import base64
import json
import sqlite3
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from src.memory.models import CandidateClaim, MemoryContext, MemoryEvent
from src.memory.store import MemoryStore
from tests.runtime import COMPACT_HARD_SECRET_CASES


def private_event(
    *,
    message_id: str = "42",
    sequence: int = 1,
    text: str = "我喜欢跑步",
) -> MemoryEvent:
    return MemoryEvent(
        context=MemoryContext(
            user_id="10001",
            session_key="private:10001",
            is_group=False,
        ),
        message_id=message_id,
        sequence=sequence,
        text=text,
        image_count=0,
        mentioned_qq_ids=("20002",),
        reply_to_message_id="previous-message",
        reply_to_user_id="20002",
    )


def group_event(
    *,
    user_id: str,
    message_id: str,
    sequence: int,
    text: str = "群消息",
    group_id: str = "30003",
) -> MemoryEvent:
    return MemoryEvent(
        context=MemoryContext(
            user_id=user_id,
            session_key=f"group:{group_id}:{user_id}",
            is_group=True,
            group_id=group_id,
        ),
        message_id=message_id,
        sequence=sequence,
        text=text,
        image_count=2,
        mentioned_qq_ids=("90009",),
        reply_to_message_id="group-previous",
        reply_to_user_id="80008",
    )


class MemoryModelTests(unittest.TestCase):
    def test_context_derives_primary_scope_and_models_are_immutable(self):
        private = MemoryContext("10001", "private:10001", False)
        group = MemoryContext("10001", "group:30003", True, "30003")
        candidate = CandidateClaim(
            subject_ref="speaker",
            predicate="likes",
            value="running",
            memory_type="preference",
            modality="asserted",
            confidence="high",
        )

        self.assertEqual(("private", "10001"), private.primary_scope)
        self.assertEqual(("group", "30003"), group.primary_scope)
        with self.assertRaises(FrozenInstanceError):
            candidate.value = "swimming"  # type: ignore[misc]


class ConfigMemoryPathTests(unittest.TestCase):
    def test_memory_database_path_is_inside_configured_data_directory(self):
        from src.config import Config

        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root).resolve()
            with mock.patch.dict(
                "os.environ",
                {
                    "CHAT_MODELS": "gemini:dummy",
                    "GEMINI_API_KEY": "dummy",
                    "DATA_DIR": str(data_dir),
                },
                clear=False,
            ):
                current = Config()

        self.assertEqual(data_dir / "memory.sqlite3", current.memory_database_path)

class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.path = Path(self.root.name) / "memory.sqlite3"
        self.store = MemoryStore(self.path)
        self.store.initialize()

    def tearDown(self):
        self.root.cleanup()

    def test_initializes_schema_fts_wal_and_version_idempotently(self):
        self.store.initialize()

        names = self.store.table_names()
        self.assertTrue(
            {
                "memory_claims",
                "memory_evidence",
                "memory_relations",
                "memory_jobs",
                "memory_deletion_audit",
                "memory_claim_tombstones",
                "schema_version",
                "memory_fts",
            }.issubset(names)
        )
        self.assertEqual(4, self.store.schema_version())
        with closing(sqlite3.connect(self.path)) as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            tombstone_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(memory_claim_tombstones)"
                )
            }
        self.assertEqual("wal", journal_mode.lower())
        self.assertEqual({"dedupe_key", "deleted_at"}, tombstone_columns)

    def test_upgrades_v1_dispute_schema_without_losing_claim_data(self):
        claim, _ = self.store.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker_qq="1002",
            subject_type="qq_user",
            subject_id="1001",
            predicate="likes",
            value="preserved-v1-data",
            memory_type="preference",
            modality="asserted",
            source_kind="message",
            source_message_id="v1-preserved-message",
            source_excerpt="preserved-v1-data",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="v1-preserved-claim",
        )
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("DROP TABLE memory_subject_disputes")
            connection.execute("DROP TABLE memory_pending_privacy_cleanup")
            connection.execute("DROP TABLE memory_claim_id_sequence")
            connection.execute(
                "DROP TABLE IF EXISTS memory_claim_tombstones"
            )
            connection.execute("DELETE FROM schema_version")
            connection.execute(
                """
                INSERT INTO schema_version(version, applied_at)
                VALUES (1, '2026-01-01T00:00:00+00:00')
                """
            )
            connection.commit()

        self.store.initialize()

        self.assertEqual(4, self.store.schema_version())
        self.assertEqual("preserved-v1-data", self.store.get_claim(claim.id).value)
        with closing(sqlite3.connect(self.path)) as connection:
            dispute_table = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name = 'memory_subject_disputes'
                """
            ).fetchone()
            dispute_index = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'index'
                  AND name = 'memory_subject_disputes_actor_idx'
                """
            ).fetchone()
            pending_table = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'memory_pending_privacy_cleanup'
                """
            ).fetchone()
            sequence_table = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'memory_claim_id_sequence'
                """
            ).fetchone()
            tombstone_table = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'memory_claim_tombstones'
                """
            ).fetchone()
            foreign_keys = connection.execute(
                "PRAGMA foreign_key_list(memory_subject_disputes)"
            ).fetchall()
        self.assertEqual(("memory_subject_disputes",), dispute_table)
        self.assertEqual(("memory_subject_disputes_actor_idx",), dispute_index)
        self.assertEqual(
            ("memory_pending_privacy_cleanup",),
            pending_table,
        )
        self.assertEqual(("memory_claim_id_sequence",), sequence_table)
        self.assertEqual(("memory_claim_tombstones",), tombstone_table)
        self.assertIn(
            ("memory_claims", "target_claim_id", "id", "CASCADE"),
            {
                (row[2], row[3], row[4], row[6])
                for row in foreign_keys
            },
        )

    def test_upgrades_legacy_jobs_with_indexed_source_message_id(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "legacy-jobs.sqlite3"
            payload = json.dumps(
                {
                    "context": {
                        "user_id": "10001",
                        "session_key": "private:10001",
                        "is_group": False,
                        "group_id": None,
                    },
                    "message_id": "legacy-source-42",
                    "sequence": 42,
                    "text": "legacy body",
                    "image_count": 0,
                    "mentioned_qq_ids": [],
                    "reply_to_message_id": None,
                    "reply_to_user_id": None,
                },
                ensure_ascii=False,
            )
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    """
                    CREATE TABLE memory_jobs (
                        id INTEGER PRIMARY KEY,
                        dedupe_key TEXT NOT NULL UNIQUE,
                        scope_key TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        state TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        retry_at TEXT,
                        error_type TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO memory_jobs(
                        dedupe_key, scope_key, sequence, payload_json, state,
                        attempts, retry_at, error_type, created_at, updated_at
                    ) VALUES (
                        'legacy-job', 'private:10001', 42, ?, 'retry',
                        1, NULL, 'temporary', '2026-01-01', '2026-01-01'
                    )
                    """,
                    (payload,),
                )
                connection.commit()

            store = MemoryStore(path)
            store.initialize()

            with closing(sqlite3.connect(path)) as connection:
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(memory_jobs)"
                    )
                }
                source_message_id = connection.execute(
                    """
                    SELECT source_message_id
                    FROM memory_jobs
                    WHERE dedupe_key = 'legacy-job'
                    """
                ).fetchone()[0]
                indexes = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA index_list(memory_jobs)"
                    )
                }

            self.assertIn("source_message_id", columns)
            self.assertEqual("legacy-source-42", source_message_id)
            self.assertIn("memory_jobs_source_message_idx", indexes)

    def test_sequence_upgrade_seeds_from_deleted_audit_and_pending_ids(self):
        cases = (
            ("audit-only", 40, None, 40),
            ("pending-only", None, 50, 50),
        )
        for label, audit_id, pending_id, expected_floor in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "legacy-sequence.sqlite3"
                store = MemoryStore(path)
                store.initialize()
                with closing(sqlite3.connect(path)) as connection:
                    connection.execute("DROP TABLE memory_claim_id_sequence")
                    if audit_id is not None:
                        connection.execute(
                            """
                            INSERT INTO memory_deletion_audit(
                                claim_id, reason, deleted_at
                            ) VALUES (?, 'private_privacy_delete', ?)
                            """,
                            (audit_id, "2026-01-01T00:00:00+00:00"),
                        )
                    if pending_id is not None:
                        connection.execute(
                            """
                            INSERT INTO memory_pending_privacy_cleanup(
                                claim_id, reason, scope_type, scope_id,
                                needs_fts_optimize, created_at
                            ) VALUES (
                                ?, 'private_privacy_delete', 'private',
                                '1001', 0, '2026-01-01T00:00:00+00:00'
                            )
                            """,
                            (pending_id,),
                        )
                    connection.commit()

                store.initialize()
                claim, _ = store.create_claim(
                    scope_type="private",
                    scope_id="1001",
                    speaker_qq="1001",
                    subject_type="qq_user",
                    subject_id="1001",
                    predicate="likes",
                    value=f"sequence-{label}",
                    memory_type="preference",
                    modality="asserted",
                    source_kind="message",
                    source_message_id=f"sequence-{label}-message",
                    source_excerpt=f"sequence-{label}",
                    extraction_confidence="high",
                    attribution_confidence="high",
                    truth_confidence="medium",
                    dedupe_key=f"sequence-{label}-claim",
                )

                self.assertGreater(claim.id, expected_floor)

    def test_precommit_database_lock_is_not_reported_as_deleted_or_partial(self):
        claim, _ = self.store.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker_qq="1001",
            subject_type="qq_user",
            subject_id="1001",
            predicate="likes",
            value="locked-before-commit",
            memory_type="preference",
            modality="asserted",
            source_kind="message",
            source_message_id="precommit-lock-message",
            source_excerpt="locked-before-commit",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="precommit-lock-claim",
        )
        locker = sqlite3.connect(self.path)
        locker.execute("BEGIN IMMEDIATE")
        original_connect = sqlite3.connect

        def short_timeout_connect(*args, **kwargs):
            kwargs["timeout"] = 0.01
            return original_connect(*args, **kwargs)

        try:
            with mock.patch.object(
                sqlite3,
                "connect",
                side_effect=short_timeout_connect,
            ):
                with self.assertRaisesRegex(
                    sqlite3.OperationalError,
                    "locked",
                ):
                    self.store.delete_claim_physically_with_outcome(
                        claim.id,
                        reason="private_privacy_delete",
                        actor_qq="1001",
                        is_admin=False,
                    )
        finally:
            locker.rollback()
            locker.close()

        self.assertIsNotNone(self.store.get_claim(claim.id))
        with closing(sqlite3.connect(self.path)) as connection:
            audit_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM memory_deletion_audit
                WHERE claim_id = ?
                """,
                (claim.id,),
            ).fetchone()[0]
        self.assertEqual(0, audit_count)

    def test_duplicate_message_job_is_idempotent_across_threads(self):
        event = private_event()

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = tuple(pool.map(lambda _: self.store.create_job(event), range(4)))

        self.assertEqual(1, sum(created for _, created in results))
        self.assertEqual(1, len({job_id for job_id, _ in results}))

    def test_job_payload_is_minimal_and_redacts_forbidden_data(self):
        bare_image_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAusB9Wl2ZQAAAABJRU5ErkJggg=="
        )
        event = private_event(
            text=(
                "保留这句话；data:image/png;base64,aGVsbG8=；"
                f"裸图片 {bare_image_base64}；"
                "api_key=sk-secret-value；卡号 4111 1111 1111 1111"
            )
        )

        job_id, _ = self.store.create_job(event)
        job = self.store.get_job(job_id)
        payload = json.loads(job.payload_json)

        self.assertEqual(
            {
                "context",
                "message_id",
                "sequence",
                "text",
                "image_count",
                "mentioned_qq_ids",
                "reply_to_message_id",
                "reply_to_user_id",
            },
            set(payload),
        )
        serialized = job.payload_json.lower()
        self.assertIn("保留这句话", serialized)
        self.assertNotIn("data:image", serialized)
        self.assertNotIn("agvsbg8", serialized)
        self.assertNotIn(bare_image_base64.lower(), serialized)
        self.assertNotIn("sk-secret-value", serialized)
        self.assertNotIn("4111 1111 1111 1111", serialized)

    def test_job_payload_redacts_all_explicitly_forbidden_secret_categories(self):
        forbidden_values = (
            "cookie-session-value",
            "generic-token-value",
            "654321",
            "123",
            "pay-credential-value",
            "bearer-credential-value",
            "ghp_abcdefghijklmnopqrstuvwxyz123456",
            "private-key-body-value",
            "chinese-password-value",
            "chinese-api-key-value",
            "chinese-payment-value",
            (
                "Qk14eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4"
                "eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4"
            ),
        )
        event = private_event(
            message_id="forbidden-categories",
            text=(
                "Cookie=cookie-session-value；"
                "token=generic-token-value；"
                "验证码=654321；CVV=123；"
                "payment_token=pay-credential-value；"
                "Authorization: Bearer bearer-credential-value；"
                "GitHub ghp_abcdefghijklmnopqrstuvwxyz123456；"
                "-----BEGIN PRIVATE KEY-----\n"
                "private-key-body-value\n"
                "-----END PRIVATE KEY-----；"
                "密码是chinese-password-value；"
                "API 密钥：chinese-api-key-value；"
                "支付凭据是chinese-payment-value；"
                f"BMP {forbidden_values[-1]}"
            ),
        )

        job_id, _ = self.store.create_job(event)
        payload = self.store.get_job(job_id).payload_json.lower()

        for forbidden in forbidden_values:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden.lower(), payload)
        self.assertIn("[redacted:credential]", payload)
        self.assertIn("[redacted:payment-data]", payload)
        self.assertIn("[redacted:image-data]", payload)

    def test_job_payload_redacts_delimited_and_compact_secrets_before_sqlite(self):
        forbidden_values = (
            "654321",
            "482915",
            "736219",
            "PASSWORD_SPACE_SENTINEL_7F2A",
            "TOKEN_SPACE_SENTINEL_93B1",
            "API_KEY_SPACE_SENTINEL_5D8C",
            "PAYMENT_SPACE_SENTINEL_4A6E",
            "4111 1111 1111 1111",
            "1234567890",
            "9876543210",
            "1357902468",
            "2468013579",
            "123",
        ) + tuple(
            dict.fromkeys(
                raw_value
                for _label, _secret, raw_value
                in COMPACT_HARD_SECRET_CASES
            )
        )
        secret_text = "；".join(
            (
                "验证码 654321",
                "OTP 482915",
                "verification code 736219",
                "密码 PASSWORD_SPACE_SENTINEL_7F2A",
                "token TOKEN_SPACE_SENTINEL_93B1",
                "API key API_KEY_SPACE_SENTINEL_5D8C",
                "支付凭据 PAYMENT_SPACE_SENTINEL_4A6E",
                "银行卡号 4111 1111 1111 1111",
                "bank account 1234567890",
                "银行账号 9876543210",
                "银行帐号 1357902468",
                "支付账号 2468013579",
                "CVV 123",
                *(
                    secret
                    for _label, secret, _raw_value
                    in COMPACT_HARD_SECRET_CASES
                ),
            )
        )
        job_id, _ = self.store.create_job(
            private_event(
                message_id="space-delimited-secret-categories",
                text=secret_text,
            )
        )

        payload = self.store.get_job(job_id).payload_json

        for forbidden in forbidden_values:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, payload)
        self.assertIn("[redacted:credential]", payload)
        self.assertIn("[redacted:payment-data]", payload)
        for database_file in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            if not database_file.exists():
                continue
            database_bytes = database_file.read_bytes()
            for sentinel in (
                "PASSWORD_SPACE_SENTINEL_7F2A",
                "TOKEN_SPACE_SENTINEL_93B1",
                "API_KEY_SPACE_SENTINEL_5D8C",
                "PAYMENT_SPACE_SENTINEL_4A6E",
                *(
                    secret
                    for _label, secret, _raw_value
                    in COMPACT_HARD_SECRET_CASES
                ),
            ):
                with self.subTest(
                    database_file=database_file.name,
                    sentinel=sentinel,
                ):
                    self.assertNotIn(
                        sentinel.encode("utf-8"),
                        database_bytes,
                    )

    def test_job_payload_redacts_mime_wrapped_binary_base64(self):
        encoded_values = []
        for prefix, marker in (
            (b"BM", b"BMP_PRIVATE_BODY"),
            (b"II*\x00", b"TIFF_PRIVATE_BODY"),
            (b"PK\x03\x04", b"ATTACHMENT_PRIVATE_BODY"),
        ):
            encoded = base64.b64encode(prefix + marker + (b"x" * 96)).decode()
            encoded_values.append(
                "\r\n".join(
                    encoded[index : index + 40]
                    for index in range(0, len(encoded), 40)
                )
            )
        event = private_event(
            message_id="mime-base64",
            text="；".join(encoded_values),
        )

        job_id, _ = self.store.create_job(event)
        stored_text = self.store.get_job(job_id).text

        for encoded in encoded_values:
            for line in encoded.splitlines():
                with self.subTest(line=line):
                    self.assertNotIn(line, stored_text)
        self.assertEqual(
            3,
            stored_text.count("[redacted:image-data]"),
        )

    def test_mime_base64_filter_does_not_consume_following_prose(self):
        event = private_event(
            message_id="mime-followed-by-prose",
            text=(
                "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkw\n"
                "Keep this ordinary prose."
            ),
        )

        job_id, _ = self.store.create_job(event)

        self.assertIn(
            "Keep this ordinary prose.",
            self.store.get_job(job_id).text,
        )

    def test_short_base64_shaped_ordinary_lines_are_not_redacted(self):
        samples = (
            "abcdefghijklmnop\nordinary",
            "abcdefghijklmnop\nqrstuvwxyzabcdef",
        )

        for index, sample in enumerate(samples):
            with self.subTest(sample=sample):
                event = private_event(
                    message_id=f"ordinary-base64-lines-{index}",
                    text=sample,
                )
                job_id, _ = self.store.create_job(event)
                self.assertEqual(sample, self.store.get_job(job_id).text)

    def test_mime_binary_redaction_stops_before_following_text(self):
        encoded = base64.b64encode(
            b"BM" + b"BMP_PRIVATE_BODY" + (b"x" * 96)
        ).decode()
        wrapped = "\r\n".join(
            encoded[index : index + 40]
            for index in range(0, len(encoded), 40)
        )

        for index, suffix in enumerate(
            ("Keep this ordinary prose.", "ordinary")
        ):
            with self.subTest(suffix=suffix):
                event = private_event(
                    message_id=f"mime-binary-boundary-{index}",
                    text=f"{wrapped}\r\n{suffix}",
                )
                job_id, _ = self.store.create_job(event)
                stored_text = self.store.get_job(job_id).text
                for line in wrapped.splitlines():
                    self.assertNotIn(line, stored_text)
                self.assertIn(suffix, stored_text)

    def test_ambiguous_unpadded_short_line_is_not_consumed_as_binary_tail(self):
        first_line = base64.b64encode(b"BM" + (b"A" * 96)).decode()[:16]

        for index, newline in enumerate(("\n", "\r\n")):
            with self.subTest(newline=repr(newline)):
                event = private_event(
                    message_id=f"ambiguous-short-tail-{index}",
                    text=f"{first_line}{newline}ordinary",
                )
                job_id, _ = self.store.create_job(event)
                self.assertEqual(
                    f"[redacted:image-data]{newline}ordinary",
                    self.store.get_job(job_id).text,
                )

    def test_structured_wrapped_binary_keeps_unpadded_and_padded_boundaries(self):
        payloads = (
            ("bmp-unpadded", b"BM" + (b"A" * 37), False),
            ("bmp-padded", b"BM" + (b"A" * 32), True),
            ("tiff-unpadded", b"II*\x00" + (b"A" * 35), False),
            ("tiff-padded", b"II*\x00" + (b"A" * 33), True),
        )

        for payload_name, payload, expects_padding in payloads:
            encoded = base64.b64encode(payload).decode()
            self.assertEqual(expects_padding, encoded.endswith("=="))
            wrapped_lines = [
                encoded[index : index + 16]
                for index in range(0, len(encoded), 16)
            ]
            self.assertGreaterEqual(
                sum(len(line) == 16 for line in wrapped_lines),
                2,
            )
            for newline_index, newline in enumerate(("\n", "\r\n")):
                with self.subTest(
                    payload=payload_name,
                    newline=repr(newline),
                ):
                    wrapped = newline.join(wrapped_lines)
                    event = private_event(
                        message_id=(
                            f"structured-binary-{payload_name}-{newline_index}"
                        ),
                        text=f"{wrapped}{newline}ordinary",
                    )
                    job_id, _ = self.store.create_job(event)
                    stored_text = self.store.get_job(job_id).text
                    for line in wrapped_lines:
                        self.assertNotIn(line, stored_text)
                    self.assertIn("ordinary", stored_text)

    def test_job_payload_redacts_the_complete_cookie_header(self):
        event = private_event(
            message_id="multi-cookie",
            text=(
                "Cookie: session=alpha-secret; csrf=beta-secret; theme=dark\n"
                "仍需保留这句话"
            ),
        )

        job_id, _ = self.store.create_job(event)
        stored_text = self.store.get_job(job_id).text

        self.assertNotIn("alpha-secret", stored_text)
        self.assertNotIn("beta-secret", stored_text)
        self.assertNotIn("theme=dark", stored_text)
        self.assertIn("[redacted:credential]", stored_text)
        self.assertIn("仍需保留这句话", stored_text)

    def test_group_job_preserves_attribution_context_and_structured_payload(self):
        event = group_event(
            user_id="10001",
            message_id="group-message",
            sequence=7,
            text="我喜欢跑步",
        )

        job_id, _ = self.store.create_job(event)
        job = self.store.get_job(job_id)

        self.assertEqual("group:30003", job.scope_key)
        self.assertEqual(event.context, job.context)
        self.assertEqual(event.message_id, job.message_id)
        self.assertEqual(event.text, job.text)
        self.assertEqual(2, job.image_count)
        self.assertEqual(("90009",), job.mentioned_qq_ids)
        self.assertEqual("group-previous", job.reply_to_message_id)
        self.assertEqual("80008", job.reply_to_user_id)

    def test_claim_next_job_preserves_scope_sequence_and_one_running_job(self):
        first_id, _ = self.store.create_job(
            private_event(message_id="first-blocker", sequence=1)
        )
        second_id, _ = self.store.create_job(
            private_event(message_id="second-ready", sequence=2)
        )
        self.store.mark_job_ready(second_id)

        self.assertIsNone(self.store.claim_next_job("private:10001"))

        self.store.mark_job_ready(first_id)
        first = self.store.claim_next_job("private:10001")
        self.assertEqual(first_id, first.id)
        self.assertIsNone(self.store.claim_next_job("private:10001"))

        future_retry = (
            datetime.now(timezone.utc) + timedelta(minutes=5)
        ).isoformat()
        self.store.fail_job(first_id, "temporary", future_retry)
        self.assertIsNone(self.store.claim_next_job("private:10001"))

        self.store.mark_job_ready(first_id)
        reclaimed = self.store.claim_next_job("private:10001")
        self.assertEqual(first_id, reclaimed.id)
        self.store.complete_job(first_id)
        second = self.store.claim_next_job("private:10001")
        self.assertEqual(second_id, second.id)

    def test_group_users_have_independent_fifo_job_scopes(self):
        first_id, _ = self.store.create_job(
            group_event(user_id="10001", message_id="group-a", sequence=1)
        )
        second_id, _ = self.store.create_job(
            group_event(user_id="20002", message_id="group-b", sequence=2)
        )
        self.store.mark_job_ready(first_id)
        self.store.mark_job_ready(second_id)

        first = self.store.claim_next_job("group:30003")
        self.assertEqual(first_id, first.id)
        self.store.complete_job(first_id)
        second = self.store.claim_next_job("group:30003")
        self.assertEqual(second_id, second.id)

    def test_cleanup_honors_cutoff_and_keeps_cleaned_jobs_readable(self):
        recent_id, _ = self.store.create_job(
            private_event(
                message_id="recent-done",
                sequence=10,
                text="RECENT_JOB_BODY_91ba",
            )
        )
        old_id, _ = self.store.create_job(
            private_event(
                message_id="old-done",
                sequence=11,
                text="OLD_JOB_BODY_a614",
            )
        )
        running_id, _ = self.store.create_job(
            private_event(
                message_id="current-running",
                sequence=12,
                text="CURRENT_CLAIM_BODY_201c",
            )
        )
        for job_id in (recent_id, old_id, running_id):
            self.store.mark_job_ready(job_id)
            claimed = self.store.claim_next_job("private:10001")
            self.assertIsNotNone(claimed)
            if job_id == running_id:
                break
            self.store.complete_job(claimed.id)

        archived, _ = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="old-archived-predicate",
            value="old-archived-value",
            memory_type="fact",
            modality="asserted",
            source_kind="message:speaker",
            source_message_id="old-archived-source",
            source_excerpt="OLD_ARCHIVED_EXCERPT_f781",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="high",
            dedupe_key="old-archived-claim",
            status="archived",
        )
        self.store.add_evidence(
            archived.id,
            source_kind="message:speaker",
            source_message_id="old-archived-evidence",
            source_excerpt="OLD_ARCHIVED_EVIDENCE_33c1",
        )
        active, _ = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="current-active-predicate",
            value="current-active-value",
            memory_type="fact",
            modality="asserted",
            source_kind="message:speaker",
            source_message_id="current-active-source",
            source_excerpt="CURRENT_ACTIVE_EXCERPT_8f12",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="high",
            dedupe_key="current-active-claim",
        )
        old_timestamp = (
            datetime.now(timezone.utc) - timedelta(days=91)
        ).isoformat()
        with self.store._connection() as connection:
            connection.execute(
                "UPDATE memory_jobs SET updated_at = ? WHERE id = ?",
                (old_timestamp, old_id),
            )
            connection.execute(
                "UPDATE memory_claims SET created_at = ? WHERE id IN (?, ?)",
                (old_timestamp, archived.id, active.id),
            )
            connection.execute(
                "UPDATE memory_evidence SET created_at = ? WHERE claim_id = ?",
                (old_timestamp, archived.id),
            )

        self.store.cleanup_old_jobs_and_excerpts(days=90)

        self.assertEqual("RECENT_JOB_BODY_91ba", self.store.get_job(recent_id).text)
        cleaned = self.store.get_job(old_id)
        self.assertEqual("", cleaned.text)
        self.assertNotIn("OLD_JOB_BODY_a614", cleaned.payload_json)
        self.assertEqual(
            "CURRENT_CLAIM_BODY_201c",
            self.store.get_job(running_id).text,
        )
        self.assertEqual("", self.store.get_claim(archived.id).source_excerpt)
        self.assertEqual("", self.store.list_evidence(archived.id)[0].source_excerpt)
        self.assertEqual(
            "CURRENT_ACTIVE_EXCERPT_8f12",
            self.store.get_claim(active.id).source_excerpt,
        )
        with self.store._connection() as connection:
            archived_fts_excerpt = connection.execute(
                "SELECT source_excerpt FROM memory_fts WHERE rowid = ?",
                (archived.id,),
            ).fetchone()[0]
        self.assertEqual("", archived_fts_excerpt)

    def test_cleanup_retries_maintenance_after_post_commit_failure(self):
        job_marker = "RETRY_JOB_BODY_711d"
        excerpt_marker = "RETRY_ARCHIVED_EXCERPT_4c20"
        job_id, _ = self.store.create_job(
            private_event(
                message_id="retry-maintenance-job",
                sequence=20,
                text=job_marker,
            )
        )
        self.store.mark_job_ready(job_id)
        claimed = self.store.claim_next_job("private:10001")
        self.store.complete_job(claimed.id)
        archived, _ = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="retry-maintenance-predicate",
            value="retry-maintenance-value",
            memory_type="fact",
            modality="asserted",
            source_kind="message:speaker",
            source_message_id="retry-maintenance-source",
            source_excerpt=excerpt_marker,
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="high",
            dedupe_key="retry-maintenance-claim",
            status="archived",
        )
        old_timestamp = (
            datetime.now(timezone.utc) - timedelta(days=91)
        ).isoformat()
        with self.store._connection() as connection:
            connection.execute(
                "UPDATE memory_jobs SET updated_at = ? WHERE id = ?",
                (old_timestamp, job_id),
            )
            connection.execute(
                "UPDATE memory_claims SET created_at = ? WHERE id = ?",
                (old_timestamp, archived.id),
            )

        with (
            mock.patch(
                "src.memory.store._fts_secure_delete_enabled",
                return_value=False,
            ),
            mock.patch.object(
                self.store,
                "_optimize_fts_for_privacy",
                side_effect=RuntimeError("maintenance interrupted"),
            ),
            self.assertRaises(RuntimeError),
        ):
            self.store.cleanup_old_jobs_and_excerpts(days=90)

        with (
            mock.patch(
                "src.memory.store._fts_secure_delete_enabled",
                return_value=False,
            ),
            mock.patch.object(
                self.store,
                "_optimize_fts_for_privacy",
                wraps=self.store._optimize_fts_for_privacy,
            ) as optimize,
            mock.patch.object(
                self.store,
                "_checkpoint_wal",
                wraps=self.store._checkpoint_wal,
            ) as checkpoint,
        ):
            self.store.cleanup_old_jobs_and_excerpts(days=90)

        optimize.assert_called_once_with()
        checkpoint.assert_called_once_with()
        self.assertEqual("", self.store.get_job(job_id).text)
        self.assertEqual("", self.store.get_claim(archived.id).source_excerpt)
        for path in (self.path, Path(f"{self.path}-wal")):
            if path.exists():
                contents = path.read_bytes()
                self.assertNotIn(job_marker.encode(), contents)
                self.assertNotIn(excerpt_marker.encode(), contents)

    def test_recovers_abandoned_running_jobs_after_reopen(self):
        job_id, _ = self.store.create_job(
            private_event(message_id="crash-recovery", sequence=1)
        )
        self.store.mark_job_ready(job_id)
        running = self.store.claim_next_job("private:10001")
        self.assertEqual("running", running.state)

        reopened = MemoryStore(self.path)
        reopened.initialize()
        self.assertIsNone(reopened.claim_next_job("private:10001"))
        self.assertEqual(1, reopened.recover_running_jobs())

        recovered = reopened.get_job(job_id)
        self.assertEqual("retry", recovered.state)
        self.assertEqual("abandoned", recovered.error_type)
        reclaimed = reopened.claim_next_job("private:10001")
        self.assertEqual(job_id, reclaimed.id)
        self.assertEqual(2, reclaimed.attempts)

    def test_retry_time_is_validated_normalized_to_utc_and_compared_by_instant(self):
        job_id, _ = self.store.create_job(
            private_event(message_id="offset-retry", sequence=1)
        )
        self.store.mark_job_ready(job_id)
        running = self.store.claim_next_job("private:10001")
        retry_instant = datetime.now(timezone.utc) - timedelta(seconds=1)
        retry_with_offset = retry_instant.astimezone(
            timezone(timedelta(hours=8))
        ).isoformat()

        self.store.fail_job(running.id, "temporary", retry_with_offset)

        normalized = self.store.get_job(job_id)
        self.assertEqual(retry_instant.isoformat(), normalized.retry_at)
        self.assertEqual(
            job_id,
            self.store.claim_next_job("private:10001").id,
        )

        second_id, _ = self.store.create_job(
            private_event(message_id="invalid-retry", sequence=2)
        )
        self.store.mark_job_ready(second_id)
        self.store.complete_job(job_id)
        second = self.store.claim_next_job("private:10001")
        with self.assertRaisesRegex(ValueError, "retry_at"):
            self.store.fail_job(second.id, "temporary", "2026-07-26T10:00:00")
        with self.assertRaisesRegex(ValueError, "retry_at"):
            self.store.fail_job(second.id, "temporary", "not-a-timestamp")

    def test_job_state_machine_retries_and_survives_reopen(self):
        first_id, _ = self.store.create_job(
            private_event(message_id="first", sequence=1)
        )
        second_id, _ = self.store.create_job(
            private_event(message_id="second", sequence=2)
        )
        self.store.mark_job_ready(first_id)
        self.store.mark_job_ready(second_id)

        first = self.store.claim_next_job("private:10001")
        self.assertEqual(first_id, first.id)
        self.assertEqual("running", first.state)
        self.assertEqual(1, first.attempts)

        retry_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        self.store.fail_job(first_id, "temporary_network", retry_at)
        reopened = MemoryStore(self.path)
        reopened.initialize()
        retried = reopened.claim_next_job("private:10001")
        self.assertEqual(first_id, retried.id)
        self.assertEqual(2, retried.attempts)
        reopened.complete_job(first_id)

        second = reopened.claim_next_job("private:10001")
        self.assertEqual(second_id, second.id)
        reopened.fail_job(second_id, "invalid_output", None)
        self.assertEqual("failed", reopened.get_job(second_id).state)

    def test_claim_crud_evidence_relations_fts_and_physical_delete(self):
        first, first_created = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="喜欢",
            value="越野跑",
            memory_type="preference",
            modality="asserted",
            source_kind="message",
            source_message_id="m-1",
            source_excerpt="我最近喜欢越野跑",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="claim:m-1:likes-running",
        )
        duplicate, duplicate_created = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="喜欢",
            value="越野跑",
            memory_type="preference",
            modality="asserted",
            source_kind="message",
            source_message_id="m-1",
            source_excerpt="我最近喜欢越野跑",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="claim:m-1:likes-running",
        )
        second, _ = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="不喜欢",
            value="越野跑",
            memory_type="preference",
            modality="negated",
            source_kind="message",
            source_message_id="m-2",
            source_excerpt="我现在不喜欢越野跑",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="claim:m-2:dislikes-running",
        )

        self.assertTrue(first_created)
        self.assertFalse(duplicate_created)
        self.assertEqual(first.id, duplicate.id)
        evidence_id, evidence_created = self.store.add_evidence(
            first.id,
            source_kind="message",
            source_message_id="m-3",
            source_excerpt="周末我又去越野跑了",
        )
        self.assertTrue(evidence_created)
        self.assertGreater(evidence_id, 0)
        self.assertEqual(1, len(self.store.list_evidence(first.id)))

        self.assertTrue(
            self.store.add_relation(first.id, second.id, "contradicts")
        )
        self.assertFalse(
            self.store.add_relation(first.id, second.id, "contradicts")
        )
        relations = self.store.list_relations(first.id)
        self.assertEqual("contradicts", relations[0].relation_type)

        found = self.store.search_claims(
            "越野跑",
            scope_type="private",
            scope_id="10001",
        )
        self.assertEqual({first.id, second.id}, {claim.id for claim in found})

        updated = self.store.update_claim(
            first.id,
            value="公路骑行",
            source_excerpt="我改为喜欢公路骑行",
            status="superseded",
        )
        self.assertEqual("公路骑行", updated.value)
        self.assertEqual((), self.store.search_claims("我最近喜欢越野跑"))
        self.assertEqual(
            (first.id,),
            tuple(
                claim.id
                for claim in self.store.search_claims(
                    "公路骑行",
                    scope_type="private",
                    scope_id="10001",
                )
            ),
        )

        self.assertTrue(
            self.store.delete_relation(first.id, second.id, "contradicts")
        )
        self.assertTrue(self.store.delete_evidence(evidence_id))
        self.assertTrue(
            self.store.delete_claim_physically(
                first.id,
                reason="user_forget",
                actor_qq="10001",
                is_admin=False,
            )
        )
        self.assertIsNone(self.store.get_claim(first.id))
        self.assertEqual((), self.store.search_claims("公路骑行"))

        with closing(sqlite3.connect(self.path)) as connection:
            residual = connection.execute(
                """
                SELECT COUNT(*) FROM memory_fts
                WHERE memory_fts MATCH ?
                """,
                ('"公路骑行"',),
            ).fetchone()[0]
            audit = connection.execute(
                """
                SELECT claim_id, reason, deleted_at
                FROM memory_deletion_audit
                WHERE claim_id = ?
                """,
                (first.id,),
            ).fetchone()
            audit_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(memory_deletion_audit)"
                )
            }
        self.assertEqual(0, residual)
        self.assertEqual(first.id, audit[0])
        self.assertEqual("user_forget", audit[1])
        self.assertTrue(audit[2])
        self.assertTrue(
            {
                "predicate",
                "value",
                "source_excerpt",
                "dedupe_key",
                "payload_json",
            }.isdisjoint(audit_columns)
        )

    def test_exact_scoped_claim_query_is_not_capped_by_fts_limit(self):
        for index in range(513):
            self.store.create_claim(
                scope_type="group",
                scope_id="900",
                speaker_qq="101",
                subject_type="qq_user",
                subject_id=(
                    "target-after-limit"
                    if index == 512
                    else f"filler-{index}"
                ),
                predicate="likes",
                value=f"value-{index}",
                memory_type="fact",
                modality="asserted",
                source_kind="message:speaker",
                source_message_id=f"exact-query-{index}",
                source_excerpt=f"value-{index}",
                extraction_confidence="high",
                attribution_confidence="high",
                truth_confidence="high",
                dedupe_key=f"exact-query-{index}",
            )

        exact = self.store.find_claims_exact(
            scope_type="group",
            scope_id="900",
            statuses=("active", "disputed"),
            subject_type="qq_user",
            subject_id="target-after-limit",
            predicates=("likes",),
        )

        self.assertEqual(1, len(exact))
        self.assertEqual("value-512", exact[0].value)

    def test_authorized_search_filters_scope_in_sql_before_fts_limit(self):
        visible, _ = self.store.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker_qq="1001",
            subject_type="qq_user",
            subject_id="1001",
            predicate="topic",
            value="AUTHORIZED-LEXICAL-MARKER",
            memory_type="fact",
            modality="asserted",
            source_kind="message:speaker",
            source_message_id="authorized-visible",
            source_excerpt="AUTHORIZED-LEXICAL-MARKER",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="high",
            dedupe_key="authorized-visible",
        )
        for index in range(20):
            self.store.create_claim(
                scope_type="group",
                scope_id="other-group",
                speaker_qq="2002",
                subject_type="qq_user",
                subject_id="2002",
                predicate="topic",
                value="AUTHORIZED-LEXICAL-MARKER",
                memory_type="fact",
                modality="asserted",
                source_kind="message:speaker",
                source_message_id=f"unauthorized-{index}",
                source_excerpt="AUTHORIZED-LEXICAL-MARKER",
                extraction_confidence="high",
                attribution_confidence="high",
                truth_confidence="high",
                dedupe_key=f"unauthorized-{index}",
            )
        context = MemoryContext(
            user_id="1001",
            session_key="group:2001:1001",
            is_group=True,
            group_id="2001",
        )

        found = self.store.search_authorized_claims(
            context,
            "AUTHORIZED-LEXICAL-MARKER",
            limit=1,
        )

        self.assertEqual([visible.id], [claim.id for claim in found])

    def test_reconciliation_transaction_rolls_back_all_writes(self):
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            with self.store.reconciliation() as transaction:
                transaction.create_claim(
                    scope_type="private",
                    scope_id="10001",
                    speaker_qq="10001",
                    subject_type="qq_user",
                    subject_id="10001",
                    predicate="likes",
                    value="transaction-private-value",
                    memory_type="preference",
                    modality="asserted",
                    source_kind="message:speaker",
                    source_message_id="transaction-message",
                    source_excerpt="我喜欢 transaction-private-value",
                    extraction_confidence="high",
                    attribution_confidence="high",
                    truth_confidence="high",
                    dedupe_key="transaction-claim",
                )
                raise RuntimeError("injected failure")

        self.assertEqual(
            (),
            self.store.search_claims("transaction-private-value"),
        )
        with closing(sqlite3.connect(self.path)) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM memory_claims"
            ).fetchone()[0]
        self.assertEqual(0, count)

    def test_constraints_and_foreign_keys_reject_invalid_records(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.create_claim(
                scope_type="room",
                scope_id="10001",
                speaker_qq="10001",
                subject_type="qq_user",
                subject_id="10001",
                predicate="喜欢",
                value="跑步",
                memory_type="preference",
                modality="asserted",
                source_kind="message",
                source_message_id="m-invalid",
                source_excerpt="喜欢跑步",
                extraction_confidence="high",
                attribution_confidence="high",
                truth_confidence="medium",
                dedupe_key="invalid-scope",
            )

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add_evidence(
                999999,
                source_kind="message",
                source_message_id="missing",
                source_excerpt="orphan",
            )

    def test_physical_delete_erases_private_body_from_database_files(self):
        marker = "PRIVATE_BODY_MARKER_7f29e521"
        claim, _ = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="likes",
            value=marker,
            memory_type="preference",
            modality="asserted",
            source_kind="message",
            source_message_id="private-marker-message",
            source_excerpt=marker,
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="private-marker-claim",
        )

        self.assertTrue(
            self.store.delete_claim_physically(
                claim.id,
                reason="user_forget",
                actor_qq="10001",
                is_admin=False,
            )
        )

        marker_bytes = marker.encode("utf-8")
        database_files = (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        )
        for database_file in database_files:
            if database_file.exists():
                with self.subTest(database_file=database_file.name):
                    self.assertNotIn(marker_bytes, database_file.read_bytes())

    def test_physical_delete_same_message_id_is_scoped(self):
        source_message_id = "onebot-collision-7788"
        private_marker = "PRIVATE_COLLISION_BODY_61A2"
        group_a_marker = "GROUP_A_COLLISION_BODY_73B4"
        group_b_marker = "GROUP_B_COLLISION_BODY_95C6"
        events = (
            private_event(
                message_id=source_message_id,
                text=private_marker,
            ),
            group_event(
                user_id="20001",
                group_id="30003",
                message_id=source_message_id,
                sequence=2,
                text=group_a_marker,
            ),
            group_event(
                user_id="20002",
                group_id="40004",
                message_id=source_message_id,
                sequence=3,
                text=group_b_marker,
            ),
        )
        job_ids = []
        for current_event in events:
            job_id, created = self.store.create_job(current_event)
            self.assertTrue(created)
            self.store.mark_job_ready(job_id)
            job_ids.append(job_id)

        def create_scoped_claim(
            scope_type: str,
            scope_id: str,
            speaker_qq: str,
            marker: str,
        ):
            claim, created = self.store.create_claim(
                scope_type=scope_type,
                scope_id=scope_id,
                speaker_qq=speaker_qq,
                subject_type="qq_user",
                subject_id=speaker_qq,
                predicate="likes",
                value=marker,
                memory_type="preference",
                modality="asserted",
                source_kind="message",
                source_message_id=source_message_id,
                source_excerpt=marker,
                extraction_confidence="high",
                attribution_confidence="high",
                truth_confidence="medium",
                dedupe_key=f"collision-{scope_type}-{scope_id}",
            )
            self.assertTrue(created)
            return claim

        private_claim = create_scoped_claim(
            "private",
            "10001",
            "10001",
            private_marker,
        )
        group_a_claim = create_scoped_claim(
            "group",
            "30003",
            "20001",
            group_a_marker,
        )
        group_b_claim = create_scoped_claim(
            "group",
            "40004",
            "20002",
            group_b_marker,
        )
        self.store.add_evidence(
            group_a_claim.id,
            source_kind="message",
            source_message_id=source_message_id,
            source_excerpt=group_a_marker,
        )
        self.store.add_evidence(
            group_b_claim.id,
            source_kind="message",
            source_message_id=source_message_id,
            source_excerpt=group_b_marker,
        )

        self.store.delete_claim_physically(
            private_claim.id,
            reason="private_privacy_delete",
            actor_qq="10001",
            is_admin=False,
        )

        self.assertEqual("failed", self.store.get_job(job_ids[0]).state)
        self.assertEqual("", self.store.get_job(job_ids[0]).text)
        self.assertEqual("ready", self.store.get_job(job_ids[1]).state)
        self.assertEqual(group_a_marker, self.store.get_job(job_ids[1]).text)
        self.assertEqual("ready", self.store.get_job(job_ids[2]).state)
        self.assertEqual(group_b_marker, self.store.get_job(job_ids[2]).text)
        self.assertEqual(
            group_a_marker,
            self.store.get_claim(group_a_claim.id).source_excerpt,
        )
        self.assertEqual(
            group_b_marker,
            self.store.get_claim(group_b_claim.id).source_excerpt,
        )
        self.assertEqual(
            group_a_marker,
            self.store.list_evidence(group_a_claim.id)[0].source_excerpt,
        )
        self.assertEqual(
            group_b_marker,
            self.store.list_evidence(group_b_claim.id)[0].source_excerpt,
        )

        self.store.delete_claim_physically(
            group_a_claim.id,
            reason="administrator_delete",
            actor_qq="90009",
            is_admin=True,
        )

        self.assertEqual("failed", self.store.get_job(job_ids[1]).state)
        self.assertEqual("", self.store.get_job(job_ids[1]).text)
        self.assertEqual("ready", self.store.get_job(job_ids[2]).state)
        self.assertEqual(group_b_marker, self.store.get_job(job_ids[2]).text)
        self.assertEqual(
            group_b_marker,
            self.store.get_claim(group_b_claim.id).source_excerpt,
        )
        self.assertEqual(
            group_b_marker,
            self.store.list_evidence(group_b_claim.id)[0].source_excerpt,
        )
        self.assertIn(
            group_b_claim.id,
            {
                claim.id
                for claim in self.store.search_claims(group_b_marker)
            },
        )

    def test_physical_delete_authorization_is_enforced_in_transaction(self):
        claim, _ = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="likes",
            value="atomic-owner-authorization",
            memory_type="preference",
            modality="asserted",
            source_kind="message",
            source_message_id="atomic-owner-authorization-message",
            source_excerpt="atomic-owner-authorization",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="atomic-owner-authorization-claim",
        )

        with self.assertRaises(PermissionError):
            self.store.delete_claim_physically_with_outcome(
                claim.id,
                reason="private_privacy_delete",
                actor_qq="10002",
                is_admin=False,
            )

        self.assertIsNotNone(self.store.get_claim(claim.id))
        with closing(sqlite3.connect(self.path)) as connection:
            audit_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM memory_deletion_audit
                WHERE claim_id = ?
                """,
                (claim.id,),
            ).fetchone()[0]
        self.assertEqual(0, audit_count)

    def test_empty_source_id_delete_does_not_scrub_unassociated_rows(self):
        target, _ = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="likes",
            value="empty-source-target",
            memory_type="preference",
            modality="asserted",
            source_kind="message",
            source_message_id="",
            source_excerpt="empty-source-target",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="empty-source-target",
        )
        sibling, _ = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="dislikes",
            value="empty-source-unrelated",
            memory_type="preference",
            modality="asserted",
            source_kind="message",
            source_message_id="",
            source_excerpt="empty-source-unrelated",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="empty-source-unrelated",
        )
        job_id, _ = self.store.create_job(
            private_event(
                message_id="",
                text="unassociated empty-id job body",
            )
        )
        self.store.mark_job_ready(job_id)

        self.store.delete_claim_physically(
            target.id,
            reason="private_privacy_delete",
            actor_qq="10001",
            is_admin=False,
        )

        self.assertEqual(
            "empty-source-unrelated",
            self.store.get_claim(sibling.id).source_excerpt,
        )
        self.assertEqual("ready", self.store.get_job(job_id).state)
        self.assertEqual(
            "unassociated empty-id job body",
            self.store.get_job(job_id).text,
        )

    def test_physical_delete_waits_for_concurrent_reader_before_success(self):
        marker = "CONCURRENT_PRIVATE_MARKER_93ad"
        claim, _ = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="likes",
            value=marker,
            memory_type="preference",
            modality="asserted",
            source_kind="message",
            source_message_id="concurrent-private-message",
            source_excerpt=marker,
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="concurrent-private-claim",
        )
        reader = sqlite3.connect(self.path)
        reader.execute("BEGIN")
        self.assertEqual(
            marker,
            reader.execute(
                "SELECT value FROM memory_claims WHERE id = ?",
                (claim.id,),
            ).fetchone()[0],
        )

        with ThreadPoolExecutor(max_workers=1) as pool:
            deletion = pool.submit(
                self.store.delete_claim_physically,
                claim.id,
                reason="user_forget",
                actor_qq="10001",
                is_admin=False,
            )
            try:
                deadline = time.monotonic() + 1.0
                while self.store.get_claim(claim.id) is not None:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)

                self.assertFalse(deletion.done())
                self.assertEqual(
                    marker,
                    reader.execute(
                        "SELECT value FROM memory_claims WHERE id = ?",
                        (claim.id,),
                    ).fetchone()[0],
                )
            finally:
                reader.rollback()
                reader.close()

            self.assertTrue(deletion.result(timeout=2.0))

        marker_bytes = marker.encode("utf-8")
        for database_file in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            if database_file.exists():
                self.assertNotIn(marker_bytes, database_file.read_bytes())

    def test_old_sqlite_uses_physical_delete_fallback_without_fts_command(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "old-compatible.sqlite3"
            store = MemoryStore(path)
            with mock.patch.object(sqlite3, "sqlite_version_info", (3, 41, 2)):
                store.initialize()

            with closing(sqlite3.connect(path)) as connection:
                secure_delete_config = connection.execute(
                    """
                    SELECT v
                    FROM memory_fts_config
                    WHERE k = 'secure-delete'
                    """
                ).fetchone()
            self.assertIsNone(secure_delete_config)

            marker = "OLD_SQLITE_PRIVATE_MARKER_14c8"
            claim, _ = store.create_claim(
                scope_type="private",
                scope_id="10001",
                speaker_qq="10001",
                subject_type="qq_user",
                subject_id="10001",
                predicate="likes",
                value=marker,
                memory_type="preference",
                modality="asserted",
                source_kind="message",
                source_message_id="old-sqlite-message",
                source_excerpt=marker,
                extraction_confidence="high",
                attribution_confidence="high",
                truth_confidence="medium",
                dedupe_key="old-sqlite-private-claim",
            )
            self.assertTrue(
                store.delete_claim_physically(
                    claim.id,
                    reason="user_forget",
                    actor_qq="10001",
                    is_admin=False,
                )
            )
            self.assertNotIn(marker.encode("utf-8"), path.read_bytes())

    def test_modern_sqlite_delete_does_not_run_full_fts_optimize(self):
        claim, _ = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="likes",
            value="modern secure delete",
            memory_type="preference",
            modality="asserted",
            source_kind="message",
            source_message_id="modern-delete-message",
            source_excerpt="modern secure delete",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="modern-delete-claim",
        )
        statements = []
        original_connect = sqlite3.connect

        def traced_connect(*args, **kwargs):
            connection = original_connect(*args, **kwargs)
            connection.set_trace_callback(statements.append)
            return connection

        with mock.patch.object(sqlite3, "connect", side_effect=traced_connect):
            self.assertTrue(
                self.store.delete_claim_physically(
                    claim.id,
                    reason="user_forget",
                    actor_qq="10001",
                    is_admin=False,
                )
            )

        optimize = [
            statement
            for statement in statements
            if "values ('optimize')" in statement.lower()
        ]
        self.assertEqual([], optimize)

    def test_old_sqlite_runs_fts_optimize_after_delete_transaction_commits(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "old-cleanup.sqlite3"
            store = MemoryStore(path)
            with mock.patch.object(sqlite3, "sqlite_version_info", (3, 41, 2)):
                store.initialize()
            claim, _ = store.create_claim(
                scope_type="private",
                scope_id="10001",
                speaker_qq="10001",
                subject_type="qq_user",
                subject_id="10001",
                predicate="likes",
                value="old fallback cleanup",
                memory_type="preference",
                modality="asserted",
                source_kind="message",
                source_message_id="old-cleanup-message",
                source_excerpt="old fallback cleanup",
                extraction_confidence="high",
                attribution_confidence="high",
                truth_confidence="medium",
                dedupe_key="old-cleanup-claim",
            )
            statements = []
            connection_number = 0
            original_connect = sqlite3.connect

            def traced_connect(*args, **kwargs):
                nonlocal connection_number
                connection = original_connect(*args, **kwargs)
                connection_number += 1
                current = connection_number
                connection.set_trace_callback(
                    lambda statement: statements.append((current, statement))
                )
                return connection

            with mock.patch.object(
                sqlite3,
                "connect",
                side_effect=traced_connect,
            ):
                self.assertTrue(
                    store.delete_claim_physically(
                        claim.id,
                        reason="user_forget",
                        actor_qq="10001",
                        is_admin=False,
                    )
                )

        delete_transaction_connections = {
            number
            for number, statement in statements
            if statement.strip().lower() == "begin immediate"
        }
        optimize_connections = {
            number
            for number, statement in statements
            if "values ('optimize')" in statement.lower()
        }
        self.assertEqual(1, len(delete_transaction_connections))
        self.assertEqual(1, len(optimize_connections))
        self.assertTrue(
            delete_transaction_connections.isdisjoint(optimize_connections)
        )

    def test_cleanup_does_not_rewrite_already_empty_old_job_payload(self):
        job_id, _ = self.store.create_job(
            private_event(
                message_id="already-empty-old-job",
                text="ordinary body",
            )
        )
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                UPDATE memory_jobs
                SET payload_json = '{}',
                    state = 'done',
                    updated_at = '2000-01-01T00:00:00+00:00'
                WHERE id = ?
                """,
                (job_id,),
            )
            connection.execute(
                """
                CREATE TRIGGER reject_empty_payload_rewrite
                BEFORE UPDATE OF payload_json ON memory_jobs
                WHEN OLD.payload_json = '{}' AND NEW.payload_json = '{}'
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'already-empty payload was rewritten'
                    );
                END
                """
            )
            connection.commit()

        self.assertEqual(
            0,
            self.store.cleanup_old_jobs_and_excerpts(days=90),
        )

    def test_archived_privacy_cleanup_optimizes_old_fts_only_once(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "old-archive-maintenance.sqlite3"
            store = MemoryStore(path)
            with mock.patch.object(
                sqlite3,
                "sqlite_version_info",
                (3, 41, 2),
            ):
                store.initialize()
            claim, _ = store.create_claim(
                scope_type="private",
                scope_id="10001",
                speaker_qq="10001",
                subject_type="qq_user",
                subject_id="10001",
                predicate="likes",
                value="archived maintenance value",
                memory_type="preference",
                modality="asserted",
                source_kind="message",
                source_message_id="archived-maintenance-message",
                source_excerpt="archived maintenance excerpt",
                extraction_confidence="high",
                attribution_confidence="high",
                truth_confidence="medium",
                dedupe_key="archived-maintenance-claim",
                status="archived",
            )
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    """
                    UPDATE memory_claims
                    SET created_at = '2000-01-01T00:00:00+00:00'
                    WHERE id = ?
                    """,
                    (claim.id,),
                )
                connection.commit()

            with mock.patch.object(
                store,
                "_optimize_fts_for_privacy",
                wraps=store._optimize_fts_for_privacy,
            ) as optimize:
                store.cleanup_old_jobs_and_excerpts(days=90)
                store.cleanup_old_jobs_and_excerpts(days=90)

            self.assertEqual(1, optimize.call_count)

    def test_archived_privacy_cleanup_retries_optimize_after_failure(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "old-archive-retry.sqlite3"
            store = MemoryStore(path)
            with mock.patch.object(
                sqlite3,
                "sqlite_version_info",
                (3, 41, 2),
            ):
                store.initialize()
            claim, _ = store.create_claim(
                scope_type="private",
                scope_id="10001",
                speaker_qq="10001",
                subject_type="qq_user",
                subject_id="10001",
                predicate="likes",
                value="archived retry value",
                memory_type="preference",
                modality="asserted",
                source_kind="message",
                source_message_id="archived-retry-message",
                source_excerpt="archived retry excerpt",
                extraction_confidence="high",
                attribution_confidence="high",
                truth_confidence="medium",
                dedupe_key="archived-retry-claim",
                status="archived",
            )
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    """
                    UPDATE memory_claims
                    SET created_at = '2000-01-01T00:00:00+00:00'
                    WHERE id = ?
                    """,
                    (claim.id,),
                )
                connection.commit()

            with mock.patch.object(
                store,
                "_optimize_fts_for_privacy",
                side_effect=[
                    RuntimeError("forced optimize failure"),
                    None,
                ],
            ) as optimize:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "forced optimize failure",
                ):
                    store.cleanup_old_jobs_and_excerpts(days=90)
                store.cleanup_old_jobs_and_excerpts(days=90)
                store.cleanup_old_jobs_and_excerpts(days=90)

            self.assertEqual(2, optimize.call_count)

    def test_physical_delete_fails_bounded_if_reader_never_releases(self):
        marker = "BLOCKED_PRIVATE_MARKER_6d31"
        claim, _ = self.store.create_claim(
            scope_type="private",
            scope_id="10001",
            speaker_qq="10001",
            subject_type="qq_user",
            subject_id="10001",
            predicate="likes",
            value=marker,
            memory_type="preference",
            modality="asserted",
            source_kind="message",
            source_message_id="blocked-private-message",
            source_excerpt=marker,
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="medium",
            dedupe_key="blocked-private-claim",
        )
        reader = sqlite3.connect(self.path)
        reader.execute("BEGIN")
        reader.execute(
            "SELECT value FROM memory_claims WHERE id = ?",
            (claim.id,),
        ).fetchone()
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(RuntimeError, "active readers"):
                self.store.delete_claim_physically(
                    claim.id,
                    reason="user_forget",
                    actor_qq="10001",
                    is_admin=False,
                )
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertEqual(
                marker,
                reader.execute(
                    "SELECT value FROM memory_claims WHERE id = ?",
                    (claim.id,),
                ).fetchone()[0],
            )
        finally:
            reader.rollback()
            reader.close()

        self.assertFalse(
            self.store.delete_claim_physically(
                claim.id,
                reason="user_forget",
                actor_qq="10001",
                is_admin=False,
            )
        )
        marker_bytes = marker.encode("utf-8")
        for database_file in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            if database_file.exists():
                self.assertNotIn(marker_bytes, database_file.read_bytes())


if __name__ == "__main__":
    unittest.main()
