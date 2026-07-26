import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from src.memory.models import CandidateClaim, MemoryContext, MemoryEvent
from src.memory.store import MemoryStore


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
                "schema_version",
                "memory_fts",
            }.issubset(names)
        )
        self.assertEqual(1, self.store.schema_version())
        with closing(sqlite3.connect(self.path)) as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual("wal", journal_mode.lower())

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
            self.store.delete_claim_physically(first.id, reason="user_forget")
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
            self.store.delete_claim_physically(claim.id, reason="user_forget")
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


if __name__ == "__main__":
    unittest.main()
