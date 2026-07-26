from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.memory.models import (
    MemoryClaim,
    MemoryContext,
    MemoryEvidence,
    MemoryEvent,
    MemoryJob,
    MemoryRelation,
)


SCHEMA_VERSION = 1
MAX_SOURCE_EXCERPT_CHARS = 500
PRIVACY_CHECKPOINT_TIMEOUT_MS = 250
PRIVACY_OPTIMIZE_TIMEOUT_MS = 250
_DATA_URL_PATTERN = re.compile(
    r"data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=\r\n]+",
    re.IGNORECASE,
)
_BARE_IMAGE_BASE64_PATTERN = re.compile(
    r"(?:iVBORw0KGgo|/9j/|R0lGOD(?:lh|dh)|UklGR)[a-z0-9+/=]{12,}",
    re.IGNORECASE,
)
_MIME_WRAPPED_BASE64_PATTERN = re.compile(
    r"(?<![a-z0-9+/=])"
    r"[a-z0-9+/]{16,}={0,2}"
    r"(?:\r?\n[a-z0-9+/]{2,}={0,2})+"
    r"(?=$|[\r\n，；])",
    re.IGNORECASE,
)
_BINARY_BASE64_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"BM",
    b"II*\x00",
    b"MM\x00*",
    b"PK\x03\x04",
    b"%PDF-",
    b"\x1f\x8b",
    b"Rar!\x1a\x07",
    b"7z\xbc\xaf'\x1c",
)
_GENERIC_BASE64_PATTERN = re.compile(
    r"(?<![a-z0-9+/])[a-z0-9+/]{48,}={0,2}(?![a-z0-9+/=])",
    re.IGNORECASE,
)
_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----"
    r".*?"
    r"-----END [^-\r\n]*PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
_BEARER_PATTERN = re.compile(
    r"\bBearer\s+[a-z0-9._~+/=-]+",
    re.IGNORECASE,
)
_PAYMENT_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?:\b(?:payment[_ -]?(?:token|credential)|cvv|cvc)\b"
    r"|支付凭据|支付密码|银行卡号)"
    r"\s*(?:[:=：]|\bis\b|是|为)\s*[^\s,;，；]+"
)
_COOKIE_HEADER_PATTERN = re.compile(
    r"(?im)\b(?:cookie|set-cookie)\s*[:=]\s*[^\r\n，；]+"
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?:\b(?:api[_ -]?key|secret|access[_ -]?token|token|password|"
    r"passwd|credential|cookie|otp|verification[_ -]?code|authorization)\b"
    r"|api\s*密钥|密钥|密码|口令|验证码|校验码)"
    r"\s*(?:[:=：]|\bis\b|是|为)\s*[^\s,;，；]+"
)
_SECRET_TOKEN_PATTERN = re.compile(
    r"\b(?:sk-[a-z0-9_-]{8,}|gh[pousr]_[a-z0-9]{20,}|AKIA[A-Z0-9]{16})\b",
    re.IGNORECASE,
)
_PAYMENT_NUMBER_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_AUDIT_REASON_PATTERN = re.compile(r"[a-z][a-z0-9_.:-]{0,63}")


def _utc_now() -> str:
    return _format_utc(datetime.now(timezone.utc))


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _normalize_retry_at(value: str) -> str:
    candidate = value.strip()
    if candidate.endswith(("Z", "z")):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        raise ValueError(
            "retry_at must be a valid timezone-aware ISO timestamp"
        ) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("retry_at must include a timezone offset")
    return _format_utc(parsed)


def _passes_luhn(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _redact_mime_wrapped_base64(match: re.Match[str]) -> str:
    candidate = match.group(0)
    lines = candidate.splitlines(keepends=True)
    first_width = len(lines[0].rstrip("\r\n"))
    prefix_line_count = 1
    saw_short_line = False
    for line in lines[1:]:
        width = len(line.rstrip("\r\n"))
        if not saw_short_line and width == first_width:
            prefix_line_count += 1
        elif not saw_short_line and width < first_width:
            prefix_line_count += 1
            saw_short_line = True
        else:
            break

    prefix_end = sum(len(line) for line in lines[:prefix_line_count])
    while prefix_end and candidate[prefix_end - 1] in "\r\n":
        prefix_end -= 1
    encoded = candidate[:prefix_end].replace("\r", "").replace("\n", "")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return candidate
    if not (
        decoded.startswith(_BINARY_BASE64_SIGNATURES)
        or (decoded.startswith(b"RIFF") and decoded[8:12] == b"WEBP")
    ):
        return candidate
    return "[redacted:image-data]" + candidate[prefix_end:]


def _redact_forbidden_payload_data(text: str) -> str:
    text = _PRIVATE_KEY_PATTERN.sub("[redacted:credential]", text)
    text = _DATA_URL_PATTERN.sub("[redacted:image-data]", text)
    text = _MIME_WRAPPED_BASE64_PATTERN.sub(
        _redact_mime_wrapped_base64,
        text,
    )
    text = _BARE_IMAGE_BASE64_PATTERN.sub("[redacted:image-data]", text)
    text = _GENERIC_BASE64_PATTERN.sub("[redacted:image-data]", text)
    text = _BEARER_PATTERN.sub("[redacted:credential]", text)
    text = _PAYMENT_ASSIGNMENT_PATTERN.sub(
        "[redacted:payment-data]",
        text,
    )
    text = _COOKIE_HEADER_PATTERN.sub("[redacted:credential]", text)
    text = _CREDENTIAL_ASSIGNMENT_PATTERN.sub(
        "[redacted:credential]",
        text,
    )
    text = _SECRET_TOKEN_PATTERN.sub("[redacted:credential]", text)
    return _PAYMENT_NUMBER_PATTERN.sub(
        lambda match: (
            "[redacted:payment-data]"
            if _passes_luhn(match.group(0))
            else match.group(0)
        ),
        text,
    )


class MemoryStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA secure_delete = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            if sqlite3.sqlite_version_info >= (3, 42, 0):
                connection.execute(
                    """
                    INSERT INTO memory_fts(memory_fts, rank)
                    VALUES ('secure-delete', 1)
                    """
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_version(version, applied_at)
                VALUES (?, ?)
                """,
                (SCHEMA_VERSION, _utc_now()),
            )

    def table_names(self) -> frozenset[str]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type IN ('table', 'view')
                """
            ).fetchall()
        return frozenset(str(row["name"]) for row in rows)

    def schema_version(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_version"
            ).fetchone()
        return int(row["version"] or 0)

    def create_job(self, event: MemoryEvent) -> tuple[int, bool]:
        scope_type, scope_id = event.context.primary_scope
        scope_key = event.context.session_key
        dedupe_key = hashlib.sha256(
            f"{scope_type}:{scope_id}\0{event.message_id}".encode("utf-8")
        ).hexdigest()
        payload = {
            "context": {
                "user_id": event.context.user_id,
                "session_key": event.context.session_key,
                "is_group": event.context.is_group,
                "group_id": event.context.group_id,
            },
            "message_id": event.message_id,
            "sequence": event.sequence,
            "text": _redact_forbidden_payload_data(event.text),
            "image_count": event.image_count,
            "mentioned_qq_ids": event.mentioned_qq_ids,
            "reply_to_message_id": event.reply_to_message_id,
            "reply_to_user_id": event.reply_to_user_id,
        }
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        timestamp = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memory_jobs(
                    dedupe_key, scope_key, sequence, payload_json, state,
                    attempts, retry_at, error_type, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'staged', 0, NULL, NULL, ?, ?)
                ON CONFLICT(dedupe_key) DO NOTHING
                """,
                (
                    dedupe_key,
                    scope_key,
                    event.sequence,
                    payload_json,
                    timestamp,
                    timestamp,
                ),
            )
            created = cursor.rowcount == 1
            row = connection.execute(
                "SELECT id FROM memory_jobs WHERE dedupe_key = ?",
                (dedupe_key,),
            ).fetchone()
        return int(row["id"]), created

    def get_job(self, job_id: int) -> MemoryJob:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM memory_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"memory job {job_id} does not exist")
        return _job_from_row(row)

    def mark_job_ready(self, job_id: int) -> None:
        self._transition_job(
            job_id,
            """
            UPDATE memory_jobs
            SET state = 'ready', retry_at = NULL, error_type = NULL,
                updated_at = ?
            WHERE id = ? AND state IN ('staged', 'retry')
            """,
        )

    def claim_next_job(self, scope_key: str) -> MemoryJob | None:
        timestamp = _utc_now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            running = connection.execute(
                """
                SELECT 1
                FROM memory_jobs
                WHERE scope_key = ? AND state = 'running'
                LIMIT 1
                """,
                (scope_key,),
            ).fetchone()
            if running is not None:
                return None
            row = connection.execute(
                """
                SELECT id, state, retry_at
                FROM memory_jobs
                WHERE scope_key = ?
                  AND state NOT IN ('done', 'failed')
                ORDER BY sequence, id
                LIMIT 1
                """,
                (scope_key,),
            ).fetchone()
            if row is None:
                return None
            if row["state"] == "staged":
                return None
            if row["state"] == "retry" and (
                row["retry_at"] is None or row["retry_at"] > timestamp
            ):
                return None
            job_id = int(row["id"])
            connection.execute(
                """
                UPDATE memory_jobs
                SET state = 'running', attempts = attempts + 1,
                    retry_at = NULL, error_type = NULL, updated_at = ?
                WHERE id = ?
                """,
                (timestamp, job_id),
            )
            claimed = connection.execute(
                "SELECT * FROM memory_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return _job_from_row(claimed)

    def complete_job(self, job_id: int) -> None:
        self._transition_job(
            job_id,
            """
            UPDATE memory_jobs
            SET state = 'done', retry_at = NULL, error_type = NULL,
                updated_at = ?
            WHERE id = ? AND state = 'running'
            """,
        )

    def fail_job(
        self,
        job_id: int,
        error_type: str,
        retry_at: str | None,
    ) -> None:
        normalized_retry_at = (
            None if retry_at is None else _normalize_retry_at(retry_at)
        )
        state = "retry" if normalized_retry_at is not None else "failed"
        timestamp = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE memory_jobs
                SET state = ?, retry_at = ?, error_type = ?, updated_at = ?
                WHERE id = ? AND state = 'running'
                """,
                (state, normalized_retry_at, error_type, timestamp, job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"memory job {job_id} is not running")

    def recover_running_jobs(self, scope_key: str | None = None) -> int:
        timestamp = _utc_now()
        clauses = ["state = 'running'"]
        parameters: list[Any] = [timestamp, timestamp]
        if scope_key is not None:
            clauses.append("scope_key = ?")
            parameters.append(scope_key)
        with self._connection() as connection:
            cursor = connection.execute(
                f"""
                UPDATE memory_jobs
                SET state = 'retry', retry_at = ?, error_type = 'abandoned',
                    updated_at = ?
                WHERE {' AND '.join(clauses)}
                """,
                parameters,
            )
        return cursor.rowcount

    def _transition_job(self, job_id: int, statement: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(statement, (_utc_now(), job_id))
            if cursor.rowcount != 1:
                raise ValueError(f"invalid state transition for memory job {job_id}")

    def create_claim(
        self,
        *,
        scope_type: str,
        scope_id: str,
        speaker_qq: str,
        subject_type: str,
        subject_id: str,
        predicate: str,
        value: str,
        memory_type: str,
        modality: str,
        source_kind: str,
        source_message_id: str,
        source_excerpt: str,
        extraction_confidence: str,
        attribution_confidence: str,
        truth_confidence: str,
        dedupe_key: str,
        status: str = "active",
        valid_from: str | None = None,
        valid_to: str | None = None,
        last_confirmed_at: str | None = None,
    ) -> tuple[MemoryClaim, bool]:
        timestamp = _utc_now()
        source_excerpt = source_excerpt[:MAX_SOURCE_EXCERPT_CHARS]
        with self._connection() as connection:
            parameters = (
                scope_type,
                scope_id,
                speaker_qq,
                subject_type,
                subject_id,
                predicate,
                value,
                memory_type,
                modality,
                source_kind,
                source_message_id,
                source_excerpt,
                extraction_confidence,
                attribution_confidence,
                truth_confidence,
                status,
                timestamp,
                valid_from,
                valid_to,
                last_confirmed_at,
                dedupe_key,
            )
            cursor = connection.execute(
                """
                INSERT INTO memory_claims(
                    scope_type, scope_id, speaker_qq, subject_type, subject_id,
                    predicate, value, memory_type, modality, source_kind,
                    source_message_id, source_excerpt, extraction_confidence,
                    attribution_confidence, truth_confidence, status,
                    created_at, valid_from, valid_to, last_confirmed_at,
                    dedupe_key
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(dedupe_key) DO NOTHING
                """,
                parameters,
            )
            created = cursor.rowcount == 1
            row = connection.execute(
                "SELECT * FROM memory_claims WHERE dedupe_key = ?",
                (dedupe_key,),
            ).fetchone()
            if created:
                _insert_fts_row(connection, row)
        return _claim_from_row(row), created

    def get_claim(self, claim_id: int) -> MemoryClaim | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM memory_claims WHERE id = ?",
                (claim_id,),
            ).fetchone()
        return None if row is None else _claim_from_row(row)

    def update_claim(self, claim_id: int, **changes: Any) -> MemoryClaim:
        if not changes:
            existing = self.get_claim(claim_id)
            if existing is None:
                raise KeyError(f"memory claim {claim_id} does not exist")
            return existing
        invalid = set(changes) - _MUTABLE_CLAIM_FIELDS
        if invalid:
            raise ValueError(
                f"unsupported memory claim fields: {', '.join(sorted(invalid))}"
            )
        if "source_excerpt" in changes:
            changes["source_excerpt"] = str(changes["source_excerpt"])[
                :MAX_SOURCE_EXCERPT_CHARS
            ]
        assignments = ", ".join(f"{name} = ?" for name in changes)
        parameters = [changes[name] for name in changes]
        parameters.append(claim_id)
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE memory_claims SET {assignments} WHERE id = ?",
                parameters,
            )
            if cursor.rowcount != 1:
                raise KeyError(f"memory claim {claim_id} does not exist")
            row = connection.execute(
                "SELECT * FROM memory_claims WHERE id = ?",
                (claim_id,),
            ).fetchone()
            connection.execute(
                "DELETE FROM memory_fts WHERE rowid = ?",
                (claim_id,),
            )
            _insert_fts_row(connection, row)
        return _claim_from_row(row)

    def delete_claim_physically(self, claim_id: int, *, reason: str) -> bool:
        if _AUDIT_REASON_PATTERN.fullmatch(reason) is None:
            raise ValueError("deletion reason must be a body-free audit code")
        deleted = False
        needs_checkpoint = False
        needs_fts_optimize = False
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            needs_fts_optimize = not _fts_secure_delete_enabled(connection)
            exists = connection.execute(
                "SELECT 1 FROM memory_claims WHERE id = ?",
                (claim_id,),
            ).fetchone()
            if exists is None:
                needs_checkpoint = (
                    connection.execute(
                        """
                        SELECT 1
                        FROM memory_deletion_audit
                        WHERE claim_id = ?
                        LIMIT 1
                        """,
                        (claim_id,),
                    ).fetchone()
                    is not None
                )
            else:
                connection.execute(
                    """
                    INSERT INTO memory_deletion_audit(claim_id, reason, deleted_at)
                    VALUES (?, ?, ?)
                    """,
                    (claim_id, reason, _utc_now()),
                )
                connection.execute(
                    "DELETE FROM memory_fts WHERE rowid = ?",
                    (claim_id,),
                )
                connection.execute(
                    "DELETE FROM memory_claims WHERE id = ?",
                    (claim_id,),
                )
                deleted = True
                needs_checkpoint = True
        if needs_checkpoint:
            if needs_fts_optimize:
                self._optimize_fts_for_privacy()
            self._checkpoint_wal()
        return deleted

    def _optimize_fts_for_privacy(self) -> None:
        deadline = time.monotonic() + PRIVACY_OPTIMIZE_TIMEOUT_MS / 1000
        with self._connection() as connection:
            connection.execute(
                f"PRAGMA busy_timeout = {PRIVACY_OPTIMIZE_TIMEOUT_MS}"
            )
            connection.set_progress_handler(
                lambda: int(time.monotonic() >= deadline),
                1000,
            )
            try:
                connection.execute(
                    "INSERT INTO memory_fts(memory_fts) VALUES ('optimize')"
                )
            except sqlite3.OperationalError as error:
                if "interrupt" not in str(error).lower():
                    raise
                raise RuntimeError(
                    "FTS privacy cleanup exceeded time limit"
                ) from error
            finally:
                connection.set_progress_handler(None, 0)

    def _checkpoint_wal(self) -> None:
        with self._connection() as connection:
            connection.execute(
                f"PRAGMA busy_timeout = {PRIVACY_CHECKPOINT_TIMEOUT_MS}"
            )
            result = connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
        if result is not None and int(result[0]) != 0:
            raise RuntimeError(
                "active readers prevent physical memory deletion checkpoint"
            )

    def add_evidence(
        self,
        claim_id: int,
        *,
        source_kind: str,
        source_message_id: str,
        source_excerpt: str,
    ) -> tuple[int, bool]:
        source_excerpt = source_excerpt[:MAX_SOURCE_EXCERPT_CHARS]
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memory_evidence(
                    claim_id, source_kind, source_message_id,
                    source_excerpt, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(claim_id, source_kind, source_message_id) DO NOTHING
                """,
                (
                    claim_id,
                    source_kind,
                    source_message_id,
                    source_excerpt,
                    _utc_now(),
                ),
            )
            created = cursor.rowcount == 1
            row = connection.execute(
                """
                SELECT id
                FROM memory_evidence
                WHERE claim_id = ? AND source_kind = ? AND source_message_id = ?
                """,
                (claim_id, source_kind, source_message_id),
            ).fetchone()
        return int(row["id"]), created

    def list_evidence(self, claim_id: int) -> tuple[MemoryEvidence, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM memory_evidence
                WHERE claim_id = ?
                ORDER BY id
                """,
                (claim_id,),
            ).fetchall()
        return tuple(_evidence_from_row(row) for row in rows)

    def delete_evidence(self, evidence_id: int) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM memory_evidence WHERE id = ?",
                (evidence_id,),
            )
        return cursor.rowcount == 1

    def add_relation(
        self,
        source_claim_id: int,
        target_claim_id: int,
        relation_type: str,
    ) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memory_relations(
                    source_claim_id, target_claim_id, relation_type, created_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(
                    source_claim_id, target_claim_id, relation_type
                ) DO NOTHING
                """,
                (
                    source_claim_id,
                    target_claim_id,
                    relation_type,
                    _utc_now(),
                ),
            )
        return cursor.rowcount == 1

    def list_relations(self, claim_id: int) -> tuple[MemoryRelation, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM memory_relations
                WHERE source_claim_id = ? OR target_claim_id = ?
                ORDER BY created_at, source_claim_id, target_claim_id
                """,
                (claim_id, claim_id),
            ).fetchall()
        return tuple(_relation_from_row(row) for row in rows)

    def delete_relation(
        self,
        source_claim_id: int,
        target_claim_id: int,
        relation_type: str,
    ) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM memory_relations
                WHERE source_claim_id = ? AND target_claim_id = ?
                  AND relation_type = ?
                """,
                (source_claim_id, target_claim_id, relation_type),
            )
        return cursor.rowcount == 1

    def search_claims(
        self,
        query: str,
        *,
        scope_type: str | None = None,
        scope_id: str | None = None,
        limit: int = 12,
    ) -> tuple[MemoryClaim, ...]:
        if not query.strip() or limit <= 0:
            return ()
        clauses = ["memory_fts MATCH ?"]
        parameters: list[Any] = [_fts_phrase(query)]
        if scope_type is not None:
            clauses.append("claim.scope_type = ?")
            parameters.append(scope_type)
        if scope_id is not None:
            clauses.append("claim.scope_id = ?")
            parameters.append(scope_id)
        parameters.append(limit)
        sql = f"""
            SELECT claim.*
            FROM memory_fts
            JOIN memory_claims AS claim ON claim.id = memory_fts.rowid
            WHERE {' AND '.join(clauses)}
            ORDER BY bm25(memory_fts), claim.id
            LIMIT ?
        """
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return tuple(_claim_from_row(row) for row in rows)


def _fts_phrase(value: str) -> str:
    return '"' + value.strip().replace('"', '""') + '"'


def _fts_secure_delete_enabled(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        """
        SELECT v
        FROM memory_fts_config
        WHERE k = 'secure-delete'
        """
    ).fetchone()
    return row is not None and int(row["v"]) == 1


def _insert_fts_row(
    connection: sqlite3.Connection,
    claim: sqlite3.Row,
) -> None:
    connection.execute(
        """
        INSERT INTO memory_fts(rowid, claim_id, predicate, value, source_excerpt)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            int(claim["id"]),
            str(claim["id"]),
            claim["predicate"],
            claim["value"],
            claim["source_excerpt"],
        ),
    )


def _claim_from_row(row: sqlite3.Row) -> MemoryClaim:
    return MemoryClaim(**{field: row[field] for field in MemoryClaim.__dataclass_fields__})


def _job_from_row(row: sqlite3.Row) -> MemoryJob:
    payload = json.loads(row["payload_json"])
    context_data = payload.get("context")
    if context_data is None:
        context = _legacy_job_context(str(row["scope_key"]))
    else:
        context = MemoryContext(
            user_id=str(context_data["user_id"]),
            session_key=str(context_data["session_key"]),
            is_group=bool(context_data["is_group"]),
            group_id=(
                None
                if context_data.get("group_id") is None
                else str(context_data["group_id"])
            ),
        )
    return MemoryJob(
        id=int(row["id"]),
        dedupe_key=str(row["dedupe_key"]),
        scope_key=str(row["scope_key"]),
        sequence=int(row["sequence"]),
        payload_json=str(row["payload_json"]),
        context=context,
        message_id=str(payload["message_id"]),
        text=str(payload["text"]),
        image_count=int(payload.get("image_count", 0)),
        mentioned_qq_ids=tuple(
            str(value) for value in payload.get("mentioned_qq_ids", ())
        ),
        reply_to_message_id=payload.get("reply_to_message_id"),
        reply_to_user_id=payload.get("reply_to_user_id"),
        state=str(row["state"]),
        attempts=int(row["attempts"]),
        retry_at=row["retry_at"],
        error_type=row["error_type"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _legacy_job_context(scope_key: str) -> MemoryContext:
    parts = scope_key.split(":")
    is_group = parts[0] == "group"
    return MemoryContext(
        user_id="" if is_group else (parts[-1] if len(parts) > 1 else ""),
        session_key=scope_key,
        is_group=is_group,
        group_id=parts[1] if is_group and len(parts) > 1 else None,
    )


def _evidence_from_row(row: sqlite3.Row) -> MemoryEvidence:
    return MemoryEvidence(
        **{field: row[field] for field in MemoryEvidence.__dataclass_fields__}
    )


def _relation_from_row(row: sqlite3.Row) -> MemoryRelation:
    return MemoryRelation(
        **{field: row[field] for field in MemoryRelation.__dataclass_fields__}
    )


_MUTABLE_CLAIM_FIELDS = frozenset(
    {
        "scope_type",
        "scope_id",
        "speaker_qq",
        "subject_type",
        "subject_id",
        "predicate",
        "value",
        "memory_type",
        "modality",
        "source_kind",
        "source_message_id",
        "source_excerpt",
        "extraction_confidence",
        "attribution_confidence",
        "truth_confidence",
        "status",
        "valid_from",
        "valid_to",
        "last_confirmed_at",
        "dedupe_key",
    }
)


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_claims (
        id INTEGER PRIMARY KEY,
        scope_type TEXT NOT NULL
            CHECK (scope_type IN ('private', 'group', 'global', 'bot')),
        scope_id TEXT NOT NULL,
        speaker_qq TEXT NOT NULL,
        subject_type TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        predicate TEXT NOT NULL,
        value TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        modality TEXT NOT NULL,
        source_kind TEXT NOT NULL,
        source_message_id TEXT NOT NULL,
        source_excerpt TEXT NOT NULL,
        extraction_confidence TEXT NOT NULL
            CHECK (extraction_confidence IN ('low', 'medium', 'high')),
        attribution_confidence TEXT NOT NULL
            CHECK (attribution_confidence IN ('low', 'medium', 'high')),
        truth_confidence TEXT NOT NULL
            CHECK (truth_confidence IN ('low', 'medium', 'high')),
        status TEXT NOT NULL
            CHECK (status IN (
                'active', 'disputed', 'superseded', 'retracted', 'archived'
            )),
        created_at TEXT NOT NULL,
        valid_from TEXT,
        valid_to TEXT,
        last_confirmed_at TEXT,
        dedupe_key TEXT NOT NULL UNIQUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_evidence (
        id INTEGER PRIMARY KEY,
        claim_id INTEGER NOT NULL
            REFERENCES memory_claims(id) ON DELETE CASCADE,
        source_kind TEXT NOT NULL,
        source_message_id TEXT NOT NULL,
        source_excerpt TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(claim_id, source_kind, source_message_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_relations (
        source_claim_id INTEGER NOT NULL
            REFERENCES memory_claims(id) ON DELETE CASCADE,
        target_claim_id INTEGER NOT NULL
            REFERENCES memory_claims(id) ON DELETE CASCADE,
        relation_type TEXT NOT NULL
            CHECK (relation_type IN (
                'supports', 'contradicts', 'supersedes', 'retracts'
            )),
        created_at TEXT NOT NULL,
        CHECK (source_claim_id <> target_claim_id),
        UNIQUE(source_claim_id, target_claim_id, relation_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_jobs (
        id INTEGER PRIMARY KEY,
        dedupe_key TEXT NOT NULL UNIQUE,
        scope_key TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        state TEXT NOT NULL
            CHECK (state IN (
                'staged', 'ready', 'running', 'retry', 'done', 'failed'
            )),
        attempts INTEGER NOT NULL DEFAULT 0,
        retry_at TEXT,
        error_type TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_deletion_audit (
        id INTEGER PRIMARY KEY,
        claim_id INTEGER NOT NULL,
        reason TEXT NOT NULL,
        deleted_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS memory_jobs_claim_idx
    ON memory_jobs(scope_key, state, retry_at, sequence, id)
    """,
    """
    CREATE INDEX IF NOT EXISTS memory_claims_scope_idx
    ON memory_claims(scope_type, scope_id, status)
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        claim_id UNINDEXED,
        predicate,
        value,
        source_excerpt,
        tokenize = 'unicode61'
    )
    """,
)
