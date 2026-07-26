from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.memory.models import (
    MemoryClaim,
    MemoryEvidence,
    MemoryEvent,
    MemoryJob,
    MemoryRelation,
)


SCHEMA_VERSION = 1
MAX_SOURCE_EXCERPT_CHARS = 500
_DATA_URL_PATTERN = re.compile(
    r"data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=\r\n]+",
    re.IGNORECASE,
)
_BARE_IMAGE_BASE64_PATTERN = re.compile(
    r"(?:iVBORw0KGgo|/9j/|R0lGOD(?:lh|dh)|UklGR)[a-z0-9+/=\r\n]{12,}",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|secret|access[_-]?token|password|passwd|credential)"
    r"\b\s*[:=]\s*[^\s,;，；]+"
)
_SECRET_TOKEN_PATTERN = re.compile(r"\bsk-[a-z0-9_-]{8,}\b", re.IGNORECASE)
_PAYMENT_NUMBER_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_AUDIT_REASON_PATTERN = re.compile(r"[a-z][a-z0-9_.:-]{0,63}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _redact_forbidden_payload_data(text: str) -> str:
    text = _DATA_URL_PATTERN.sub("[redacted:image-data]", text)
    text = _BARE_IMAGE_BASE64_PATTERN.sub("[redacted:image-data]", text)
    text = _SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}=[redacted:credential]",
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
        scope_key = f"{scope_type}:{scope_id}"
        dedupe_key = hashlib.sha256(
            f"{scope_key}\0{event.message_id}".encode("utf-8")
        ).hexdigest()
        payload = {
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
            row = connection.execute(
                """
                SELECT id
                FROM memory_jobs
                WHERE scope_key = ?
                  AND (
                    state = 'ready'
                    OR (state = 'retry' AND retry_at IS NOT NULL AND retry_at <= ?)
                  )
                ORDER BY sequence, id
                LIMIT 1
                """,
                (scope_key, timestamp),
            ).fetchone()
            if row is None:
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
        state = "retry" if retry_at is not None else "failed"
        timestamp = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE memory_jobs
                SET state = ?, retry_at = ?, error_type = ?, updated_at = ?
                WHERE id = ? AND state = 'running'
                """,
                (state, retry_at, error_type, timestamp, job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"memory job {job_id} is not running")

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
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM memory_claims WHERE id = ?",
                (claim_id,),
            ).fetchone()
            if exists is None:
                return False
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
        return True

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
    return MemoryJob(**{field: row[field] for field in MemoryJob.__dataclass_fields__})


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
