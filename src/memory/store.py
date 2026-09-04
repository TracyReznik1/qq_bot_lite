from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from src.memory.models import (
    MemoryClaim,
    MemoryContext,
    MemoryEvidence,
    MemoryEvent,
    MemoryJob,
    MemoryRelation,
    PhysicalDeleteOutcome,
)
from src.memory.privacy import redact_hard_secrets


SCHEMA_VERSION = 4
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
_AUDIT_REASON_PATTERN = re.compile(r"[a-z][a-z0-9_.:-]{0,63}")
_POLICY_DEDUPE_KEY_PATTERN = re.compile(r"policy:[0-9a-f]{64}")


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


def _redact_mime_wrapped_base64(match: re.Match[str]) -> str:
    candidate = match.group(0)
    lines = candidate.splitlines(keepends=True)
    first_line = lines[0].rstrip("\r\n")
    first_width = len(first_line)
    prefix_line_count = 1
    full_width_line_count = 1
    if not first_line.endswith("="):
        for line in lines[1:]:
            content = line.rstrip("\r\n")
            width = len(content)
            if content.endswith("="):
                prefix_line_count += 1
                break
            if width == first_width:
                prefix_line_count += 1
                full_width_line_count += 1
            elif width < first_width and full_width_line_count >= 2:
                prefix_line_count += 1
                break
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
    text = redact_hard_secrets(text)
    text = _DATA_URL_PATTERN.sub("[redacted:image-data]", text)
    text = _MIME_WRAPPED_BASE64_PATTERN.sub(
        _redact_mime_wrapped_base64,
        text,
    )
    text = _BARE_IMAGE_BASE64_PATTERN.sub("[redacted:image-data]", text)
    text = _GENERIC_BASE64_PATTERN.sub("[redacted:image-data]", text)
    return text


@dataclass(frozen=True)
class _DeleteMutation:
    deleted: bool
    needs_checkpoint: bool
    needs_fts_optimize: bool


def _physical_delete_is_authorized(
    *,
    scope_type: str,
    scope_id: str,
    actor_qq: str,
    is_admin: bool,
) -> bool:
    return bool(
        is_admin
        or (
            scope_type == "private"
            and scope_id == str(actor_qq)
        )
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
            _migrate_memory_claims_privacy_maintenance(connection)
            _migrate_memory_jobs_source_message_id(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS memory_jobs_source_message_idx
                ON memory_jobs(source_message_id, state, id)
                """
            )
            connection.execute(
                """
                INSERT INTO memory_claim_id_sequence(singleton, last_id)
                VALUES (
                    1,
                    (
                        SELECT COALESCE(MAX(candidate_id), 0)
                        FROM (
                            SELECT id AS candidate_id
                            FROM memory_claims
                            UNION ALL
                            SELECT claim_id AS candidate_id
                            FROM memory_deletion_audit
                            UNION ALL
                            SELECT claim_id AS candidate_id
                            FROM memory_pending_privacy_cleanup
                        )
                    )
                )
                ON CONFLICT(singleton) DO UPDATE SET
                    last_id = MAX(
                        memory_claim_id_sequence.last_id,
                        excluded.last_id
                    )
                """
            )
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

    def integrity_check(self) -> str:
        with self._connection() as connection:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row else "failed"

    def cleanup_old_jobs_and_excerpts(self, days: int = 90) -> int:
        cutoff = _format_utc(
            datetime.now(timezone.utc) - timedelta(days=days)
        )
        cleaned = 0
        maintenance_claim_ids: tuple[int, ...] = ()
        archived_claim_ids: tuple[int, ...] = ()
        needs_fts_optimize = False
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE memory_jobs
                SET payload_json = '{}'
                WHERE state = 'done' AND updated_at < ?
                  AND payload_json <> '{}'
                """,
                (cutoff,),
            )
            cleaned += cursor.rowcount
            archived_rows = connection.execute(
                """
                SELECT id, source_excerpt
                FROM memory_claims
                WHERE status = 'archived'
                  AND created_at < ?
                  AND privacy_maintenance_at IS NULL
                """,
                (cutoff,),
            ).fetchall()
            maintenance_claim_ids = tuple(
                int(row["id"]) for row in archived_rows
            )
            archived_claim_ids = tuple(
                int(row["id"])
                for row in archived_rows
                if str(row["source_excerpt"])
            )
            needs_fts_optimize = bool(
                maintenance_claim_ids
            ) and not _fts_secure_delete_enabled(connection)
            if archived_claim_ids:
                placeholders = ",".join("?" for _ in archived_claim_ids)
                cursor = connection.execute(
                    f"""
                    UPDATE memory_claims
                    SET source_excerpt = ''
                    WHERE id IN ({placeholders})
                    """,
                    archived_claim_ids,
                )
                cleaned += cursor.rowcount
                for claim_id in archived_claim_ids:
                    connection.execute(
                        "DELETE FROM memory_fts WHERE rowid = ?",
                        (claim_id,),
                    )
                    claim = connection.execute(
                        "SELECT * FROM memory_claims WHERE id = ?",
                        (claim_id,),
                    ).fetchone()
                    _insert_fts_row(connection, claim)
            cursor = connection.execute(
                """
                UPDATE memory_evidence
                SET source_excerpt = ''
                WHERE created_at < ?
                  AND source_excerpt <> ''
                  AND claim_id IN (
                      SELECT id
                      FROM memory_claims
                      WHERE status = 'archived'
                  )
                """,
                (cutoff,),
            )
            cleaned += cursor.rowcount
        if needs_fts_optimize:
            self._optimize_fts_for_privacy()
        if maintenance_claim_ids:
            placeholders = ",".join(
                "?" for _ in maintenance_claim_ids
            )
            with self._connection() as connection:
                connection.execute(
                    f"""
                    UPDATE memory_claims
                    SET privacy_maintenance_at = ?
                    WHERE id IN ({placeholders})
                      AND privacy_maintenance_at IS NULL
                    """,
                    (_utc_now(), *maintenance_claim_ids),
                )
        self._checkpoint_wal()
        return cleaned

    def create_job(self, event: MemoryEvent) -> tuple[int, bool]:
        scope_type, scope_id = event.context.primary_scope
        scope_key = f"{scope_type}:{scope_id}"
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
            "prior_dialogue_context": [
                [role, content]
                for role, content in event.prior_dialogue_context
            ],
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
                    dedupe_key, scope_key, sequence, source_message_id,
                    payload_json, state, attempts, retry_at, error_type,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'staged', 0, NULL, NULL, ?, ?)
                ON CONFLICT(dedupe_key) DO NOTHING
                """,
                (
                    dedupe_key,
                    scope_key,
                    event.sequence,
                    event.message_id,
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

    def mark_job_ready(self, job_id: int) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE memory_jobs
                SET state = 'ready', retry_at = NULL, error_type = NULL,
                    updated_at = ?
                WHERE id = ? AND state IN ('staged', 'ready', 'retry')
                """,
                (_utc_now(), job_id),
            )
        return cursor.rowcount == 1

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
        with self._connection() as connection:
            _complete_job_in_connection(connection, job_id)

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

    def recover_running_job(self, job_id: int, error_type: str) -> bool:
        timestamp = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE memory_jobs
                SET state = 'retry', retry_at = ?, error_type = ?,
                    updated_at = ?
                WHERE id = ? AND state = 'running'
                """,
                (timestamp, error_type, timestamp, job_id),
            )
        return cursor.rowcount == 1

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

    def recover_staged_jobs(self) -> int:
        timestamp = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE memory_jobs
                SET state = 'ready', retry_at = NULL, error_type = NULL,
                    updated_at = ?
                WHERE state = 'staged'
                """,
                (timestamp,),
            )
        return cursor.rowcount

    def max_job_sequence(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM memory_jobs"
            ).fetchone()
        return int(row["sequence"] or 0)

    def _transition_job(self, job_id: int, statement: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(statement, (_utc_now(), job_id))
            if cursor.rowcount != 1:
                raise ValueError(f"invalid state transition for memory job {job_id}")

    @contextmanager
    def reconciliation(self) -> Iterator["_MemoryReconciliation"]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            yield _MemoryReconciliation(connection)

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
        with self._connection() as connection:
            return _create_claim_in_connection(
                connection,
                scope_type=scope_type,
                scope_id=scope_id,
                speaker_qq=speaker_qq,
                subject_type=subject_type,
                subject_id=subject_id,
                predicate=predicate,
                value=value,
                memory_type=memory_type,
                modality=modality,
                source_kind=source_kind,
                source_message_id=source_message_id,
                source_excerpt=source_excerpt,
                extraction_confidence=extraction_confidence,
                attribution_confidence=attribution_confidence,
                truth_confidence=truth_confidence,
                dedupe_key=dedupe_key,
                status=status,
                valid_from=valid_from,
                valid_to=valid_to,
                last_confirmed_at=last_confirmed_at,
            )

    def get_claim(self, claim_id: int) -> MemoryClaim | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM memory_claims WHERE id = ?",
                (claim_id,),
            ).fetchone()
        return None if row is None else _claim_from_row(row)

    def update_claim(self, claim_id: int, **changes: Any) -> MemoryClaim:
        with self._connection() as connection:
            return _update_claim_in_connection(connection, claim_id, **changes)

    def delete_claim_physically(
        self,
        claim_id: int,
        *,
        reason: str,
        actor_qq: str,
        is_admin: bool,
    ) -> bool:
        mutation = self._delete_claim_transaction(
            claim_id,
            reason=reason,
            actor_qq=actor_qq,
            is_admin=is_admin,
        )
        self._run_privacy_maintenance(
            needs_checkpoint=mutation.needs_checkpoint,
            needs_fts_optimize=mutation.needs_fts_optimize,
        )
        if mutation.needs_checkpoint:
            self._clear_pending_privacy_cleanup(claim_id)
        return mutation.deleted

    def _delete_claim_transaction(
        self,
        claim_id: int,
        *,
        reason: str,
        actor_qq: str,
        is_admin: bool,
    ) -> _DeleteMutation:
        """Commit the row mutation and return required post-commit maintenance."""
        if _AUDIT_REASON_PATTERN.fullmatch(reason) is None:
            raise ValueError("deletion reason must be a body-free audit code")
        deleted = False
        needs_checkpoint = False
        needs_fts_optimize = False
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            needs_fts_optimize = not _fts_secure_delete_enabled(connection)
            exists = connection.execute(
                """
                SELECT scope_type, scope_id, source_message_id, dedupe_key
                FROM memory_claims
                WHERE id = ?
                """,
                (claim_id,),
            ).fetchone()
            if exists is None:
                pending = connection.execute(
                    """
                    SELECT scope_type, scope_id, needs_fts_optimize
                    FROM memory_pending_privacy_cleanup
                    WHERE claim_id = ?
                    """,
                    (claim_id,),
                ).fetchone()
                if pending is not None and not _physical_delete_is_authorized(
                    scope_type=str(pending["scope_type"]),
                    scope_id=str(pending["scope_id"]),
                    actor_qq=actor_qq,
                    is_admin=is_admin,
                ):
                    raise PermissionError(
                        "physical delete is not authorized"
                    )
                needs_checkpoint = pending is not None
                if pending is not None:
                    needs_fts_optimize = bool(
                        pending["needs_fts_optimize"]
                    )
            else:
                scope_type = str(exists["scope_type"])
                scope_id = str(exists["scope_id"])
                if not _physical_delete_is_authorized(
                    scope_type=scope_type,
                    scope_id=scope_id,
                    actor_qq=actor_qq,
                    is_admin=is_admin,
                ):
                    raise PermissionError(
                        "physical delete is not authorized"
                    )
                scope_key = f"{scope_type}:{scope_id}"
                source_message_id = str(exists["source_message_id"] or "")
                dedupe_key = str(exists["dedupe_key"] or "")
                if _POLICY_DEDUPE_KEY_PATTERN.fullmatch(dedupe_key):
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO memory_claim_tombstones(
                            dedupe_key, deleted_at
                        ) VALUES (?, ?)
                        """,
                        (dedupe_key, _utc_now()),
                    )
                connection.execute(
                    """
                    INSERT INTO memory_deletion_audit(claim_id, reason, deleted_at)
                    VALUES (?, ?, ?)
                    """,
                    (claim_id, reason, _utc_now()),
                )
                connection.execute(
                    """
                    INSERT INTO memory_pending_privacy_cleanup(
                        claim_id, reason, scope_type, scope_id,
                        needs_fts_optimize, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(claim_id) DO UPDATE SET
                        reason = excluded.reason,
                        scope_type = excluded.scope_type,
                        scope_id = excluded.scope_id,
                        needs_fts_optimize = excluded.needs_fts_optimize,
                        created_at = excluded.created_at
                    """,
                    (
                        claim_id,
                        reason,
                        str(exists["scope_type"]),
                        str(exists["scope_id"]),
                        int(needs_fts_optimize),
                        _utc_now(),
                    ),
                )
                same_source_claim_ids = (claim_id,)
                if source_message_id:
                    connection.execute(
                        """
                        UPDATE memory_jobs
                        SET payload_json = ?,
                            state = CASE
                                WHEN state IN (
                                    'staged', 'ready', 'running', 'retry'
                                ) THEN 'failed'
                                ELSE state
                            END,
                            retry_at = CASE
                                WHEN state IN (
                                    'staged', 'ready', 'running', 'retry'
                                ) THEN NULL
                                ELSE retry_at
                            END,
                            error_type = CASE
                                WHEN state IN (
                                    'staged', 'ready', 'running', 'retry'
                                ) THEN 'source_deleted'
                                ELSE error_type
                            END,
                            updated_at = ?
                        WHERE source_message_id = ?
                          AND scope_key = ?
                        """,
                        (
                            _body_free_job_payload(source_message_id),
                            _utc_now(),
                            source_message_id,
                            scope_key,
                        ),
                    )
                    same_source_rows = connection.execute(
                        """
                        SELECT id
                        FROM memory_claims
                        WHERE source_message_id = ?
                          AND scope_type = ?
                          AND scope_id = ?
                        """,
                        (source_message_id, scope_type, scope_id),
                    ).fetchall()
                    same_source_claim_ids = tuple(
                        int(row["id"]) for row in same_source_rows
                    )
                if same_source_claim_ids:
                    placeholders = ",".join(
                        "?" for _ in same_source_claim_ids
                    )
                    connection.execute(
                        f"""
                        UPDATE memory_claims
                        SET source_excerpt = ''
                        WHERE id IN ({placeholders})
                          AND source_excerpt <> ''
                        """,
                        same_source_claim_ids,
                    )
                    connection.execute(
                        f"""
                        DELETE FROM memory_fts
                        WHERE rowid IN ({placeholders})
                        """,
                        same_source_claim_ids,
                    )
                if source_message_id:
                    connection.execute(
                        """
                        UPDATE memory_evidence
                        SET source_excerpt = ''
                        WHERE source_message_id = ?
                          AND source_excerpt <> ''
                          AND EXISTS (
                              SELECT 1
                              FROM memory_claims AS scoped_claim
                              WHERE scoped_claim.id = memory_evidence.claim_id
                                AND scoped_claim.scope_type = ?
                                AND scoped_claim.scope_id = ?
                          )
                        """,
                        (source_message_id, scope_type, scope_id),
                    )
                connection.execute(
                    "DELETE FROM memory_claims WHERE id = ?",
                    (claim_id,),
                )
                for sibling_id in same_source_claim_ids:
                    if sibling_id == claim_id:
                        continue
                    sibling = connection.execute(
                        "SELECT * FROM memory_claims WHERE id = ?",
                        (sibling_id,),
                    ).fetchone()
                    if sibling is not None:
                        _insert_fts_row(connection, sibling)
                deleted = True
                needs_checkpoint = True
        return _DeleteMutation(
            deleted=deleted,
            needs_checkpoint=needs_checkpoint,
            needs_fts_optimize=needs_fts_optimize,
        )

    def _run_privacy_maintenance(
        self,
        *,
        needs_checkpoint: bool,
        needs_fts_optimize: bool,
    ) -> None:
        if needs_checkpoint:
            if needs_fts_optimize:
                self._optimize_fts_for_privacy()
            self._checkpoint_wal()

    def delete_claim_physically_with_outcome(
        self,
        claim_id: int,
        *,
        reason: str,
        actor_qq: str,
        is_admin: bool,
    ) -> PhysicalDeleteOutcome:
        """Delete and report post-commit privacy maintenance precisely."""
        had_audit = self._has_deletion_audit(claim_id)
        mutation = self._delete_claim_transaction(
            claim_id,
            reason=reason,
            actor_qq=actor_qq,
            is_admin=is_admin,
        )
        try:
            self._run_privacy_maintenance(
                needs_checkpoint=mutation.needs_checkpoint,
                needs_fts_optimize=mutation.needs_fts_optimize,
            )
            if mutation.needs_checkpoint:
                self._clear_pending_privacy_cleanup(claim_id)
        except Exception:
            if mutation.needs_checkpoint:
                return PhysicalDeleteOutcome(
                    status="partial",
                    row_deleted=True,
                    cleanup_complete=False,
                    retryable=True,
                )
            raise
        if mutation.deleted:
            return PhysicalDeleteOutcome(
                status="deleted",
                row_deleted=True,
                cleanup_complete=True,
                retryable=False,
            )
        if had_audit or self._has_deletion_audit(claim_id):
            return PhysicalDeleteOutcome(
                status="cleanup_completed",
                row_deleted=False,
                cleanup_complete=True,
                retryable=False,
            )
        return PhysicalDeleteOutcome(
            status="no_op",
            row_deleted=False,
            cleanup_complete=True,
            retryable=False,
        )

    def retry_pending_delete_cleanup(
        self,
        claim_id: int,
        *,
        actor_qq: str,
        is_admin: bool,
    ) -> tuple[PhysicalDeleteOutcome, str] | None:
        with self._connection() as connection:
            pending = connection.execute(
                """
                SELECT scope_type, scope_id, needs_fts_optimize
                FROM memory_pending_privacy_cleanup
                WHERE claim_id = ?
                """,
                (claim_id,),
            ).fetchone()
        if pending is None:
            return None
        scope_type = str(pending["scope_type"])
        scope_id = str(pending["scope_id"])
        needs_fts_optimize = bool(pending["needs_fts_optimize"])
        authorized = _physical_delete_is_authorized(
            scope_type=scope_type,
            scope_id=scope_id,
            actor_qq=actor_qq,
            is_admin=is_admin,
        )
        if not authorized:
            return None
        try:
            self._run_privacy_maintenance(
                needs_checkpoint=True,
                needs_fts_optimize=needs_fts_optimize,
            )
            self._clear_pending_privacy_cleanup(claim_id)
        except Exception:
            return (
                PhysicalDeleteOutcome(
                    status="partial",
                    row_deleted=True,
                    cleanup_complete=False,
                    retryable=True,
                ),
                f"{scope_type}:{scope_id}",
            )
        return (
            PhysicalDeleteOutcome(
                status="cleanup_completed",
                row_deleted=True,
                cleanup_complete=True,
                retryable=False,
            ),
            f"{scope_type}:{scope_id}",
        )

    def _clear_pending_privacy_cleanup(self, claim_id: int) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                DELETE FROM memory_pending_privacy_cleanup
                WHERE claim_id = ?
                """,
                (claim_id,),
            )

    def _has_deletion_audit(self, claim_id: int) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM memory_deletion_audit
                WHERE claim_id = ?
                LIMIT 1
                """,
                (claim_id,),
            ).fetchone()
        return row is not None

    def register_subject_dispute(
        self,
        claim_id: int,
        *,
        actor_qq: str,
        group_id: str,
        source_message_id: str,
    ) -> MemoryClaim | None:
        """Suppress a foreign group claim when its subject disputes its use."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM memory_claims WHERE id = ?",
                (claim_id,),
            ).fetchone()
            if row is None:
                return None
            target = _claim_from_row(row)
            if (
                target.scope_type != "group"
                or target.scope_id != group_id
                or target.subject_type != "qq_user"
                or target.subject_id != actor_qq
                or target.speaker_qq == actor_qq
                or target.status not in ("active", "disputed")
            ):
                return None
            connection.execute(
                """
                INSERT INTO memory_subject_disputes(
                    target_claim_id, actor_qq, source_message_id, created_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(target_claim_id, actor_qq) DO UPDATE SET
                    source_message_id = excluded.source_message_id,
                    created_at = excluded.created_at
                """,
                (
                    target.id,
                    actor_qq,
                    source_message_id,
                    _utc_now(),
                ),
            )
        return target

    def subject_dispute_suppressed_ids(
        self,
        claim_ids: tuple[int, ...],
    ) -> frozenset[int]:
        if not claim_ids:
            return frozenset()
        with self._connection() as connection:
            return _subject_dispute_suppressed_ids_in_connection(
                connection,
                claim_ids,
            )

    def retract_group_claim(
        self,
        claim_id: int,
        *,
        actor_qq: str,
        group_id: str,
    ) -> bool:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE memory_claims
                SET status = 'retracted'
                WHERE id = ?
                  AND scope_type = 'group'
                  AND scope_id = ?
                  AND speaker_qq = ?
                  AND status IN ('active', 'disputed')
                """,
                (claim_id, group_id, actor_qq),
            )
        return cursor.rowcount == 1

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
        with self._connection() as connection:
            return _add_evidence_in_connection(
                connection,
                claim_id,
                source_kind=source_kind,
                source_message_id=source_message_id,
                source_excerpt=source_excerpt,
            )

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
            return _add_relation_in_connection(
                connection,
                source_claim_id,
                target_claim_id,
                relation_type,
            )

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

    def find_claims_exact(
        self,
        *,
        scope_type: str,
        scope_id: str,
        statuses: tuple[str, ...] = (),
        subject_type: str | None = None,
        subject_id: str | None = None,
        predicates: tuple[str, ...] = (),
        value: str | None = None,
        speaker_qq: str | None = None,
    ) -> tuple[MemoryClaim, ...]:
        with self._connection() as connection:
            return _find_claims_exact(
                connection,
                scope_type=scope_type,
                scope_id=scope_id,
                statuses=statuses,
                subject_type=subject_type,
                subject_id=subject_id,
                predicates=predicates,
                value=value,
                speaker_qq=speaker_qq,
            )

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

    def search_authorized_claims(
        self,
        context: MemoryContext,
        query: str,
        *,
        limit: int,
    ) -> tuple[MemoryClaim, ...]:
        if limit <= 0:
            return ()
        permission_sql, permission_parameters = (
            _authorized_claim_scope_sql(
                context,
                include_private_personalization=True,
            )
        )
        query_text = str(query or "").strip()
        parameters: list[Any] = []
        if query_text:
            sql = f"""
                SELECT claim.*
                FROM memory_fts
                JOIN memory_claims AS claim
                  ON claim.id = memory_fts.rowid
                WHERE ({permission_sql})
                  AND claim.status IN ('active', 'disputed')
                  AND memory_fts MATCH ?
                ORDER BY bm25(memory_fts), claim.id DESC
                LIMIT ?
            """
            parameters.extend(permission_parameters)
            parameters.extend((_fts_phrase(query_text), limit))
        else:
            sql = f"""
                SELECT claim.*
                FROM memory_claims AS claim
                WHERE ({permission_sql})
                  AND claim.status IN ('active', 'disputed')
                ORDER BY claim.id DESC
                LIMIT ?
            """
            parameters.extend(permission_parameters)
            parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return tuple(_claim_from_row(row) for row in rows)

    def list_authorized_claims(
        self,
        context: MemoryContext,
        *,
        include_private_personalization: bool,
    ) -> tuple[MemoryClaim, ...]:
        permission_sql, parameters = _authorized_claim_scope_sql(
            context,
            include_private_personalization=(
                include_private_personalization
            ),
        )
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT claim.*
                FROM memory_claims AS claim
                WHERE ({permission_sql})
                  AND claim.status IN ('active', 'disputed')
                ORDER BY claim.id DESC
                """,
                parameters,
            ).fetchall()
        return tuple(_claim_from_row(row) for row in rows)

    def find_reserved_authorized_claims(
        self,
        context: MemoryContext,
        *,
        predicates: tuple[str, ...],
        subject_id: str | None,
        limit: int,
    ) -> tuple[MemoryClaim, ...]:
        if not predicates or limit <= 0:
            return ()
        permission_sql, parameters = _authorized_claim_scope_sql(
            context,
            include_private_personalization=True,
        )
        placeholders = ",".join("?" for _ in predicates)
        clauses = [
            f"({permission_sql})",
            "claim.status IN ('active', 'disputed')",
            f"claim.predicate IN ({placeholders})",
        ]
        parameters.extend(predicates)
        if subject_id is not None:
            clauses.extend(
                (
                    "claim.subject_type = 'qq_user'",
                    "claim.subject_id = ?",
                )
            )
            parameters.append(subject_id)
        parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT claim.*
                FROM memory_claims AS claim
                WHERE {' AND '.join(clauses)}
                ORDER BY
                    CASE
                        WHEN claim.predicate = 'preferred_name' THEN 0
                        WHEN claim.predicate IN ('name', 'real_name') THEN 1
                        WHEN claim.predicate = 'likes' THEN 2
                        WHEN claim.predicate = 'response_style' THEN 3
                        ELSE 4
                    END,
                    claim.id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return tuple(_claim_from_row(row) for row in rows)

    def authorized_conflict_group(
        self,
        context: MemoryContext,
        claim_id: int,
    ) -> tuple[MemoryClaim, ...] | None:
        permission_sql, parameters = _authorized_claim_scope_sql(
            context,
            include_private_personalization=True,
        )
        with self._connection() as connection:
            id_rows = connection.execute(
                """
                WITH RECURSIVE conflict_ids(id) AS (
                    SELECT ?
                    UNION
                    SELECT
                        CASE
                            WHEN relation.source_claim_id = conflict_ids.id
                            THEN relation.target_claim_id
                            ELSE relation.source_claim_id
                        END
                    FROM memory_relations AS relation
                    JOIN conflict_ids
                      ON relation.source_claim_id = conflict_ids.id
                      OR relation.target_claim_id = conflict_ids.id
                    WHERE relation.relation_type = 'contradicts'
                )
                SELECT id FROM conflict_ids
                """,
                (claim_id,),
            ).fetchall()
            claim_ids = tuple(int(row["id"]) for row in id_rows)
            if not claim_ids:
                return ()
            placeholders = ",".join("?" for _ in claim_ids)
            rows = connection.execute(
                f"""
                SELECT claim.*
                FROM memory_claims AS claim
                WHERE claim.id IN ({placeholders})
                  AND claim.status IN ('active', 'disputed')
                  AND ({permission_sql})
                ORDER BY claim.id
                """,
                [*claim_ids, *parameters],
            ).fetchall()
        if len(rows) != len(claim_ids):
            return None
        return tuple(_claim_from_row(row) for row in rows)

    def relation_types_for_claims(
        self,
        claim_ids: tuple[int, ...],
    ) -> dict[int, tuple[str, ...]]:
        unique_ids = tuple(dict.fromkeys(int(value) for value in claim_ids))
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT source_claim_id, target_claim_id, relation_type
                FROM memory_relations
                WHERE source_claim_id IN ({placeholders})
                  AND target_claim_id IN ({placeholders})
                ORDER BY relation_type, source_claim_id, target_claim_id
                """,
                [*unique_ids, *unique_ids],
            ).fetchall()
        relation_types: dict[int, set[str]] = {
            claim_id: set() for claim_id in unique_ids
        }
        for row in rows:
            relation_type = str(row["relation_type"])
            relation_types[int(row["source_claim_id"])].add(relation_type)
            relation_types[int(row["target_claim_id"])].add(relation_type)
        return {
            claim_id: tuple(sorted(values))
            for claim_id, values in relation_types.items()
        }


class _MemoryReconciliation:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create_claim(self, **fields: Any) -> tuple[MemoryClaim, bool]:
        return _create_claim_in_connection(self.connection, **fields)

    def update_claim(self, claim_id: int, **changes: Any) -> MemoryClaim:
        return _update_claim_in_connection(
            self.connection,
            claim_id,
            **changes,
        )

    def add_evidence(
        self,
        claim_id: int,
        *,
        source_kind: str,
        source_message_id: str,
        source_excerpt: str,
    ) -> tuple[int, bool]:
        return _add_evidence_in_connection(
            self.connection,
            claim_id,
            source_kind=source_kind,
            source_message_id=source_message_id,
            source_excerpt=source_excerpt,
        )

    def add_relation(
        self,
        source_claim_id: int,
        target_claim_id: int,
        relation_type: str,
    ) -> bool:
        return _add_relation_in_connection(
            self.connection,
            source_claim_id,
            target_claim_id,
            relation_type,
        )

    def find_claims_exact(self, **filters: Any) -> tuple[MemoryClaim, ...]:
        return _find_claims_exact(self.connection, **filters)

    def claim_is_tombstoned(self, dedupe_key: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1
            FROM memory_claim_tombstones
            WHERE dedupe_key = ?
            LIMIT 1
            """,
            (dedupe_key,),
        ).fetchone()
        return row is not None

    def subject_dispute_suppressed_ids(
        self,
        claim_ids: tuple[int, ...],
    ) -> frozenset[int]:
        return _subject_dispute_suppressed_ids_in_connection(
            self.connection,
            claim_ids,
        )

    def complete_job(self, job_id: int) -> None:
        _complete_job_in_connection(self.connection, job_id)


def _complete_job_in_connection(
    connection: sqlite3.Connection,
    job_id: int,
) -> None:
    cursor = connection.execute(
        """
        UPDATE memory_jobs
        SET state = 'done', retry_at = NULL, error_type = NULL,
            updated_at = ?
        WHERE id = ? AND state = 'running'
        """,
        (_utc_now(), job_id),
    )
    if cursor.rowcount != 1:
        raise ValueError(f"memory job {job_id} is not running")


def _create_claim_in_connection(
    connection: sqlite3.Connection,
    **fields: Any,
) -> tuple[MemoryClaim, bool]:
    fields.setdefault("status", "active")
    fields.setdefault("valid_from", None)
    fields.setdefault("valid_to", None)
    fields.setdefault("last_confirmed_at", None)
    fields["source_excerpt"] = str(fields["source_excerpt"])[
        :MAX_SOURCE_EXCERPT_CHARS
    ]
    sequence_row = connection.execute(
        """
        UPDATE memory_claim_id_sequence
        SET last_id = last_id + 1
        WHERE singleton = 1
        RETURNING last_id
        """
    ).fetchone()
    if sequence_row is None:
        raise RuntimeError("memory claim ID sequence is not initialized")
    fields["id"] = int(sequence_row["last_id"])
    columns = (
        "id",
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
        "created_at",
        "valid_from",
        "valid_to",
        "last_confirmed_at",
        "dedupe_key",
    )
    fields["created_at"] = _utc_now()
    parameters = tuple(fields[column] for column in columns)
    cursor = connection.execute(
        """
        INSERT INTO memory_claims(
            id, scope_type, scope_id, speaker_qq, subject_type, subject_id,
            predicate, value, memory_type, modality, source_kind,
            source_message_id, source_excerpt, extraction_confidence,
            attribution_confidence, truth_confidence, status,
            created_at, valid_from, valid_to, last_confirmed_at,
            dedupe_key
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT(dedupe_key) DO NOTHING
        """,
        parameters,
    )
    created = cursor.rowcount == 1
    row = connection.execute(
        "SELECT * FROM memory_claims WHERE dedupe_key = ?",
        (fields["dedupe_key"],),
    ).fetchone()
    if created:
        _insert_fts_row(connection, row)
    return _claim_from_row(row), created


def _update_claim_in_connection(
    connection: sqlite3.Connection,
    claim_id: int,
    **changes: Any,
) -> MemoryClaim:
    if not changes:
        row = connection.execute(
            "SELECT * FROM memory_claims WHERE id = ?",
            (claim_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"memory claim {claim_id} does not exist")
        return _claim_from_row(row)
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


def _add_evidence_in_connection(
    connection: sqlite3.Connection,
    claim_id: int,
    *,
    source_kind: str,
    source_message_id: str,
    source_excerpt: str,
) -> tuple[int, bool]:
    source_excerpt = source_excerpt[:MAX_SOURCE_EXCERPT_CHARS]
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


def _add_relation_in_connection(
    connection: sqlite3.Connection,
    source_claim_id: int,
    target_claim_id: int,
    relation_type: str,
) -> bool:
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


def _fts_phrase(value: str) -> str:
    return '"' + value.strip().replace('"', '""') + '"'


def _find_claims_exact(
    connection: sqlite3.Connection,
    *,
    scope_type: str,
    scope_id: str,
    statuses: tuple[str, ...] = (),
    subject_type: str | None = None,
    subject_id: str | None = None,
    predicates: tuple[str, ...] = (),
    value: str | None = None,
    speaker_qq: str | None = None,
) -> tuple[MemoryClaim, ...]:
    clauses = ["scope_type = ?", "scope_id = ?"]
    parameters: list[Any] = [scope_type, scope_id]
    for column, column_value in (
        ("subject_type", subject_type),
        ("subject_id", subject_id),
        ("value", value),
        ("speaker_qq", speaker_qq),
    ):
        if column_value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(column_value)
    for column, values in (("status", statuses), ("predicate", predicates)):
        if values:
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"{column} IN ({placeholders})")
            parameters.extend(values)
    rows = connection.execute(
        f"""
        SELECT *
        FROM memory_claims
        WHERE {' AND '.join(clauses)}
        ORDER BY id
        """,
        parameters,
    ).fetchall()
    return tuple(_claim_from_row(row) for row in rows)


def _authorized_claim_scope_sql(
    context: MemoryContext,
    *,
    include_private_personalization: bool,
) -> tuple[str, list[Any]]:
    user_id = str(context.user_id)
    if context.is_group:
        group_id = str(context.group_id or "")
        clauses = [
            "(claim.scope_type = 'group' AND claim.scope_id = ?)",
            "(claim.scope_type = 'global' AND claim.scope_id = 'global')",
        ]
        parameters: list[Any] = [group_id]
        if include_private_personalization:
            clauses.append(
                """
                (
                    claim.scope_type = 'private'
                    AND claim.scope_id = ?
                    AND claim.subject_type = 'qq_user'
                    AND claim.subject_id = ?
                    AND claim.speaker_qq = ?
                    AND (
                        claim.predicate IN (
                            'preferred_name', 'response_style'
                        )
                        OR claim.memory_type = 'preferred_name'
                    )
                )
                """
            )
            parameters.extend((user_id, user_id, user_id))
        return " OR ".join(clauses), parameters
    return (
        """
        (claim.scope_type = 'private' AND claim.scope_id = ?)
        OR
        (claim.scope_type = 'global' AND claim.scope_id = 'global')
        """,
        [user_id],
    )


def _subject_dispute_suppressed_ids_in_connection(
    connection: sqlite3.Connection,
    claim_ids: tuple[int, ...],
) -> frozenset[int]:
    if not claim_ids:
        return frozenset()
    placeholders = ", ".join("?" for _ in claim_ids)
    rows = connection.execute(
        f"""
        SELECT target_claim_id
        FROM memory_subject_disputes
        WHERE target_claim_id IN ({placeholders})
        """,
        claim_ids,
    ).fetchall()
    return frozenset(int(row["target_claim_id"]) for row in rows)


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


def _migrate_memory_jobs_source_message_id(
    connection: sqlite3.Connection,
) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(memory_jobs)"
        ).fetchall()
    }
    if "source_message_id" not in columns:
        connection.execute(
            """
            ALTER TABLE memory_jobs
            ADD COLUMN source_message_id TEXT NOT NULL DEFAULT ''
            """
        )
    rows = connection.execute(
        """
        SELECT id, payload_json
        FROM memory_jobs
        WHERE source_message_id = ''
        """
    ).fetchall()
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        source_message_id = str(payload.get("message_id") or "").strip()
        if source_message_id:
            connection.execute(
                """
                UPDATE memory_jobs
                SET source_message_id = ?
                WHERE id = ?
                """,
                (source_message_id, int(row["id"])),
            )


def _migrate_memory_claims_privacy_maintenance(
    connection: sqlite3.Connection,
) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(memory_claims)"
        ).fetchall()
    }
    if "privacy_maintenance_at" not in columns:
        connection.execute(
            """
            ALTER TABLE memory_claims
            ADD COLUMN privacy_maintenance_at TEXT
            """
        )


def _body_free_job_payload(source_message_id: str) -> str:
    return json.dumps(
        {
            "message_id": source_message_id,
            "text": "",
        },
        ensure_ascii=False,
        separators=(",", ":"),
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
        message_id=str(
            payload.get("message_id")
            or (
                row["source_message_id"]
                if "source_message_id" in row.keys()
                else ""
            )
        ),
        text=str(payload.get("text", "")),
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
        prior_dialogue_context=tuple(
            (str(turn[0]), str(turn[1]))
            for turn in payload.get("prior_dialogue_context", ())
            if isinstance(turn, (list, tuple)) and len(turn) == 2
        ),
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
        privacy_maintenance_at TEXT,
        dedupe_key TEXT NOT NULL UNIQUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_claim_id_sequence (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        last_id INTEGER NOT NULL CHECK (last_id >= 0)
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
        source_message_id TEXT NOT NULL,
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
    CREATE TABLE IF NOT EXISTS memory_claim_tombstones (
        dedupe_key TEXT PRIMARY KEY
            CHECK (
                length(dedupe_key) = 71
                AND substr(dedupe_key, 1, 7) = 'policy:'
                AND substr(dedupe_key, 8) NOT GLOB '*[^0-9a-f]*'
            ),
        deleted_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_pending_privacy_cleanup (
        claim_id INTEGER PRIMARY KEY,
        reason TEXT NOT NULL,
        scope_type TEXT NOT NULL
            CHECK (scope_type IN ('private', 'group', 'global', 'bot')),
        scope_id TEXT NOT NULL,
        needs_fts_optimize INTEGER NOT NULL
            CHECK (needs_fts_optimize IN (0, 1)),
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_subject_disputes (
        target_claim_id INTEGER NOT NULL
            REFERENCES memory_claims(id) ON DELETE CASCADE,
        actor_qq TEXT NOT NULL,
        source_message_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(target_claim_id, actor_qq)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS memory_subject_disputes_actor_idx
    ON memory_subject_disputes(actor_qq, target_claim_id)
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
