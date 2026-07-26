from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from threading import Condition, Lock, Thread
from typing import Sequence

from src.config import config
from src.memory.extractor import MemoryExtractor
from src.memory.models import MemoryEvent, MemoryJob
from src.memory.policy import MemoryPolicy
from src.memory.store import MemoryStore

logger = logging.getLogger("qq-bot")


def _utc_now_offset(seconds: int) -> str:
    dt = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return dt.isoformat()


class MemoryService:
    def __init__(
        self,
        store: MemoryStore | None = None,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        db_path = store.path if store else config.data_dir / "memory.db"
        self.store = store or MemoryStore(db_path)
        self._extractor = extractor or MemoryExtractor()
        self._policy = MemoryPolicy(self.store)
        self._ephemeral_images: dict[int, tuple[str, ...]] = {}
        self._lock = Lock()
        self._cond = Condition(self._lock)
        self._running = False
        self._workers: list[Thread] = []

    def start(self, worker_count: int = 2) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self.store.initialize()
            self.store.recover_running_jobs()
            self._workers = [
                Thread(target=self._worker_loop, daemon=True, name=f"memory-worker-{i}")
                for i in range(worker_count)
            ]
            for w in self._workers:
                w.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._cond.notify_all()
        for w in self._workers:
            w.join(timeout=2.0)
        self._workers.clear()

    def stage_event(self, event: MemoryEvent) -> int:
        job_id, _created = self.store.create_job(event)
        return job_id

    def release_job(self, job_id: int, image_data_urls: Sequence[str] = ()) -> None:
        with self._lock:
            if image_data_urls:
                self._ephemeral_images[job_id] = tuple(image_data_urls)
            self.store.mark_job_ready(job_id)
            self._cond.notify_all()

    def wait_for_scope(self, scope_key: str, timeout: float = 5.0) -> bool:
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        while datetime.now(timezone.utc) < deadline:
            if not self._has_pending_jobs(scope_key):
                return True
            with self._lock:
                self._cond.wait(timeout=0.05)
        return not self._has_pending_jobs(scope_key)

    def _has_pending_jobs(self, scope_key: str) -> bool:
        with self.store._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM memory_jobs
                WHERE scope_key = ? AND state IN ('staged', 'ready', 'running')
                LIMIT 1
                """,
                (scope_key,),
            ).fetchone()
        return row is not None

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break

            job = self._find_and_claim_next_job()
            if job is None:
                with self._lock:
                    if not self._running:
                        break
                    self._cond.wait(timeout=0.2)
                continue

            self._process_claimed_job(job)

    def _find_and_claim_next_job(self) -> MemoryJob | None:
        now_str = _utc_now_offset(0)
        with self.store._connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT scope_key
                FROM memory_jobs
                WHERE state = 'ready'
                   OR (state = 'retry' AND (retry_at IS NULL OR retry_at <= ?))
                """,
                (now_str,),
            ).fetchall()
        for r in rows:
            scope_key = str(r["scope_key"])
            claimed = self.store.claim_next_job(scope_key)
            if claimed is not None:
                return claimed
        return None

    def _process_claimed_job(self, job: MemoryJob) -> None:
        with self._lock:
            images = self._ephemeral_images.get(job.id, ())

        try:
            event = MemoryEvent(
                context=job.context,
                message_id=job.message_id,
                sequence=job.sequence,
                text=job.text,
                image_count=job.image_count,
                mentioned_qq_ids=job.mentioned_qq_ids,
                reply_to_message_id=job.reply_to_message_id,
                reply_to_user_id=job.reply_to_user_id,
            )
            candidates = self._extractor.extract(
                text=job.text,
                image_data_urls=list(images),
                mentioned_qq_ids=job.mentioned_qq_ids,
                reply_to_user_id=job.reply_to_user_id,
            )
            self._policy.apply(event, candidates)
            self.store.complete_job(job.id)
            logger.info("Memory job completed job_id=%s scope_key=%s attempts=%s", job.id, job.scope_key, job.attempts)
        except Exception as err:
            error_type = type(err).__name__
            if job.attempts < 4:
                delays = [2, 10, 30]
                delay = delays[min(job.attempts - 1, 2)]
                retry_at = _utc_now_offset(delay)
                self.store.fail_job(job.id, error_type=error_type, retry_at=retry_at)
                logger.warning("Memory job retry job_id=%s scope_key=%s attempts=%s error_type=%s", job.id, job.scope_key, job.attempts, error_type)
            else:
                self.store.fail_job(job.id, error_type=error_type, retry_at=None)
                logger.error("Memory job failed job_id=%s scope_key=%s attempts=%s error_type=%s", job.id, job.scope_key, job.attempts, error_type)
        finally:
            with self._lock:
                self._ephemeral_images.pop(job.id, None)
                self._cond.notify_all()


_global_memory_service: MemoryService | None = None
_service_lock = Lock()


def get_memory_service() -> MemoryService:
    global _global_memory_service
    with _service_lock:
        if _global_memory_service is None:
            _global_memory_service = MemoryService()
        return _global_memory_service
