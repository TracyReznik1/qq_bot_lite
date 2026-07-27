import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.memory.models import CandidateClaim, MemoryContext, MemoryEvent
from src.memory.service import MemoryService, get_memory_service
from src.memory.store import MemoryStore


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_integrity_check_returns_ok(self):
        self.assertEqual("ok", self.store.integrity_check())

    def test_default_service_uses_config_memory_database_path(self):
        expected_path = Path(self.temp_dir) / "configured" / "memory.sqlite3"
        fake_config = SimpleNamespace(memory_database_path=expected_path)

        with patch("src.memory.service.config", fake_config):
            service = MemoryService(extractor=MagicMock())

        self.assertEqual(expected_path, service.store.path)

    def test_stage_event_and_release_job(self):
        service = MemoryService(store=self.store)
        ctx = MemoryContext(user_id="1001", session_key="private:1001", is_group=False)
        event = MemoryEvent(
            context=ctx,
            message_id="msg_100",
            sequence=1,
            text="我叫小明",
        )

        job_id = service.stage_event(event)
        job = self.store.get_job(job_id)
        self.assertEqual("staged", job.state)
        self.assertEqual("private:1001", job.scope_key)

        service.release_job(job_id, image_data_urls=["data:image/png;base64,123"])
        job_after = self.store.get_job(job_id)
        self.assertEqual("ready", job_after.state)
        # Check ephemeral images stored in memory
        self.assertIn(job_id, service._ephemeral_images)
        self.assertEqual(("data:image/png;base64,123",), service._ephemeral_images[job_id])

    def test_worker_processes_job_to_completion(self):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = (
            CandidateClaim(
                subject_ref="speaker",
                predicate="preferred_name",
                value="小明",
                memory_type="preferred_name",
                modality="asserted",
                confidence="high",
            ),
        )

        service = MemoryService(store=self.store, extractor=mock_extractor)
        service.start()
        try:
            ctx = MemoryContext(user_id="1001", session_key="private:1001", is_group=False)
            event = MemoryEvent(
                context=ctx,
                message_id="msg_101",
                sequence=1,
                text="叫我小明",
            )
            job_id = service.stage_event(event)
            service.release_job(job_id)

            finished = service.wait_for_scope("private:1001", timeout=5.0)
            self.assertTrue(finished)

            job = self.store.get_job(job_id)
            self.assertEqual("done", job.state)
            self.assertNotIn(job_id, service._ephemeral_images)

            claims = self.store.find_claims_exact(
                scope_type="private",
                scope_id="1001",
                statuses=("active",),
            )
            self.assertEqual(1, len(claims))
            self.assertEqual("小明", claims[0].value)
        finally:
            service.stop()

    def test_retry_on_transient_failure_and_permanent_failure(self):
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = RuntimeError("llm connection reset")

        service = MemoryService(store=self.store, extractor=mock_extractor)
        service.start()
        try:
            ctx = MemoryContext(user_id="1001", session_key="private:1001", is_group=False)
            event = MemoryEvent(context=ctx, message_id="msg_fail", sequence=1, text="测试失败")
            job_id = service.stage_event(event)
            service.release_job(job_id)

            service.wait_for_scope("private:1001", timeout=5.0)
            job = self.store.get_job(job_id)
            self.assertEqual("retry", job.state)
            self.assertEqual("RuntimeError", job.error_type)
            self.assertEqual(1, job.attempts)
        finally:
            service.stop()

    def test_abandoned_job_recovery(self):
        ctx = MemoryContext(user_id="1001", session_key="private:1001", is_group=False)
        event = MemoryEvent(context=ctx, message_id="msg_abandoned", sequence=1, text="中途崩溃")
        job_id, _ = self.store.create_job(event)
        self.store.mark_job_ready(job_id)
        job_claimed = self.store.claim_next_job("private:1001")
        self.assertIsNotNone(job_claimed)
        self.assertEqual("running", job_claimed.state)

        # Re-initialize service and recover running jobs
        recovered = self.store.recover_running_jobs()
        self.assertEqual(1, recovered)

        job_after = self.store.get_job(job_id)
        self.assertEqual("retry", job_after.state)
        self.assertEqual("abandoned", job_after.error_type)


if __name__ == "__main__":
    unittest.main()
