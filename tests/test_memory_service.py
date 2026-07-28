import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
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

    def test_start_schedules_ninety_day_cleanup_before_workers(self):
        service = MemoryService(store=self.store, extractor=MagicMock())

        with patch.object(
            self.store,
            "cleanup_old_jobs_and_excerpts",
            wraps=self.store.cleanup_old_jobs_and_excerpts,
        ) as cleanup:
            service.start(worker_count=0)
            try:
                cleanup.assert_called_once_with(days=90)
            finally:
                service.stop()

    def test_release_failure_does_not_retain_ephemeral_images(self):
        service = MemoryService(store=self.store, extractor=MagicMock())
        event = MemoryEvent(
            context=MemoryContext(
                user_id="1001",
                session_key="private:1001",
                is_group=False,
            ),
            message_id="release-failure",
            sequence=1,
            text="ordinary text",
        )
        job_id = service.stage_event(event)

        with patch.object(
            self.store,
            "mark_job_ready",
            side_effect=RuntimeError("database failure"),
        ):
            with self.assertRaises(RuntimeError):
                service.release_job(
                    job_id,
                    image_data_urls=["data:image/png;base64,private-image"],
                )

        self.assertNotIn(job_id, service._ephemeral_images)

    def test_duplicate_completed_job_release_never_retains_images(self):
        service = MemoryService(store=self.store, extractor=MagicMock())
        event = MemoryEvent(
            context=MemoryContext(
                user_id="1001",
                session_key="private:1001",
                is_group=False,
            ),
            message_id="duplicate-completed",
            sequence=1,
            text="ordinary text",
        )
        job_id = service.stage_event(event)
        service.release_job(job_id)
        claimed = self.store.claim_next_job("private:1001")
        self.store.complete_job(claimed.id)

        duplicate_id = service.stage_event(event)
        service.release_job(
            duplicate_id,
            image_data_urls=["data:image/png;base64,duplicate-private-image"],
        )

        self.assertEqual(job_id, duplicate_id)
        self.assertEqual("done", self.store.get_job(job_id).state)
        self.assertNotIn(job_id, service._ephemeral_images)

    def test_duplicate_failed_job_release_never_retains_images(self):
        service = MemoryService(store=self.store, extractor=MagicMock())
        event = MemoryEvent(
            context=MemoryContext(
                user_id="1001",
                session_key="private:1001",
                is_group=False,
            ),
            message_id="duplicate-failed",
            sequence=1,
            text="ordinary text",
        )
        job_id = service.stage_event(event)
        service.release_job(job_id)
        claimed = self.store.claim_next_job("private:1001")
        self.store.fail_job(
            claimed.id,
            error_type="RuntimeError",
            retry_at=None,
        )

        duplicate_id = service.stage_event(event)
        service.release_job(
            duplicate_id,
            image_data_urls=["data:image/png;base64,duplicate-private-image"],
        )

        self.assertEqual(job_id, duplicate_id)
        self.assertEqual("failed", self.store.get_job(job_id).state)
        self.assertNotIn(job_id, service._ephemeral_images)

    def test_stop_removes_unclaimed_ephemeral_images(self):
        service = MemoryService(store=self.store, extractor=MagicMock())
        service.start(worker_count=0)
        event = MemoryEvent(
            context=MemoryContext(
                user_id="1001",
                session_key="private:1001",
                is_group=False,
            ),
            message_id="stopped-before-claim",
            sequence=1,
            text="ordinary text",
        )
        job_id = service.stage_event(event)
        service.release_job(
            job_id,
            image_data_urls=["data:image/png;base64,private-image"],
        )

        service.stop()

        self.assertEqual({}, service._ephemeral_images)

    def test_worker_state_failure_is_redacted_and_clears_ephemeral_images(self):
        extractor = MagicMock()
        extractor.extract.side_effect = RuntimeError(
            "incoming-body-private-marker"
        )
        service = MemoryService(store=self.store, extractor=extractor)
        event = MemoryEvent(
            context=MemoryContext(
                user_id="1001",
                session_key="private:1001",
                is_group=False,
            ),
            message_id="worker-failure",
            sequence=1,
            text="claim-value-private-marker",
        )
        job_id = service.stage_event(event)
        service.release_job(
            job_id,
            image_data_urls=["data:image/png;base64,private-image-marker"],
        )
        claimed = self.store.claim_next_job("private:1001")

        with (
            patch.object(
                self.store,
                "fail_job",
                side_effect=ValueError("database-private-marker"),
            ),
            self.assertLogs("qq-bot", level="ERROR") as captured,
        ):
            service._process_claimed_job(claimed)

        logged = "\n".join(captured.output)
        self.assertIn(f"job_id={job_id}", logged)
        self.assertIn("scope_key=private:1001", logged)
        self.assertIn("RuntimeError", logged)
        self.assertIn("ValueError", logged)
        self.assertNotIn("incoming-body-private-marker", logged)
        self.assertNotIn("claim-value-private-marker", logged)
        self.assertNotIn("database-private-marker", logged)
        self.assertNotIn("private-image-marker", logged)
        self.assertNotIn(job_id, service._ephemeral_images)

    def test_state_update_failure_recovers_only_current_job_and_unblocks_scope(self):
        extractor = MagicMock()
        extractor.extract.side_effect = [RuntimeError("first failure"), (), ()]
        service = MemoryService(store=self.store, extractor=extractor)
        first = MemoryEvent(
            context=MemoryContext("1001", "private:1001", False),
            message_id="recover-first",
            sequence=1,
            text="first",
        )
        second = MemoryEvent(
            context=MemoryContext("1001", "private:1001", False),
            message_id="recover-second",
            sequence=2,
            text="second",
        )
        other = MemoryEvent(
            context=MemoryContext("2002", "private:2002", False),
            message_id="other-running",
            sequence=3,
            text="other",
        )
        first_id = service.stage_event(first)
        second_id = service.stage_event(second)
        other_id = service.stage_event(other)
        for job_id in (first_id, second_id, other_id):
            service.release_job(job_id)
        first_claim = self.store.claim_next_job("private:1001")
        other_claim = self.store.claim_next_job("private:2002")

        with patch.object(
            self.store,
            "fail_job",
            side_effect=ValueError("state write failed"),
        ):
            service._process_claimed_job(first_claim)

        self.assertEqual("retry", self.store.get_job(first_id).state)
        self.assertEqual("running", self.store.get_job(other_claim.id).state)
        reclaimed = self.store.claim_next_job("private:1001")
        self.assertEqual(first_id, reclaimed.id)
        service._process_claimed_job(reclaimed)
        following = self.store.claim_next_job("private:1001")
        self.assertEqual(second_id, following.id)
        service._process_claimed_job(following)
        self.assertEqual("done", self.store.get_job(second_id).state)

    def test_worker_survives_redacted_claim_failure(self):
        service = MemoryService(store=self.store, extractor=MagicMock())
        first_attempt = Event()
        second_attempt = Event()
        release_second = Event()
        attempts = 0

        def claim():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                first_attempt.set()
                raise RuntimeError("claim-private-marker")
            second_attempt.set()
            release_second.wait(timeout=2)
            return None

        with (
            patch.object(service, "_find_and_claim_next_job", side_effect=claim),
            self.assertLogs("qq-bot", level="ERROR") as captured,
        ):
            service.start(worker_count=1)
            try:
                self.assertTrue(first_attempt.wait(timeout=2))
                self.assertTrue(second_attempt.wait(timeout=2))
            finally:
                release_second.set()
                service.stop()

        logged = "\n".join(captured.output)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn("claim-private-marker", logged)

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
