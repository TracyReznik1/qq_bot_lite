import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.memory.models import MemoryContext, MemoryEvent
from src.memory.service import MemoryService
from src.memory.store import MemoryStore
from src.messaging import MessageQueue


class MemoryEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_shared_group_memory_commit_scope_key(self):
        # Group jobs must have scope_key = "group:<group_id>"
        ctx = MemoryContext(user_id="1001", session_key="group:2001:1001", is_group=True, group_id="2001")
        event = MemoryEvent(context=ctx, message_id="msg_g1", sequence=1, text="群消息")
        job_id, _ = self.store.create_job(event)

        job = self.store.get_job(job_id)
        self.assertEqual("group:2001", job.scope_key)
        self.assertEqual("group:2001:1001", job.context.session_key)

    def test_message_queue_assigns_qqbot_sequence(self):
        mq = MessageQueue(max_workers=2)
        data = {"user_id": "1001", "raw_message": "hello", "message_type": "private"}
        processed = []

        def dummy_processor(msg):
            processed.append(msg)

        mq.enqueue(data, dummy_processor)
        self.assertIn("_qqbot_sequence", data)
        self.assertGreater(data["_qqbot_sequence"], 0)

    def test_blocked_extractor_does_not_delay_reply(self):
        # Background extractor sleeps for 2 seconds
        def slow_extract(*args, **kwargs):
            time.sleep(1.0)
            return ()

        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = slow_extract

        service = MemoryService(store=self.store, extractor=mock_extractor)
        service.start()
        try:
            ctx = MemoryContext(user_id="1001", session_key="private:1001", is_group=False)
            event = MemoryEvent(context=ctx, message_id="msg_slow", sequence=1, text="慢提取")

            start_t = time.monotonic()
            job_id = service.stage_event(event)
            service.release_job(job_id)
            elapsed = time.monotonic() - start_t

            # release_job must return immediately (< 0.2s)
            self.assertLess(elapsed, 0.2)
        finally:
            service.stop()

    def test_group_sequence_two_cannot_be_claimed_before_late_staged_sequence_one(self):
        service = MemoryService(store=self.store, extractor=MagicMock())
        first = MemoryEvent(
            context=MemoryContext(
                user_id="1001",
                session_key="group:2001:1001",
                is_group=True,
                group_id="2001",
            ),
            message_id="first",
            sequence=1,
            text="first ordinary message",
        )
        second = MemoryEvent(
            context=MemoryContext(
                user_id="1002",
                session_key="group:2001:1002",
                is_group=True,
                group_id="2001",
            ),
            message_id="second",
            sequence=2,
            text="second ordinary message",
        )
        scope_key = "group:2001"
        service.register_pending_sequence(scope_key, first.sequence)
        service.register_pending_sequence(scope_key, second.sequence)

        second_id = service.stage_event(second)
        service.release_job(second_id)

        self.assertIsNone(service._find_and_claim_next_job())

        first_id = service.stage_event(first)
        service.release_job(first_id)
        first_claim = service._find_and_claim_next_job()
        self.assertEqual(first_id, first_claim.id)
        service._process_claimed_job(first_claim)

        second_claim = service._find_and_claim_next_job()
        self.assertEqual(second_id, second_claim.id)


if __name__ == "__main__":
    unittest.main()
