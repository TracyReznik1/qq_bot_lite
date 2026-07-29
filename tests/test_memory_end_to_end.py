import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.commands import CommandContext, handle_command
from src.memory.models import CandidateClaim, MemoryContext, MemoryEvent
from src.memory.service import MemoryService
from src.memory.store import MemoryStore
from src.messaging import MessageQueue
from src.router import Route


class MemoryEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_memory.sqlite3"
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
        mq.executor.shutdown(wait=True)

    def test_new_callback_sequence_exceeds_durable_restart_high_water(self):
        context = MemoryContext(
            user_id="1001",
            session_key="private:1001",
            is_group=False,
        )
        self.store.create_job(
            MemoryEvent(
                context=context,
                message_id="durable-sequence-100",
                sequence=100,
                text="older durable message",
            )
        )
        queue = MessageQueue(max_workers=1)
        processed = []
        try:
            queue.ensure_sequence_at_least(
                self.store.max_job_sequence()
            )
            event = {
                "user_id": "1001",
                "raw_message": "new callback",
                "message_type": "private",
            }
            queue.enqueue(event, processed.append)
        finally:
            queue.executor.shutdown(wait=True)

        self.assertEqual(1, len(processed))
        self.assertGreater(event["_qqbot_sequence"], 100)

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

    def test_owner_forget_erases_auto_job_and_all_same_source_excerpts(self):
        marker = "AUTO_DELETE_PRIVATE_MARKER_72ca"
        extractor = MagicMock()
        extractor.extract.return_value = (
            CandidateClaim(
                subject_ref="speaker",
                predicate="likes",
                value=marker,
                memory_type="preference",
                modality="asserted",
                confidence="high",
            ),
            CandidateClaim(
                subject_ref="speaker",
                predicate="likes",
                value="香蕉",
                memory_type="preference",
                modality="asserted",
                confidence="high",
            ),
        )
        service = MemoryService(store=self.store, extractor=extractor)
        context = MemoryContext("1001", "private:1001", False)
        event = MemoryEvent(
            context=context,
            message_id="auto-delete-source",
            sequence=1,
            text=f"我喜欢 {marker}，也喜欢香蕉",
        )
        job_id = service.stage_event(event)
        service.release_job(job_id)
        service._process_claimed_job(
            self.store.claim_next_job("private:1001")
        )
        claims = self.store.find_claims_exact(
            scope_type="private",
            scope_id="1001",
            statuses=("active",),
        )
        target = next(claim for claim in claims if claim.value == marker)
        sibling = next(claim for claim in claims if claim.value == "香蕉")
        self.store.add_evidence(
            sibling.id,
            source_kind="message:speaker",
            source_message_id=event.message_id,
            source_excerpt=f"same source retained {marker}",
        )

        result = handle_command(
            Route(
                handler="command",
                action="command",
                command="forget",
                query=str(target.id),
            ),
            CommandContext(
                uid="1001",
                session_key="private:1001",
                raw_message=f"/forget {target.id}",
                memory_context=context,
                message_id="forget-auto-delete",
            ),
            store=self.store,
        )

        self.assertEqual("deleted", result.outcome.status)
        self.assertIsNone(self.store.get_claim(target.id))
        remaining = self.store.get_claim(sibling.id)
        self.assertEqual("香蕉", remaining.value)
        self.assertEqual("", remaining.source_excerpt)
        self.assertEqual(
            [""],
            [
                evidence.source_excerpt
                for evidence in self.store.list_evidence(sibling.id)
            ],
        )
        job = self.store.get_job(job_id)
        self.assertEqual("done", job.state)
        self.assertEqual("", json.loads(job.payload_json)["text"])
        self.assertEqual((), self.store.search_claims(marker))
        self.assertEqual(
            [sibling.id],
            [
                claim.id
                for claim in self.store.search_claims(
                    "香蕉",
                    scope_type="private",
                    scope_id="1001",
                )
            ],
        )
        marker_bytes = marker.encode("utf-8")
        for database_file in (
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            if database_file.exists():
                self.assertNotIn(marker_bytes, database_file.read_bytes())

    def test_delete_tombstones_retry_job_and_duplicate_callback_cannot_resurrect(self):
        marker = "RETRY_DELETE_MARKER_0be7"
        candidate = CandidateClaim(
            subject_ref="speaker",
            predicate="likes",
            value=marker,
            memory_type="preference",
            modality="asserted",
            confidence="high",
        )
        extractor = MagicMock()
        extractor.extract.return_value = (candidate,)
        service = MemoryService(store=self.store, extractor=extractor)
        context = MemoryContext("1001", "private:1001", False)
        event = MemoryEvent(
            context=context,
            message_id="retry-delete-source",
            sequence=1,
            text=f"我喜欢 {marker}",
        )
        job_id = service.stage_event(event)
        service.release_job(job_id)
        running = self.store.claim_next_job("private:1001")
        decisions = service._policy.apply(event, (candidate,))
        self.store.fail_job(
            running.id,
            error_type="legacy_completion_failure",
            retry_at="2026-07-29T00:00:00+00:00",
        )
        target = decisions[0].claim

        result = handle_command(
            Route(
                handler="command",
                action="command",
                command="forget",
                query=str(target.id),
            ),
            CommandContext(
                uid="1001",
                session_key="private:1001",
                raw_message=f"/forget {target.id}",
                memory_context=context,
                message_id="forget-retry-delete",
            ),
            store=self.store,
        )
        duplicate_id = service.stage_event(event)
        service.release_job(duplicate_id)
        retry = self.store.claim_next_job("private:1001")
        if retry is not None:
            service._process_claimed_job(retry)
        late_callback = service._policy.apply(event, (candidate,))

        self.assertEqual("deleted", result.outcome.status)
        self.assertEqual((), late_callback)
        self.assertEqual(job_id, duplicate_id)
        tombstoned = self.store.get_job(job_id)
        self.assertEqual("failed", tombstoned.state)
        self.assertEqual("source_deleted", tombstoned.error_type)
        self.assertEqual("", json.loads(tombstoned.payload_json)["text"])
        self.assertIsNone(self.store.get_claim(target.id))
        self.assertEqual(
            (),
            self.store.find_claims_exact(
                scope_type="private",
                scope_id="1001",
                statuses=("active", "disputed"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
