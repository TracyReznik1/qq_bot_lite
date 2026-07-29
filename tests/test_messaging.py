import os
import unittest
from threading import Event, Lock
from unittest import mock
from unittest.mock import patch

import src.main as main
from src.config import Config
from src.messaging import MessageQueue, get_event_session_key


WAIT_TIMEOUT = 2


def config_with(value: str | None) -> Config:
    environment = {
        "CHAT_MODELS": "gemini:test-gemini",
        "GEMINI_API_KEY": "test-gemini-key",
    }
    if value is not None:
        environment["MESSAGE_WORKERS"] = value
    with mock.patch.dict(os.environ, environment, clear=True):
        return Config()


def private_message(user_id: int, raw_message: str) -> dict:
    return {
        "message_type": "private",
        "user_id": user_id,
        "raw_message": raw_message,
    }


class MessageWorkerConfigurationTests(unittest.TestCase):
    def test_defaults_to_eight_when_environment_variable_is_missing(self):
        self.assertEqual(8, config_with(None).message_workers)

    def test_uses_positive_environment_value(self):
        self.assertEqual(16, config_with("16").message_workers)

    def test_normalizes_non_positive_values_to_one(self):
        self.assertEqual(1, config_with("0").message_workers)
        self.assertEqual(1, config_with("-3").message_workers)

    def test_falls_back_to_eight_for_invalid_value(self):
        self.assertEqual(8, config_with("many").message_workers)


class MainMessageQueueConfigurationTests(unittest.TestCase):
    def test_global_queue_uses_configured_worker_count(self):
        self.assertEqual(
            main.config.message_workers,
            main.message_queue.executor._max_workers,
        )

    def test_global_queue_registers_group_memory_sequence_on_acceptance(self):
        service = mock.Mock()
        event = {
            "message_type": "group",
            "group_id": 20,
            "user_id": 7,
            "raw_message": "ordinary",
            "_qqbot_sequence": 42,
        }

        with mock.patch.object(main, "get_memory_service", return_value=service):
            main.message_queue.on_accepted(event)

        service.register_pending_sequence.assert_called_once_with("group:20", 42)

    def test_global_queue_clears_group_memory_sequence_on_rejection(self):
        service = mock.Mock()
        event = {
            "message_type": "group",
            "group_id": 20,
            "user_id": 7,
            "raw_message": "ordinary",
            "_qqbot_sequence": 43,
        }

        with mock.patch.object(main, "get_memory_service", return_value=service):
            main.message_queue.on_rejected(event)

        service.clear_pending_sequence.assert_called_once_with("group:20", 43)

    def test_startup_seeds_receive_sequence_before_callbacks_are_accepted(self):
        service = mock.Mock()
        service.store.max_job_sequence.return_value = 100
        queue = mock.Mock()
        main._startup_initialized = False
        try:
            with (
                mock.patch.object(
                    main,
                    "get_persona",
                    return_value=mock.Mock(),
                ),
                mock.patch.object(
                    main,
                    "get_memory_service",
                    return_value=service,
                ),
                mock.patch.object(main, "message_queue", queue),
            ):
                main.startup()
        finally:
            main._startup_initialized = False

        service.start.assert_called_once_with()
        queue.ensure_sequence_at_least.assert_called_once_with(100)


class MessageQueueConcurrencyTests(unittest.TestCase):
    def test_submit_failure_rolls_back_reservation_and_session_state(self):
        accepted = []
        rejected = []
        queue = MessageQueue(
            max_workers=1,
            on_accepted=lambda data: accepted.append(
                data["_qqbot_sequence"]
            ),
            on_rejected=lambda data: rejected.append(
                data["_qqbot_sequence"]
            ),
        )
        first = private_message(7, "first")
        first_session = get_event_session_key(first)
        processed = Event()

        try:
            with (
                mock.patch.object(
                    queue.executor,
                    "submit",
                    side_effect=RuntimeError("executor unavailable"),
                ),
                self.assertRaisesRegex(RuntimeError, "executor unavailable"),
            ):
                queue.enqueue(first, lambda _data: None)

            self.assertEqual(
                [first["_qqbot_sequence"]],
                accepted,
            )
            self.assertEqual(
                [first["_qqbot_sequence"]],
                rejected,
            )
            self.assertNotIn(first_session, queue.session_message_queues)
            self.assertNotIn(first_session, queue.active_session_workers)

            queue.enqueue(
                private_message(7, "second"),
                lambda _data: processed.set(),
            )
            self.assertTrue(processed.wait(WAIT_TIMEOUT))
        finally:
            queue.executor.shutdown(wait=True)

    def test_acceptance_hook_runs_with_sequence_before_processing_starts(self):
        accepted = []
        processed = []
        queue = MessageQueue(
            max_workers=1,
            on_accepted=lambda data: accepted.append(data["_qqbot_sequence"]),
        )

        try:
            queue.enqueue(
                private_message(7, "first"),
                lambda data: processed.append(
                    (data["_qqbot_sequence"], tuple(accepted))
                ),
            )
        finally:
            queue.executor.shutdown(wait=True)

        self.assertEqual(1, len(accepted))
        self.assertEqual([(accepted[0], (accepted[0],))], processed)

    def test_different_private_sessions_can_run_in_parallel(self):
        queue = MessageQueue(max_workers=2)
        first_started = Event()
        second_started = Event()
        release_first = Event()
        first_timed_out = Event()

        def process(data):
            if data["raw_message"] == "first":
                first_started.set()
                if not release_first.wait(WAIT_TIMEOUT):
                    first_timed_out.set()
            else:
                second_started.set()

        try:
            queue.enqueue(private_message(7, "first"), process)
            self.assertTrue(first_started.wait(WAIT_TIMEOUT))

            queue.enqueue(private_message(8, "second"), process)

            self.assertTrue(second_started.wait(WAIT_TIMEOUT))
            self.assertFalse(first_timed_out.is_set())
        finally:
            release_first.set()
            queue.executor.shutdown(wait=True)

    def test_same_private_session_is_strictly_fifo_and_never_concurrent(self):
        queue = MessageQueue(max_workers=2)
        first_started = Event()
        release_first = Event()
        all_finished = Event()
        first_timed_out = Event()
        state_lock = Lock()
        order = []
        active = 0
        max_active = 0

        def process(data):
            nonlocal active, max_active
            sequence = int(data["raw_message"])
            with state_lock:
                active += 1
                max_active = max(max_active, active)
                order.append(sequence)

            try:
                if sequence == 1:
                    first_started.set()
                    if not release_first.wait(WAIT_TIMEOUT):
                        first_timed_out.set()
            finally:
                with state_lock:
                    active -= 1
                    if len(order) == 3:
                        all_finished.set()

        try:
            queue.enqueue(private_message(7, "1"), process)
            self.assertTrue(first_started.wait(WAIT_TIMEOUT))

            queue.enqueue(private_message(7, "2"), process)
            queue.enqueue(private_message(7, "3"), process)
            with state_lock:
                self.assertEqual([1], order)
                self.assertEqual(1, active)
                self.assertEqual(1, max_active)

            release_first.set()
            self.assertTrue(all_finished.wait(WAIT_TIMEOUT))

            with state_lock:
                self.assertEqual([1, 2, 3], order)
                self.assertEqual(0, active)
                self.assertEqual(1, max_active)
            self.assertFalse(first_timed_out.is_set())
        finally:
            release_first.set()
            queue.executor.shutdown(wait=True)

    def test_session_continues_after_processing_exception(self):
        queue = MessageQueue(max_workers=1)
        first_started = Event()
        release_first = Event()
        second_finished = Event()
        first_timed_out = Event()
        state_lock = Lock()
        order = []

        def process(data):
            sequence = int(data["raw_message"])
            with state_lock:
                order.append(sequence)
            if sequence == 1:
                first_started.set()
                if not release_first.wait(WAIT_TIMEOUT):
                    first_timed_out.set()
                raise RuntimeError("first message failed")
            second_finished.set()

        try:
            with self.assertLogs("qq-bot", level="ERROR") as captured:
                queue.enqueue(private_message(7, "1"), process)
                self.assertTrue(first_started.wait(WAIT_TIMEOUT))
                queue.enqueue(private_message(7, "2"), process)

                release_first.set()
                self.assertTrue(second_finished.wait(WAIT_TIMEOUT))

                with state_lock:
                    self.assertEqual([1, 2], order)
                self.assertFalse(first_timed_out.is_set())
                logged = "\n".join(captured.output)
                self.assertIn("RuntimeError", logged)
                self.assertNotIn("first message failed", logged)
        finally:
            release_first.set()
            queue.executor.shutdown(wait=True)

    def test_session_keys_separate_private_group_user_and_group_scopes(self):
        private_event = private_message(7, "private")
        group_event = {
            "message_type": "group",
            "user_id": 7,
            "group_id": 10,
            "raw_message": "group",
        }
        same_group_and_user_event = {
            **group_event,
            "raw_message": "another message",
        }
        same_user_other_group_event = {**group_event, "group_id": 11}
        same_group_other_user_event = {**group_event, "user_id": 8}

        private_key = get_event_session_key(private_event)
        group_key = get_event_session_key(group_event)
        same_group_and_user_key = get_event_session_key(same_group_and_user_event)
        same_user_other_group_key = get_event_session_key(
            same_user_other_group_event
        )
        same_group_other_user_key = get_event_session_key(
            same_group_other_user_event
        )

        self.assertEqual("private:7", private_key)
        self.assertEqual("group:10:7", group_key)
        self.assertEqual(group_key, same_group_and_user_key)
        self.assertNotEqual(group_key, same_user_other_group_key)
        self.assertNotEqual(group_key, same_group_other_user_key)
        self.assertNotEqual(private_key, group_key)
        self.assertNotEqual(same_user_other_group_key, same_group_other_user_key)


if __name__ == "__main__":
    unittest.main()
