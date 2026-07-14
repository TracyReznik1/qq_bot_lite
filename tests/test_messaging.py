import logging
import unittest
from threading import Event, Lock
from unittest.mock import patch

from src.messaging import MessageQueue, get_event_session_key


WAIT_TIMEOUT = 2


def private_message(user_id: int, raw_message: str) -> dict:
    return {
        "message_type": "private",
        "user_id": user_id,
        "raw_message": raw_message,
    }


class MessageQueueConcurrencyTests(unittest.TestCase):
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

        logger = logging.getLogger("qq-bot")
        try:
            with patch.object(logger, "exception") as log_exception:
                queue.enqueue(private_message(7, "1"), process)
                self.assertTrue(first_started.wait(WAIT_TIMEOUT))
                queue.enqueue(private_message(7, "2"), process)

                release_first.set()
                self.assertTrue(second_finished.wait(WAIT_TIMEOUT))

                with state_lock:
                    self.assertEqual([1, 2], order)
                self.assertFalse(first_timed_out.is_set())
                log_exception.assert_called_once_with(
                    "Background message processing failed"
                )
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
