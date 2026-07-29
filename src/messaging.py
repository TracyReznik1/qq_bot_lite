from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
from threading import Lock
from typing import Any, Callable

from src.utils.storage import safe_id


def build_session_key(uid: str, data: dict[str, Any], is_group: bool) -> str:
    if is_group:
        return f"group:{safe_id(data.get('group_id'))}:{safe_id(uid)}"
    return f"private:{safe_id(uid)}"


def get_event_session_key(data: dict[str, Any]) -> str | None:
    uid = str(data.get("user_id", ""))
    raw_msg = str(data.get("raw_message", "")).strip()
    if not uid or not raw_msg:
        return None

    is_group = data.get("message_type") == "group"
    return build_session_key(uid, data, is_group)


def get_event_memory_scope_key(data: dict[str, Any]) -> str | None:
    uid = str(data.get("user_id", "")).strip()
    if not uid:
        return None
    if data.get("message_type") == "group":
        group_id = str(data.get("group_id", "")).strip()
        return f"group:{safe_id(group_id)}" if group_id else None
    return f"private:{safe_id(uid)}"


def build_message_dedupe_key(data: dict[str, Any], message_id: Any) -> str:
    message_type = str(data.get("message_type") or "unknown")
    if message_type == "group":
        scope_kind = "group"
        scope_id = data.get("group_id")
    else:
        scope_kind = "user"
        scope_id = data.get("user_id")

    return ":".join(
        [
            safe_id(data.get("self_id")),
            safe_id(message_type),
            scope_kind,
            safe_id(scope_id),
            safe_id(message_id),
        ]
    )


_sequence_lock = Lock()
_sequence_counter = 0


def next_qqbot_sequence() -> int:
    global _sequence_counter
    with _sequence_lock:
        _sequence_counter += 1
        return _sequence_counter


def ensure_qqbot_sequence_at_least(sequence: int) -> None:
    global _sequence_counter
    minimum = max(int(sequence), 0)
    with _sequence_lock:
        _sequence_counter = max(_sequence_counter, minimum)


class MessageQueue:
    def __init__(
        self,
        max_workers: int = 4,
        max_processed_message_ids: int = 500,
        on_accepted: Callable[[dict[str, Any]], None] | None = None,
        on_rejected: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="qq-message")
        self.processed_message_ids: set[str] = set()
        self.processed_message_order: deque[str] = deque()
        self.processed_message_lock = Lock()
        self.session_queue_lock = Lock()
        self.session_message_queues: dict[str, deque[dict[str, Any]]] = {}
        self.active_session_workers: set[str] = set()
        self.max_processed_message_ids = max_processed_message_ids
        self.on_accepted = on_accepted
        self.on_rejected = on_rejected

    def ensure_sequence_at_least(self, sequence: int) -> None:
        ensure_qqbot_sequence_at_least(sequence)

    def mark_seen(self, data: dict[str, Any]) -> bool:
        message_id = data.get("message_id")
        if message_id is None:
            return True

        key = build_message_dedupe_key(data, message_id)
        with self.processed_message_lock:
            if key in self.processed_message_ids:
                return False

            self.processed_message_ids.add(key)
            self.processed_message_order.append(key)
            while len(self.processed_message_order) > self.max_processed_message_ids:
                old_key = self.processed_message_order.popleft()
                self.processed_message_ids.discard(old_key)
            return True

    def enqueue(self, data: dict[str, Any], process_func) -> None:
        session_key = get_event_session_key(data)
        if session_key is None:
            return

        should_start_worker = False
        with self.session_queue_lock:
            if "_qqbot_sequence" not in data:
                data["_qqbot_sequence"] = next_qqbot_sequence()
            if self.on_accepted is not None:
                try:
                    self.on_accepted(data)
                except Exception as error:
                    logging.getLogger("qq-bot").error(
                        "Message acceptance hook failed error_type=%s",
                        type(error).__name__,
                    )
            queue = self.session_message_queues.setdefault(session_key, deque())
            queue.append(data)
            if session_key not in self.active_session_workers:
                self.active_session_workers.add(session_key)
                should_start_worker = True

        if should_start_worker:
            try:
                self.executor.submit(
                    self._drain_session,
                    session_key,
                    process_func,
                )
            except Exception:
                with self.session_queue_lock:
                    rejected = tuple(
                        self.session_message_queues.pop(session_key, ())
                    )
                    self.active_session_workers.discard(session_key)
                if self.on_rejected is not None:
                    for rejected_data in rejected:
                        try:
                            self.on_rejected(rejected_data)
                        except Exception as error:
                            logging.getLogger("qq-bot").error(
                                "Message rejection hook failed error_type=%s",
                                type(error).__name__,
                            )
                raise

    def _drain_session(self, session_key: str, process_func) -> None:
        while True:
            with self.session_queue_lock:
                queue = self.session_message_queues.get(session_key)
                if not queue:
                    self.session_message_queues.pop(session_key, None)
                    self.active_session_workers.discard(session_key)
                    return
                data = queue.popleft()

            try:
                process_func(data)
            except Exception as error:
                logging.getLogger("qq-bot").error(
                    "Background message processing failed error_type=%s",
                    type(error).__name__,
                )


def mark_message_seen(data: dict[str, Any], queue: MessageQueue) -> bool:
    return queue.mark_seen(data)


def enqueue_message(data: dict[str, Any], queue: MessageQueue, process_func) -> None:
    queue.enqueue(data, process_func)
