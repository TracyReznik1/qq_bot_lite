# Concurrent Conversations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make qqbot process up to eight different conversations concurrently by default while preserving strict FIFO processing and reply order within each conversation.

**Architecture:** Keep the existing `MessageQueue` design: one shared bounded `ThreadPoolExecutor`, one FIFO deque per `session_key`, and at most one drain worker per session. Add a startup-only `MESSAGE_WORKERS` configuration value, wire it into the global queue, and prove cross-session parallelism, same-session serialization, and failure recovery with event-driven concurrency tests.

**Tech Stack:** Python 3.11+, `concurrent.futures.ThreadPoolExecutor`, `threading.Event`/`Lock`, Flask, `unittest`, python-dotenv.

---

## File map

- Create `tests/test_messaging.py`: deterministic queue concurrency, FIFO, failure recovery, session-boundary, configuration, and main-wiring tests.
- Modify `src/config.py`: define startup configuration `message_workers`, defaulting to 8 and clamped to at least 1.
- Modify `src/main.py`: construct the global `MessageQueue` with `config.message_workers`.
- Modify `.env.example`: publish `MESSAGE_WORKERS=8`.
- Modify `README.md`: explain concurrent conversations and the configuration value.
- Modify local ignored `.env`: add `MESSAGE_WORKERS=8` only when the key is absent; never stage this file.

### Task 1: Characterize the existing session scheduler

**Files:**
- Create: `tests/test_messaging.py`
- Read: `src/messaging.py`

- [ ] **Step 1: Add deterministic tests for cross-session parallelism, same-session FIFO, failure recovery, and session boundaries**

Create `tests/test_messaging.py` with the following content. These are characterization tests for behavior already present in `MessageQueue`, so they are expected to pass before production changes:

```python
import logging
import threading
import unittest
from unittest import mock

from src.messaging import MessageQueue, get_event_session_key


def private_event(user_id: int, message_id: int) -> dict:
    return {
        "message_type": "private",
        "user_id": user_id,
        "message_id": message_id,
        "raw_message": f"message-{message_id}",
    }


def group_event(group_id: int, user_id: int, message_id: int) -> dict:
    return {
        "message_type": "group",
        "group_id": group_id,
        "user_id": user_id,
        "message_id": message_id,
        "raw_message": f"message-{message_id}",
    }


class MessageQueueConcurrencyTests(unittest.TestCase):
    def test_different_sessions_run_concurrently(self):
        queue = MessageQueue(max_workers=2)
        first_started = threading.Event()
        second_started = threading.Event()
        release = threading.Event()

        def process(data):
            if data["user_id"] == 1:
                first_started.set()
            else:
                second_started.set()
            release.wait(timeout=2)

        try:
            queue.enqueue(private_event(1, 1), process)
            self.assertTrue(first_started.wait(timeout=1))
            queue.enqueue(private_event(2, 2), process)
            self.assertTrue(second_started.wait(timeout=1))
        finally:
            release.set()
            queue.executor.shutdown(wait=True, cancel_futures=True)

    def test_same_session_runs_one_message_at_a_time_in_fifo_order(self):
        queue = MessageQueue(max_workers=3)
        first_started = threading.Event()
        release_first = threading.Event()
        all_done = threading.Event()
        state_lock = threading.Lock()
        order = []
        active = 0
        max_active = 0

        def process(data):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
                order.append(data["message_id"])
            try:
                if data["message_id"] == 1:
                    first_started.set()
                    release_first.wait(timeout=2)
                if data["message_id"] == 3:
                    all_done.set()
            finally:
                with state_lock:
                    active -= 1

        try:
            queue.enqueue(private_event(1, 1), process)
            self.assertTrue(first_started.wait(timeout=1))
            queue.enqueue(private_event(1, 2), process)
            queue.enqueue(private_event(1, 3), process)
            with state_lock:
                self.assertEqual([1], order)
                self.assertEqual(1, active)
            release_first.set()
            self.assertTrue(all_done.wait(timeout=1))
            with state_lock:
                self.assertEqual([1, 2, 3], order)
                self.assertEqual(1, max_active)
        finally:
            release_first.set()
            queue.executor.shutdown(wait=True, cancel_futures=True)

    def test_failed_message_does_not_block_next_message_in_same_session(self):
        queue = MessageQueue(max_workers=2)
        second_done = threading.Event()
        processed = []

        def process(data):
            processed.append(data["message_id"])
            if data["message_id"] == 1:
                raise RuntimeError("injected failure")
            second_done.set()

        try:
            with mock.patch.object(logging.getLogger("qq-bot"), "exception"):
                queue.enqueue(private_event(1, 1), process)
                queue.enqueue(private_event(1, 2), process)
                self.assertTrue(second_done.wait(timeout=1))
            self.assertEqual([1, 2], processed)
        finally:
            queue.executor.shutdown(wait=True, cancel_futures=True)

    def test_session_key_keeps_each_group_conversation_separate(self):
        self.assertEqual("private:7", get_event_session_key(private_event(7, 1)))
        self.assertEqual("group:10:7", get_event_session_key(group_event(10, 7, 2)))
        self.assertEqual("group:10:7", get_event_session_key(group_event(10, 7, 3)))
        self.assertEqual("group:20:7", get_event_session_key(group_event(20, 7, 4)))
        self.assertEqual("group:10:8", get_event_session_key(group_event(10, 8, 5)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the characterization tests**

Run:

```powershell
python -m unittest tests.test_messaging.MessageQueueConcurrencyTests -v
```

Expected: 4 tests pass. A failure means the current queue does not satisfy the approved behavioral foundation; stop and diagnose before changing configuration.

- [ ] **Step 3: Commit the characterization tests**

```powershell
git add tests/test_messaging.py
git commit -m "test: cover concurrent message scheduling"
```

### Task 2: Add configurable worker capacity with TDD

**Files:**
- Modify: `tests/test_messaging.py`
- Modify: `src/config.py`

- [ ] **Step 1: Write failing configuration tests**

Add these imports and test class to `tests/test_messaging.py`:

```python
import os

from src.config import Config


class MessageWorkerConfigurationTests(unittest.TestCase):
    def config_with(self, value):
        environment = {} if value is None else {"MESSAGE_WORKERS": value}
        with mock.patch.dict(os.environ, environment, clear=True):
            return Config()

    def test_message_workers_defaults_to_eight(self):
        self.assertEqual(8, self.config_with(None).message_workers)

    def test_positive_message_workers_value_is_used(self):
        self.assertEqual(16, self.config_with("16").message_workers)

    def test_non_positive_message_workers_value_is_clamped_to_one(self):
        self.assertEqual(1, self.config_with("0").message_workers)
        self.assertEqual(1, self.config_with("-3").message_workers)

    def test_invalid_message_workers_value_falls_back_to_eight(self):
        self.assertEqual(8, self.config_with("many").message_workers)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m unittest tests.test_messaging.MessageWorkerConfigurationTests -v
```

Expected: all 4 tests error with `AttributeError: 'Config' object has no attribute 'message_workers'`.

- [ ] **Step 3: Implement the minimum configuration field**

In `src/config.py`, add this field immediately after `persist_history`:

```python
    message_workers: int = field(
        default_factory=lambda: max(env_int("MESSAGE_WORKERS", 8), 1)
    )
```

This uses `default_factory` so each explicit `Config()` construction reads its current environment. The module-level `config = Config()` remains startup-only and therefore still requires a restart after `.env` changes.

- [ ] **Step 4: Run the configuration and characterization tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_messaging -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit the configuration change**

```powershell
git add src/config.py tests/test_messaging.py
git commit -m "feat: configure concurrent message workers"
```

### Task 3: Wire configured capacity into the application

**Files:**
- Modify: `tests/test_messaging.py`
- Modify: `src/main.py`

- [ ] **Step 1: Write a failing main-wiring test**

Add this import and test class to `tests/test_messaging.py`:

```python
import src.main as main


class MainMessageQueueConfigurationTests(unittest.TestCase):
    def test_global_message_queue_uses_configured_worker_count(self):
        self.assertEqual(
            main.config.message_workers,
            main.message_queue.executor._max_workers,
        )
```

The test reads the executor capacity only for verification; production code does not depend on this private standard-library attribute.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m unittest tests.test_messaging.MainMessageQueueConfigurationTests -v
```

Expected: FAIL because the global queue still has 4 workers while `config.message_workers` defaults to 8.

- [ ] **Step 3: Replace the hard-coded capacity**

In `src/main.py`, change only the queue construction:

```python
message_queue = MessageQueue(
    max_workers=config.message_workers,
    max_processed_message_ids=MAX_PROCESSED_MESSAGE_IDS,
)
```

- [ ] **Step 4: Run all messaging tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_messaging -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Commit the application wiring**

```powershell
git add src/main.py tests/test_messaging.py
git commit -m "feat: enable concurrent conversations by configuration"
```

### Task 4: Document and apply the concurrency setting

**Files:**
- Modify: `tests/test_qqbot_branding.py`
- Modify: `.env.example`
- Modify: `README.md`
- Modify locally but do not stage: `.env`

- [ ] **Step 1: Write a failing operator-documentation test**

Add this method to `QqbotBrandingTests` in `tests/test_qqbot_branding.py`:

```python
    def test_operator_files_describe_concurrent_conversation_setting(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertIn("MESSAGE_WORKERS=8", env_example)
        self.assertIn("MESSAGE_WORKERS=8", readme)
        self.assertIn("不同会话可以并行", readme)
        self.assertIn("同一会话仍按顺序", readme)
```

- [ ] **Step 2: Run the documentation test and verify RED**

Run:

```powershell
python -m unittest tests.test_qqbot_branding.QqbotBrandingTests.test_operator_files_describe_concurrent_conversation_setting -v
```

Expected: FAIL because `MESSAGE_WORKERS=8` and the concurrency explanation are absent.

- [ ] **Step 3: Add the example configuration**

Append this line to `.env.example` after `PERSIST_HISTORY=true`:

```env
MESSAGE_WORKERS=8
```

- [ ] **Step 4: Update the README configuration block and behavior explanation**

Add this line to the README `.env` block after `PERSIST_HISTORY=true`:

```env
MESSAGE_WORKERS=8      # 同时处理的活跃会话数
```

Add this paragraph immediately after the configuration block:

```markdown
默认可同时处理 8 个活跃会话。不同会话可以并行处理，同一会话仍按顺序处理和回复；可通过 `MESSAGE_WORKERS` 调整并发会话数，修改后需要重启。
```

- [ ] **Step 5: Make the minimum local `.env` update without exposing or rewriting other values**

First, check only whether the key exists:

```powershell
if (Select-String -Path .env -Pattern '^MESSAGE_WORKERS=' -Quiet) { 'present' } else { 'missing' }
```

If the result is `missing`, use `apply_patch` to append this single line at the end of `.env`:

```env
MESSAGE_WORKERS=8
```

If the result is `present`, leave `.env` unchanged. Do not print, normalize, or rewrite any other `.env` value.

- [ ] **Step 6: Run the documentation tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_qqbot_branding -v
```

Expected: all branding tests pass.

- [ ] **Step 7: Confirm the ignored local environment file is not staged**

Run:

```powershell
git check-ignore -v .env
git diff --cached --name-only
```

Expected: `.env` is ignored by `.gitignore`, and `.env` is absent from the staged-file list.

- [ ] **Step 8: Commit operator-facing configuration**

```powershell
git add .env.example README.md tests/test_qqbot_branding.py
git commit -m "docs: describe concurrent conversations"
```

### Task 5: Verify the complete change

**Files:**
- Verify: `src/config.py`
- Verify: `src/main.py`
- Verify: `src/messaging.py`
- Verify: `tests/test_messaging.py`
- Verify: `tests/test_qqbot_branding.py`
- Verify: `.env.example`
- Verify: `README.md`

- [ ] **Step 1: Run the full test suite with identity isolation**

```powershell
$env:BOT_NAME='qqbot'
python -m unittest discover -s tests -v
```

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 2: Compile source and tests**

```powershell
python -m compileall -q src tests
```

Expected: exit code 0 and no compilation errors.

- [ ] **Step 3: Check patch formatting and repository state**

```powershell
git diff --check
git status --short --branch
```

Expected: `git diff --check` exits 0; only intentional files appear before their commits, and `.env` never appears as tracked or staged.

- [ ] **Step 4: Verify the effective local concurrency configuration without printing other settings**

```powershell
python -c "from src.config import config; print(config.message_workers)"
```

Expected: prints `8` unless the user intentionally kept another existing positive `MESSAGE_WORKERS` value.

- [ ] **Step 5: Perform final requirements review**

Confirm from fresh test output and the final diff that:

```text
- different session keys can execute concurrently;
- one session executes one message at a time in FIFO order;
- a failed message does not block the next message;
- the default worker count is 8 and environment overrides are validated;
- the global application queue uses the configured count;
- README and .env.example explain the behavior;
- .env is ignored and only receives the permitted local key when absent.
```

No additional production commit is needed if Tasks 1–4 left the working tree clean.
