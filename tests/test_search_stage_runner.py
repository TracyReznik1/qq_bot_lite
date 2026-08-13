import math
import threading
import time
import unittest

from src.search import stage_runner
from src.search.stage_runner import StageCallResult, run_stage


class SearchStageRunnerTests(unittest.TestCase):
    def test_completed_stage_retains_its_return_value(self):
        value = ("completed", 7)

        result = run_stage(lambda: value, timeout_seconds=0.5)

        self.assertEqual(StageCallResult(True, value), result)

    def test_stage_callable_receives_no_runner_arguments(self):
        # Closure capture is outside this generic runner's enforcement boundary.
        result = run_stage(lambda *args, **kwargs: (args, kwargs), timeout_seconds=0.5)

        self.assertEqual(StageCallResult(True, ((), {})), result)

    def test_queued_stage_is_cancelled_after_timeout(self):
        release = threading.Event()
        started = threading.Semaphore(0)
        queued_started = threading.Event()

        def block_worker():
            started.release()
            release.wait(1.0)

        blockers = [stage_runner._STAGE_EXECUTOR.submit(block_worker) for _ in range(8)]
        try:
            for _ in blockers:
                self.assertTrue(started.acquire(timeout=0.5))

            result = run_stage(queued_started.set, timeout_seconds=0.05)
            self.assertEqual(StageCallResult(False, None), result)
        finally:
            release.set()
            for blocker in blockers:
                blocker.result(timeout=1.0)

        self.assertFalse(queued_started.wait(0.1))

    def test_running_stage_timeout_returns_promptly_and_seals_its_timeout_result(self):
        release = threading.Event()
        started = threading.Event()
        finished = threading.Event()

        def late_call():
            started.set()
            release.wait(1.0)
            finished.set()
            return "late"

        started_at = time.monotonic()
        result = run_stage(late_call, timeout_seconds=0.05)
        elapsed = time.monotonic() - started_at

        self.assertTrue(started.is_set())
        self.assertEqual(StageCallResult(False, None), result)
        self.assertLess(elapsed, 0.25)
        release.set()
        self.assertTrue(finished.wait(1.0))
        self.assertEqual(StageCallResult(False, None), result)

    def test_non_timeout_errors_propagate_to_the_stage_owner(self):
        with self.assertRaisesRegex(RuntimeError, "stage failed"):
            run_stage(lambda: (_ for _ in ()).throw(RuntimeError("stage failed")), timeout_seconds=0.5)

    def test_worker_timeout_error_propagates_to_the_stage_owner(self):
        with self.assertRaisesRegex(TimeoutError, "worker timeout"):
            run_stage(
                lambda: (_ for _ in ()).throw(TimeoutError("worker timeout")),
                timeout_seconds=0.5,
            )

    def test_executor_recovers_after_queued_timeout_when_workers_are_released(self):
        release = threading.Event()
        started = threading.Semaphore(0)

        def block_worker():
            started.release()
            release.wait(1.0)

        blockers = [stage_runner._STAGE_EXECUTOR.submit(block_worker) for _ in range(8)]
        try:
            for _ in blockers:
                self.assertTrue(started.acquire(timeout=0.5))

            self.assertEqual(
                StageCallResult(False, None),
                run_stage(lambda: "queued", timeout_seconds=0.05),
            )
        finally:
            release.set()
            for blocker in blockers:
                blocker.result(timeout=1.0)

        self.assertEqual(
            StageCallResult(True, "healthy"),
            run_stage(lambda: "healthy", timeout_seconds=0.5),
        )

    def test_call_and_timeout_validation_are_closed(self):
        with self.assertRaises(TypeError):
            run_stage(None, timeout_seconds=0.5)

        for value in (True, 0, -1, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    run_stage(lambda: None, timeout_seconds=value)
