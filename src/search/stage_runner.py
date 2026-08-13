"""Bounded execution for isolated non-provider pipeline stages."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import math
from typing import Callable


_STAGE_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="search-stage")


@dataclass(frozen=True)
class StageCallResult:
    completed: bool
    value: object | None


@dataclass(frozen=True)
class _StageOutcome:
    value: object | None = None
    error: BaseException | None = None


def _run_call(call: Callable[[], object]) -> _StageOutcome:
    try:
        return _StageOutcome(value=call())
    except BaseException as exc:
        return _StageOutcome(error=exc)


def run_stage(call: Callable[[], object], *, timeout_seconds: float) -> StageCallResult:
    """Run a cooperative zero-argument stage and seal the caller's timeout result.

    A timeout and cancellation do not terminate a running worker thread. Callers
    must provide bounded/cooperative callables and must not rely on this runner
    to force-stop their work.
    """

    if not callable(call):
        raise TypeError("call must be callable")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a finite positive number")
    future = _STAGE_EXECUTOR.submit(_run_call, call)
    try:
        outcome = future.result(timeout=float(timeout_seconds))
    except FuturesTimeoutError:
        future.cancel()
        return StageCallResult(False, None)
    if outcome.error is not None:
        raise outcome.error
    return StageCallResult(True, outcome.value)
