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


def run_stage(call: Callable[[], object], *, timeout_seconds: float) -> StageCallResult:
    """Run a zero-argument stage and seal a timeout result for its caller."""

    if not callable(call):
        raise TypeError("call must be callable")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a finite positive number")
    future = _STAGE_EXECUTOR.submit(call)
    try:
        return StageCallResult(True, future.result(timeout=float(timeout_seconds)))
    except FuturesTimeoutError:
        future.cancel()
        return StageCallResult(False, None)
