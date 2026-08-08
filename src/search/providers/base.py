"""Provider-neutral search interface and the ordered registry."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Iterable, Protocol

from src.search.models import (
    ProviderAttempt,
    ProviderReadiness,
    ProviderResult,
    ProviderStatus,
    SearchQuery,
    SearchTier,
)


class SearchProvider(Protocol):
    """A neutral search adapter. Subclasses expose ``name``."""

    name: str

    def readiness(self) -> ProviderReadiness: ...

    def search(
        self,
        query: SearchQuery,
        *,
        tier: SearchTier,
        max_results: int,
        timeout_seconds: float,
    ) -> ProviderResult: ...


@dataclass(frozen=True)
class ProviderSearchOutcome:
    """One registry call plus immutable attempts local to that call."""

    result: ProviderResult
    attempts: tuple[ProviderAttempt, ...] = ()

    @property
    def provider(self) -> str:
        return self.result.provider

    @property
    def status(self) -> ProviderStatus:
        return self.result.status

    @property
    def hits(self) -> tuple[Any, ...]:
        return self.result.hits

    @property
    def latency_ms(self) -> int | float:
        return self.result.latency_ms


# One fixed-capacity pool bounds non-cooperative adapter calls across all
# registries. Timed-out queued futures are canceled; timed-out running calls can
# occupy at most this fixed number of workers until their dependency returns.
_ADAPTER_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="search-adapter",
)

AttemptStartedObserver = Callable[[str, SearchQuery, ProviderReadiness, float], None]
AttemptFinishedObserver = Callable[[ProviderAttempt], None]


_TAVILY_FALLBACK_RESERVE_SECONDS = {
    SearchTier.LIGHT: 3.5,
    SearchTier.STANDARD: 5.0,
    SearchTier.DEEP: 8.0,
}


class ProviderRegistry:
    """Select DDGS first and use Tavily as the bounded fallback."""

    def __init__(self, providers: Iterable[SearchProvider] | None = None) -> None:
        self._providers: list[SearchProvider] = list(providers or ())

    def available_providers(self) -> list[str]:
        return [provider.name for provider in self._providers if provider.readiness().available]

    def readiness(self) -> tuple[ProviderReadiness, ...]:
        return tuple(provider.readiness() for provider in self._providers)

    def configured(self) -> bool:
        return any(provider.readiness().configured for provider in self._providers)

    def search(
        self,
        query: SearchQuery,
        *,
        tier: SearchTier,
        max_results: int,
        timeout_seconds: float,
    ) -> ProviderResult:
        """Preserve the original public ProviderResult contract."""
        return self.search_with_attempts(
            query,
            tier=tier,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        ).result

    def search_with_attempts(
        self,
        query: SearchQuery,
        *,
        tier: SearchTier,
        max_results: int,
        timeout_seconds: float,
        on_attempt_started: AttemptStartedObserver | None = None,
        on_attempt_finished: AttemptFinishedObserver | None = None,
    ) -> ProviderSearchOutcome:
        """Search with immutable, request-local attempt truth for orchestration."""
        attempts: list[ProviderAttempt] = []
        scheduled_result: ProviderResult | None = None
        scheduled_attempt: ProviderAttempt | None = None
        duration = max(float(timeout_seconds), 0.0)
        deadline = time.monotonic() + duration
        if duration <= 0:
            return ProviderSearchOutcome(
                ProviderResult(
                    provider=self._readiness_provider_name(),
                    status=ProviderStatus.TIMEOUT,
                    hits=(),
                    latency_ms=0,
                ),
                (),
            )

        primary = self._primary_provider()
        fallback = self._fallback_provider()
        if (
            primary is not None
            and primary.readiness().available
            and self._remaining(deadline) > 0
        ):
            result, attempt = self._call_until_deadline(
                primary,
                query,
                tier=tier,
                max_results=max_results,
                deadline=self._primary_deadline(deadline, tier, fallback),
                on_attempt_started=on_attempt_started,
                on_attempt_finished=on_attempt_finished,
            )
            scheduled_result = result
            scheduled_attempt = attempt
            if attempt is not None:
                attempts.append(attempt)
            if result.status is ProviderStatus.SUCCESS and result.hits:
                return self._outcome(result, attempts)

        remaining = self._remaining(deadline)
        if fallback is not None and fallback.readiness().available and remaining > 0:
            result, attempt = self._call_until_deadline(
                fallback,
                query,
                tier=tier,
                max_results=max_results,
                deadline=deadline,
                on_attempt_started=on_attempt_started,
                on_attempt_finished=on_attempt_finished,
            )
            scheduled_result = result
            scheduled_attempt = attempt
            if attempt is not None:
                attempts.append(attempt)
            if result.status is ProviderStatus.SUCCESS and result.hits:
                return self._outcome(result, attempts)

        if scheduled_result is not None and scheduled_attempt is None:
            return ProviderSearchOutcome(
                ProviderResult(
                    provider=scheduled_result.provider,
                    status=scheduled_result.status,
                    hits=scheduled_result.hits,
                    latency_ms=(
                        sum(attempt.latency_ms for attempt in attempts)
                        + scheduled_result.latency_ms
                    ),
                ),
                tuple(attempts),
            )
        if not attempts:
            # No adapter was actually invoked: this is a readiness failure, not
            # an attempted invocation.
            return ProviderSearchOutcome(
                ProviderResult(
                    provider=self._readiness_provider_name(),
                    status=ProviderStatus.NOT_CONFIGURED if not self.configured() else ProviderStatus.UNAVAILABLE,
                    hits=(),
                    latency_ms=0,
                ),
                (),
            )
        last_status = attempts[-1].status
        return ProviderSearchOutcome(
            ProviderResult(
                provider=attempts[-1].provider,
                status=last_status,
                hits=(),
                latency_ms=sum(attempt.latency_ms for attempt in attempts),
            ),
            tuple(attempts),
        )

    def _call_until_deadline(
        self,
        provider: SearchProvider,
        query: SearchQuery,
        *,
        tier: SearchTier,
        max_results: int,
        deadline: float,
        on_attempt_started: AttemptStartedObserver | None,
        on_attempt_finished: AttemptFinishedObserver | None,
    ) -> tuple[ProviderResult, ProviderAttempt | None]:
        remaining = self._remaining(deadline)
        if remaining <= 0:
            return (
                ProviderResult(provider.name, ProviderStatus.TIMEOUT, (), 0),
                None,
            )
        readiness = provider.readiness()
        invocation: dict[str, Any] = {"cancelled": False, "started": None}
        invocation_gate = Lock()

        def invoke() -> ProviderResult | None:
            with invocation_gate:
                if invocation["cancelled"] or self._remaining(deadline) <= 0:
                    invocation["cancelled"] = True
                    return None
                invocation["started"] = time.monotonic()
                started_at = invocation["started"]
            if on_attempt_started is not None:
                try:
                    on_attempt_started(provider.name, query, readiness, started_at)
                except Exception:
                    pass
            return provider.search(
                query,
                tier=tier,
                max_results=max_results,
                timeout_seconds=max(self._remaining(deadline), 0.001),
            )

        future = _ADAPTER_EXECUTOR.submit(invoke)
        try:
            result = future.result(timeout=remaining)
        except FuturesTimeoutError:
            with invocation_gate:
                started = invocation["started"]
                if started is None:
                    invocation["cancelled"] = True
            future.cancel()
            elapsed_ms = int((time.monotonic() - (started or time.monotonic())) * 1000)
            result = ProviderResult(
                provider=provider.name,
                status=ProviderStatus.TIMEOUT,
                hits=(),
                latency_ms=max(elapsed_ms, 0),
            )
            if started is None:
                return result, None
        except Exception as exc:
            with invocation_gate:
                started = invocation["started"] or time.monotonic()
            elapsed_ms = int((time.monotonic() - started) * 1000)
            result = ProviderResult(
                provider=provider.name,
                status=(
                    ProviderStatus.TIMEOUT
                    if isinstance(exc, TimeoutError)
                    else ProviderStatus.ERROR
                ),
                hits=(),
                latency_ms=max(elapsed_ms, 0),
            )
        else:
            if result is None:
                return (
                    ProviderResult(provider.name, ProviderStatus.TIMEOUT, (), 0),
                    None,
                )
            with invocation_gate:
                started = invocation["started"] or time.monotonic()
            elapsed_ms = int((time.monotonic() - started) * 1000)
            result = ProviderResult(
                provider=provider.name,
                status=result.status,
                hits=tuple(result.hits),
                latency_ms=max(elapsed_ms, 0),
            )

        attempt = self._attempt(provider, query, result, readiness)
        if on_attempt_finished is not None:
            try:
                on_attempt_finished(attempt)
            except Exception:
                pass
        return result, attempt

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(deadline - time.monotonic(), 0.0)

    @staticmethod
    def _attempt(
        provider: SearchProvider,
        query: SearchQuery,
        result: ProviderResult,
        readiness: ProviderReadiness,
    ) -> ProviderAttempt:
        return ProviderAttempt(
            provider=provider.name,
            status=result.status,
            count=1,
            latency_ms=result.latency_ms,
            query_id=query.query_id,
            configured=readiness.configured,
            available=readiness.available,
            invocation_started=True,
        )

    @staticmethod
    def _outcome(result: ProviderResult, attempts: list[ProviderAttempt]) -> ProviderSearchOutcome:
        return ProviderSearchOutcome(
            ProviderResult(
                provider=result.provider,
                status=result.status,
                hits=result.hits,
                latency_ms=sum(attempt.latency_ms for attempt in attempts),
            ),
            tuple(attempts),
        )

    def _primary_provider(self) -> SearchProvider | None:
        return self._provider_named("ddgs")

    def _fallback_provider(self) -> SearchProvider | None:
        return self._provider_named("tavily")

    def _provider_named(self, name: str) -> SearchProvider | None:
        return next(
            (provider for provider in self._providers if provider.name == name),
            None,
        )

    def _primary_deadline(
        self,
        deadline: float,
        tier: SearchTier,
        fallback: SearchProvider | None,
    ) -> float:
        reserve = _TAVILY_FALLBACK_RESERVE_SECONDS[tier]
        if (
            fallback is not None
            and fallback.readiness().available
            and self._remaining(deadline) > reserve
        ):
            return deadline - reserve
        return deadline

    def _readiness_provider_name(self) -> str:
        if self._providers:
            return self._providers[0].name
        return "registry"
