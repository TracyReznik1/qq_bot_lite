"""Provider-neutral search interface and the ordered registry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

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


class ProviderRegistry:
    """Owns ordered adapters; selects Tavily first, DDGS as availability fallback."""

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
    ) -> ProviderSearchOutcome:
        attempts: list[ProviderAttempt] = []
        started = time.monotonic()

        primary = self._primary_provider()
        if primary is not None and primary.readiness().available:
            result = self._call(
                primary,
                query,
                tier=tier,
                max_results=max_results,
                timeout_seconds=self._remaining(timeout_seconds, started),
            )
            attempts.append(self._attempt(primary, query, result))
            if result.status is ProviderStatus.SUCCESS and result.hits:
                return self._outcome(result, attempts)

        fallback = self._fallback_provider()
        remaining = self._remaining(timeout_seconds, started)
        if fallback is not None and fallback.readiness().available and remaining > 0:
            result = self._call(
                fallback,
                query,
                tier=tier,
                max_results=max_results,
                timeout_seconds=remaining,
            )
            attempts.append(self._attempt(fallback, query, result))
            if result.status is ProviderStatus.SUCCESS and result.hits:
                return self._outcome(result, attempts)

        if not attempts:
            # No adapter was actually invoked: this is a readiness failure, not
            # an attempted invocation.
            return ProviderSearchOutcome(
                ProviderResult(
                    provider="tavily",
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

    def _call(
        self,
        provider: SearchProvider,
        query: SearchQuery,
        *,
        tier: SearchTier,
        max_results: int,
        timeout_seconds: float,
    ) -> ProviderResult:
        started = time.monotonic()
        try:
            result = provider.search(
                query,
                tier=tier,
                max_results=max_results,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            return ProviderResult(
                provider=provider.name,
                status=ProviderStatus.ERROR,
                hits=(),
                latency_ms=elapsed_ms,
            )
        # Registry timing is authoritative for this individual adapter call.
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return ProviderResult(
            provider=result.provider,
            status=result.status,
            hits=tuple(result.hits),
            latency_ms=elapsed_ms,
        )

    @staticmethod
    def _remaining(timeout_seconds: float, started: float) -> float:
        return max(float(timeout_seconds) - (time.monotonic() - started), 0.0)

    @staticmethod
    def _attempt(provider: SearchProvider, query: SearchQuery, result: ProviderResult) -> ProviderAttempt:
        readiness = provider.readiness()
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
        for provider in self._providers:
            if provider.name == "tavily":
                return provider
        return None

    def _fallback_provider(self) -> SearchProvider | None:
        for provider in self._providers:
            if provider.name == "ddgs":
                return provider
        return None
