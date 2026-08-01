"""Provider-neutral search interface and the ordered registry."""

from __future__ import annotations

import time
from typing import Any, Iterable, Protocol

from src.search.models import (
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


class ProviderRegistry:
    """Owns ordered adapters; selects Tavily first, DDGS as availability fallback."""

    def __init__(self, providers: Iterable[SearchProvider] | None = None) -> None:
        self._providers: list[SearchProvider] = list(providers or ())
        self.last_attempts: tuple[Any, ...] = ()

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
        attempts: list[Any] = []
        primary_error: ProviderStatus | None = None

        primary = self._primary_provider()
        if primary is not None and primary.readiness().available:
            result = self._call(primary, query, tier=tier, max_results=max_results, timeout_seconds=timeout_seconds)
            attempts.append(result)
            if result.status is ProviderStatus.SUCCESS and result.hits:
                self.last_attempts = tuple(attempts)
                return result
            primary_error = result.status

        fallback = self._fallback_provider()
        if fallback is not None and fallback.readiness().available:
            result = self._call(fallback, query, tier=tier, max_results=max_results, timeout_seconds=timeout_seconds)
            attempts.append(result)
            if result.status is ProviderStatus.SUCCESS and result.hits:
                self.last_attempts = tuple(attempts)
                return result

        self.last_attempts = tuple(attempts)
        if not attempts:
            # No adapter was actually invoked: this is a readiness failure, not
            # an attempted invocation.
            return ProviderResult(
                provider="tavily",
                status=ProviderStatus.NOT_CONFIGURED if not self.configured() else ProviderStatus.UNAVAILABLE,
                hits=(),
                latency_ms=0,
            )
        last_status = attempts[-1].status
        return ProviderResult(
            provider="tavily",
            status=last_status,
            hits=(),
            latency_ms=sum(a.latency_ms for a in attempts),
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
        # Fill real latency on successful adapter returns (adapters may leave
        # latency at zero).
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if result.latency_ms == 0 and elapsed_ms > 0:
            return ProviderResult(
                provider=result.provider,
                status=result.status,
                hits=result.hits,
                latency_ms=elapsed_ms,
            )
        return result

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
