"""DDGS search provider for the simple search runtime."""

from __future__ import annotations

import time
from typing import Any

from src.search.simple.models import SearchMode, SearchQuery
from src.search.simple.providers import (
    ProviderHit,
    ProviderReadiness,
    ProviderResult,
    ProviderStatus,
    SearchProvider,
)

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None


class DDGSSearchProvider(SearchProvider):
    name = "ddgs"

    def __init__(self, *, proxy_url: str, timeout_seconds: float) -> None:
        self._proxy_url = proxy_url
        self._timeout_seconds = timeout_seconds

    def readiness(self) -> ProviderReadiness:
        configured = True
        available = DDGS is not None
        return ProviderReadiness("ddgs", configured=configured, available=available)

    def search(
        self,
        query: SearchQuery,
        *,
        mode: SearchMode,
        max_results: int,
        timeout_seconds: float,
    ) -> ProviderResult:
        del mode
        readiness = self.readiness()
        if not readiness.available:
            return ProviderResult(
                provider=self.name,
                status=ProviderStatus.UNAVAILABLE,
                hits=(),
                latency_ms=0.0,
            )

        started_at = time.monotonic()
        per_call_timeout = min(
            self._timeout_seconds, max(float(timeout_seconds), 0.001)
        )
        try:
            with DDGS(
                proxy=self._proxy_url or None, timeout=per_call_timeout
            ) as client:
                raw_results = client.text(
                    query.text,
                    max_results=max_results,
                    region=(
                        "cn-zh"
                        if any("一" <= ch <= "鿿" for ch in query.text)
                        else "us-en"
                    ),
                )
        except TimeoutError:
            return ProviderResult(
                provider=self.name,
                status=ProviderStatus.TIMEOUT,
                hits=(),
                latency_ms=_elapsed_ms(started_at),
            )
        except Exception:
            return ProviderResult(
                provider=self.name,
                status=ProviderStatus.ERROR,
                hits=(),
                latency_ms=_elapsed_ms(started_at),
            )

        latency = _elapsed_ms(started_at)
        if not raw_results:
            return ProviderResult(
                provider=self.name,
                status=ProviderStatus.EMPTY,
                hits=(),
                latency_ms=latency,
            )

        hits = [
            ProviderHit(
                provider=self.name,
                query_id=query.query_id,
                title=str(item.get("title") or "").strip(),
                url=str(item.get("href") or item.get("url") or "").strip(),
                snippet=str(item.get("body") or "").strip() or None,
                score=None,
                raw_content=None,
            )
            for item in raw_results
            if isinstance(item, dict)
            and str(item.get("href") or item.get("url") or "").strip()
        ]

        status = ProviderStatus.SUCCESS if hits else ProviderStatus.EMPTY
        return ProviderResult(
            provider=self.name,
            status=status,
            hits=tuple(hits),
            latency_ms=latency,
        )


def _elapsed_ms(started_at: float) -> float:
    return max((time.monotonic() - started_at) * 1000.0, 0.0)
