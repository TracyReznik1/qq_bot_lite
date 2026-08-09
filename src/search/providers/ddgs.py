"""DDGS primary search adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.search.models import (
    ProviderHit,
    ProviderReadiness,
    ProviderResult,
    ProviderStatus,
    SearchFailureCode,
    SearchQuery,
    SearchTier,
)
from src.search.providers.base import SearchProvider

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - optional dependency
    DDGS = None


class DDGSSearchProvider(SearchProvider):
    name = "ddgs"

    def __init__(self, *, proxy_url: str, timeout_seconds: float) -> None:
        self._proxy_url = proxy_url
        self._timeout_seconds = timeout_seconds

    def readiness(self) -> ProviderReadiness:
        if DDGS is None:
            return ProviderReadiness("ddgs", True, False, SearchFailureCode.PROVIDER_UNAVAILABLE)
        return ProviderReadiness("ddgs", True, True, None)

    def search(
        self,
        query: SearchQuery,
        *,
        tier: SearchTier,
        max_results: int,
        timeout_seconds: float,
    ) -> ProviderResult:
        del tier
        readiness = self.readiness()
        if not readiness.available:
            return ProviderResult(
                provider=self.name,
                status=ProviderStatus.UNAVAILABLE,
                hits=(),
                latency_ms=0,
            )
        try:
            per_call_timeout = min(self._timeout_seconds, max(float(timeout_seconds), 0.001))
            with DDGS(proxy=self._proxy_url or None, timeout=per_call_timeout) as client:
                raw_results = client.text(
                    query.text,
                    max_results=max_results,
                    region="cn-zh" if any("一" <= ch <= "鿿" for ch in query.text) else "us-en",
                )
        except TimeoutError:
            return ProviderResult(provider=self.name, status=ProviderStatus.TIMEOUT, hits=(), latency_ms=0)
        except Exception:
            return ProviderResult(provider=self.name, status=ProviderStatus.ERROR, hits=(), latency_ms=0)

        if not raw_results:
            return ProviderResult(provider=self.name, status=ProviderStatus.EMPTY, hits=(), latency_ms=0)
        hits = tuple(_ddgs_hit(item, query.query_id) for item in raw_results if isinstance(item, dict))
        if not hits:
            return ProviderResult(provider=self.name, status=ProviderStatus.EMPTY, hits=(), latency_ms=0)
        return ProviderResult(provider=self.name, status=ProviderStatus.SUCCESS, hits=hits, latency_ms=0)


def _ddgs_hit(item: dict[str, Any], query_id: str) -> ProviderHit:
    return ProviderHit(
        provider="ddgs",
        query_id=query_id,
        title=str(item.get("title") or ""),
        url=str(item.get("href") or item.get("url") or ""),
        snippet=_optional_text(item.get("body")),
        score=None,
        published_at=_parse_datetime(item.get("date")),
        raw_content=None,
        quality_flags=(),
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text[:10]):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None
