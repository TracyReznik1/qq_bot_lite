"""Tavily search adapter."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Sequence

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
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - optional dependency
    TavilyClient = None

_TIMEOUT_STATUS = ProviderStatus.TIMEOUT


class TavilySearchProvider(SearchProvider):
    name = "tavily"

    def __init__(self, *, api_key: str, proxy_url: str) -> None:
        self._api_key = api_key
        self._proxy_url = proxy_url
        self._client = None

    def readiness(self) -> ProviderReadiness:
        if not self._api_key:
            return ProviderReadiness("tavily", False, False, SearchFailureCode.PROVIDER_NOT_CONFIGURED)
        if TavilyClient is None:
            return ProviderReadiness("tavily", True, False, SearchFailureCode.PROVIDER_UNAVAILABLE)
        return ProviderReadiness("tavily", True, True, None)

    def search(
        self,
        query: SearchQuery,
        *,
        tier: SearchTier,
        max_results: int,
        timeout_seconds: float,
    ) -> ProviderResult:
        readiness = self.readiness()
        if not readiness.available:
            return ProviderResult(
                provider=self.name,
                status=ProviderStatus.UNAVAILABLE if readiness.configured else ProviderStatus.NOT_CONFIGURED,
                hits=(),
                latency_ms=0,
            )
        client = self._get_client()
        depth = "basic" if tier is SearchTier.LIGHT else "advanced"
        params: dict[str, Any] = {
            "search_depth": depth,
            "max_results": max_results,
            "timeout": timeout_seconds,
            "include_raw_content": tier is not SearchTier.LIGHT,
        }
        if query.date_from is not None:
            params["start_date"] = query.date_from.isoformat()
        if query.date_to is not None:
            params["end_date"] = query.date_to.isoformat()
        if query.include_domains:
            params["include_domains"] = query.include_domains
        if query.exclude_domains:
            params["exclude_domains"] = query.exclude_domains
        if query.purpose.value == "time_bounded":
            params["topic"] = "news"

        try:
            response = client.search(query.text, **params)
        except TimeoutError:
            return ProviderResult(provider=self.name, status=_TIMEOUT_STATUS, hits=(), latency_ms=0)
        except Exception:
            return ProviderResult(provider=self.name, status=ProviderStatus.ERROR, hits=(), latency_ms=0)

        results = response.get("results", []) if isinstance(response, dict) else []
        if not results:
            return ProviderResult(provider=self.name, status=ProviderStatus.EMPTY, hits=(), latency_ms=0)

        hits = tuple(_tavily_hit(item, query.query_id) for item in results if isinstance(item, dict))
        if not hits:
            return ProviderResult(provider=self.name, status=ProviderStatus.EMPTY, hits=(), latency_ms=0)
        return ProviderResult(provider=self.name, status=ProviderStatus.SUCCESS, hits=hits, latency_ms=0)

    def _get_client(self) -> Any:
        if self._client is None:
            proxies = {"http": self._proxy_url, "https": self._proxy_url} if self._proxy_url else None
            self._client = TavilyClient(api_key=self._api_key, proxies=proxies)
        return self._client


def _tavily_hit(item: dict[str, Any], query_id: str) -> ProviderHit:
    published = _parse_datetime(item.get("published_date"))
    raw_content = item.get("raw_content") or item.get("content") or None
    return ProviderHit(
        provider="tavily",
        query_id=query_id,
        title=str(item.get("title") or ""),
        url=str(item.get("url") or ""),
        snippet=_optional_text(item.get("content")),
        score=_optional_float(item.get("score")),
        published_at=published,
        raw_content=_optional_text(raw_content),
        quality_flags=(),
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return parsed
    return None
