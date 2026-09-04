"""Tavily search provider for the simple search runtime."""

from __future__ import annotations

from datetime import date
import time
from typing import Any

from src.search.simple.models import SearchMode, SearchQuery
from src.search.simple.providers import (
    ProviderErrorCode,
    ProviderHit,
    ProviderReadiness,
    ProviderResult,
    ProviderStatus,
    SearchProvider,
)

try:
    from tavily import TavilyClient
    from tavily.errors import BadRequestError, TimeoutError as TavilyTimeoutError
except ImportError:
    TavilyClient = None
    BadRequestError = ()
    TavilyTimeoutError = ()


class TavilySearchProvider(SearchProvider):
    name = "tavily"

    def __init__(self, *, api_key: str, proxy_url: str) -> None:
        self._api_key = api_key
        self._proxy_url = proxy_url
        self._client: Any = None

    def readiness(self) -> ProviderReadiness:
        configured = bool(self._api_key)
        available = bool(self._api_key and TavilyClient is not None)
        return ProviderReadiness("tavily", configured=configured, available=available)

    def search(
        self,
        query: SearchQuery,
        *,
        mode: SearchMode,
        max_results: int,
        timeout_seconds: float,
    ) -> ProviderResult:
        readiness = self.readiness()
        if not readiness.available:
            return ProviderResult(
                provider=self.name,
                status=(
                    ProviderStatus.UNAVAILABLE
                    if readiness.configured
                    else ProviderStatus.NOT_CONFIGURED
                ),
                hits=(),
                latency_ms=0.0,
            )

        client = self._get_client()
        params, date_filter_normalized = _tavily_params(
            query,
            mode=mode,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )
        started_at = time.monotonic()
        parameter_retry_attempted = False
        error_code: ProviderErrorCode | None = None

        try:
            response = client.search(query.text, **params)
        except BadRequestError:
            if not _has_date_bounds(params):
                return _error_result(
                    self.name,
                    started_at,
                    ProviderErrorCode.INVALID_PARAMETERS,
                    date_filter_normalized=date_filter_normalized,
                    parameter_retry_attempted=parameter_retry_attempted,
                )
            parameter_retry_attempted = True
            date_filter_normalized = True
            remaining = max(0.0, timeout_seconds - (time.monotonic() - started_at))
            retry_params = dict(params)
            retry_params.pop("start_date", None)
            retry_params.pop("end_date", None)
            retry_params["timeout"] = remaining
            try:
                response = client.search(query.text, **retry_params)
            except BadRequestError:
                return _error_result(
                    self.name,
                    started_at,
                    ProviderErrorCode.INVALID_PARAMETERS,
                    date_filter_normalized=True,
                    parameter_retry_attempted=True,
                )
            except (TimeoutError, TavilyTimeoutError):
                return _timeout_result(
                    self.name,
                    started_at,
                    date_filter_normalized=True,
                    parameter_retry_attempted=True,
                )
            except (ConnectionError, OSError):
                return _error_result(
                    self.name,
                    started_at,
                    ProviderErrorCode.CONNECTION,
                    date_filter_normalized=True,
                    parameter_retry_attempted=True,
                )
            except Exception:
                return _error_result(
                    self.name,
                    started_at,
                    ProviderErrorCode.UNKNOWN,
                    date_filter_normalized=True,
                    parameter_retry_attempted=True,
                )
            error_code = ProviderErrorCode.INVALID_PARAMETERS
        except (TimeoutError, TavilyTimeoutError):
            return _timeout_result(
                self.name,
                started_at,
                date_filter_normalized=date_filter_normalized,
                parameter_retry_attempted=parameter_retry_attempted,
            )
        except (ConnectionError, OSError):
            return _error_result(
                self.name,
                started_at,
                ProviderErrorCode.CONNECTION,
                date_filter_normalized=date_filter_normalized,
                parameter_retry_attempted=parameter_retry_attempted,
            )
        except Exception:
            return _error_result(
                self.name,
                started_at,
                ProviderErrorCode.UNKNOWN,
                date_filter_normalized=date_filter_normalized,
                parameter_retry_attempted=parameter_retry_attempted,
            )

        latency = _elapsed_ms(started_at)
        raw_results = response.get("results", []) if isinstance(response, dict) else []
        hits = [
            ProviderHit(
                provider=self.name,
                query_id=query.query_id,
                title=str(item.get("title") or "").strip(),
                url=str(item.get("url") or "").strip(),
                snippet=str(item.get("content") or "").strip() or None,
                score=_safe_float(item.get("score")),
                raw_content=str(item.get("raw_content") or "").strip() or None,
            )
            for item in raw_results
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        ]

        status = ProviderStatus.SUCCESS if hits else ProviderStatus.EMPTY
        return ProviderResult(
            provider=self.name,
            status=status,
            hits=tuple(hits),
            latency_ms=latency,
            error_code=error_code,
            date_filter_normalized=date_filter_normalized,
            parameter_retry_attempted=parameter_retry_attempted,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            proxies = (
                {"http": self._proxy_url, "https": self._proxy_url}
                if self._proxy_url
                else None
            )
            self._client = TavilyClient(api_key=self._api_key, proxies=proxies)
        return self._client


def _tavily_params(
    query: SearchQuery,
    *,
    mode: SearchMode,
    max_results: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], bool]:
    depth = "basic" if mode is SearchMode.LIGHT else "advanced"
    params: dict[str, Any] = {
        "search_depth": depth,
        "max_results": max_results,
        "timeout": timeout_seconds,
        "include_raw_content": mode is not SearchMode.LIGHT,
    }
    equal_bounds = query.date_from is not None and query.date_from == query.date_to
    if not equal_bounds:
        if query.date_from is not None:
            params["start_date"] = query.date_from.isoformat()
        if query.date_to is not None:
            params["end_date"] = query.date_to.isoformat()
    if query.include_domains:
        params["include_domains"] = query.include_domains
    if query.exclude_domains:
        params["exclude_domains"] = query.exclude_domains
    if query.news:
        params["topic"] = "news"
    return params, equal_bounds


def _has_date_bounds(params: dict[str, Any]) -> bool:
    return "start_date" in params or "end_date" in params


def _elapsed_ms(started_at: float) -> float:
    return max((time.monotonic() - started_at) * 1000.0, 0.0)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timeout_result(
    provider: str,
    started_at: float,
    *,
    date_filter_normalized: bool,
    parameter_retry_attempted: bool,
) -> ProviderResult:
    return ProviderResult(
        provider=provider,
        status=ProviderStatus.TIMEOUT,
        hits=(),
        latency_ms=_elapsed_ms(started_at),
        date_filter_normalized=date_filter_normalized,
        parameter_retry_attempted=parameter_retry_attempted,
    )


def _error_result(
    provider: str,
    started_at: float,
    error_code: ProviderErrorCode,
    *,
    date_filter_normalized: bool,
    parameter_retry_attempted: bool,
) -> ProviderResult:
    return ProviderResult(
        provider=provider,
        status=ProviderStatus.ERROR,
        hits=(),
        latency_ms=_elapsed_ms(started_at),
        error_code=error_code,
        date_filter_normalized=date_filter_normalized,
        parameter_retry_attempted=parameter_retry_attempted,
    )
