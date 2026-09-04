"""Request-local Tavily-first search with DDGS fallback."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import ipaddress
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from src.search.simple.models import SearchMode, SearchPlan, SearchQuery, SearchResult, SearchTrace
from src.search.url_policy import canonicalize_public_http_url

_TRACKING_QUERY_KEYS = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "ref_src",
    }
)


class ProviderRunner:
    """Run named legacy providers concurrently, preferring Tavily per query."""

    def __init__(
        self,
        providers: Iterable[object],
        tavily_timeout: float,
        ddgs_timeout: float,
        max_results_per_query: int,
    ) -> None:
        self._providers = {
            str(provider.name).strip().lower(): provider
            for provider in providers
            if str(getattr(provider, "name", "")).strip()
        }
        self._timeouts = {
            "tavily": tavily_timeout,
            "ddgs": ddgs_timeout,
        }
        self._max_results_per_query = max_results_per_query

    def run(self, plan: SearchPlan, trace: SearchTrace) -> tuple[SearchResult, ...]:
        queries = tuple(plan.queries)
        if not queries:
            trace.candidate_count = 0
            return ()

        executor = ThreadPoolExecutor(
            max_workers=len(queries),
            thread_name_prefix="simple-search-provider",
        )
        statuses: dict[str, list[str]] = {}
        collected_hits: list[object] = []
        unresolved = list(queries)
        try:
            tavily_results = self._run_provider(
                executor,
                "tavily",
                queries,
                plan.mode,
                statuses,
            )
            unresolved = []
            for query, result in zip(queries, tavily_results):
                if result.status == "success" and result.hits:
                    collected_hits.extend(result.hits)
                else:
                    unresolved.append(query)

            if unresolved:
                ddgs_results = self._run_provider(
                    executor,
                    "ddgs",
                    tuple(unresolved),
                    plan.mode,
                    statuses,
                )
                for result in ddgs_results:
                    if result.status == "success" and result.hits:
                        collected_hits.extend(result.hits)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        trace.provider_statuses.update(
            (provider, _summarize_statuses(values))
            for provider, values in statuses.items()
        )
        limit = 5 if plan.mode is SearchMode.LIGHT else 8
        results = _convert_hits(collected_hits, limit=limit)
        trace.candidate_count = len(results)
        return results

    def _run_provider(
        self,
        executor: ThreadPoolExecutor,
        provider_name: str,
        queries: tuple[SearchQuery, ...],
        mode: SearchMode,
        statuses: dict[str, list[str]],
    ) -> tuple["_CallResult", ...]:
        provider = self._providers.get(provider_name)
        readiness_status = _readiness_status(provider)
        if readiness_status is not None:
            statuses.setdefault(provider_name, []).extend(
                [readiness_status] * len(queries)
            )
            return tuple(_CallResult(readiness_status) for _ in queries)

        futures = [
            executor.submit(
                provider.search,
                _legacy_query(query, mode),
                tier=_legacy_tier(mode),
                max_results=self._max_results_per_query,
                timeout_seconds=self._timeouts[provider_name],
            )
            for query in queries
        ]
        output: list[_CallResult] = []
        for future in futures:
            try:
                provider_result = future.result()
            except TimeoutError:
                result = _CallResult("timeout")
            except Exception:
                result = _CallResult("error")
            else:
                status = _status_value(getattr(provider_result, "status", "error"))
                hits = tuple(getattr(provider_result, "hits", ()) or ())
                if status == "success" and not hits:
                    status = "empty"
                result = _CallResult(status, hits if status == "success" else ())
            statuses.setdefault(provider_name, []).append(result.status)
            output.append(result)
        return tuple(output)


class _CallResult:
    __slots__ = ("status", "hits")

    def __init__(self, status: str, hits: tuple[object, ...] = ()) -> None:
        self.status = status
        self.hits = hits


def _readiness_status(provider: object | None) -> str | None:
    if provider is None:
        return "unavailable"
    try:
        readiness = provider.readiness()
    except Exception:
        return "unavailable"
    if not bool(getattr(readiness, "configured", False)):
        return "not_configured"
    if not bool(getattr(readiness, "available", False)):
        return "unavailable"
    return None


def _status_value(status: object) -> str:
    value = getattr(status, "value", status)
    normalized = str(value).strip().lower()
    return normalized if normalized in {
        "success", "empty", "timeout", "error", "not_configured", "unavailable"
    } else "error"


def _summarize_statuses(statuses: list[str]) -> str:
    if not statuses:
        return "unavailable"
    if len(set(statuses)) == 1:
        return statuses[0]
    counts = Counter(statuses)
    return ",".join(f"{status}:{counts[status]}" for status in sorted(counts))


def _convert_hits(hits: Iterable[object], *, limit: int) -> tuple[SearchResult, ...]:
    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for hit in hits:
        canonical = _safe_canonical_url(str(getattr(hit, "url", "") or ""))
        if canonical is None or canonical in seen_urls:
            continue
        seen_urls.add(canonical)
        score = getattr(hit, "score", None)
        try:
            normalized_score = 0.5 if score is None else float(score)
        except (TypeError, ValueError):
            normalized_score = 0.5
        results.append(
            SearchResult(
                result_id=f"R{len(results) + 1}",
                title=str(getattr(hit, "title", "") or "").strip(),
                url=canonical,
                excerpt=str(
                    getattr(hit, "snippet", None)
                    or getattr(hit, "raw_content", None)
                    or ""
                ).strip(),
                provider=str(getattr(hit, "provider", "") or "").strip(),
                score=normalized_score,
            )
        )
        if len(results) >= limit:
            break
    return tuple(results)


def _safe_canonical_url(raw_url: str) -> str | None:
    canonical = canonicalize_public_http_url(raw_url)
    if canonical is None:
        return None
    parsed = urlparse(canonical)
    hostname = (parsed.hostname or "").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        return None
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            return None

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in _TRACKING_QUERY_KEYS
    ]
    without_tracking = urlunparse(
        parsed._replace(query=urlencode(filtered_query, doseq=True))
    )
    return canonicalize_public_http_url(without_tracking)


def _legacy_query(query: SearchQuery, mode: SearchMode):
    from src.search.models import QueryPurpose, SearchQuery as OldQuery, SearchRoundKind
    return OldQuery(
        query_id=query.query_id,
        query_index=int(query.query_id.removeprefix("q")),
        round_kind=SearchRoundKind.INITIAL,
        purpose=QueryPurpose.DIRECT,
        text=query.text,
    )


def _legacy_tier(mode: SearchMode):
    from src.search.models import SearchTier
    return SearchTier.LIGHT if mode is SearchMode.LIGHT else SearchTier.STANDARD
