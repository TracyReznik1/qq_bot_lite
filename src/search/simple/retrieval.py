"""Request-local Tavily-first search with DDGS fallback."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
import ipaddress
from typing import Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from src.search.simple.models import (
    SearchMode,
    SearchPlan,
    SearchQuery,
    SearchResult,
    SearchTrace,
)
from src.search.simple.providers import (
    ProviderHit,
    ProviderResult,
    ProviderStatus,
    SearchProvider,
)
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
    """Run simple providers concurrently, preferring Tavily per query."""

    def __init__(
        self,
        providers: Sequence[SearchProvider],
        tavily_timeout: float,
        ddgs_timeout: float,
        max_results_per_query: int,
    ) -> None:
        self._providers = {
            str(provider.name).strip().lower(): provider
            for provider in providers
            if hasattr(provider, "name") and str(provider.name).strip()
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

        statuses: dict[str, list[str]] = {}
        collected_hits: list[ProviderHit] = []

        tavily_hits, unresolved = self._execute_provider_batch(
            "tavily",
            queries,
            plan.mode,
            self._timeouts["tavily"],
            statuses,
        )
        collected_hits.extend(tavily_hits)

        if unresolved:
            ddgs_hits, _ = self._execute_provider_batch(
                "ddgs",
                tuple(unresolved),
                plan.mode,
                self._timeouts["ddgs"],
                statuses,
            )
            collected_hits.extend(ddgs_hits)

        trace.provider_statuses.update(
            (provider, _summarize_statuses(values))
            for provider, values in statuses.items()
        )
        limit = 5 if plan.mode is SearchMode.LIGHT else 8
        results = _convert_hits(collected_hits, limit=limit)
        trace.candidate_count = len(results)
        return results

    def _execute_provider_batch(
        self,
        provider_name: str,
        queries: tuple[SearchQuery, ...],
        mode: SearchMode,
        timeout: float,
        statuses: dict[str, list[str]],
    ) -> tuple[list[ProviderHit], list[SearchQuery]]:
        provider = self._providers.get(provider_name)
        if provider is None:
            statuses.setdefault(provider_name, []).extend(["unavailable"] * len(queries))
            return [], list(queries)

        readiness = _check_readiness(provider)
        if readiness is not None:
            statuses.setdefault(provider_name, []).extend([readiness] * len(queries))
            return [], list(queries)

        executor = ThreadPoolExecutor(
            max_workers=len(queries),
            thread_name_prefix=f"simple-search-{provider_name}",
        )
        futures_map = {}
        try:
            for query in queries:
                future = executor.submit(
                    provider.search,
                    query,
                    mode=mode,
                    max_results=self._max_results_per_query,
                    timeout_seconds=timeout,
                )
                futures_map[query] = future

            wait(tuple(futures_map.values()), timeout=timeout)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        resolved_hits: list[ProviderHit] = []
        unresolved_queries: list[SearchQuery] = []

        for query in queries:
            future = futures_map[query]
            if not future.done():
                statuses.setdefault(provider_name, []).append("timeout")
                unresolved_queries.append(query)
                continue

            try:
                res = future.result()
            except TimeoutError:
                statuses.setdefault(provider_name, []).append("timeout")
                unresolved_queries.append(query)
            except Exception:
                statuses.setdefault(provider_name, []).append("error")
                unresolved_queries.append(query)
            else:
                status_val = getattr(res.status, "value", str(res.status)).lower()
                usable_hits = [
                    hit
                    for hit in res.hits
                    if _safe_canonical_url(hit.url) is not None
                ]
                if status_val == "success" and usable_hits:
                    statuses.setdefault(provider_name, []).append("success")
                    resolved_hits.extend(usable_hits)
                else:
                    norm_status = status_val if status_val in {
                        "empty", "timeout", "error", "not_configured", "unavailable"
                    } else "empty"
                    statuses.setdefault(provider_name, []).append(norm_status)
                    unresolved_queries.append(query)

        return resolved_hits, unresolved_queries


def _check_readiness(provider: SearchProvider) -> str | None:
    try:
        readiness = provider.readiness()
    except Exception:
        return "unavailable"
    if not bool(getattr(readiness, "configured", False)):
        return "not_configured"
    if not bool(getattr(readiness, "available", False)):
        return "unavailable"
    return None


def _summarize_statuses(statuses: list[str]) -> str:
    if not statuses:
        return "unavailable"
    if len(set(statuses)) == 1:
        return statuses[0]
    counts = Counter(statuses)
    return ",".join(f"{status}:{counts[status]}" for status in sorted(counts))


def _convert_hits(
    hits: Sequence[ProviderHit],
    *,
    limit: int,
) -> tuple[SearchResult, ...]:
    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for hit in hits:
        canonical = _safe_canonical_url(hit.url)
        if canonical is None or canonical in seen_urls:
            continue
        seen_urls.add(canonical)
        score = hit.score if hit.score is not None else 0.5
        results.append(
            SearchResult(
                result_id=f"R{len(results) + 1}",
                title=hit.title,
                url=canonical,
                excerpt=hit.snippet or hit.raw_content or "",
                provider=hit.provider,
                score=score,
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
