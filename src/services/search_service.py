"""Compatibility facade over the unified evidence-search orchestrator.

New chat/command code must call ``get_search_orchestrator()`` directly. This
module only keeps the legacy public names working so older call sites do not
break; it flattens admitted Evidence into a bounded text format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.search import get_search_orchestrator, reset_search_orchestrator
from src.search.models import (
    RequestSource,
    RetrievalRequest,
    SearchFailureCode,
    SearchTier,
)

QUERY_MAX_CHARS = 500


@dataclass(frozen=True)
class SearchResult:
    ok: bool
    status: str
    text: str


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _truncate_text(text: str, limit: int) -> str:
    text = _collapse_spaces(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip("，,。；;：:")


def normalize_search_query(query: str) -> str:
    """Unicode-normalize, trim whitespace, drop /search prefix, cap at 500 chars."""
    query = _collapse_spaces(query)
    if not query:
        return ""
    query = re.sub(r"^/(?:search|s)(?:\s+|$)", "", query, flags=re.IGNORECASE).strip()
    return _truncate_text(query, QUERY_MAX_CHARS)


def search(query: str) -> SearchResult:
    normalized = normalize_search_query(query)
    if not normalized:
        return SearchResult(ok=False, status="empty_query", text="没有可搜索的关键词。")

    request = RetrievalRequest(
        normalized,
        force_search=True,
        request_source=RequestSource.COMPATIBILITY,
    )
    result = get_search_orchestrator().run(request)

    if result.decision.route is SearchTier.SKIP:
        return SearchResult(
            ok=False,
            status="skip",
            text="本次请求未执行在线检索。",
        )

    if result.evidence is None or result.failure_code is not None:
        failure = result.failure_code or SearchFailureCode.PROVIDER_NOT_CONFIGURED
        return SearchResult(
            ok=False,
            status=failure.value,
            text="在线检索未完成。",
        )

    evidence = result.evidence
    lines = ["搜索状态：success", f"搜索词：{normalized}", ""]
    for index, item in enumerate(evidence.evidence_items, 1):
        lines.append(
            f"{index}. {item.title or item.url}\n"
            f"{(item.excerpt or '')[:300]}\n"
            f"{item.url}"
        )
    return SearchResult(ok=True, status="success", text="\n\n".join(lines))


def web_search(query: str) -> str:
    return search(query).text


def has_search_results(search_result: SearchResult) -> bool:
    return search_result.ok
