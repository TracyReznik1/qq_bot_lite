"""Compatibility facade over the deterministic simple search pipeline.

This module keeps the legacy public names working so older call sites do not
break; it always executes SearchMode.STANDARD and flattens search results
into a bounded text format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.search.simple.factory import (
    get_simple_search_pipeline,
    reset_simple_search_pipeline,
)
from src.search.simple.models import (
    RequestSource,
    SearchMode,
    SearchRequest,
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


def reset_search_service() -> None:
    reset_simple_search_pipeline()


def search(query: str) -> SearchResult:
    normalized = normalize_search_query(query)
    if not normalized:
        return SearchResult(ok=False, status="empty_query", text="没有可搜索的关键词。")

    request = SearchRequest(
        mode=SearchMode.STANDARD,
        text=normalized,
        images=(),
        source=RequestSource.COMPATIBILITY,
    )
    outcome = get_simple_search_pipeline().run(request)
    if outcome.failure is not None:
        return SearchResult(
            ok=False,
            status=outcome.failure.value,
            text="在线检索未完成。",
        )

    lines: list[str] = []
    for index, item in enumerate(outcome.results, 1):
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
