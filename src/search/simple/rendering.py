from __future__ import annotations

import re
from typing import Sequence

from src.search.simple.models import (
    OutputKind,
    SearchFailure,
    SearchResponse,
    SearchResult,
    SearchTrace,
)

_FAILURE_MESSAGES = {
    SearchFailure.PROVIDER_UNAVAILABLE: "在线搜索暂时不可用，请稍后再试。",
    SearchFailure.NO_RESULTS: "没有找到可用的在线搜索结果。",
}


def split_qq_reply(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not text:
        return []

    parts: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        idx = window.rfind("\n")
        if idx != -1:
            cut = idx + 1
        else:
            cut = max_chars

        part = remaining[:cut]
        if part:
            parts.append(part)
        remaining = remaining[cut:]

    if remaining:
        parts.append(remaining)

    return parts


def render_search_failure(
    failure: SearchFailure,
    *,
    qq_limit: int = 1700,
    trace: SearchTrace,
) -> SearchResponse:
    trace.output_kind = OutputKind.SEARCH_FAILURE
    msg = _FAILURE_MESSAGES.get(failure, "在线搜索暂时不可用，请稍后再试。")
    if qq_limit > 0 and len(msg) > qq_limit:
        msg = msg[:qq_limit]
    return SearchResponse(text=msg, sources=(), trace=trace)


def render_search_answer(
    text: str,
    results: tuple[SearchResult, ...] | list[SearchResult],
    *,
    warning: str | None = None,
    show_sources: bool = False,
    qq_limit: int = 1700,
    trace: SearchTrace,
) -> SearchResponse:
    if text.startswith("根据搜索结果："):
        trace.output_kind = OutputKind.SUMMARY_FALLBACK
    else:
        trace.output_kind = OutputKind.MODEL_ANSWER

    shown_sources: list[SearchResult] = []
    if show_sources:
        seen_urls: set[str] = set()
        for r in results:
            if r.url and r.url not in seen_urls:
                seen_urls.add(r.url)
                shown_sources.append(r)
                if len(shown_sources) >= 3:
                    break

    sources_suffix = ""
    if show_sources and shown_sources:
        source_lines = []
        for i, r in enumerate(shown_sources, 1):
            title = " ".join(r.title.split()).strip()
            if title:
                source_lines.append(f"{i}. {title} {r.url}")
            else:
                source_lines.append(f"{i}. {r.url}")
        sources_suffix = "\n\n来源：\n" + "\n".join(source_lines)

    body = text.strip()
    lines = body.splitlines()
    while lines:
        last_line = lines[-1].strip()
        if not last_line:
            lines.pop()
            continue
        is_disclaimer = (
            any(prefix in last_line for prefix in ("注：", "注:", "提示：", "提示:", "免责声明：", "免责声明:"))
            and any(kw in last_line for kw in ("不完整", "仅供参考", "可靠", "时效", "准确"))
        ) or (
            any(kw in last_line for kw in ("仅供参考", "无法保证完全可靠", "可靠性未知", "信息可能不完整", "搜索信息可能不完整"))
            and len(last_line) <= 60
        )
        if is_disclaimer:
            lines.pop()
        else:
            break
    body = "\n".join(lines).strip()

    cleaned_warning = (warning or "").strip()
    if cleaned_warning in ("信息可能不完整。", "信息不完整。", "信息不完整", "信息可能不完整"):
        cleaned_warning = ""
    warn_suffix = f"\n\n{cleaned_warning}" if cleaned_warning else ""

    if cleaned_warning and cleaned_warning in body:
        body = body.replace(cleaned_warning, "").strip()

    suffix = warn_suffix + sources_suffix
    if qq_limit > 0:
        if len(suffix) >= qq_limit:
            if warn_suffix and len(warn_suffix) <= qq_limit:
                suffix = warn_suffix
            else:
                suffix = suffix[:qq_limit]
            available_body = max(0, qq_limit - len(suffix))
            body = body[:available_body].rstrip()
        else:
            available_body = qq_limit - len(suffix)
            if len(body) > available_body:
                body = body[:available_body].rstrip()

    final_text = (body + suffix).strip()
    if qq_limit > 0 and len(final_text) > qq_limit:
        final_text = final_text[:qq_limit]

    return SearchResponse(
        text=final_text,
        sources=tuple(shown_sources) if show_sources else (),
        trace=trace,
    )
