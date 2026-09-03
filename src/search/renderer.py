"""Deterministic view rendering: citations, disclosures, conflicts, QQ chunks."""

from __future__ import annotations

import re
import time
from typing import Mapping, Sequence

from src.search.models import (
    AnswerBlock,
    DisclosureCode,
    EvidenceConflict,
    EvidenceItem,
    RenderedReply,
    RenderState,
    SearchTrace,
    WarningCode,
)

_PARTIAL_PREFIX = "以下只回答已获得证据支持的部分；其余部分暂无法确认。"
_CONFLICT_PREFIX = "来源之间存在未解决差异，下面分别列出，不合并为单一结论。"
_DYNAMIC_REFUSAL = "我暂时无法完成在线核验，因此不能确认当前结论。"
_NO_WEB_DYNAMIC_LIMIT = "根据你的要求，本次没有联网核验；涉及当前状态的结论无法确认。"
_VALIDATION_FAILED = "回答未能通过证据核验，已移除无法确认的内容。"
_VALIDATION_UNAVAILABLE = "已获得网页材料，但本次未能完成语义支撑核验；以下表述应谨慎看待。"
_JUDGE_UNAVAILABLE = "已找到网页内容，但暂时无法可靠判断其是否直接支持问题。"

_HIGH_CONSEQUENCE_WARNING = "重要提示：搜索结果可能不完整或不准确，不能替代适当的专业判断。"
_NO_WEB_HIGH_CONSEQUENCE_WARNING = (
    "重要提示：本次未联网核验，以下内容不能替代适当的专业判断。"
)

_DISCLOSURE_TEXT = {
    DisclosureCode.ONLINE_VERIFICATION_FAILED: _DYNAMIC_REFUSAL,
    DisclosureCode.JUDGE_UNAVAILABLE: _JUDGE_UNAVAILABLE,
    DisclosureCode.PARTIAL_EVIDENCE: _PARTIAL_PREFIX,
    DisclosureCode.SOURCE_CONFLICT: _CONFLICT_PREFIX,
    DisclosureCode.VALIDATION_UNAVAILABLE: _VALIDATION_UNAVAILABLE,
    DisclosureCode.VALIDATION_FAILED: _VALIDATION_FAILED,
    DisclosureCode.USER_FORBID_WEB: _NO_WEB_DYNAMIC_LIMIT,
}

_SOURCE_MARKER = re.compile(r"来源[：:]\s*https?://\S+")
_SOURCE_HEADING = re.compile(r"^来源[：:]", re.MULTILINE)
_URL_PATTERN = re.compile(r"https?://[^\s<>'\"\]]+")


def split_qq_reply(text: str, limit: int) -> list[str]:
    limit = max(int(limit or 0), 1)
    text = str(text or "")
    if not text:
        return []

    source_split = _split_source_section(text, limit)
    if source_split is not None:
        body, source_chunks = source_split
        return [*_split_general_text(body, limit), *source_chunks]
    return _split_general_text(text, limit)


def _split_general_text(text: str, limit: int) -> list[str]:
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        url_match = _URL_PATTERN.match(remaining)
        if url_match is not None and url_match.end() > limit:
            parts.append(url_match.group(0))
            remaining = remaining[url_match.end():].strip()
            continue
        cut = _split_boundary(remaining, limit)
        if cut <= 0:
            parts.append(remaining[:limit])
            remaining = remaining[limit:].strip()
            continue
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return [part for part in parts if part]


def _split_source_section(text: str, limit: int) -> tuple[str, list[str]] | None:
    marker = "\n\n来源：\n"
    if marker not in text:
        return None
    body, raw_sources = text.split(marker, 1)
    lines = raw_sources.splitlines()
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        title_line = lines[index].strip()
        if (
            not re.match(r"^\[\d+\]\s+\S", title_line)
            or index + 1 >= len(lines)
            or _URL_PATTERN.fullmatch(lines[index + 1].strip()) is None
        ):
            return None
        entries.append((title_line, lines[index + 1].strip()))
        index += 2

    chunks: list[str] = []
    current = "来源："
    for title_line, url in entries:
        entry = f"{title_line}\n{url}"
        if len(url) > limit:
            if current:
                title_chunk = f"{current}\n{title_line}" if current else title_line
                chunks.extend(_split_general_text(title_chunk, limit))
                current = ""
            chunks.append(url)
            continue
        candidate = f"{current}\n{entry}" if current else entry
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(entry) <= limit:
            current = entry
            continue
        bounded_title = _source_title_line_within_budget(title_line, url, limit)
        if bounded_title is not None:
            current = f"{bounded_title}\n{url}"
            continue
        title_chunks = _split_general_text(title_line, limit)
        chunks.extend(title_chunks)
        current = url
    if current:
        chunks.append(current)
    return body, [chunk for chunk in chunks if chunk]


def _split_boundary(text: str, limit: int) -> int:
    window = text[:limit]
    for match in _URL_PATTERN.finditer(window):
        if match.start() > 0 and match.end() > limit - 1:
            return match.start()
    for separator in ("\n", "。", "！", "？", "；"):
        index = window.rfind(separator)
        if index > limit // 2:
            return index + 1
    if window and _URL_PATTERN.search(window):
        last_url = list(_URL_PATTERN.finditer(window))[-1]
        if last_url.end() >= limit:
            return last_url.start()
    cut = window.rfind(" ")
    if cut > limit // 2:
        return cut + 1
    return limit


def _source_title_line_within_budget(
    title_line: str,
    url: str,
    limit: int,
) -> str | None:
    available = limit - len(url) - 1
    match = re.fullmatch(r"(?P<prefix>\[\d+\]\s+)(?P<title>.+)", title_line)
    if match is None or available <= len(match.group("prefix")):
        return None
    prefix = match.group("prefix")
    title = _bounded_title(match.group("title"), available - len(prefix))
    return f"{prefix}{title}"


def render_search_reply(state: RenderState, *, qq_limit: int) -> RenderedReply:
    """Render a deterministic view state into a QQ reply."""
    body = _render_blocks(state.visible_blocks)
    conflicts = _render_conflicts(state.conflict_groups)
    warnings = _render_warning_codes(state.warning_codes, state.disclosure_codes)
    disclosures = _render_disclosure_codes(state.disclosure_codes, state.warning_codes)
    used_ids, shown_urls = _source_metadata(
        state.used_sources,
        state.citation_map,
    )
    return _finish_render(
        body,
        conflicts,
        disclosures,
        warnings,
        used_ids,
        shown_urls,
        qq_limit,
    )


def _render_blocks(blocks: Sequence[AnswerBlock]) -> str:
    return "\n".join(block.text for block in blocks).strip()


def _render_conflicts(
    conflicts: Sequence[EvidenceConflict],
) -> list[str]:
    sections: list[str] = []
    for conflict in conflicts:
        values = list(
            dict.fromkeys(
                member.value for member in conflict.members if member.value
            )
        )
        if values:
            sections.append(
                f"冲突点（{conflict.conflict_key}）：\n"
                + "\n".join(f"- {value}" for value in values)
            )
    return sections


def _render_disclosure_codes(
    codes: Sequence[DisclosureCode],
    warning_codes: Sequence[WarningCode],
) -> list[str]:
    result: list[str] = []
    for code in codes:
        if (
            code is DisclosureCode.USER_FORBID_WEB
            and WarningCode.HIGH_CONSEQUENCE in warning_codes
        ):
            continue
        if code in _DISCLOSURE_TEXT:
            result.append(_DISCLOSURE_TEXT[code])
    return result


def _render_warning_codes(
    codes: Sequence[WarningCode],
    disclosure_codes: Sequence[DisclosureCode],
) -> list[str]:
    if WarningCode.HIGH_CONSEQUENCE not in codes:
        return []
    if DisclosureCode.USER_FORBID_WEB in disclosure_codes:
        return [_NO_WEB_HIGH_CONSEQUENCE_WARNING]
    return [_HIGH_CONSEQUENCE_WARNING]


def _source_metadata(
    sources: Sequence[EvidenceItem],
    citation_map: Mapping[str, int],
) -> tuple[list[str], list[str]]:
    used_ids: list[str] = []
    shown_urls: list[str] = []
    for item in sources:
        if citation_map.get(item.evidence_id) is None or not item.url:
            continue
        used_ids.append(item.evidence_id)
        shown_urls.append(item.url)
    return used_ids, shown_urls


def _finish_render(
    body: str,
    conflicts: list[str],
    disclosures: list[str],
    warnings: list[str],
    used_ids: list[str],
    shown_urls: list[str],
    qq_limit: int,
) -> RenderedReply:
    prefix = _dedupe_strings([*disclosures, *warnings])
    sections: list[str] = []
    if body:
        sections.append(body)
    if conflicts:
        sections.append("\n\n".join(conflicts))
    body_text = "\n\n".join(sections)
    text = _compose_disclosures(body_text, prefix)
    chunks = split_qq_reply(text, qq_limit)
    return RenderedReply(
        text=text,
        chunks=tuple(chunks),
        used_evidence_ids=tuple(used_ids),
        shown_source_urls=tuple(shown_urls),
        degradation_disclosures=tuple(prefix),
    )


def render_plain_reply(text: str, *, trace: SearchTrace, qq_limit: int) -> RenderedReply:
    render_started = time.monotonic()
    cleaned = _strip_source_markers(text)
    chunks = split_qq_reply(cleaned, qq_limit)
    rendered = RenderedReply(
        text=cleaned,
        chunks=tuple(chunks),
        used_evidence_ids=(),
        shown_source_urls=(),
        degradation_disclosures=(),
    )
    trace.qq_render_latency_ms = max((time.monotonic() - render_started) * 1000.0, 0.0)
    trace.citation_count = 0
    return rendered


def _strip_source_markers(text: str) -> str:
    text = str(text or "")
    text = _SOURCE_HEADING.sub("\n@@SOURCE_BOUNDARY@@", text)
    text = text.split("@@SOURCE_BOUNDARY@@", 1)[0]
    text = _SOURCE_MARKER.sub("", text)
    text = re.sub(r"\[(?:SRCH|MEM|CHAT):[^\]]*\]", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _compose_disclosures(body: str, disclosures: Sequence[str]) -> str:
    cleaned = str(body or "").strip()
    ordered = _dedupe_strings(disclosures)
    for disclosure in ordered:
        cleaned = cleaned.replace(disclosure, "").strip()
    prefix = "\n".join(ordered)
    if prefix and cleaned:
        return f"{prefix}\n\n{cleaned}"
    return prefix or cleaned


def _bounded_title(title: str, limit: int = 80) -> str:
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    if len(title) <= limit:
        return title
    if limit <= 1:
        return title[:limit]
    return f"{title[:limit - 1]}…"
