"""Deterministic rendering: citations, disclosures, conflicts, QQ chunks."""

from __future__ import annotations

import re
import time
from typing import Sequence

from src.search.models import (
    EvidenceBundle,
    EvidenceState,
    GroundedDraft,
    PotentialHarm,
    RenderedReply,
    SearchFailureCode,
    SearchPipelineResult,
    SearchTier,
    SearchTrace,
    TriggerCode,
    ValidationReport,
)

_STABLE_FALLBACK_PREFIX = "在线检索未完成。以下仅按已有知识作有限说明，可能不完整或已经过时："
_DYNAMIC_REFUSAL = "我暂时无法完成在线核验，因此不能确认当前结论。"
_PARTIAL_PREFIX = "以下只回答已获得证据支持的部分；其余部分暂无法确认。"
_CONFLICT_PREFIX = "来源之间存在未解决差异，下面分别列出，不合并为单一结论。"
_EXPLICIT_SEARCH_FAILED = "你要求了在线搜索，但本次检索未成功完成。"
_SEMANTIC_UNAVAILABLE = "已获得网页材料，但本次未能完成语义支撑核验；以下表述应谨慎看待。"
_NO_WEB_DYNAMIC_LIMIT = "根据你的要求，本次没有联网核验；涉及当前状态的结论无法确认。"
_HIGH_CONSEQUENCE_WARNING = "重要提示：搜索结果可能不完整或不准确，不能替代适当的专业判断。"
_NO_WEB_HIGH_CONSEQUENCE_WARNING = (
    "重要提示：本次未联网核验，以下内容不能替代适当的专业判断。"
)

_LIMITATION_DISCLOSURES = {
    "single_source_authority": "证据限制：本次结论主要依赖单一权威来源，缺少独立来源交叉核验。",
    "weak_source_topics": "证据限制：部分主题仅有较弱来源支持。",
    "no_citable_evidence": "证据限制：本次没有可引用的直接证据。",
}

_FIXED_DISCLOSURES = {
    SearchFailureCode.PROVIDER_NOT_CONFIGURED: "当前搜索服务未配置，无法完成在线核验。",
    SearchFailureCode.PROVIDER_UNAVAILABLE: _DYNAMIC_REFUSAL,
    SearchFailureCode.PROVIDER_TIMEOUT: _DYNAMIC_REFUSAL,
    SearchFailureCode.NO_RESULTS: _DYNAMIC_REFUSAL,
    SearchFailureCode.CONTENT_UNREADABLE: _DYNAMIC_REFUSAL,
    SearchFailureCode.INSUFFICIENT_EVIDENCE: _DYNAMIC_REFUSAL,
    SearchFailureCode.PARTIAL_EVIDENCE: _PARTIAL_PREFIX,
    SearchFailureCode.SOURCE_CONFLICT: _CONFLICT_PREFIX,
    SearchFailureCode.VALIDATION_FAILED: "回答未能通过证据核验，已移除无法确认的内容。",
    SearchFailureCode.USER_FORBID_WEB: _NO_WEB_DYNAMIC_LIMIT,
}

_SOURCE_MARKER = re.compile(r"来源[：:]\s*https?://\S+")
_SOURCE_HEADING = re.compile(r"^来源[：:]", re.MULTILINE)
_SEARCH_SUCCESS_STATUS = (
    r"(?:(?:本次|在线)\s*)?(?:搜索|检索)(?:状态)?\s*(?:[：:]\s*)?"
    r"(?:success|successful|succeeded|completed|成功|已成功|完成|已完成)"
)
_PROFESSIONAL_WARNING_CUE_START = (
    r"(?:[\u4e00-\u9fff]{0,4}提示|请注意|注意|警告|免责声明)\s*[：:,，]?"
)
_PROFESSIONAL_WARNING_SUBJECT_START = (
    r"(?:(?:搜索|检索)结果|(?:以下|上述|此|该)(?:内容|信息|回答|建议)|"
    r"本(?:次)?(?:内容|信息|回答|答复|未联网核验))"
)
_PROFESSIONAL_WARNING_CORE_PATTERN = (
    r"(?:"
    r"(?:不能|不可|不应|请勿|勿)[^。\n！？!?]{0,32}?"
    r"(?:替代|代替|取代|当作|视为|作为)[^。\n！？!?]{0,24}?专业(?:判断|建议|意见)"
    r"|不构成[^。\n！？!?]{0,24}?专业(?:判断|建议|意见)"
    r")"
)
_PROFESSIONAL_WARNING_CUE = re.compile(r"^" + _PROFESSIONAL_WARNING_CUE_START)
_PROFESSIONAL_WARNING_SUBJECT = re.compile(r"^" + _PROFESSIONAL_WARNING_SUBJECT_START)
_PROFESSIONAL_WARNING_CORE = re.compile(_PROFESSIONAL_WARNING_CORE_PATTERN)
_DISCLOSURE_ATOM_START = re.compile(
    r"(?:^|(?<=[。！？!?；;，,：:\n]))[ \t]*", re.MULTILINE
)
_DISCLOSURE_ATOM_END = re.compile(r"[。！？!?；;\n]")
_SEARCH_SUCCESS_STATUS_ATOM = re.compile(
    r"(?:^|(?<=[。！？!?；;，,：:\n]))[ \t]*"
    + _SEARCH_SUCCESS_STATUS
    + r"(?:[ \t]*[：:][ \t]*|[ \t]*(?:[。！？!?；;]+|(?=$|\n)))",
    re.IGNORECASE | re.MULTILINE,
)
_URGENT_DYNAMIC_CODES = {
    SearchFailureCode.PROVIDER_UNAVAILABLE,
    SearchFailureCode.PROVIDER_TIMEOUT,
    SearchFailureCode.NO_RESULTS,
    SearchFailureCode.CONTENT_UNREADABLE,
    SearchFailureCode.INSUFFICIENT_EVIDENCE,
}


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
        # An unbreakable URL longer than the limit is emitted alone, without
        # absorbing the remaining text into the same chunk.
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
        # There is not enough room for even the source number plus the URL.
        # Preserve the URL and keep the title separate as the only possible
        # fallback for this mathematically unrepresentable limit.
        title_chunks = _split_general_text(title_line, limit)
        chunks.extend(title_chunks)
        current = url
    if current:
        chunks.append(current)
    return body, [chunk for chunk in chunks if chunk]


def _split_boundary(text: str, limit: int) -> int:
    window = text[:limit]
    # A URL starting near the boundary must stay whole.
    for match in _URL_PATTERN.finditer(window):
        if match.start() > 0 and match.end() > limit - 1:
            return match.start()
    for separator in ("\n", "。", "！", "？", "；"):
        index = window.rfind(separator)
        if index > limit // 2:
            return index + 1
    # If the window ends inside a URL, cut before the URL.
    if window and _URL_PATTERN.search(window):
        last_url = list(_URL_PATTERN.finditer(window))[-1]
        if last_url.end() >= limit:
            return last_url.start()
    cut = window.rfind(" ")
    if cut > limit // 2:
        return cut + 1
    return limit


_URL_PATTERN = re.compile(r"https?://[^\s<>'\"\]]+")


def _source_title_line_within_budget(
    title_line: str,
    url: str,
    limit: int,
) -> str | None:
    available = limit - len(url) - 1  # one newline between title and URL
    match = re.fullmatch(r"(?P<prefix>\[\d+\]\s+)(?P<title>.+)", title_line)
    if match is None or available <= len(match.group("prefix")):
        return None
    prefix = match.group("prefix")
    title = _bounded_title(match.group("title"), available - len(prefix))
    return f"{prefix}{title}"


def render_search_reply(
    result: SearchPipelineResult,
    validation: ValidationReport | None,
    *,
    knowledge_fallback_text: str = "",
    qq_limit: int,
) -> RenderedReply:
    render_started = time.monotonic()
    decision = result.decision

    if decision.route is SearchTier.SKIP:
        disclosures: list[str] = []
        is_no_web = bool(
            decision.skip_reason
            and decision.skip_reason.value == "user_forbid_web"
        )
        is_high_consequence = _is_high_consequence(result)
        if is_no_web and not is_high_consequence:
            disclosures.append(_NO_WEB_DYNAMIC_LIMIT)
        disclosures.extend(_search_warning_disclosures(result))
        disclosures = _dedupe_strings(disclosures)
        body = _strip_markers(
            knowledge_fallback_text,
            strip_search_disclosures=True,
        )
        if is_no_web and is_high_consequence:
            body = body.replace(_NO_WEB_DYNAMIC_LIMIT, "").strip()
        text = _compose_disclosures(
            body,
            disclosures,
        )
        chunks = split_qq_reply(text, qq_limit)
        rendered = RenderedReply(
            text=text,
            chunks=tuple(chunks),
            used_evidence_ids=(),
            shown_source_urls=(),
            degradation_disclosures=tuple(disclosures),
        )
        return _finish_render(
            result, rendered, render_started,
            knowledge_fallback=bool(knowledge_fallback_text),
        )

    failure = result.failure_code
    evidence = result.evidence

    # Deterministic failure rendering.
    if failure in _FIXED_DISCLOSURES and (evidence is None or failure in _URGENT_DYNAMIC_CODES):
        base = _FIXED_DISCLOSURES[failure]
        explicit_add = ""
        if _explicit_search_failed(result):
            explicit_add = _EXPLICIT_SEARCH_FAILED
        if failure is SearchFailureCode.PARTIAL_EVIDENCE and validation is not None:
            rendered = _render_validated(result, validation, evidence, qq_limit)
            return _finish_render(result, rendered, render_started)
        if failure is SearchFailureCode.SOURCE_CONFLICT and validation is not None:
            rendered = _render_validated(result, validation, evidence, qq_limit)
            return _finish_render(result, rendered, render_started)
        disclosures: list[str] = []
        if knowledge_fallback_text:
            base = _STABLE_FALLBACK_PREFIX
            disclosures.append(_STABLE_FALLBACK_PREFIX)
            text = _strip_markers(
                knowledge_fallback_text,
                strip_search_disclosures=True,
            )
        else:
            disclosures.append(base)
            text = ""
        if explicit_add:
            disclosures.append(_EXPLICIT_SEARCH_FAILED)
        disclosures.extend(_search_warning_disclosures(result))
        disclosures = _dedupe_strings(disclosures)
        text = _compose_disclosures(text, disclosures)
        chunks = split_qq_reply(text, qq_limit)
        rendered = RenderedReply(
            text=text,
            chunks=tuple(chunks),
            used_evidence_ids=(),
            shown_source_urls=(),
            degradation_disclosures=tuple(disclosures),
        )
        return _finish_render(
            result, rendered, render_started,
            knowledge_fallback=bool(knowledge_fallback_text),
        )

    if failure is SearchFailureCode.VALIDATION_FAILED:
        if evidence is not None:
            rendered = _render_validation_failure(result, evidence, qq_limit)
            return _finish_render(result, rendered, render_started)
        disclosures = _dedupe_strings([
            _FIXED_DISCLOSURES[SearchFailureCode.VALIDATION_FAILED],
            *_search_warning_disclosures(result),
        ])
        text = _compose_disclosures("", disclosures)
        rendered = RenderedReply(
            text=text,
            chunks=tuple(split_qq_reply(text, qq_limit)),
            used_evidence_ids=(),
            shown_source_urls=(),
            degradation_disclosures=tuple(disclosures),
        )
        return _finish_render(result, rendered, render_started)

    if validation is None or evidence is None:
        if validation is None and evidence is not None:
            result.trace.degradation_reason = SearchFailureCode.VALIDATION_FAILED
            rendered = _render_validation_failure(result, evidence, qq_limit)
            return _finish_render(result, rendered, render_started)
        else:
            disclosures = _search_warning_disclosures(result)
            text = _compose_disclosures(
                _strip_markers(
                    knowledge_fallback_text or "",
                    strip_search_disclosures=True,
                ),
                disclosures,
            )
        chunks = split_qq_reply(text, qq_limit)
        rendered = RenderedReply(
            text=text,
            chunks=tuple(chunks),
            used_evidence_ids=(),
            shown_source_urls=(),
            degradation_disclosures=tuple(disclosures),
        )
        return _finish_render(
            result, rendered, render_started,
            knowledge_fallback=bool(knowledge_fallback_text),
        )

    rendered = _render_validated(result, validation, evidence, qq_limit)
    return _finish_render(result, rendered, render_started)


def _render_validation_failure(
    result: SearchPipelineResult,
    evidence: EvidenceBundle,
    qq_limit: int,
) -> RenderedReply:
    evidence_by_id = {item.evidence_id: item for item in evidence.evidence_items}
    conflict_member_ids = [
        member.evidence_id
        for conflict in evidence.conflicts
        for member in conflict.members
    ]
    numbered, source_order = _number_sources(conflict_member_ids, evidence_by_id)
    body_parts = _render_conflict_sections(evidence, evidence_by_id, numbered)

    disclosures = [_FIXED_DISCLOSURES[SearchFailureCode.VALIDATION_FAILED]]
    if evidence.evidence_state is EvidenceState.CONFLICTING:
        disclosures.append(_CONFLICT_PREFIX)
    disclosures.extend(_limitation_disclosures(evidence))
    disclosures.extend(_search_warning_disclosures(result))
    disclosures = _dedupe_strings(disclosures)

    source_lines: list[str] = []
    shown_urls: list[str] = []
    for evidence_id in source_order:
        item = evidence_by_id[evidence_id]
        if not item.url or item.url in shown_urls:
            continue
        shown_urls.append(item.url)
        number = numbered[evidence_id]
        title = _bounded_source_title(item.title or item.url, item.url, number, qq_limit)
        source_lines.extend((f"[{number}] {title}", item.url))
    if source_lines:
        source_lines.insert(0, "来源：")

    body = "\n".join(body_parts).strip()
    if source_lines:
        body = f"{body}\n\n{'\n'.join(source_lines)}"
    text = _compose_disclosures(body, disclosures)
    return RenderedReply(
        text=text,
        chunks=tuple(split_qq_reply(text, qq_limit)),
        used_evidence_ids=tuple(source_order),
        shown_source_urls=tuple(shown_urls),
        degradation_disclosures=tuple(disclosures),
    )


def render_plain_reply(text: str, *, trace: SearchTrace, qq_limit: int) -> RenderedReply:
    render_started = time.monotonic()
    cleaned = _strip_markers(text)
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


def _render_validated(
    result: SearchPipelineResult,
    validation: ValidationReport,
    evidence: EvidenceBundle,
    qq_limit: int,
) -> RenderedReply:
    evidence_by_id = {item.evidence_id: item for item in evidence.evidence_items}
    used_order: list[str] = []
    for claim in validation.retained_claims:
        for evidence_id in claim.evidence_ids:
            if evidence_id in evidence_by_id and evidence_id not in used_order:
                used_order.append(evidence_id)

    for conflict in evidence.conflicts:
        for member in conflict.members:
            item = evidence_by_id.get(member.evidence_id)
            if (
                item is not None
                and item.citable
                and item.url
                and member.evidence_id not in used_order
            ):
                used_order.append(member.evidence_id)

    numbered, source_order = _number_sources(used_order, evidence_by_id)

    body_parts: list[str] = []
    disclosures: list[str] = []
    if result.evidence is not None and result.evidence.evidence_state is EvidenceState.PARTIAL:
        disclosures.append(_PARTIAL_PREFIX)
    if result.evidence is not None and result.evidence.evidence_state is EvidenceState.CONFLICTING:
        disclosures.append(_CONFLICT_PREFIX)
    if _semantic_unavailable(validation):
        disclosures.append(_SEMANTIC_UNAVAILABLE)
    disclosures.extend(_limitation_disclosures(evidence))
    disclosures.extend(_search_warning_disclosures(result))
    disclosures = _dedupe_strings(disclosures)

    for block in validation.retained_blocks:
        # Collect evidence ids referenced by this block's claims.
        used_here: list[int] = []
        for claim in validation.retained_claims:
            if claim.block_id == block.block_id:
                for eid in claim.evidence_ids:
                    if eid in numbered and numbered[eid] not in used_here:
                        used_here.append(numbered[eid])
        citations = used_here
        protected_texts = tuple(
            claim.text
            for claim in validation.retained_claims
            if claim.block_id == block.block_id
        )
        text = _strip_markers(
            block.text,
            strip_search_disclosures=True,
            protected_texts=protected_texts,
        )
        if citations:
            text = f"{text}{''.join(f'[{number}]' for number in citations)}"
        body_parts.append(text)

    body_parts.extend(_render_conflict_sections(evidence, evidence_by_id, numbered))

    source_lines: list[str] = []
    shown_urls: list[str] = []
    seen_urls: set[str] = set()
    for evidence_id in source_order:
        item = evidence_by_id[evidence_id]
        if not item.url or item.url in seen_urls:
            continue
        seen_urls.add(item.url)
        shown_urls.append(item.url)
        number = numbered[evidence_id]
        title = _bounded_source_title(item.title or item.url, item.url, number, qq_limit)
        source_lines.append(f"[{number}] {_bounded_title(title)}")
        source_lines.append(item.url)

    if source_order:
        source_lines.insert(0, "来源：")

    body = "\n".join(body_parts).strip()
    if source_lines:
        body = f"{body}\n\n{'\n'.join(source_lines)}"
    body = _compose_disclosures(body, disclosures)

    chunks = split_qq_reply(body, qq_limit)
    return RenderedReply(
        text=body,
        chunks=tuple(chunks),
        used_evidence_ids=tuple(source_order),
        shown_source_urls=tuple(shown_urls),
        degradation_disclosures=tuple(disclosures),
    )


def _render_partial(
    result: SearchPipelineResult,
    validation: ValidationReport | None,
    qq_limit: int,
) -> RenderedReply:
    if validation is None:
        return render_search_reply(
            SearchPipelineResult(result.decision, result.plan, None, result.trace, SearchFailureCode.PROVIDER_UNAVAILABLE),
            None,
            qq_limit=qq_limit,
        )
    return _render_validated(result, validation, result.evidence, qq_limit)


def _render_conflict(
    result: SearchPipelineResult,
    validation: ValidationReport | None,
    qq_limit: int,
) -> RenderedReply:
    if validation is None:
        return render_search_reply(
            SearchPipelineResult(result.decision, result.plan, None, result.trace, SearchFailureCode.PROVIDER_UNAVAILABLE),
            None,
            qq_limit=qq_limit,
        )
    return _render_validated(result, validation, result.evidence, qq_limit)


def _explicit_search_failed(result: SearchPipelineResult) -> bool:
    trigger_codes = result.decision.trigger_codes
    return any(code.value in {"explicit_search", "explicit_verification", "explicit_source_request"} for code in trigger_codes)


def _semantic_unavailable(validation: ValidationReport) -> bool:
    return any("semantic_verification_unavailable" in limitation for limitation in validation.limitations)


def _number_sources(
    used_order: Sequence[str],
    evidence_by_id: dict[str, object],
) -> tuple[dict[str, int], list[str]]:
    numbered: dict[str, int] = {}
    number_by_url: dict[str, int] = {}
    source_order: list[str] = []
    for evidence_id in used_order:
        item = evidence_by_id.get(evidence_id)
        if item is None or not getattr(item, "citable", False):
            continue
        url = str(getattr(item, "url", "") or "")
        if _URL_PATTERN.fullmatch(url) is None:
            continue
        key = str(getattr(item, "canonical_url", "") or url)
        if key in number_by_url:
            numbered[evidence_id] = number_by_url[key]
            continue
        number = len(source_order) + 1
        number_by_url[key] = number
        numbered[evidence_id] = number
        source_order.append(evidence_id)
    return numbered, source_order


def _render_conflict_sections(
    evidence: EvidenceBundle,
    evidence_by_id: dict[str, object],
    numbered: dict[str, int],
) -> list[str]:
    sections: list[str] = []
    for conflict in evidence.conflicts:
        member_lines: list[str] = []
        for member in conflict.members:
            item = evidence_by_id.get(member.evidence_id)
            number = numbered.get(member.evidence_id)
            if item is None or number is None:
                continue
            title = _bounded_title(
                str(getattr(item, "title", "") or getattr(item, "publisher", "") or member.evidence_id),
                60,
            )
            published_at = member.published_at or getattr(item, "published_at", None)
            details: list[str] = []
            if published_at is not None:
                if hasattr(published_at, "date"):
                    details.append(f"日期：{published_at.date().isoformat()}")
                elif hasattr(published_at, "isoformat"):
                    details.append(f"日期：{published_at.isoformat()}")
            if member.relation == "claims_supersession":
                details.append("该来源声称为后续更新")
            detail_text = f"（{'；'.join(details)}）" if details else ""
            member_lines.append(f"- {title}：{member.value}{detail_text}[{number}]")
        if member_lines:
            sections.append(
                f"冲突点（{conflict.conflict_key}）：\n" + "\n".join(member_lines)
            )
    return sections


def _limitation_disclosures(evidence: EvidenceBundle) -> list[str]:
    present = set(evidence.limitations)
    return [
        disclosure
        for code, disclosure in _LIMITATION_DISCLOSURES.items()
        if code in present
    ]


def _is_high_consequence(result: SearchPipelineResult) -> bool:
    decision = result.decision
    return (
        decision.potential_harm is PotentialHarm.HIGH
        or TriggerCode.HIGH_CONSEQUENCE_ACTION in decision.trigger_codes
    )


def _search_warning_disclosures(result: SearchPipelineResult) -> list[str]:
    if not _is_high_consequence(result):
        return []
    decision = result.decision
    if (
        decision.route is SearchTier.SKIP
        and decision.skip_reason is not None
        and decision.skip_reason.value == "user_forbid_web"
    ):
        return [_NO_WEB_HIGH_CONSEQUENCE_WARNING]
    return [_HIGH_CONSEQUENCE_WARNING]


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


def _bounded_source_title(title: str, url: str, number: int, limit: int) -> str:
    if len(url) > limit:
        return _bounded_title(title)
    fixed_length = len(f"[{number}] \n{url}")
    available = max(limit - fixed_length, 1)
    return _bounded_title(title, min(80, available))


def _finish_render(
    result: SearchPipelineResult,
    rendered: RenderedReply,
    started: float,
    *,
    knowledge_fallback: bool = False,
) -> RenderedReply:
    result.trace.qq_render_latency_ms = max((time.monotonic() - started) * 1000.0, 0.0)
    result.trace.citation_count = len(rendered.shown_source_urls)
    result.trace.knowledge_fallback_used = bool(
        result.trace.knowledge_fallback_used or knowledge_fallback
    )
    return rendered


def _bounded_title(title: str, limit: int = 80) -> str:
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    if len(title) <= limit:
        return title
    if limit <= 1:
        return title[:limit]
    return title[: limit - 1].rstrip() + "…"


def _strip_markers(
    text: str,
    *,
    strip_search_disclosures: bool = False,
    protected_texts: Sequence[str] = (),
) -> str:
    text = str(text or "")
    # Discard everything from a model-written source heading onward; the
    # deterministic renderer owns the source list.
    text = _SOURCE_HEADING.sub("\n@@SOURCE_BOUNDARY@@", text)
    text = text.split("@@SOURCE_BOUNDARY@@", 1)[0]
    text = _SOURCE_MARKER.sub("", text)
    text = re.sub(r"\[(?:SRCH|MEM|CHAT):[^\]]*\]", "", text)
    if strip_search_disclosures:
        text = _strip_program_owned_search_disclosures(
            text, protected_texts=protected_texts,
        )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _strip_program_owned_search_disclosures(
    text: str, *, protected_texts: Sequence[str] = (),
) -> str:
    """Remove renderer-owned status and warning atoms from model prose.

    Status atoms remove only the status prefix and its delimiter.  Warning
    atoms are bounded clauses; exact retained-claim spans inside them are
    protected so filtering cannot silently truncate grounded facts.
    """
    text, status_count = _SEARCH_SUCCESS_STATUS_ATOM.subn("", text)
    warning_spans = _professional_warning_spans(text)
    if warning_spans:
        protected_spans = _exact_text_spans(text, protected_texts)
        delete = [False] * len(text)
        for start, end in warning_spans:
            for index in range(start, end):
                delete[index] = True
        for start, end in protected_spans:
            for index in range(start, end):
                delete[index] = False
        text = "".join(char for index, char in enumerate(text) if not delete[index])

    if status_count or warning_spans:
        text = "\n".join(
            line.rstrip("，,；;：: ") for line in text.splitlines() if line.strip()
        )
    return text


def _professional_warning_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return complete, bounded model-owned professional-warning clauses."""
    spans: list[tuple[int, int]] = []
    for boundary in _DISCLOSURE_ATOM_START.finditer(text):
        start = boundary.end()
        end_match = _DISCLOSURE_ATOM_END.search(text, start)
        end = end_match.end() if end_match else len(text)
        body = text[start:end].rstrip("。！？!?；;\n \t")
        if not body or len(body) > 180:
            continue
        core = _PROFESSIONAL_WARNING_CORE.search(body)
        if core is None or body[core.end():].strip():
            continue
        if not (
            _PROFESSIONAL_WARNING_CUE.match(body)
            or _PROFESSIONAL_WARNING_SUBJECT.match(body)
            or core.start() == 0
        ):
            continue
        spans.append((start, end))
    return tuple(_merge_spans(spans))


def _exact_text_spans(
    text: str, protected_texts: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for protected in protected_texts:
        protected = str(protected or "")
        if not protected:
            continue
        offset = 0
        while (start := text.find(protected, offset)) >= 0:
            spans.append((start, start + len(protected)))
            offset = start + len(protected)
    return tuple(_merge_spans(spans))


def _merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
