"""Deterministic rendering: citations, disclosures, conflicts, QQ chunks."""

from __future__ import annotations

import re
from typing import Sequence

from src.search.models import (
    EvidenceBundle,
    EvidenceState,
    GroundedDraft,
    RenderedReply,
    SearchFailureCode,
    SearchPipelineResult,
    SearchTier,
    SearchTrace,
    ValidationReport,
)

_STABLE_FALLBACK_PREFIX = "在线检索未完成。以下仅按已有知识作有限说明，可能不完整或已经过时："
_DYNAMIC_REFUSAL = "我暂时无法完成在线核验，因此不能确认当前结论。"
_PARTIAL_PREFIX = "以下只回答已获得证据支持的部分；其余部分暂无法确认。"
_CONFLICT_PREFIX = "来源之间存在未解决差异，下面分别列出，不合并为单一结论。"
_EXPLICIT_SEARCH_FAILED = "你要求了在线搜索，但本次检索未成功完成。"
_SEMANTIC_UNAVAILABLE = "已获得网页材料，但本次未能完成语义支撑核验；以下表述应谨慎看待。"
_NO_WEB_DYNAMIC_LIMIT = "根据你的要求，本次没有联网核验；涉及当前状态的结论无法确认。"

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


def render_search_reply(
    result: SearchPipelineResult,
    validation: ValidationReport | None,
    *,
    knowledge_fallback_text: str = "",
    qq_limit: int,
) -> RenderedReply:
    decision = result.decision
    disclosures: list[str] = []

    if decision.route is SearchTier.SKIP:
        if decision.skip_reason and decision.skip_reason.value == "user_forbid_web":
            disclosures.append(_NO_WEB_DYNAMIC_LIMIT)
        text = _strip_markers(knowledge_fallback_text)
        if disclosures:
            text = f"{'\n'.join(disclosures)}\n{text}".strip()
        chunks = split_qq_reply(text, qq_limit)
        return RenderedReply(
            text=text,
            chunks=tuple(chunks),
            used_evidence_ids=(),
            shown_source_urls=(),
            degradation_disclosures=tuple(disclosures),
        )

    failure = result.failure_code
    evidence = result.evidence

    # Deterministic failure rendering.
    if failure in _FIXED_DISCLOSURES and (evidence is None or failure in _URGENT_DYNAMIC_CODES):
        base = _FIXED_DISCLOSURES[failure]
        explicit_add = ""
        if _explicit_search_failed(result):
            explicit_add = _EXPLICIT_SEARCH_FAILED
        if failure is SearchFailureCode.PARTIAL_EVIDENCE:
            return _render_partial(result, validation, qq_limit)
        if failure is SearchFailureCode.SOURCE_CONFLICT:
            return _render_conflict(result, validation, qq_limit)
        if knowledge_fallback_text:
            base = _STABLE_FALLBACK_PREFIX
            disclosures.append(_STABLE_FALLBACK_PREFIX)
            text = f"{_STABLE_FALLBACK_PREFIX}\n{_strip_markers(knowledge_fallback_text)}"
        else:
            text = base
        if explicit_add:
            text = f"{explicit_add}\n{text}"
            disclosures.append(_EXPLICIT_SEARCH_FAILED)
        chunks = split_qq_reply(text, qq_limit)
        return RenderedReply(
            text=text,
            chunks=tuple(chunks),
            used_evidence_ids=(),
            shown_source_urls=(),
            degradation_disclosures=tuple(disclosures),
        )

    if validation is None or evidence is None:
        text = _strip_markers(knowledge_fallback_text or "")
        chunks = split_qq_reply(text, qq_limit)
        return RenderedReply(
            text=text,
            chunks=tuple(chunks),
            used_evidence_ids=(),
            shown_source_urls=(),
            degradation_disclosures=tuple(disclosures),
        )

    return _render_validated(result, validation, evidence, qq_limit)


def render_plain_reply(text: str, *, trace: SearchTrace, qq_limit: int) -> RenderedReply:
    del trace
    cleaned = _strip_markers(text)
    chunks = split_qq_reply(cleaned, qq_limit)
    return RenderedReply(
        text=cleaned,
        chunks=tuple(chunks),
        used_evidence_ids=(),
        shown_source_urls=(),
        degradation_disclosures=(),
    )


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

    numbered = {evidence_id: index for index, evidence_id in enumerate(used_order, 1)}

    body_parts: list[str] = []
    disclosures: list[str] = []
    if result.evidence is not None and result.evidence.evidence_state is EvidenceState.PARTIAL:
        disclosures.append(_PARTIAL_PREFIX)
    if result.evidence is not None and result.evidence.evidence_state is EvidenceState.CONFLICTING:
        disclosures.append(_CONFLICT_PREFIX)
    if _semantic_unavailable(validation):
        disclosures.append(_SEMANTIC_UNAVAILABLE)

    for block in validation.retained_blocks:
        # Collect evidence ids referenced by this block's claims.
        used_here: list[int] = []
        for claim in validation.retained_claims:
            if claim.block_id == block.block_id:
                for eid in claim.evidence_ids:
                    if eid in numbered and numbered[eid] not in used_here:
                        used_here.append(numbered[eid])
        citations = used_here
        text = _strip_markers(block.text)
        if citations:
            text = f"{text}[{''.join(str(n) for n in citations)}]"
        body_parts.append(text)

    if result.evidence is not None and result.evidence.evidence_state is EvidenceState.CONFLICTING:
        body_parts.append(_CONFLICT_PREFIX)

    source_lines: list[str] = []
    shown_urls: list[str] = []
    seen_urls: set[str] = set()
    # In a conflict, every recorded member must be shown even if the draft
    # omitted it; never select a winner.
    if result.evidence is not None and result.evidence.evidence_state is EvidenceState.CONFLICTING:
        for evidence_id in evidence_by_id:
            if evidence_id not in used_order:
                used_order.append(evidence_id)
                numbered[evidence_id] = len(used_order)
    for evidence_id in used_order:
        item = evidence_by_id[evidence_id]
        if not item.url or item.url in seen_urls:
            continue
        seen_urls.add(item.url)
        shown_urls.append(item.url)
        number = numbered[evidence_id]
        title = item.title or item.url
        source_lines.append(f"[{number}] {_bounded_title(title)}")
        source_lines.append(item.url)

    if used_order:
        source_lines.insert(0, "来源：")

    body = "\n".join(body_parts).strip()
    if source_lines:
        body = f"{body}\n\n{'\n'.join(source_lines)}"
    if disclosures and not body.startswith(tuple(d for d in disclosures if d)):
        body = f"{'\n'.join(disclosures)}\n\n{body}".strip()
    else:
        body = body.strip()

    chunks = split_qq_reply(body, qq_limit)
    return RenderedReply(
        text=body,
        chunks=tuple(chunks),
        used_evidence_ids=tuple(used_order),
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


def _bounded_title(title: str, limit: int = 80) -> str:
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    if len(title) <= limit:
        return title
    return title[:limit].rstrip() + "…"


def _strip_markers(text: str) -> str:
    text = str(text or "")
    # Discard everything from a model-written source heading onward; the
    # deterministic renderer owns the source list.
    text = _SOURCE_HEADING.sub("\n@@SOURCE_BOUNDARY@@", text)
    text = text.split("@@SOURCE_BOUNDARY@@", 1)[0]
    text = _SOURCE_MARKER.sub("", text)
    text = re.sub(r"\[(?:SRCH|MEM|CHAT):[^\]]*\]", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
