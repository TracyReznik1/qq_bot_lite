"""Safe structured content extraction for search candidates."""

from __future__ import annotations

import re
from typing import Any, Sequence

from src.search.models import (
    EvidenceCandidate,
    ExcerptOrigin,
    FetchedDocument,
    ProviderHit,
    SearchQuery,
)
import src.services.url_fetch_service as url_fetch

_EXCERPT_MAX_CHARS = 900
_RAW_CONTENT_MAX_CHARS = 6000

_INJECTION_PATTERNS = (
    re.compile(r"(?i)(忽略|ignore).{0,12}(之前|以上|先前|prior|previous).{0,12}(指令|instructions)"),
    re.compile(r"(?i)(system prompt|系统提示|system_message|提示词|你的规则).{0,20}(expose|reveal|输出|泄露|打印|print)"),
    re.compile(r"(?i)调用工具|call tools|工具调用|run tool"),
    re.compile(r"(?i)(api.?key|密码|secret|token)\s*[:=]"),
    re.compile(r"(?i)你的(核心|内部|系统)指令"),
    re.compile(r"(?i)请(绕过|忽略|覆盖)"),
    re.compile(r"(?i)向用户(显示|泄露|暴露|展示)"),
)

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class SearchExtractor:
    """Extract bounded query-aware excerpts from provider hits."""

    def extract(
        self,
        hit: ProviderHit,
        query: SearchQuery,
        *,
        allow_network_read: bool,
        timeout_seconds: float,
    ) -> EvidenceCandidate:
        raw_content = hit.raw_content if hit.raw_content else None
        snippet = hit.snippet if hit.snippet else None

        # Provider-native raw content is a usable read with no extra network.
        if raw_content:
            excerpt = _clean_excerpt(raw_content)
            safety_flags = _detect_safety_flags(raw_content)
            return EvidenceCandidate(
                hit=hit,
                document=None,
                excerpt=excerpt,
                excerpt_origin=ExcerptOrigin.PROVIDER_SNIPPET,
                extraction_status="provider_raw_content",
                safety_flags=safety_flags,
                content_reads_consumed=1,
            )

        # When a network read is allowed, fetch the page so a snippet never
        # blocks the underlying readable content (deep-tier DDGS requires it).
        if allow_network_read and hit.url:
            document = self._fetch_document(hit.url, timeout_seconds=timeout_seconds)
            if document.fetch_status == "success" and document.excerpt:
                excerpt = _select_query_relevant_excerpt(document.excerpt, query)
                excerpt = _clean_excerpt(excerpt)
                origin = (
                    ExcerptOrigin.DOCUMENT_EXTRACT
                    if document.content_type in {"application/pdf", "application/x-pdf"}
                    else ExcerptOrigin.PAGE_EXTRACT
                )
                status = "document_extract" if origin is ExcerptOrigin.DOCUMENT_EXTRACT else "page_extract"
                return EvidenceCandidate(
                    hit=hit,
                    document=document,
                    excerpt=excerpt,
                    excerpt_origin=origin,
                    extraction_status=status,
                    safety_flags=_detect_safety_flags(excerpt),
                    content_reads_consumed=1,
                )

        if snippet:
            excerpt = _clean_excerpt(snippet)
            safety_flags = _detect_safety_flags(snippet)
            return EvidenceCandidate(
                hit=hit,
                document=None,
                excerpt=excerpt,
                excerpt_origin=ExcerptOrigin.PROVIDER_SNIPPET,
                extraction_status="search_result_snippet",
                safety_flags=safety_flags,
                content_reads_consumed=0,
            )

        return EvidenceCandidate(
            hit=hit,
            document=None,
            excerpt=None,
            excerpt_origin=None,
            extraction_status="no_content",
            safety_flags=(),
            content_reads_consumed=0,
        )

    def _fetch_document(self, url: str, *, timeout_seconds: float) -> FetchedDocument:
        result = url_fetch.fetch_document(url, timeout_seconds=timeout_seconds)
        if not result.ok:
            return FetchedDocument(
                requested_url=url,
                final_url=result.final_url or None,
                content_type=result.content_type or None,
                title=None,
                excerpt=None,
                fetch_status=result.status,
                untrusted_content_flags=(),
            )
        return FetchedDocument(
            requested_url=url,
            final_url=result.final_url,
            content_type=result.content_type,
            title=result.title or None,
            excerpt=result.text[: _RAW_CONTENT_MAX_CHARS] if result.text else None,
            fetch_status="success",
            untrusted_content_flags=_detect_safety_flags(result.text or ""),
        )


def _clean_excerpt(text: str) -> str:
    text = str(text or "")
    text = _CONTROL_CHARACTERS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= _EXCERPT_MAX_CHARS:
        return text
    return text[:_EXCERPT_MAX_CHARS].rstrip("，,。；;：:") + "…"


def _detect_safety_flags(text: str) -> tuple[str, ...]:
    flags: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(str(text or "")):
            flags.append("prompt_injection")
            break
    return tuple(dict.fromkeys(flags))


def _select_query_relevant_excerpt(text: str, query: SearchQuery) -> str:
    text = str(text or "")
    terms = _query_terms(query.text)
    if not terms:
        return text
    sentences = re.split(r"(?<=[。！？；!?；])", text)
    best = ""
    best_score = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        score = sum(1 for term in terms if term in sentence)
        if score > best_score:
            best_score = score
            best = sentence
    if best_score == 0:
        return text
    return best


def _query_terms(text: str) -> tuple[str, ...]:
    lowered = str(text or "").casefold()
    terms: list[str] = []
    for candidate in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{1,30}|[一-鿿]{2,8}", lowered):
        terms.append(candidate)
    stop = {"什么是", "什么", "怎么", "如何", "为什么", "区别", "最近", "最新", "今天"}
    return tuple(dict.fromkeys(term for term in terms if term not in stop))[:20]
