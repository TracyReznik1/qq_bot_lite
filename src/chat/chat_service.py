import copy
import json
import logging
import re
import time
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any

from src.chat.prompt import _ensure_context, build_untrusted_context
from src.memory.models import MemoryContext
from src.config import config
from src.search import get_search_orchestrator, reset_search_orchestrator
from src.search.models import (
    AllowedClaimScope,
    AnswerCertainty,
    AnswerGenerationMode,
    AnswerState,
    DisclosureCode,
    EvidenceState,
    RequestSource,
    RetrievalRequest,
    SearchFailureCode,
    SearchPipelineResult,
    SearchTier,
    SkipReason,
    ValidatorRequirement,
    WarningCode,
)
from src.search.policy import build_render_state, decide_answer_state
from src.search.renderer import render_search_reply, render_plain_reply
from src.services.llm_client import get_llm_client
from src.services.llm_types import ChatResponse
from src.utils.storage import read_json, safe_id, write_json


logger = logging.getLogger("qq-bot")

llm = get_llm_client()
chat_history: dict[str, list[dict[str, str]]] = {}
chat_history_lock = Lock()

_search_orchestrator = None


# ── generic provider tool-protocol helpers ─────────────────────────────
# These remain so provider clients keep tool support for other callers;
# ordinary chat no longer supplies a search tool.

def normalize_chat_response(response: ChatResponse | str) -> ChatResponse:
    if isinstance(response, ChatResponse):
        return response
    return ChatResponse(content=str(response or ""))


def _tool_result(name: str, query: str) -> str:
    """Generic provider-protocol tool result. Ordinary chat never supplies tools."""
    del query
    if name == "search_web":
        return "搜索已完成（兼容工具占位）。"
    return ""


def build_tool_messages(
    tool_calls: list[dict[str, Any]],
    fallback_query: str,
    provider_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build assistant + tool result messages for a provider tool round."""
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }
    if provider_context is not None:
        assistant_message["_provider_context"] = copy.deepcopy(provider_context)
    messages: list[dict[str, Any]] = [assistant_message]
    for index, tool_call in enumerate(tool_calls, 1):
        function = tool_call.get("function") if isinstance(tool_call, dict) else {}
        arguments = function.get("arguments") if isinstance(function, dict) else "{}"
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        query = str(args.get("query") or args.get("url") or fallback_query).strip()
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(tool_call.get("id") or f"tool_{index}"),
                "name": str(function.get("name") or ""),
                "content": _tool_result(str(function.get("name") or ""), query),
            }
        )
    return messages


def get_search_orchestrator_for_chat():
    global _search_orchestrator
    if _search_orchestrator is None:
        _search_orchestrator = get_search_orchestrator()
    return _search_orchestrator


def reset_chat_search_orchestrator() -> None:
    global _search_orchestrator
    _search_orchestrator = None
    reset_search_orchestrator()


def _history_path(session_key: str) -> Path:
    return config.data_dir / "history" / f"{safe_id(session_key)}.json"


def _load_history_unlocked(session_key: str) -> list[dict[str, str]]:
    if not config.persist_history:
        return []
    data = read_json(_history_path(session_key), {"messages": []})
    messages = data.get("messages", []) if isinstance(data, dict) else []
    limit = max(config.history_turns, 1) * 2
    return [msg for msg in messages[-limit:] if isinstance(msg, dict) and "role" in msg and "content" in msg]


def _save_history_unlocked(session_key: str, history: list[dict[str, str]]) -> None:
    if not config.persist_history:
        return
    write_json(_history_path(session_key), {"messages": history})


def _remove_history_file(session_key: str) -> None:
    path = _history_path(session_key)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.debug("Failed to remove history file: %s", path)


def append_history(session_key: str, user_text: str, assistant_text: str) -> None:
    with chat_history_lock:
        history = chat_history.setdefault(session_key, [])
        history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        limit = max(config.history_turns, 1) * 2
        history[:] = history[-limit:]
        _save_history_unlocked(session_key, history)


def reset_history(session_key: str) -> None:
    with chat_history_lock:
        chat_history.pop(session_key, None)
    _remove_history_file(session_key)


def _ensure_history_loaded(session_key: str) -> None:
    with chat_history_lock:
        history = chat_history.setdefault(session_key, [])
        if not history and config.persist_history:
            loaded = _load_history_unlocked(session_key)
            history.extend(loaded)


def build_user_content(text: str, image_data_urls: list[str]):
    text = str(text or "").strip()
    if not image_data_urls:
        return text
    content: list[dict[str, Any]] = [
        {"type": "text", "text": text or "请识别图片内容并回答。"}
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_data_url}}
        for image_data_url in image_data_urls
    )
    return content


def history_user_text(text: str, image_count: int) -> str:
    parts = ["[图片]"] * max(image_count, 0)
    text = str(text or "").strip()
    if text:
        parts.append(text)
    return "\n".join(parts)


def _build_evidence_payload(result: SearchPipelineResult) -> str:
    evidence = result.evidence
    if evidence is None:
        return ""
    rows = []
    for item in evidence.evidence_items:
        rows.append(
            {
                "evidence_id": item.evidence_id,
                "title": item.title,
                "url": item.url,
                "excerpt": (item.excerpt or "")[:400],
                "published_at": item.published_at.isoformat() if item.published_at else None,
                "source_relation": item.source_relation.value,
                "supported_topics": list(item.supported_topics),
            }
        )
    payload = {
        "evidence_items": rows,
        "conflict_groups": list(evidence.conflict_groups),
        "conflicts": [
            {
                "conflict_id": conflict.conflict_id,
                "conflict_key": conflict.conflict_key,
                "members": [
                    {
                        "evidence_id": member.evidence_id,
                        "value": member.value,
                        "published_at": member.published_at.isoformat() if member.published_at else None,
                        "relation": member.relation,
                    }
                    for member in conflict.members
                ],
            }
            for conflict in evidence.conflicts
        ],
        "missing_claim_topics": list(evidence.missing_claim_topics),
        "evidence_state": evidence.evidence_state.value,
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_messages(
    mem_ctx: MemoryContext,
    text: str,
    images: list[str],
    *,
    evidence_payload: str,
    include_memories: bool,
) -> list[dict[str, Any]]:
    _ensure_history_loaded(mem_ctx.session_key)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt_for(mem_ctx, evidence_payload)},
        {
            "role": "user",
            "content": build_untrusted_context(
                mem_ctx,
                query=text,
                evidence_payload=evidence_payload,
                include_memories=include_memories,
            ),
        },
    ]
    with chat_history_lock:
        messages.extend(chat_history.get(mem_ctx.session_key, []).copy())
    messages.append(
        {
            "role": "user",
            "content": build_user_content(text, images),
        }
    )
    return messages


def _system_prompt_for(mem_ctx: MemoryContext, evidence_payload: str) -> str:
    from src.chat.prompt import build_system_prompt
    return build_system_prompt(mem_ctx, evidence_payload=evidence_payload)


def _generate_answer(trace, messages, *, temperature: float) -> ChatResponse:
    """Time the answer-model stage even when the provider raises."""
    answer_started = time.monotonic()
    try:
        return llm.chat(messages, temperature=temperature)
    finally:
        trace.answer_generation_latency_ms += max(
            (time.monotonic() - answer_started) * 1000.0,
            0.0,
        )


def _grounded_generation(
    mem_ctx, text, images, result, answer_state
) -> tuple[str, SearchPipelineResult]:
    evidence_payload = _build_evidence_payload(result)
    messages = _build_messages(mem_ctx, text, images, evidence_payload=evidence_payload, include_memories=True)
    response = _generate_answer(result.trace, messages, temperature=0.2)
    structural_started = time.monotonic()
    try:
        draft = _parse_draft(response.content)
    except ValueError:
        result.trace.structural_validation_latency_ms += max(
            (time.monotonic() - structural_started) * 1000.0,
            0.0,
        )
        return _handle_draft_failure(result, answer_state)
    result.trace.structural_validation_latency_ms += max(
        (time.monotonic() - structural_started) * 1000.0,
        0.0,
    )
    try:
        report = _validate_draft(draft, result, answer_state)
    except Exception:
        logger.debug("grounded draft validation failed", exc_info=True)
        return _handle_draft_failure(result, answer_state)
    if report is None:
        return _handle_draft_failure(result, answer_state)
    render_state = build_render_state(answer_state, report, result.evidence)
    rendered = _render_view(render_state, result)
    return rendered.text, result


def _handle_draft_failure(
    result: SearchPipelineResult,
    answer_state: AnswerState,
) -> tuple[str, SearchPipelineResult]:
    """A malformed draft cannot produce a definite grounded answer."""
    result.trace.degradation_reason = SearchFailureCode.VALIDATION_FAILED
    failed_result = replace(result, failure_code=SearchFailureCode.VALIDATION_FAILED)
    disclosures = [DisclosureCode.VALIDATION_FAILED]
    if (
        result.evidence is not None
        and result.evidence.evidence_state is EvidenceState.CONFLICTING
    ):
        disclosures.append(DisclosureCode.SOURCE_CONFLICT)
    failed_state = AnswerState(
        None,
        AnswerGenerationMode.FIXED,
        AnswerCertainty.UNVERIFIED,
        AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS,
        tuple(disclosures),
        answer_state.warning_codes,
        answer_state.validator_requirement,
    )
    render_state = build_render_state(failed_state, None, failed_result.evidence)
    rendered = _render_view(render_state, failed_result)
    return rendered.text, failed_result


class SemanticVerificationUnavailable(RuntimeError):
    """Raised when the semantic verifier cannot run (e.g. provider failure)."""


class _Verifier:
    """Small semantic verifier wrapper over the LLM chain."""

    def verify(self, payload):
        prompt = (
            "Judge each claim against the provided evidence excerpts. "
            "Return a JSON object mapping claim_id to one of: supported, partial, conflict, unsupported."
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            response = llm.chat(messages, temperature=0.0)
            parsed = json.loads(response.content)
            if not isinstance(parsed, dict):
                raise SemanticVerificationUnavailable("verifier returned non-object")
            return parsed
        except SemanticVerificationUnavailable:
            raise
        except Exception as exc:
            raise SemanticVerificationUnavailable(str(exc)) from exc


def generate_reply(
    context: MemoryContext | str,
    text: str,
    image_data_urls: list[str] | None = None,
    *,
    force_search: bool = False,
    history_text: str | None = None,
) -> str:
    response_started = time.monotonic()
    mem_ctx = _ensure_context(context)
    session_key = mem_ctx.session_key
    images = list(image_data_urls or [])
    request_source = RequestSource.COMMAND if force_search else RequestSource.CHAT

    request = RetrievalRequest(
        text,
        force_search=force_search,
        has_images=bool(images),
        request_source=request_source,
    )
    result = get_search_orchestrator_for_chat().run(request)
    result.trace.response_started_at = response_started
    final_result = result
    reply: str | None = None
    try:
        answer_state = decide_answer_state(
            result.analysis,
            result.evidence,
            result.failure_code,
        )
        if answer_state.generation_mode is AnswerGenerationMode.PLAIN:
            reply, final_result = _handle_plain(mem_ctx, text, images, result)
        elif answer_state.generation_mode is AnswerGenerationMode.GROUNDED:
            reply, final_result = _grounded_generation(
                mem_ctx, text, images, result, answer_state
            )
        else:
            reply, final_result = _handle_fixed(
                mem_ctx, text, images, result, answer_state
            )
        return reply
    finally:
        finalize_search_trace(final_result, history_text)
        if reply is not None:
            stored_user_text = (
                history_text
                if history_text is not None
                else history_user_text(text, len(images))
            )
            append_history(session_key, stored_user_text, reply)


def _handle_plain(mem_ctx, text, images, result) -> tuple[str, SearchPipelineResult]:
    # Ordinary closed tasks: normal answer call, no search tool, no citations.
    messages = _build_messages(mem_ctx, text, images, evidence_payload="", include_memories=True)
    response = _generate_answer(result.trace, messages, temperature=0.75)
    rendered = render_plain_reply(
        response.content,
        trace=result.trace,
        qq_limit=_qq_limit(),
    )
    return rendered.text, result


def _handle_fixed(
    mem_ctx,
    text,
    images,
    result,
    answer_state,
) -> tuple[str, SearchPipelineResult]:
    # Fixed output (no-web limitation or external-fact failure): never invoke the
    # ordinary answer model to invent facts.
    del mem_ctx, text, images
    render_state = build_render_state(answer_state, None, result.evidence)
    rendered = _render_view(render_state, result)
    return rendered.text, result


def _render_view(
    render_state,
    result: SearchPipelineResult,
) -> "RenderedReply":
    started = time.monotonic()
    rendered = render_search_reply(render_state, qq_limit=_qq_limit())
    result.trace.qq_render_latency_ms = max(
        (time.monotonic() - started) * 1000.0,
        0.0,
    )
    result.trace.citation_count = len(rendered.shown_source_urls)
    return rendered


def _fixed_answer_state(
    answer_state: AnswerState,
    disclosure: DisclosureCode,
) -> AnswerState:
    return AnswerState(
        None,
        AnswerGenerationMode.FIXED,
        AnswerCertainty.UNVERIFIED,
        AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS,
        (disclosure,),
        answer_state.warning_codes,
        answer_state.validator_requirement,
    )


def _parse_draft(content: str):
    from src.search.validation import parse_grounded_draft
    return parse_grounded_draft(content)


def _validate_draft(draft, result, answer_state):
    from src.search.validation import LLMClaimDiscoverer, validate_and_filter
    verifier = _Verifier()
    return validate_and_filter(
        draft,
        result.evidence,
        answer_state,
        claim_discoverer=LLMClaimDiscoverer(llm),
        semantic_verifier=verifier,
        trace=result.trace,
    )


def finalize_search_trace(result: SearchPipelineResult, history_text: str | None) -> None:
    del history_text
    from src.search.orchestrator import finalize_search_trace as _finalize
    _finalize(result.trace, response_finished_at=time.monotonic())


def _qq_limit() -> int:
    """Return the shared QQ limit without coupling tests to unrelated config."""
    return max(int(getattr(config, "max_reply_chars", 1700)), 200)
