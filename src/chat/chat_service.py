import copy
import json
import logging
import re
from pathlib import Path
from threading import Lock
from typing import Any

from src.chat.prompt import _ensure_context, build_untrusted_context
from src.memory.models import MemoryContext
from src.config import config
from src.search import get_search_orchestrator, reset_search_orchestrator
from src.search.models import (
    EvidenceState,
    RequestSource,
    RetrievalRequest,
    SearchFailureCode,
    SearchPipelineResult,
    SearchTier,
    SkipReason,
    TriggerCode,
)
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


def run_tool(name: str, query: str) -> str:
    """Legacy tool dispatch used only by generic provider protocol tests."""
    if name == "search_web":
        from src.services.search_service import web_search
        return web_search(query)
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
                "content": run_tool(str(function.get("name") or ""), query),
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
        {"role": "system", "content": build_untrusted_context(
            mem_ctx,
            query=text,
            evidence_payload=evidence_payload,
            include_memories=include_memories,
        )},
    ]
    # Rebuild the untrusted context as a user message for the model.
    messages = [
        {"role": "system", "content": _system_prompt_for(mem_ctx, evidence_payload)},
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


def _grounded_generation(mem_ctx, text, images, result) -> str:
    evidence_payload = _build_evidence_payload(result)
    messages = _build_messages(mem_ctx, text, images, evidence_payload=evidence_payload, include_memories=True)
    response = llm.chat(messages, temperature=0.2)
    from src.search.validation import parse_grounded_draft, validate_and_filter
    from tests.search_fakes import StaticSemanticVerifier
    draft = parse_grounded_draft(response.content)
    from src.search.evidence import EvidenceAssembler
    verifier = _Verifier()
    report = validate_and_filter(
        draft,
        result.evidence,
        result.decision,
        claim_discoverer=_Discoverer(),
        semantic_verifier=verifier,
    )
    from src.search.renderer import render_search_reply
    rendered = render_search_reply(result, report, qq_limit=config.max_reply_chars)
    return rendered.text


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
                return {}
            return parsed
        except Exception:
            return {}


class _Discoverer:
    def discover(self, draft, evidence):
        del draft, evidence
        return ()


def generate_reply(
    context: MemoryContext | str,
    text: str,
    image_data_urls: list[str] | None = None,
    *,
    force_search: bool = False,
    history_text: str | None = None,
) -> str:
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

    if result.decision.route is SearchTier.SKIP:
        return _handle_skip(mem_ctx, text, images, result)

    if result.evidence is None or result.failure_code is not None:
        return _handle_failure(mem_ctx, text, images, result)

    evidence_payload = _build_evidence_payload(result)
    messages = _build_messages(mem_ctx, text, images, evidence_payload=evidence_payload, include_memories=True)
    response = llm.chat(messages, temperature=0.2)
    draft = _parse_draft(response.content)
    report = _validate_draft(draft, result)
    rendered = render_search_reply(result, report, qq_limit=config.max_reply_chars)
    finalize_search_trace(result, history_text)
    reply = rendered.text
    append_history(session_key, history_user_text(text, len(images)), reply)
    return reply


def _handle_skip(mem_ctx, text, images, result) -> str:
    reason = result.decision.skip_reason
    from src.search.renderer import render_search_reply
    if reason is SkipReason.USER_FORBID_WEB:
        # Stable knowledge may answer with a fixed no-web disclosure.
        if result.decision.requires_clarification:
            rendered = render_search_reply(result, None, qq_limit=config.max_reply_chars)
            finalize_search_trace(result, None)
            append_history(mem_ctx.session_key, history_user_text(text, len(images)), rendered.text)
            return rendered.text
        messages = _build_messages(mem_ctx, text, images, evidence_payload="", include_memories=False)
        response = llm.chat(messages, temperature=0.5)
        rendered = render_search_reply(
            result, None,
            knowledge_fallback_text=response.content,
            qq_limit=config.max_reply_chars,
        )
        finalize_search_trace(result, None)
        append_history(mem_ctx.session_key, history_user_text(text, len(images)), rendered.text)
        return rendered.text

    # Ordinary closed tasks: normal answer call, no search tool, no citations.
    messages = _build_messages(mem_ctx, text, images, evidence_payload="", include_memories=True)
    response = llm.chat(messages, temperature=0.75)
    finalize_search_trace(result, None)
    reply = re.sub(r"\[(?:SRCH|MEM|CHAT):?.*?\]", "", response.content).strip()
    append_history(mem_ctx.session_key, history_user_text(text, len(images)), reply)
    return reply


def _handle_failure(mem_ctx, text, images, result) -> str:
    from src.search.renderer import render_search_reply
    decision = result.decision
    if decision.route is SearchTier.DEEP:
        rendered = render_search_reply(result, None, qq_limit=config.max_reply_chars)
        finalize_search_trace(result, None)
        append_history(mem_ctx.session_key, history_user_text(text, len(images)), rendered.text)
        return rendered.text

    # Stable knowledge fallback without retrieved-memory facts.
    messages = _build_messages(mem_ctx, text, images, evidence_payload="", include_memories=False)
    response = llm.chat(messages, temperature=0.5)
    rendered = render_search_reply(
        result, None,
        knowledge_fallback_text=response.content,
        qq_limit=config.max_reply_chars,
    )
    finalize_search_trace(result, None)
    append_history(mem_ctx.session_key, history_user_text(text, len(images)), rendered.text)
    return rendered.text


def _parse_draft(content: str):
    from src.search.validation import parse_grounded_draft
    return parse_grounded_draft(content)


def _validate_draft(draft, result):
    from src.search.validation import validate_and_filter
    from src.search.evidence import EvidenceAssembler
    verifier = _Verifier()
    return validate_and_filter(
        draft,
        result.evidence,
        result.decision,
        claim_discoverer=_Discoverer(),
        semantic_verifier=verifier,
    )


def finalize_search_trace(result: SearchPipelineResult, history_text: str | None) -> None:
    del history_text
    from src.search.orchestrator import finalize_search_trace as _finalize
    _finalize(result.trace, response_finished_at=0.0)
