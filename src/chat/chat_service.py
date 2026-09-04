from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

from src.chat.prompt import (
    _ensure_context,
    build_search_system_prompt,
    build_untrusted_context,
)
from src.config import config
from src.memory.models import MemoryContext
from src.search.simple.answering import SearchAnswerer
from src.search.simple.factory import (
    get_simple_search_pipeline,
    reset_simple_search_pipeline,
)
from src.search.simple.planning import IMAGE_ONLY_FALLBACK_QUERY
from src.search.simple.models import (
    RequestSource,
    SearchMode,
    SearchRequest,
)

from src.search.simple.rendering import (
    render_search_answer,
    render_search_failure,
)
from src.services.llm_client import get_llm_client
from src.services.llm_types import ChatResponse
from src.utils.storage import read_json, safe_id, write_json

logger = logging.getLogger("qq-bot")

llm = get_llm_client()
chat_history: dict[str, list[dict[str, str]]] = {}
chat_history_lock = Lock()

_simple_search_pipeline = None


def get_simple_search_pipeline_for_chat():
    global _simple_search_pipeline
    if _simple_search_pipeline is None:
        _simple_search_pipeline = get_simple_search_pipeline()
    return _simple_search_pipeline


def reset_chat_search_pipeline() -> None:
    global _simple_search_pipeline
    _simple_search_pipeline = None
    reset_simple_search_pipeline()


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


def _history_path(session_key: str) -> Path:
    return config.data_dir / "history" / f"{safe_id(session_key)}.json"


def _load_history_unlocked(session_key: str) -> list[dict[str, str]]:
    if not config.persist_history:
        return []
    data = read_json(_history_path(session_key), {"messages": []})
    messages = data.get("messages", []) if isinstance(data, dict) else []
    limit = max(config.history_turns, 1) * 2
    return [
        msg
        for msg in messages[-limit:]
        if isinstance(msg, dict) and "role" in msg and "content" in msg
    ]


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


def _qq_limit() -> int:
    """Return the shared QQ limit without coupling tests to unrelated config."""
    return max(int(getattr(config, "max_reply_chars", 1700)), 200)


def _build_base_messages(
    mem_ctx: MemoryContext,
    text: str,
    images: list[str],
) -> list[dict[str, Any]]:
    _ensure_history_loaded(mem_ctx.session_key)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_search_system_prompt(mem_ctx)},
        {
            "role": "user",
            "content": build_untrusted_context(
                mem_ctx,
                query=text,
                evidence_payload="",
                include_memories=True,
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


def _plain_reply(
    mem_ctx: MemoryContext,
    text: str,
    images: list[str],
    *,
    timeout_seconds: float,
) -> str:
    from src.chat.prompt import build_system_prompt

    _ensure_history_loaded(mem_ctx.session_key)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(mem_ctx)},
        {
            "role": "user",
            "content": build_untrusted_context(
                mem_ctx,
                query=text,
                evidence_payload="",
                include_memories=True,
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
    response = llm.chat(messages, temperature=0.7, timeout_seconds=timeout_seconds)
    content = getattr(response, "content", "")
    return str(content or "").strip()


def generate_reply(
    context: MemoryContext | str,
    text: str,
    image_data_urls: list[str] | None = None,
    *,
    mode: SearchMode,
    history_text: str | None = None,
) -> str:
    mem_ctx = _ensure_context(context)
    session_key = mem_ctx.session_key
    normalized_text = " ".join(str(text or "").split())
    images = [img for img in (image_data_urls or []) if str(img or "").strip()]

    reply: str | None = None
    try:
        try:
            timeout = float(getattr(config, "search_answer_timeout", 20.0))
            if mode is SearchMode.SKIP:
                reply = _plain_reply(
                    mem_ctx,
                    normalized_text,
                    images,
                    timeout_seconds=timeout,
                )
                return reply

            request = SearchRequest(
                mode=mode,
                text=normalized_text,
                images=tuple(images),
                source=RequestSource.CHAT if mode is SearchMode.LIGHT else RequestSource.COMMAND,
            )
            pipeline = get_simple_search_pipeline_for_chat()
            outcome = pipeline.run(request)

            if outcome.failure is not None:
                rendered = render_search_failure(
                    outcome.failure,
                    qq_limit=_qq_limit(),
                    trace=outcome.trace,
                )
                reply = rendered.text
                return reply

            base_messages = _build_base_messages(mem_ctx, normalized_text, images)
            answerer = SearchAnswerer(llm)
            answer_res = answerer.answer(
                question=normalized_text or IMAGE_ONLY_FALLBACK_QUERY,
                results=outcome.results,
                base_messages=base_messages,
                timeout_seconds=timeout,
                context=mem_ctx,
            )
            outcome.trace.answer_degraded = answer_res.degraded
            warning = outcome.warning
            if answer_res.degraded and not warning:
                warning = "信息可能不完整。"

            rendered = render_search_answer(
                answer_res.text,
                outcome.results,
                warning=warning,
                show_sources=(mode is SearchMode.STANDARD),
                qq_limit=_qq_limit(),
                trace=outcome.trace,
            )
            reply = rendered.text
            return reply
        except Exception as exc:
            logger.warning("generate_reply failed (%s)", type(exc).__name__)
            reply = "在线搜索暂时不可用，请稍后再试。"
            return reply
    finally:
        if reply is not None:
            stored_user_text = (
                history_text
                if history_text is not None
                else history_user_text(text, len(images))
            )
            append_history(session_key, stored_user_text, reply)
