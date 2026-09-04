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
    format_external_webpage_sandbox,
)
from src.services.url_fetch_service import extract_first_url, fetch_document
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
from src.search.simple.router import SearchRouter

from src.search.simple.rendering import (
    render_search_answer,
    render_search_failure,
)
from src.services.llm_client import ImageRecognitionUnavailable, get_llm_client
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


def get_recent_dialogue_context(
    session_key: str,
    turns: int = 1,
) -> tuple[tuple[str, str], ...]:
    """Return the most recent dialogue turns as a tuple of (role, content) pairs.

    Each turn is typically a user message and assistant reply (up to 2 messages per turn).
    Safe to call concurrently.
    """
    _ensure_history_loaded(session_key)
    with chat_history_lock:
        messages = chat_history.get(session_key, [])
        if not messages:
            return ()
        message_count = max(int(turns), 1) * 2
        slice_messages = messages[-message_count:]
        result: list[tuple[str, str]] = []
        for msg in slice_messages:
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", "")).strip()
            if role and content:
                result.append((role, content))
        return tuple(result)


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
    webpage_payload: str = "",
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
                webpage_payload=webpage_payload,
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
    try:
        response = llm.chat(messages, temperature=0.7, timeout_seconds=timeout_seconds)
        content = getattr(response, "content", "")
        text_reply = str(content or "").strip()
        return text_reply or "暂时无法回复，请稍后再试。"
    except ImageRecognitionUnavailable as err:
        return str(err)
    except Exception as exc:
        logger.warning("_plain_reply failed (%s)", type(exc).__name__)
        return "暂时无法回复，请稍后再试。"


def generate_reply(
    context: MemoryContext | str,
    text: str,
    image_data_urls: list[str] | None = None,
    *,
    mode: SearchMode | None = None,
    history_text: str | None = None,
    router: SearchRouter | None = None,
) -> str:
    mem_ctx = _ensure_context(context)
    session_key = mem_ctx.session_key
    normalized_text = " ".join(str(text or "").split())
    images = [img for img in (image_data_urls or []) if str(img or "").strip()]

    reply: str | None = None
    try:
        try:
            timeout = float(getattr(config, "search_answer_timeout", 20.0))

            # ── 聊天 URL 前置自动直读与注入 ──
            url = extract_first_url(normalized_text)
            webpage_payload = ""
            if url:
                try:
                    doc = fetch_document(url, timeout_seconds=5.0)
                    if doc.ok:
                        webpage_payload = format_external_webpage_sandbox(
                            url=doc.final_url or url,
                            title=doc.title,
                            text=doc.text,
                        )
                        logger.info(
                            "URL direct fetch succeeded url=%s title=%s chars=%s",
                            url,
                            doc.title,
                            len(doc.text),
                        )
                    else:
                        logger.info("URL direct fetch degraded url=%s status=%s", url, doc.status)
                except Exception as fetch_err:
                    logger.warning(
                        "URL direct fetch unexpected error url=%s err=%s",
                        url,
                        type(fetch_err).__name__,
                    )

            if webpage_payload:
                # 抓取到完整网页正文后，直接基于正文回答，短路冗余网络搜索
                reply = _plain_reply(
                    mem_ctx,
                    normalized_text,
                    images,
                    timeout_seconds=timeout,
                    webpage_payload=webpage_payload,
                )
                return reply

            # ── 模式裁决与检索主题 ──
            active_topics: tuple[str, ...] = ()
            effective_mode: SearchMode = mode if mode is not None else SearchMode.SKIP
            if mode is None:
                # 无命令普通消息：由 SearchRouter 依据检索收益裁决 (仅在 skip 与 light 间选择)
                search_router = router if router is not None else SearchRouter()
                decision = search_router.route(normalized_text, tuple(images))
                logger.info(
                    "Search router decision mode=%s reason=%s topics=%s",
                    decision.mode.value,
                    decision.reason_code,
                    decision.retrieval_topics,
                )
                if decision.mode is SearchMode.SKIP:
                    reply = _plain_reply(
                        mem_ctx,
                        normalized_text,
                        images,
                        timeout_seconds=timeout,
                    )
                    return reply
                effective_mode = SearchMode.LIGHT
                active_topics = decision.retrieval_topics
            else:
                effective_mode = mode

            if effective_mode is SearchMode.SKIP:
                reply = _plain_reply(
                    mem_ctx,
                    normalized_text,
                    images,
                    timeout_seconds=timeout,
                )
                return reply

            request = SearchRequest(
                mode=effective_mode,
                text=normalized_text,
                images=tuple(images),
                source=RequestSource.CHAT if effective_mode is SearchMode.LIGHT else RequestSource.COMMAND,
                topics=active_topics,
            )
            pipeline = get_simple_search_pipeline_for_chat()
            outcome = pipeline.run(request)

            if outcome.failure is not None:
                if effective_mode is SearchMode.LIGHT:
                    logger.info(
                        "Light search yielded no usable results (%s), falling back to plain conversation",
                        outcome.failure.value,
                    )
                    reply = _plain_reply(
                        mem_ctx,
                        normalized_text,
                        images,
                        timeout_seconds=timeout,
                    )
                    return reply

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
            rendered = render_search_answer(
                answer_res.text,
                outcome.results,
                warning=None,
                show_sources=(effective_mode is SearchMode.STANDARD),
                qq_limit=_qq_limit(),
                trace=outcome.trace,
            )
            reply = rendered.text
            return reply
        except Exception as exc:
            logger.warning("generate_reply failed (%s)", type(exc).__name__)
            if effective_mode is SearchMode.SKIP:
                reply = "暂时无法回复，请稍后再试。"
            else:
                reply = "在线搜索暂时不可用，请稍后再试。"
            return reply
    finally:
        if reply is not None:
            from src.memory.privacy import redact_hard_secrets

            reply = redact_hard_secrets(reply)
            stored_user_text = (
                history_text
                if history_text is not None
                else history_user_text(text, len(images))
            )
            append_history(session_key, stored_user_text, reply)
