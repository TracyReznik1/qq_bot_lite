import copy
import json
import logging
import re
from pathlib import Path
from threading import Lock
from typing import Any

from src.chat.prompt import build_system_prompt, build_untrusted_context
from src.services.search_service import web_search as search_web
from src.services.search_service import normalize_search_query
from src.services.search_service import search_query_specificity_score
from src.config import config
from src.services.llm_client import get_llm_client
from src.services.llm_types import ChatResponse
from src.utils.storage import read_json, safe_id, write_json


logger = logging.getLogger("qq-bot")

llm = get_llm_client()
chat_history: dict[str, list[dict[str, str]]] = {}
chat_history_lock = Lock()
HISTORY_DIR = config.data_dir / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
MAX_TOOL_CALL_ROUNDS = 2
TOOL_CALL_LIMIT_FALLBACK = "我搜到了信息，但没能整理出可靠回答。可以换个问法再试一次。"

SEARCH_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "搜索网页，并按来源优先级自动安全读取部分结果页正文摘录，由模型自行判断是否调用。闲聊、情绪回应、角色语气、记忆已能回答的问题不要搜索；"
            "遇到最新/实时信息、不懂、不确定、新梗、黑话、缩写、圈内 ID、人名、公开项目、产品、版本或当前事件等必须搜索。"
            "用户问最近为什么火、趋势、原因、评价或舆论变化时，即使主题看似熟悉也要搜索。"
            "搜索结果只能作为参考，最终回答必须由模型加工。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "搜索关键词，从用户最新消息中提取。"
                        "规则：提取核心名词、专有名词、事件名、作品名、人名、梗/黑话，用空格分隔；"
                        "保留限定词，例如平台、作者、版本、时间、作品名、产品名、原因、趋势、评价、舆论；不要只给单个泛词；"
                        "去掉语气词、追问、闲聊成分和已在前文解释过的上下文；"
                        "不要用完整问句，不要带\"怎么\"、\"什么\"、\"为什么\"。"
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

SUPPORTED_TOOL_NAMES = {
    "search_web",
}


def chat_tools_for_text(text: str) -> list[dict[str, Any]]:
    """qqbot_lite: ordinary chat only exposes search_web."""
    return [SEARCH_WEB_TOOL]


def tool_function_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "")


def filter_tool_calls(tool_calls: list[dict[str, Any]], allowed_names: set[str]) -> list[dict[str, Any]]:
    supported_calls = []
    for tool_call in tool_calls:
        name = tool_function_name(tool_call)
        if name in allowed_names and name in SUPPORTED_TOOL_NAMES:
            supported_calls.append(tool_call)
    return supported_calls


def run_tool(name: str, query: str) -> str:
    """qqbot_lite: only search_web is supported in ordinary chat."""
    return search_web(query)


def normalize_chat_response(response: ChatResponse | str) -> ChatResponse:
    if isinstance(response, ChatResponse):
        return response
    return ChatResponse(content=str(response or ""))


def tool_call_query(tool_call: dict[str, Any], fallback: str) -> str:
    function = tool_call.get("function") if isinstance(tool_call, dict) else {}
    arguments = function.get("arguments") if isinstance(function, dict) else "{}"
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return str(args.get("query") or args.get("url") or fallback).strip()


SEARCH_QUERY_INTENT_MARKERS = (
    "为什么",
    "原因",
    "趋势",
    "评价",
    "舆论",
    "最近",
    "最新",
    "当前",
    "发布",
    "更新",
    "火",
    "爆火",
    "走红",
)


def _compact_query_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").casefold())


def _fallback_has_more_search_intent(normalized: str, fallback: str, fallback_normalized: str) -> bool:
    normalized_compact = _compact_query_text(normalized)
    fallback_compact = _compact_query_text(fallback_normalized)
    if not normalized_compact or not fallback_compact:
        return False
    if normalized_compact not in fallback_compact:
        return False
    if len(fallback_compact) <= len(normalized_compact) + 1:
        return False
    marker_text = f"{fallback} {fallback_normalized}"
    return any(marker in marker_text for marker in SEARCH_QUERY_INTENT_MARKERS)


def normalize_tool_query(name: str, query: str, fallback: str) -> str:
    """qqbot_lite: only search_web normalization is needed."""
    if name != "search_web":
        return str(query or fallback).strip()
    normalized = normalize_search_query(query)
    fallback_normalized = normalize_search_query(fallback)
    if normalized:
        if (
            fallback_normalized
            and search_query_specificity_score(fallback_normalized) > search_query_specificity_score(normalized)
        ):
            return fallback_normalized
        if fallback_normalized and _fallback_has_more_search_intent(normalized, fallback, fallback_normalized):
            return fallback_normalized
        return normalized
    return fallback_normalized or str(query or fallback).strip()


def build_tool_messages(
    tool_calls: list[dict[str, Any]],
    fallback_query: str,
    provider_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }
    if provider_context is not None:
        assistant_message["_provider_context"] = copy.deepcopy(
            provider_context
        )

    messages: list[dict[str, Any]] = [assistant_message]
    for index, tool_call in enumerate(tool_calls, 1):
        name = tool_function_name(tool_call)
        query = normalize_tool_query(
            name,
            tool_call_query(tool_call, ""),
            fallback_query,
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(
                    tool_call.get("id") or f"{name}_{index}"
                ),
                "name": name,
                "content": run_tool(name, query),
            }
        )
    return messages


def _history_path(session_key: str) -> Path:
    return HISTORY_DIR / f"{safe_id(session_key)}.json"


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


def generate_reply(
    session_key: str,
    text: str,
    tool_context: str = "",
    image_data_urls: list[str] | None = None,
) -> str:
    images = list(image_data_urls or [])
    _ensure_history_loaded(session_key)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(session_key, tool_context)},
        {"role": "user", "content": build_untrusted_context(session_key, tool_context)},
    ]
    with chat_history_lock:
        messages.extend(chat_history.get(session_key, []).copy())
    messages.append({"role": "user", "content": build_user_content(text, images)})

    if tool_context.strip():
        reply = normalize_chat_response(llm.chat(messages, temperature=0.75)).content
    else:
        reply = ""
        needs_final_summary = False
        tools = chat_tools_for_text(text)
        allowed_tool_names = {
            str(tool.get("function", {}).get("name") or "")
            for tool in tools
            if isinstance(tool.get("function"), dict)
        }
        for _round in range(MAX_TOOL_CALL_ROUNDS):
            response = normalize_chat_response(
                llm.chat(
                    messages,
                    temperature=0.75,
                    tools=tools,
                    tool_choice="auto",
                )
            )
            reply = response.content
            tool_calls = filter_tool_calls(response.tool_calls, allowed_tool_names)
            if not tool_calls:
                needs_final_summary = False
                break
            messages.extend(
                build_tool_messages(
                    tool_calls,
                    text or "图片内容",
                    provider_context=response.provider_context,
                )
            )
            needs_final_summary = True

        if needs_final_summary:
            reply = normalize_chat_response(
                llm.chat(messages, temperature=0.75, tools=tools)
            ).content
            if not reply.strip():
                reply = TOOL_CALL_LIMIT_FALLBACK

    reply = re.sub(r"\[(?:SRCH|MEM|CHAT):?.*?\]", "", reply).strip()
    append_history(session_key, history_user_text(text, len(images)), reply)
    return reply
