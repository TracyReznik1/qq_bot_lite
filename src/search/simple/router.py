"""Retrieval benefit search router for ordinary chat turns."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.config import config
from src.search.simple.models import SearchMode, SearchRouteDecision
from src.services.llm_client import get_router_llm_client

logger = logging.getLogger("qq-bot")

_ROUTER_SYSTEM_PROMPT = """You are a search routing decision model.
Your task is to determine whether answering the user's message requires an external web search based strictly on Retrieval Benefit.
Retrieval Benefit means: "Would external information substantially improve the accuracy, recency, completeness, or verifiability of the response?"

Decision Rules:
1. Do NOT evaluate whether you already know the answer. If the question depends on external real-world facts, entities, technical concepts, or verification, favor "light".
2. Favor "skip" (no search) when:
   - The message is casual chat, greeting, emotional banter, personality play, or social interaction (e.g. "你是笨蛋吗", "晚上好", "今天好累").
   - The message is a self-contained instruction (translation, text rewriting, creative writing, code generation without external APIs).
   - The message refers purely to the current conversation or user-provided content.
3. Favor "light" (search needed) when:
   - The message asks about external facts, definitions, events, people, places, products, or technical specs (e.g. "ANSYS 的 Bonded MPC 是什么").
   - The message asks for recent/timely information or fact-checking (e.g. "Gemini 最近有什么更新", "查一下这个消息是真是假").
   - You are unsure whether search has benefit on an external factual question (default to "light").
4. Never output "standard". The search_mode MUST strictly be either "skip" or "light".
5. Output format: Return JSON ONLY with this exact schema:
{
  "search_mode": "skip" | "light",
  "reason_code": "conversation" | "self_contained" | "external_fact" | "recency" | "verification",
  "retrieval_topics": ["concise query topic"]
}
If search_mode is "skip", retrieval_topics must be [].
If search_mode is "light", retrieval_topics must contain exactly 1 concise search query topic.
"""


class SearchRouter:
    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def _get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        return get_router_llm_client()

    def route(
        self,
        text: str,
        images: tuple[str, ...] = (),
        *,
        timeout_seconds: float | None = None,
    ) -> SearchRouteDecision:
        normalized_text = " ".join(str(text or "").split())
        timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(getattr(config, "search_router_timeout", 5.0))
        )
        try:
            llm = self._get_llm()
            messages = _router_messages(normalized_text, images)
            response = llm.chat(
                messages,
                temperature=0.0,
                max_tokens=200,
                tools=None,
                tool_choice="none",
                timeout_seconds=timeout,
            )
            content = getattr(response, "content", "")
            return _parse_decision(content, normalized_text)
        except Exception as error:
            logger.warning(
                "SearchRouter execution failed (%s), degrading to LIGHT",
                type(error).__name__,
            )
            return SearchRouteDecision(
                mode=SearchMode.LIGHT,
                reason_code="degraded_fallback",
                retrieval_topics=(normalized_text,) if normalized_text else (),
            )


def _router_messages(text: str, images: tuple[str, ...]) -> list[dict[str, Any]]:
    if not images:
        user_content: Any = text or "请判断是否需要联网搜索。"
    else:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": text or "请根据图片及内容判断是否需要联网搜索。"}
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": image}}
            for image in images
        )
        user_content = content
    return [
        {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_decision(content: str, raw_text: str) -> SearchRouteDecision:
    payload = _first_json_object(str(content or ""))
    if payload is None:
        return SearchRouteDecision(
            mode=SearchMode.LIGHT,
            reason_code="degraded_fallback",
            retrieval_topics=(raw_text,) if raw_text else (),
        )
    raw_mode = str(payload.get("search_mode", "")).strip().lower()
    reason_code = str(payload.get("reason_code", "")).strip() or "unspecified"
    raw_topics = payload.get("retrieval_topics", [])

    if raw_mode == "skip":
        return SearchRouteDecision(
            mode=SearchMode.SKIP,
            reason_code=reason_code,
            retrieval_topics=(),
        )

    topics = []
    if isinstance(raw_topics, list):
        for item in raw_topics:
            if isinstance(item, str) and item.strip():
                clean_topic = " ".join(item.split())
                if clean_topic:
                    topics.append(clean_topic[:500])
    selected_topics = tuple(topics[:1]) if topics else ((raw_text,) if raw_text else ())

    return SearchRouteDecision(
        mode=SearchMode.LIGHT,
        reason_code=reason_code,
        retrieval_topics=selected_topics,
    )


def _first_json_object(content: str) -> dict[str, Any] | None:
    start = content.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(content)):
            character = content[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(content[start : index + 1])
                    except (TypeError, ValueError):
                        start = content.find("{", index + 1)
                        break
                    if isinstance(value, dict):
                        return value
                    start = content.find("{", index + 1)
                    break
        else:
            start = content.find("{", start + 1)
    return None
