"""Fixed-mode multimodal query planning with deterministic degradation."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.search.simple.models import SearchMode, SearchPlan, SearchQuery

logger = logging.getLogger("qq-bot")

IMAGE_ONLY_FALLBACK_QUERY = "识别并查找图片中的主体、事件或内容"

_PLANNER_SYSTEM_PROMPT = """Analyze the user's message and images to generate concise web search queries.
Return exactly one JSON object with only this field:
{"queries":["concise query"]}
For light mode, return 1 concise query. For standard mode, return 1 to 3 diverse queries. Do not call tools and do not include any other fields.
"""


class QueryPlanner:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def plan(
        self,
        *,
        mode: SearchMode,
        text: str,
        images: tuple[str, ...] = (),
        timeout_seconds: float,
    ) -> SearchPlan:
        if mode is SearchMode.SKIP:
            raise ValueError("skip mode must not invoke QueryPlanner")
        fallback = " ".join(text.split()) or IMAGE_ONLY_FALLBACK_QUERY
        try:
            response = self._llm.chat(
                _planner_messages(text, images),
                temperature=0.0,
                max_tokens=256,
                tools=None,
                tool_choice="none",
                timeout_seconds=timeout_seconds,
            )
            queries = _parse_queries(getattr(response, "content", ""))
        except Exception as error:
            logger.debug("query planner failed error_type=%s", type(error).__name__)
            queries = ()
        limit = 1 if mode is SearchMode.LIGHT else 3
        selected = queries[:limit] or (fallback,)
        return SearchPlan(
            mode=mode,
            queries=tuple(
                SearchQuery(f"q{index}", query)
                for index, query in enumerate(selected, 1)
            ),
            planner_degraded=not bool(queries),
        )


def _planner_messages(text: str, images: tuple[str, ...]) -> list[dict[str, Any]]:
    if not images:
        user_content: Any = text or "请生成联网搜索词。"
    else:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": text or "请根据图片生成联网搜索词。"}
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": image}}
            for image in images
        )
        user_content = content
    return [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_queries(content: str) -> tuple[str, ...]:
    payload = _first_json_object(str(content or ""))
    if payload is None:
        return ()
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list):
        return ()
    output: list[str] = []
    seen: set[str] = set()
    for item in raw_queries:
        if not isinstance(item, str):
            continue
        query = " ".join(item.split())
        if not query or query in seen:
            continue
        seen.add(query)
        output.append(query[:500])
    return tuple(output)


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
