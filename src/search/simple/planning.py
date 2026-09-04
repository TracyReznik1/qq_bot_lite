"""One-call route and query planning with deterministic degradation."""

from __future__ import annotations

import json
import re
from typing import Any

from .models import RequestSource, SearchMode, SearchPlan, SearchQuery, SearchRequest


_PLANNER_SYSTEM_PROMPT = """Decide whether answering the user's message needs web search and create concise search queries.
Return exactly one JSON object with only these fields:
{"mode":"skip|light|standard","queries":["query text"]}
Use skip for social chat, creative writing, transformations of user-provided text, and arithmetic.
Use light for one straightforward factual lookup and standard for broader research.
Light has at most one query; standard has at most three. Do not call tools.
"""

_GREETING_RE = re.compile(
    r"^(?:你?好|您好|嗨|哈[喽啰罗]|hi|hello|hey|早上好|上午好|下午好|晚上好|晚安)"
    r"(?:呀|啊|哦|哟|！|!|。|\.|～|~|\s)*$",
    re.IGNORECASE,
)
_CREATIVE_RE = re.compile(
    r"^(?:请|帮我|请帮我|能否|可以|给我|请给我)?\s*"
    r"(?:(?:写|创作|编|续写|作|起草|生成|画)|讲(?:一个|个)?(?:故事|笑话)|tell\s+me\s+(?:a\s+)?(?:story|joke)\b)",
    re.IGNORECASE,
)
_TEXT_TRANSFORM_RE = re.compile(
    r"^(?:请|帮我|请帮我|能否|可以)?\s*"
    r"(?:翻译|改写|重写|润色|校对|纠错|缩写|扩写|总结|概括|摘要|paraphrase|rewrite|translate|summari[sz]e)",
    re.IGNORECASE,
)
_ARITHMETIC_RE = re.compile(r"^[\d\s+\-*/%().=^]+$")


class RoutePlanner:
    def __init__(self, llm: Any):
        self._llm = llm

    def plan(self, request: SearchRequest, *, timeout_seconds: float) -> SearchPlan:
        try:
            response = self._llm.chat(
                _planner_messages(request.question, request.has_images),
                temperature=0.0,
                max_tokens=256,
                tools=None,
                tool_choice="none",
                timeout_seconds=timeout_seconds,
            )
            parsed = _parse_plan(getattr(response, "content", ""), request)
            if parsed is not None:
                return parsed
        except Exception:
            pass
        return _fallback_plan(request)


def _planner_messages(question: str, has_images: bool) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"question": question, "has_images": bool(has_images)},
                ensure_ascii=False,
            ),
        },
    ]


def _parse_plan(content: Any, request: SearchRequest) -> SearchPlan | None:
    payload = _first_json_object(str(content or ""))
    if payload is None:
        return None

    try:
        mode = SearchMode(payload.get("mode"))
    except (TypeError, ValueError):
        return None

    queries = _normalized_queries(payload.get("queries"))
    forced = request.force_search or request.request_source is RequestSource.COMMAND
    if forced:
        mode = SearchMode.STANDARD

    if mode is SearchMode.SKIP:
        return SearchPlan(mode, ())

    if not queries:
        queries = [request.question]
    limit = 1 if mode is SearchMode.LIGHT else 3
    return SearchPlan(
        mode,
        tuple(SearchQuery(f"q{index}", text) for index, text in enumerate(queries[:limit], 1)),
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
                        return None
                    return value if isinstance(value, dict) else None
        start = content.find("{", start + 1)
    return None


def _normalized_queries(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        query = " ".join(item.split())
        if not query or query in seen:
            continue
        seen.add(query)
        output.append(query)
    return output


def _fallback_plan(request: SearchRequest) -> SearchPlan:
    forced = request.force_search or request.request_source is RequestSource.COMMAND
    if forced:
        mode = SearchMode.STANDARD
    elif _obviously_no_search(request.question):
        mode = SearchMode.SKIP
    else:
        mode = SearchMode.LIGHT

    queries = () if mode is SearchMode.SKIP else (SearchQuery("q1", request.question),)
    return SearchPlan(mode, queries, planner_degraded=True)


def _obviously_no_search(question: str) -> bool:
    text = " ".join(question.strip().split())
    return bool(
        _GREETING_RE.fullmatch(text)
        or _CREATIVE_RE.match(text)
        or _TEXT_TRANSFORM_RE.match(text)
        or (_ARITHMETIC_RE.fullmatch(text) and any(char.isdigit() for char in text))
    )
