"""One-call route and query planning with deterministic degradation."""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from .models import RequestSource, SearchMode, SearchPlan, SearchQuery, SearchRequest


logger = logging.getLogger("qq-bot")


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
_SELF_CONTAINED_CREATIVE_RE = re.compile(
    r"(?:写|创作|编|续写|作|起草|生成|画).*(?:诗|故事|小说|歌词|文案|笑话|对联|剧本|信|邮件)"
    r"|讲(?:一个|个)?(?:故事|笑话)"
    r"|tell\s+me\s+(?:a\s+)?(?:story|joke)\b",
    re.IGNORECASE,
)
_FACT_LOOKUP_CUE_RE = re.compile(
    r"(?:最新|最近|近期|今天|今日|当前|现在|实时|新闻|漏洞|现任|今年|本周|本月|latest|current|today|news|recent)",
    re.IGNORECASE,
)
_TEXT_TRANSFORM_RE = re.compile(
    r"(?:翻译|改写|重写|润色|校对|纠错|缩写|扩写|总结|概括|摘要|paraphrase|rewrite|translate|summari[sz]e)",
    re.IGNORECASE,
)
_QUOTED_TEXT_RE = re.compile(r"[“‘\"']\s*([^”’\"']+)\s*[”’\"']")
_FRESHNESS_CUE_RE = re.compile(
    r"(?:最新|最近|近期|今天|今日|当前|现在|实时|今年|本周|本月|latest|current|today|recent)",
    re.IGNORECASE,
)
_LOOKUP_TOPIC_CUE_RE = re.compile(
    r"(?:新闻|消息|动态|进展|漏洞|版本|天气|价格|股价|汇率|行情|比分|排名|政策|规定|数据|"
    r"news|updates?|vulnerabilit(?:y|ies)|version|weather|price|score|ranking)",
    re.IGNORECASE,
)
_SUPPLIED_CONTENT_CUE_RE = re.compile(
    r"[,，。；;！!]|(?:新闻|消息|动态|进展|漏洞|版本|天气|价格|股价|汇率|行情|比分|排名|政策|规定|数据)"
    r"\s*(?:是(?!什么|否|怎么|如何)|为(?!什么|何)|包括|有|称|显示|指出|报道|宣布)",
    re.IGNORECASE,
)
_ALLOWED_ARITHMETIC_OPERATORS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitXor,
    ast.UAdd,
    ast.USub,
)


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
        except Exception as error:
            logger.debug("search route planner call failed (%s)", type(error).__name__)
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
                        start = content.find("{", index + 1)
                        break
                    if isinstance(value, dict):
                        return value
                    start = content.find("{", index + 1)
                    break
        else:
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
        or _is_self_contained_creative(text)
        or _is_supplied_text_transform(text)
        or _is_valid_arithmetic_expression(text)
    )


def _is_self_contained_creative(text: str) -> bool:
    return bool(
        _SELF_CONTAINED_CREATIVE_RE.search(text)
        and not _FACT_LOOKUP_CUE_RE.search(text)
    )


def _is_supplied_text_transform(text: str) -> bool:
    if not _TEXT_TRANSFORM_RE.search(text):
        return False

    source_texts: list[str] = []
    delimiter = re.search(r"[:：]\s*(\S.*)$", text)
    if delimiter:
        source_texts.append(delimiter.group(1).strip())
    source_texts.extend(match.group(1).strip() for match in _QUOTED_TEXT_RE.finditer(text))
    return bool(
        source_texts
        and not any(_is_fresh_lookup_topic(source) for source in source_texts)
    )


def _is_fresh_lookup_topic(text: str) -> bool:
    return bool(
        _FRESHNESS_CUE_RE.search(text)
        and _LOOKUP_TOPIC_CUE_RE.search(text)
        and not _SUPPLIED_CONTENT_CUE_RE.search(text)
    )


def _is_valid_arithmetic_expression(text: str) -> bool:
    try:
        root = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError):
        return False

    has_binary_operation = False
    for node in ast.walk(root):
        if isinstance(node, ast.BinOp):
            has_binary_operation = True
        elif isinstance(node, ast.UnaryOp):
            continue
        elif isinstance(node, (ast.operator, ast.unaryop)):
            if not isinstance(node, _ALLOWED_ARITHMETIC_OPERATORS):
                return False
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float, complex)):
                return False
        elif not isinstance(node, (ast.Expression, ast.Load)):
            return False
    return has_binary_operation
