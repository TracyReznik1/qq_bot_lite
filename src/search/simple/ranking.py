"""Tolerant relevance ranking for simple search evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
import math
from typing import Any

from src.search.simple.models import SearchResult

logger = logging.getLogger("qq-bot")

_RANKER_SYSTEM_PROMPT = """Score how directly and reliably each candidate search result answers the user's question.
Return exactly one JSON object with a "scores" mapping from result_id to a float between 0.0 and 1.0.
Results that are irrelevant, unhelpful, or spam must be assigned 0.0.
Example: {"scores":{"R1":0.9,"R2":0.0}}
Do not include explanations or any other fields.
"""


@dataclass(frozen=True)
class RankingResult:
    results: tuple[SearchResult, ...]
    degraded: bool


class EvidenceRanker:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def rank(
        self,
        question: str,
        results: tuple[SearchResult, ...],
        *,
        timeout_seconds: float,
    ) -> RankingResult:
        if not results:
            return RankingResult((), degraded=False)

        messages = _build_ranker_messages(question, results)
        try:
            response = self._llm.chat(
                messages,
                temperature=0.0,
                max_tokens=512,
                tools=None,
                tool_choice="none",
                timeout_seconds=timeout_seconds,
            )
            scores = _parse_scores(getattr(response, "content", ""))
        except Exception as error:
            logger.debug("evidence ranker call failed (%s)", type(error).__name__)
            return RankingResult(results, degraded=True)

        if scores is None:
            return RankingResult(results, degraded=True)

        known_ids = {r.result_id for r in results}
        has_valid_known_score = any(
            rid in known_ids and _is_finite_number(val)
            for rid, val in scores.items()
        )
        if not has_valid_known_score:
            return RankingResult(results, degraded=True)

        ranked: list[SearchResult] = []
        for r in results:
            raw_score = scores.get(r.result_id)
            if raw_score is None:
                final_score = 0.5
            elif _is_finite_number(raw_score):
                num = float(raw_score)
                if num == 0.0:
                    continue  # explicit zero removes the result
                final_score = min(max(num, 0.0), 1.0)
            else:
                final_score = 0.5

            ranked.append(replace(r, score=final_score))

        ranked.sort(key=lambda item: item.score, reverse=True)
        return RankingResult(tuple(ranked), degraded=False)


def _build_ranker_messages(
    question: str,
    results: tuple[SearchResult, ...],
) -> list[dict[str, Any]]:
    items_text = []
    for r in results:
        items_text.append(f"[{r.result_id}] {r.title}\n{r.excerpt}")
    user_content = f"Question: {question}\n\nCandidates:\n" + "\n\n".join(items_text)
    return [
        {"role": "system", "content": _RANKER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def _parse_scores(content: str) -> dict[str, Any] | None:
    payload = _first_json_object(str(content or ""))
    if payload is None:
        return None
    scores = payload.get("scores")
    if isinstance(scores, dict):
        return scores
    return None


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
