from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any
import unicodedata

from src.chat.prompt import build_search_system_prompt
from src.search.simple.models import SearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerResult:
    text: str
    degraded: bool


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"https?://\S+", "", text)
    text = "".join(
        ch
        for ch in text
        if ch == "\n" or not unicodedata.category(ch).startswith("C")
    )
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _summary_fallback(
    results: tuple[SearchResult, ...] | list[SearchResult],
) -> str:
    lines = ["根据搜索结果："]
    for i, r in enumerate(results[:5], 1):
        title = " ".join(r.title.split()).strip()
        excerpt = " ".join(r.excerpt.split()).strip()
        if len(excerpt) > 300:
            excerpt = excerpt[:300] + "..."
        if excerpt:
            lines.append(f"{i}. {title}：{excerpt}")
        else:
            lines.append(f"{i}. {title}")
    return "\n".join(lines)


class SearchAnswerer:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def answer(
        self,
        question: str = "",
        results: tuple[SearchResult, ...] | list[SearchResult] = (),
        *,
        base_messages: list[dict[str, Any]] | None = None,
        timeout_seconds: float = 20.0,
        context: Any = "",
        **kwargs: Any,
    ) -> AnswerResult:
        if "question" in kwargs:
            question = kwargs.pop("question")
        if "results" in kwargs:
            results = kwargs.pop("results")

        system_content = build_search_system_prompt(context)
        msgs: list[dict[str, Any]] = []
        has_system = False
        if base_messages:
            for i, m in enumerate(base_messages):
                if i == 0 and m.get("role") == "system":
                    msgs.append({"role": "system", "content": system_content})
                    has_system = True
                else:
                    msgs.append(dict(m))
        if not has_system:
            msgs.insert(0, {"role": "system", "content": system_content})

        evidence = [
            {"title": r.title, "excerpt": r.excerpt[:1500]}
            for r in results
        ]
        user_payload = {
            "question": question,
            "search_results": evidence,
        }
        msgs.append(
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            }
        )

        try:
            response = self._llm.chat(
                msgs,
                temperature=0.2,
                timeout_seconds=timeout_seconds,
            )
            raw_content = getattr(response, "content", "")
            cleaned = _clean_text(str(raw_content or ""))
            if cleaned:
                return AnswerResult(text=cleaned, degraded=False)
        except Exception:
            logger.debug("search answer model call failed", exc_info=True)

        return AnswerResult(text=_summary_fallback(results), degraded=True)
