from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from src.persona import get_persona


logger = logging.getLogger("qq-bot")
_ALLOWED_TONES = frozenset({"plain", "warm", "firm", "playful"})


@dataclass(frozen=True)
class TrustedCommandFacts:
    code: str
    status: str
    scope: str
    cause: str
    details: tuple[str, ...] = ()


class PersonaCommandRenderer:
    """Let a model select tone while deterministic code owns every fact."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def render(
        self,
        facts: TrustedCommandFacts,
        fallback_reply: str,
    ) -> str:
        try:
            persona = get_persona()
            tone = "warm"
            if self.model is not None:
                response = self.model.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "角色设定：\n"
                                f"<persona>\n{persona.content}\n</persona>\n"
                                f"请为角色 {persona.name} 选择回复语气。"
                                "只能输出以下一个英文标签：plain、warm、firm、playful。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "受信命令事实："
                                + json.dumps(
                                    asdict(facts),
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=8,
                )
                tone = str(getattr(response, "content", "") or "").strip()
                if tone not in _ALLOWED_TONES:
                    return fallback_reply
            return _apply_tone(persona.name, tone, fallback_reply)
        except Exception as error:
            logger.warning(
                "Command persona rendering failed error_type=%s",
                type(error).__name__,
            )
            return fallback_reply


def _apply_tone(persona_name: str, tone: str, exact_reply: str) -> str:
    if tone == "plain":
        return exact_reply
    if tone == "firm":
        return f"{persona_name}认真地说：{exact_reply}"
    if tone == "playful":
        return f"{persona_name}：{exact_reply} ✦"
    return f"{persona_name}：{exact_reply}"
