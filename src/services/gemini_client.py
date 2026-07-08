"""Gemini (Google AI Studio) client via OpenAI-compatible endpoint.

Uses ``try_proxied_post`` for consistency with other HTTP clients in this project.
"""

import logging
from typing import Any

from src.config import Config
from src.services.llm_types import ChatResponse
from src.util import try_proxied_post

logger = logging.getLogger("qq-bot")


class GeminiClient:
    """Gemini chat-completion client over the OpenAI-compatible endpoint.

    The Google AI Studio base URL is configured as ``GEMINI_URL``
    (default: https://generativelanguage.googleapis.com/v1beta/openai/chat/completions).
    Authentication uses ``GEMINI_API_KEY`` as a Bearer token.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    # ── public ──────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResponse:
        if not self._cfg.gemini_api_key:
            raise RuntimeError(
                "Gemini API key is missing. Set GEMINI_API_KEY in .env."
            )

        payload: dict[str, Any] = {
            "model": model or self._cfg.gemini_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        response = try_proxied_post(
            self._cfg.gemini_url,
            proxies=self._cfg.proxies,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._cfg.gemini_api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._cfg.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        raw_tool_calls = message.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raw_tool_calls = []

        return ChatResponse(
            content=(message.get("content") or "").strip(),
            tool_calls=raw_tool_calls,
        )
