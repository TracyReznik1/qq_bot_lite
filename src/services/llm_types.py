"""Shared types for LLM provider clients.

All LLM clients MUST return ``ChatResponse`` so callers don't depend on
a specific provider.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatResponse:
    """Unified response plus optional current-turn provider protocol state."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    provider_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMModelSpec:
    """Describes one model in the fallback chain."""

    provider: str       # "gemini" | "deepseek"
    model: str          # e.g. "gemini-3.1-flash-lite"
    supports_tools: bool = True
