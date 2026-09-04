"""Fallback LLM client that chains multiple providers.

Builds a model chain from ``Config`` and tries each model in order.  When a
model fails (network error, HTTP 429 / 5xx, malformed response, or missing
API key) the next model in the chain is tried automatically.

Models with ``supports_tools=False`` are skipped when the caller passes a
non-empty ``tools`` list to ``chat()``.

After all models have been exhausted, a unified error is raised instead of
dumping a raw traceback into the QQ chat.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from src.config import config
from src.services.deepseek_client import DeepSeekClient
from src.services.gemini_client import GeminiClient
from src.services.llm_types import ChatResponse, LLMModelSpec

logger = logging.getLogger("qq-bot")

# ── model capability registry ──────────────────────────────────────────
# Default: gemma-4-26b-a4b-it does NOT support tool calling.
# This should be updated once real API testing confirms capability.
_MODEL_CAPABILITIES: dict[str, bool] = {
    "gemma-4-26b-a4b-it": False,
}


def _model_supports_tools(provider: str, model_name: str) -> bool:
    """Check whether *model_name* supports function-calling / tools."""
    key = model_name.lower()
    if key in _MODEL_CAPABILITIES:
        return _MODEL_CAPABILITIES[key]
    # For providers not in the deny-list we assume tool support.
    return True


# ── fallback logic ─────────────────────────────────────────────────────

_FALLBACK_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


class ImageRecognitionUnavailable(RuntimeError):
    """Raised after every configured model fails a request with images."""


def _messages_have_images(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        if any(
            isinstance(item, dict) and item.get("type") == "image_url"
            for item in content
        ):
            return True
    return False


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True when *exc* signals a transient failure worth retrying."""
    if isinstance(exc, requests.HTTPError):
        status = (
            getattr(getattr(exc, "response", None), "status_code", None) or 0
        )
        return status in _FALLBACK_HTTP_STATUSES
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


def _tool_affinity(
    messages: list[dict[str, Any]],
) -> tuple[str, str] | None:
    for message in reversed(messages):
        if (
            not isinstance(message, dict)
            or message.get("role") != "assistant"
            or not message.get("tool_calls")
        ):
            continue
        context = message.get("_provider_context")
        if not isinstance(context, dict):
            continue
        provider = context.get("provider")
        model = context.get("model")
        if (
            isinstance(provider, str)
            and provider
            and isinstance(model, str)
            and model
        ):
            return provider, model
    return None


class FallbackLLMClient:
    """Unified client that tries models from *chain* in order.

    Parameters
    ----------
    chain:
        Ordered list of ``LLMModelSpec`` entries describing the fallback
        sequence.
    """

    def __init__(
        self,
        chain: list[LLMModelSpec],
        cfg: Config | None = None,
        *,
        gemini_api_key: str | None = None,
        deepseek_api_key: str | None = None,
    ) -> None:
        self._chain = chain
        self._cfg = cfg or config
        self._gemini_api_key = gemini_api_key.strip() if gemini_api_key else None
        self._deepseek_api_key = deepseek_api_key.strip() if deepseek_api_key else None
        self._clients: dict[str, DeepSeekClient | GeminiClient] = {}

    # ── public API ──────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> ChatResponse:
        has_tools = bool(tools)
        contains_images = _messages_have_images(messages)
        affinity = _tool_affinity(messages)

        started = time.monotonic()
        for spec in self._chain:
            if affinity is not None and affinity != (spec.provider, spec.model):
                continue
            if has_tools and not spec.supports_tools:
                logger.info(
                    "LLM skipping model (no tool support) provider=%s model=%s",
                    spec.provider,
                    spec.model,
                )
                continue

            client = self._get_client(spec)

            try:
                remaining = timeout_seconds
                if timeout_seconds is not None:
                    remaining = max(timeout_seconds - (time.monotonic() - started), 0.0)
                    if remaining <= 0:
                        raise TimeoutError("LLM deadline expired")
                result = client.chat(
                    messages,
                    model=spec.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    timeout_seconds=remaining,
                )
                # Guard against responses that are structurally broken:
                # content is empty AND tool_calls is empty → treat as failure.
                if not result.content and not result.tool_calls:
                    logger.warning(
                        "LLM returned empty content+tools provider=%s model=%s — treating as failure",
                        spec.provider,
                        spec.model,
                    )
                    continue

                if result.tool_calls:
                    result.provider_context = {
                        **(result.provider_context or {}),
                        "provider": spec.provider,
                        "model": spec.model,
                    }
                return result
            except RuntimeError as exc:
                # Missing API key → do NOT retry; log and skip this model.
                logger.warning(
                    "LLM config error provider=%s model=%s error_type=%s",
                    spec.provider,
                    spec.model,
                    type(exc).__name__,
                )
                continue
            except Exception as exc:
                if _is_retryable_error(exc):
                    logger.warning(
                        "LLM call failed provider=%s model=%s error_type=%s — trying next fallback",
                        spec.provider,
                        spec.model,
                        type(exc).__name__,
                    )
                    continue
                # For unexpected errors, log and continue (don't crash the bot).
                logger.warning(
                    "LLM call failed (non-retryable) provider=%s model=%s error_type=%s",
                    spec.provider,
                    spec.model,
                    type(exc).__name__,
                )
                continue

        # All models exhausted
        if contains_images:
            raise ImageRecognitionUnavailable("当前模型无法识别该图片。")
        raise RuntimeError("所有模型暂时不可用，请稍后再试。")

    # ── helpers ─────────────────────────────────────────────────────

    def _get_client(self, spec: LLMModelSpec) -> DeepSeekClient | GeminiClient:
        key = f"{spec.provider}:{spec.model}"
        if key not in self._clients:
            if spec.provider == "deepseek":
                self._clients[key] = DeepSeekClient(
                    self._cfg,
                    api_key=self._deepseek_api_key,
                )
            elif spec.provider == "gemini":
                self._clients[key] = GeminiClient(
                    self._cfg,
                    api_key=self._gemini_api_key,
                )
            else:
                raise RuntimeError(
                    f"Unknown LLM provider: {spec.provider}"
                )
        return self._clients[key]


# ── chain builder ──────────────────────────────────────────────────────

def _build_chain(cfg=None, models=None) -> list[LLMModelSpec]:
    """Build the exact ordered chain validated by Config."""
    if cfg is None:
        cfg = config
    selected = tuple(models or cfg.chat_models)
    return [
        LLMModelSpec(
            provider=item.provider,
            model=item.model,
            supports_tools=_model_supports_tools(
                item.provider,
                item.model,
            ),
        )
        for item in selected
    ]


# ── singleton accessor ─────────────────────────────────────────────────

_llm_client: FallbackLLMClient | None = None
_memory_llm_client: FallbackLLMClient | None = None


def get_llm_client() -> FallbackLLMClient:
    """Return the singleton ``FallbackLLMClient`` built from config."""
    global _llm_client
    if _llm_client is None:
        _llm_client = FallbackLLMClient(
            _build_chain(),
            cfg=config,
            gemini_api_key=getattr(config, "gemini_api_key", None),
            deepseek_api_key=getattr(config, "deepseek_api_key", None),
        )
    return _llm_client


def get_memory_llm_client() -> FallbackLLMClient:
    """Return the memory-extraction client with its own fallback chain."""
    global _memory_llm_client
    if _memory_llm_client is None:
        memory_gemini_key = getattr(
            config,
            "memory_gemini_api_key",
            None,
        ) or getattr(config, "gemini_api_key", None)
        memory_deepseek_key = getattr(
            config,
            "memory_deepseek_api_key",
            None,
        ) or getattr(config, "deepseek_api_key", None)
        _memory_llm_client = FallbackLLMClient(
            _build_chain(models=config.memory_models),
            cfg=config,
            gemini_api_key=memory_gemini_key,
            deepseek_api_key=memory_deepseek_key,
        )
    return _memory_llm_client
