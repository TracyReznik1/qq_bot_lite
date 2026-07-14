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


class FallbackLLMClient:
    """Unified client that tries models from *chain* in order.

    Parameters
    ----------
    chain:
        Ordered list of ``LLMModelSpec`` entries describing the fallback
        sequence.
    """

    def __init__(self, chain: list[LLMModelSpec]) -> None:
        self._chain = chain
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
    ) -> ChatResponse:
        has_tools = bool(tools)
        has_images = _messages_have_images(messages)

        for spec in self._chain:
            if has_tools and not spec.supports_tools:
                logger.info(
                    "LLM skipping model (no tool support) provider=%s model=%s",
                    spec.provider,
                    spec.model,
                )
                continue

            client = self._get_client(spec)

            try:
                result = client.chat(
                    messages,
                    model=spec.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
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

                return result
            except RuntimeError as exc:
                # Missing API key → do NOT retry; log and skip this model.
                logger.warning(
                    "LLM config error provider=%s model=%s: %s",
                    spec.provider,
                    spec.model,
                    exc,
                )
                continue
            except Exception as exc:
                if _is_retryable_error(exc):
                    logger.warning(
                        "LLM call failed provider=%s model=%s reason=%s — trying next fallback",
                        spec.provider,
                        spec.model,
                        exc,
                    )
                    continue
                # For unexpected errors, log and continue (don't crash the bot).
                logger.warning(
                    "LLM call failed (non-retryable) provider=%s model=%s reason=%s",
                    spec.provider,
                    spec.model,
                    exc,
                )
                continue

        # All models exhausted
        if has_images:
            raise ImageRecognitionUnavailable("当前模型无法识别该图片。")
        raise RuntimeError("所有模型暂时不可用，请稍后再试。")

    # ── helpers ─────────────────────────────────────────────────────

    def _get_client(self, spec: LLMModelSpec) -> DeepSeekClient | GeminiClient:
        key = f"{spec.provider}:{spec.model}"
        if key not in self._clients:
            if spec.provider == "deepseek":
                self._clients[key] = DeepSeekClient(config)
            elif spec.provider == "gemini":
                self._clients[key] = GeminiClient(config)
            else:
                raise RuntimeError(
                    f"Unknown LLM provider: {spec.provider}"
                )
        return self._clients[key]


# ── chain builder ──────────────────────────────────────────────────────

def _build_chain(cfg=None) -> list[LLMModelSpec]:
    """Build the ordered model chain from ``Config``."""
    if cfg is None:
        cfg = config
    chain: list[LLMModelSpec] = []

    # Primary
    primary_model = cfg.llm_primary_model
    if not primary_model:
        if cfg.llm_primary_provider == "gemini":
            primary_model = cfg.gemini_model
        elif cfg.llm_primary_provider == "deepseek":
            primary_model = cfg.deepseek_model
        else:
            primary_model = cfg.gemini_model
    chain.append(
        LLMModelSpec(
            provider=cfg.llm_primary_provider,
            model=primary_model,
            supports_tools=_model_supports_tools(
                cfg.llm_primary_provider, primary_model
            ),
        )
    )

    # Fallback slots — only add when provider is non-empty
    for fb_provider_attr, fb_model_attr in (
        ("llm_fallback_1_provider", "llm_fallback_1_model"),
        ("llm_fallback_2_provider", "llm_fallback_2_model"),
        ("llm_fallback_3_provider", "llm_fallback_3_model"),
    ):
        provider = getattr(cfg, fb_provider_attr, "").strip()
        model_name = getattr(cfg, fb_model_attr, "").strip()
        if not provider:
            continue
        chain.append(
            LLMModelSpec(
                provider=provider,
                model=model_name,
                supports_tools=_model_supports_tools(provider, model_name),
            )
        )

    # Deduplicate while preserving order
    seen: set[tuple[str, str]] = set()
    deduped: list[LLMModelSpec] = []
    for spec in chain:
        key = (spec.provider, spec.model)
        if key not in seen:
            seen.add(key)
            deduped.append(spec)
    return deduped


# ── singleton accessor ─────────────────────────────────────────────────

_llm_client: FallbackLLMClient | None = None


def get_llm_client() -> FallbackLLMClient:
    """Return the singleton ``FallbackLLMClient`` built from config."""
    global _llm_client
    if _llm_client is None:
        _llm_client = FallbackLLMClient(_build_chain())
    return _llm_client
