from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
from urllib.parse import urlsplit


SUPPORTED_PROVIDERS = frozenset({"gemini", "deepseek"})
PROVIDER_KEY_VARIABLES = {
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


class ModelConfigurationError(ValueError):
    """Raised when the environment model chain is not safe to start."""


@dataclass(frozen=True)
class ConfiguredModel:
    provider: str
    model: str


def parse_model_chain(
    value: str | None,
    setting_name: str,
) -> tuple[ConfiguredModel, ...]:
    raw = str(value or "")
    if not raw.strip():
        raise ModelConfigurationError(f"{setting_name} 不能为空")

    parsed: list[ConfiguredModel] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw.split(","), 1):
        stripped = item.strip()
        if not stripped:
            raise ModelConfigurationError(
                f"{setting_name} 第 {index} 项为空"
            )
        if ":" not in stripped:
            raise ModelConfigurationError(
                f"{setting_name} 第 {index} 项缺少英文冒号"
            )

        provider_text, model_text = stripped.split(":", 1)
        provider = provider_text.strip().lower()
        model = model_text.strip()
        if not provider:
            raise ModelConfigurationError(
                f"{setting_name} 第 {index} 项缺少提供商"
            )
        if not model:
            raise ModelConfigurationError(
                f"{setting_name} 第 {index} 项缺少模型名"
            )
        if provider not in SUPPORTED_PROVIDERS:
            raise ModelConfigurationError(
                f"{setting_name} 第 {index} 项提供商仅支持 gemini 或 deepseek"
            )

        key = (provider, model)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(ConfiguredModel(provider=provider, model=model))

    return tuple(parsed)


def parse_chat_models(value: str | None) -> tuple[ConfiguredModel, ...]:
    return parse_model_chain(value, "CHAT_MODELS")


def validate_model_configuration(
    models: Sequence[ConfiguredModel],
    *,
    provider_api_keys: Mapping[str, str],
    gemini_url: str,
    setting_name: str = "CHAT_MODELS",
) -> None:
    referenced = {model.provider for model in models}
    for provider in sorted(referenced):
        if not str(provider_api_keys.get(provider, "")).strip():
            variable = PROVIDER_KEY_VARIABLES[provider]
            raise ModelConfigurationError(
                f"{setting_name} 使用 {provider}，但 {variable} 未配置"
            )

    if "gemini" not in referenced:
        return

    parsed_url = urlsplit(str(gemini_url or "").strip())
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ModelConfigurationError(
            "GEMINI_URL 必须是完整的 HTTP/HTTPS 基础地址"
        )
    normalized_path = parsed_url.path.casefold()
    if "/openai/" in normalized_path or ":generatecontent" in normalized_path:
        raise ModelConfigurationError(
            "GEMINI_URL 必须是原生 API 基础地址，不能填写完整方法地址"
        )
