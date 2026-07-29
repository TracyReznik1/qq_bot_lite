import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


from src.model_config import (
    ConfiguredModel,
    parse_model_chain,
    parse_chat_models,
    validate_model_configuration,
)


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


DEFAULT_DATA_DIR_NAME = "qqbot_data"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_csv_set(name: str) -> frozenset[str]:
    value = os.getenv(name, "")
    items = [item.strip() for item in value.replace(";", ",").split(",")]
    return frozenset(item for item in items if item)


def resolve_path(value: str, default: str) -> Path:
    path = Path(value or default)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


@dataclass(frozen=True)
class Config:
    persona_path: Path = field(default_factory=lambda: BASE_DIR / "config" / "persona.md")
    # ── Gemini / Google AI Studio native API ──
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip()
    )
    gemini_url: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_URL",
            "https://generativelanguage.googleapis.com/v1",
        ).rstrip("/")
    )

    # ── DeepSeek ──
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "").strip()
    )
    deepseek_url: str = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_URL",
            "https://api.deepseek.com/chat/completions",
        ).strip()
    )

    _chat_models_raw: str = field(
        default_factory=lambda: os.getenv("CHAT_MODELS", ""),
        repr=False,
    )
    _memory_models_raw: str = field(
        default_factory=lambda: os.getenv("MEMORY_MODELS", ""),
        repr=False,
    )
    chat_models: tuple[ConfiguredModel, ...] = field(init=False)
    memory_models: tuple[ConfiguredModel, ...] = field(init=False)

    def __post_init__(self) -> None:
        chat_models = parse_chat_models(self._chat_models_raw)
        provider_api_keys = {
            "gemini": self.gemini_api_key,
            "deepseek": self.deepseek_api_key,
        }
        validate_model_configuration(
            chat_models,
            provider_api_keys=provider_api_keys,
            gemini_url=self.gemini_url,
        )
        if str(self._memory_models_raw or "").strip():
            memory_models = parse_model_chain(
                self._memory_models_raw,
                "MEMORY_MODELS",
            )
            validate_model_configuration(
                memory_models,
                provider_api_keys=provider_api_keys,
                gemini_url=self.gemini_url,
                setting_name="MEMORY_MODELS",
            )
        else:
            memory_models = chat_models
        object.__setattr__(self, "chat_models", chat_models)
        object.__setattr__(self, "memory_models", memory_models)
    onebot_url: str = os.getenv("ONEBOT_API_URL", "http://127.0.0.1:3000").rstrip("/")
    onebot_access_token: str = os.getenv("ONEBOT_ACCESS_TOKEN", "")
    callback_secret: str = os.getenv("CALLBACK_SECRET", "")
    proxy_url: str = os.getenv("PROXY_URL", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    host: str = os.getenv("BOT_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port: int = env_int("BOT_PORT", 5000)
    require_group_at: bool = env_bool("REQUIRE_GROUP_AT", True)
    admin_qq_ids: frozenset[str] = env_csv_set("ADMIN_QQ_IDS")
    data_dir: Path = field(
        default_factory=lambda: resolve_path(
            os.getenv("DATA_DIR", ""),
            DEFAULT_DATA_DIR_NAME,
        )
    )

    @property
    def memory_database_path(self) -> Path:
        return self.data_dir / "memory.sqlite3"

    search_max_results: int = env_int("SEARCH_MAX_RESULTS", 4)
    history_turns: int = env_int("HISTORY_TURNS", 8)
    persist_history: bool = env_bool("PERSIST_HISTORY", True)
    message_workers: int = field(default_factory=lambda: max(env_int("MESSAGE_WORKERS", 8), 1))
    request_timeout: float = env_float("REQUEST_TIMEOUT", 18.0)
    max_reply_chars: int = env_int("MAX_REPLY_CHARS", 1700)

    @property
    def proxies(self) -> dict[str, str] | None:
        if not self.proxy_url:
            return None
        return {"http": self.proxy_url, "https": self.proxy_url}


config = Config()
