import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


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
    bot_name: str = os.getenv("BOT_NAME", "ATRI")
    bot_persona: str = os.getenv(
        "BOT_PERSONA",
        "你是 ATRI，一个 QQ 聊天机器人。说话自然、可爱、有一点吐槽感，但要友好、简洁、靠谱。",
    )
    # ── Gemini / Google AI Studio ──
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    gemini_url: str = os.getenv(
        "GEMINI_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )

    # ── DeepSeek ──
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    deepseek_url: str = os.getenv(
        "DEEPSEEK_URL", "https://api.deepseek.com/chat/completions"
    )

    # Not yet used; reserved for media pipeline
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # ── LLM provider chain ──
    # Backward compatibility: LLM_PROVIDER=deepseek  →  primary = deepseek
    # If LLM_PROVIDER is set (non-empty), it overrides LLM_PRIMARY_PROVIDER.
    _llm_provider_compat: str = os.getenv("LLM_PROVIDER", "").strip().lower()
    llm_primary_provider: str = os.getenv(
        "LLM_PRIMARY_PROVIDER",
        _llm_provider_compat if _llm_provider_compat else "gemini",
    ).strip().lower()
    llm_primary_model: str = os.getenv("LLM_PRIMARY_MODEL", "").strip()

    llm_fallback_1_provider: str = os.getenv("LLM_FALLBACK_1_PROVIDER", "gemini").strip().lower()
    llm_fallback_1_model: str = os.getenv("LLM_FALLBACK_1_MODEL", "gemma-4-26b-a4b-it").strip()
    llm_fallback_2_provider: str = os.getenv("LLM_FALLBACK_2_PROVIDER", "deepseek").strip().lower()
    llm_fallback_2_model: str = os.getenv("LLM_FALLBACK_2_MODEL", "deepseek-v4-flash").strip()
    llm_fallback_3_provider: str = os.getenv("LLM_FALLBACK_3_PROVIDER", "deepseek").strip().lower()
    llm_fallback_3_model: str = os.getenv("LLM_FALLBACK_3_MODEL", "deepseek-v4-pro").strip()
    onebot_url: str = os.getenv("ONEBOT_API_URL", "http://127.0.0.1:3000").rstrip("/")
    onebot_access_token: str = os.getenv("ONEBOT_ACCESS_TOKEN", "")
    callback_secret: str = os.getenv("CALLBACK_SECRET", "")
    proxy_url: str = os.getenv("PROXY_URL", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    host: str = os.getenv("BOT_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port: int = env_int("BOT_PORT", 5000)
    require_group_at: bool = env_bool("REQUIRE_GROUP_AT", True)
    admin_qq_ids: frozenset[str] = env_csv_set("ADMIN_QQ_IDS")
    data_dir: Path = resolve_path(os.getenv("DATA_DIR", ""), "atri_data")
    search_max_results: int = env_int("SEARCH_MAX_RESULTS", 4)
    video_enable_media_pipeline: bool = env_bool("VIDEO_ENABLE_MEDIA_PIPELINE", False)
    video_max_download_mb: int = env_int("VIDEO_MAX_DOWNLOAD_MB", 50)
    video_max_seconds: int = env_int("VIDEO_MAX_SECONDS", 600)
    history_turns: int = env_int("HISTORY_TURNS", 8)
    memory_limit: int = env_int("MEMORY_LIMIT", 30)
    persist_history: bool = env_bool("PERSIST_HISTORY", True)
    request_timeout: float = env_float("REQUEST_TIMEOUT", 18.0)
    max_reply_chars: int = env_int("MAX_REPLY_CHARS", 1700)

    # ── Local ComfyUI Image Generation ──
    image_enable: bool = env_bool("IMAGE_ENABLE", False)
    image_chat_tool_enable: bool = env_bool("IMAGE_CHAT_TOOL_ENABLE", False)
    image_admin_only: bool = env_bool("IMAGE_ADMIN_ONLY", True)

    comfyui_base_url: str = os.getenv("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
    comfyui_timeout_seconds: int = env_int("COMFYUI_TIMEOUT_SECONDS", 600)
    comfyui_client_id: str = os.getenv("COMFYUI_CLIENT_ID", "atri-qq-bot")

    image_project_dir: str = os.getenv("IMAGE_PROJECT_DIR", "")
    comfyui_workflow_api_path: str = os.getenv("COMFYUI_WORKFLOW_API_PATH", "")
    image_default_preset: str = os.getenv("IMAGE_DEFAULT_PRESET", "highres_output")
    image_fallback_preset: str = os.getenv("IMAGE_FALLBACK_PRESET", "final_output")
    image_output_dir: str = os.getenv("IMAGE_OUTPUT_DIR", "atri_data/generated_images")
    image_max_batch_size: int = env_int("IMAGE_MAX_BATCH_SIZE", 1)
    image_single_task_lock: bool = env_bool("IMAGE_SINGLE_TASK_LOCK", True)

    image_use_character_lora: bool = env_bool("IMAGE_USE_CHARACTER_LORA", False)
    image_character_lora_name: str = os.getenv("IMAGE_CHARACTER_LORA_NAME", "")
    image_character_lora_strength: float = env_float("IMAGE_CHARACTER_LORA_STRENGTH", 0.0)

    image_use_style_lora: bool = env_bool("IMAGE_USE_STYLE_LORA", False)
    image_style_lora_name: str = os.getenv("IMAGE_STYLE_LORA_NAME", "")
    image_style_lora_strength: float = env_float("IMAGE_STYLE_LORA_STRENGTH", 0.0)

    # Optional ComfyUI node ID overrides (leave empty for auto-detection)
    comfyui_positive_node_id: str = os.getenv("COMFYUI_POSITIVE_NODE_ID", "").strip()
    comfyui_negative_node_id: str = os.getenv("COMFYUI_NEGATIVE_NODE_ID", "").strip()
    comfyui_preset_node_id: str = os.getenv("COMFYUI_PRESET_NODE_ID", "").strip()
    comfyui_seed_node_id: str = os.getenv("COMFYUI_SEED_NODE_ID", "").strip()
    comfyui_width_node_id: str = os.getenv("COMFYUI_WIDTH_NODE_ID", "").strip()
    comfyui_height_node_id: str = os.getenv("COMFYUI_HEIGHT_NODE_ID", "").strip()
    comfyui_steps_node_id: str = os.getenv("COMFYUI_STEPS_NODE_ID", "").strip()
    comfyui_cfg_node_id: str = os.getenv("COMFYUI_CFG_NODE_ID", "").strip()
    comfyui_character_lora_node_id: str = os.getenv("COMFYUI_CHARACTER_LORA_NODE_ID", "").strip()
    comfyui_style_lora_node_id: str = os.getenv("COMFYUI_STYLE_LORA_NODE_ID", "").strip()
    comfyui_save_prefix_node_id: str = os.getenv("COMFYUI_SAVE_PREFIX_NODE_ID", "").strip()

    @property
    def proxies(self) -> dict[str, str] | None:
        if not self.proxy_url:
            return None
        return {"http": self.proxy_url, "https": self.proxy_url}


config = Config()
