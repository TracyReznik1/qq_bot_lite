# Environment Chat Model Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed model-selection variables with one strictly validated `CHAT_MODELS` chain and migrate Gemini from the OpenAI-compatible endpoint to the native stateless `generateContent` REST API.

**Architecture:** A focused `src/model_config.py` module parses and validates the environment model chain. `Config` owns the immutable parsed chain, `FallbackLLMClient` consumes it without applying defaults, and `GeminiClient` translates the existing provider-neutral chat contract to Gemini native request and response structures. Local history remains the only conversation state shared by Gemini and DeepSeek.

**Tech Stack:** Python 3.11+, `dataclasses`, `urllib.parse`, `requests`, Flask, `python-dotenv`, `unittest`.

## Global Constraints

- Supported model providers are exactly `gemini` and `deepseek`.
- `CHAT_MODELS` is required; the first item is primary and later items are fallbacks.
- Remove `GEMINI_MODEL`, `DEEPSEEK_MODEL`, `LLM_PROVIDER`, `LLM_PRIMARY_*`, and all `LLM_FALLBACK_*` runtime compatibility.
- Every provider named in `CHAT_MODELS` must have its corresponding API Key.
- Gemini uses native REST `generateContent`; do not use the OpenAI compatibility endpoint, Interactions API, Google GenAI SDK, remote conversation IDs, streaming, background execution, or managed agents.
- Default `GEMINI_URL` is the base URL `https://generativelanguage.googleapis.com/v1`.
- Local persisted history remains authoritative and is sent with each provider request.
- Do not read, modify, stage, or commit the real `.env`.
- Do not add dependencies.
- Preserve all chat, web search, image understanding, memory, OneBot, and multi-session FIFO behavior.
- At execution start, verify `git hash-object README.md` equals `git rev-parse HEAD:README.md`. If it does not, stop and ask how to preserve the user change before editing README. The hashes were equal when this plan was written even though Git reported a metadata-only modified state.
- Execute feature work in an isolated worktree created with `superpowers:using-git-worktrees`.

---

## File Map

- Create `src/model_config.py`: immutable model entry, strict parser, provider-Key and Gemini base-URL validation.
- Modify `src/config.py`: replace old model fields with `chat_models`.
- Modify `src/services/llm_client.py`: build the fallback chain directly from `chat_models`.
- Replace `src/services/gemini_client.py`: native `generateContent` REST adapter.
- Modify `src/services/llm_types.py`: carry short-lived provider context for tool-call protocol metadata.
- Modify `src/chat/chat_service.py`: attach provider context only to the current tool loop.
- Modify `src/services/deepseek_client.py`: remove private provider context before OpenAI-compatible requests.
- Modify `run_bot.py`: convert startup model-configuration failures into concise Chinese stderr and exit code 2.
- Modify `src/main.py`: expose the non-secret model chain through `/health`.
- Modify `tests/__init__.py`: provide deterministic fake model configuration before package-based test discovery imports runtime modules.
- Create `tests/test_model_config.py`: parser and validation tests.
- Create `tests/test_model_chain_configuration.py`: `Config`, chain builder, and startup error tests.
- Create `tests/test_gemini_native_client.py`: native request/response adapter tests.
- Create `tests/test_health.py`: health response and secret-exclusion tests.
- Modify `tests/test_messaging.py`: include required model variables in the environment-clearing helper.
- Modify `tests/test_multimodal_chat.py`: rename the stale OpenAI-specific internal-content test.
- Modify `tests/test_product_scope.py`: prevent reintroduction of the Gemini OpenAI endpoint or Interactions state.
- Modify `.env.example`: replace eleven model-selection variables with `CHAT_MODELS` and the native Gemini base URL.
- Modify `README.md`: document the new chain, native Gemini endpoint, migration break, validation, and troubleshooting.
- Modify `tests/test_readme_guide.py`: assert the new user-facing model configuration and retain source-derived environment coverage.

---

### Task 1: Add the Strict Model-Chain Parser

**Files:**
- Create: `src/model_config.py`
- Create: `tests/test_model_config.py`

**Interfaces:**
- Produces: `ConfiguredModel(provider: str, model: str)`.
- Produces: `ModelConfigurationError(ValueError)`.
- Produces: `parse_chat_models(value: str | None) -> tuple[ConfiguredModel, ...]`.
- Produces: `validate_model_configuration(models, *, provider_api_keys, gemini_url) -> None`.
- Consumes: no runtime clients and performs no network access.

- [ ] **Step 1: Write parser and validator tests**

Create `tests/test_model_config.py`:

```python
import unittest

from src.model_config import (
    ConfiguredModel,
    ModelConfigurationError,
    parse_chat_models,
    validate_model_configuration,
)


class ChatModelParserTests(unittest.TestCase):
    def test_parses_order_trims_provider_and_preserves_model_text(self):
        models = parse_chat_models(
            " Gemini : Gemini-3.6-Flash , DEEPSEEK : deepseek:reasoner "
        )

        self.assertEqual(
            (
                ConfiguredModel("gemini", "Gemini-3.6-Flash"),
                ConfiguredModel("deepseek", "deepseek:reasoner"),
            ),
            models,
        )

    def test_deduplicates_exact_provider_and_model_pairs_in_order(self):
        models = parse_chat_models(
            "gemini:a,deepseek:b,gemini:a,gemini:A"
        )

        self.assertEqual(
            (
                ConfiguredModel("gemini", "a"),
                ConfiguredModel("deepseek", "b"),
                ConfiguredModel("gemini", "A"),
            ),
            models,
        )

    def test_rejects_missing_or_empty_chain(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ModelConfigurationError,
                    "CHAT_MODELS.*不能为空",
                ):
                    parse_chat_models(value)

    def test_rejects_empty_items_and_bad_item_shapes(self):
        cases = {
            "gemini:a,": "第 2 项为空",
            "gemini:a,,deepseek:b": "第 2 项为空",
            "gemini-a": "第 1 项缺少英文冒号",
            ":a": "第 1 项缺少提供商",
            "gemini:": "第 1 项缺少模型名",
            "openai:gpt": "第 1 项提供商仅支持 gemini 或 deepseek",
        }
        for value, message in cases.items():
            with self.subTest(value=value):
                with self.assertRaisesRegex(ModelConfigurationError, message):
                    parse_chat_models(value)


class ChatModelValidationTests(unittest.TestCase):
    def test_accepts_keys_for_every_referenced_provider(self):
        models = parse_chat_models("gemini:a,deepseek:b")

        validate_model_configuration(
            models,
            provider_api_keys={"gemini": "g-key", "deepseek": "d-key"},
            gemini_url="https://generativelanguage.googleapis.com/v1",
        )

    def test_requires_key_for_every_referenced_provider(self):
        models = parse_chat_models("gemini:a,deepseek:b")

        with self.assertRaisesRegex(
            ModelConfigurationError,
            "DEEPSEEK_API_KEY",
        ):
            validate_model_configuration(
                models,
                provider_api_keys={"gemini": "g-key", "deepseek": " "},
                gemini_url="https://generativelanguage.googleapis.com/v1",
            )

    def test_rejects_non_base_gemini_urls(self):
        models = parse_chat_models("gemini:a")
        invalid_urls = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/openai/chat/completions",
            "https://generativelanguage.googleapis.com/"
            "v1/models/model:generateContent",
        )
        for invalid_url in invalid_urls:
            with self.subTest(invalid_url=invalid_url):
                with self.assertRaisesRegex(
                    ModelConfigurationError,
                    "GEMINI_URL.*基础地址",
                ):
                    validate_model_configuration(
                        models,
                        provider_api_keys={
                            "gemini": "g-key",
                            "deepseek": "",
                        },
                        gemini_url=invalid_url,
                    )

    def test_does_not_put_key_values_in_errors(self):
        secret = "secret-value-that-must-not-appear"
        models = parse_chat_models("deepseek:b")

        with self.assertRaises(ModelConfigurationError) as raised:
            validate_model_configuration(
                models,
                provider_api_keys={"gemini": secret, "deepseek": ""},
                gemini_url="https://generativelanguage.googleapis.com/v1",
            )

        self.assertNotIn(secret, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m unittest tests.test_model_config -v
```

Expected: import failure because `src.model_config` does not exist.

- [ ] **Step 3: Implement the parser and validator**

Create `src/model_config.py`:

```python
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


def parse_chat_models(value: str | None) -> tuple[ConfiguredModel, ...]:
    raw = str(value or "")
    if not raw.strip():
        raise ModelConfigurationError("CHAT_MODELS 不能为空")

    parsed: list[ConfiguredModel] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw.split(","), 1):
        stripped = item.strip()
        if not stripped:
            raise ModelConfigurationError(
                f"CHAT_MODELS 第 {index} 项为空"
            )
        if ":" not in stripped:
            raise ModelConfigurationError(
                f"CHAT_MODELS 第 {index} 项缺少英文冒号"
            )

        provider_text, model_text = stripped.split(":", 1)
        provider = provider_text.strip().lower()
        model = model_text.strip()
        if not provider:
            raise ModelConfigurationError(
                f"CHAT_MODELS 第 {index} 项缺少提供商"
            )
        if not model:
            raise ModelConfigurationError(
                f"CHAT_MODELS 第 {index} 项缺少模型名"
            )
        if provider not in SUPPORTED_PROVIDERS:
            raise ModelConfigurationError(
                f"CHAT_MODELS 第 {index} 项提供商仅支持 gemini 或 deepseek"
            )

        key = (provider, model)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(ConfiguredModel(provider=provider, model=model))

    return tuple(parsed)


def validate_model_configuration(
    models: Sequence[ConfiguredModel],
    *,
    provider_api_keys: Mapping[str, str],
    gemini_url: str,
) -> None:
    referenced = {model.provider for model in models}
    for provider in sorted(referenced):
        if not str(provider_api_keys.get(provider, "")).strip():
            variable = PROVIDER_KEY_VARIABLES[provider]
            raise ModelConfigurationError(
                f"CHAT_MODELS 使用 {provider}，但 {variable} 未配置"
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
```

- [ ] **Step 4: Run parser tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_model_config -v
```

Expected: all parser and validator tests pass.

- [ ] **Step 5: Check and commit the isolated component**

Run:

```powershell
python -m compileall -q src/model_config.py tests/test_model_config.py
git diff --check
git add src/model_config.py tests/test_model_config.py
git diff --cached --check
git commit -m "feat: parse env chat model chain"
```

Expected: one commit containing only the parser and its tests.

---

### Task 2: Integrate Strict Configuration, Startup Failure, and Chain Building

**Files:**
- Modify: `tests/__init__.py`
- Create: `tests/test_model_chain_configuration.py`
- Modify: `tests/test_messaging.py`
- Modify: `src/config.py`
- Modify: `src/services/llm_client.py`
- Modify: `run_bot.py`

**Interfaces:**
- Consumes: Task 1 `parse_chat_models()` and `validate_model_configuration()`.
- Produces: `Config.chat_models: tuple[ConfiguredModel, ...]`.
- Produces: `_build_chain(cfg) -> list[LLMModelSpec]` with exact environment order.
- Produces: `run_bot.main() -> int`.

- [ ] **Step 1: Add deterministic test-environment defaults**

Replace `tests/__init__.py` with:

```python
import os


os.environ["CHAT_MODELS"] = "gemini:test-gemini"
os.environ["GEMINI_API_KEY"] = "test-gemini-key"
```

All full-suite commands from this task onward must use package-aware discovery:

```powershell
python -m unittest discover -s tests -t . -v
```

This imports `tests` before test modules import runtime configuration.

- [ ] **Step 2: Write configuration, chain, and startup tests**

Create `tests/test_model_chain_configuration.py`:

```python
import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import run_bot
from src.config import Config
from src.model_config import ConfiguredModel, ModelConfigurationError
from src.services.llm_client import _build_chain


VALID_ENV = {
    "CHAT_MODELS": "gemini:primary,deepseek:fallback",
    "GEMINI_API_KEY": "g-key",
    "DEEPSEEK_API_KEY": "d-key",
    "GEMINI_URL": "https://generativelanguage.googleapis.com/v1",
}


class ConfiguredChainTests(unittest.TestCase):
    def test_config_exposes_only_the_new_model_chain(self):
        with mock.patch.dict(os.environ, VALID_ENV, clear=True):
            current = Config()

        self.assertEqual(
            (
                ConfiguredModel("gemini", "primary"),
                ConfiguredModel("deepseek", "fallback"),
            ),
            current.chat_models,
        )
        for old_name in (
            "gemini_model",
            "deepseek_model",
            "_llm_provider_compat",
            "llm_primary_provider",
            "llm_primary_model",
            "llm_fallback_1_provider",
            "llm_fallback_1_model",
            "llm_fallback_2_provider",
            "llm_fallback_2_model",
            "llm_fallback_3_provider",
            "llm_fallback_3_model",
        ):
            self.assertFalse(hasattr(current, old_name), old_name)

    def test_chain_builder_preserves_config_order(self):
        cfg = SimpleNamespace(
            chat_models=(
                ConfiguredModel("deepseek", "first"),
                ConfiguredModel("gemini", "second"),
                ConfiguredModel("gemini", "third"),
            )
        )

        chain = _build_chain(cfg)

        self.assertEqual(
            [
                ("deepseek", "first"),
                ("gemini", "second"),
                ("gemini", "third"),
            ],
            [(item.provider, item.model) for item in chain],
        )


class StartupModelConfigurationTests(unittest.TestCase):
    def test_startup_error_is_concise_and_returns_two(self):
        secret = "must-not-leak"
        error = ModelConfigurationError(
            "CHAT_MODELS 使用 gemini，但 GEMINI_API_KEY 未配置"
        )
        stderr = io.StringIO()

        with (
            mock.patch.object(run_bot, "load_application", side_effect=error),
            contextlib.redirect_stderr(stderr),
        ):
            code = run_bot.main()

        output = stderr.getvalue()
        self.assertEqual(2, code)
        self.assertIn("模型配置错误", output)
        self.assertIn("GEMINI_API_KEY", output)
        self.assertNotIn(secret, output)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Update the environment-clearing messaging helper**

In `tests/test_messaging.py`, replace `config_with()` with:

```python
def config_with(value: str | None) -> Config:
    environment = {
        "CHAT_MODELS": "gemini:test-gemini",
        "GEMINI_API_KEY": "test-gemini-key",
    }
    if value is not None:
        environment["MESSAGE_WORKERS"] = value
    with mock.patch.dict(os.environ, environment, clear=True):
        return Config()
```

- [ ] **Step 4: Run focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_model_chain_configuration tests.test_messaging -v
```

Expected failures:

- `Config` has no `chat_models`.
- old model fields still exist.
- `_build_chain()` still expects fixed primary/fallback attributes.
- `run_bot` has no `load_application()` or `main()`.

- [ ] **Step 5: Replace model fields in `Config`**

In `src/config.py`:

1. Import `ConfiguredModel`, `parse_chat_models`, and `validate_model_configuration`.
2. Replace the Gemini, DeepSeek, and old LLM chain fields with:

```python
    # Gemini / Google AI Studio native API
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip()
    )
    gemini_url: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_URL",
            "https://generativelanguage.googleapis.com/v1",
        ).rstrip("/")
    )

    # DeepSeek OpenAI-compatible API
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
    chat_models: tuple[ConfiguredModel, ...] = field(init=False)

    def __post_init__(self) -> None:
        models = parse_chat_models(self._chat_models_raw)
        validate_model_configuration(
            models,
            provider_api_keys={
                "gemini": self.gemini_api_key,
                "deepseek": self.deepseek_api_key,
            },
            gemini_url=self.gemini_url,
        )
        object.__setattr__(self, "chat_models", models)
```

Delete all eleven old model-selection environment reads.

- [ ] **Step 6: Simplify `_build_chain()`**

Replace `_build_chain()` in `src/services/llm_client.py` with:

```python
def _build_chain(cfg=None) -> list[LLMModelSpec]:
    """Build the exact ordered chain validated by Config."""
    if cfg is None:
        cfg = config
    return [
        LLMModelSpec(
            provider=item.provider,
            model=item.model,
            supports_tools=_model_supports_tools(
                item.provider,
                item.model,
            ),
        )
        for item in cfg.chat_models
    ]
```

Do not leave provider fallback inference or a second deduplication pass.

- [ ] **Step 7: Make startup configuration errors user-facing**

Replace `run_bot.py` with:

```python
import sys

from src.model_config import ModelConfigurationError


def load_application():
    from src import main

    return main


def main() -> int:
    try:
        application = load_application()
    except ModelConfigurationError as error:
        print(f"模型配置错误：{error}", file=sys.stderr)
        print(
            "正确格式：CHAT_MODELS=gemini:模型名,deepseek:模型名",
            file=sys.stderr,
        )
        return 2

    application.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run focused and full tests**

Run:

```powershell
python -m unittest tests.test_model_config tests.test_model_chain_configuration tests.test_messaging tests.test_identity_configuration -v
python -m unittest discover -s tests -t . -v
```

Expected: all tests pass. `tests/test_messaging.py` is the only existing test helper that clears the complete environment, so no other test should require production-validation exceptions.

- [ ] **Step 9: Compile, inspect deleted variables, and commit**

Run:

```powershell
python -m compileall -q src tests run_bot.py
rg -n "GEMINI_MODEL|DEEPSEEK_MODEL|LLM_PROVIDER|LLM_PRIMARY_|LLM_FALLBACK_" src run_bot.py
git diff --check
git add src/config.py src/services/llm_client.py src/model_config.py run_bot.py tests/__init__.py tests/test_model_chain_configuration.py tests/test_messaging.py
git diff --cached --check
git commit -m "feat: configure ordered chat model chain"
```

Expected: `rg` reports no old runtime model variables; commit contains only configuration integration and tests.

---

### Task 3: Replace Gemini OpenAI Compatibility with Native `generateContent`

**Files:**
- Create: `tests/test_gemini_native_client.py`
- Replace: `src/services/gemini_client.py`
- Modify: `src/services/llm_types.py`
- Modify: `src/chat/chat_service.py`
- Modify: `src/services/deepseek_client.py`

**Interfaces:**
- Consumes: existing provider-neutral `GeminiClient.chat(messages, model, temperature, max_tokens, tools, tool_choice)`.
- Produces: `ChatResponse(content, tool_calls, provider_context)`; context is temporary and never persisted.
- Uses: existing `try_proxied_post()` and `Config` proxy/timeout fields.
- Preserves: the public `DeepSeekClient.chat()` and `FallbackLLMClient.chat()` signatures.

- [ ] **Step 1: Write native protocol tests**

Create `tests/test_gemini_native_client.py`:

```python
import unittest
from types import SimpleNamespace
from unittest import mock

from src.chat import chat_service
from src.services.deepseek_client import DeepSeekClient
from src.services.gemini_client import GeminiClient
from src.services.llm_types import ChatResponse


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def config():
    return SimpleNamespace(
        gemini_api_key="g-key",
        gemini_url="https://generativelanguage.googleapis.com/v1",
        proxies=None,
        request_timeout=18,
    )


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "搜索网页",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


class GeminiNativeRequestTests(unittest.TestCase):
    def test_uses_native_url_api_key_header_and_generation_config(self):
        response = FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "回答"}]}}
                ]
            }
        )
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ) as post:
            result = GeminiClient(config()).chat(
                [
                    {"role": "system", "content": "系统规则"},
                    {"role": "user", "content": "你好"},
                ],
                model="gemini:model/one",
                temperature=0.25,
                max_tokens=512,
            )

        self.assertEqual("回答", result.content)
        url = post.call_args.args[0]
        kwargs = post.call_args.kwargs
        self.assertEqual(
            "https://generativelanguage.googleapis.com/v1/"
            "models/gemini%3Amodel%2Fone:generateContent",
            url,
        )
        self.assertEqual("g-key", kwargs["headers"]["x-goog-api-key"])
        self.assertNotIn("Authorization", kwargs["headers"])
        payload = kwargs["json"]
        self.assertNotIn("model", payload)
        self.assertNotIn("messages", payload)
        self.assertEqual(
            {"parts": [{"text": "系统规则"}]},
            payload["systemInstruction"],
        )
        self.assertEqual(
            {"temperature": 0.25, "maxOutputTokens": 512},
            payload["generationConfig"],
        )

    def test_converts_image_and_tool_declaration(self):
        response = FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "id": "remote-call-1",
                                        "name": "search_web",
                                        "args": {"query": "测试"},
                                    },
                                    "thoughtSignature": "signature-1",
                                }
                            ]
                        }
                    }
                ]
            }
        )
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ) as post:
            result = GeminiClient(config()).chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "看图"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,cG5n"
                                },
                            },
                        ],
                    }
                ],
                model="vision",
                tools=[SEARCH_TOOL],
                tool_choice="auto",
            )

        payload = post.call_args.kwargs["json"]
        parts = payload["contents"][0]["parts"]
        self.assertEqual({"text": "看图"}, parts[0])
        self.assertEqual(
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": "cG5n",
                }
            },
            parts[1],
        )
        declaration = payload["tools"][0]["functionDeclarations"][0]
        self.assertEqual("search_web", declaration["name"])
        self.assertEqual(
            "AUTO",
            payload["toolConfig"]["functionCallingConfig"]["mode"],
        )
        self.assertEqual("search_web", result.tool_calls[0]["function"]["name"])
        self.assertEqual(
            '{"query":"测试"}',
            result.tool_calls[0]["function"]["arguments"],
        )
        self.assertEqual("remote-call-1", result.tool_calls[0]["id"])
        self.assertEqual(
            "signature-1",
            result.provider_context["content"]["parts"][0][
                "thoughtSignature"
            ],
        )

    def test_returns_raw_function_call_content_with_signature_and_id(self):
        native_content = {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "id": "remote-call-1",
                        "name": "search_web",
                        "args": {"query": "测试"},
                    },
                    "thoughtSignature": "signature-1",
                }
            ],
        }
        first_response = FakeResponse(
            {"candidates": [{"content": native_content}]}
        )
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=first_response,
        ):
            result = GeminiClient(config()).chat(
                [{"role": "user", "content": "查一下"}],
                model="gemini-test",
                tools=[SEARCH_TOOL],
                tool_choice="auto",
            )

        self.assertEqual(
            {"provider": "gemini", "content": native_content},
            result.provider_context,
        )

    def test_converts_signed_tool_round_trip_messages(self):
        response = FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "整理结果"}]}}
                ]
            }
        )
        messages = [
            {"role": "user", "content": "查一下"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"测试"}',
                        },
                    }
                ],
                "_provider_context": {
                    "provider": "gemini",
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "remote-call-1",
                                    "name": "search_web",
                                    "args": {"query": "测试"},
                                },
                                "thoughtSignature": "signature-1",
                            }
                        ],
                    },
                },
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search_web",
                "content": "搜索结果",
            },
        ]
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ) as post:
            GeminiClient(config()).chat(messages, model="gemini-test")

        contents = post.call_args.kwargs["json"]["contents"]
        self.assertEqual(
            {
                "functionCall": {
                    "id": "remote-call-1",
                    "name": "search_web",
                    "args": {"query": "测试"},
                },
                "thoughtSignature": "signature-1",
            },
            contents[1]["parts"][0],
        )
        self.assertEqual(
            {
                "functionResponse": {
                    "id": "remote-call-1",
                    "name": "search_web",
                    "response": {"result": "搜索结果"},
                }
            },
            contents[2]["parts"][0],
        )

    def test_preserves_parallel_call_order_ids_and_signature_placement(self):
        response = FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "整理结果"}]}}
                ]
            }
        )
        native_parts = [
            {
                "functionCall": {
                    "id": "remote-1",
                    "name": "search_web",
                    "args": {"query": "一"},
                },
                "thoughtSignature": "parallel-signature",
            },
            {
                "functionCall": {
                    "id": "remote-2",
                    "name": "search_web",
                    "args": {"query": "二"},
                }
            },
        ]
        calls = [
            {
                "id": "local-1",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": '{"query":"一"}',
                },
            },
            {
                "id": "local-2",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": '{"query":"二"}',
                },
            },
        ]
        messages = [
            {"role": "user", "content": "查两项"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": calls,
                "_provider_context": {
                    "provider": "gemini",
                    "content": {
                        "role": "model",
                        "parts": native_parts,
                    },
                },
            },
            {
                "role": "tool",
                "tool_call_id": "local-1",
                "name": "search_web",
                "content": "结果一",
            },
            {
                "role": "tool",
                "tool_call_id": "local-2",
                "name": "search_web",
                "content": "结果二",
            },
        ]

        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ) as post:
            GeminiClient(config()).chat(messages, model="gemini-test")

        contents = post.call_args.kwargs["json"]["contents"]
        self.assertEqual(native_parts, contents[1]["parts"])
        self.assertEqual(
            ["remote-1", "remote-2"],
            [
                part["functionResponse"]["id"]
                for part in contents[2]["parts"]
            ],
        )
        self.assertNotIn(
            "thoughtSignature",
            contents[1]["parts"][1],
        )

    def test_cross_provider_tool_call_gets_official_dummy_signature(self):
        response = FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "整理结果"}]}}
                ]
            }
        )
        messages = [
            {"role": "user", "content": "查一下"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "deepseek-call",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"测试"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "deepseek-call",
                "name": "search_web",
                "content": "搜索结果",
            },
        ]
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ) as post:
            GeminiClient(config()).chat(messages, model="gemini-test")

        function_part = post.call_args.kwargs["json"]["contents"][1][
            "parts"
        ][0]
        self.assertEqual(
            "skip_thought_signature_validator",
            function_part["thoughtSignature"],
        )


class GeminiNativeResponseTests(unittest.TestCase):
    def test_joins_text_parts_and_parses_multiple_calls(self):
        response = FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "第一段"},
                                {"text": "第二段"},
                                {
                                    "functionCall": {
                                        "name": "search_web",
                                        "args": {"query": "一"},
                                    }
                                },
                                {
                                    "functionCall": {
                                        "name": "search_web",
                                        "args": {"query": "二"},
                                    }
                                },
                            ]
                        }
                    }
                ]
            }
        )
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ):
            result = GeminiClient(config()).chat(
                [{"role": "user", "content": "测试"}],
                model="gemini-test",
            )

        self.assertEqual("第一段\n第二段", result.content)
        self.assertEqual(2, len(result.tool_calls))
        self.assertNotEqual(result.tool_calls[0]["id"], result.tool_calls[1]["id"])

    def test_rejects_empty_candidates(self):
        with (
            mock.patch(
                "src.services.gemini_client.try_proxied_post",
                return_value=FakeResponse({"candidates": []}),
            ),
            self.assertRaisesRegex(RuntimeError, "Gemini.*候选"),
        ):
            GeminiClient(config()).chat(
                [{"role": "user", "content": "测试"}],
                model="gemini-test",
            )


class ProviderContextBoundaryTests(unittest.TestCase):
    def test_chat_tool_messages_keep_context_only_on_temporary_assistant(self):
        context = {
            "provider": "gemini",
            "content": {"role": "model", "parts": []},
        }
        calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": '{"query":"测试"}',
                },
            }
        ]

        with mock.patch.object(
            chat_service,
            "run_tool",
            return_value="搜索结果",
        ):
            messages = chat_service.build_tool_messages(
                calls,
                "测试",
                provider_context=context,
            )

        self.assertEqual(context, messages[0]["_provider_context"])
        self.assertNotIn("_provider_context", messages[1])

    def test_deepseek_removes_private_provider_context(self):
        cfg = SimpleNamespace(
            deepseek_api_key="d-key",
            deepseek_url="https://api.deepseek.com/chat/completions",
            proxies=None,
            request_timeout=18,
        )
        response = FakeResponse(
            {
                "choices": [
                    {"message": {"content": "回答", "tool_calls": []}}
                ]
            }
        )
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [],
                "_provider_context": {
                    "provider": "gemini",
                    "content": {"parts": []},
                },
            }
        ]

        with mock.patch(
            "src.services.deepseek_client.try_proxied_post",
            return_value=response,
        ) as post:
            result = DeepSeekClient(cfg).chat(
                messages,
                model="deepseek-test",
            )

        self.assertEqual("回答", result.content)
        sent_message = post.call_args.kwargs["json"]["messages"][0]
        self.assertNotIn("_provider_context", sent_message)

    def test_string_response_normalization_has_no_provider_context(self):
        normalized = chat_service.normalize_chat_response("回答")

        self.assertEqual(
            ChatResponse(content="回答", provider_context=None),
            normalized,
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run native client tests and verify RED**

Run:

```powershell
python -m unittest tests.test_gemini_native_client -v
```

Expected: failures showing the client still sends the OpenAI URL, Bearer header, `messages`, and parses `choices`.

- [ ] **Step 3: Add short-lived provider context to the tool loop**

In `src/services/llm_types.py`, extend `ChatResponse`:

```python
@dataclass
class ChatResponse:
    """Unified response plus optional current-turn provider protocol state."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    provider_context: dict[str, Any] | None = None
```

In `src/chat/chat_service.py`:

1. Add `import copy`.
2. Replace `build_tool_messages()` with:

```python
def build_tool_messages(
    tool_calls: list[dict[str, Any]],
    fallback_query: str,
    provider_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }
    if provider_context is not None:
        assistant_message["_provider_context"] = copy.deepcopy(
            provider_context
        )

    messages: list[dict[str, Any]] = [assistant_message]
    for index, tool_call in enumerate(tool_calls, 1):
        name = tool_function_name(tool_call)
        query = normalize_tool_query(
            name,
            tool_call_query(tool_call, ""),
            fallback_query,
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(
                    tool_call.get("id") or f"{name}_{index}"
                ),
                "name": name,
                "content": run_tool(name, query),
            }
        )
    return messages
```

3. In `generate_reply()`, pass the context from the same response:

```python
            messages.extend(
                build_tool_messages(
                    tool_calls,
                    text or "图片内容",
                    provider_context=response.provider_context,
                )
            )
```

Do not add `_provider_context` to `append_history()` or any storage function.

In `src/services/deepseek_client.py`, sanitize only the private provider field before building the payload:

```python
        model_name = str(model or "").strip()
        if not model_name:
            raise RuntimeError("DeepSeek model is not configured")
        clean_messages = [
            {
                key: value
                for key, value in message.items()
                if key != "_provider_context"
            }
            for message in messages
        ]
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": clean_messages,
            "temperature": temperature,
        }
```

Keep the remaining DeepSeek request and response behavior unchanged.

- [ ] **Step 4: Replace `GeminiClient` with the native adapter**

Replace `src/services/gemini_client.py` with:

```python
"""Gemini Developer API client using native generateContent REST."""

from __future__ import annotations

import base64
import copy
import json
from typing import Any
from urllib.parse import quote

from src.config import Config
from src.services.llm_types import ChatResponse
from src.util import try_proxied_post


DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def _append_content(
    contents: list[dict[str, Any]],
    role: str,
    parts: list[dict[str, Any]],
) -> None:
    if not parts:
        return
    if contents and contents[-1]["role"] == role:
        contents[-1]["parts"].extend(parts)
        return
    contents.append({"role": role, "parts": parts})


def _data_url_part(url: str) -> dict[str, Any]:
    header, separator, encoded = str(url or "").partition(",")
    if (
        not separator
        or not header.startswith("data:")
        or ";base64" not in header.casefold()
    ):
        raise RuntimeError("Gemini 图片必须是 base64 data URL")
    mime_type = header[5:].split(";", 1)[0].strip().lower()
    if not mime_type.startswith("image/"):
        raise RuntimeError("Gemini 图片 data URL 缺少有效图片 MIME 类型")
    try:
        base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise RuntimeError("Gemini 图片 base64 数据无效") from error
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": encoded,
        }
    }


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = str(item.get("text") or "")
            if text:
                parts.append({"text": text})
        elif item_type == "image_url":
            image = item.get("image_url")
            url = image.get("url") if isinstance(image, dict) else ""
            parts.append(_data_url_part(str(url or "")))
    return parts


def _function_call_part(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise RuntimeError("内部工具调用缺少 function")
    name = str(function.get("name") or "").strip()
    arguments = function.get("arguments", "{}")
    try:
        parsed_arguments = (
            json.loads(arguments) if isinstance(arguments, str) else arguments
        )
    except json.JSONDecodeError as error:
        raise RuntimeError("内部工具调用参数不是有效 JSON") from error
    if not name or not isinstance(parsed_arguments, dict):
        raise RuntimeError("内部工具调用名称或参数无效")
    return {
        "functionCall": {
            "name": name,
            "args": parsed_arguments,
        }
    }


def _native_messages(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system_parts: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []
    native_call_ids: dict[str, str] = {}

    for message in messages:
        role = str(message.get("role") or "")
        if role == "system":
            system_parts.extend(_content_parts(message.get("content")))
            continue
        if role == "tool":
            name = str(message.get("name") or "").strip()
            if not name:
                raise RuntimeError("内部工具结果缺少名称")
            function_response: dict[str, Any] = {
                "name": name,
                "response": {
                    "result": str(message.get("content") or "")
                },
            }
            internal_id = str(message.get("tool_call_id") or "")
            native_id = native_call_ids.get(internal_id)
            if native_id:
                function_response["id"] = native_id
            _append_content(
                contents,
                "user",
                [
                    {
                        "functionResponse": function_response
                    }
                ],
            )
            continue

        native_role = "model" if role == "assistant" else "user"
        parts = _content_parts(message.get("content"))
        if role == "assistant":
            raw_calls = message.get("tool_calls") or []
            provider_context = message.get("_provider_context")
            native_content = (
                provider_context.get("content")
                if (
                    isinstance(provider_context, dict)
                    and provider_context.get("provider") == "gemini"
                )
                else None
            )
            if isinstance(native_content, dict):
                native_parts = native_content.get("parts")
                if not isinstance(native_parts, list):
                    raise RuntimeError(
                        "Gemini provider context 缺少原始 parts"
                    )
                internal_calls = (
                    raw_calls if isinstance(raw_calls, list) else []
                )
                returned_calls = [
                    part.get("functionCall")
                    for part in native_parts
                    if (
                        isinstance(part, dict)
                        and isinstance(part.get("functionCall"), dict)
                    )
                ]
                for internal_call, returned_call in zip(
                    internal_calls,
                    returned_calls,
                ):
                    internal_id = str(
                        internal_call.get("id") or ""
                    )
                    returned_id = str(
                        returned_call.get("id") or ""
                    )
                    if internal_id and returned_id:
                        native_call_ids[internal_id] = returned_id
                contents.append(copy.deepcopy(native_content))
                continue

            if isinstance(raw_calls, list):
                call_parts = [
                    _function_call_part(call)
                    for call in raw_calls
                    if isinstance(call, dict)
                ]
                if call_parts:
                    call_parts[0][
                        "thoughtSignature"
                    ] = DUMMY_THOUGHT_SIGNATURE
                    parts.extend(call_parts)
        _append_content(contents, native_role, parts)

    system_instruction = (
        {"parts": system_parts} if system_parts else None
    )
    return system_instruction, contents


def _native_tools(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        declaration = {
            "name": name,
            "description": str(function.get("description") or ""),
            "parameters": function.get("parameters") or {
                "type": "object",
                "properties": {},
            },
        }
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}] if declarations else []


def _native_tool_config(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if tool_choice is None or tool_choice == "auto":
        mode = "AUTO"
        allowed_names: list[str] = []
    elif tool_choice == "none":
        mode = "NONE"
        allowed_names = []
    elif tool_choice in {"required", "any"}:
        mode = "ANY"
        allowed_names = []
    elif isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        name = (
            str(function.get("name") or "").strip()
            if isinstance(function, dict)
            else ""
        )
        mode = "ANY"
        allowed_names = [name] if name else []
    else:
        raise RuntimeError("Gemini 不支持当前 tool_choice")

    config: dict[str, Any] = {"mode": mode}
    if allowed_names:
        config["allowedFunctionNames"] = allowed_names
    return {"functionCallingConfig": config}


def _parse_response(data: dict[str, Any]) -> ChatResponse:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        feedback = data.get("promptFeedback")
        reason = (
            str(feedback.get("blockReason") or "")
            if isinstance(feedback, dict)
            else ""
        )
        suffix = f"：{reason}" if reason else ""
        raise RuntimeError(f"Gemini 未返回候选结果{suffix}")

    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise RuntimeError("Gemini 候选结果缺少 content.parts")

    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for index, part in enumerate(parts, 1):
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
        function_call = part.get("functionCall")
        if not isinstance(function_call, dict):
            continue
        name = str(function_call.get("name") or "").strip()
        arguments = function_call.get("args")
        if not name or not isinstance(arguments, dict):
            continue
        call_id = str(
            function_call.get("id") or f"gemini_call_{index}"
        )
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        )

    result = ChatResponse(
        content="\n".join(texts).strip(),
        tool_calls=tool_calls,
        provider_context={
            "provider": "gemini",
            "content": copy.deepcopy(content),
        },
    )
    if not result.content and not result.tool_calls:
        raise RuntimeError("Gemini 返回空内容且没有函数调用")
    return result


class GeminiClient:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

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
            raise RuntimeError("GEMINI_API_KEY is not configured")
        model_name = str(model or "").strip()
        if not model_name:
            raise RuntimeError("Gemini model is not configured")

        system_instruction, contents = _native_messages(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        native_tools = _native_tools(tools)
        if native_tools:
            payload["tools"] = native_tools
            payload["toolConfig"] = _native_tool_config(tool_choice)

        encoded_model = quote(model_name, safe="")
        url = (
            f"{self._cfg.gemini_url.rstrip('/')}/models/"
            f"{encoded_model}:generateContent"
        )
        response = try_proxied_post(
            url,
            proxies=self._cfg.proxies,
            json=payload,
            headers={
                "x-goog-api-key": self._cfg.gemini_api_key,
                "Content-Type": "application/json",
            },
            timeout=self._cfg.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Gemini 返回的 JSON 不是对象")
        return _parse_response(data)
```

- [ ] **Step 5: Run native tests and fix only protocol mismatches**

Run:

```powershell
python -m unittest tests.test_gemini_native_client -v
```

Expected: all native client tests pass. Preserve the provider-context boundary and DeepSeek sanitizer established in Step 3 while fixing protocol mismatches.

- [ ] **Step 6: Run provider and image regression tests**

Run:

```powershell
python -m unittest tests.test_gemini_native_client tests.test_llm_image_fallback tests.test_multimodal_chat tests.test_identity_configuration -v
```

Expected: all tests pass; image fallback and identity prompts remain provider-neutral.

- [ ] **Step 7: Compile and commit**

Run:

```powershell
python -m compileall -q src/services tests/test_gemini_native_client.py src/chat/chat_service.py
git diff --check
git add src/services/gemini_client.py src/services/deepseek_client.py src/services/llm_types.py src/chat/chat_service.py tests/test_gemini_native_client.py
git diff --cached --check
git commit -m "feat: use native Gemini generateContent"
```

Expected: the native Gemini adapter, temporary provider-context bridge, DeepSeek sanitizer, and their direct tests are committed.

---

### Task 4: Add Integration Guardrails and Health Visibility

**Files:**
- Create: `tests/test_health.py`
- Modify: `tests/test_multimodal_chat.py`
- Modify: `tests/test_product_scope.py`
- Modify: `src/main.py`

**Interfaces:**
- Consumes: `config.chat_models`.
- Produces: `/health` field `chat_models: list[{"provider": str, "model": str}]`.
- Adds no secrets and no remote state IDs.

- [ ] **Step 1: Add health and product-boundary tests**

Create `tests/test_health.py`:

```python
import unittest
from types import SimpleNamespace
from unittest import mock

import src.main as main
from src.model_config import ConfiguredModel


class HealthModelChainTests(unittest.TestCase):
    def test_health_lists_models_without_secrets(self):
        fake_config = SimpleNamespace(
            bot_name="qqbot",
            gemini_api_key="secret-g",
            deepseek_api_key="secret-d",
            onebot_url="http://127.0.0.1:3000",
            require_group_at=True,
            chat_models=(
                ConfiguredModel("gemini", "gemini-test"),
                ConfiguredModel("deepseek", "deepseek-test"),
            ),
        )

        with mock.patch.object(main, "config", fake_config):
            response = main.health()

        self.assertEqual(
            [
                {"provider": "gemini", "model": "gemini-test"},
                {"provider": "deepseek", "model": "deepseek-test"},
            ],
            response["chat_models"],
        )
        serialized = str(response)
        self.assertNotIn("secret-g", serialized)
        self.assertNotIn("secret-d", serialized)
        self.assertNotIn("api_key", serialized.casefold())


if __name__ == "__main__":
    unittest.main()
```

In `tests/test_product_scope.py`, add:

```python
    def test_gemini_runtime_uses_native_stateless_generate_content(self):
        source = (
            ROOT / "src" / "services" / "gemini_client.py"
        ).read_text(encoding="utf-8")

        self.assertIn(":generateContent", source)
        self.assertIn("x-goog-api-key", source)
        self.assertNotIn("/openai/chat/completions", source)
        self.assertNotIn("previous_interaction_id", source)
        self.assertNotIn("interactions.create", source)
```

Rename the stale method in `tests/test_multimodal_chat.py`:

```python
    def test_builds_provider_neutral_multimodal_content(self):
```

Keep its assertions unchanged because the internal chat contract remains provider-neutral and the Gemini adapter owns native conversion.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_health tests.test_product_scope tests.test_multimodal_chat -v
```

Expected: health test fails because `chat_models` is absent; native product test passes only after Task 3; renamed multimodal test remains green.

- [ ] **Step 3: Add non-secret model-chain health output**

In `src/main.py`, add to the `health()` result:

```python
        "chat_models": [
            {"provider": item.provider, "model": item.model}
            for item in config.chat_models
        ],
```

Do not add raw environment values, API Keys, URLs containing query strings, or remote IDs.

- [ ] **Step 4: Run integration and full tests**

Run:

```powershell
python -m unittest tests.test_health tests.test_product_scope tests.test_multimodal_chat tests.test_main_image_flow tests.test_llm_image_fallback -v
python -m unittest discover -s tests -t . -v
```

Expected: all tests pass.

- [ ] **Step 5: Compile and commit**

Run:

```powershell
python -m compileall -q src tests
git diff --check
git add src/main.py tests/test_health.py tests/test_product_scope.py tests/test_multimodal_chat.py
git diff --cached --check
git commit -m "test: guard native Gemini model integration"
```

Expected: integration health and regression guardrails committed separately from docs.

---

### Task 5: Migrate the Safe Template and README

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/test_readme_guide.py`
- Verify: `tests/test_qqbot_branding.py`

**Interfaces:**
- Documents: `CHAT_MODELS=provider:model[,provider:model...]`.
- Documents: `GEMINI_URL` as a base URL.
- Removes: all eleven old model-selection variables.
- Does not touch: real `.env`.

- [ ] **Step 1: Add documentation regression assertions**

In `tests/test_readme_guide.py`, add:

```python
    def test_readme_documents_chat_models_and_native_gemini(self):
        reference = readme_section(
            self.readme,
            "完整 `.env` 参数参考",
        )
        self.assertIn("`CHAT_MODELS`", reference)
        self.assertIn(
            "https://generativelanguage.googleapis.com/v1",
            self.readme,
        )
        self.assertIn("generateContent", self.readme)
        self.assertIn(
            "CHAT_MODELS=gemini:",
            self.readme,
        )

    def test_operator_docs_remove_old_model_variables(self):
        env_example = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
        combined = self.readme + "\n" + env_example
        for old_name in (
            "GEMINI_MODEL",
            "DEEPSEEK_MODEL",
            "LLM_PROVIDER",
            "LLM_PRIMARY_PROVIDER",
            "LLM_PRIMARY_MODEL",
            "LLM_FALLBACK_1_PROVIDER",
            "LLM_FALLBACK_1_MODEL",
            "LLM_FALLBACK_2_PROVIDER",
            "LLM_FALLBACK_2_MODEL",
            "LLM_FALLBACK_3_PROVIDER",
            "LLM_FALLBACK_3_MODEL",
        ):
            with self.subTest(old_name=old_name):
                self.assertNotIn(old_name, combined)
```

- [ ] **Step 2: Run documentation tests and verify RED**

Run:

```powershell
python -m unittest tests.test_readme_guide -v
```

Expected failures:

- runtime/template environment sets differ;
- README has no `CHAT_MODELS` row;
- old model variables remain;
- native Gemini base URL and `generateContent` are not documented.

- [ ] **Step 3: Replace the model section in `.env.example`**

Use exactly:

```dotenv
# 对话模型链：第一个是主模型，后续按顺序回退。
# 格式：提供商:模型名，支持 gemini 和 deepseek。
CHAT_MODELS=gemini:gemini-3.1-flash-lite

# Gemini 原生 generateContent API
GEMINI_API_KEY=
GEMINI_URL=https://generativelanguage.googleapis.com/v1

# DeepSeek OpenAI 兼容 API
DEEPSEEK_API_KEY=
DEEPSEEK_URL=https://api.deepseek.com/chat/completions
```

Delete the eleven old assignments rather than commenting them out.

- [ ] **Step 4: Update README configuration examples**

Use these exact minimum patterns:

Gemini only:

```dotenv
CHAT_MODELS=gemini:填写账号可用的模型名
GEMINI_API_KEY=填写你的_Gemini_Key
GEMINI_URL=https://generativelanguage.googleapis.com/v1
```

Gemini with DeepSeek fallback:

```dotenv
CHAT_MODELS=gemini:填写主模型名,deepseek:填写回退模型名
GEMINI_API_KEY=填写你的_Gemini_Key
DEEPSEEK_API_KEY=填写你的_DeepSeek_Key
```

DeepSeek only:

```dotenv
CHAT_MODELS=deepseek:填写账号可用的模型名
DEEPSEEK_API_KEY=填写你的_DeepSeek_Key
```

Add one `CHAT_MODELS` row to the complete parameter table:

```markdown
| `CHAT_MODELS` | 必需 | 无 | 对话模型链，格式为 `提供商:模型名`；第一个是主模型，后续依次回退。支持 `gemini` 和 `deepseek`，列出的每个提供商都必须配置对应 API Key。 |
```

Replace the `GEMINI_URL` row with:

```markdown
| `GEMINI_URL` | 可选 | `https://generativelanguage.googleapis.com/v1` | Gemini Developer API 基础地址；客户端会追加 `/models/{model}:generateContent`。不要填写旧 `/openai/chat/completions` 地址。 |
```

Remove all eleven old parameter rows and replace the old primary/fallback explanation with:

```markdown
模型按照 `CHAT_MODELS` 从左到右尝试。重复的“提供商 + 模型名”只保留第一次；格式错误、未知提供商或对应 Key 缺失时，机器人会在启动阶段停止并给出中文错误。模型名是否存在及是否支持工具或图片由实际 API 响应决定。
```

Add troubleshooting entries for:

```markdown
### `CHAT_MODELS` 配置错误

确认使用英文逗号分隔模型，并使用第一个英文冒号分隔提供商和模型名，例如 `gemini:模型名,deepseek:模型名`。不支持 Gemini、DeepSeek 之外的提供商。

### Gemini 返回 404 或模型不存在

确认 `GEMINI_URL=https://generativelanguage.googleapis.com/v1`，并确认 `CHAT_MODELS` 中的 Gemini 模型名确实对当前账号开放。不要把旧 OpenAI 兼容端点填入 `GEMINI_URL`。
```

State explicitly that Gemini uses native stateless `generateContent`, while DeepSeek keeps its OpenAI-compatible endpoint, and that both consume the same local history.

Replace any README full-suite command with the package-aware form used by this plan:

```powershell
python -m unittest discover -s tests -t . -v
```

- [ ] **Step 5: Run docs and product tests**

Run:

```powershell
python -m unittest tests.test_readme_guide tests.test_qqbot_branding tests.test_product_scope -v
python -c "from dotenv import dotenv_values; values=dotenv_values('.env.example'); print(len(values), len(set(values)))"
```

Expected:

- all tests pass;
- dotenv reports equal total and unique counts with no parse warning;
- source-derived environment-variable equality passes without a hardcoded count.

- [ ] **Step 6: Verify old names are absent from active operator surfaces**

Run:

```powershell
rg -n "GEMINI_MODEL|DEEPSEEK_MODEL|LLM_PROVIDER|LLM_PRIMARY_|LLM_FALLBACK_" src README.md .env.example
rg -n "v1beta/openai/chat/completions" src/services/gemini_client.py .env.example
rg -n "Authorization.*Bearer" src/services/gemini_client.py
git status --short
git ls-files .env
```

Expected:

- all three `rg` commands have no matches on the scanned Gemini runtime and
  template surfaces;
- `.env` is not tracked or staged;
- only intended feature files are changed.

- [ ] **Step 7: Commit documentation migration**

Run:

```powershell
git diff --check
git add README.md .env.example tests/test_readme_guide.py
git diff --cached --check
git diff --cached --name-only
git commit -m "docs: document configurable chat model chain"
```

Expected staged names are exactly `.env.example`, `README.md`, and `tests/test_readme_guide.py`.

---

### Task 6: Full Verification, Review, and Handoff

**Files:**
- Verify: every file changed by Tasks 1–5.
- Preserve: real `.env` and local data.

**Interfaces:**
- Final behavior: strict `CHAT_MODELS`, native Gemini, unchanged DeepSeek contract, local history, ordered fallback.

- [ ] **Step 1: Run the complete test suite**

Run:

```powershell
python -m unittest discover -s tests -t . -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Run compilation and patch validation**

Run:

```powershell
python -m compileall -q src tests run_bot.py
git diff --check origin/main..HEAD
git status --short --branch
git ls-files .env
git check-ignore -v .env
```

Expected:

- compilation succeeds;
- diff check is clean;
- `.env` is ignored and untracked;
- no unrelated working-tree changes exist.

- [ ] **Step 3: Independently validate the model configuration surfaces**

Run:

```powershell
python -m unittest tests.test_model_config tests.test_model_chain_configuration tests.test_gemini_native_client tests.test_health tests.test_readme_guide -v
python -c "from dotenv import dotenv_values; values=dotenv_values('.env.example'); print('keys', len(values), 'unique', len(set(values)))"
rg -n "GEMINI_MODEL|DEEPSEEK_MODEL|LLM_PROVIDER|LLM_PRIMARY_|LLM_FALLBACK_" src README.md .env.example
rg -n "v1beta/openai|previous_interaction_id|interactions.create" src README.md .env.example
```

Expected:

- focused tests pass;
- dotenv totals are equal;
- both legacy scans return no matches.

- [ ] **Step 4: Request final review**

The reviewer must inspect the complete feature diff and verify:

```text
- CHAT_MODELS parsing is strict, ordered, deduplicated, and provider-limited.
- every referenced provider requires its Key without leaking values;
- no legacy model-selection variable is read by runtime or presented to users;
- Gemini URL, x-goog-api-key, contents, systemInstruction, inlineData,
  functionDeclarations, functionCall, and functionResponse match native REST;
- Gemini never sends OpenAI messages/choices payloads or uses Interactions state;
- DeepSeek keeps its existing OpenAI-compatible request/response behavior
  except for removing the private `_provider_context` field before requests;
- Gemini tool-call round trips preserve the raw Part order, original call IDs,
  and exact `thoughtSignature` placement;
- cross-provider DeepSeek-to-Gemini tool context uses the documented dummy
  signature only when no native Gemini signature exists;
- provider context is neither persisted nor leaked into DeepSeek payloads;
- local history, search tool loop, images, fallback, memory, concurrency, and reset remain intact;
- /health contains model names but no Keys;
- real .env and local data are absent from commits;
- no Critical or Important issue remains.
```

- [ ] **Step 5: Apply review fixes and rerun all verification**

For every Critical or Important finding:

1. add or adjust a focused failing test;
2. implement the smallest correction;
3. rerun the focused test;
4. request re-review;
5. rerun Steps 1–3 after approval.

Do not broaden the product beyond the confirmed design.

- [ ] **Step 6: Finish the branch**

Use `superpowers:finishing-a-development-branch`. Do not push, merge, delete a branch, or modify the real `.env` without the user selecting or explicitly authorizing that action.
