import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import run_bot
from src.config import Config
from src.model_config import ConfiguredModel, ModelConfigurationError
from src.services.llm_client import _build_chain, _model_supports_tools


VALID_ENV = {
    "CHAT_MODELS": "gemini:primary,deepseek:fallback",
    "GEMINI_API_KEY": "g-key",
    "DEEPSEEK_API_KEY": "d-key",
    "GEMINI_URL": "https://generativelanguage.googleapis.com/v1",
}


class ConfiguredChainTests(unittest.TestCase):
    def test_tool_support_is_not_inferred_from_model_name_substrings(self):
        self.assertTrue(
            _model_supports_tools(
                "deepseek",
                "custom-reasoner-with-tools",
            )
        )
        self.assertTrue(
            _model_supports_tools(
                "deepseek",
                "custom-r1-with-tools",
            )
        )
        self.assertFalse(
            _model_supports_tools(
                "gemini",
                "gemma-4-26b-a4b-it",
            )
        )

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
