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
