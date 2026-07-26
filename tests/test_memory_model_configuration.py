import os
import unittest
from types import SimpleNamespace
from unittest import mock

from src.config import Config
from src.model_config import ModelConfigurationError, parse_model_chain
from src.services import llm_client


class MemoryModelConfigurationTests(unittest.TestCase):
    def test_blank_memory_models_reuses_chat_chain(self):
        with mock.patch.dict(os.environ, {
            "CHAT_MODELS": "gemini:chat-main,deepseek:chat-fallback",
            "MEMORY_MODELS": "",
            "GEMINI_API_KEY": "g",
            "DEEPSEEK_API_KEY": "d",
        }, clear=False):
            current = Config()

        self.assertEqual(current.chat_models, current.memory_models)

    def test_explicit_memory_chain_is_independent(self):
        with mock.patch.dict(os.environ, {
            "CHAT_MODELS": "gemini:chat-main",
            "MEMORY_MODELS": "deepseek:memory-cheap",
            "GEMINI_API_KEY": "g",
            "DEEPSEEK_API_KEY": "d",
        }, clear=False):
            current = Config()

        self.assertEqual("memory-cheap", current.memory_models[0].model)
        self.assertEqual("chat-main", current.chat_models[0].model)

    def test_memory_chain_key_error_names_memory_models(self):
        with mock.patch.dict(os.environ, {
            "CHAT_MODELS": "gemini:chat-main",
            "MEMORY_MODELS": "deepseek:memory-cheap",
            "GEMINI_API_KEY": "g",
            "DEEPSEEK_API_KEY": "",
        }, clear=False):
            with self.assertRaisesRegex(
                ModelConfigurationError,
                "MEMORY_MODELS.*DEEPSEEK_API_KEY",
            ):
                Config()

    def test_memory_parser_error_names_memory_models(self):
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "MEMORY_MODELS.*提供商仅支持",
        ):
            parse_model_chain("openai:gpt", "MEMORY_MODELS")

    def test_memory_client_uses_a_separate_cached_memory_chain(self):
        cfg = SimpleNamespace(
            chat_models=(
                SimpleNamespace(provider="gemini", model="chat-main"),
            ),
            memory_models=(
                SimpleNamespace(provider="deepseek", model="memory-cheap"),
            ),
        )
        with (
            mock.patch.object(llm_client, "config", cfg),
            mock.patch.object(llm_client, "_llm_client", None),
            mock.patch.object(llm_client, "_memory_llm_client", None),
        ):
            chat_client = llm_client.get_llm_client()
            memory_client = llm_client.get_memory_llm_client()

        self.assertIsNot(chat_client, memory_client)
        self.assertEqual("chat-main", chat_client._chain[0].model)
        self.assertEqual("memory-cheap", memory_client._chain[0].model)


if __name__ == "__main__":
    unittest.main()
