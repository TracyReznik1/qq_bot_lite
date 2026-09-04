import os
import unittest
from unittest import mock

from src.config import Config
from src.services import llm_client
from src.services.llm_types import LLMModelSpec


class MemoryApiKeyIsolationTests(unittest.TestCase):
    def test_gemini_api_key_isolation_when_configured(self):
        with mock.patch.dict(os.environ, {
            "CHAT_MODELS": "gemini:gemini-2.5-flash",
            "MEMORY_MODELS": "gemini:gemini-2.0-flash-lite",
            "GEMINI_API_KEY": "chat-key-1",
            "MEMORY_GEMINI_API_KEY": "memory-key-2",
        }, clear=False):
            cfg = Config()

        with (
            mock.patch.object(llm_client, "config", cfg),
            mock.patch.object(llm_client, "_llm_client", None),
            mock.patch.object(llm_client, "_memory_llm_client", None),
        ):
            chat_client = llm_client.get_llm_client()
            memory_client = llm_client.get_memory_llm_client()

            # Ensure they are distinct instances
            self.assertIsNot(chat_client, memory_client)

            # Check the underlying GeminiClient instances
            chat_gemini = chat_client._get_client(
                LLMModelSpec(provider="gemini", model="gemini-2.5-flash", supports_tools=True)
            )
            memory_gemini = memory_client._get_client(
                LLMModelSpec(provider="gemini", model="gemini-2.0-flash-lite", supports_tools=True)
            )

            self.assertEqual("chat-key-1", chat_gemini.api_key)
            self.assertEqual("memory-key-2", memory_gemini.api_key)

    def test_gemini_api_key_fallback_when_unset(self):
        with mock.patch.dict(os.environ, {
            "CHAT_MODELS": "gemini:gemini-2.5-flash",
            "MEMORY_MODELS": "gemini:gemini-2.0-flash-lite",
            "GEMINI_API_KEY": "shared-key",
            "MEMORY_GEMINI_API_KEY": "",
        }, clear=False):
            cfg = Config()

        self.assertEqual("", cfg.memory_gemini_api_key)
        self.assertEqual("shared-key", cfg.gemini_api_key)

        with (
            mock.patch.object(llm_client, "config", cfg),
            mock.patch.object(llm_client, "_llm_client", None),
            mock.patch.object(llm_client, "_memory_llm_client", None),
        ):
            memory_client = llm_client.get_memory_llm_client()
            memory_gemini = memory_client._get_client(
                LLMModelSpec(provider="gemini", model="gemini-2.0-flash-lite", supports_tools=True)
            )
            self.assertEqual("shared-key", memory_gemini.api_key)

    def test_deepseek_api_key_isolation_when_configured(self):
        with mock.patch.dict(os.environ, {
            "CHAT_MODELS": "deepseek:deepseek-chat",
            "MEMORY_MODELS": "deepseek:deepseek-coder",
            "DEEPSEEK_API_KEY": "deepseek-chat-key",
            "MEMORY_DEEPSEEK_API_KEY": "deepseek-memory-key",
        }, clear=False):
            cfg = Config()

        with (
            mock.patch.object(llm_client, "config", cfg),
            mock.patch.object(llm_client, "_llm_client", None),
            mock.patch.object(llm_client, "_memory_llm_client", None),
        ):
            chat_client = llm_client.get_llm_client()
            memory_client = llm_client.get_memory_llm_client()

            chat_ds = chat_client._get_client(
                LLMModelSpec(provider="deepseek", model="deepseek-chat", supports_tools=True)
            )
            memory_ds = memory_client._get_client(
                LLMModelSpec(provider="deepseek", model="deepseek-coder", supports_tools=True)
            )

            self.assertEqual("deepseek-chat-key", chat_ds.api_key)
            self.assertEqual("deepseek-memory-key", memory_ds.api_key)

    def test_deepseek_api_key_fallback_when_unset(self):
        with mock.patch.dict(os.environ, {
            "CHAT_MODELS": "deepseek:deepseek-chat",
            "MEMORY_MODELS": "deepseek:deepseek-coder",
            "DEEPSEEK_API_KEY": "shared-ds-key",
            "MEMORY_DEEPSEEK_API_KEY": "",
        }, clear=False):
            cfg = Config()

        with (
            mock.patch.object(llm_client, "config", cfg),
            mock.patch.object(llm_client, "_llm_client", None),
            mock.patch.object(llm_client, "_memory_llm_client", None),
        ):
            memory_client = llm_client.get_memory_llm_client()
            memory_ds = memory_client._get_client(
                LLMModelSpec(provider="deepseek", model="deepseek-coder", supports_tools=True)
            )
            self.assertEqual("shared-ds-key", memory_ds.api_key)

    def test_memory_models_uses_dedicated_key_in_config_validation(self):
        with mock.patch.dict(os.environ, {
            "CHAT_MODELS": "deepseek:deepseek-chat",
            "MEMORY_MODELS": "gemini:gemini-2.0-flash-lite",
            "DEEPSEEK_API_KEY": "ds-key",
            "GEMINI_API_KEY": "",
            "MEMORY_GEMINI_API_KEY": "dedicated-gemini-key",
        }, clear=False):
            cfg = Config()
            self.assertEqual("dedicated-gemini-key", cfg.memory_gemini_api_key)


if __name__ == "__main__":
    unittest.main()
