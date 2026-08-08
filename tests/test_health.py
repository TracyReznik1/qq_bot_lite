import unittest
from types import SimpleNamespace
from unittest import mock

import src.main as main
from src.model_config import ConfiguredModel
from src.persona import Persona


class HealthModelChainTests(unittest.TestCase):
    def test_health_lists_models_without_secrets(self):
        fake_config = SimpleNamespace(
            gemini_api_key="secret-g",
            deepseek_api_key="secret-d",
            onebot_url="http://127.0.0.1:3000",
            require_group_at=True,
            chat_models=(
                ConfiguredModel("gemini", "gemini-test"),
                ConfiguredModel("deepseek", "deepseek-test"),
            ),
        )

        with (
            mock.patch.object(main, "config", fake_config),
            mock.patch.object(main, "get_persona", return_value=Persona("小Q", "角色内容")),
        ):
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
        self.assertEqual("小Q", response["bot_name"])
        self.assertIn("search_ready", response)
        self.assertIn("search_providers", response)

    def test_health_search_providers_without_secrets(self):
        from src.search.models import ProviderReadiness
        fake_config = SimpleNamespace(
            gemini_api_key="secret-g",
            deepseek_api_key="secret-d",
            onebot_url="http://127.0.0.1:3000",
            require_group_at=True,
            chat_models=(
                ConfiguredModel("gemini", "gemini-test"),
                ConfiguredModel("deepseek", "deepseek-test"),
            ),
        )
        readiness = (
            ProviderReadiness("ddgs", True, True, None),
            ProviderReadiness("tavily", True, True, None),
        )
        orchestrator = SimpleNamespace(_providers=(_ReadinessProvider(readiness[0]), _ReadinessProvider(readiness[1])))

        with (
            mock.patch.object(main, "config", fake_config),
            mock.patch.object(main, "get_persona", return_value=Persona("小Q", "角色内容")),
            mock.patch.object(main, "_search_readiness", return_value=list(readiness)),
        ):
            response = main.health()

        self.assertTrue(response["search_ready"])
        self.assertEqual(2, len(response["search_providers"]))
        self.assertEqual("ddgs", response["search_providers"][0]["provider"])
        self.assertEqual("tavily", response["search_providers"][1]["provider"])
        self.assertTrue(response["search_providers"][0]["configured"])
        self.assertTrue(response["search_providers"][0]["available"])
        self.assertTrue(response["search_providers"][1]["configured"])
        self.assertTrue(response["search_providers"][1]["available"])
        serialized = str(response)
        self.assertNotIn("api_key", serialized.casefold())


class _ReadinessProvider:
    def __init__(self, readiness):
        self._readiness = readiness
        self.name = readiness.provider

    def readiness(self):
        return self._readiness


if __name__ == "__main__":
    unittest.main()
