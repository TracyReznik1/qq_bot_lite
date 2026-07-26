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


if __name__ == "__main__":
    unittest.main()
