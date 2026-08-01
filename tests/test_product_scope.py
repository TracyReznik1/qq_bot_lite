import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import src.commands.search as search_command
import src.services.search_service as search_service
import src.services.url_fetch_service as url_fetch_service
from src.commands import COMMANDS
from src.config import Config
from src.services.onebot_client import OneBotClient
from src.services.search_service import SearchResult


ROOT = Path(__file__).resolve().parents[1]


class ProductScopeTests(unittest.TestCase):
    def test_generation_and_video_configuration_is_absent(self):
        fields = set(Config.__dataclass_fields__)
        residual = sorted(
            name
            for name in fields
            if name == "openai_api_key"
            or name.startswith("video_")
            or name.startswith("image_")
            or name.startswith("comfyui_")
        )
        self.assertEqual([], residual)

    def test_onebot_client_has_no_outbound_image_method(self):
        self.assertFalse(hasattr(OneBotClient, "send_image"))

    def test_direct_url_wrapper_interfaces_are_absent(self):
        self.assertFalse(hasattr(url_fetch_service, "has_url"))
        self.assertFalse(hasattr(url_fetch_service, "url_fetch"))

    def test_expected_commands_are_preserved(self):
        self.assertTrue(
            {"search", "help", "reset", "remember", "globalremember"}.issubset(COMMANDS)
        )

    def test_search_with_url_uses_keyword_search_not_direct_fetch(self):
        with (
            mock.patch.object(
                search_command,
                "normalize_search_query",
                return_value="https://example.com/page",
            ),
            mock.patch.object(
                search_command,
                "generate_reply",
                return_value="answer",
            ) as generate,
        ):
            result = search_command.search_reply(
                "https://example.com/page", "private:1", "/search https://example.com/page"
            )

        self.assertEqual("answer", result)
        generate.assert_called_once()
        self.assertTrue(generate.call_args.kwargs["force_search"])

    def test_search_internal_page_fetch_is_preserved(self):
        from src.search import get_search_orchestrator, reset_search_orchestrator
        reset_search_orchestrator()
        self.assertTrue(callable(get_search_orchestrator))

    def test_gemini_runtime_uses_native_stateless_generate_content(self):
        source = (
            ROOT / "src" / "services" / "gemini_client.py"
        ).read_text(encoding="utf-8")

        self.assertIn(":generateContent", source)
        self.assertIn("x-goog-api-key", source)
        self.assertNotIn("/openai/chat/completions", source)
        self.assertNotIn("previous_interaction_id", source)
        self.assertNotIn("interactions.create", source)


if __name__ == "__main__":
    unittest.main()
