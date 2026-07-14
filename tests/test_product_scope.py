import unittest
from types import SimpleNamespace
from unittest import mock

import src.commands.search as search_command
import src.services.search_service as search_service
from src.commands import COMMANDS
from src.config import Config
from src.services.onebot_client import OneBotClient
from src.services.search_service import SearchResult


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

    def test_expected_commands_are_preserved(self):
        self.assertTrue(
            {"search", "help", "reset", "remember", "globalremember"}.issubset(COMMANDS)
        )

    def test_search_with_url_uses_keyword_search_not_direct_fetch(self):
        with (
            mock.patch.object(
                search_command,
                "extract_first_url",
                return_value="https://example.com/page",
                create=True,
            ),
            mock.patch.object(
                search_command,
                "fetch_url",
                return_value=SimpleNamespace(ok=True, text="direct page"),
                create=True,
            ) as direct_fetch,
            mock.patch.object(
                search_command,
                "search",
                return_value=SearchResult(ok=True, status="success", text="search result"),
            ) as keyword_search,
            mock.patch.object(search_command, "generate_reply", return_value="answer"),
        ):
            result = search_command.search_reply(
                "https://example.com/page", "private:1", "/search https://example.com/page"
            )

        self.assertEqual("answer", result)
        keyword_search.assert_called_once_with("https://example.com/page")
        direct_fetch.assert_not_called()

    def test_search_internal_page_fetch_is_preserved(self):
        self.assertTrue(hasattr(search_service, "fetch_url"))


if __name__ == "__main__":
    unittest.main()
