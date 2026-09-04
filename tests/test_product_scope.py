import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import src.commands.search as search_command
import src.services.search_service as search_service
import src.services.url_fetch_service as url_fetch_service
from src.commands import COMMANDS, CommandContext
from src.config import Config
from src.search.simple.models import (
    RequestSource,
    SearchFailure,
    SearchMode,
    SearchOutcome,
    SearchPlan,
    SearchQuery,
    SearchResult as SimpleSearchResult,
    SearchTrace,
)
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
            {"search", "help", "reset", "remember", "globalremember", "skip"}.issubset(COMMANDS)
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
                "https://example.com/page",
                CommandContext(
                    uid="1",
                    session_key="private:1",
                    raw_message="/search https://example.com/page",
                ),
            )

        self.assertEqual("answer", result)
        generate.assert_called_once()
        self.assertEqual(SearchMode.STANDARD, generate.call_args.kwargs["mode"])

    def test_search_pipeline_factory_is_preserved(self):
        from src.search import get_simple_search_pipeline, reset_simple_search_pipeline
        reset_simple_search_pipeline()
        self.assertTrue(callable(get_simple_search_pipeline))

    def test_compatibility_search_failure_has_no_status_banner(self):
        outcome = SearchOutcome(
            plan=SearchPlan(SearchMode.STANDARD, (SearchQuery("q1", "什么是光合作用"),)),
            results=(),
            trace=SearchTrace("req-1", RequestSource.COMPATIBILITY, SearchMode.STANDARD),
            failure=SearchFailure.PROVIDER_UNAVAILABLE,
        )
        pipeline = mock.Mock(run=mock.Mock(return_value=outcome))
        with mock.patch.object(search_service, "get_simple_search_pipeline", return_value=pipeline):
            result = search_service.search("什么是光合作用")
        self.assertFalse(result.ok)
        self.assertEqual("在线检索未完成。", result.text)

    def test_compatibility_search_success_has_no_status_banner(self):
        outcome = SearchOutcome(
            plan=SearchPlan(SearchMode.STANDARD, (SearchQuery("q1", "当前版本是什么"),)),
            results=(
                SimpleSearchResult(
                    result_id="r1",
                    title="Example",
                    url="https://example.com/page",
                    excerpt="版本是3.2",
                    provider="tavily",
                ),
            ),
            trace=SearchTrace("req-1", RequestSource.COMPATIBILITY, SearchMode.STANDARD),
        )
        pipeline = mock.Mock(run=mock.Mock(return_value=outcome))
        with mock.patch.object(
            search_service,
            "get_simple_search_pipeline",
            return_value=pipeline,
        ):
            result = search_service.search("当前版本是什么")

        self.assertTrue(result.ok)
        self.assertNotIn("搜索状态：success", result.text)
        self.assertNotIn("搜索成功", result.text)
        self.assertNotIn("检索完成", result.text)
        self.assertIn("版本是3.2", result.text)
        self.assertIn("https://example.com/page", result.text)

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
