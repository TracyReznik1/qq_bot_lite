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
from src.search.models import RequestSource, SearchFailureCode, SearchTier, SearchTrace


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

    def test_compatibility_search_finalizes_trace_once_on_early_failure(self):
        trace = SearchTrace("req-1", RequestSource.COMPATIBILITY, SearchTier.LIGHT)
        pipeline_result = SimpleNamespace(
            decision=SimpleNamespace(route=SearchTier.LIGHT),
            evidence=None,
            failure_code=SearchFailureCode.PROVIDER_NOT_CONFIGURED,
            trace=trace,
        )
        orchestrator = mock.Mock(run=mock.Mock(return_value=pipeline_result))
        with (
            mock.patch.object(search_service, "get_search_orchestrator", return_value=orchestrator),
            mock.patch("src.search.orchestrator.finalize_search_trace") as finalize,
        ):
            result = search_service.search("什么是光合作用")
        self.assertFalse(result.ok)
        finalize.assert_called_once()
        self.assertIsNotNone(trace.response_started_at)

    def test_compatibility_search_success_has_no_status_banner(self):
        trace = SearchTrace("req-1", RequestSource.COMPATIBILITY, SearchTier.LIGHT)
        evidence = SimpleNamespace(
            evidence_items=(
                SimpleNamespace(
                    title="Example",
                    url="https://example.com/page",
                    excerpt="版本是3.2",
                ),
            ),
        )
        pipeline_result = SimpleNamespace(
            decision=SimpleNamespace(route=SearchTier.LIGHT),
            evidence=evidence,
            failure_code=None,
            trace=trace,
        )
        orchestrator = mock.Mock(run=mock.Mock(return_value=pipeline_result))
        with (
            mock.patch.object(
                search_service,
                "get_search_orchestrator",
                return_value=orchestrator,
            ),
            mock.patch("src.search.orchestrator.finalize_search_trace"),
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
