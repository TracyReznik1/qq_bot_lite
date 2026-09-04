import unittest
from unittest.mock import MagicMock, patch

from src.chat import chat_service
from src.chat.chat_service import generate_reply, reset_history
from src.search.simple.models import (
    OutputKind,
    RequestSource,
    SearchMode,
    SearchOutcome,
    SearchPlan,
    SearchQuery,
    SearchResult,
    SearchTrace,
)
from src.services.llm_types import ChatResponse
from src.services.url_fetch_service import UrlDocumentResult


class UrlChatFlowTests(unittest.TestCase):
    def setUp(self):
        reset_history("private:url_test_user")

    def tearDown(self):
        reset_history("private:url_test_user")

    @patch("src.chat.chat_service.get_simple_search_pipeline_for_chat")
    @patch("src.chat.chat_service.fetch_document")
    def test_url_direct_fetch_succeeds_and_short_circuits_search(
        self,
        mock_fetch,
        mock_get_pipeline,
    ):
        mock_fetch.return_value = UrlDocumentResult(
            ok=True,
            status="success",
            requested_url="https://example.com/test-article",
            final_url="https://example.com/test-article",
            title="测试文章标题",
            content_type="text/html",
            text="这是测试文章的详细正文内容，介绍了相关技术。",
        )
        fake_llm = MagicMock()
        fake_llm.chat.return_value = ChatResponse(content="这篇文章主要介绍了相关技术。")

        with patch.object(chat_service, "llm", fake_llm):
            reply = generate_reply(
                "private:url_test_user",
                "请总结这篇文章 https://example.com/test-article",
                mode=SearchMode.LIGHT,
            )

        self.assertEqual("这篇文章主要介绍了相关技术。", reply)
        # Search pipeline must NOT be created or called (short-circuited)
        mock_get_pipeline.assert_not_called()
        mock_fetch.assert_called_once_with("https://example.com/test-article", timeout_seconds=5.0)

        # Check prompt payload passed to LLM
        called_messages = fake_llm.chat.call_args.args[0]
        untrusted_msg = called_messages[1]["content"]
        self.assertIn("<external_webpage_content url=\"https://example.com/test-article\" title=\"测试文章标题\">", untrusted_msg)
        self.assertIn("这是测试文章的详细正文内容", untrusted_msg)

        # Check history contains user message but NOT the webpage body
        history = chat_service.chat_history["private:url_test_user"]
        user_history_entry = history[-2]["content"]
        self.assertIn("https://example.com/test-article", user_history_entry)
        self.assertNotIn("这是测试文章的详细正文内容", user_history_entry)

    @patch("src.chat.chat_service.get_simple_search_pipeline_for_chat")
    @patch("src.chat.chat_service.fetch_document")
    def test_url_direct_fetch_fails_gracefully_and_falls_back_to_search(
        self,
        mock_fetch,
        mock_get_pipeline,
    ):
        mock_fetch.return_value = UrlDocumentResult(
            ok=False,
            status="timeout",
            requested_url="https://example.com/broken",
            final_url="https://example.com/broken",
            title="",
            content_type="",
            text="网页读取超时。",
        )
        fake_pipeline = MagicMock()
        fake_pipeline.run.return_value = SearchOutcome(
            plan=SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "broken"),)),
            results=(),
            trace=SearchTrace("r1", RequestSource.CHAT, SearchMode.LIGHT),
        )
        mock_get_pipeline.return_value = fake_pipeline

        fake_llm = MagicMock()
        fake_llm.chat.return_value = ChatResponse(content="基于搜索结果回答。")

        with patch.object(chat_service, "llm", fake_llm):
            reply = generate_reply(
                "private:url_test_user",
                "看看这个链接 https://example.com/broken",
                mode=SearchMode.LIGHT,
            )

        mock_fetch.assert_called_once()
        fake_pipeline.run.assert_called_once()
        self.assertIsNotNone(reply)

    @patch("src.chat.chat_service.fetch_document")
    def test_chat_without_url_does_not_call_fetch_document(self, mock_fetch):
        fake_llm = MagicMock()
        fake_llm.chat.return_value = ChatResponse(content="普通回复")
        with patch.object(chat_service, "llm", fake_llm):
            generate_reply(
                "private:url_test_user",
                "今天天气怎么样",
                mode=SearchMode.SKIP,
            )
        mock_fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
