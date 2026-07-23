import unittest
from unittest import mock

import src.chat.chat_service as chat_service
from src.services.llm_types import ChatResponse


class MultimodalChatTests(unittest.TestCase):
    def test_builds_provider_neutral_multimodal_content(self):
        content = chat_service.build_user_content(
            "这是什么？", ["data:image/png;base64,cG5n"]
        )
        self.assertEqual(
            [
                {"type": "text", "text": "这是什么？"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}},
            ],
            content,
        )

    def test_image_only_message_gets_a_text_instruction(self):
        content = chat_service.build_user_content("", ["data:image/png;base64,cG5n"])
        self.assertEqual("请识别图片内容并回答。", content[0]["text"])

    def test_history_uses_placeholders_and_never_contains_image_data(self):
        history_text = chat_service.history_user_text(
            "帮我看看", 2
        )
        self.assertEqual("[图片]\n[图片]\n帮我看看", history_text)
        self.assertNotIn("base64", history_text)
        self.assertNotIn("http", history_text)

    def test_text_only_content_remains_a_string(self):
        self.assertEqual("你好", chat_service.build_user_content("你好", []))

    def test_generate_reply_persists_placeholder_not_image_data(self):
        session_key = "test:image-history"
        chat_service.chat_history.pop(session_key, None)
        with (
            mock.patch.object(chat_service, "_ensure_history_loaded"),
            mock.patch.object(chat_service, "_save_history_unlocked"),
            mock.patch.object(
                chat_service.llm,
                "chat",
                return_value=ChatResponse(content="看到了"),
            ),
        ):
            reply = chat_service.generate_reply(
                session_key,
                "帮我看看",
                tool_context="已有上下文",
                image_data_urls=["data:image/png;base64,cG5n"],
            )
        history = chat_service.chat_history.pop(session_key)
        self.assertEqual("看到了", reply)
        self.assertEqual("[图片]\n帮我看看", history[0]["content"])
        self.assertNotIn("base64", str(history))


if __name__ == "__main__":
    unittest.main()
