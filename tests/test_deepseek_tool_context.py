import unittest
from types import SimpleNamespace
from unittest import mock

from src.services.deepseek_client import DeepSeekClient


class FakeResponse:
    def __init__(self, message):
        self._message = message

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": self._message}]}


def config():
    return SimpleNamespace(
        deepseek_api_key="d-key",
        deepseek_url="https://api.deepseek.com/chat/completions",
        proxies=None,
        request_timeout=18,
    )


TOOL_CALL = {
    "id": "call-1",
    "type": "function",
    "function": {
        "name": "search_web",
        "arguments": '{"query":"测试"}',
    },
}


class DeepSeekToolContextTests(unittest.TestCase):
    def test_tool_response_captures_reasoning_content(self):
        response = FakeResponse(
            {
                "content": "",
                "reasoning_content": "先搜索再整理",
                "tool_calls": [TOOL_CALL],
            }
        )

        with mock.patch(
            "src.services.deepseek_client.try_proxied_post",
            return_value=response,
        ):
            result = DeepSeekClient(config()).chat(
                [{"role": "user", "content": "查一下"}],
                model="deepseek-v4-flash",
                tools=[{"type": "function", "function": {"name": "search_web"}}],
            )

        self.assertEqual(
            {
                "provider": "deepseek",
                "reasoning_content": "先搜索再整理",
            },
            result.provider_context,
        )

    def test_deepseek_context_is_restored_without_private_field(self):
        response = FakeResponse({"content": "整理结果"})
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [TOOL_CALL],
                "_provider_context": {
                    "provider": "deepseek",
                    "reasoning_content": "先搜索再整理",
                },
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search_web",
                "content": "搜索结果",
            },
        ]

        with mock.patch(
            "src.services.deepseek_client.try_proxied_post",
            return_value=response,
        ) as post:
            DeepSeekClient(config()).chat(
                messages,
                model="deepseek-v4-flash",
            )

        sent_assistant = post.call_args.kwargs["json"]["messages"][0]
        self.assertEqual(
            "先搜索再整理",
            sent_assistant["reasoning_content"],
        )
        self.assertNotIn("_provider_context", sent_assistant)

    def test_gemini_context_is_stripped_without_reasoning_conversion(self):
        response = FakeResponse({"content": "整理结果"})
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [TOOL_CALL],
                "_provider_context": {
                    "provider": "gemini",
                    "content": {
                        "role": "model",
                        "parts": [],
                    },
                },
            }
        ]

        with mock.patch(
            "src.services.deepseek_client.try_proxied_post",
            return_value=response,
        ) as post:
            DeepSeekClient(config()).chat(
                messages,
                model="deepseek-v4-flash",
            )

        sent_assistant = post.call_args.kwargs["json"]["messages"][0]
        self.assertNotIn("_provider_context", sent_assistant)
        self.assertNotIn("reasoning_content", sent_assistant)

    def test_text_response_does_not_retain_reasoning_context(self):
        response = FakeResponse(
            {
                "content": "普通回答",
                "reasoning_content": "无需跨轮保存",
                "tool_calls": [],
            }
        )

        with mock.patch(
            "src.services.deepseek_client.try_proxied_post",
            return_value=response,
        ):
            result = DeepSeekClient(config()).chat(
                [{"role": "user", "content": "你好"}],
                model="deepseek-v4-flash",
            )

        self.assertEqual("普通回答", result.content)
        self.assertIsNone(result.provider_context)


if __name__ == "__main__":
    unittest.main()
