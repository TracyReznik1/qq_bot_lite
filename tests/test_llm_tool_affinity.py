import unittest
from unittest import mock

from src.services.llm_client import FallbackLLMClient
from src.services.llm_types import ChatResponse, LLMModelSpec


TOOL_CALL = {
    "id": "call-1",
    "type": "function",
    "function": {
        "name": "search_web",
        "arguments": '{"query":"测试"}',
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {"name": "search_web"},
}


class StaticClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


def continuation_messages(provider: str, model: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [TOOL_CALL],
            "_provider_context": {
                "provider": provider,
                "model": model,
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


class ToolAffinityTests(unittest.TestCase):
    def test_tool_response_is_stamped_with_exact_provider_and_model(self):
        provider_response = ChatResponse(
            tool_calls=[TOOL_CALL],
            provider_context={
                "provider": "deepseek",
                "reasoning_content": "先搜索再整理",
            },
        )
        provider_client = StaticClient(response=provider_response)
        fallback = FallbackLLMClient(
            [
                LLMModelSpec(
                    provider="deepseek",
                    model="deepseek-v4-flash",
                )
            ]
        )

        with mock.patch.object(
            fallback,
            "_get_client",
            return_value=provider_client,
        ):
            result = fallback.chat(
                [{"role": "user", "content": "查一下"}],
                tools=[SEARCH_TOOL],
                tool_choice="auto",
            )

        self.assertEqual(
            {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "reasoning_content": "先搜索再整理",
            },
            result.provider_context,
        )

    def test_continuation_skips_models_before_and_after_affinity(self):
        first = StaticClient(response=ChatResponse(content="错误模型一"))
        pinned = StaticClient(response=ChatResponse(content="正确模型"))
        last = StaticClient(response=ChatResponse(content="错误模型二"))
        clients = {
            ("gemini", "first"): first,
            ("deepseek", "pinned"): pinned,
            ("gemini", "last"): last,
        }
        fallback = FallbackLLMClient(
            [
                LLMModelSpec(provider="gemini", model="first"),
                LLMModelSpec(provider="deepseek", model="pinned"),
                LLMModelSpec(provider="gemini", model="last"),
            ]
        )

        with mock.patch.object(
            fallback,
            "_get_client",
            side_effect=lambda spec: clients[(spec.provider, spec.model)],
        ):
            result = fallback.chat(
                continuation_messages("deepseek", "pinned"),
                tools=[SEARCH_TOOL],
                tool_choice="auto",
            )

        self.assertEqual("正确模型", result.content)
        self.assertEqual(0, first.calls)
        self.assertEqual(1, pinned.calls)
        self.assertEqual(0, last.calls)

    def test_pinned_model_failure_does_not_cross_fallback(self):
        pinned = StaticClient(error=RuntimeError("pinned unavailable"))
        other = StaticClient(response=ChatResponse(content="不应使用"))
        clients = {
            ("deepseek", "pinned"): pinned,
            ("gemini", "other"): other,
        }
        fallback = FallbackLLMClient(
            [
                LLMModelSpec(provider="deepseek", model="pinned"),
                LLMModelSpec(provider="gemini", model="other"),
            ]
        )

        with (
            mock.patch.object(
                fallback,
                "_get_client",
                side_effect=lambda spec: clients[(spec.provider, spec.model)],
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "所有模型暂时不可用",
            ),
        ):
            fallback.chat(
                continuation_messages("deepseek", "pinned"),
                tools=[SEARCH_TOOL],
                tool_choice="none",
            )

        self.assertEqual(1, pinned.calls)
        self.assertEqual(0, other.calls)


if __name__ == "__main__":
    unittest.main()
