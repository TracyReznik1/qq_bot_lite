import unittest
from types import SimpleNamespace
from unittest import mock

from src.chat import chat_service
from src.services.deepseek_client import DeepSeekClient
from src.services.gemini_client import GeminiClient
from src.services.llm_types import ChatResponse


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def config():
    return SimpleNamespace(
        gemini_api_key="g-key",
        gemini_url="https://generativelanguage.googleapis.com/v1",
        proxies=None,
        request_timeout=18,
    )


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "搜索网页",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


class GeminiNativeRequestTests(unittest.TestCase):
    def test_call_timeout_is_capped_by_remaining_retrieval_time(self):
        response = FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "回答"}]}}]}
        )
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ) as post:
            GeminiClient(config()).chat(
                [{"role": "user", "content": "你好"}],
                model="gemini:model/one",
                timeout_seconds=0.25,
            )

        self.assertEqual(post.call_args.kwargs["timeout"], 0.25)

    def test_uses_native_url_api_key_header_and_generation_config(self):
        response = FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "回答"}]}}
                ]
            }
        )
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ) as post:
            result = GeminiClient(config()).chat(
                [
                    {"role": "system", "content": "系统规则"},
                    {"role": "user", "content": "你好"},
                ],
                model="gemini:model/one",
                temperature=0.25,
                max_tokens=512,
            )

        self.assertEqual("回答", result.content)
        url = post.call_args.args[0]
        kwargs = post.call_args.kwargs
        self.assertEqual(
            "https://generativelanguage.googleapis.com/v1/"
            "models/gemini%3Amodel%2Fone:generateContent",
            url,
        )
        self.assertEqual("g-key", kwargs["headers"]["x-goog-api-key"])
        self.assertNotIn("Authorization", kwargs["headers"])
        payload = kwargs["json"]
        self.assertNotIn("model", payload)
        self.assertNotIn("messages", payload)
        self.assertEqual(
            {"parts": [{"text": "系统规则"}]},
            payload["systemInstruction"],
        )
        self.assertEqual(
            {"temperature": 0.25, "maxOutputTokens": 512},
            payload["generationConfig"],
        )

    def test_converts_image_and_tool_declaration(self):
        response = FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "id": "remote-call-1",
                                        "name": "search_web",
                                        "args": {"query": "测试"},
                                    },
                                    "thoughtSignature": "signature-1",
                                }
                            ]
                        }
                    }
                ]
            }
        )
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ) as post:
            result = GeminiClient(config()).chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "看图"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,cG5n"
                                },
                            },
                        ],
                    }
                ],
                model="vision",
                tools=[SEARCH_TOOL],
                tool_choice="auto",
            )

        payload = post.call_args.kwargs["json"]
        parts = payload["contents"][0]["parts"]
        self.assertEqual({"text": "看图"}, parts[0])
        self.assertEqual(
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": "cG5n",
                }
            },
            parts[1],
        )
        declaration = payload["tools"][0]["functionDeclarations"][0]
        self.assertEqual("search_web", declaration["name"])
        self.assertEqual(
            "AUTO",
            payload["toolConfig"]["functionCallingConfig"]["mode"],
        )
        self.assertEqual("search_web", result.tool_calls[0]["function"]["name"])
        self.assertEqual(
            '{"query":"测试"}',
            result.tool_calls[0]["function"]["arguments"],
        )
        self.assertEqual("remote-call-1", result.tool_calls[0]["id"])
        self.assertEqual(
            "signature-1",
            result.provider_context["content"]["parts"][0][
                "thoughtSignature"
            ],
        )

    def test_returns_raw_function_call_content_with_signature_and_id(self):
        native_content = {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "id": "remote-call-1",
                        "name": "search_web",
                        "args": {"query": "测试"},
                    },
                    "thoughtSignature": "signature-1",
                }
            ],
        }
        first_response = FakeResponse(
            {"candidates": [{"content": native_content}]}
        )
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=first_response,
        ):
            result = GeminiClient(config()).chat(
                [{"role": "user", "content": "查一下"}],
                model="gemini-test",
                tools=[SEARCH_TOOL],
                tool_choice="auto",
            )

        self.assertEqual(
            {"provider": "gemini", "content": native_content},
            result.provider_context,
        )

    def test_converts_signed_tool_round_trip_messages(self):
        response = FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "整理结果"}]}}
                ]
            }
        )
        messages = [
            {"role": "user", "content": "查一下"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"测试"}',
                        },
                    }
                ],
                "_provider_context": {
                    "provider": "gemini",
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "remote-call-1",
                                    "name": "search_web",
                                    "args": {"query": "测试"},
                                },
                                "thoughtSignature": "signature-1",
                            }
                        ],
                    },
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
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ) as post:
            GeminiClient(config()).chat(messages, model="gemini-test")

        contents = post.call_args.kwargs["json"]["contents"]
        self.assertEqual(
            {
                "functionCall": {
                    "id": "remote-call-1",
                    "name": "search_web",
                    "args": {"query": "测试"},
                },
                "thoughtSignature": "signature-1",
            },
            contents[1]["parts"][0],
        )
        self.assertEqual(
            {
                "functionResponse": {
                    "id": "remote-call-1",
                    "name": "search_web",
                    "response": {"result": "搜索结果"},
                }
            },
            contents[2]["parts"][0],
        )

    def test_preserves_parallel_call_order_ids_and_signature_placement(self):
        response = FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "整理结果"}]}}
                ]
            }
        )
        native_parts = [
            {
                "functionCall": {
                    "id": "remote-1",
                    "name": "search_web",
                    "args": {"query": "一"},
                },
                "thoughtSignature": "parallel-signature",
            },
            {
                "functionCall": {
                    "id": "remote-2",
                    "name": "search_web",
                    "args": {"query": "二"},
                }
            },
        ]
        calls = [
            {
                "id": "local-1",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": '{"query":"一"}',
                },
            },
            {
                "id": "local-2",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": '{"query":"二"}',
                },
            },
        ]
        messages = [
            {"role": "user", "content": "查两项"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": calls,
                "_provider_context": {
                    "provider": "gemini",
                    "content": {
                        "role": "model",
                        "parts": native_parts,
                    },
                },
            },
            {
                "role": "tool",
                "tool_call_id": "local-1",
                "name": "search_web",
                "content": "结果一",
            },
            {
                "role": "tool",
                "tool_call_id": "local-2",
                "name": "search_web",
                "content": "结果二",
            },
        ]

        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ) as post:
            GeminiClient(config()).chat(messages, model="gemini-test")

        contents = post.call_args.kwargs["json"]["contents"]
        self.assertEqual(native_parts, contents[1]["parts"])
        self.assertEqual(
            ["remote-1", "remote-2"],
            [
                part["functionResponse"]["id"]
                for part in contents[2]["parts"]
            ],
        )
        self.assertNotIn(
            "thoughtSignature",
            contents[1]["parts"][1],
        )

    def test_cross_provider_tool_call_gets_official_dummy_signature(self):
        response = FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "整理结果"}]}}
                ]
            }
        )
        messages = [
            {"role": "user", "content": "查一下"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "deepseek-call",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"测试"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "deepseek-call",
                "name": "search_web",
                "content": "搜索结果",
            },
        ]
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ) as post:
            GeminiClient(config()).chat(messages, model="gemini-test")

        function_part = post.call_args.kwargs["json"]["contents"][1][
            "parts"
        ][0]
        self.assertEqual(
            "skip_thought_signature_validator",
            function_part["thoughtSignature"],
        )


class GeminiNativeResponseTests(unittest.TestCase):
    def test_joins_text_parts_and_parses_multiple_calls(self):
        response = FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "第一段"},
                                {"text": "第二段"},
                                {
                                    "functionCall": {
                                        "name": "search_web",
                                        "args": {"query": "一"},
                                    }
                                },
                                {
                                    "functionCall": {
                                        "name": "search_web",
                                        "args": {"query": "二"},
                                    }
                                },
                            ]
                        }
                    }
                ]
            }
        )
        with mock.patch(
            "src.services.gemini_client.try_proxied_post",
            return_value=response,
        ):
            result = GeminiClient(config()).chat(
                [{"role": "user", "content": "测试"}],
                model="gemini-test",
            )

        self.assertEqual("第一段\n第二段", result.content)
        self.assertEqual(2, len(result.tool_calls))
        self.assertNotEqual(result.tool_calls[0]["id"], result.tool_calls[1]["id"])

    def test_rejects_empty_candidates(self):
        with (
            mock.patch(
                "src.services.gemini_client.try_proxied_post",
                return_value=FakeResponse({"candidates": []}),
            ),
            self.assertRaisesRegex(RuntimeError, "Gemini.*候选"),
        ):
            GeminiClient(config()).chat(
                [{"role": "user", "content": "测试"}],
                model="gemini-test",
            )


class ProviderContextBoundaryTests(unittest.TestCase):
    def test_chat_tool_messages_keep_context_only_on_temporary_assistant(self):
        context = {
            "provider": "gemini",
            "content": {"role": "model", "parts": []},
        }
        calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": '{"query":"测试"}',
                },
            }
        ]

        with mock.patch.object(
            chat_service,
            "_tool_result",
            return_value="搜索结果",
        ):
            messages = chat_service.build_tool_messages(
                calls,
                "测试",
                provider_context=context,
            )

        self.assertEqual(context, messages[0]["_provider_context"])
        self.assertNotIn("_provider_context", messages[1])

    def test_deepseek_removes_private_provider_context(self):
        cfg = SimpleNamespace(
            deepseek_api_key="d-key",
            deepseek_url="https://api.deepseek.com/chat/completions",
            proxies=None,
            request_timeout=18,
        )
        response = FakeResponse(
            {
                "choices": [
                    {"message": {"content": "回答", "tool_calls": []}}
                ]
            }
        )
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [],
                "_provider_context": {
                    "provider": "gemini",
                    "content": {"parts": []},
                },
            }
        ]

        with mock.patch(
            "src.services.deepseek_client.try_proxied_post",
            return_value=response,
        ) as post:
            result = DeepSeekClient(cfg).chat(
                messages,
                model="deepseek-test",
            )

        self.assertEqual("回答", result.content)
        sent_message = post.call_args.kwargs["json"]["messages"][0]
        self.assertNotIn("_provider_context", sent_message)

    def test_string_response_normalization_has_no_provider_context(self):
        normalized = chat_service.normalize_chat_response("回答")

        self.assertEqual(
            ChatResponse(content="回答", provider_context=None),
            normalized,
        )


    def test_deepseek_rejects_empty_api_key(self):
        cfg = SimpleNamespace(
            deepseek_api_key="",
            deepseek_url="https://api.deepseek.com/chat/completions",
            proxies=None,
            request_timeout=18,
        )
        with self.assertRaises(RuntimeError) as ctx:
            DeepSeekClient(cfg).chat(
                [{"role": "user", "content": "hi"}],
                model="deepseek-test",
            )
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
