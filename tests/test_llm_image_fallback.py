import unittest
from unittest import mock

import src.main as main
import src.services.llm_client as llm_client
from src.services.llm_types import ChatResponse, LLMModelSpec


class FailingClient:
    def chat(self, *args, **kwargs):
        raise RuntimeError("model does not support image input")


class SuccessfulClient:
    def chat(self, *args, **kwargs):
        return ChatResponse(content="识别成功")


def image_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "看看"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,cG5n"},
                },
            ],
        }
    ]


class LlmImageFallbackTests(unittest.TestCase):
    def image_error_type(self):
        error_type = getattr(llm_client, "ImageRecognitionUnavailable", None)
        self.assertIsNotNone(
            error_type, "ImageRecognitionUnavailable is missing"
        )
        return error_type

    def test_image_exhaustion_uses_specific_error(self):
        error_type = self.image_error_type()
        client = llm_client.FallbackLLMClient(
            [LLMModelSpec(provider="gemini", model="vision-test")]
        )
        with (
            mock.patch.object(
                client, "_get_client", return_value=FailingClient()
            ),
            self.assertRaisesRegex(
                error_type,
                "当前模型无法识别该图片",
            ),
        ):
            client.chat(image_messages())

    def test_image_failure_still_tries_the_remaining_model_chain(self):
        client = llm_client.FallbackLLMClient(
            [
                LLMModelSpec(provider="gemini", model="vision-first"),
                LLMModelSpec(provider="deepseek", model="vision-second"),
            ]
        )
        with mock.patch.object(
            client,
            "_get_client",
            side_effect=[FailingClient(), SuccessfulClient()],
        ) as get_client:
            response = client.chat(image_messages())

        self.assertEqual("识别成功", response.content)
        self.assertEqual(2, get_client.call_count)

    def test_text_exhaustion_keeps_generic_error(self):
        client = llm_client.FallbackLLMClient(
            [LLMModelSpec(provider="gemini", model="text-test")]
        )
        with (
            mock.patch.object(
                client, "_get_client", return_value=FailingClient()
            ),
            self.assertRaisesRegex(RuntimeError, "所有模型暂时不可用"),
        ):
            client.chat([{"role": "user", "content": "你好"}])

    def test_main_sends_image_specific_error_without_config_prefix(self):
        event = {
            "post_type": "message",
            "message_type": "private",
            "user_id": 1,
            "self_id": 9,
            "message_id": 12,
            "raw_message": (
                "[CQ:image,file=a.png,url=https://img.example/a.png]"
            ),
            "message": [
                {
                    "type": "image",
                    "data": {"url": "https://img.example/a.png"},
                }
            ],
        }
        error_type = self.image_error_type()
        with (
            mock.patch.object(
                main,
                "load_chat_images",
                return_value=["data:image/png;base64,cG5n"],
            ),
            mock.patch.object(
                main,
                "generate_reply",
                side_effect=error_type("当前模型无法识别该图片。"),
            ),
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        send.assert_called_once_with(
            "1", "当前模型无法识别该图片。", is_group=False
        )


if __name__ == "__main__":
    unittest.main()
