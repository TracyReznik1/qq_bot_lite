import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import src.chat.chat_service as chat_service
import src.main as main
from src.commands import CommandResult
from src.services.llm_types import ChatResponse


IMAGE_URL = "https://img.example/a.png?token=temporary-secret"
IMAGE_DATA_URL = "data:image/png;base64,cG5n"


def private_event(raw_message, message, message_id=10):
    return {
        "post_type": "message",
        "message_type": "private",
        "user_id": 1,
        "self_id": 9,
        "message_id": message_id,
        "raw_message": raw_message,
        "message": message,
    }


def group_event(raw_message, message, message_id=11):
    return {
        "post_type": "message",
        "message_type": "group",
        "group_id": 20,
        "user_id": 1,
        "self_id": 9,
        "message_id": message_id,
        "raw_message": raw_message,
        "message": message,
    }


class MainImageFlowTests(unittest.TestCase):
    def test_private_cq_file_image_is_resolved_only_for_chat_loading(self):
        event = private_event(
            "[CQ:image,file=opaque-file-id,sub_type=0,url=,file_size=123]",
            "[CQ:image,file=opaque-file-id,sub_type=0,url=,file_size=123]",
        )
        with (
            mock.patch.object(main, "load_chat_images", return_value=[IMAGE_DATA_URL]) as load,
            mock.patch.object(main, "generate_reply", return_value="看到了") as generate,
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        load.assert_called_once_with(
            ("",),
            image_file_ids=("opaque-file-id",),
            image_url_resolver=main.onebot.get_image_url,
        )
        generate.assert_called_once_with(
            "private:1", "", image_data_urls=[IMAGE_DATA_URL]
        )
        send.assert_called_once_with("1", "看到了", is_group=False)

    def test_private_image_only_message_reaches_multimodal_chat(self):
        event = private_event(
            f"[CQ:image,file=a.png,url={IMAGE_URL}]",
            [{"type": "image", "data": {"url": IMAGE_URL}}],
        )
        with (
            mock.patch.object(main, "load_chat_images", return_value=[IMAGE_DATA_URL], create=True) as load,
            mock.patch.object(main, "generate_reply", return_value="看到了") as generate,
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        load.assert_called_once_with(
            (IMAGE_URL,),
            image_file_ids=("a.png",),
            image_url_resolver=main.onebot.get_image_url,
        )
        generate.assert_called_once_with(
            "private:1", "", image_data_urls=[IMAGE_DATA_URL]
        )
        send.assert_called_once_with("1", "看到了", is_group=False)

    def test_private_image_plus_text_passes_only_visible_text_to_chat(self):
        event = private_event(
            f"帮我看看 [CQ:image,file=a.png,url={IMAGE_URL}]",
            [
                {"type": "text", "data": {"text": "帮我看看 "}},
                {"type": "image", "data": {"url": IMAGE_URL}},
            ],
        )
        with (
            mock.patch.object(main, "load_chat_images", return_value=[IMAGE_DATA_URL], create=True),
            mock.patch.object(main, "generate_reply", return_value="答案") as generate,
            mock.patch.object(main.onebot, "send_msg"),
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        generate.assert_called_once_with(
            "private:1", "帮我看看", image_data_urls=[IMAGE_DATA_URL]
        )

    def test_group_image_requires_and_accepts_bot_mention(self):
        event = group_event(
            f"[CQ:at,qq=9] [CQ:image,file=a.png,url={IMAGE_URL}]",
            [
                {"type": "at", "data": {"qq": "9"}},
                {"type": "image", "data": {"url": IMAGE_URL}},
            ],
        )
        with (
            mock.patch.object(main, "load_chat_images", return_value=[IMAGE_DATA_URL], create=True),
            mock.patch.object(main, "generate_reply", return_value="群图") as generate,
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        generate.assert_called_once_with(
            "group:20:1", "", image_data_urls=[IMAGE_DATA_URL]
        )
        send.assert_called_once_with(20, "群图", is_group=True)

    def test_group_image_without_bot_mention_is_ignored_before_image_loading(self):
        event = group_event(
            f"[CQ:image,file=a.png,url={IMAGE_URL}]",
            [{"type": "image", "data": {"url": IMAGE_URL}}],
        )
        with (
            mock.patch.object(main, "load_chat_images", create=True) as load,
            mock.patch.object(main, "generate_reply") as generate,
            mock.patch.object(main.onebot, "send_msg") as send,
        ):
            main.process_message(event)

        load.assert_not_called()
        generate.assert_not_called()
        send.assert_not_called()

    def test_command_with_image_routes_without_downloading_image(self):
        event = private_event(
            f"/help [CQ:image,file=a.png,url={IMAGE_URL}]",
            [
                {"type": "text", "data": {"text": "/help "}},
                {"type": "image", "data": {"url": IMAGE_URL}},
            ],
        )
        with (
            mock.patch.object(main, "load_chat_images", create=True) as load,
            mock.patch.object(
                main, "handle_command", return_value=CommandResult(True, "帮助")
            ) as handle,
            mock.patch.object(main, "generate_reply") as generate,
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        load.assert_not_called()
        generate.assert_not_called()
        self.assertEqual("help", handle.call_args.args[0].command)
        self.assertEqual("/help", handle.call_args.args[1].raw_message)
        send.assert_called_once_with("1", "帮助", is_group=False)

    def test_command_with_file_only_image_does_not_resolve_or_download(self):
        event = private_event(
            "/help [CQ:image,file=opaque-file-id,sub_type=0,url=,file_size=123]",
            "/help [CQ:image,file=opaque-file-id,sub_type=0,url=,file_size=123]",
        )
        with (
            mock.patch.object(main, "load_chat_images") as load,
            mock.patch.object(main.onebot, "get_image_url") as resolve,
            mock.patch.object(
                main, "handle_command", return_value=CommandResult(True, "帮助")
            ),
            mock.patch.object(main, "generate_reply") as generate,
            mock.patch.object(main.onebot, "send_msg"),
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        resolve.assert_not_called()
        load.assert_not_called()
        generate.assert_not_called()

    def test_parse_error_is_returned_to_user_without_chat_or_download(self):
        event = private_event(
            "images",
            [
                {"type": "image", "data": {"url": f"https://img.example/{i}.png"}}
                for i in range(5)
            ],
        )
        with (
            mock.patch.object(main, "load_chat_images", create=True) as load,
            mock.patch.object(main, "generate_reply") as generate,
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        load.assert_not_called()
        generate.assert_not_called()
        send.assert_called_once_with(
            "1", "每条消息最多发送 4 张图片。", is_group=False
        )

    def test_image_loading_error_is_returned_to_user(self):
        event = private_event(
            f"[CQ:image,file=a.png,url={IMAGE_URL}]",
            [{"type": "image", "data": {"url": IMAGE_URL}}],
        )
        from src.services.image_input_service import ImageInputError

        with (
            mock.patch.object(
                main,
                "load_chat_images",
                side_effect=ImageInputError("图片读取失败，请重新发送。"),
                create=True,
            ),
            mock.patch.object(main, "generate_reply") as generate,
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        generate.assert_not_called()
        send.assert_called_once_with(
            "1", "图片读取失败，请重新发送。", is_group=False
        )

    def test_image_routing_logs_use_placeholder_not_temporary_url(self):
        event = group_event(
            f"[CQ:at,qq=9] [CQ:image,file=a.png,url={IMAGE_URL}]",
            [
                {"type": "at", "data": {"qq": "9"}},
                {"type": "image", "data": {"url": IMAGE_URL}},
            ],
        )
        with (
            mock.patch.object(main, "load_chat_images", return_value=[IMAGE_DATA_URL], create=True),
            mock.patch.object(main, "generate_reply", return_value="群图"),
            mock.patch.object(main.onebot, "send_msg"),
            mock.patch.object(main.time, "sleep"),
            mock.patch.object(main.logger, "info") as log_info,
        ):
            main.process_message(event)

        logged = repr(log_info.call_args_list)
        self.assertNotIn(IMAGE_URL, logged)
        self.assertNotIn("base64", logged)
        self.assertIn("[图片]", logged)

    def test_real_history_file_omits_image_data_and_temporary_url(self):
        event = private_event(
            f"帮我看看 [CQ:image,file=a.png,url={IMAGE_URL}]",
            [
                {"type": "text", "data": {"text": "帮我看看 "}},
                {"type": "image", "data": {"url": IMAGE_URL}},
            ],
            message_id=210,
        )
        session_key = "private:1"
        chat_service.chat_history.pop(session_key, None)
        self.addCleanup(chat_service.chat_history.pop, session_key, None)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch.object(main, "load_chat_images", return_value=[IMAGE_DATA_URL]),
                mock.patch.object(chat_service, "HISTORY_DIR", Path(temp_dir)),
                mock.patch.object(
                    chat_service,
                    "config",
                    SimpleNamespace(persist_history=True, history_turns=8),
                ),
                mock.patch.object(
                    chat_service.llm,
                    "chat",
                    return_value=ChatResponse(content="看到了"),
                ),
                mock.patch.object(main.onebot, "send_msg"),
                mock.patch.object(main.time, "sleep"),
            ):
                main.process_message(event)

            history_files = list(Path(temp_dir).glob("*.json"))
            self.assertEqual(1, len(history_files))
            raw_history = history_files[0].read_text(encoding="utf-8")
            persisted = json.loads(raw_history)

        self.assertEqual("[图片]\n帮我看看", persisted["messages"][0]["content"])
        self.assertNotIn("data:image", raw_history)
        self.assertNotIn("base64", raw_history)
        self.assertNotIn(IMAGE_URL, raw_history)
        self.assertNotIn("temporary-secret", raw_history)


if __name__ == "__main__":
    unittest.main()
