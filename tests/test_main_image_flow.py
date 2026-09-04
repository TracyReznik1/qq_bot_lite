import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import src.chat.chat_service as chat_service
import src.main as main
from src.commands import CommandResult
from src.search.simple.models import (
    RequestSource,
    SearchMode,
    SearchOutcome,
    SearchPlan,
    SearchQuery,
    SearchResult,
    SearchTrace,
)
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


def _light_outcome():
    return SearchOutcome(
        plan=SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "帮我看看"),)),
        results=(
            SearchResult(
                "R1",
                "Title",
                "https://example.com/1",
                "Excerpt",
                "tavily",
                0.9,
            ),
        ),
        trace=SearchTrace("req-1", RequestSource.CHAT, SearchMode.LIGHT),
    )



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
    def setUp(self):
        self.memory_service = mock.Mock()
        self.memory_service.stage_event.return_value = 1
        service_patch = mock.patch.object(
            main,
            "get_memory_service",
            return_value=self.memory_service,
        )
        service_patch.start()
        self.addCleanup(service_patch.stop)

    def test_memory_stage_failure_does_not_change_successful_chat_reply(self):
        event = private_event("hello", "hello", message_id=301)
        memory_service = mock.Mock()
        memory_service.stage_event.side_effect = RuntimeError(
            "incoming-private-text-must-not-be-logged"
        )

        with (
            mock.patch.object(main, "get_memory_service", return_value=memory_service),
            mock.patch.object(main, "load_chat_images", return_value=[]),
            mock.patch.object(main, "generate_reply", return_value="successful reply"),
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        send.assert_called_once_with("1", "successful reply", is_group=False)
        memory_service.release_job.assert_not_called()

    def test_memory_release_failure_does_not_send_a_second_error_reply(self):
        event = private_event("hello", "hello", message_id=302)
        memory_service = mock.Mock()
        memory_service.stage_event.return_value = 91
        memory_service.release_job.side_effect = RuntimeError(
            "successful-reply-must-not-be-logged"
        )

        with (
            mock.patch.object(main, "get_memory_service", return_value=memory_service),
            mock.patch.object(main, "load_chat_images", return_value=[]),
            mock.patch.object(main, "generate_reply", return_value="successful reply"),
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        send.assert_called_once_with("1", "successful reply", is_group=False)
        memory_service.release_job.assert_called_once_with(91, [])

    def test_structured_reply_reference_resolves_author_before_staging(self):
        event = private_event(
            "[CQ:reply,id=reply-77]他喜欢跑步",
            [
                {"type": "reply", "data": {"id": "reply-77"}},
                {"type": "text", "data": {"text": "他喜欢跑步"}},
            ],
            message_id=303,
        )
        memory_service = mock.Mock()

        with (
            mock.patch.object(main, "get_memory_service", return_value=memory_service),
            mock.patch.object(
                main.onebot,
                "get_message_author",
                return_value="456",
            ) as get_author,
            mock.patch.object(main, "load_chat_images", return_value=[]),
            mock.patch.object(main, "generate_reply", return_value="ok"),
            mock.patch.object(main.onebot, "send_msg"),
            mock.patch.object(main.time, "sleep"),
        ):
            memory_service.stage_event.side_effect = (
                lambda _event: 92 if get_author.called else self.fail(
                    "reply author lookup must precede staging"
                )
            )
            main.process_message(event)

        get_author.assert_called_once_with("reply-77")
        staged_event = memory_service.stage_event.call_args.args[0]
        self.assertEqual("reply-77", staged_event.reply_to_message_id)
        self.assertEqual("456", staged_event.reply_to_user_id)
        self.assertEqual("他喜欢跑步", staged_event.text)

    def test_cq_only_reply_with_attributes_around_id_uses_raw_fallback(self):
        raw_message = (
            "[CQ:reply,seq=12,id=reply-88,time=1720000000]他喜欢游泳"
        )
        event = private_event(raw_message, raw_message, message_id=307)
        memory_service = mock.Mock()

        with (
            mock.patch.object(main, "get_memory_service", return_value=memory_service),
            mock.patch.object(
                main.onebot,
                "get_message_author",
                return_value="789",
            ) as get_author,
            mock.patch.object(main, "load_chat_images", return_value=[]),
            mock.patch.object(main, "generate_reply", return_value="ok"),
            mock.patch.object(main.onebot, "send_msg"),
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        get_author.assert_called_once_with("reply-88")
        staged_event = memory_service.stage_event.call_args.args[0]
        self.assertEqual("reply-88", staged_event.reply_to_message_id)
        self.assertEqual("789", staged_event.reply_to_user_id)
        self.assertEqual("他喜欢游泳", staged_event.text)

    def test_message_and_reply_bodies_are_absent_from_captured_logs(self):
        incoming = "incoming-private-marker-4b739"
        reply = "reply-private-marker-85d20"
        event = private_event(incoming, incoming, message_id=304)
        memory_service = mock.Mock()
        memory_service.stage_event.return_value = 93

        with (
            mock.patch.object(main, "get_memory_service", return_value=memory_service),
            mock.patch.object(main, "load_chat_images", return_value=[]),
            mock.patch.object(main, "generate_reply", return_value=reply),
            mock.patch.object(main.onebot, "send_msg"),
            mock.patch.object(main.time, "sleep"),
            self.assertLogs("qq-bot", level="INFO") as captured,
        ):
            main.process_message(event)

        logged = "\n".join(captured.output)
        self.assertNotIn(incoming, logged)
        self.assertNotIn(reply, logged)

    def test_command_clears_acceptance_reservation_without_staging(self):
        event = private_event("/help", "/help", message_id=305)
        event["_qqbot_sequence"] = 305
        memory_service = mock.Mock()

        with (
            mock.patch.object(main, "get_memory_service", return_value=memory_service),
            mock.patch.object(
                main,
                "handle_command",
                return_value=CommandResult(True, "help"),
            ),
            mock.patch.object(main.onebot, "send_msg"),
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        memory_service.stage_event.assert_not_called()
        memory_service.clear_pending_sequence.assert_called_once_with(
            "private:1",
            305,
        )

    def test_command_receives_real_message_id_and_memory_context(self):
        event = group_event(
            "[CQ:at,qq=9] /remember 我喜欢简洁回答",
            "[CQ:at,qq=9] /remember 我喜欢简洁回答",
            message_id=98765,
        )

        with (
            mock.patch.object(
                main,
                "config",
                replace(main.config, require_group_at=True),
            ),
            mock.patch.object(
                main,
                "handle_command",
                return_value=CommandResult(True, "ok"),
            ) as handle,
            mock.patch.object(main.onebot, "send_msg"),
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        command_context = handle.call_args.args[1]
        self.assertEqual("98765", command_context.message_id)
        self.assertEqual("1", command_context.memory_context.user_id)
        self.assertTrue(command_context.memory_context.is_group)
        self.assertEqual("20", command_context.memory_context.group_id)

    def test_chat_exception_log_contains_only_error_class_metadata(self):
        incoming = "incoming-error-marker-d19c"
        exception_body = "provider-private-marker-651e"
        event = private_event(incoming, incoming, message_id=306)
        memory_service = mock.Mock()
        memory_service.stage_event.return_value = 94

        with (
            mock.patch.object(main, "get_memory_service", return_value=memory_service),
            mock.patch.object(main, "load_chat_images", return_value=[]),
            mock.patch.object(
                main,
                "generate_reply",
                side_effect=RuntimeError(exception_body),
            ),
            mock.patch.object(main.onebot, "send_msg"),
            mock.patch.object(main.time, "sleep"),
            self.assertLogs("qq-bot", level="ERROR") as captured,
        ):
            main.process_message(event)

        logged = "\n".join(captured.output)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn(incoming, logged)
        self.assertNotIn(exception_body, logged)

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

    def test_search_command_downloads_and_passes_images(self):
        event = private_event(
            f"/search 看看 [CQ:image,file=a.png,url={IMAGE_URL}]",
            [
                {"type": "text", "data": {"text": "/search 看看 "}},
                {"type": "image", "data": {"url": IMAGE_URL}},
            ],
        )
        with (
            mock.patch.object(main, "load_chat_images", return_value=[IMAGE_DATA_URL], create=True) as load,
            mock.patch.object(
                main, "handle_command", return_value=CommandResult(True, "结果")
            ) as handle,
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        load.assert_called_once()
        self.assertEqual((IMAGE_DATA_URL,), handle.call_args.args[1].image_data_urls)
        send.assert_called_once_with("1", "结果", is_group=False)

    def test_skip_command_downloads_file_only_image(self):
        event = private_event(
            "/skip [CQ:image,file=opaque-file-id,sub_type=0,url=,file_size=123]",
            "/skip [CQ:image,file=opaque-file-id,sub_type=0,url=,file_size=123]",
        )
        with (
            mock.patch.object(main, "load_chat_images", return_value=[IMAGE_DATA_URL]) as load,
            mock.patch.object(main.onebot, "get_image_url") as resolve,
            mock.patch.object(
                main, "handle_command", return_value=CommandResult(True, "跳过")
            ) as handle,
            mock.patch.object(main.onebot, "send_msg"),
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        load.assert_called_once_with(
            ("",),
            image_file_ids=("opaque-file-id",),
            image_url_resolver=resolve,
        )
        self.assertEqual((IMAGE_DATA_URL,), handle.call_args.args[1].image_data_urls)

    def test_command_image_loading_error_is_returned_to_user(self):
        event = private_event(
            f"/search 看看 [CQ:image,file=a.png,url={IMAGE_URL}]",
            [
                {"type": "text", "data": {"text": "/search 看看 "}},
                {"type": "image", "data": {"url": IMAGE_URL}},
            ],
        )
        from src.services.image_input_service import ImageInputError

        with (
            mock.patch.object(
                main,
                "load_chat_images",
                side_effect=ImageInputError("图片读取失败，请重新发送。"),
                create=True,
            ),
            mock.patch.object(main, "handle_command") as handle,
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        handle.assert_not_called()
        send.assert_called_once_with(
            "1", "图片读取失败，请重新发送。", is_group=False
        )


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

    def test_image_routing_logs_omit_placeholder_and_temporary_url(self):
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
        self.assertNotIn("[图片]", logged)

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
                mock.patch.object(
                    chat_service,
                    "config",
                    SimpleNamespace(
                        data_dir=Path(temp_dir),
                        persist_history=True,
                        history_turns=8,
                    ),
                ),
                mock.patch("src.chat.prompt.MemoryRetriever", return_value=mock.Mock(retrieve=lambda ctx, query: [])),
                mock.patch.object(
                    chat_service,
                    "build_untrusted_context",
                    return_value="[非可信上下文]暂无",
                ),
                mock.patch.object(
                    chat_service.llm,
                    "chat",
                    return_value=ChatResponse(content="看到了"),
                ),
                mock.patch.object(
                    chat_service,
                    "get_simple_search_pipeline_for_chat",
                    return_value=SimpleNamespace(run=lambda req: _light_outcome()),
                ),
                mock.patch.object(main.onebot, "send_msg"),
                mock.patch.object(main.time, "sleep"),
            ):
                main.process_message(event)

            history_files = list(
                (Path(temp_dir) / "history").glob("*.json")
            )
            self.assertEqual(1, len(history_files))
            raw_history = history_files[0].read_text(encoding="utf-8")
            persisted = json.loads(raw_history)

        self.assertEqual("[图片]\n帮我看看", persisted["messages"][0]["content"])
        self.assertNotIn("data:image", raw_history)
        self.assertNotIn("base64", raw_history)
        self.assertNotIn(IMAGE_URL, raw_history)
        self.assertNotIn("temporary-secret", raw_history)


class MainDeterministicModeTests(unittest.TestCase):
    def setUp(self):
        self.memory_service = mock.Mock()
        self.memory_service.stage_event.return_value = 1
        service_patch = mock.patch.object(
            main,
            "get_memory_service",
            return_value=self.memory_service,
        )
        service_patch.start()
        self.addCleanup(service_patch.stop)

    def test_ordinary_text_image_and_mixed_messages_delegate_to_router_without_explicit_mode(self):
        for text, images in (("hello", []), ("", [IMAGE_DATA_URL]), ("hello", [IMAGE_DATA_URL])):
            with self.subTest(text=text, images=images):
                raw_msg = f"{text} [CQ:image,url={IMAGE_URL}]" if images else text
                parsed_msg = [
                    *([{"type": "text", "data": {"text": text}}] if text else []),
                    *([{"type": "image", "data": {"url": IMAGE_URL}}] if images else []),
                ]
                event = private_event(raw_msg, parsed_msg if parsed_msg else raw_msg)
                with (
                    mock.patch.object(main, "load_chat_images", return_value=images, create=True),
                    mock.patch.object(main, "generate_reply", return_value="reply") as generate_reply,
                    mock.patch.object(main.onebot, "send_msg"),
                    mock.patch.object(main.time, "sleep"),
                ):
                    main.process_message(event)
                    generate_reply.assert_called_once_with(
                        "private:1", text, image_data_urls=images
                    )

    def test_search_command_downloads_and_passes_images(self):
        event = private_event(
            f"/search 看看 [CQ:image,file=a.png,url={IMAGE_URL}]",
            [
                {"type": "text", "data": {"text": "/search 看看 "}},
                {"type": "image", "data": {"url": IMAGE_URL}},
            ],
        )
        with (
            mock.patch.object(main, "load_chat_images", return_value=[IMAGE_DATA_URL], create=True) as load,
            mock.patch.object(main, "handle_command", return_value=CommandResult(True, "结果")) as handle,
            mock.patch.object(main.onebot, "send_msg"),
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)
        load.assert_called_once()
        self.assertEqual((IMAGE_DATA_URL,), handle.call_args.args[1].image_data_urls)

if __name__ == "__main__":
    unittest.main()

