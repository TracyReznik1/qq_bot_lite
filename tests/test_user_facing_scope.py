import unittest
from pathlib import Path
from unittest import mock

import src.main as main
from src.chat.prompt import build_system_prompt
from src.commands.help import help_text
from src.persona import PersonaConfigurationError, get_persona


ROOT = Path(__file__).resolve().parents[1]


class UserFacingScopeTests(unittest.TestCase):
    def setUp(self):
        service_patch = mock.patch.object(
            main,
            "get_memory_service",
            return_value=mock.Mock(),
        )
        service_patch.start()
        self.addCleanup(service_patch.stop)

    def test_help_mentions_image_understanding_not_generation(self):
        text = help_text()
        self.assertIn("发送图片", text)
        self.assertIn("识别", text)
        self.assertIn("不支持图片生成", text)

    def test_system_prompt_allows_input_images_but_forbids_output_images(self):
        prompt = build_system_prompt("private:1")
        self.assertIn("理解用户随消息提供的图片", prompt)
        self.assertIn("不能生成、编辑或主动发送图片", prompt)

    def test_readme_describes_keyword_search_and_image_input(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("图片理解", readme)
        self.assertIn("/search <关键词>", readme)
        self.assertIn("不提供独立 URL 直读", readme)

    def test_user_facing_search_paths_do_not_show_success_status_banner(self):
        for relative_path in (
            "src/search/renderer.py",
            "src/services/search_service.py",
            "src/commands/search.py",
        ):
            with self.subTest(relative_path=relative_path):
                content = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertNotIn("搜索状态：success", content)

    def test_help_uses_the_persona_identity(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        help_message = help_text()

        self.assertNotIn("ATRI", readme)
        self.assertIn(get_persona().name, help_message)

    def test_runtime_provider_failure_is_not_reported_as_configuration(self):
        event = {
            "post_type": "message",
            "message_type": "private",
            "user_id": 1001,
            "self_id": 9,
            "message_id": 7001,
            "raw_message": "/remember 私密内容",
            "message": "/remember 私密内容",
        }
        with (
            mock.patch.object(
                main,
                "handle_command",
                side_effect=RuntimeError("provider-private-marker"),
            ),
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
            self.assertLogs("qq-bot", level="ERROR") as captured,
        ):
            main.process_message(event)

        sent_text = send.call_args.args[1]
        logged = "\n".join(captured.output)
        self.assertNotIn("配置还没好", sent_text)
        self.assertIn("处理失败", sent_text)
        self.assertNotIn("Configuration error", logged)
        self.assertIn("Message handling failed", logged)
        self.assertNotIn("provider-private-marker", logged)

    def test_persona_configuration_failure_keeps_configuration_reply(self):
        event = {
            "post_type": "message",
            "message_type": "private",
            "user_id": 1001,
            "self_id": 9,
            "message_id": 7002,
            "raw_message": "/help",
            "message": "/help",
        }
        with (
            mock.patch.object(
                main,
                "handle_command",
                side_effect=PersonaConfigurationError("角色文件为空"),
            ),
            mock.patch.object(main.onebot, "send_msg") as send,
            mock.patch.object(main.time, "sleep"),
        ):
            main.process_message(event)

        self.assertIn("配置还没好", send.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
