import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.chat import chat_service
from src.chat.prompt import build_system_prompt
import src.commands.search as search_command
from src.config import BASE_DIR, Config
from src.persona import Persona
from src.services.llm_types import ChatResponse


class CapturingLlm:
    def __init__(self):
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        if len(self.messages) == 1:
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": "search_1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"测试关键词"}',
                        },
                    }
                ],
            )
        return ChatResponse(content="按身份整理后的回答")


class ReplyingLlm:
    def __init__(self):
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        return ChatResponse(content="按身份回答")


class IdentityConfigurationTests(unittest.TestCase):
    def test_identity_environment_variables_are_not_configuration(self):
        with patch.dict(os.environ, {"BOT_NAME": "小Q", "BOT_PERSONA": "冷静、专业。"}):
            current = Config()

        self.assertEqual(BASE_DIR / "config" / "persona.md", current.persona_path)
        self.assertFalse(hasattr(current, "bot_name"))
        self.assertFalse(hasattr(current, "bot_persona"))

    def test_system_prompt_uses_full_persona_file_content(self):
        persona = Persona("小Q", "# 角色\n\n- 名字：小Q\n\n冷静、专业，先给结论。")
        with patch("src.chat.prompt.get_persona", return_value=persona):
            prompt = build_system_prompt("private:1")

        self.assertIn("你扮演 小Q。", prompt)
        self.assertIn(persona.content, prompt)
        for fixed_trait in ("温柔", "日系", "治愈", "偶尔玩梗"):
            self.assertNotIn(fixed_trait, prompt)

    def test_every_model_call_keeps_the_identity_system_message(self):
        fake_llm = CapturingLlm()
        persona = Persona("小Q", "# 角色\n\n- 名字：小Q\n\n冷静、专业，先给结论。")
        with (
            patch.object(chat_service, "llm", fake_llm),
            patch("src.chat.prompt.get_persona", return_value=persona),
            patch.object(chat_service, "_ensure_history_loaded"),
            patch.object(chat_service, "build_untrusted_context", return_value="[非可信上下文]暂无"),
            patch.object(chat_service, "run_tool", return_value="搜索结果"),
            patch.object(chat_service, "append_history"),
        ):
            reply = chat_service.generate_reply("identity:test", "请查测试关键词")

        self.assertEqual("按身份整理后的回答", reply)
        self.assertEqual(2, len(fake_llm.messages))
        for messages in fake_llm.messages:
            self.assertEqual("system", messages[0]["role"])
            self.assertIn("你扮演 小Q。", messages[0]["content"])
            self.assertIn(persona.content, messages[0]["content"])

    def test_search_failure_context_uses_current_role_not_atri(self):
        failed_result = SimpleNamespace(text="没有可靠结果")
        with (
            patch.object(search_command, "normalize_search_query", return_value="测试"),
            patch.object(search_command, "search", return_value=failed_result),
            patch.object(search_command, "has_search_results", return_value=False),
            patch.object(search_command, "generate_reply", return_value="无法确认") as generate,
        ):
            search_command.search_reply("测试", "private:1", "/search 测试")

        tool_context = generate.call_args.args[2]
        self.assertIn("按当前角色设定回答", tool_context)
        self.assertNotIn("ATRI", tool_context)

    def test_search_command_model_call_keeps_identity(self):
        fake_llm = ReplyingLlm()
        persona = Persona("小Q", "# 角色\n\n- 名字：小Q\n\n冷静、专业，先给结论。")
        result = SimpleNamespace(text="[1] 搜索结果")
        with (
            patch.object(search_command, "search", return_value=result),
            patch.object(search_command, "has_search_results", return_value=True),
            patch.object(chat_service, "llm", fake_llm),
            patch("src.chat.prompt.get_persona", return_value=persona),
            patch.object(chat_service, "_ensure_history_loaded"),
            patch.object(chat_service, "build_untrusted_context", return_value="[非可信上下文]搜索结果"),
            patch.object(chat_service, "append_history"),
        ):
            reply = search_command.search_reply("测试", "identity:search", "/search 测试")

        self.assertEqual("按身份回答", reply)
        self.assertIn("你扮演 小Q。", fake_llm.messages[0][0]["content"])

    def test_multimodal_model_call_keeps_identity(self):
        fake_llm = ReplyingLlm()
        persona = Persona("小Q", "# 角色\n\n- 名字：小Q\n\n冷静、专业，先给结论。")
        with (
            patch.object(chat_service, "llm", fake_llm),
            patch("src.chat.prompt.get_persona", return_value=persona),
            patch.object(chat_service, "_ensure_history_loaded"),
            patch.object(chat_service, "build_untrusted_context", return_value="[非可信上下文]暂无"),
            patch.object(chat_service, "append_history"),
        ):
            reply = chat_service.generate_reply(
                "identity:image",
                "请看图",
                image_data_urls=["data:image/png;base64,cG5n"],
            )

        self.assertEqual("按身份回答", reply)
        self.assertIn("你扮演 小Q。", fake_llm.messages[0][0]["content"])
        self.assertEqual("image_url", fake_llm.messages[0][-1]["content"][1]["type"])


if __name__ == "__main__":
    unittest.main()
