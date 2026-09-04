import copy
import os
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

from src.chat import chat_service
from src.chat.prompt import build_system_prompt
from src.commands import CommandContext
import src.commands.search as search_command
from src.config import BASE_DIR, Config
from src.persona import Persona
from src.search.simple.models import SearchMode
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
        fake_llm = ReplyingLlm()
        persona = Persona("小Q", "# 角色\n\n- 名字：小Q\n\n冷静、专业，先给结论。")
        with (
            patch.object(chat_service, "llm", fake_llm),
            patch("src.chat.prompt.get_persona", return_value=persona),
            patch.object(chat_service, "_ensure_history_loaded"),
            patch("src.chat.prompt.MemoryRetriever", return_value=SimpleNamespace(retrieve=lambda ctx, query: [])),
            patch.object(chat_service, "append_history"),
        ):
            reply = chat_service.generate_reply("identity:test", "你好", mode=SearchMode.SKIP)

        self.assertEqual("按身份回答", reply)
        for messages in fake_llm.messages:
            self.assertEqual("system", messages[0]["role"])
            self.assertIn("你扮演 小Q。", messages[0]["content"])
            self.assertIn(persona.content, messages[0]["content"])

    def test_search_failure_context_uses_current_role_not_atri(self):
        with (
            patch.object(search_command, "normalize_search_query", return_value="测试"),
            patch.object(search_command, "generate_reply", return_value="无法确认") as generate,
        ):
            reply = search_command.search_reply(
                "测试",
                CommandContext(uid="1", session_key="private:1", raw_message="/search 测试"),
            )

        self.assertEqual("无法确认", reply)
        self.assertEqual(SearchMode.STANDARD, generate.call_args.kwargs["mode"])

    def test_search_command_model_call_keeps_identity(self):
        fake_llm = ReplyingLlm()
        persona = Persona("小Q", "# 角色\n\n- 名字：小Q\n\n冷静、专业，先给结论。")
        from src.search.simple.models import (
            SearchOutcome,
            SearchPlan,
            SearchQuery,
            SearchResult,
            SearchTrace,
            RequestSource,
        )

        plan = SearchPlan(SearchMode.STANDARD, (SearchQuery("q1", "测试"),))
        trace = SearchTrace("req-1", RequestSource.COMMAND, SearchMode.STANDARD)
        results = (SearchResult(result_id="1", title="title", url="https://example.com", excerpt="snippet", provider="tavily"),)
        mock_pipeline = mock.Mock()
        mock_pipeline.run.return_value = SearchOutcome(plan=plan, results=results, trace=trace)

        with (
            patch.object(chat_service, "llm", fake_llm),
            patch("src.chat.chat_service.get_simple_search_pipeline_for_chat", return_value=mock_pipeline),
            patch("src.chat.prompt.get_persona", return_value=persona),
            patch.object(chat_service, "_ensure_history_loaded"),
            patch("src.chat.prompt.MemoryRetriever", return_value=SimpleNamespace(retrieve=lambda ctx, query: [])),
            patch.object(chat_service, "append_history"),
        ):
            reply = search_command.search_reply(
                "测试",
                CommandContext(uid="1", session_key="identity:search", raw_message="/search 测试"),
            )

        self.assertIn("按身份回答", reply)
        self.assertIn("你扮演 小Q。", fake_llm.messages[0][0]["content"])

    def test_multimodal_model_call_keeps_identity(self):
        fake_llm = ReplyingLlm()
        persona = Persona("小Q", "# 角色\n\n- 名字：小Q\n\n冷静、专业，先给结论。")
        with (
            patch.object(chat_service, "llm", fake_llm),
            patch("src.chat.prompt.get_persona", return_value=persona),
            patch.object(chat_service, "_ensure_history_loaded"),
            patch("src.chat.prompt.MemoryRetriever", return_value=SimpleNamespace(retrieve=lambda ctx, query: [])),
            patch.object(chat_service, "append_history"),
        ):
            reply = chat_service.generate_reply(
                "identity:image",
                "请看图",
                image_data_urls=["data:image/png;base64,cG5n"],
                mode=SearchMode.SKIP,
            )

        self.assertEqual("按身份回答", reply)
        self.assertIn("你扮演 小Q。", fake_llm.messages[0][0]["content"])
        self.assertEqual("image_url", fake_llm.messages[0][-1]["content"][1]["type"])


if __name__ == "__main__":
    unittest.main()
