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
from tests.search_fakes import make_analysis


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
            # A social/emotional skip uses a single plain answer call.
            from src.search.models import SearchTier, SkipReason, SearchTrace, RetrievalDecision, Factuality, Freshness, RiskLevel, RequestSource, SearchPipelineResult, Actionability, PotentialHarm
            skip = RetrievalDecision(
                SearchTier.SKIP, SkipReason.SOCIAL_OR_EMOTIONAL, False, (),
                frozenset(), Factuality.NON_FACTUAL, False, Freshness.NONE,
                RiskLevel.LOW, Actionability.NONE, PotentialHarm.NONE,
                None, None, (),
            )
            chat_service._search_orchestrator = SimpleNamespace(run=lambda req: SearchPipelineResult(
                skip, None, None, SearchTrace("req-1", RequestSource.CHAT, SearchTier.SKIP), None,
                analysis=make_analysis(skip_reason=SkipReason.SOCIAL_OR_EMOTIONAL),
            ))
            try:
                reply = chat_service.generate_reply("identity:test", "你好")
            finally:
                chat_service._search_orchestrator = None

        self.assertEqual("按身份回答", reply)
        for messages in fake_llm.messages:
            self.assertEqual("system", messages[0]["role"])
            self.assertIn("你扮演 小Q。", messages[0]["content"])
            self.assertIn(persona.content, messages[0]["content"])

    def test_search_failure_context_uses_current_role_not_atri(self):
        from src.search.models import (
            SearchTier,
            SearchTrace,
            RetrievalDecision,
            Factuality,
            Freshness,
            RiskLevel,
            RequestSource,
            SearchPipelineResult,
            Actionability,
            PotentialHarm,
            SearchFailureCode,
        )
        failed = RetrievalDecision(
            SearchTier.LIGHT, None, True, (), frozenset(), Factuality.FACTUAL,
            True, Freshness.NONE, RiskLevel.LOW, Actionability.NONE,
            PotentialHarm.NONE, SearchTier.LIGHT, None, (),
        )
        empty_plan = None
        orchestrator = SimpleNamespace(run=lambda req: SearchPipelineResult(
            failed, empty_plan, None, SearchTrace("req-1", RequestSource.CHAT, SearchTier.LIGHT),
            SearchFailureCode.PROVIDER_NOT_CONFIGURED,
            analysis=make_analysis(),
        ))
        with (
            patch.object(search_command, "normalize_search_query", return_value="测试"),
            patch.object(search_command, "generate_reply", return_value="无法确认") as generate,
        ):
            chat_service._search_orchestrator = orchestrator
            try:
                search_command.search_reply("测试", "private:1", "/search 测试")
            finally:
                chat_service._search_orchestrator = None

        self.assertEqual("无法确认", generate.return_value)

    def test_search_command_model_call_keeps_identity(self):
        fake_llm = ReplyingLlm()
        persona = Persona("小Q", "# 角色\n\n- 名字：小Q\n\n冷静、专业，先给结论。")
        from src.search.models import (
            SearchTier,
            SkipReason,
            SearchTrace,
            RetrievalDecision,
            Factuality,
            Freshness,
            RiskLevel,
            RequestSource,
            SearchPipelineResult,
            Actionability,
            PotentialHarm,
        )
        skip = RetrievalDecision(
            SearchTier.SKIP, SkipReason.SOCIAL_OR_EMOTIONAL, False, (),
            frozenset(), Factuality.NON_FACTUAL, False, Freshness.NONE,
            RiskLevel.LOW, Actionability.NONE, PotentialHarm.NONE,
            None, None, (),
        )
        orchestrator = SimpleNamespace(run=lambda req: SearchPipelineResult(
            skip, None, None, SearchTrace("req-1", RequestSource.CHAT, SearchTier.SKIP), None,
            analysis=make_analysis(skip_reason=SkipReason.SOCIAL_OR_EMOTIONAL),
        ))
        with (
            patch.object(search_command, "normalize_search_query", return_value="测试"),
            patch.object(chat_service, "llm", fake_llm),
            patch("src.chat.prompt.get_persona", return_value=persona),
            patch.object(chat_service, "_ensure_history_loaded"),
            patch("src.chat.prompt.MemoryRetriever", return_value=SimpleNamespace(retrieve=lambda ctx, query: [])),
            patch.object(chat_service, "append_history"),
        ):
            chat_service._search_orchestrator = orchestrator
            try:
                reply = search_command.search_reply("测试", "identity:search", "/search 测试")
            finally:
                chat_service._search_orchestrator = None

        self.assertEqual("按身份回答", reply)
        self.assertIn("你扮演 小Q。", fake_llm.messages[0][0]["content"])

    def test_multimodal_model_call_keeps_identity(self):
        fake_llm = ReplyingLlm()
        persona = Persona("小Q", "# 角色\n\n- 名字：小Q\n\n冷静、专业，先给结论。")
        from src.search.models import SearchTier, SkipReason, SearchTrace, RetrievalDecision, Factuality, Freshness, RiskLevel, RequestSource, SearchPipelineResult, Actionability, PotentialHarm
        skip = RetrievalDecision(
            SearchTier.SKIP, SkipReason.SOCIAL_OR_EMOTIONAL, False, (),
            frozenset(), Factuality.NON_FACTUAL, False, Freshness.NONE,
            RiskLevel.LOW, Actionability.NONE, PotentialHarm.NONE,
            None, None, (),
        )
        orchestrator = SimpleNamespace(run=lambda req: SearchPipelineResult(
            skip, None, None, SearchTrace("req-1", RequestSource.CHAT, SearchTier.SKIP), None,
            analysis=make_analysis(skip_reason=SkipReason.SOCIAL_OR_EMOTIONAL),
        ))
        with (
            patch.object(chat_service, "llm", fake_llm),
            patch("src.chat.prompt.get_persona", return_value=persona),
            patch.object(chat_service, "_ensure_history_loaded"),
            patch("src.chat.prompt.MemoryRetriever", return_value=SimpleNamespace(retrieve=lambda ctx, query: [])),
            patch.object(chat_service, "append_history"),
        ):
            chat_service._search_orchestrator = orchestrator
            try:
                reply = chat_service.generate_reply(
                    "identity:image",
                    "请看图",
                    image_data_urls=["data:image/png;base64,cG5n"],
                )
            finally:
                chat_service._search_orchestrator = None

        self.assertEqual("按身份回答", reply)
        self.assertIn("你扮演 小Q。", fake_llm.messages[0][0]["content"])
        self.assertEqual("image_url", fake_llm.messages[0][-1]["content"][1]["type"])


if __name__ == "__main__":
    unittest.main()
