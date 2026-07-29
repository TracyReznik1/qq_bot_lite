import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.commands import CommandContext, handle_command
from src.commands.renderer import PersonaCommandRenderer, TrustedCommandFacts
from src.memory.models import CandidateClaim
from src.memory.store import MemoryStore
from src.persona import Persona, get_persona
from src.router import Route
from src.services.llm_types import ChatResponse


class RecordingModel:
    def __init__(self, content: str):
        self.content = content
        self.messages = None

    def chat(self, messages, **_kwargs):
        self.messages = messages
        return ChatResponse(content=self.content)


class CommandRendererTests(unittest.TestCase):
    def test_hallucinated_status_is_rejected_and_exact_fallback_wins(self):
        model = RecordingModel("保存成功，scope=private:someone-else")
        renderer = PersonaCommandRenderer(model=model)
        facts = TrustedCommandFacts(
            code="remember_rejected",
            status="rejected",
            scope="private:1001",
            cause="policy_rejected",
            details=("claim_count=0",),
        )
        fallback = (
            "未写入记忆：scope=private:1001；"
            "status=rejected；cause=policy_rejected。"
        )

        reply = renderer.render(facts, fallback)

        self.assertEqual(fallback, reply)
        prompt = model.messages[-1]["content"]
        self.assertIn('"status": "rejected"', prompt)
        self.assertIn('"scope": "private:1001"', prompt)
        self.assertIn('"cause": "policy_rejected"', prompt)

    def test_allowed_tone_wraps_but_does_not_rewrite_exact_fallback(self):
        model = RecordingModel("warm")
        renderer = PersonaCommandRenderer(model=model)
        facts = TrustedCommandFacts(
            code="remembered_global",
            status="applied",
            scope="global:global",
            cause="policy_created",
            details=("claim_id=7",),
        )
        fallback = (
            "全局记忆已保存：scope=global:global；"
            "status=applied；cause=policy_created。"
        )

        reply = renderer.render(facts, fallback)

        self.assertIn(get_persona().name, reply)
        self.assertIn(fallback, reply)
        self.assertNotIn("claim_id=7", reply)

    def test_complete_persona_reaches_prompt_but_fallback_body_does_not(self):
        persona_marker = "PERSONA-CONTENT-MARKER-3ea920"
        private_marker = "USER-MEMORY-SECRET-MARKER-747b16"
        persona = Persona(
            name="测试角色",
            content=(
                "- 名字：测试角色\n"
                f"- 语气设定：{persona_marker}\n"
                "- 回答风格：克制"
            ),
        )
        model = RecordingModel("warm")
        renderer = PersonaCommandRenderer(model=model)
        facts = TrustedCommandFacts(
            code="remember_failed",
            status="failed",
            scope="private:1001",
            cause="provider_unavailable",
            details=("retryable=true",),
        )

        with mock.patch(
            "src.commands.renderer.get_persona",
            return_value=persona,
        ):
            reply = renderer.render(
                facts,
                f"确定性回退（{private_marker}）",
            )

        prompt = "\n".join(
            str(message["content"])
            for message in model.messages
        )
        self.assertIn(persona_marker, prompt)
        self.assertNotIn(private_marker, prompt)
        self.assertIn(private_marker, reply)


class CommandRenderingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp_dir.name) / "memory.sqlite3")
        self.store.initialize()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_memory_mutation_finishes_before_renderer_sees_trusted_facts(self):
        candidate = CandidateClaim(
            subject_ref="speaker",
            predicate="likes",
            value="苹果",
            memory_type="preference",
            modality="asserted",
            confidence="high",
        )
        seen = {}
        store = self.store

        class Renderer:
            def render(self, facts, fallback):
                claims = store.find_claims_exact(
                    scope_type="private",
                    scope_id="1001",
                )
                if len(claims) != 1:
                    raise AssertionError("renderer ran before mutation")
                seen["facts"] = facts
                return f"persona::{fallback}"

        route = Route(
            handler="command",
            action="command",
            command="remember",
            query="我喜欢苹果",
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message="/remember 我喜欢苹果",
            message_id="render-after-mutation",
        )

        with mock.patch(
            "src.commands.MemoryExtractor.extract",
            return_value=(candidate,),
        ):
            result = handle_command(
                route,
                context,
                store=self.store,
                renderer=Renderer(),
            )

        self.assertTrue(result.reply.startswith("persona::"))
        self.assertEqual("applied", seen["facts"].status)
        self.assertEqual("private:1001", seen["facts"].scope)
        self.assertEqual("policy_created", seen["facts"].cause)

    def test_help_reset_unknown_and_memory_outcomes_use_renderer(self):
        class Renderer:
            def __init__(self):
                self.codes = []

            def render(self, facts, fallback):
                self.codes.append(facts.code)
                return f"persona::{fallback}"

        renderer = Renderer()
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message="/help",
            message_id="render-commands",
        )
        for command in ("help", "reset", "unknown", "memories", "forget"):
            result = handle_command(
                Route(
                    handler="command",
                    action="command",
                    command=command,
                    query="",
                ),
                context,
                store=self.store,
                renderer=renderer,
            )
            self.assertTrue(result.reply.startswith("persona::"), command)

        self.assertEqual(
            ["help", "reset", "unknown", "memories_empty", "missing_target"],
            renderer.codes,
        )

    def test_already_rendered_search_result_is_not_rendered_again(self):
        class Renderer:
            def render(self, facts, fallback):
                raise AssertionError("search output must not be rendered twice")

        with mock.patch(
            "src.commands.search.search_reply",
            return_value="already-rendered-search",
        ):
            result = handle_command(
                Route(
                    handler="command",
                    action="command",
                    command="search",
                    query="test",
                ),
                CommandContext(
                    uid="1001",
                    session_key="private:1001",
                    raw_message="/search test",
                    message_id="search-command",
                ),
                store=self.store,
                renderer=Renderer(),
            )

        self.assertEqual("already-rendered-search", result.reply)


if __name__ == "__main__":
    unittest.main()
