"""Program-first chat retrieval: ordinary chat uses the same orchestrator."""

from __future__ import annotations

import json
import unittest
from datetime import date
from unittest import mock

import src.chat.chat_service as chat_service
from src.chat.prompt import build_untrusted_context
from src.search.models import (
    EvidenceBundle,
    EvidenceGapAnalysis,
    EvidenceItem,
    EvidenceState,
    ExcerptOrigin,
    Factuality,
    Freshness,
    RepairPlan,
    RequestSource,
    RetrievalDecision,
    RetrievalRequest,
    RiskLevel,
    SearchPipelineResult,
    SearchPlan,
    SearchQuery,
    SearchRoundKind,
    SearchTier,
    SearchTrace,
    SkipReason,
    SourceRelation,
)
from src.services.llm_types import ChatResponse


def models():
    return __import__("src.search.models", fromlist=["RetrievalRequest"])


def decision(route=SearchTier.LIGHT):
    m = models()
    return m.RetrievalDecision(
        route, None, False, (), frozenset(), Factuality.FACTUAL,
        True, Freshness.NONE, RiskLevel.LOW, m.Actionability.NONE,
        m.PotentialHarm.NONE, route, None, (),
    )


def query():
    return SearchQuery("q1", SearchRoundKind.INITIAL, models().QueryPurpose.DIRECT, "q")


def plan(route=SearchTier.LIGHT):
    m = models()
    d = decision(route)
    return SearchPlan(
        d, "什么是光合作用", m.PlanningStatus.NORMAL, (), None, (query(),),
        (), frozenset({SourceRelation.PRIMARY}), (), m.DEFAULT_TIER_BUDGETS[route],
    )


def item(eid="E1", url="https://example.com/page", title="Title"):
    m = models()
    return m.EvidenceItem(
        eid, "q1", "tavily", title, url, url, "example.com", "Example",
        SourceRelation.INDEPENDENT, None, None, None, "光合作用定义",
        ExcerptOrigin.PROVIDER_SNIPPET, "ok", 1.0, 1.0, True, Freshness.NONE,
        True, (), ("定义",), "g1",
    )


def bundle(evidence=(), state=EvidenceState.SUFFICIENT):
    m = models()
    p = plan()
    return m.EvidenceBundle(
        "req-1", p.decision, p, (), tuple(e.evidence_id for e in evidence),
        m.EvidenceGapAnalysis((), (), False, None, ()),
        m.RepairPlan(False, (), None), 1, tuple(evidence), state,
        (), (), (), (),
    )


def search_result(route=SearchTier.LIGHT, evidence=None, failure=None, skip_reason=None):
    m = models()
    if skip_reason is not None:
        d = m.RetrievalDecision(
            SearchTier.SKIP, skip_reason, False, (), frozenset(),
            Factuality.NON_FACTUAL, False, Freshness.NONE, RiskLevel.LOW,
            m.Actionability.NONE, m.PotentialHarm.NONE, None, None, (),
        )
        return m.SearchPipelineResult(d, None, None, SearchTrace("req-1", RequestSource.CHAT, SearchTier.SKIP), failure)
    d = decision(route)
    p = plan(route)
    if evidence is None and failure is not None and failure is not m.SearchFailureCode.PROVIDER_NOT_CONFIGURED:
        evidence = bundle((), state=EvidenceState.INSUFFICIENT)
    return m.SearchPipelineResult(d, p, evidence, SearchTrace("req-1", RequestSource.CHAT, route), failure)


def _patch_memory():
    return mock.patch(
        "src.chat.prompt.MemoryRetriever",
        return_value=mock.Mock(retrieve=lambda ctx, query: ()),
    )


class SkipFlowTests(unittest.TestCase):
    def _run(self, result, text="你好"):
        with (
            _patch_memory(),
            mock.patch.object(chat_service.llm, "chat", return_value=ChatResponse(content="你好呀")) as llm_chat,
            mock.patch.object(chat_service, "append_history"),
        ):
            chat_service._search_orchestrator = mock.Mock(run=lambda req: result)
            try:
                reply = chat_service.generate_reply("private:1", text)
            finally:
                chat_service._search_orchestrator = None
        return reply, llm_chat

    def test_social_skip_zero_provider_and_normal_answer(self):
        reply, llm_chat = self._run(
            search_result(skip_reason=SkipReason.SOCIAL_OR_EMOTIONAL), "你好",
        )
        self.assertEqual("你好呀", reply)
        llm_chat.assert_called_once()
        self.assertNotIn("tools", llm_chat.call_args.kwargs)


class SearchFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = getattr(chat_service, "_search_orchestrator", None)

    def tearDown(self) -> None:
        chat_service._search_orchestrator = self._old

    def _run(self, result, text="什么是光合作用", force_search=False):
        fake_orchestrator = mock.Mock()
        fake_orchestrator.run.return_value = result
        chat_service._search_orchestrator = fake_orchestrator
        draft_json_payload = json.dumps({
            "answer_blocks": [{"block_id": "B1", "kind": "factual", "text": "版本是3.2", "claim_ids": ["C1"]}],
            "claims": [{"claim_id": "C1", "block_id": "B1", "text": "版本是3.2", "material": True, "evidence_ids": ["E1"]}],
            "limitations": [],
            "conflict_summary": [],
            "used_knowledge_fallback": False,
        })

        def _sequenced(*args, **kwargs):
            if not hasattr(_sequenced, "count"):
                _sequenced.count = 0
            _sequenced.count += 1
            if _sequenced.count == 1:
                return ChatResponse(content=draft_json_payload)
            return ChatResponse(content='{"C1": "supported"}')

        with (
            _patch_memory(),
            mock.patch.object(chat_service.llm, "chat", side_effect=_sequenced) as llm_chat,
            mock.patch.object(chat_service, "append_history"),
            mock.patch.object(chat_service, "finalize_search_trace"),
        ):
            reply = chat_service.generate_reply("private:1", text, force_search=force_search)
        return reply, llm_chat, fake_orchestrator

    def test_stable_fact_starts_orchestrator_even_if_model_knows(self):
        result = search_result(SearchTier.LIGHT, bundle((item(),)))
        reply, llm_chat, orch = self._run(result)
        orch.run.assert_called_once()
        self.assertIn("版本是3.2", reply)

    def test_evidence_payload_reaches_answer_model(self):
        result = search_result(SearchTier.LIGHT, bundle((item("E1", "https://example.com/page"),)))
        _reply, llm_chat, _orch = self._run(result)
        # Capture the messages sent to the answer-generation call (call #1).
        messages = llm_chat.call_args_list[0].args[0]
        joined = "\n".join(
            str(msg.get("content", ""))
            for msg in messages
            if isinstance(msg, dict)
        )
        self.assertIn("E1", joined)
        self.assertIn("https://example.com/page", joined)
        self.assertIn("光合作用定义", joined)

    def test_force_search_uses_command_source(self):
        result = search_result(SearchTier.LIGHT, bundle((item(),)))
        _, _, orch = self._run(result, force_search=True)
        request = orch.run.call_args.args[0]
        self.assertIs(request.request_source, RequestSource.COMMAND)

    def test_skipped_no_web_dynamic_emits_limitation(self):
        result = search_result(skip_reason=SkipReason.USER_FORBID_WEB)
        reply, _llm_chat = SkipFlowTests()._run(result, "不要联网，现在股票怎么样")
        self.assertIn("没有联网核验", reply)


class FailureFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = getattr(chat_service, "_search_orchestrator", None)

    def tearDown(self) -> None:
        chat_service._search_orchestrator = self._old

    def _run(self, result, text="什么是光合作用"):
        chat_service._search_orchestrator = mock.Mock(run=lambda req: result)
        with (
            _patch_memory(),
            mock.patch.object(chat_service.llm, "chat", return_value=ChatResponse(content="有限知识")) as llm_chat,
            mock.patch.object(chat_service, "append_history"),
            mock.patch.object(chat_service, "finalize_search_trace"),
        ):
            reply = chat_service.generate_reply("private:1", text)
        return reply, llm_chat

    def test_stable_failure_uses_fixed_disclosure_and_no_memory(self):
        result = search_result(SearchTier.LIGHT, failure=models().SearchFailureCode.NO_RESULTS)
        reply, llm_chat = self._run(result)
        self.assertIn("在线检索未完成", reply)
        self.assertEqual(1, llm_chat.call_count)


class NoToolsTests(unittest.TestCase):
    def test_no_llm_call_receives_tools(self):
        old = getattr(chat_service, "_search_orchestrator", None)
        chat_service._search_orchestrator = mock.Mock(run=lambda req: search_result(skip_reason=SkipReason.SOCIAL_OR_EMOTIONAL))
        try:
            with (
                _patch_memory(),
                mock.patch.object(chat_service.llm, "chat", return_value=ChatResponse(content="回答")) as llm_chat,
                mock.patch.object(chat_service, "append_history"),
            ):
                chat_service.generate_reply("private:1", "你好")
            for call in llm_chat.call_args_list:
                self.assertNotIn("tools", call.kwargs)
        finally:
            chat_service._search_orchestrator = old


class HistoryAppendTests(unittest.TestCase):
    def test_history_appended_exactly_once_with_final_reply(self):
        old = getattr(chat_service, "_search_orchestrator", None)
        chat_service._search_orchestrator = mock.Mock(run=lambda req: search_result(skip_reason=SkipReason.SOCIAL_OR_EMOTIONAL))
        try:
            with (
                _patch_memory(),
                mock.patch.object(chat_service.llm, "chat", return_value=ChatResponse(content="你好呀")),
                mock.patch.object(chat_service, "append_history") as append,
            ):
                chat_service.generate_reply("private:1", "你好")
            append.assert_called_once()
            self.assertEqual("你好呀", append.call_args.args[2])
        finally:
            chat_service._search_orchestrator = old


class PartialConflictFlowTests(unittest.TestCase):
    """C5/I11: partial and conflicting Evidence stay grounded, not generic fallback."""

    def _run_grounded(self, result):
        old = getattr(chat_service, "_search_orchestrator", None)
        chat_service._search_orchestrator = mock.Mock(run=lambda req: result)
        draft_payload = json.dumps({
            "answer_blocks": [{"block_id": "B1", "kind": "factual", "text": "版本是3.2", "claim_ids": ["C1"]}],
            "claims": [{"claim_id": "C1", "block_id": "B1", "text": "版本是3.2", "material": True, "evidence_ids": ["E1"]}],
            "limitations": [],
            "conflict_summary": [],
            "used_knowledge_fallback": False,
        })

        def _sequenced(*args, **kwargs):
            if not hasattr(_sequenced, "count"):
                _sequenced.count = 0
            _sequenced.count += 1
            if _sequenced.count == 1:
                return ChatResponse(content=draft_payload)
            return ChatResponse(content='{"C1": "supported"}')

        try:
            with (
                _patch_memory(),
                mock.patch.object(chat_service.llm, "chat", side_effect=_sequenced) as llm_chat,
                mock.patch.object(chat_service, "append_history"),
                mock.patch.object(chat_service, "finalize_search_trace"),
            ):
                reply = chat_service.generate_reply("private:1", "当前版本是什么")
        finally:
            chat_service._search_orchestrator = old
        return reply

    def test_partial_bundle_answers_supported_only(self):
        m = models()
        p = plan(SearchTier.STANDARD)
        partial_bundle = m.EvidenceBundle(
            "req-1", p.decision, p, (), ("E1",),
            m.EvidenceGapAnalysis(("历史",), (), False, None, ()),
            m.RepairPlan(False, (), None), 1, (item("E1", "https://a.example.com"),),
            m.EvidenceState.PARTIAL, ("历史",), (), (), (),
        )
        result = search_result(SearchTier.STANDARD, partial_bundle, failure=m.SearchFailureCode.PARTIAL_EVIDENCE)
        reply = self._run_grounded(result)
        self.assertIn("版本是3.2", reply)
        self.assertIn("以下只回答已获得证据支持的部分", reply)
        self.assertIn("https://a.example.com", reply)

    def test_conflict_bundle_shows_sources(self):
        m = models()
        p = plan(SearchTier.STANDARD)
        conflict_bundle = m.EvidenceBundle(
            "req-1", p.decision, p, (), ("E1", "E2"),
            m.EvidenceGapAnalysis((), (), False, None, ()),
            m.RepairPlan(False, (), None), 1,
            (item("E1", "https://a.example.com", title="Source A"), item("E2", "https://b.example.com", title="Source B")),
            m.EvidenceState.CONFLICTING, (), (), ("conflict:版本",), (),
        )
        result = search_result(SearchTier.STANDARD, conflict_bundle, failure=m.SearchFailureCode.SOURCE_CONFLICT)
        reply = self._run_grounded(result)
        self.assertIn("来源之间存在未解决差异", reply)
        self.assertIn("https://a.example.com", reply)
        self.assertIn("https://b.example.com", reply)


if __name__ == "__main__":
    unittest.main()
