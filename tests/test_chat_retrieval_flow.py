"""Program-first chat retrieval: ordinary chat uses the same orchestrator."""

from __future__ import annotations

import json
import time
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

    def test_no_web_high_consequence_uses_no_provider_or_answer_model(self):
        from src.search.orchestrator import SearchOrchestrator
        from src.search.router import RetrievalBenefitRouter
        from tests.search_fakes import StaticRouterAdvisor

        provider = mock.Mock()
        orchestrator = SearchOrchestrator(
            router=RetrievalBenefitRouter(StaticRouterAdvisor({})),
            planner=mock.Mock(),
            judge=mock.Mock(),
            providers=(provider,),
        )
        old = getattr(chat_service, "_search_orchestrator", None)
        chat_service._search_orchestrator = orchestrator
        warning = "重要提示：搜索结果可能不完整或不准确，不能替代适当的专业判断。"
        try:
            with (
                _patch_memory(),
                mock.patch.object(
                    chat_service.llm,
                    "chat",
                    return_value=ChatResponse(content="你应该服用99毫克布洛芬。"),
                ) as answer_chat,
                mock.patch.object(chat_service, "append_history"),
            ):
                reply = chat_service.generate_reply(
                    "private:1", "不要联网，我发烧39度，该吃多少布洛芬？"
                )
        finally:
            chat_service._search_orchestrator = old

        provider.search.assert_not_called()
        answer_chat.assert_not_called()
        self.assertIn("没有联网核验", reply)
        self.assertEqual(1, reply.count(warning))
        self.assertNotIn("99毫克", reply)


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
            if _sequenced.count == 2:
                return ChatResponse(content='{"spans": []}')
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

    def test_production_claim_discoverer_removes_hidden_fact_before_rendering(self):
        result = search_result(SearchTier.LIGHT, bundle((item(),)))
        fake_orchestrator = mock.Mock()
        fake_orchestrator.run.return_value = result
        chat_service._search_orchestrator = fake_orchestrator
        draft_payload = json.dumps({
            "answer_blocks": [
                {"block_id": "B1", "kind": "non_factual", "text": "顺带说版本是9.9", "claim_ids": []},
                {"block_id": "B2", "kind": "non_factual", "text": "请以来源为准。", "claim_ids": []},
            ],
            "claims": [],
            "limitations": [],
            "conflict_summary": [],
            "used_knowledge_fallback": False,
        }, ensure_ascii=False)
        discovery_payload = json.dumps({
            "spans": [{
                "block_id": "B1",
                "text": "版本是9.9",
                "material": True,
                "external_fact": True,
                "claim_id": None,
            }]
        }, ensure_ascii=False)
        with (
            _patch_memory(),
            mock.patch.object(
                chat_service.llm,
                "chat",
                side_effect=(ChatResponse(content=draft_payload), ChatResponse(content=discovery_payload)),
            ) as llm_chat,
            mock.patch.object(chat_service, "append_history"),
            mock.patch.object(chat_service, "finalize_search_trace"),
        ):
            reply = chat_service.generate_reply("private:1", "当前版本是什么")
        self.assertEqual(2, llm_chat.call_count)
        self.assertNotIn("版本是9.9", reply)
        self.assertIn("请以来源为准", reply)


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

    def test_deep_malformed_draft_returns_validation_failed_and_trace_agrees(self):
        result = search_result(SearchTier.DEEP, bundle((item(),)))
        chat_service._search_orchestrator = mock.Mock(run=lambda req: result)
        with (
            _patch_memory(),
            mock.patch.object(chat_service.llm, "chat", return_value=ChatResponse(content="not-json")),
            mock.patch.object(chat_service, "append_history") as append,
            mock.patch.object(chat_service, "finalize_search_trace") as finalize,
        ):
            reply = chat_service.generate_reply("private:1", "我该吃多少药")

        self.assertEqual("回答未能通过证据核验，已移除无法确认的内容。", reply)
        finalize.assert_called_once()
        finalized_result = finalize.call_args.args[0]
        self.assertIs(finalized_result.failure_code, models().SearchFailureCode.VALIDATION_FAILED)
        self.assertIs(finalized_result.trace.degradation_reason, models().SearchFailureCode.VALIDATION_FAILED)
        append.assert_called_once()
        self.assertTrue(append.call_args.args[2])

    def test_deep_cross_block_claim_mapping_is_validation_failed_without_citation_shift(self):
        result = search_result(SearchTier.DEEP, bundle((item(),)))
        chat_service._search_orchestrator = mock.Mock(run=lambda req: result)
        cross_block = json.dumps({
            "answer_blocks": [
                {"block_id": "B1", "kind": "factual", "text": "危险剂量是99毫克", "claim_ids": ["C1"]},
                {"block_id": "B2", "kind": "non_factual", "text": "请咨询专业人士", "claim_ids": []},
            ],
            "claims": [
                {"claim_id": "C1", "block_id": "B2", "text": "安全提醒", "material": True, "evidence_ids": ["E1"]},
            ],
        }, ensure_ascii=False)
        with (
            _patch_memory(),
            mock.patch.object(chat_service.llm, "chat", return_value=ChatResponse(content=cross_block)) as llm_chat,
            mock.patch.object(chat_service, "append_history"),
            mock.patch.object(chat_service, "finalize_search_trace") as finalize,
        ):
            reply = chat_service.generate_reply("private:1", "我该吃多少药")

        self.assertEqual(1, llm_chat.call_count)
        self.assertEqual("回答未能通过证据核验，已移除无法确认的内容。", reply)
        self.assertNotIn("99毫克", reply)
        self.assertNotIn("[1]", reply)
        finalized_result = finalize.call_args.args[0]
        self.assertIs(finalized_result.failure_code, models().SearchFailureCode.VALIDATION_FAILED)
        self.assertIs(finalized_result.trace.degradation_reason, models().SearchFailureCode.VALIDATION_FAILED)

    def test_deep_malformed_claim_discovery_is_validation_failed(self):
        result = search_result(SearchTier.DEEP, bundle((item(),)))
        chat_service._search_orchestrator = mock.Mock(run=lambda req: result)
        valid_draft = json.dumps({
            "answer_blocks": [
                {"block_id": "B1", "kind": "factual", "text": "版本是3.2", "claim_ids": ["C1"]},
            ],
            "claims": [
                {"claim_id": "C1", "block_id": "B1", "text": "版本是3.2", "material": True, "evidence_ids": ["E1"]},
            ],
        }, ensure_ascii=False)
        with (
            _patch_memory(),
            mock.patch.object(
                chat_service.llm,
                "chat",
                side_effect=(ChatResponse(content=valid_draft), ChatResponse(content="not-json")),
            ) as llm_chat,
            mock.patch.object(chat_service, "append_history"),
            mock.patch.object(chat_service, "finalize_search_trace") as finalize,
        ):
            reply = chat_service.generate_reply("private:1", "当前版本是什么")

        self.assertEqual(2, llm_chat.call_count)
        self.assertEqual("回答未能通过证据核验，已移除无法确认的内容。", reply)
        self.assertIs(
            finalize.call_args.args[0].failure_code,
            models().SearchFailureCode.VALIDATION_FAILED,
        )

    def test_malformed_draft_keeps_structured_conflict_and_limitation_disclosures(self):
        m = models()
        p = plan(SearchTier.STANDARD)
        conflict_bundle = m.EvidenceBundle(
            "req-1", p.decision, p, (), ("E1", "E2"),
            m.EvidenceGapAnalysis((), (), False, None, ()),
            m.RepairPlan(False, (), None), 1,
            (
                item("E1", "https://a.example.com", title="Source A"),
                item("E2", "https://b.example.com", title="Source B"),
            ),
            m.EvidenceState.CONFLICTING, (), (), ("conflict:版本",),
            ("weak_source_topics",),
            (
                m.EvidenceConflict(
                    "conflict-1", "版本",
                    (
                        m.EvidenceConflictMember("E1", "3.2", None, "contradicts"),
                        m.EvidenceConflictMember("E2", "3.3", None, "contradicts"),
                    ),
                ),
            ),
        )
        result = search_result(
            SearchTier.STANDARD,
            conflict_bundle,
            failure=m.SearchFailureCode.SOURCE_CONFLICT,
        )
        chat_service._search_orchestrator = mock.Mock(run=lambda req: result)
        with (
            _patch_memory(),
            mock.patch.object(chat_service.llm, "chat", return_value=ChatResponse(content="not-json")),
            mock.patch.object(chat_service, "append_history"),
            mock.patch.object(chat_service, "finalize_search_trace"),
        ):
            reply = chat_service.generate_reply("private:1", "当前版本是什么")

        self.assertIn("回答未能通过证据核验", reply)
        self.assertIn("来源之间存在未解决差异", reply)
        self.assertIn("Source A：3.2", reply)
        self.assertIn("Source B：3.3", reply)
        self.assertIn("部分主题仅有较弱来源支持", reply)
        self.assertIn("https://a.example.com", reply)
        self.assertIn("https://b.example.com", reply)


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

    def test_search_command_stores_raw_history_once_but_orchestrates_normalized_text(self):
        old = getattr(chat_service, "_search_orchestrator", None)
        orchestrator = mock.Mock()
        orchestrator.run.return_value = search_result(
            SearchTier.LIGHT,
            failure=models().SearchFailureCode.NO_RESULTS,
        )
        chat_service._search_orchestrator = orchestrator
        raw = "/search   什么是光合作用"
        try:
            with (
                _patch_memory(),
                mock.patch.object(chat_service.llm, "chat", return_value=ChatResponse(content="有限知识")),
                mock.patch.object(chat_service, "append_history") as append,
                mock.patch.object(chat_service, "finalize_search_trace"),
            ):
                chat_service.generate_reply(
                    "private:1",
                    "什么是光合作用",
                    force_search=True,
                    history_text=raw,
                )
            request = orchestrator.run.call_args.args[0]
            self.assertEqual("什么是光合作用", request.question)
            self.assertNotIn("/search", request.question)
            append.assert_called_once()
            self.assertEqual(raw, append.call_args.args[1])
        finally:
            chat_service._search_orchestrator = old


class TraceLifecycleTests(unittest.TestCase):
    def test_grounded_path_populates_answer_validation_render_counts_and_timestamps(self):
        result = search_result(SearchTier.LIGHT, bundle((item(),)))
        old = getattr(chat_service, "_search_orchestrator", None)
        chat_service._search_orchestrator = mock.Mock(run=lambda req: result)
        draft_payload = json.dumps({
            "answer_blocks": [{"block_id": "B1", "kind": "factual", "text": "版本是3.2", "claim_ids": ["C1"]}],
            "claims": [{"claim_id": "C1", "block_id": "B1", "text": "版本是3.2", "material": True, "evidence_ids": ["E1"]}],
            "limitations": [],
            "conflict_summary": [],
            "used_knowledge_fallback": False,
        })
        no_hidden_spans = json.dumps({"spans": []})
        try:
            with (
                _patch_memory(),
                mock.patch.object(
                    chat_service.llm,
                    "chat",
                    side_effect=(
                        ChatResponse(content=draft_payload),
                        ChatResponse(content=no_hidden_spans),
                        ChatResponse(content='{"C1":"supported"}'),
                    ),
                ),
                mock.patch.object(chat_service, "append_history"),
                mock.patch("src.search.orchestrator.logger.info"),
            ):
                reply = chat_service.generate_reply("private:1", "当前版本是什么")
        finally:
            chat_service._search_orchestrator = old

        self.assertIn("版本是3.2[1]", reply)
        trace = result.trace
        self.assertGreater(trace.answer_generation_latency_ms, 0)
        self.assertGreater(trace.structural_validation_latency_ms, 0)
        self.assertGreater(trace.semantic_validation_latency_ms, 0)
        self.assertGreater(trace.qq_render_latency_ms, 0)
        self.assertGreater(trace.total_response_latency_ms, 0)
        self.assertEqual(1, trace.claim_count)
        self.assertEqual(1, trace.supported_claim_count)
        self.assertEqual(1, trace.citation_count)
        self.assertFalse(trace.knowledge_fallback_used)
        self.assertIsNotNone(trace.response_started_at)
        self.assertIsNotNone(trace.response_finished_at)

    def test_delayed_claim_discovery_exception_finalizes_validation_failure_trace_once(self):
        result = search_result(SearchTier.DEEP, bundle((item(),)))
        old = getattr(chat_service, "_search_orchestrator", None)
        chat_service._search_orchestrator = mock.Mock(run=lambda req: result)
        draft_payload = json.dumps({
            "answer_blocks": [{"block_id": "B1", "kind": "factual", "text": "版本是3.2", "claim_ids": ["C1"]}],
            "claims": [{"claim_id": "C1", "block_id": "B1", "text": "版本是3.2", "material": True, "evidence_ids": ["E1"]}],
            "limitations": [],
            "conflict_summary": [],
            "used_knowledge_fallback": False,
        })
        llm_calls = 0

        def delayed_discovery_failure(*_args, **_kwargs):
            nonlocal llm_calls
            llm_calls += 1
            if llm_calls == 1:
                return ChatResponse(content=draft_payload)
            time.sleep(0.02)
            raise RuntimeError("boom")

        try:
            with (
                _patch_memory(),
                mock.patch.object(chat_service.llm, "chat", side_effect=delayed_discovery_failure),
                mock.patch.object(chat_service, "append_history") as append,
                mock.patch.object(
                    chat_service,
                    "finalize_search_trace",
                    wraps=chat_service.finalize_search_trace,
                ) as finalize,
                mock.patch("src.search.orchestrator.logger.info"),
            ):
                reply = chat_service.generate_reply("private:1", "当前版本是什么")
        finally:
            chat_service._search_orchestrator = old

        self.assertEqual("回答未能通过证据核验，已移除无法确认的内容。", reply)
        self.assertEqual(2, llm_calls)
        finalize.assert_called_once()
        append.assert_called_once()
        finalized_result = finalize.call_args.args[0]
        self.assertIs(finalized_result.failure_code, models().SearchFailureCode.VALIDATION_FAILED)
        self.assertIs(result.trace.degradation_reason, models().SearchFailureCode.VALIDATION_FAILED)
        self.assertTrue(result.trace.finalized)
        self.assertGreater(result.trace.semantic_validation_latency_ms, 0)
        self.assertEqual(1, result.trace.claim_count)
        self.assertEqual(0, result.trace.supported_claim_count)

    def test_trace_finalizes_once_even_when_answer_generation_raises(self):
        result = search_result(skip_reason=SkipReason.SOCIAL_OR_EMOTIONAL)
        old = getattr(chat_service, "_search_orchestrator", None)
        chat_service._search_orchestrator = mock.Mock(run=lambda req: result)
        def delayed_failure(*_args, **_kwargs):
            time.sleep(0.02)
            raise RuntimeError("boom")
        try:
            with (
                _patch_memory(),
                mock.patch.object(chat_service.llm, "chat", side_effect=delayed_failure),
                mock.patch.object(chat_service, "append_history") as append,
                mock.patch.object(
                    chat_service,
                    "finalize_search_trace",
                    wraps=chat_service.finalize_search_trace,
                ) as finalize,
                mock.patch("src.search.orchestrator.logger.info"),
            ):
                with self.assertRaises(RuntimeError):
                    chat_service.generate_reply("private:1", "你好")
            finalize.assert_called_once()
            append.assert_not_called()
            self.assertTrue(result.trace.finalized)
            self.assertGreater(result.trace.answer_generation_latency_ms, 0)
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
            if _sequenced.count == 2:
                return ChatResponse(content='{"spans": []}')
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
            (
                m.EvidenceConflict(
                    "conflict-1",
                    "版本",
                    (
                        m.EvidenceConflictMember("E1", "3.2", None, "contradicts"),
                        m.EvidenceConflictMember("E2", "3.3", None, "contradicts"),
                    ),
                ),
            ),
        )
        result = search_result(SearchTier.STANDARD, conflict_bundle, failure=m.SearchFailureCode.SOURCE_CONFLICT)
        reply = self._run_grounded(result)
        self.assertIn("来源之间存在未解决差异", reply)
        self.assertIn("https://a.example.com", reply)
        self.assertIn("https://b.example.com", reply)


if __name__ == "__main__":
    unittest.main()
