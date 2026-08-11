"""Renderer tests: deterministic citations, disclosures, conflicts, QQ chunks."""

from __future__ import annotations

import importlib
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.search.models import (
    AnswerBlock,
    Claim,
    EvidenceBundle,
    EvidenceConflict,
    EvidenceConflictMember,
    EvidenceGapAnalysis,
    EvidenceItem,
    EvidenceState,
    ExcerptOrigin,
    Factuality,
    Freshness,
    GroundedDraft,
    RepairPlan,
    RequestSource,
    RetrievalDecision,
    RiskLevel,
    SearchFailureCode,
    SearchPlan,
    SearchQuery,
    SearchRoundKind,
    SearchTier,
    SearchTrace,
    SkipReason,
    SourceRelation,
    SupportLabel,
    TriggerCode,
    ValidationReport,
)
from src.search.validation import parse_grounded_draft, validate_and_filter
from tests.search_fakes import StaticSemanticVerifier


def renderer_module():
    try:
        return importlib.import_module("src.search.renderer")
    except ModuleNotFoundError:
        raise AssertionError("src.search.renderer must exist") from None


def models():
    return importlib.import_module("src.search.models")


def analyzed_decision(question: str, analyzer_content: str = "{}"):
    """Use Task 3's analyzer -> context router path in renderer fixtures."""
    from src.search.orchestrator import _with_legacy_analysis_metadata
    from src.search.router import LLMRequestAnalyzer, RetrievalBenefitRouter

    class StaticLLM:
        def chat(self, *_args, **_kwargs):
            return SimpleNamespace(content=analyzer_content)

    request = models().RetrievalRequest(question, request_source=RequestSource.CHAT)
    analysis = LLMRequestAnalyzer(StaticLLM()).analyze(request)
    routed = RetrievalBenefitRouter().decide(analysis.retrieval)
    return _with_legacy_analysis_metadata(routed, analysis), analysis


def decision(tier=SearchTier.STANDARD):
    m = models()
    return m.RetrievalDecision(
        tier, None, False, (), frozenset(), Factuality.FACTUAL,
        True, Freshness.NONE, RiskLevel.LOW, m.Actionability.NONE,
        m.PotentialHarm.NONE, tier, None, (),
    )


def query():
    return SearchQuery("q1", SearchRoundKind.INITIAL, __import__("src.search.models", fromlist=["QueryPurpose"]).QueryPurpose.DIRECT, "q")


def plan(required=("版本",)):
    m = models()
    d = decision()
    return SearchPlan(
        d, "当前版本是什么", m.PlanningStatus.NORMAL, (), None, (query(),),
        tuple(required), frozenset({SourceRelation.PRIMARY}), (), m.DEFAULT_TIER_BUDGETS[SearchTier.STANDARD],
    )


def item(eid="E1", url="https://example.com/page", title="Example Page"):
    m = models()
    return m.EvidenceItem(
        eid, "q1", "tavily", title, url, url, "example.com", "Example",
        SourceRelation.INDEPENDENT, None, None, None, "版本是3.2",
        ExcerptOrigin.PROVIDER_SNIPPET, "ok", 1.0, 1.0, True, Freshness.NONE,
        True, (), ("版本",), "g1",
    )


def bundle(
    evidence=(),
    state=EvidenceState.SUFFICIENT,
    conflicts=(),
    missing=(),
    *,
    structured_conflicts=(),
    limitations=(),
):
    m = models()
    requested_missing = tuple(missing)
    p = plan(tuple(dict.fromkeys(("版本", *requested_missing))))
    supports_evidence = bool(evidence) and state not in {
        EvidenceState.CONFLICTING,
        EvidenceState.INSUFFICIENT,
    }
    missing_labels = {
        topic.label for topic in p.required_topics if topic.label in requested_missing
    }
    assessments = tuple(
        m.TopicAssessment(
            topic.topic_id,
            m.FreshnessEligibility.NOT_REQUIRED,
            tuple(item.evidence_id for item in evidence)
            if supports_evidence and topic.label not in missing_labels
            else (),
        )
        for topic in p.required_topics
        if topic.material
    )
    supported_topic_ids = tuple(
        assessment.topic_id
        for assessment in assessments
        if assessment.supporting_evidence_ids
    )
    missing_topic_ids = tuple(
        assessment.topic_id
        for assessment in assessments
        if not assessment.supporting_evidence_ids
    )
    actual_missing = tuple(
        topic.label
        for topic in p.required_topics
        if topic.material and topic.topic_id in missing_topic_ids
    )
    actual_structured_conflicts = tuple(structured_conflicts)
    if conflicts and not actual_structured_conflicts:
        actual_structured_conflicts = tuple(
            m.EvidenceConflict(
                conflict_id,
                conflict_id,
                (
                    m.EvidenceConflictMember(
                        evidence[0].evidence_id,
                        "position-1",
                        None,
                        "contradicts",
                    ),
                    m.EvidenceConflictMember(
                        evidence[1].evidence_id,
                        "position-2",
                        None,
                        "contradicts",
                    ),
                ),
            )
            for conflict_id in conflicts
        )
    actual_conflict_groups = tuple(
        conflict.conflict_id for conflict in actual_structured_conflicts
    )
    return m.EvidenceBundle(
        "req-1", p.decision, p, (), tuple(e.evidence_id for e in evidence),
        m.EvidenceGapAnalysis(actual_missing, (), False, None, ()),
        m.RepairPlan(False, (), None), 1, tuple(evidence), state,
        actual_missing, (), actual_conflict_groups, tuple(limitations),
        actual_structured_conflicts,
        topic_assessments=assessments,
        supported_topic_ids=supported_topic_ids,
        missing_topic_ids=missing_topic_ids,
    )


def trace(route=SearchTier.STANDARD):
    return SearchTrace("req-1", RequestSource.CHAT, route)


def result(evidence=None, failure=None, route=SearchTier.STANDARD):
    p = plan()
    d = decision(route)
    if evidence is None and failure is not None and failure is not SearchFailureCode.PROVIDER_NOT_CONFIGURED:
        evidence = bundle((), state=EvidenceState.INSUFFICIENT)
    return models().SearchPipelineResult(
        d, p, evidence, trace(route), failure,
    )


def supported_report():
    draft = GroundedDraft(
        (AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
        (Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
        (),
        (),
        False,
    )
    return ValidationReport(
        draft,
        draft.answer_blocks,
        draft.claims,
        (),
        {},
        (),
    )


def high_consequence_result(evidence=None, failure=None):
    m = models()
    d = replace(
        decision(SearchTier.DEEP),
        trigger_codes=(TriggerCode.HIGH_CONSEQUENCE_ACTION,),
        risk=m.RiskLevel.HIGH,
        actionability=m.Actionability.PERSONALIZED,
        potential_harm=m.PotentialHarm.HIGH,
        final_reason_codes=(TriggerCode.HIGH_CONSEQUENCE_ACTION,),
    )
    source_plan = evidence.plan if evidence is not None else plan()
    p = replace(
        source_plan,
        decision=d,
        budget=m.DEFAULT_TIER_BUDGETS[SearchTier.DEEP],
    )
    if evidence is not None:
        evidence = replace(evidence, decision=d, plan=p)
    if evidence is None and failure not in {None, SearchFailureCode.PROVIDER_NOT_CONFIGURED}:
        evidence = bundle((), state=EvidenceState.INSUFFICIENT)
        evidence = replace(evidence, decision=d, plan=p)
    return m.SearchPipelineResult(d, p, evidence, trace(SearchTier.DEEP), failure)


class CitationRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = renderer_module()

    def test_numbers_sources_by_first_use(self):
        b = bundle((item("E1", "https://a.example.com"), item("E2", "https://b.example.com")), state=EvidenceState.SUFFICIENT)
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (Claim("C1", "B1", "版本是3.2", True, ("E2",)),),
            (), (), False,
        )
        report = ValidationReport(draft, (draft.answer_blocks[0],), draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(
            result(b), report, qq_limit=1700,
        )
        self.assertIn("[1]", rendered.text)
        self.assertIn("https://b.example.com", rendered.text)
        self.assertIn("来源：", rendered.text)
        self.assertEqual(rendered.used_evidence_ids, ("E2",))
        self.assertEqual(rendered.shown_source_urls, ("https://b.example.com",))

    def test_unused_source_suppressed(self):
        b = bundle((item("E1", "https://a.example.com"), item("E2", "https://b.example.com")), state=EvidenceState.SUFFICIENT)
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, (draft.answer_blocks[0],), draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b), report, qq_limit=1700)
        self.assertIn("https://a.example.com", rendered.text)
        self.assertNotIn("https://b.example.com", rendered.text)

    def test_nonexistent_evidence_rejected(self):
        b = bundle((item("E1", "https://a.example.com"),), state=EvidenceState.SUFFICIENT)
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (Claim("C1", "B1", "版本是3.2", True, ("E99",)),),
            (), (), False,
        )
        report = ValidationReport(draft, (draft.answer_blocks[0],), draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b), report, qq_limit=1700)
        self.assertNotIn("E99", rendered.text)
        self.assertEqual(rendered.used_evidence_ids, ())

    def test_model_written_source_section_stripped(self):
        b = bundle((item("E1", "https://a.example.com"),), state=EvidenceState.SUFFICIENT)
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "版本是3.2 来源：http://fake.example.com", ("C1",)),),
            (Claim("C1", "B1", "版本是3.2 来源：http://fake.example.com", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, (draft.answer_blocks[0],), draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b), report, qq_limit=1700)
        self.assertIn("版本是3.2", rendered.text)
        self.assertNotIn("来源：http://fake.example.com", rendered.text)


class FailureRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = renderer_module()

    def test_provider_not_configured_disclosure(self):
        rendered = self.module.render_search_reply(
            result(None, SearchFailureCode.PROVIDER_NOT_CONFIGURED), None, qq_limit=1700,
        )
        self.assertIn("当前搜索服务未配置", rendered.text)

    def test_ordinary_failure_keeps_failure_disclosure_without_risk_warning(self):
        warning = "温馨提示：以下内容不构成专业建议。"
        variant = "风险提示：本回答不能作为专业建议。"
        rendered = self.module.render_search_reply(
            result(None, SearchFailureCode.NO_RESULTS, route=SearchTier.LIGHT),
            None,
            knowledge_fallback_text=f"有限知识仍保留。{warning}\n{variant}",
            qq_limit=1700,
        )
        self.assertIn("在线检索未完成", rendered.text)
        self.assertIn("有限知识仍保留", rendered.text)
        self.assertNotIn("不构成专业建议", rendered.text)
        self.assertNotIn("不能作为专业建议", rendered.text)
        self.assertNotIn("温馨提示", rendered.text)
        self.assertNotIn("风险提示", rendered.text)

    def test_dynamic_high_consequence_without_evidence_refusal(self):
        rendered = self.module.render_search_reply(
            result(None, SearchFailureCode.PROVIDER_UNAVAILABLE, route=SearchTier.DEEP), None, qq_limit=1700,
        )
        self.assertIn("无法完成在线核验", rendered.text)

    def test_failure_has_zero_citations(self):
        rendered = self.module.render_search_reply(
            result(None, SearchFailureCode.NO_RESULTS), None, qq_limit=1700,
        )
        self.assertNotIn("来源：", rendered.text)
        self.assertEqual(rendered.shown_source_urls, ())
        self.assertNotIn("[1]", rendered.text)

    def test_explicit_search_failure_additive(self):
        from src.search.models import TriggerCode
        explicit_decision = models().RetrievalDecision(
            SearchTier.STANDARD, None, True, (TriggerCode.EXPLICIT_SEARCH,),
            frozenset(), Factuality.FACTUAL, True, Freshness.NONE,
            RiskLevel.LOW, models().Actionability.NONE, models().PotentialHarm.NONE,
            SearchTier.STANDARD, None, (TriggerCode.EXPLICIT_SEARCH,),
        )
        p = plan()
        empty_bundle = bundle((), state=EvidenceState.INSUFFICIENT)
        explicit_result = models().SearchPipelineResult(
            explicit_decision, p, empty_bundle, trace(), SearchFailureCode.NO_RESULTS,
        )
        rendered = self.module.render_search_reply(
            explicit_result, None, qq_limit=1700,
            knowledge_fallback_text="有限说明",
        )
        self.assertIn("你要求了在线搜索", rendered.text)


class ConflictRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = renderer_module()

    def test_conflict_positions_shown_separately(self):
        b = bundle(
            (
                item("E1", "https://a.example.com", title="Source A"),
                item("E2", "https://b.example.com", title="Source B"),
            ),
            state=EvidenceState.CONFLICTING,
            conflicts=("conflict:版本",),
        )
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "版本说法不一", ("C1",)),),
            (Claim("C1", "B1", "版本说法不一", True, ("E1", "E2")),),
            (), (), False,
        )
        report = ValidationReport(draft, (draft.answer_blocks[0],), draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b, SearchFailureCode.SOURCE_CONFLICT), report, qq_limit=1700)
        self.assertIn("来源之间存在未解决差异", rendered.text)
        self.assertIn("https://a.example.com", rendered.text)
        self.assertIn("https://b.example.com", rendered.text)
        self.assertNotIn("不能替代适当的专业判断", rendered.text)

    def test_conflict_section_mandatory_even_if_draft_omits(self):
        b = bundle(
            (item("E1", "https://a.example.com"), item("E2", "https://b.example.com")),
            state=EvidenceState.CONFLICTING,
            conflicts=("conflict:版本",),
        )
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, (draft.answer_blocks[0],), draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b, SearchFailureCode.SOURCE_CONFLICT), report, qq_limit=1700)
        self.assertIn("来源之间存在未解决差异", rendered.text)
        self.assertNotIn("不能替代适当的专业判断", rendered.text)

    def test_structured_conflict_renders_only_members_with_value_date_and_citation(self):
        first_date = datetime(2026, 7, 1, tzinfo=timezone.utc)
        second_date = datetime(2026, 7, 2, tzinfo=timezone.utc)
        conflict = EvidenceConflict(
            "conflict-1",
            "current_version",
            (
                EvidenceConflictMember("E1", "3.2", first_date, "contradicts"),
                EvidenceConflictMember("E2", "3.3", second_date, "claims_supersession"),
            ),
        )
        evidence = (
            replace(item("E1", "https://a.example.com", title="Source A"), published_at=first_date),
            replace(item("E2", "https://b.example.com", title="Source B"), published_at=second_date),
            item("E3", "https://not-a-member.example.com", title="Unrelated Source"),
        )
        b = bundle(
            evidence,
            state=EvidenceState.CONFLICTING,
            conflicts=("conflict-1",),
            structured_conflicts=(conflict,),
        )
        draft = GroundedDraft(
            (AnswerBlock("B1", "non_factual", "现有来源说法不一。", ()),),
            (), (), (), False,
        )
        report = ValidationReport(draft, draft.answer_blocks, (), (), {}, ())

        rendered = self.module.render_search_reply(
            result(b, SearchFailureCode.SOURCE_CONFLICT), report, qq_limit=1700,
        )

        self.assertIn("current_version", rendered.text)
        self.assertIn("Source A：3.2", rendered.text)
        self.assertIn("2026-07-01", rendered.text)
        self.assertIn("[1]", rendered.text)
        self.assertIn("Source B：3.3", rendered.text)
        self.assertIn("2026-07-02", rendered.text)
        self.assertIn("[2]", rendered.text)
        self.assertNotIn("Unrelated Source", rendered.text)
        self.assertNotIn("not-a-member.example.com", rendered.text)
        self.assertEqual(1, rendered.text.count("来源之间存在未解决差异"))
        self.assertEqual(("E1", "E2"), rendered.used_evidence_ids)
        self.assertNotIn("不能替代适当的专业判断", rendered.text)

    def test_validation_failed_still_renders_conflict_members_limitations_and_warning(self):
        conflict = EvidenceConflict(
            "conflict-1",
            "版本",
            (
                EvidenceConflictMember("E1", "3.2", datetime(2026, 7, 1, tzinfo=timezone.utc), "contradicts"),
                EvidenceConflictMember("E2", "3.3", datetime(2026, 7, 2, tzinfo=timezone.utc), "contradicts"),
            ),
        )
        b = bundle(
            (
                item("E1", "https://a.example.com", title="Source A"),
                item("E2", "https://b.example.com", title="Source B"),
                item("E3", "https://unrelated.example.com", title="Unrelated"),
            ),
            state=EvidenceState.CONFLICTING,
            structured_conflicts=(conflict,),
            limitations=("weak_source_topics",),
        )
        rendered = self.module.render_search_reply(
            high_consequence_result(b, SearchFailureCode.VALIDATION_FAILED),
            None,
            qq_limit=1700,
        )

        self.assertIn("回答未能通过证据核验", rendered.text)
        self.assertEqual(1, rendered.text.count("来源之间存在未解决差异"))
        self.assertIn("Source A：3.2", rendered.text)
        self.assertIn("Source B：3.3", rendered.text)
        self.assertIn("https://a.example.com", rendered.text)
        self.assertIn("https://b.example.com", rendered.text)
        self.assertNotIn("https://unrelated.example.com", rendered.text)
        self.assertIn("部分主题仅有较弱来源支持", rendered.text)
        self.assertEqual(1, rendered.text.count("搜索结果可能不完整或不准确"))

    def test_bundle_limitations_render_once_in_deterministic_order(self):
        b = bundle(
            (item(),),
            limitations=("single_source_authority", "single_source_authority", "weak_source_topics"),
        )
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, draft.answer_blocks, draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b), report, qq_limit=1700)
        self.assertEqual(1, rendered.text.count("单一权威来源"))
        self.assertEqual(1, rendered.text.count("较弱来源"))


class PartialRenderingTests(unittest.TestCase):
    def test_partial_scope_disclosure(self):
        module = renderer_module()
        b = bundle((item("E1", "https://a.example.com"),), state=EvidenceState.PARTIAL, missing=("历史",))
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, (draft.answer_blocks[0],), draft.claims, (), {}, ())
        rendered = module.render_search_reply(result(b, SearchFailureCode.PARTIAL_EVIDENCE), report, qq_limit=1700)
        self.assertIn("以下只回答已获得证据支持的部分", rendered.text)
        self.assertNotIn("不能替代适当的专业判断", rendered.text)

    def test_high_consequence_partial_keeps_scope_disclosure_and_one_warning(self):
        module = renderer_module()
        b = bundle(
            (item("E1", "https://a.example.com"),),
            state=EvidenceState.PARTIAL,
            missing=("历史",),
        )
        rendered = module.render_search_reply(
            high_consequence_result(b, SearchFailureCode.PARTIAL_EVIDENCE),
            supported_report(),
            qq_limit=1700,
        )
        self.assertIn("以下只回答已获得证据支持的部分", rendered.text)
        self.assertEqual(1, rendered.text.count("不能替代适当的专业判断"))


class QQSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = renderer_module()

    def test_split_respects_limit(self):
        text = "A" * 3000
        chunks = self.module.split_qq_reply(text, 500)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 500)
        self.assertEqual("".join(chunks), text)

    def test_split_keeps_urls_intact(self):
        url = "https://example.com/" + "x" * 100
        text = f"正文内容{url}"
        chunks = self.module.split_qq_reply(text, 100)
        self.assertTrue(any(url in chunk for chunk in chunks))

    def test_oversize_url_emitted_alone(self):
        url = "https://example.com/" + "y" * 300
        chunks = self.module.split_qq_reply(url, 100)
        self.assertIn(url, chunks)

    def test_oversize_url_does_not_absorb_trailing_text(self):
        url = "https://example.com/" + "y" * 300
        chunks = self.module.split_qq_reply(url + " TAIL", 50)
        self.assertTrue(any(url in chunk for chunk in chunks))
        self.assertFalse(any("TAIL" in chunk for chunk in chunks if url in chunk))

    def test_citation_numbers_single_bracket(self):
        b = bundle((item("E1", "https://a.example.com"), item("E2", "https://b.example.com")), state=EvidenceState.SUFFICIENT)
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (Claim("C1", "B1", "版本是3.2", True, ("E1", "E2")),),
            (), (), False,
        )
        report = ValidationReport(draft, (draft.answer_blocks[0],), draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b), report, qq_limit=1700)
        self.assertIn("版本是3.2[1][2]", rendered.text)
        self.assertNotIn("[12]", rendered.text)

    def test_model_source_section_discarded_from_heading_onward(self):
        b = bundle((item("E1", "https://a.example.com"),), state=EvidenceState.SUFFICIENT)
        block_text = "版本是3.2\n来源：\n[1] 假来源\nhttps://fake.example.com"
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", block_text, ("C1",)),),
            (Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, (draft.answer_blocks[0],), draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b), report, qq_limit=1700)
        self.assertNotIn("假来源", rendered.text)
        self.assertNotIn("fake.example.com", rendered.text)

    def test_cjk_punctuation_boundaries(self):
        text = "第一句。第二句。第三句。"
        chunks = self.module.split_qq_reply(text, 8)
        self.assertTrue(chunks)

    def test_body_then_sources_chunks(self):
        b = bundle((item("E1", "https://a.example.com"),), state=EvidenceState.SUFFICIENT)
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, (draft.answer_blocks[0],), draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b), report, qq_limit=45)
        self.assertTrue(rendered.chunks)
        for chunk in rendered.chunks:
            self.assertLessEqual(len(chunk), 45)
        # A source entry is atomic: its number/title and URL share one chunk.
        source_chunk = next(chunk for chunk in rendered.chunks if "https://a.example.com" in chunk)
        self.assertIn("[1] Example Page", source_chunk)

    def test_long_body_starts_atomic_source_entry_in_later_chunk(self):
        b = bundle((item("E1", "https://a.example.com/x", title="Source A"),))
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "正文" * 30, ("C1",)),),
            (Claim("C1", "B1", "事实", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, draft.answer_blocks, draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b), report, qq_limit=50)
        source_chunk = next(chunk for chunk in rendered.chunks if "https://a.example.com/x" in chunk)
        self.assertIn("[1] Source A", source_chunk)
        self.assertLessEqual(len(source_chunk), 50)

    def test_truncated_source_title_still_shares_its_url_chunk(self):
        url = "https://a.example.com/path"
        b = bundle((item("E1", url, title="A title that is much too long for this chunk"),))
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "事实", ("C1",)),),
            (Claim("C1", "B1", "事实", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, draft.answer_blocks, draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b), report, qq_limit=45)
        source_chunk = next(chunk for chunk in rendered.chunks if url in chunk)
        self.assertIn("[1] ", source_chunk)
        self.assertLessEqual(len(source_chunk), 45)

    def test_public_splitter_truncates_normal_source_title_but_keeps_url_atomic(self):
        url = "https://a.example/x"
        text = f"正文\n\n来源：\n[1] {'很长标题' * 12}\n{url}"
        chunks = self.module.split_qq_reply(text, 30)
        source_chunk = next(chunk for chunk in chunks if url in chunk)
        self.assertIn("[1] ", source_chunk)
        self.assertLessEqual(len(source_chunk), 30)

    def test_oversize_source_url_is_alone_and_does_not_absorb_title(self):
        url = "https://example.com/" + "z" * 120
        b = bundle((item("E1", url, title="Oversize Source"),))
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "事实", ("C1",)),),
            (Claim("C1", "B1", "事实", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, draft.answer_blocks, draft.claims, (), {}, ())
        rendered = self.module.render_search_reply(result(b), report, qq_limit=50)
        self.assertIn(url, rendered.chunks)
        self.assertFalse(any("Oversize Source" in chunk for chunk in rendered.chunks if url in chunk))


class RenderPlainReplyTests(unittest.TestCase):
    def test_render_plain_reply_never_adds_citations(self):
        module = renderer_module()
        t = trace()
        rendered = module.render_plain_reply("普通回答", trace=t, qq_limit=1700)
        self.assertEqual(rendered.text, "普通回答")
        self.assertEqual(rendered.shown_source_urls, ())
        self.assertEqual(rendered.used_evidence_ids, ())


class HighConsequenceWarningTests(unittest.TestCase):
    def test_empty_advisor_language_miss_warns_on_search_success_and_failure(self):
        module = renderer_module()
        warning = "重要提示：搜索结果可能不完整或不准确，不能替代适当的专业判断。"
        d, analysis = analyzed_decision("一条腿突然没力气，要去急诊吗？")
        self.assertIs(d.route, SearchTier.LIGHT)
        self.assertTrue(analysis.risk.high_consequence)
        p = replace(
            plan(),
            decision=d,
            budget=models().DEFAULT_TIER_BUDGETS[d.route],
        )
        b = replace(bundle((item(),)), decision=d, plan=p)
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", "请根据检索结果及时就医", ("C1",)),),
            (Claim("C1", "B1", "请根据检索结果及时就医", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(
            draft, draft.answer_blocks, draft.claims, (), {}, (),
        )
        success = models().SearchPipelineResult(
            d, p, b, trace(d.route), None,
        )
        failure = models().SearchPipelineResult(
            d, p, None, trace(d.route),
            SearchFailureCode.PROVIDER_NOT_CONFIGURED,
        )

        for path, search_result, validation in (
            ("success", success, report),
            ("failure", failure, None),
        ):
            with self.subTest(path=path):
                rendered = module.render_search_reply(
                    search_result, validation, qq_limit=1700,
                )
                self.assertEqual(1, rendered.text.count(warning))
                self.assertIn(warning, rendered.degradation_disclosures)

    def test_ordinary_success_has_answer_and_source_without_warning_or_status(self):
        module = renderer_module()
        rendered = module.render_search_reply(
            result(bundle((item(),))), supported_report(), qq_limit=1700,
        )

        self.assertIn("版本是3.2", rendered.text)
        self.assertIn("https://example.com/page", rendered.text)
        self.assertNotIn("搜索结果可能不完整或不准确", rendered.text)
        self.assertNotIn("检索完成", rendered.text)
        self.assertNotIn("搜索成功", rendered.text)
        self.assertNotIn("搜索状态：success", rendered.text)

    def test_ordinary_model_warning_is_removed_from_grounded_answer(self):
        module = renderer_module()
        warning = "重要提示：搜索结果可能不完整或不准确，不能替代适当的专业判断。"
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", f"版本是3.2\n{warning}", ("C1",)),),
            (Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, draft.answer_blocks, draft.claims, (), {}, ())

        rendered = module.render_search_reply(
            result(bundle((item(),))), report, qq_limit=1700,
        )

        self.assertIn("版本是3.2", rendered.text)
        self.assertIn("https://example.com/page", rendered.text)
        self.assertNotIn(warning, rendered.text)

    def test_ordinary_program_disclosures_are_removed_without_dropping_normal_text(self):
        module = renderer_module()
        risk_warning = "重要提示：以下内容不能替代适当的专业判断。"
        risk_warning_variant = "风险提示：本回答不构成专业建议。"
        quoted_fact = "资料原文含有“不能替代适当的专业判断”这句话。"
        draft = GroundedDraft(
            (
                AnswerBlock("B1", "factual", quoted_fact, ("C1",)),
                AnswerBlock(
                    "B2",
                    "non_factual",
                    "检索完成\n搜索成功\n搜索状态：success\n普通说明保留。检索完成\n"
                    f"{risk_warning}\n普通说明仍保留。{risk_warning_variant}\n"
                    "冒号说明保留：检索完成\n"
                    f"冒号说明仍保留：{risk_warning_variant}",
                    (),
                ),
            ),
            (Claim("C1", "B1", quoted_fact, True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, draft.answer_blocks, draft.claims, (), {}, ())

        rendered = module.render_search_reply(
            result(bundle((item(),))), report, qq_limit=1700,
        )

        self.assertIn(quoted_fact, rendered.text)
        self.assertIn("普通说明保留", rendered.text)
        self.assertIn("普通说明仍保留", rendered.text)
        self.assertIn("冒号说明保留", rendered.text)
        self.assertIn("冒号说明仍保留", rendered.text)
        self.assertIn("https://example.com/page", rendered.text)
        for forbidden in (
            "检索完成",
            "搜索成功",
            "搜索状态：success",
            risk_warning,
            risk_warning_variant,
            "重要提示",
            "风险提示",
        ):
            self.assertNotIn(forbidden, rendered.text)

    def test_status_prefix_atom_is_removed_without_dropping_following_text(self):
        module = renderer_module()
        draft = GroundedDraft(
            (
                AnswerBlock(
                    "B1",
                    "non_factual",
                    "检索完成：中文正文保留。\n搜索成功: ASCII 正文保留。\n"
                    "本次检索完成\n在线检索完成\n"
                    "本次检索完成后整理结果。完成某任务需要三步。检索完成率是九成。",
                    (),
                ),
            ),
            (), (), (), False,
        )
        report = ValidationReport(draft, draft.answer_blocks, (), (), {}, ())

        rendered = module.render_search_reply(
            result(bundle((item(),))), report, qq_limit=1700,
        )

        self.assertIn("中文正文保留", rendered.text)
        self.assertIn("ASCII 正文保留", rendered.text)
        self.assertIn("本次检索完成后整理结果", rendered.text)
        self.assertIn("完成某任务需要三步", rendered.text)
        self.assertIn("检索完成率是九成", rendered.text)
        self.assertNotIn("检索完成：", rendered.text)
        self.assertNotIn("搜索成功:", rendered.text)
        self.assertNotIn("在线检索完成", rendered.text)
        self.assertNotIn("本次检索完成", rendered.text.splitlines())

    def test_warning_atoms_are_removed_without_truncating_supported_facts(self):
        module = renderer_module()
        quoted_fact = "不能替代适当的专业判断是这份文件的免责声明内容。"
        embedded_fact = "版本是3.2"
        advice_fact = "专业建议栏目位于第二页。"
        block_text = (
            f"{quoted_fact}{advice_fact}"
            f"请注意：{embedded_fact}，不能代替专业建议。"
            "风险提示：本回答不能代替专业建议。"
        )
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", block_text, ("C1", "C2", "C3")),),
            (
                Claim("C1", "B1", quoted_fact, True, ("E1",)),
                Claim("C2", "B1", embedded_fact, True, ("E1",)),
                Claim("C3", "B1", advice_fact, True, ("E1",)),
            ),
            (), (), False,
        )
        report = ValidationReport(draft, draft.answer_blocks, draft.claims, (), {}, ())

        rendered = module.render_search_reply(
            result(bundle((item(),))), report, qq_limit=1700,
        )

        self.assertIn(quoted_fact, rendered.text)
        self.assertIn(embedded_fact, rendered.text)
        self.assertIn(advice_fact, rendered.text)
        self.assertNotIn("请注意", rendered.text)
        self.assertNotIn("风险提示", rendered.text)
        self.assertNotIn("不能代替专业建议", rendered.text)

    def test_high_consequence_model_warning_is_replaced_by_one_deterministic_warning(self):
        module = renderer_module()
        warning = "重要提示：搜索结果可能不完整或不准确，不能替代适当的专业判断。"
        model_warning = "温馨提示：以下内容不构成专业建议。风险提示：本回答不能作为专业建议。"
        b = bundle((item(),))
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", f"版本是3.2。{model_warning}", ("C1",)),),
            (Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(
            draft, draft.answer_blocks, draft.claims, (), {}, (),
        )

        rendered = module.render_search_reply(
            high_consequence_result(b), report, qq_limit=1700,
        )

        self.assertEqual(1, rendered.text.count(warning))
        self.assertEqual(1, rendered.text.count("不能替代适当的专业判断"))
        self.assertEqual(1, rendered.text.count("重要提示"))
        self.assertNotIn(model_warning, rendered.text)
        self.assertNotIn("温馨提示", rendered.text)
        self.assertNotIn("风险提示", rendered.text)
        self.assertNotIn("不能作为专业建议", rendered.text)
        self.assertEqual(1, rendered.degradation_disclosures.count(warning))

    def test_closed_skip_has_no_warning_but_no_web_high_consequence_gets_fixed_warning(self):
        module = renderer_module()
        warning = "重要提示：搜索结果可能不完整或不准确，不能替代适当的专业判断。"
        social = models().RetrievalDecision(
            SearchTier.SKIP,
            SkipReason.SOCIAL_OR_EMOTIONAL,
            False,
            (),
            frozenset(),
            Factuality.NON_FACTUAL,
            False,
            Freshness.NONE,
            RiskLevel.LOW,
            models().Actionability.NONE,
            models().PotentialHarm.NONE,
            None,
            None,
            (),
        )
        no_web_high_consequence, analysis = analyzed_decision(
            "不要联网，我发烧39度，该吃多少布洛芬？"
        )

        social_rendered = module.render_search_reply(
            models().SearchPipelineResult(
                social, None, None, trace(SearchTier.SKIP), None,
            ),
            None,
            knowledge_fallback_text="你好呀",
            qq_limit=1700,
        )
        no_web_rendered = module.render_search_reply(
            models().SearchPipelineResult(
                no_web_high_consequence, None, None, trace(SearchTier.SKIP), None,
            ),
            None,
            qq_limit=1700,
        )

        self.assertIs(social.route, SearchTier.SKIP)
        self.assertEqual("你好呀", social_rendered.text)
        self.assertNotIn(warning, social_rendered.text)
        self.assertIs(no_web_high_consequence.route, SearchTier.SKIP)
        self.assertTrue(analysis.risk.high_consequence)
        self.assertEqual(1, no_web_rendered.text.count("本次未联网核验"))
        self.assertNotIn("本次没有联网核验", no_web_rendered.text)
        self.assertNotIn(warning, no_web_rendered.text)

    def test_no_web_high_consequence_uses_no_web_warning_once(self):
        module = renderer_module()
        decision, analysis = analyzed_decision(
            "不要联网，我发烧39度，该吃多少布洛芬？"
        )
        search_result = models().SearchPipelineResult(
            decision,
            None,
            None,
            trace(SearchTier.SKIP),
            SearchFailureCode.USER_FORBID_WEB,
        )
        rendered = module.render_search_reply(
            search_result,
            None,
            knowledge_fallback_text=(
                "有限说明\n"
                "根据你的要求，本次没有联网核验；涉及当前状态的结论无法确认。"
            ),
            qq_limit=1700,
        )

        self.assertEqual(1, rendered.text.count("本次未联网核验"))
        self.assertNotIn("本次没有联网核验", rendered.text)
        self.assertNotIn("搜索结果可能不完整或不准确", rendered.text)
        self.assertIn("有限说明", rendered.text)
        self.assertTrue(analysis.risk.high_consequence)

    def test_warning_is_additive_to_grounded_answer_and_not_duplicated(self):
        module = renderer_module()
        warning = "重要提示：搜索结果可能不完整或不准确，不能替代适当的专业判断。"
        b = bundle((item(),))
        draft = GroundedDraft(
            (AnswerBlock("B1", "factual", f"剂量信息见来源。\n{warning}", ("C1",)),),
            (Claim("C1", "B1", "剂量信息见来源", True, ("E1",)),),
            (), (), False,
        )
        report = ValidationReport(draft, draft.answer_blocks, draft.claims, (), {}, ())
        rendered = module.render_search_reply(high_consequence_result(b), report, qq_limit=1700)
        self.assertEqual(1, rendered.text.count(warning))
        self.assertIn(warning, rendered.degradation_disclosures)
        self.assertIn("剂量信息见来源", rendered.text)
        self.assertIn("https://example.com/page", rendered.text)

    def test_warning_is_additive_to_search_failure_without_citations(self):
        module = renderer_module()
        rendered = module.render_search_reply(
            high_consequence_result(None, SearchFailureCode.PROVIDER_UNAVAILABLE),
            None,
            qq_limit=1700,
        )
        self.assertIn("搜索结果可能不完整或不准确", rendered.text)
        self.assertIn("无法完成在线核验", rendered.text)
        self.assertNotIn("来源：", rendered.text)
        self.assertEqual(1, rendered.text.count("搜索结果可能不完整或不准确"))

    def test_empty_and_malformed_advisor_warning_renders_on_success_and_failure(self):
        analyzer_contents = (
            ("empty", "{}"),
            ("malformed", "not-json"),
        )
        module = renderer_module()
        warning = "重要提示：搜索结果可能不完整或不准确，不能替代适当的专业判断。"
        for label, analyzer_content in analyzer_contents:
            d, analysis = analyzed_decision(
                "我发烧39度，该吃多少布洛芬？",
                analyzer_content,
            )
            self.assertTrue(analysis.risk.high_consequence)
            p = replace(
                plan(),
                decision=d,
                budget=models().DEFAULT_TIER_BUDGETS[d.route],
            )
            b = replace(bundle((item(),)), decision=d, plan=p)
            draft = GroundedDraft(
                (AnswerBlock("B1", "factual", "请核对剂量", ("C1",)),),
                (Claim("C1", "B1", "请核对剂量", True, ("E1",)),),
                (), (), False,
            )
            report = ValidationReport(draft, draft.answer_blocks, draft.claims, (), {}, ())
            success = models().SearchPipelineResult(d, p, b, trace(d.route), None)
            failure = models().SearchPipelineResult(
                d, p, None, trace(d.route), SearchFailureCode.PROVIDER_NOT_CONFIGURED,
            )
            with self.subTest(advisor=label, path="success"):
                rendered = module.render_search_reply(success, report, qq_limit=1700)
                self.assertEqual(1, rendered.text.count(warning))
            with self.subTest(advisor=label, path="failure"):
                rendered = module.render_search_reply(failure, None, qq_limit=1700)
                self.assertEqual(1, rendered.text.count(warning))


class RequestAnalysisPropagationTests(unittest.TestCase):
    """Renderer fallback copies must preserve Task 3 analysis identity."""

    def test_partial_and_conflict_fallbacks_keep_supplied_analysis(self):
        module = renderer_module()
        m = models()
        analysis = m.RequestAnalysis(
            m.RetrievalContext(
                False,
                None,
                m.Factuality.FACTUAL,
                True,
                (),
                m.SourceRequirement.ANY_RELEVANT,
            ),
            m.FreshnessContext(
                m.FreshnessRequirement.NOT_REQUIRED,
                None,
                None,
                None,
                None,
            ),
            m.RiskContext(False, False, False),
        )
        source = m.SearchPipelineResult(
            decision(),
            plan(),
            None,
            trace(),
            m.SearchFailureCode.PROVIDER_NOT_CONFIGURED,
            analysis,
        )
        copied_results = []

        def build_copied(*args):
            copied = SimpleNamespace(analysis=args[5] if len(args) > 5 else None)
            copied_results.append(copied)
            return copied

        def capture(copied, validation, *, qq_limit, **_kwargs):
            return m.RenderedReply("captured", (), (), (), ())

        with patch.object(module, "SearchPipelineResult", side_effect=build_copied):
            with patch.object(module, "render_search_reply", side_effect=capture):
                partial = module._render_partial(source, None, 1700)
                conflict = module._render_conflict(source, None, 1700)

        self.assertEqual("captured", partial.text)
        self.assertEqual("captured", conflict.text)
        self.assertEqual(2, len(copied_results))
        for copied in copied_results:
            self.assertIs(analysis, copied.analysis)


if __name__ == "__main__":
    unittest.main()
