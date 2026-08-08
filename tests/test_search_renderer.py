"""Renderer tests: deterministic citations, disclosures, conflicts, QQ chunks."""

from __future__ import annotations

import importlib
import unittest
from dataclasses import replace
from datetime import datetime, timezone

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


def decision(tier=SearchTier.STANDARD):
    m = models()
    return m.RetrievalDecision(
        tier, None, False, (), frozenset(), Factuality.FACTUAL,
        True, Freshness.NONE, RiskLevel.LOW, m.Actionability.NONE,
        m.PotentialHarm.NONE, tier, None, (),
    )


def query():
    return SearchQuery("q1", SearchRoundKind.INITIAL, __import__("src.search.models", fromlist=["QueryPurpose"]).QueryPurpose.DIRECT, "q")


def plan():
    m = models()
    d = decision()
    return SearchPlan(
        d, "当前版本是什么", m.PlanningStatus.NORMAL, (), None, (query(),),
        (), frozenset({SourceRelation.PRIMARY}), (), m.DEFAULT_TIER_BUDGETS[SearchTier.STANDARD],
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
    p = plan()
    return m.EvidenceBundle(
        "req-1", p.decision, p, (), tuple(e.evidence_id for e in evidence),
        m.EvidenceGapAnalysis(missing, (), False, None, ()),
        m.RepairPlan(False, (), None), 1, tuple(evidence), state,
        tuple(missing), (), tuple(conflicts), tuple(limitations),
        tuple(structured_conflicts),
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
    p = replace(plan(), decision=d, budget=m.DEFAULT_TIER_BUDGETS[SearchTier.DEEP])
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
        from src.search.router import LLMRoutingAdvisor, RetrievalBenefitRouter
        from src.services.llm_types import ChatResponse
        from tests.search_fakes import StaticRouterAdvisor

        class MalformedLLM:
            def chat(self, *_args, **_kwargs):
                return ChatResponse(content="not-json")

        routers = (
            ("empty", RetrievalBenefitRouter(StaticRouterAdvisor({}))),
            ("malformed", RetrievalBenefitRouter(LLMRoutingAdvisor(MalformedLLM()))),
        )
        module = renderer_module()
        warning = "重要提示：搜索结果可能不完整或不准确，不能替代适当的专业判断。"
        for label, router in routers:
            d = router.decide(models().RetrievalRequest("我发烧39度，该吃多少布洛芬？"))
            p = replace(
                plan(),
                decision=d,
                budget=models().DEFAULT_TIER_BUDGETS[SearchTier.DEEP],
            )
            b = replace(bundle((item(),)), decision=d, plan=p)
            draft = GroundedDraft(
                (AnswerBlock("B1", "factual", "请核对剂量", ("C1",)),),
                (Claim("C1", "B1", "请核对剂量", True, ("E1",)),),
                (), (), False,
            )
            report = ValidationReport(draft, draft.answer_blocks, draft.claims, (), {}, ())
            success = models().SearchPipelineResult(d, p, b, trace(SearchTier.DEEP), None)
            failure = models().SearchPipelineResult(
                d, p, None, trace(SearchTier.DEEP), SearchFailureCode.PROVIDER_NOT_CONFIGURED,
            )
            with self.subTest(advisor=label, path="success"):
                rendered = module.render_search_reply(success, report, qq_limit=1700)
                self.assertEqual(1, rendered.text.count(warning))
            with self.subTest(advisor=label, path="failure"):
                rendered = module.render_search_reply(failure, None, qq_limit=1700)
                self.assertEqual(1, rendered.text.count(warning))


if __name__ == "__main__":
    unittest.main()
