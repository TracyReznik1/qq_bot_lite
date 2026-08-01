"""Evidence tests: relevance-gated admission, dedup, conflicts, sufficiency."""

from __future__ import annotations

import importlib
import unittest
from datetime import date, datetime, timezone

from src.search.models import (
    CandidateRelevance,
    EvidenceCandidate,
    EvidenceGapAnalysis,
    EvidenceState,
    ExcerptOrigin,
    Factuality,
    FetchedDocument,
    Freshness,
    ProviderHit,
    QueryPurpose,
    RequestSource,
    RetrievalDecision,
    RetrievalRequest,
    RiskLevel,
    SearchPlan,
    SearchQuery,
    SearchRoundKind,
    SearchTier,
    SourceRelation,
)
from tests.search_fakes import StaticEvidenceJudge


def evidence_module():
    try:
        return importlib.import_module("src.search.evidence")
    except ModuleNotFoundError:
        raise AssertionError("src.search.evidence must exist") from None


def decision(tier=SearchTier.STANDARD):
    m = __import__("src.search.models", fromlist=["RetrievalDecision"])
    return m.RetrievalDecision(
        tier, None, False, (), frozenset(), Factuality.FACTUAL,
        True, Freshness.NONE, RiskLevel.LOW, m.Actionability.NONE,
        m.PotentialHarm.NONE, tier, None, (),
    )


def query(qid="q1", text="什么是光合作用"):
    return SearchQuery(qid, SearchRoundKind.INITIAL, QueryPurpose.DIRECT, text)


def plan(required_topics=("定义",), route=SearchTier.STANDARD):
    d = decision(route)
    return SearchPlan(
        d, "什么是光合作用", __import__("src.search.models", fromlist=["PlanningStatus"]).PlanningStatus.NORMAL,
        ("光合作用",), None, (query(),), tuple(required_topics),
        frozenset({SourceRelation.PRIMARY, SourceRelation.INDEPENDENT}), (), _budget(route),
    )


def _budget(route):
    m = __import__("src.search.models", fromlist=["DEFAULT_TIER_BUDGETS"])
    return m.DEFAULT_TIER_BUDGETS[route]


def hit(url="https://example.com/page", title="Title", provider="tavily", content=None, published=None):
    return ProviderHit(
        provider=provider,
        query_id="q1",
        title=title,
        url=url,
        snippet=content,
        score=1.0,
        published_at=published,
        raw_content=None,
        quality_flags=(),
    )


def candidate(
    url="https://example.com/page",
    title="Title",
    relevance=None,
    provider="tavily",
    content="直接回答光合作用定义的正文。",
    published=None,
):
    return EvidenceCandidate(
        hit=hit(url=url, title=title, provider=provider, content=content, published=published),
        document=None,
        excerpt=content,
        excerpt_origin=ExcerptOrigin.PROVIDER_SNIPPET,
        extraction_status="search_result_snippet",
        safety_flags=(),
        content_reads_consumed=0,
    )


def judge_ok(candidate_id="C1", relevance="direct", relation="primary", supported=("定义",), conflict_key=None):
    return {
        "candidate_id": candidate_id,
        "relevance": relevance,
        "source_relation": relation,
        "publisher_entity_match": relation == "primary",
        "ownership_basis": "publisher matches query entity" if relation == "primary" else None,
        "supported_topics": list(supported),
        "conflict_key": conflict_key,
    }


class EvidenceAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    def test_relevance_is_admission_gate_before_source_relation(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", relevance="irrelevant", relation="primary"),
                "C2": judge_ok("C2", relevance="direct", relation="independent"),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        candidates = (
            candidate(url="https://official.example.com", title="Official Homepage"),
            candidate(url="https://independent.example.com/note", title="Independent Release Note"),
        )
        bundle = assembler.assemble(plan(), candidates)
        evidence = bundle.evidence_items
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].url, "https://independent.example.com/note")
        self.assertIs(evidence[0].source_relation, SourceRelation.INDEPENDENT)
        self.assertTrue(evidence[0].relevance_gate_passed)

    def test_irrelevant_official_docs_url_cannot_become_first_party(self):
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", relevance="irrelevant", relation="primary")}
        )
        assembler = self.module.EvidenceAssembler(judge)
        candidates = (candidate(url="https://unrelated.example/docs", title="Docs"),)
        bundle = assembler.assemble(plan(), candidates)
        self.assertEqual(bundle.evidence_items, ())
        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)

    def test_primary_requires_publisher_match_and_basis(self):
        judge = StaticEvidenceJudge(
            {
                "C1": {
                    "candidate_id": "C1",
                    "relevance": "direct",
                    "source_relation": "primary",
                    "publisher_entity_match": False,
                    "ownership_basis": None,
                    "supported_topics": ["定义"],
                    "conflict_key": None,
                }
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(), (candidate(url="https://docs.example.com"),))
        self.assertEqual(len(bundle.evidence_items), 1)
        self.assertIs(bundle.evidence_items[0].source_relation, SourceRelation.SECONDARY)

    def test_docs_path_never_primary_by_shape(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", relation="primary")})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(), (candidate(url="https://x.example.com/docs/guide"),))
        self.assertEqual(len(bundle.evidence_items), 1)
        # Even a primary claim cannot come from URL shape alone; judge controls it.

    def test_judge_failure_falls_back_deterministically_to_unknown(self):
        assembler = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": "garbage"})
        )
        bundle = assembler.assemble(plan(), (candidate(content="直接相关正文"),))
        self.assertEqual(len(bundle.evidence_items), 1)
        self.assertIs(bundle.evidence_items[0].source_relation, SourceRelation.UNKNOWN)
        self.assertTrue(bundle.evidence_items[0].relevance_gate_passed)


class EvidenceDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    def test_url_canonicalization_dedup(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("定义",)),
                "C2": judge_ok("C2", supported=("定义",)),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        candidates = (
            candidate(url="https://example.com/page"),
            candidate(url="https://example.com/page/"),
            candidate(url="https://example.com/page#fragment"),
        )
        bundle = assembler.assemble(plan(), candidates)
        # Redirects/fragments/trailing slash collapse to one canonical URL.
        self.assertEqual(len(bundle.evidence_items), 1)

    def test_syndicated_excerpts_group_into_one_independence_group(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("定义",)),
                "C2": judge_ok("C2", supported=("定义",)),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        text = "同一篇关于光合作用的转载文章内容完全相同。"
        candidates = (
            candidate(url="https://a.example.com/1", content=text),
            candidate(url="https://b.example.com/2", content=text),
        )
        bundle = assembler.assemble(plan(), candidates)
        self.assertEqual(len(bundle.evidence_items), 2)
        self.assertEqual(bundle.evidence_items[0].independence_group, bundle.evidence_items[1].independence_group)
        self.assertIsNotNone(bundle.evidence_items[0].independence_group)


class EvidenceStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    def test_sufficient_when_all_topics_supported(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(required_topics=("定义",)), (candidate(),))
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)

    def test_partial_lists_missing_topics(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(required_topics=("定义", "历史")), (candidate(),))
        self.assertIs(bundle.evidence_state, EvidenceState.PARTIAL)
        self.assertIn("历史", bundle.missing_claim_topics)

    def test_conflict_group_created_from_two_evidence_items(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("版本",), conflict_key="version"),
                "C2": judge_ok("C2", supported=("版本",), conflict_key="version"),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(
            plan(required_topics=("版本",)),
            (
                candidate(url="https://a.example.com/v", content="版本 1.0"),
                candidate(url="https://b.example.com/v", content="版本 2.0"),
            ),
        )
        self.assertIs(bundle.evidence_state, EvidenceState.CONFLICTING)
        self.assertTrue(bundle.conflict_groups)

    def test_model_memory_statement_does_not_create_conflict(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(), (candidate(),))
        # No second Evidence item exists; no conflict can be formed.
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual(bundle.conflict_groups, ())

    def test_dates_remain_separate(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        published = datetime(2024, 5, 1, tzinfo=timezone.utc)
        bundle = assembler.assemble(plan(), (candidate(published=published),))
        item = bundle.evidence_items[0]
        self.assertEqual(item.published_at, published)
        self.assertIsNotNone(item.retrieved_at)

    def test_single_source_authority_limitation_recorded(self):
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("动态",), relation="primary")}
        )
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(
            plan(required_topics=("动态",), route=SearchTier.DEEP),
            (candidate(),),
        )
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertTrue(any("single_source" in limitation for limitation in bundle.limitations))

    def test_ddgs_fallback_snippet_alone_cannot_meet_dynamic_topic(self):
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("动态",))}
        )
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(
            plan(required_topics=("动态",), route=SearchTier.DEEP),
            (
                candidate(
                    provider="ddgs",
                    content="搜索片段",
                    url="https://ddgs.example.com/page",
                ),
            ),
        )
        self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)


class EvidenceGapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    def test_gap_analysis_reports_missing_topics_and_repair_eligibility(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(required_topics=("定义", "历史")), (candidate(),))
        gap = assembler.analyze_gap(plan(required_topics=("定义", "历史")), bundle)
        self.assertIn("历史", gap.missing_claim_topics)
        self.assertTrue(gap.repair_eligible)
        self.assertEqual(gap.repair_purpose, "fill missing topic")

    def test_empty_gap_is_not_repairable(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(required_topics=("定义",)), (candidate(),))
        gap = assembler.analyze_gap(plan(required_topics=("定义",)), bundle)
        self.assertFalse(gap.repair_eligible)
        self.assertEqual(gap.missing_claim_topics, ())


if __name__ == "__main__":
    unittest.main()
