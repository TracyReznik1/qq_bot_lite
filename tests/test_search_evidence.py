"""Evidence tests: relevance-gated admission, dedup, conflicts, sufficiency."""

from __future__ import annotations

import importlib
import unittest
from dataclasses import replace
from datetime import date, datetime, timezone
from itertools import permutations

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


def judge_ok(
    candidate_id="C1",
    relevance="direct",
    relation="primary",
    supported=("定义",),
    conflict_key=None,
    conflict_value=None,
    conflict_relation=None,
    publisher=None,
):
    if conflict_key is not None and conflict_relation is None:
        conflict_relation = "contradicts"
    return {
        "candidate_id": candidate_id,
        "relevance": relevance,
        "source_relation": relation,
        "publisher_entity_match": relation == "primary",
        "ownership_basis": "publisher matches query entity" if relation == "primary" else None,
        "publisher": publisher,
        "supported_topics": list(supported),
        "conflict_key": conflict_key,
        "conflict_value": conflict_value,
        "conflict_relation": conflict_relation,
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

    def test_contradictory_primary_row_is_rejected(self):
        judge = StaticEvidenceJudge(
            {
                "C1": {
                    "candidate_id": "C1",
                    "relevance": "direct",
                    "source_relation": "primary",
                    "publisher_entity_match": False,
                    "ownership_basis": None,
                    "publisher": None,
                    "supported_topics": ["定义"],
                    "conflict_key": None,
                    "conflict_value": None,
                    "conflict_relation": None,
                }
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(), (candidate(url="https://docs.example.com"),))
        self.assertEqual(bundle.evidence_items, ())

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
        self.assertEqual(bundle.evidence_items, ())
        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)

    def test_only_direct_relevance_is_admitted(self):
        assembler = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": judge_ok("C1", relevance="contextual")})
        )
        bundle = assembler.assemble(plan(), (candidate(content="背景相关但不回答问题"),))
        self.assertEqual(bundle.evidence_items, ())

    def test_judge_receives_required_topics_and_remaining_time(self):
        calls = []

        class RecordingJudge:
            def judge(self, question, candidates, *, required_topics, timeout_seconds):
                calls.append((question, tuple(required_topics), timeout_seconds, len(candidates)))
                return {"C1": judge_ok("C1")}

        assembler = self.module.EvidenceAssembler(RecordingJudge())
        assembler.assemble(plan(required_topics=("定义", "历史")), (candidate(),), timeout_seconds=0.4)

        self.assertEqual(calls, [("什么是光合作用", ("定义", "历史"), 0.4, 1)])

    def test_partial_direct_judge_row_is_rejected_as_a_whole(self):
        class PartialJudge:
            def judge(self, *_args, **_kwargs):
                return {"C1": {"relevance": "direct", "supported_topics": ["定义"]}}

        bundle = self.module.EvidenceAssembler(PartialJudge()).assemble(
            plan(required_topics=("定义",)), (candidate(),)
        )

        self.assertEqual(bundle.evidence_items, ())
        self.assertEqual(bundle.evidence_state, EvidenceState.INSUFFICIENT)

    def test_missing_or_unknown_judge_field_rejects_the_complete_row(self):
        missing = judge_ok("C1")
        missing.pop("source_relation")
        extra = judge_ok("C2")
        extra["unreviewed_field"] = "must not be ignored"

        bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": missing, "C2": extra})
        ).assemble(
            plan(required_topics=("定义",)),
            (
                candidate(url="https://one.example/item"),
                candidate(url="https://two.example/item"),
            ),
        )

        self.assertEqual(bundle.evidence_items, ())

    def test_malformed_or_contradictory_conflict_contract_rejects_row(self):
        missing_value = judge_ok("C1", conflict_key="version", conflict_value=None)
        value_without_key = judge_ok("C2", conflict_key=None, conflict_value="2.0")

        bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": missing_value, "C2": value_without_key})
        ).assemble(
            plan(required_topics=("版本",)),
            (
                candidate(url="https://one.example/version", content="版本 1.0"),
                candidate(url="https://two.example/version", content="版本 2.0"),
            ),
        )

        self.assertEqual(bundle.evidence_items, ())
        self.assertEqual(bundle.conflicts, ())


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

    def test_same_domain_is_not_independent_when_wording_differs(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", relation="primary"),
                "C2": judge_ok("C2", relation="independent"),
            }
        )
        deep_candidates = tuple(
            replace(
                item,
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for item in (
                candidate(url="https://same.example/a", content="正文表述甲：光合作用定义。"),
                candidate(url="https://same.example/b", content="完全不同措辞乙：植物把光转成化学能。"),
            )
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("定义",), route=SearchTier.DEEP),
            deep_candidates,
        )
        self.assertEqual(
            bundle.evidence_items[0].independence_group,
            bundle.evidence_items[1].independence_group,
        )
        self.assertIn("single_source_authority", bundle.limitations)

    def test_same_registrable_parent_subdomains_cannot_corroborate(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", relation="primary", supported=("动态",)),
                "C2": judge_ok("C2", relation="independent", supported=("动态",)),
            }
        )
        candidates = tuple(
            replace(
                candidate(url=url, content=text),
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for url, text in (
                ("https://www.vendor.example/status", "厂商主站确认当前状态。"),
                ("https://news.vendor.example/report", "新闻子站使用完全不同的措辞报道。"),
            )
        )

        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("动态",), route=SearchTier.DEEP), candidates
        )

        self.assertEqual(
            bundle.evidence_items[0].independence_group,
            bundle.evidence_items[1].independence_group,
        )
        self.assertIn("single_source_authority", bundle.limitations)

    def test_same_publisher_across_domains_cannot_corroborate(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok(
                    "C1", relation="primary", supported=("动态",), publisher="Example Wire"
                ),
                "C2": judge_ok(
                    "C2", relation="independent", supported=("动态",), publisher="Example Wire"
                ),
            }
        )
        candidates = tuple(
            replace(
                candidate(url=url, content=text),
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for url, text in (
                ("https://authority.example/status", "权威页面陈述动态状态。"),
                ("https://affiliate.invalid/report", "关联出版方采用不同措辞。"),
            )
        )

        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("动态",), route=SearchTier.DEEP), candidates
        )

        self.assertEqual(
            bundle.evidence_items[0].publisher,
            bundle.evidence_items[1].publisher,
        )
        self.assertEqual(
            bundle.evidence_items[0].independence_group,
            bundle.evidence_items[1].independence_group,
        )
        self.assertIn("single_source_authority", bundle.limitations)

    def test_domain_to_publisher_bridge_is_transitive_in_every_order(self):
        specs = (
            ("https://www.vendor.example/status", "Vendor Org", (), "厂商主站确认当前状态。"),
            ("https://news.vendor.example/report", "Affiliate Media", (), "新闻子站采用不同措辞报道。"),
            ("https://affiliate.invalid/report", "Affiliate Media", (), "关联出版方独立撰写另一篇报道。"),
        )

        for ordered_specs in permutations(specs):
            with self.subTest(order=tuple(spec[0] for spec in ordered_specs)):
                bundle = self._deep_bridge_bundle(ordered_specs)
                self.assertEqual(len(bundle.evidence_items), 3)
                self.assertEqual(
                    len({item.independence_group for item in bundle.evidence_items}),
                    1,
                )
                self.assertIn("single_source_authority", bundle.limitations)

    def test_publisher_to_canonical_marker_bridge_is_transitive_in_every_order(self):
        specs = (
            ("https://publisher.example/a", "Shared Publisher", (), "来源甲采用独立表述。"),
            (
                "https://bridge.invalid/b",
                "Shared Publisher",
                ("canonical_source:wire-story-42",),
                "桥接来源乙采用另一种表述。",
            ),
            (
                "https://copy.invalid/c",
                "Copy Publisher",
                ("canonical_source:wire-story-42",),
                "转载来源丙采用第三种表述。",
            ),
        )

        for ordered_specs in permutations(specs):
            with self.subTest(order=tuple(spec[0] for spec in ordered_specs)):
                bundle = self._deep_bridge_bundle(ordered_specs)
                self.assertEqual(
                    len({item.independence_group for item in bundle.evidence_items}),
                    1,
                )

    def test_text_to_syndication_bridge_is_transitive_in_every_order(self):
        syndicated_text = "同一篇转载文章逐字说明当前动态状态与发布日期。"
        specs = (
            ("https://origin.example/a", "Origin Publisher", (), syndicated_text),
            (
                "https://bridge.invalid/b",
                "Bridge Publisher",
                ("syndication_source:wire-story-99",),
                syndicated_text,
            ),
            (
                "https://copy.invalid/c",
                "Copy Publisher",
                ("syndication_source:wire-story-99",),
                "转载方针对同一稿件另写摘要。",
            ),
        )

        for ordered_specs in permutations(specs):
            with self.subTest(order=tuple(spec[0] for spec in ordered_specs)):
                bundle = self._deep_bridge_bundle(ordered_specs)
                self.assertEqual(
                    len({item.independence_group for item in bundle.evidence_items}),
                    1,
                )

    def _deep_bridge_bundle(self, specs):
        judged = {
            f"C{index}": judge_ok(
                f"C{index}",
                relation="primary" if index == 1 else "independent",
                supported=("动态",),
                publisher=publisher,
            )
            for index, (_url, publisher, _flags, _text) in enumerate(specs, 1)
        }
        candidates = tuple(
            replace(
                candidate(
                    url=url,
                    content=text,
                ),
                hit=replace(candidate(url=url, content=text).hit, quality_flags=flags),
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for url, _publisher, flags, text in specs
        )
        return self.module.EvidenceAssembler(StaticEvidenceJudge(judged)).assemble(
            plan(required_topics=("动态",), route=SearchTier.DEEP),
            candidates,
        )


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
                "C1": judge_ok(
                    "C1", supported=("版本",), conflict_key="version",
                    conflict_value="1.0",
                ),
                "C2": judge_ok(
                    "C2", supported=("版本",), conflict_key="version",
                    conflict_value="2.0",
                    conflict_relation="claims_supersession",
                ),
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
        self.assertEqual(bundle.conflicts[0].conflict_key, "version")
        self.assertEqual(
            [(member.evidence_id, member.value) for member in bundle.conflicts[0].members],
            [("E1", "1.0"), ("E2", "2.0")],
        )
        self.assertEqual(bundle.conflicts[0].members[1].relation, "claims_supersession")

    def test_conflict_members_preserve_each_publication_date(self):
        older = datetime(2025, 1, 1, tzinfo=timezone.utc)
        newer = datetime(2026, 1, 1, tzinfo=timezone.utc)
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("版本",), conflict_key="version", conflict_value="1.0"),
                "C2": judge_ok("C2", supported=("版本",), conflict_key="version", conflict_value="2.0"),
            }
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("版本",)),
            (
                candidate(url="https://a.example/v", content="版本 1.0", published=older),
                candidate(url="https://b.example/v", content="版本 2.0", published=newer),
            ),
        )
        self.assertEqual(
            [member.published_at for member in bundle.conflicts[0].members],
            [older, newer],
        )

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
        authoritative = replace(
            candidate(),
            excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
            extraction_status="page_extract",
            content_reads_consumed=1,
        )
        bundle = assembler.assemble(
            plan(required_topics=("动态",), route=SearchTier.DEEP),
            (authoritative,),
        )
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertTrue(any("single_source" in limitation for limitation in bundle.limitations))

    def test_deep_requires_primary_plus_genuinely_independent_corroboration(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("动态",), relation="primary"),
                "C2": judge_ok("C2", supported=("动态",), relation="independent"),
            }
        )
        candidates = tuple(
            replace(
                candidate(url=url, content=text),
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for url, text in (
                ("https://authority.example/status", "权威来源确认动态状态。"),
                ("https://independent.example/report", "独立来源交叉确认该动态状态。"),
            )
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("动态",), route=SearchTier.DEEP), candidates
        )
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertNotIn("single_source_authority", bundle.limitations)

    def test_deep_two_independent_secondary_sources_do_not_replace_primary(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("动态",), relation="independent"),
                "C2": judge_ok("C2", supported=("动态",), relation="independent"),
            }
        )
        candidates = tuple(
            replace(
                candidate(url=url, content="动态页面正文"),
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for url in ("https://a.example/status", "https://b.example/report")
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("动态",), route=SearchTier.DEEP), candidates
        )
        self.assertIsNot(bundle.evidence_state, EvidenceState.SUFFICIENT)

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

    def test_tavily_snippet_after_failed_fetch_is_not_strong_dynamic_support(self):
        failed_document = FetchedDocument(
            "https://tavily.example/status",
            "https://tavily.example/status",
            None,
            None,
            None,
            "request_error",
            (),
        )
        weak = replace(
            candidate(provider="tavily", content="当前状态片段", url="https://tavily.example/status"),
            document=failed_document,
            extraction_status="search_result_snippet_after_fetch_failure",
            content_reads_consumed=1,
        )
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("动态",), relation="primary")}
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("动态",), route=SearchTier.DEEP), (weak,)
        )
        self.assertIsNot(bundle.evidence_state, EvidenceState.SUFFICIENT)

    def test_standard_high_risk_failed_fetch_snippet_cannot_satisfy_topic(self):
        risky_plan = plan(required_topics=("剂量",), route=SearchTier.STANDARD)
        risky_plan = replace(
            risky_plan,
            decision=replace(risky_plan.decision, risk=RiskLevel.HIGH),
        )
        weak = replace(
            candidate(content="药物剂量搜索片段"),
            extraction_status="search_result_snippet_after_fetch_failure",
            content_reads_consumed=1,
        )
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("剂量",), relation="primary")}
        )

        bundle = self.module.EvidenceAssembler(judge).assemble(risky_plan, (weak,))

        self.assertIsNot(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIn("剂量", bundle.missing_claim_topics)

    def test_standard_high_freshness_failed_fetch_snippet_cannot_satisfy_topic(self):
        dynamic_plan = plan(required_topics=("当前版本",), route=SearchTier.STANDARD)
        dynamic_plan = replace(
            dynamic_plan,
            decision=replace(dynamic_plan.decision, freshness=Freshness.HIGH),
        )
        weak = replace(
            candidate(content="当前版本是 2.0 的搜索片段"),
            extraction_status="search_result_snippet_after_fetch_failure",
            content_reads_consumed=1,
        )
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("当前版本",), relation="primary")}
        )

        bundle = self.module.EvidenceAssembler(judge).assemble(dynamic_plan, (weak,))

        self.assertIsNot(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIn("当前版本", bundle.missing_claim_topics)

    def test_two_weak_dynamic_snippets_cannot_form_material_conflict(self):
        weak_candidates = tuple(
            replace(
                candidate(url=url, content=text),
                extraction_status="search_result_snippet_after_fetch_failure",
                content_reads_consumed=1,
            )
            for url, text in (
                ("https://a.example/version", "当前版本 1.0"),
                ("https://b.example/version", "当前版本 2.0"),
            )
        )
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok(
                    "C1", supported=("当前版本",), conflict_key="version", conflict_value="1.0"
                ),
                "C2": judge_ok(
                    "C2", supported=("当前版本",), conflict_key="version", conflict_value="2.0"
                ),
            }
        )

        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("当前版本",), route=SearchTier.DEEP), weak_candidates
        )

        self.assertNotEqual(bundle.evidence_state, EvidenceState.CONFLICTING)
        self.assertEqual(bundle.conflict_groups, ())
        self.assertEqual(bundle.conflicts, ())
        self.assertIn("当前版本", bundle.missing_claim_topics)

    def test_exact_required_topic_label_is_supported(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok(
                    "C1", supported=("Rust 和 Go 的并发模型差异",), relation="primary"
                )
            }
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("Rust 和 Go 的并发模型差异",)),
            (candidate(content="Rust 和 Go 的并发模型存在明确差异。"),),
        )
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual(bundle.missing_claim_topics, ())

    def test_broad_subtopic_cannot_satisfy_composite_required_topic(self):
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("Go",), relation="primary")}
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("Rust 和 Go 的并发模型差异",)),
            (candidate(content="这里只介绍 Go，不比较 Rust，也未讨论并发模型。"),),
        )

        self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIn("Rust 和 Go 的并发模型差异", bundle.missing_claim_topics)

    def test_overlapping_product_name_is_not_a_topic_alias(self):
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("JavaScript",), relation="primary")}
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("Java",)),
            (candidate(content="本文只讨论 JavaScript。"),),
        )

        self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIn("Java", bundle.missing_claim_topics)

    def test_cjk_narrow_label_cannot_satisfy_full_required_topic(self):
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("苹果",), relation="primary")}
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("苹果公司的季度营收",)),
            (candidate(content="这里只介绍苹果这一名称。"),),
        )

        self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIn("苹果公司的季度营收", bundle.missing_claim_topics)

    def test_japanese_narrow_label_cannot_satisfy_full_required_topic(self):
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("東京大学",), relation="primary")}
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("東京大学の入学要件",)),
            (candidate(content="東京大学という名称だけを紹介する。"),),
        )

        self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIn("東京大学の入学要件", bundle.missing_claim_topics)

    def test_meaningful_topic_symbols_do_not_create_c_cpp_csharp_aliases(self):
        unequal_labels = (
            ("C", "C++"),
            ("C++", "C"),
            ("C", "C#"),
            ("C#", "C"),
            ("C++", "C#"),
            ("C#", "C++"),
        )

        for required, supported in unequal_labels:
            with self.subTest(required=required, supported=supported):
                judge = StaticEvidenceJudge(
                    {"C1": judge_ok("C1", supported=(supported,), relation="primary")}
                )
                bundle = self.module.EvidenceAssembler(judge).assemble(
                    plan(required_topics=(required,)),
                    (candidate(content=f"正文只支持 {supported} 标签。"),),
                )
                self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)
                self.assertIn(required, bundle.missing_claim_topics)

    def test_nfc_and_nfd_topic_labels_are_equivalent_in_both_directions(self):
        nfc = "がん治療ガイド"
        nfd = "か\u3099ん治療カ\u3099イト\u3099"

        for required, supported in ((nfc, nfd), (nfd, nfc)):
            with self.subTest(required=required, supported=supported):
                judge = StaticEvidenceJudge(
                    {"C1": judge_ok("C1", supported=(supported,), relation="primary")}
                )
                bundle = self.module.EvidenceAssembler(judge).assemble(
                    plan(required_topics=(required,)),
                    (candidate(content="がん治療ガイドについて説明する本文。"),),
                )
                self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
                self.assertEqual(bundle.missing_claim_topics, ())

    def test_topic_identity_normalizes_case_and_whitespace(self):
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("rust   言語",), relation="primary")}
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("  RUST\t言語  ",)),
            (candidate(content="Rust 言語について説明する本文。"),),
        )

        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual(bundle.missing_claim_topics, ())


class EvidenceGapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    def test_zero_citable_evidence_is_unconditionally_insufficient(self):
        judge = StaticEvidenceJudge(
            {"C1": {
                "candidate_id": "C1",
                "relevance": "direct",
                "source_relation": "independent",
                "publisher_entity_match": False,
                "ownership_basis": None,
                "publisher": None,
                "supported_topics": ["定义"],
                "conflict_key": None,
                "conflict_value": None,
                "conflict_relation": None,
            }}
        )
        assembler = self.module.EvidenceAssembler(judge)
        p = plan(required_topics=("定义",), route=SearchTier.DEEP)
        # A candidate whose excerpt is empty cannot be citable.
        from src.search.models import EvidenceCandidate, ExcerptOrigin
        empty = EvidenceCandidate(
            candidate(hit=None) if False else __import__("src.search.models", fromlist=["ProviderHit"]).ProviderHit(
                "tavily", "q1", "t", "https://example.com/x", None, None, None, None, (),
            ),
            None, None, None, "no_content", (), 0,
        )
        bundle = assembler.assemble(p, (empty,))
        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)
        self.assertTrue(bundle.limitations)

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
