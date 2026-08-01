"""Planner tests: bounded natural-language query planning and one repair."""

from __future__ import annotations

import importlib
import unittest
from datetime import date

from src.search.models import (
    EvidenceGapAnalysis,
    Factuality,
    Freshness,
    PlanningStatus,
    QueryPurpose,
    RequestSource,
    RetrievalDecision,
    RetrievalRequest,
    RiskLevel,
    SearchTier,
    SearchRoundKind,
    SearchTrace,
    SkipReason,
    TriggerCode,
)
from tests.search_fakes import StaticPlannerModel


def planner_module():
    try:
        return importlib.import_module("src.search.planner")
    except ModuleNotFoundError:
        raise AssertionError("src.search.planner must exist") from None


def decision(tier: SearchTier) -> RetrievalDecision:
    m = __import__("src.search.models", fromlist=["RetrievalDecision"])
    return m.RetrievalDecision(
        tier, None, False, (), frozenset(), Factuality.FACTUAL,
        True, Freshness.NONE, RiskLevel.LOW, m.Actionability.NONE,
        m.PotentialHarm.NONE, tier, None, (),
    )


def request(question: str) -> RetrievalRequest:
    return RetrievalRequest(
        question,
        force_search=False,
        request_source=RequestSource.CHAT,
    )


def light_decision():
    return decision(SearchTier.LIGHT)


def standard_decision():
    return decision(SearchTier.STANDARD)


def deep_decision():
    return decision(SearchTier.DEEP)


def gap(missing=(), conflict=(), eligible=True):
    return EvidenceGapAnalysis(
        tuple(missing),
        tuple(conflict),
        eligible,
        "fill the gap" if eligible else None,
        ("missing_topic",) if eligible else (),
    )


class PlannerLightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def test_light_uses_exactly_one_direct_query_equal_to_question(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("什么是光合作用"), light_decision())
        self.assertEqual(plan.planning_status, PlanningStatus.NORMAL)
        self.assertEqual(len(plan.initial_queries), 1)
        query = plan.initial_queries[0]
        self.assertIs(query.purpose, QueryPurpose.DIRECT)
        self.assertEqual(query.text, "什么是光合作用")
        self.assertEqual(plan.original_question, "什么是光合作用")
        self.assertEqual(plan.budget.max_initial_queries, 1)
        self.assertEqual(plan.budget.max_total_queries, 1)

    def test_light_never_calls_model(self):
        model = StaticPlannerModel()
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        planner.plan(request("什么是光合作用"), light_decision())
        self.assertEqual(model.calls, [])

    def test_light_cannot_repair(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("什么是光合作用"), light_decision())
        repair = planner.plan_repair(plan, gap(missing=("x",)))
        self.assertFalse(repair.triggered)
        self.assertIsNone(repair.repair_query)


class PlannerStandardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def _plan(self, model=None):
        model = model if model is not None else StaticPlannerModel()
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        return planner.plan(request("Rust 和 Go 的并发模型有什么区别"), standard_decision())

    def test_standard_emits_at_most_three_queries_with_original_direct(self):
        plan = self._plan()
        self.assertLessEqual(len(plan.initial_queries), 3)
        purposes = {q.purpose for q in plan.initial_queries}
        self.assertIn(QueryPurpose.DIRECT, purposes)
        direct = next(q for q in plan.initial_queries if q.purpose is QueryPurpose.DIRECT)
        self.assertEqual(direct.text, "Rust 和 Go 的并发模型有什么区别")

    def test_standard_fallback_adds_primary_and_independent(self):
        plan = self._plan()
        texts = [q.text for q in plan.initial_queries]
        self.assertIn(
            "Rust 和 Go 的并发模型有什么区别 Rust 官方文档 Go 官方文档",
            texts,
        )
        self.assertIn(
            "Rust 和 Go 的并发模型有什么区别 独立技术对比",
            texts,
        )

    def test_standard_repair_emits_at_most_one_distinct_query(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        repair = planner.plan_repair(plan, gap(missing=("内存安全",)))
        self.assertTrue(repair.triggered)
        self.assertIsNotNone(repair.repair_query)
        self.assertIs(repair.repair_query.round_kind, SearchRoundKind.REPAIR)
        self.assertIs(repair.repair_query.purpose, QueryPurpose.REPAIR)
        self.assertEqual(repair.repair_query.text, "内存安全 Rust 和 Go 并发模型")
        self.assertEqual(plan.budget.max_total_queries, 4)

    def test_standard_rejects_duplicate_repair(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        first = planner.plan_repair(plan, gap(missing=("x",)))
        self.assertTrue(first.triggered)
        # A second repair for the same request is refused via the
        # repair_already_planned flag (no cross-request instance state).
        second = planner.plan_repair(plan, gap(missing=("x",)), repair_already_planned=True)
        self.assertFalse(second.triggered)
        self.assertIsNone(second.repair_query)

    def test_standard_empty_gap_cannot_repair(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        repair = planner.plan_repair(plan, gap(missing=(), conflict=(), eligible=False))
        self.assertFalse(repair.triggered)

    def test_repair_state_does_not_leak_across_requests(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan_a = planner.plan(request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        first = planner.plan_repair(plan_a, gap(missing=("内存",)))
        self.assertTrue(first.triggered)
        # An unrelated request must be able to plan its own repair.
        plan_b = planner.plan(request("什么是光合作用"), standard_decision())
        second = planner.plan_repair(plan_b, gap(missing=("机制",)))
        self.assertTrue(second.triggered)


class PlannerDeepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def _plan(self):
        model = StaticPlannerModel()
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        return planner.plan(request("北京今天有什么新闻"), deep_decision())

    def test_deep_emits_at_most_five_with_time_bounded(self):
        plan = self._plan()
        self.assertLessEqual(len(plan.initial_queries), 5)
        purposes = {q.purpose for q in plan.initial_queries}
        self.assertIn(QueryPurpose.DIRECT, purposes)
        self.assertIn(QueryPurpose.TIME_BOUNDED, purposes)

    def test_deep_direct_preserves_original_cjk(self):
        plan = self._plan()
        direct = next(q for q in plan.initial_queries if q.purpose is QueryPurpose.DIRECT)
        self.assertEqual(direct.text, "北京今天有什么新闻")

    def test_deep_fallback_preserves_required_queries(self):
        plan = self._plan()
        texts = [q.text for q in plan.initial_queries]
        self.assertIn("北京 2026-07-29 新闻 重要事件", texts)
        self.assertIn("北京 2026-07-29 官方 通报", texts)
        self.assertIn("北京 2026-07-29 新闻 重要事件 独立报道", texts)

    def test_deep_repair_allows_one(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("北京今天有什么新闻"), deep_decision())
        repair = planner.plan_repair(plan, gap(missing=("事件细节",)))
        self.assertTrue(repair.triggered)
        self.assertEqual(plan.budget.max_total_queries, 6)
        self.assertEqual(plan.budget.max_repair_queries, 1)


class PlannerRedactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def _plan(self, question, tier=SearchTier.STANDARD):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        return planner.plan(request(question), decision(tier))

    def test_removes_cq_control_codes(self):
        plan = self._plan("CQ:image,file=x.png 什么是光合作用")
        joined = " ".join(q.text for q in plan.initial_queries)
        self.assertNotIn("CQ:", joined)
        self.assertNotIn("file=x.png", joined)
        self.assertIn("什么是光合作用", joined)

    def test_removes_data_urls(self):
        plan = self._plan("data:image/png;base64,AAAA 什么是光合作用")
        joined = " ".join(q.text for q in plan.initial_queries)
        self.assertNotIn("data:image", joined)
        self.assertNotIn("base64", joined)

    def test_cq_and_data_url_redactions_have_explicit_audit_codes(self):
        plan = self._plan("[CQ:image,file=x.png] data:image/png;base64,AAAA 什么是光合作用")
        joined = " ".join(query.text for query in plan.initial_queries)
        self.assertNotIn("CQ:", joined)
        self.assertNotIn("data:image", joined)
        self.assertIn("cq_control_code", plan.query_redaction_codes)
        self.assertIn("data_url", plan.query_redaction_codes)

    def test_removes_api_keys(self):
        plan = self._plan("API密钥 AIzaSyExampleKey123 是什么")
        joined = " ".join(q.text for q in plan.initial_queries)
        self.assertNotIn("AIza", joined)

    def test_removes_one_time_codes(self):
        plan = self._plan("验证码 654321 是什么")
        joined = " ".join(q.text for q in plan.initial_queries)
        self.assertNotIn("654321", joined)

    def test_explicit_search_of_exact_phone_keeps_value(self):
        plan = self._plan("请搜索并核实这个号码 13800138000")
        joined = " ".join(q.text for q in plan.initial_queries)
        self.assertIn("13800138000", joined)

    def test_removes_callback_secret(self):
        plan = self._plan("回调签名 abcdef123456 是什么")
        joined = " ".join(q.text for q in plan.initial_queries)
        self.assertNotIn("abcdef123456", joined)

    def test_degraded_when_redaction_removes_everything(self):
        plan = self._plan("secret: sk-1234567890abcdef", tier=SearchTier.LIGHT)
        self.assertEqual(plan.planning_status, PlanningStatus.DEGRADED)
        joined = " ".join(q.text for q in plan.initial_queries)
        self.assertNotIn("sk-1234567890abcdef", joined)
        self.assertTrue(joined)

    def test_memory_text_in_context_never_reaches_plan(self):
        fake_context = "用户记忆：用户喜欢蓝色，偏好安静的环境"
        question = f"{fake_context} 什么是光合作用"
        plan = self._plan(question, tier=SearchTier.LIGHT)
        joined = " ".join(q.text for q in plan.initial_queries)
        self.assertNotIn("蓝色", joined)
        self.assertNotIn("安静", joined)
        self.assertIn("什么是光合作用", joined)

    def test_light_direct_phone_removed_without_explicit_search(self):
        plan = self._plan("这个号码13800138000是谁", tier=SearchTier.LIGHT)
        joined = " ".join(q.text for q in plan.initial_queries)
        self.assertNotIn("13800138000", joined)
        self.assertIn("phone_number", plan.query_redaction_codes)

    def test_deterministic_fallback_phone_removed(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(
            request("这个号码 13800138000 是谁"),
            decision(SearchTier.STANDARD),
        )
        joined = " ".join(q.text for q in plan.initial_queries)
        self.assertNotIn("13800138000", joined)

    def test_repair_query_phone_removed(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("什么是光合作用"), standard_decision())
        repair = planner.plan_repair(plan, gap(missing=("13800138000 是什么号码",)))
        if repair.triggered:
            self.assertNotIn("13800138000", repair.repair_query.text)

    def test_repair_redaction_codes_are_request_scoped_and_auditable(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("什么是光合作用"), standard_decision())
        repair = planner.plan_repair(plan, gap(missing=("回调签名 abcdef123456 如何处理",)))
        self.assertTrue(repair.triggered)
        self.assertNotIn("abcdef123456", repair.repair_query.text)
        self.assertIn("callback_secret", repair.query_redaction_codes)

        unrelated_plan = planner.plan(request("什么是光合作用"), standard_decision())
        unrelated_repair = planner.plan_repair(unrelated_plan, gap(missing=("证据缺口",)))
        self.assertNotIn("callback_secret", unrelated_repair.query_redaction_codes)


class PlannerDomainValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def _plan_with_domains(self, include, exclude):
        model = StaticPlannerModel(
            {
                "planning_status": "normal",
                "entities": [],
                "initial_queries": [
                    {
                        "purpose": "primary",
                        "text": "光合作用 官方 介绍",
                        "include_domains": include,
                        "exclude_domains": exclude,
                    }
                ],
            }
        )
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        return planner.plan(request("什么是光合作用"), standard_decision())

    def test_rejects_urls_in_domain_lists(self):
        plan = self._plan_with_domains(["https://example.com"], [])
        self.assertEqual(plan.initial_queries[1].include_domains, ())

    def test_rejects_private_and_local_names(self):
        plan = self._plan_with_domains(["127.0.0.1", "localhost", "example.com"], [])
        self.assertNotIn("127.0.0.1", plan.initial_queries[1].include_domains)
        self.assertNotIn("localhost", plan.initial_queries[1].include_domains)
        self.assertIn("example.com", plan.initial_queries[1].include_domains)

    def test_deduplicates_and_caps_domain_list(self):
        plan = self._plan_with_domains(
            ["a.com", "a.com", "b.com", "c.com", "d.com", "e.com", "f.com"],
            [],
        )
        self.assertEqual(len(plan.initial_queries[1].include_domains), 5)


class PlannerDegradationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def test_invalid_model_output_is_degraded_without_lowering_route(self):
        model = StaticPlannerModel(payload={"bogus": True})
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        self.assertEqual(plan.planning_status, PlanningStatus.DEGRADED)
        self.assertEqual(len(plan.initial_queries), 3)
        purposes = {q.purpose for q in plan.initial_queries}
        self.assertIn(QueryPurpose.DIRECT, purposes)
        self.assertIn(QueryPurpose.PRIMARY, purposes)
        self.assertIn(QueryPurpose.INDEPENDENT, purposes)

    def test_model_exception_is_degraded(self):
        model = StaticPlannerModel(raise_error=RuntimeError("boom"))
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        self.assertEqual(plan.planning_status, PlanningStatus.DEGRADED)
        self.assertGreaterEqual(len(plan.initial_queries), 1)
        purposes = {q.purpose for q in plan.initial_queries}
        self.assertIn(QueryPurpose.DIRECT, purposes)

    def test_unknown_query_purpose_is_rejected(self):
        model = StaticPlannerModel(
            {
                "planning_status": "normal",
                "entities": [],
                "initial_queries": [{"purpose": "made_up", "text": "什么是光合作用"}],
            }
        )
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("什么是光合作用"), standard_decision())
        self.assertEqual(plan.planning_status, PlanningStatus.DEGRADED)

    def test_model_cannot_remove_cjk_direct_query(self):
        model = StaticPlannerModel(
            {
                "planning_status": "normal",
                "entities": [],
                "initial_queries": [{"purpose": "primary", "text": "机械关键词"}],
            }
        )
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("北京今天有什么新闻"), deep_decision())
        texts = [q.text for q in plan.initial_queries]
        self.assertIn("北京今天有什么新闻", texts)
        self.assertIn(QueryPurpose.DIRECT, {q.purpose for q in plan.initial_queries})

    def test_deep_dynamic_plan_always_has_time_bounded_query(self):
        from src.search.models import Freshness as F
        model = StaticPlannerModel(
            {
                "planning_status": "normal",
                "entities": [],
                "initial_queries": [{"purpose": "direct", "text": "北京今天有什么新闻"}],
            }
        )
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        d = decision(SearchTier.DEEP)
        d = __import__("src.search.models", fromlist=["RetrievalDecision"]).RetrievalDecision(
            SearchTier.DEEP, None, False, (), frozenset(), Factuality.FACTUAL,
            True, F.HIGH, RiskLevel.LOW, __import__("src.search.models", fromlist=["Actionability"]).Actionability.NONE,
            __import__("src.search.models", fromlist=["PotentialHarm"]).PotentialHarm.NONE,
            SearchTier.DEEP, None, (),
        )
        plan = planner.plan(request("北京今天有什么新闻"), d)
        self.assertIn(QueryPurpose.TIME_BOUNDED, {q.purpose for q in plan.initial_queries})

    def test_deep_high_freshness_replaces_a_full_model_slot_with_time_bounded_query(self):
        model = StaticPlannerModel(
            {
                "planning_status": "normal",
                "entities": [],
                "initial_queries": [
                    {"purpose": "primary", "text": f"北京新闻来源 {index}"}
                    for index in range(5)
                ],
            }
        )
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        d = __import__("src.search.models", fromlist=["RetrievalDecision"]).RetrievalDecision(
            SearchTier.DEEP, None, False, (), frozenset(), Factuality.FACTUAL,
            True, Freshness.HIGH, RiskLevel.LOW,
            __import__("src.search.models", fromlist=["Actionability"]).Actionability.NONE,
            __import__("src.search.models", fromlist=["PotentialHarm"]).PotentialHarm.NONE,
            SearchTier.DEEP, None, (),
        )
        plan = planner.plan(request("北京今天有什么新闻"), d)
        self.assertEqual(len(plan.initial_queries), plan.budget.max_initial_queries)
        self.assertEqual(plan.initial_queries[0].text, "北京今天有什么新闻")
        time_bounded = [query for query in plan.initial_queries if query.purpose is QueryPurpose.TIME_BOUNDED]
        self.assertEqual(len(time_bounded), 1)
        self.assertEqual(time_bounded[0].date_from, date(2026, 7, 29))
        self.assertEqual(time_bounded[0].date_to, date(2026, 7, 29))

    def test_deep_high_freshness_validates_final_slots_for_genuine_time_bounds(self):
        def model_queries(time_query):
            return [
                {"purpose": "primary", "text": f"北京新闻来源 {index}"}
                for index in range(4)
            ] + [time_query]

        cases = {
            "fifth_slot": {
                "purpose": "time_bounded",
                "text": "北京 2026-07-29 新闻",
                "date_from": "2026-07-29",
                "date_to": "2026-07-29",
            },
            "unbounded": {
                "purpose": "time_bounded",
                "text": "北京新闻",
            },
            "partial": {
                "purpose": "time_bounded",
                "text": "北京 2026-07-29 新闻",
                "date_from": "2026-07-29",
            },
            "reversed": {
                "purpose": "time_bounded",
                "text": "北京 2026-07-28 至 2026-07-29 新闻",
                "date_from": "2026-07-29",
                "date_to": "2026-07-28",
            },
        }
        d = __import__("src.search.models", fromlist=["RetrievalDecision"]).RetrievalDecision(
            SearchTier.DEEP, None, False, (), frozenset(), Factuality.FACTUAL,
            True, Freshness.HIGH, RiskLevel.LOW,
            __import__("src.search.models", fromlist=["Actionability"]).Actionability.NONE,
            __import__("src.search.models", fromlist=["PotentialHarm"]).PotentialHarm.NONE,
            SearchTier.DEEP, None, (),
        )
        for name, time_query in cases.items():
            with self.subTest(case=name):
                model = StaticPlannerModel(
                    {"planning_status": "normal", "entities": [], "initial_queries": model_queries(time_query)}
                )
                planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
                plan = planner.plan(request("北京今天有什么新闻"), d)
                self.assertEqual(plan.initial_queries[0].text, "北京今天有什么新闻")
                genuine = [
                    query
                    for query in plan.initial_queries
                    if query.purpose is QueryPurpose.TIME_BOUNDED
                    and query.date_from is not None
                    and query.date_to is not None
                    and query.date_from <= query.date_to
                    and query.date_from.isoformat() in query.text
                    and query.date_to.isoformat() in query.text
                ]
                self.assertEqual(len(genuine), 1)

    def test_deep_plan_assigns_unique_final_ids_after_dedupe_and_time_replacement(self):
        model = StaticPlannerModel(
            {
                "planning_status": "normal",
                "entities": [],
                "initial_queries": [
                    {"purpose": "direct", "text": "北京今天有什么新闻"},
                    {"purpose": "primary", "text": "北京官方来源"},
                ],
            }
        )
        d = __import__("src.search.models", fromlist=["RetrievalDecision"]).RetrievalDecision(
            SearchTier.DEEP, None, False, (), frozenset(), Factuality.FACTUAL,
            True, Freshness.HIGH, RiskLevel.LOW,
            __import__("src.search.models", fromlist=["Actionability"]).Actionability.NONE,
            __import__("src.search.models", fromlist=["PotentialHarm"]).PotentialHarm.NONE,
            SearchTier.DEEP, None, (),
        )
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("北京今天有什么新闻"), d)
        self.assertEqual("北京今天有什么新闻", plan.initial_queries[0].text)
        self.assertEqual(
            [f"initial-{index}" for index in range(1, len(plan.initial_queries) + 1)],
            [query.query_id for query in plan.initial_queries],
        )
        time_bounded = next(query for query in plan.initial_queries if query.purpose is QueryPurpose.TIME_BOUNDED)
        from src.search.models import ProviderHit
        hit = ProviderHit(
            "tavily", time_bounded.query_id, "title", "https://example.com/time",
            "snippet", None, None, None, (),
        )
        orchestrator = importlib.import_module("src.search.orchestrator")
        self.assertEqual(time_bounded, orchestrator._query_for_hit(plan, hit))
        trace = SearchTrace(
            "req-ids", RequestSource.CHAT, SearchTier.DEEP,
            executed_queries=tuple(plan.initial_queries),
        )
        self.assertEqual(len(plan.initial_queries), trace.to_log_dict()["semantic_query_count"])


class PlannerQueryCapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def test_total_semantic_queries_never_exceed_budget(self):
        model = StaticPlannerModel(
            {
                "planning_status": "normal",
                "entities": [],
                "initial_queries": [
                    {"purpose": "direct", "text": f"q{i}"} for i in range(20)
                ],
            }
        )
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("什么是光合作用"), deep_decision())
        self.assertLessEqual(len(plan.initial_queries), 5)
        self.assertLessEqual(
            len(plan.initial_queries) + (1 if plan.decision.route in (SearchTier.STANDARD, SearchTier.DEEP) else 0),
            plan.budget.max_total_queries,
        )

    def test_query_text_capped_at_500_chars(self):
        long_text = "长" * 2000
        model = StaticPlannerModel(
            {
                "planning_status": "normal",
                "entities": [],
                "initial_queries": [{"purpose": "direct", "text": long_text}],
            }
        )
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("什么是光合作用"), standard_decision())
        for query in plan.initial_queries:
            self.assertLessEqual(len(query.text), 500)


if __name__ == "__main__":
    unittest.main()
