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
        second = planner.plan_repair(plan, gap(missing=("x",)))
        self.assertFalse(second.triggered)
        self.assertIsNone(second.repair_query)

    def test_standard_empty_gap_cannot_repair(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = planner.plan(request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        repair = planner.plan_repair(plan, gap(missing=(), conflict=(), eligible=False))
        self.assertFalse(repair.triggered)


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
                        "purpose": "direct",
                        "text": "什么是光合作用",
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
        self.assertEqual(plan.initial_queries[0].include_domains, ())

    def test_rejects_private_and_local_names(self):
        plan = self._plan_with_domains(["127.0.0.1", "localhost", "example.com"], [])
        self.assertNotIn("127.0.0.1", plan.initial_queries[0].include_domains)
        self.assertNotIn("localhost", plan.initial_queries[0].include_domains)
        self.assertIn("example.com", plan.initial_queries[0].include_domains)

    def test_deduplicates_and_caps_domain_list(self):
        plan = self._plan_with_domains(
            ["a.com", "a.com", "b.com", "c.com", "d.com", "e.com", "f.com"],
            [],
        )
        self.assertEqual(len(plan.initial_queries[0].include_domains), 5)


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
