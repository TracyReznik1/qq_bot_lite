"""Planner tests: bounded natural-language query planning and one repair."""

from __future__ import annotations

import importlib
import unittest
from datetime import date

from src.search.models import (
    EvidenceGapAnalysis,
    Factuality,
    Freshness,
    FreshnessContext,
    FreshnessRequirement,
    PlanningStatus,
    QueryPurpose,
    RequestSource,
    RetrievalDecision,
    RetrievalContext,
    RetrievalRequest,
    RiskLevel,
    SearchTier,
    SearchRoundKind,
    SearchTrace,
    SkipReason,
    SourceRequirement,
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


def retrieval_context(
    source_requirement: SourceRequirement = SourceRequirement.ANY_RELEVANT,
) -> RetrievalContext:
    return RetrievalContext(
        must_search=True,
        skip_reason=None,
        factuality=Factuality.FACTUAL,
        external_fact_required=True,
        complexity_codes=(),
        source_requirement=source_requirement,
    )


def freshness_context(
    requirement: FreshnessRequirement = FreshnessRequirement.NOT_REQUIRED,
    *,
    date_from: date | None = None,
    date_to: date | None = None,
    version_constraint: str | None = None,
) -> FreshnessContext:
    return FreshnessContext(
        requirement=requirement,
        as_of=None,
        date_from=date_from,
        date_to=date_to,
        version_constraint=version_constraint,
    )


def plan_with_context(
    planner,
    retrieval_request,
    retrieval_decision,
    *,
    source_requirement: SourceRequirement = SourceRequirement.ANY_RELEVANT,
    freshness: FreshnessContext | None = None,
    **kwargs,
):
    return planner.plan(
        retrieval_request,
        retrieval_decision,
        retrieval_context(source_requirement),
        freshness_context() if freshness is None else freshness,
        **kwargs,
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


def strict_payload(*, supplements=(), topics=()):
    return {
        "supplemental_queries": list(supplements),
        "required_topics": list(topics),
    }


def model_topic(
    label,
    *,
    material=True,
    freshness_requirement="not_required",
    version_constraint=None,
    source_requirement="any_relevant",
):
    return {
        "label": label,
        "material": material,
        "freshness_requirement": freshness_requirement,
        "date_from": None,
        "date_to": None,
        "version_constraint": version_constraint,
        "source_requirement": source_requirement,
    }


def supplemental_query(
    purpose,
    text,
    *,
    targets=("topic-1",),
    date_from=None,
    date_to=None,
    include_domains=(),
    exclude_domains=(),
):
    return {
        "purpose": purpose,
        "text": text,
        "target_topic_ids": list(targets),
        "date_from": date_from,
        "date_to": date_to,
        "include_domains": list(include_domains),
        "exclude_domains": list(exclude_domains),
    }


class PlannerDirectQueryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def _plan(self, model, question="比较 Rust 和 Go 的并发模型"):
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        return planner.plan(
            request(question),
            standard_decision(),
            retrieval_context(),
            freshness_context(),
        )

    def test_three_model_supplements_leave_only_two_non_direct_slots(self):
        model = StaticPlannerModel(strict_payload(
            topics=(
                model_topic("Rust 并发 API"),
                model_topic("Go 并发 API"),
                model_topic("并发模型对比"),
            ),
            supplements=(
                {"purpose": "primary", "text": "Rust 官方并发模型", "target_topic_ids": ["topic-1"], "date_from": None, "date_to": None},
                {"purpose": "independent", "text": "Go 并发模型独立比较", "target_topic_ids": ["topic-1"], "date_from": None, "date_to": None},
                {"purpose": "primary", "text": "Rust Go 并发 API 对比", "target_topic_ids": ["topic-1"], "date_from": None, "date_to": None},
            ),
        ))

        plan = self._plan(model)

        self.assertLessEqual(len(plan.initial_queries), 3)
        self.assertIs(plan.initial_queries[0].purpose, QueryPurpose.DIRECT)
        self.assertLessEqual(
            sum(query.purpose is not QueryPurpose.DIRECT for query in plan.initial_queries),
            2,
        )
        self.assertEqual(
            ("topic-1", "topic-2", "topic-3"),
            plan.initial_queries[0].target_topic_ids,
        )
        self.assertEqual(
            ["比较 Rust 和 Go 的并发模型", "Rust 官方并发模型", "Go 并发模型独立比较"],
            [query.text for query in plan.initial_queries],
        )

    def test_valid_model_with_no_supplements_uses_only_the_direct_query(self):
        plan = self._plan(StaticPlannerModel(strict_payload(
            topics=(model_topic("并发模型"),),
            supplements=(),
        )))

        self.assertEqual(1, len(plan.initial_queries))
        self.assertIs(plan.initial_queries[0].purpose, QueryPurpose.DIRECT)

    def test_current_bounds_do_not_add_queries_and_bound_the_direct_query(self):
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic("Python 并发 API"),),
            supplements=(
                {"purpose": "primary", "text": "Python 并发 API 官方文档", "target_topic_ids": ["topic-1"], "date_from": None, "date_to": None},
            ),
        ))
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        stable = planner.plan(
            request("比较 Python 并发 API"), standard_decision(),
            retrieval_context(), freshness_context(),
        )
        current = planner.plan(
            request("比较 Python 并发 API"), standard_decision(),
            retrieval_context(), freshness_context(FreshnessRequirement.CURRENT),
        )

        self.assertEqual(len(stable.initial_queries), len(current.initial_queries))
        self.assertEqual(date(2026, 7, 29), current.initial_queries[0].date_from)
        self.assertEqual(date(2026, 7, 29), current.initial_queries[0].date_to)

    def test_version_context_overrides_model_topics_and_drops_unversioned_supplements(self):
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic(
                "Python 3.13 并发 API",
                freshness_requirement="not_required",
                source_requirement="independent_corroboration",
            ),),
            supplements=(
                {"purpose": "primary", "text": "Python 官方并发 API", "target_topic_ids": ["topic-1"], "date_from": None, "date_to": None},
                {"purpose": "independent", "text": "Python 3.13 并发 API 比较", "target_topic_ids": ["topic-1"], "date_from": None, "date_to": None},
            ),
        ))
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))

        plan = planner.plan(
            request("比较 Python 3.13 的两个并发 API"), standard_decision(),
            retrieval_context(),
            freshness_context(FreshnessRequirement.VERSION, version_constraint="3.13"),
        )

        self.assertTrue(all(item.material for item in plan.required_topics))
        self.assertTrue(all(
            item.freshness_requirement is FreshnessRequirement.VERSION
            and item.version_constraint == "3.13"
            and item.source_requirement is SourceRequirement.ANY_RELEVANT
            for item in plan.required_topics
        ))
        self.assertEqual(
            ["比较 Python 3.13 的两个并发 API", "Python 3.13 并发 API 比较"],
            [query.text for query in plan.initial_queries],
        )

    def test_unknown_or_nonmaterial_targets_are_dropped_after_topic_sealing(self):
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic("主主题"), model_topic("背景", material=False)),
            supplements=(
                {"purpose": "primary", "text": "未知目标", "target_topic_ids": ["topic-9"], "date_from": None, "date_to": None},
                {"purpose": "primary", "text": "空目标", "target_topic_ids": [], "date_from": None, "date_to": None},
                {"purpose": "primary", "text": "非材料目标", "target_topic_ids": ["topic-2"], "date_from": None, "date_to": None},
                {"purpose": "primary", "text": "材料目标", "target_topic_ids": ["topic-1"], "date_from": None, "date_to": None},
            ),
        ))

        plan = self._plan(model)

        self.assertEqual(["比较 Rust 和 Go 的并发模型", "材料目标"], [query.text for query in plan.initial_queries])
        self.assertEqual(("topic-1",), plan.initial_queries[1].target_topic_ids)

    def test_all_nonmaterial_model_topics_receive_a_deterministic_material_fallback(self):
        model = StaticPlannerModel(strict_payload(
            topics=(
                model_topic("背景", material=False),
                model_topic("附注", material=False),
            ),
            supplements=(
                supplemental_query("primary", "背景资料", targets=("topic-1",)),
            ),
        ))

        first = self._plan(model)
        second = self._plan(model)
        material_ids = tuple(
            topic.topic_id for topic in first.required_topics if topic.material
        )

        self.assertEqual(tuple(topic.label for topic in first.required_topics), tuple(
            topic.label for topic in second.required_topics
        ))
        self.assertTrue(material_ids)
        self.assertEqual(material_ids, first.initial_queries[0].target_topic_ids)
        self.assertEqual(1, len(first.initial_queries))


class PlannerLightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def test_light_uses_exactly_one_direct_query_equal_to_question(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("什么是光合作用"), light_decision())
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
        plan_with_context(planner, request("什么是光合作用"), light_decision())
        self.assertEqual(model.calls, [])

    def test_light_cannot_repair(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("什么是光合作用"), light_decision())
        repair = planner.plan_repair(plan, gap(missing=("x",)))
        self.assertFalse(repair.triggered)
        self.assertIsNone(repair.repair_query)


class PlannerStandardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def _plan(self, model=None):
        model = model if model is not None else StaticPlannerModel()
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        return plan_with_context(planner, request("Rust 和 Go 的并发模型有什么区别"), standard_decision())

    def test_standard_emits_at_most_three_queries_with_original_direct(self):
        plan = self._plan()
        self.assertLessEqual(len(plan.initial_queries), 3)
        purposes = {q.purpose for q in plan.initial_queries}
        self.assertIn(QueryPurpose.DIRECT, purposes)
        direct = next(q for q in plan.initial_queries if q.purpose is QueryPurpose.DIRECT)
        self.assertEqual(direct.text, "Rust 和 Go 的并发模型有什么区别")

    def test_planner_passes_the_smaller_remaining_time_to_model(self):
        model = StaticPlannerModel()
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))

        plan_with_context(
            planner,
            request("Rust 和 Go 的并发模型有什么区别"),
            standard_decision(),
            timeout_seconds=0.25,
        )

        timeout = model.calls[0][1]["timeout_seconds"]
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, 0.25)

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
        plan = plan_with_context(planner, request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        repair = planner.plan_repair(plan, gap(missing=("内存安全",)))
        self.assertTrue(repair.triggered)
        self.assertIsNotNone(repair.repair_query)
        self.assertIs(repair.repair_query.round_kind, SearchRoundKind.REPAIR)
        self.assertIs(repair.repair_query.purpose, QueryPurpose.REPAIR)
        self.assertEqual(repair.repair_query.text, "内存安全 Rust 和 Go 并发模型")
        self.assertEqual(plan.budget.max_total_queries, 4)

    def test_standard_rejects_duplicate_repair(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        first = planner.plan_repair(plan, gap(missing=("x",)))
        self.assertTrue(first.triggered)
        # A second repair for the same request is refused via the
        # repair_already_planned flag (no cross-request instance state).
        second = planner.plan_repair(plan, gap(missing=("x",)), repair_already_planned=True)
        self.assertFalse(second.triggered)
        self.assertIsNone(second.repair_query)

    def test_standard_empty_gap_cannot_repair(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        repair = planner.plan_repair(plan, gap(missing=(), conflict=(), eligible=False))
        self.assertFalse(repair.triggered)

    def test_repair_state_does_not_leak_across_requests(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan_a = plan_with_context(planner, request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        first = planner.plan_repair(plan_a, gap(missing=("内存",)))
        self.assertTrue(first.triggered)
        # An unrelated request must be able to plan its own repair.
        plan_b = plan_with_context(planner, request("什么是光合作用"), standard_decision())
        second = planner.plan_repair(plan_b, gap(missing=("机制",)))
        self.assertTrue(second.triggered)


class PlannerDeepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def _plan(self):
        model = StaticPlannerModel()
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        return plan_with_context(planner, request("北京今天有什么新闻"), deep_decision())

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
        plan = plan_with_context(planner, request("北京今天有什么新闻"), deep_decision())
        repair = planner.plan_repair(plan, gap(missing=("事件细节",)))
        self.assertTrue(repair.triggered)
        self.assertEqual(plan.budget.max_total_queries, 6)
        self.assertEqual(plan.budget.max_repair_queries, 1)


class PlannerRedactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def _plan(self, question, tier=SearchTier.STANDARD):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        return plan_with_context(planner, request(question), decision(tier))

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
        plan = plan_with_context(
            planner,
            request("这个号码 13800138000 是谁"),
            decision(SearchTier.STANDARD),
        )
        joined = " ".join(q.text for q in plan.initial_queries)
        self.assertNotIn("13800138000", joined)

    def test_repair_query_phone_removed(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("什么是光合作用"), standard_decision())
        repair = planner.plan_repair(plan, gap(missing=("13800138000 是什么号码",)))
        if repair.triggered:
            self.assertNotIn("13800138000", repair.repair_query.text)

    def test_repair_redaction_codes_are_request_scoped_and_auditable(self):
        planner = self.module.SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("什么是光合作用"), standard_decision())
        repair = planner.plan_repair(plan, gap(missing=("回调签名 abcdef123456 如何处理",)))
        self.assertTrue(repair.triggered)
        self.assertNotIn("abcdef123456", repair.repair_query.text)
        self.assertIn("callback_secret", repair.query_redaction_codes)

        unrelated_plan = plan_with_context(planner, request("什么是光合作用"), standard_decision())
        unrelated_repair = planner.plan_repair(unrelated_plan, gap(missing=("证据缺口",)))
        self.assertNotIn("callback_secret", unrelated_repair.query_redaction_codes)


class PlannerDomainValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def _plan_with_domains(self, include, exclude):
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic("光合作用"),),
            supplements=(
                supplemental_query(
                    "primary",
                    "光合作用 官方 介绍",
                    include_domains=include,
                    exclude_domains=exclude,
                ),
            ),
        ))
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        return plan_with_context(planner, request("什么是光合作用"), standard_decision())

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
        plan = plan_with_context(planner, request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        self.assertEqual(plan.planning_status, PlanningStatus.DEGRADED)
        self.assertEqual(len(plan.initial_queries), 3)
        purposes = {q.purpose for q in plan.initial_queries}
        self.assertIn(QueryPurpose.DIRECT, purposes)
        self.assertIn(QueryPurpose.PRIMARY, purposes)
        self.assertIn(QueryPurpose.INDEPENDENT, purposes)

    def test_model_exception_is_degraded(self):
        model = StaticPlannerModel(raise_error=RuntimeError("boom"))
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("Rust 和 Go 的并发模型有什么区别"), standard_decision())
        self.assertEqual(plan.planning_status, PlanningStatus.DEGRADED)
        self.assertGreaterEqual(len(plan.initial_queries), 1)
        purposes = {q.purpose for q in plan.initial_queries}
        self.assertIn(QueryPurpose.DIRECT, purposes)

    def test_unknown_query_purpose_is_rejected(self):
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic("光合作用"),),
            supplements=(supplemental_query("made_up", "什么是光合作用"),),
        ))
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("什么是光合作用"), standard_decision())
        self.assertEqual(plan.planning_status, PlanningStatus.DEGRADED)

    def test_model_cannot_remove_cjk_direct_query(self):
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic("北京新闻"),),
            supplements=(supplemental_query("primary", "机械关键词"),),
        ))
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("北京今天有什么新闻"), deep_decision())
        texts = [q.text for q in plan.initial_queries]
        self.assertIn("北京今天有什么新闻", texts)
        self.assertIn(QueryPurpose.DIRECT, {q.purpose for q in plan.initial_queries})

    def test_deep_dynamic_plan_always_has_time_bounded_query(self):
        from src.search.models import Freshness as F
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic("北京新闻"),),
            supplements=(
                supplemental_query(
                    "time_bounded",
                    "北京 2026-07-29 新闻",
                    date_from="2026-07-29",
                    date_to="2026-07-29",
                ),
            ),
        ))
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        d = decision(SearchTier.DEEP)
        d = __import__("src.search.models", fromlist=["RetrievalDecision"]).RetrievalDecision(
            SearchTier.DEEP, None, False, (), frozenset(), Factuality.FACTUAL,
            True, F.HIGH, RiskLevel.LOW, __import__("src.search.models", fromlist=["Actionability"]).Actionability.NONE,
            __import__("src.search.models", fromlist=["PotentialHarm"]).PotentialHarm.NONE,
            SearchTier.DEEP, None, (),
        )
        plan = plan_with_context(planner, request("北京今天有什么新闻"), d)
        self.assertIn(QueryPurpose.TIME_BOUNDED, {q.purpose for q in plan.initial_queries})

    def test_deep_high_freshness_replaces_a_full_model_slot_with_time_bounded_query(self):
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic("北京新闻"),),
            supplements=tuple(
                supplemental_query("primary", f"北京新闻来源 {index}")
                for index in range(3)
            ) + (
                supplemental_query(
                    "time_bounded",
                    "北京 2026-07-29 新闻",
                    date_from="2026-07-29",
                    date_to="2026-07-29",
                ),
                supplemental_query("primary", "北京新闻来源 额外槽位"),
            ),
        ))
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        d = __import__("src.search.models", fromlist=["RetrievalDecision"]).RetrievalDecision(
            SearchTier.DEEP, None, False, (), frozenset(), Factuality.FACTUAL,
            True, Freshness.HIGH, RiskLevel.LOW,
            __import__("src.search.models", fromlist=["Actionability"]).Actionability.NONE,
            __import__("src.search.models", fromlist=["PotentialHarm"]).PotentialHarm.NONE,
            SearchTier.DEEP, None, (),
        )
        plan = plan_with_context(planner, request("北京今天有什么新闻"), d)
        self.assertEqual(len(plan.initial_queries), plan.budget.max_initial_queries)
        self.assertEqual(plan.initial_queries[0].text, "北京今天有什么新闻")
        time_bounded = [query for query in plan.initial_queries if query.purpose is QueryPurpose.TIME_BOUNDED]
        self.assertEqual(len(time_bounded), 1)
        self.assertEqual(time_bounded[0].date_from, date(2026, 7, 29))
        self.assertEqual(time_bounded[0].date_to, date(2026, 7, 29))

    def test_deep_high_freshness_validates_final_slots_for_genuine_time_bounds(self):
        def model_queries(time_query):
            return [
                supplemental_query("primary", f"北京新闻来源 {index}")
                for index in range(3)
            ] + [time_query, supplemental_query("primary", "北京新闻来源 额外")]

        cases = {
            "fifth_slot": supplemental_query(
                "time_bounded", "北京 2026-07-29 新闻",
                date_from="2026-07-29", date_to="2026-07-29",
            ),
            "unbounded": supplemental_query("time_bounded", "北京新闻"),
            "partial": supplemental_query(
                "time_bounded", "北京 2026-07-29 新闻", date_from="2026-07-29",
            ),
            "reversed": supplemental_query(
                "time_bounded", "北京 2026-07-28 至 2026-07-29 新闻",
                date_from="2026-07-29", date_to="2026-07-28",
            ),
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
                model = StaticPlannerModel(strict_payload(
                    topics=(model_topic("北京新闻"),),
                    supplements=model_queries(time_query),
                ))
                planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
                plan = plan_with_context(planner, request("北京今天有什么新闻"), d)
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
                self.assertEqual(len(genuine), 1 if name == "fifth_slot" else 0)

    def test_deep_plan_assigns_unique_final_ids_after_dedupe_and_time_replacement(self):
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic("北京新闻"),),
            supplements=(
                supplemental_query("primary", "北京官方来源"),
                supplemental_query(
                    "time_bounded",
                    "北京 2026-07-29 新闻",
                    date_from="2026-07-29",
                    date_to="2026-07-29",
                ),
            ),
        ))
        d = __import__("src.search.models", fromlist=["RetrievalDecision"]).RetrievalDecision(
            SearchTier.DEEP, None, False, (), frozenset(), Factuality.FACTUAL,
            True, Freshness.HIGH, RiskLevel.LOW,
            __import__("src.search.models", fromlist=["Actionability"]).Actionability.NONE,
            __import__("src.search.models", fromlist=["PotentialHarm"]).PotentialHarm.NONE,
            SearchTier.DEEP, None, (),
        )
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("北京今天有什么新闻"), d)
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

    def test_deep_plan_removes_every_invalid_selected_time_query_before_dispatch(self):
        invalid_cases = {
            "unbounded": supplemental_query("time_bounded", "北京错误无界窗口"),
            "partial": supplemental_query(
                "time_bounded", "北京错误单边窗口 2026-07-29",
                date_from="2026-07-29",
            ),
            "reversed": supplemental_query(
                "time_bounded", "北京错误倒序窗口 2026-07-30 2026-07-29",
                date_from="2026-07-30", date_to="2026-07-29",
            ),
        }
        d = __import__("src.search.models", fromlist=["RetrievalDecision"]).RetrievalDecision(
            SearchTier.DEEP, None, False, (), frozenset(), Factuality.FACTUAL,
            True, Freshness.HIGH, RiskLevel.LOW,
            __import__("src.search.models", fromlist=["Actionability"]).Actionability.NONE,
            __import__("src.search.models", fromlist=["PotentialHarm"]).PotentialHarm.NONE,
            SearchTier.DEEP, None, (),
        )
        for name, invalid_query in invalid_cases.items():
            with self.subTest(case=name):
                model = StaticPlannerModel(strict_payload(
                    topics=(model_topic("北京新闻"),),
                    supplements=(
                        supplemental_query("primary", "北京官方来源"),
                        supplemental_query("primary", "北京官方来源"),
                        invalid_query,
                        supplemental_query("independent", "北京独立报道"),
                    ),
                ))
                planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
                plan = plan_with_context(planner, request("北京今天有什么新闻"), d)

                self.assertEqual("北京今天有什么新闻", plan.initial_queries[0].text)
                self.assertNotIn(invalid_query["text"], {query.text for query in plan.initial_queries})
                time_queries = [
                    query for query in plan.initial_queries
                    if query.purpose is QueryPurpose.TIME_BOUNDED
                ]
                self.assertEqual(0, len(time_queries))
                self.assertEqual(
                    [f"initial-{index}" for index in range(1, len(plan.initial_queries) + 1)],
                    [query.query_id for query in plan.initial_queries],
                )

    def test_final_plan_never_keeps_malformed_date_metadata_on_any_query_purpose(self):
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic("北京新闻"),),
            supplements=(
                supplemental_query(
                    "primary", "北京错误单边主查询", date_to="2026-07-29",
                ),
                supplemental_query(
                    "independent", "北京错误倒序独立查询",
                    date_from="2026-07-30", date_to="2026-07-29",
                ),
                supplemental_query("primary", "北京有效官方来源"),
            ),
        ))
        d = __import__("src.search.models", fromlist=["RetrievalDecision"]).RetrievalDecision(
            SearchTier.DEEP, None, False, (), frozenset(), Factuality.FACTUAL,
            True, Freshness.HIGH, RiskLevel.LOW,
            __import__("src.search.models", fromlist=["Actionability"]).Actionability.NONE,
            __import__("src.search.models", fromlist=["PotentialHarm"]).PotentialHarm.NONE,
            SearchTier.DEEP, None, (),
        )
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("北京今天有什么新闻"), d)

        self.assertNotIn("北京错误单边主查询", {query.text for query in plan.initial_queries})
        self.assertNotIn("北京错误倒序独立查询", {query.text for query in plan.initial_queries})
        self.assertTrue(all(
            (query.date_from is None and query.date_to is None)
            or (
                query.date_from is not None
                and query.date_to is not None
                and query.date_from <= query.date_to
            )
            for query in plan.initial_queries
        ))


class PlannerQueryCapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = planner_module()

    def test_total_semantic_queries_never_exceed_budget(self):
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic("光合作用"),),
            supplements=tuple(
                supplemental_query("primary", f"q{i}") for i in range(20)
            ),
        ))
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("什么是光合作用"), deep_decision())
        self.assertLessEqual(len(plan.initial_queries), 5)
        self.assertLessEqual(
            len(plan.initial_queries) + (1 if plan.decision.route in (SearchTier.STANDARD, SearchTier.DEEP) else 0),
            plan.budget.max_total_queries,
        )

    def test_query_text_capped_at_500_chars(self):
        long_text = "长" * 2000
        model = StaticPlannerModel(strict_payload(
            topics=(model_topic("光合作用"),),
            supplements=(supplemental_query("primary", long_text),),
        ))
        planner = self.module.SearchPlanner(model, today_provider=lambda: date(2026, 7, 29))
        plan = plan_with_context(planner, request("什么是光合作用"), standard_decision())
        for query in plan.initial_queries:
            self.assertLessEqual(len(query.text), 500)


if __name__ == "__main__":
    unittest.main()
