import math
import unittest
from dataclasses import replace

from src.search.budget import (
    DEFAULT_SEARCH_BUDGET_POLICY,
    RouteStageBudget,
    SearchBudgetPolicy,
)
from src.search.models import SearchTier


class SearchBudgetPolicyTests(unittest.TestCase):
    def test_light_stages_are_independent_and_watchdog_is_derived(self):
        budget = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.LIGHT)

        self.assertEqual(
            {
                "analysis_route_seconds": 3,
                "planner_seconds": 0,
                "initial_ddgs_seconds": 30,
                "initial_tavily_seconds": 6,
                "initial_reader_seconds": 4,
                "initial_judge_seconds": 4,
                "gap_seconds": 0,
                "repair_planner_seconds": 0,
                "repair_ddgs_seconds": 0,
                "repair_tavily_seconds": 0,
                "repair_reader_seconds": 0,
                "repair_judge_seconds": 0,
                "answer_seconds": 4,
                "validator_seconds": 4,
                "renderer_seconds": 1,
                "scheduling_margin_seconds": 2,
            },
            vars(budget),
        )
        self.assertEqual(58, DEFAULT_SEARCH_BUDGET_POLICY.maximum_request_seconds(SearchTier.LIGHT))

    def test_standard_stages_and_watchdog_are_derived(self):
        budget = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.STANDARD)

        self.assertEqual(
            {
                "analysis_route_seconds": 3,
                "planner_seconds": 4,
                "initial_ddgs_seconds": 30,
                "initial_tavily_seconds": 8,
                "initial_reader_seconds": 6,
                "initial_judge_seconds": 5,
                "gap_seconds": 1,
                "repair_planner_seconds": 2,
                "repair_ddgs_seconds": 30,
                "repair_tavily_seconds": 5,
                "repair_reader_seconds": 3,
                "repair_judge_seconds": 4,
                "answer_seconds": 4,
                "validator_seconds": 4,
                "renderer_seconds": 1,
                "scheduling_margin_seconds": 2,
            },
            vars(budget),
        )
        self.assertEqual(112, DEFAULT_SEARCH_BUDGET_POLICY.maximum_request_seconds(SearchTier.STANDARD))

    def test_every_active_ddgs_stage_has_thirty_second_budget(self):
        light = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.LIGHT)
        standard = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.STANDARD)

        self.assertEqual(30, light.initial_ddgs_seconds)
        self.assertEqual(30, standard.initial_ddgs_seconds)
        self.assertEqual(30, standard.repair_ddgs_seconds)

    def test_watchdog_changes_when_a_stage_changes_without_a_second_constant(self):
        original = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.LIGHT)
        changed = replace(original, initial_reader_seconds=original.initial_reader_seconds + 2)

        self.assertEqual(
            DEFAULT_SEARCH_BUDGET_POLICY.maximum_for_budget(original) + 2,
            DEFAULT_SEARCH_BUDGET_POLICY.maximum_for_budget(changed),
        )

    def test_skip_has_no_budget_but_is_not_a_route_budget(self):
        self.assertEqual(0, DEFAULT_SEARCH_BUDGET_POLICY.maximum_request_seconds(SearchTier.SKIP))
        with self.assertRaises(ValueError):
            DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.SKIP)

    def test_stage_budget_rejects_boolean_non_finite_and_negative_values(self):
        valid = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.LIGHT)
        for value in (True, -1, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    replace(valid, initial_ddgs_seconds=value)

    def test_policy_mapping_is_immutable(self):
        with self.assertRaises(TypeError):
            DEFAULT_SEARCH_BUDGET_POLICY.budgets[SearchTier.LIGHT] = RouteStageBudget(*([0] * 16))

        with self.assertRaises(ValueError):
            SearchBudgetPolicy({SearchTier.SKIP: DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.LIGHT)})


class IndependentStageBudgetExecutionTests(unittest.TestCase):
    def test_analysis_llm_receives_3_seconds(self):
        from src.search.models import RetrievalRequest
        from src.search.router import LLMRequestAnalyzer
        from types import SimpleNamespace

        captured = {}

        class CapturingLLM:
            def chat(self, messages, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(content='{"search_needed": false}')

        analyzer = LLMRequestAnalyzer(CapturingLLM())
        analyzer.analyze(RetrievalRequest("什么是光合作用"))
        self.assertEqual(3.0, captured.get("timeout_seconds"))

    def test_standard_planner_receives_4_seconds(self):
        from datetime import date
        import src.search.models as m
        from src.search.planner import SearchPlanner
        from tests.test_search_models import SearchModelFixtures
        from types import SimpleNamespace

        captured = {}

        class CapturingPlannerModel:
            def chat(self, messages, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    content='{"queries": [], "required_topics": [{"topic_id": "topic-1", "label": "t", "material": true}]}'
                )

        planner = SearchPlanner(CapturingPlannerModel(), today_provider=lambda: date(2026, 7, 29))
        fixtures = SearchModelFixtures(m)
        decision = m.RetrievalDecision(m.SearchTier.STANDARD, None, True, ())
        analysis = fixtures.analysis()
        planner.plan(
            m.RetrievalRequest("比较 Python 并发 API"),
            decision,
            analysis.retrieval,
            analysis.freshness,
        )
        self.assertEqual(4.0, captured.get("timeout_seconds"))

    def test_validator_receives_one_4_second_local_stage_budget(self):
        import src.search.models as m
        from src.search.validation import validate_and_filter
        from tests.test_search_models import SearchModelFixtures

        discovered_timeouts = []
        verified_timeouts = []

        class CapturingDiscoverer:
            def discover(self, draft, bundle, **kwargs):
                discovered_timeouts.append(kwargs.get("timeout_seconds"))
                return ()

        class CapturingVerifier:
            def verify(self, payload, **kwargs):
                verified_timeouts.append(kwargs.get("timeout_seconds"))
                return {}

        fixtures = SearchModelFixtures(m)
        draft = m.GroundedDraft(
            answer_blocks=(m.AnswerBlock("B1", "factual", "答案", ("C1",)),),
            claims=(m.Claim("C1", "B1", "答案", True, ()),),
            limitations=(),
            conflict_summary=(),
            used_knowledge_fallback=False,
        )
        bundle = fixtures.bundle()
        answer_state = m.AnswerState(
            None,
            m.AnswerGenerationMode.GROUNDED,
            m.AnswerCertainty.VERIFIED,
            m.AllowedClaimScope.ALL_SUPPORTED,
            (),
            (),
            m.ValidatorRequirement.NORMAL,
        )

        res = validate_and_filter(
            draft,
            bundle,
            answer_state,
            claim_discoverer=CapturingDiscoverer(),
            semantic_verifier=CapturingVerifier(),
            timeout_seconds=4.0,
        )
        self.assertIsNotNone(res)
        self.assertEqual(1, len(discovered_timeouts))
        self.assertAlmostEqual(4.0, discovered_timeouts[0], delta=0.1)

    def test_no_separate_watchdog_constants_exist(self):
        import src.search.budget as budget_mod
        import src.search.models as models_mod
        for mod in (budget_mod, models_mod):
            self.assertFalse(hasattr(mod, "LIGHT_WATCHDOG"))
            self.assertFalse(hasattr(mod, "STANDARD_WATCHDOG"))

    def test_renderer_timeout_returns_timeout_outcome(self):
        from src.chat.chat_service import _render_view
        import src.search.models as m
        from tests.test_search_models import SearchModelFixtures
        import time

        fixtures = SearchModelFixtures(m)
        render_state = m.RenderState(
            outcome=m.RenderOutcome.ANSWER,
            visible_blocks=(m.AnswerBlock("B1", "factual", "text", ()),),
            visible_claims=(),
            citation_map={},
            used_sources=(),
            conflict_groups=(),
            disclosure_codes=(),
            warning_codes=(),
        )
        trace = m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT)
        result = m.SearchPipelineResult(
            fixtures.decision(),
            fixtures.plan(),
            fixtures.bundle(),
            trace,
            m.SearchFailureCode.INSUFFICIENT_EVIDENCE,
            analysis=fixtures.analysis(),
        )

        with unittest.mock.patch("src.chat.chat_service.render_search_reply", side_effect=lambda *args, **kwargs: (time.sleep(0.1), None)[1]):
            rendered = _render_view(render_state, result, timeout_seconds=0.01)
            self.assertEqual("回复格式化超时，请稍后重试。", rendered.text)
            self.assertEqual(m.RenderOutcome.TIMEOUT, result.trace.render_outcome)
            self.assertEqual(0, result.trace.render_citation_count)
            self.assertEqual(0, result.trace.render_source_count)
