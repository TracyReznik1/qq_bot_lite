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

        self.assertEqual(3, budget.analysis_route_seconds)
        self.assertEqual(6, budget.initial_ddgs_seconds)
        self.assertEqual(6, budget.initial_tavily_seconds)
        self.assertEqual(4, budget.initial_reader_seconds)
        self.assertEqual(4, budget.initial_judge_seconds)
        self.assertEqual(0, budget.planner_seconds)
        self.assertEqual(0, budget.repair_ddgs_seconds)
        self.assertEqual(34, DEFAULT_SEARCH_BUDGET_POLICY.maximum_request_seconds(SearchTier.LIGHT))

    def test_standard_stages_and_watchdog_are_derived(self):
        budget = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.STANDARD)

        self.assertEqual(
            (4, 8, 8, 6, 5),
            (
                budget.planner_seconds,
                budget.initial_ddgs_seconds,
                budget.initial_tavily_seconds,
                budget.initial_reader_seconds,
                budget.initial_judge_seconds,
            ),
        )
        self.assertEqual(
            (1, 2, 5, 5, 3, 4),
            (
                budget.gap_seconds,
                budget.repair_planner_seconds,
                budget.repair_ddgs_seconds,
                budget.repair_tavily_seconds,
                budget.repair_reader_seconds,
                budget.repair_judge_seconds,
            ),
        )
        self.assertEqual(65, DEFAULT_SEARCH_BUDGET_POLICY.maximum_request_seconds(SearchTier.STANDARD))

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
