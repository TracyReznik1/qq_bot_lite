import math
import unittest

from src.config import Config
from src.search.simple.models import (
    SearchMode, SearchPlan, SearchQuery, SearchRequest, SearchResult,
)


class SimpleSearchModelTests(unittest.TestCase):
    def test_mode_has_only_skip_light_and_standard(self):
        self.assertEqual(["skip", "light", "standard"], [item.value for item in SearchMode])

    def test_light_plan_rejects_multiple_queries(self):
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"), SearchQuery("q2", "b")))

    def test_plan_coerces_raw_mode(self):
        plan = SearchPlan("light", (SearchQuery("q1", "a"),))
        self.assertIs(SearchMode.LIGHT, plan.mode)

    def test_raw_skip_mode_still_rejects_queries(self):
        with self.assertRaises(ValueError):
            SearchPlan("skip", (SearchQuery("q1", "a"),))

    def test_plan_rejects_unknown_raw_mode(self):
        with self.assertRaises(ValueError):
            SearchPlan("unknown", ())

    def test_standard_plan_rejects_more_than_three_queries(self):
        queries = tuple(SearchQuery(f"q{i}", str(i)) for i in range(1, 5))
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.STANDARD, queries)

    def test_search_result_clamps_score(self):
        result = SearchResult("R1", "title", "https://example.com", "body", "tavily", 1.8)
        self.assertEqual(1.0, result.score)

    def test_force_search_request_is_nonempty(self):
        with self.assertRaises(ValueError):
            SearchRequest("", force_search=True)

    def test_config_normalizes_non_finite_search_timeouts_to_safe_minimum(self):
        timeout_fields = (
            "search_planner_timeout",
            "search_tavily_timeout",
            "search_ddgs_timeout",
            "search_reader_timeout",
            "search_ranker_timeout",
            "search_answer_timeout",
        )
        for non_finite in (math.nan, math.inf, -math.inf):
            for field_name in timeout_fields:
                with self.subTest(value=non_finite, field=field_name):
                    current = Config(**{field_name: non_finite})
                    self.assertEqual(0.1, getattr(current, field_name))
