from pathlib import Path
import unittest

from src.search.simple.models import (
    SearchMode, SearchPlan, SearchQuery, SearchRequest, SearchResult,
)


class SimpleSearchModelTests(unittest.TestCase):
    def test_mode_has_only_skip_light_and_standard(self):
        self.assertEqual(["skip", "light", "standard"], [item.value for item in SearchMode])

    def test_light_plan_rejects_multiple_queries(self):
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"), SearchQuery("q2", "b")))

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
