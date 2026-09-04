import math
import unittest

from src.config import Config
from src.search.simple.models import (
    OutputKind,
    RequestSource,
    SearchMode,
    SearchPlan,
    SearchQuery,
    SearchRequest,
    SearchTrace,
)


class SimpleSearchModelTests(unittest.TestCase):
    def test_request_normalizes_text_images_and_owns_mode(self):
        request = SearchRequest(
            mode=SearchMode.STANDARD,
            text="  看看   这个  ",
            images=[" data:image/png;base64,AAA ", ""],
            source=RequestSource.COMMAND,
        )
        self.assertIs(SearchMode.STANDARD, request.mode)
        self.assertEqual("看看 这个", request.text)
        self.assertEqual(("data:image/png;base64,AAA",), request.images)
        self.assertFalse(hasattr(request, "force_" + "search"))
        self.assertFalse(hasattr(request, "has_" + "images"))

    def test_text_or_image_is_required(self):
        with self.assertRaisesRegex(ValueError, "text or images"):
            SearchRequest(mode=SearchMode.LIGHT, text="", images=())

    def test_plan_enforces_fixed_query_counts(self):
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.LIGHT, ())
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"), SearchQuery("q2", "b")))
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.STANDARD, tuple(SearchQuery(f"q{i}", str(i)) for i in range(4)))
        self.assertEqual((), SearchPlan(SearchMode.SKIP, ()).queries)

    def test_safe_trace_has_closed_metadata_and_no_request_content(self):
        trace = SearchTrace(
            "r1",
            source=RequestSource.CHAT,
            mode=SearchMode.LIGHT,
            query_count=1,
        )
        trace.provider_statuses["tavily"] = "error"
        trace.output_kind = OutputKind.SEARCH_FAILURE
        safe = trace.to_safe_dict()
        self.assertEqual("chat", safe["source"])
        self.assertEqual("light", safe["mode"])
        self.assertEqual("error", safe["provider_statuses"]["tavily"])
        self.assertNotIn("text", safe)
        self.assertNotIn("images", safe)
        self.assertNotIn("url", repr(safe).lower())

    def test_all_search_timeouts_are_finite_and_at_least_point_one(self):
        fields = (
            "search_planner_timeout",
            "search_tavily_timeout",
            "search_ddgs_timeout",
            "search_reader_timeout",
            "search_ranker_timeout",
            "search_answer_timeout",
        )
        for field in fields:
            for value in (math.nan, math.inf, -math.inf, 0.0, -1.0):
                with self.subTest(field=field, value=value):
                    self.assertEqual(0.1, getattr(Config(**{field: value}), field))
