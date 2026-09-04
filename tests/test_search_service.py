import unittest
from unittest.mock import MagicMock, patch

from src.search.simple.models import (
    RequestSource,
    SearchFailure,
    SearchMode,
    SearchOutcome,
    SearchPlan,
    SearchQuery,
    SearchResult,
    SearchTrace,
)
from src.services.search_service import (
    has_search_results,
    normalize_search_query,
    reset_search_service,
    search,
    web_search,
)


def standard_outcome(
    failure: SearchFailure | None = None,
    results: tuple[SearchResult, ...] = (),
) -> SearchOutcome:
    plan = SearchPlan(
        mode=SearchMode.STANDARD,
        queries=(SearchQuery(query_id="q1", text="current news"),),
    )
    trace = SearchTrace(
        request_id="req-compat-1",
        source=RequestSource.COMPATIBILITY,
        mode=SearchMode.STANDARD,
    )
    return SearchOutcome(
        plan=plan,
        results=results,
        trace=trace,
        failure=failure,
    )


class SearchServiceCompatibilityTests(unittest.TestCase):
    @patch("src.services.search_service.get_simple_search_pipeline")
    def test_compatibility_search_is_always_standard(self, factory):
        factory.return_value.run.return_value = standard_outcome(
            results=(
                SearchResult(
                    result_id="r1",
                    title="News Title",
                    url="https://example.com/news",
                    excerpt="News summary",
                    provider="tavily",
                ),
            )
        )
        result = search("  current   news ")
        request = factory.return_value.run.call_args.args[0]
        self.assertIs(SearchMode.STANDARD, request.mode)
        self.assertIs(RequestSource.COMPATIBILITY, request.source)
        self.assertEqual("current news", request.text)
        self.assertEqual((), request.images)
        self.assertTrue(result.ok)
        self.assertEqual("success", result.status)
        self.assertIn("1. News Title\nNews summary\nhttps://example.com/news", result.text)

    @patch("src.services.search_service.get_simple_search_pipeline")
    def test_compatibility_search_failure_mapping(self, factory):
        factory.return_value.run.return_value = standard_outcome(
            failure=SearchFailure.PROVIDER_UNAVAILABLE
        )
        result = search("current news")
        self.assertFalse(result.ok)
        self.assertEqual(SearchFailure.PROVIDER_UNAVAILABLE.value, result.status)
        self.assertEqual("在线检索未完成。", result.text)

    def test_compatibility_search_empty_query(self):
        result = search("   ")
        self.assertFalse(result.ok)
        self.assertEqual("empty_query", result.status)
        self.assertEqual("没有可搜索的关键词。", result.text)

    @patch("src.services.search_service.reset_simple_search_pipeline")
    def test_reset_delegates_only_to_simple_factory(self, reset):
        reset_search_service()
        reset.assert_called_once_with()

    @patch("src.services.search_service.get_simple_search_pipeline")
    def test_web_search_and_has_search_results_helpers(self, factory):
        factory.return_value.run.return_value = standard_outcome(
            results=(
                SearchResult(
                    result_id="r1",
                    title="News Title",
                    url="https://example.com/news",
                    excerpt="News summary",
                    provider="tavily",
                ),
            )
        )
        text = web_search("news")
        self.assertIn("News Title", text)
        res = search("news")
        self.assertTrue(has_search_results(res))


if __name__ == "__main__":
    unittest.main()
