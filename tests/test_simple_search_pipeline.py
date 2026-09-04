from dataclasses import replace
import time
import unittest

from src.search.simple.models import (
    RequestSource,
    SearchFailure,
    SearchMode,
    SearchPlan,
    SearchQuery,
    SearchRequest,
    SearchResult,
    SearchTrace,
)
from src.search.simple.pipeline import SearchTimeouts, SimpleSearchPipeline
from src.search.simple.ranking import RankingResult


def short_results(count: int = 3) -> tuple[SearchResult, ...]:
    return tuple(
        SearchResult(
            result_id=f"R{i}",
            title=f"Title {i}",
            url=f"https://example.com/{i}",
            excerpt="短",
            provider="tavily",
            score=0.5,
        )
        for i in range(1, count + 1)
    )


class FakePlanner:
    def __init__(self, plan=None, error=None, calls=None):
        self.plan_result = plan
        self.error = error
        self.calls = calls if calls is not None else []

    def plan(self, *, mode, text, images=(), timeout_seconds=8.0):
        self.calls.append(f"planner:{mode.value}:{len(images) or 1}")
        if self.error:
            raise self.error
        if self.plan_result is not None:
            return self.plan_result
        return SearchPlan(mode, (SearchQuery("q1", text or "fallback"),))


class FakeRetriever:
    def __init__(self, results=(), calls=None):
        self.results = results
        self.calls = calls if calls is not None else []

    def run(self, plan, trace):
        self.calls.append(f"retriever:{plan.mode.value}")
        return self.results


class FakeReader:
    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []

    def enrich(self, results, *, limit, timeout_seconds, trace):
        self.calls.append(f"reader:{limit}")
        return results


class FakeRanker:
    def __init__(self, degraded=False, calls=None):
        self.degraded = degraded
        self.calls = calls if calls is not None else []

    def rank(self, question, results, *, timeout_seconds):
        self.calls.append("ranker")
        return RankingResult(results, degraded=self.degraded)


def make_pipeline(
    results=None,
    planner_result=None,
    planner_error=None,
    ranker_degraded=False,
):
    calls = []
    planner = FakePlanner(plan=planner_result, error=planner_error, calls=calls)
    retriever = FakeRetriever(
        results=results if results is not None else short_results(3),
        calls=calls,
    )
    reader = FakeReader(calls=calls)
    ranker = FakeRanker(degraded=ranker_degraded, calls=calls)
    timeouts = SearchTimeouts(
        planner=8.0,
        tavily=8.0,
        ddgs=15.0,
        reader=5.0,
        ranker=10.0,
        answer=20.0,
    )
    pipeline = SimpleSearchPipeline(
        planner,
        retriever,
        reader,
        ranker,
        timeouts=timeouts,
    )
    return pipeline, calls


class SimpleSearchPipelineTests(unittest.TestCase):
    def test_skip_returns_before_planner_and_all_search_dependencies(self):
        pipeline, calls = make_pipeline()
        outcome = pipeline.run(SearchRequest(SearchMode.SKIP, "你好"))
        self.assertIs(SearchMode.SKIP, outcome.plan.mode)
        self.assertEqual([], calls)

    def test_light_mode_is_passed_unchanged_and_reads_one(self):
        pipeline, calls = make_pipeline(results=short_results(3))
        outcome = pipeline.run(
            SearchRequest(
                SearchMode.LIGHT,
                "q",
                ("data:image/png;base64,AAA",),
            )
        )
        self.assertEqual(
            ("planner:light:1", "retriever:light", "reader:1", "ranker"),
            tuple(calls),
        )
        self.assertIs(SearchMode.LIGHT, outcome.plan.mode)

    def test_standard_reads_two_and_never_accepts_planner_mode_change(self):
        pipeline, calls = make_pipeline(
            planner_result=SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "q"),))
        )
        outcome = pipeline.run(SearchRequest(SearchMode.STANDARD, "q"))
        self.assertIs(SearchMode.STANDARD, outcome.plan.mode)
        self.assertEqual(SearchFailure.PROVIDER_UNAVAILABLE, outcome.failure)

    def test_ranker_degradation_preserves_results_and_adds_warning(self):
        pipeline, _ = make_pipeline(ranker_degraded=True)
        outcome = pipeline.run(SearchRequest(SearchMode.LIGHT, "q"))
        self.assertTrue(outcome.results)
        self.assertTrue(outcome.trace.ranker_degraded)
        self.assertEqual("信息可能不完整。", outcome.warning)

    def test_no_usable_result_and_unexpected_exception_are_nonthrowing(self):
        empty, _ = make_pipeline(results=())
        broken, _ = make_pipeline(planner_error=RuntimeError("private"))
        self.assertEqual(
            SearchFailure.NO_RESULTS,
            empty.run(SearchRequest(SearchMode.LIGHT, "q")).failure,
        )
        self.assertEqual(
            SearchFailure.PROVIDER_UNAVAILABLE,
            broken.run(SearchRequest(SearchMode.LIGHT, "q")).failure,
        )


if __name__ == "__main__":
    unittest.main()
