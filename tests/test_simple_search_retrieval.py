from __future__ import annotations

from types import SimpleNamespace
import unittest

from src.search.models import ProviderHit, ProviderResult, ProviderStatus
from src.search.simple.models import SearchMode, SearchPlan, SearchQuery, SearchTrace
from src.search.simple.retrieval import ProviderRunner


def _hit(
    query_id: str,
    url: str,
    title: str = "title",
    snippet: str = "body",
    *,
    provider: str = "tavily",
    score: float | None = 0.8,
) -> ProviderHit:
    return ProviderHit(
        provider=provider,
        query_id=query_id,
        title=title,
        url=url,
        snippet=snippet,
        score=score,
        published_at=None,
        raw_content=None,
        quality_flags=(),
    )


def _success(*hits: ProviderHit, provider: str | None = None) -> ProviderResult:
    provider_name = provider or hits[0].provider
    return ProviderResult(provider_name, ProviderStatus.SUCCESS, hits, 1)


def _empty(provider: str) -> ProviderResult:
    return ProviderResult(provider, ProviderStatus.EMPTY, (), 1)


class FakeProvider:
    def __init__(self, name: str, results: dict[str, ProviderResult | BaseException]):
        self.name = name
        self.results = results
        self.calls = []
        self.timeout_values = []

    def readiness(self):
        return SimpleNamespace(provider=self.name, configured=True, available=True)

    def search(self, query, **kwargs):
        self.calls.append(query)
        self.timeout_values.append(kwargs["timeout_seconds"])
        result = self.results[query.query_id]
        if isinstance(result, BaseException):
            raise result
        return result


class ProviderRunnerTests(unittest.TestCase):
    def test_tavily_success_prevents_ddgs_call(self):
        plan = SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "EDG"),))
        tavily = FakeProvider(
            "tavily",
            {"q1": _success(_hit("q1", "https://example.com/a", "A"))},
        )
        ddgs = FakeProvider(
            "ddgs",
            {"q1": _success(_hit("q1", "https://example.com/b", "B", provider="ddgs"))},
        )

        results = ProviderRunner((ddgs, tavily), 8, 15, 4).run(plan, SearchTrace("r1"))

        self.assertEqual(1, len(results))
        self.assertEqual("tavily", results[0].provider)
        self.assertEqual([], ddgs.calls)

    def test_ddgs_receives_only_unresolved_queries(self):
        plan = SearchPlan(
            SearchMode.STANDARD,
            (SearchQuery("q1", "a"), SearchQuery("q2", "b")),
        )
        tavily = FakeProvider(
            "tavily",
            {
                "q1": _success(_hit("q1", "https://example.com/q1")),
                "q2": _empty("tavily"),
            },
        )
        ddgs = FakeProvider(
            "ddgs",
            {"q2": _success(_hit("q2", "https://example.com/q2", provider="ddgs"))},
        )

        ProviderRunner((tavily, ddgs), 8, 15, 4).run(plan, SearchTrace("r1"))

        self.assertEqual(["q2"], [call.query_id for call in ddgs.calls])

    def test_unsafe_urls_are_dropped_and_tracking_variants_are_deduplicated(self):
        plan = SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"),))
        tavily = FakeProvider(
            "tavily",
            {
                "q1": _success(
                    _hit("q1", "http://127.0.0.1/private"),
                    _hit("q1", "javascript:alert(1)"),
                    _hit("q1", "https://example.com/a?utm_source=first"),
                    _hit("q1", "https://EXAMPLE.com/a/?utm_medium=second#fragment"),
                )
            },
        )
        ddgs = FakeProvider("ddgs", {})

        results = ProviderRunner((tavily, ddgs), 8, 15, 4).run(plan, SearchTrace("r1"))

        self.assertEqual(["https://example.com/a"], [item.url for item in results])

    def test_configured_per_provider_timeouts_are_forwarded(self):
        plan = SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"),))
        tavily = FakeProvider("tavily", {"q1": _empty("tavily")})
        ddgs = FakeProvider(
            "ddgs",
            {"q1": _success(_hit("q1", "https://example.com/a", provider="ddgs"))},
        )

        ProviderRunner((tavily, ddgs), 8, 15, 4).run(plan, SearchTrace("r1"))

        self.assertEqual([8], tavily.timeout_values)
        self.assertEqual([15], ddgs.timeout_values)

    def test_provider_exception_falls_back_and_trace_contains_only_statuses_and_count(self):
        plan = SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "secret query"),))
        tavily = FakeProvider("tavily", {"q1": RuntimeError("secret response body")})
        ddgs = FakeProvider(
            "ddgs",
            {"q1": _success(_hit("q1", "https://example.com/a", provider="ddgs"))},
        )
        trace = SearchTrace("r1")

        results = ProviderRunner((tavily, ddgs), 8, 15, 4).run(plan, trace)

        self.assertEqual(1, len(results))
        self.assertEqual("error", trace.provider_statuses["tavily"])
        self.assertEqual("success", trace.provider_statuses["ddgs"])
        self.assertEqual(1, trace.candidate_count)
        self.assertNotIn("secret", repr(trace.to_safe_dict()))


if __name__ == "__main__":
    unittest.main()
