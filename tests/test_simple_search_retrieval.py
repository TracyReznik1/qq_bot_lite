from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace
import unittest

from src.search.simple.models import (
    RequestSource,
    SearchMode,
    SearchPlan,
    SearchQuery,
    SearchResult,
    SearchTrace,
)
from src.search.simple.providers import (
    ProviderHit,
    ProviderReadiness,
    ProviderResult,
    ProviderStatus,
)
from src.search.simple.retrieval import ProviderRunner


def hit(
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
        raw_content=None,
    )


def result(status: str, *hits: ProviderHit, provider: str = "tavily") -> ProviderResult:
    return ProviderResult(
        provider=provider,
        status=ProviderStatus(status),
        hits=hits,
        latency_ms=1.0,
    )


class FakeProvider:
    def __init__(self, name: str, results: dict[str, ProviderResult | BaseException]):
        self.name = name
        self.results = results
        self.calls = []
        self.timeout_values = []

    def readiness(self) -> ProviderReadiness:
        return ProviderReadiness(self.name, configured=True, available=True)

    def search(self, query: SearchQuery, *, mode: SearchMode, max_results: int, timeout_seconds: float) -> ProviderResult:
        self.calls.append(query)
        self.timeout_values.append(timeout_seconds)
        res = self.results.get(query.query_id)
        if isinstance(res, BaseException):
            raise res
        if res is None:
            return ProviderResult(self.name, ProviderStatus.EMPTY, (), 0.0)
        return res


class BlockingProvider:
    def __init__(self, name: str):
        self.name = name

    def readiness(self) -> ProviderReadiness:
        return ProviderReadiness(self.name, configured=True, available=True)

    def search(self, query: SearchQuery, *, mode: SearchMode, max_results: int, timeout_seconds: float) -> ProviderResult:
        time.sleep(1.0)
        return ProviderResult(self.name, ProviderStatus.TIMEOUT, (), 1000.0)


class EmptyProvider:
    def __init__(self, name: str):
        self.name = name

    def readiness(self) -> ProviderReadiness:
        return ProviderReadiness(self.name, configured=True, available=True)

    def search(self, query: SearchQuery, *, mode: SearchMode, max_results: int, timeout_seconds: float) -> ProviderResult:
        return ProviderResult(self.name, ProviderStatus.EMPTY, (), 0.0)


class ProviderRunnerTests(unittest.TestCase):
    def trace(self, mode: SearchMode = SearchMode.LIGHT) -> SearchTrace:
        return SearchTrace("r1", RequestSource.CHAT, mode)

    def test_tavily_usable_hit_resolves_query_without_ddgs(self):
        plan = SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"),))
        tavily = FakeProvider("tavily", {"q1": result("success", hit("q1", "https://example.com/a"))})
        ddgs = FakeProvider("ddgs", {})
        output = ProviderRunner((tavily, ddgs), 8, 15, 4).run(plan, self.trace())
        self.assertEqual(("https://example.com/a",), tuple(item.url for item in output))
        self.assertEqual([], ddgs.calls)

    def test_tavily_invalid_urls_fall_back_to_ddgs(self):
        plan = SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"),))
        tavily = FakeProvider("tavily", {"q1": result("success", hit("q1", "http://127.0.0.1/private"))})
        ddgs = FakeProvider("ddgs", {"q1": result("success", hit("q1", "https://example.com/ddgs", provider="ddgs"))})
        output = ProviderRunner((tavily, ddgs), 8, 15, 4).run(plan, self.trace())
        self.assertEqual(["q1"], [call.query_id for call in ddgs.calls])
        self.assertEqual("https://example.com/ddgs", output[0].url)

    def test_noncooperative_calls_are_bounded(self):
        started = time.monotonic()
        output = ProviderRunner(
            (BlockingProvider("tavily"), EmptyProvider("ddgs")), 0.05, 0.05, 4,
        ).run(SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"),)), self.trace())
        self.assertEqual((), output)
        self.assertLess(time.monotonic() - started, 0.4)

    def test_caps_light_at_five_and_standard_at_eight(self):
        hits_ten = tuple(hit("q1", f"https://example.com/{i}", f"T{i}") for i in range(10))
        tavily = FakeProvider("tavily", {"q1": result("success", *hits_ten)})
        ddgs = FakeProvider("ddgs", {})

        light_res = ProviderRunner((tavily, ddgs), 8, 15, 10).run(
            SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"),)), self.trace(SearchMode.LIGHT)
        )
        self.assertEqual(5, len(light_res))

        standard_res = ProviderRunner((tavily, ddgs), 8, 15, 10).run(
            SearchPlan(SearchMode.STANDARD, (SearchQuery("q1", "a"),)), self.trace(SearchMode.STANDARD)
        )
        self.assertEqual(8, len(standard_res))

    def test_safe_trace_contains_no_sensitive_data(self):
        trace = self.trace()
        tavily = FakeProvider("tavily", {"q1": result("success", hit("q1", "https://example.com/secret_url?api_key=123"))})
        ddgs = FakeProvider("ddgs", {})
        ProviderRunner((tavily, ddgs), 8, 15, 4).run(
            SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "sensitive query"),)), trace
        )
        safe = trace.to_safe_dict()
        self.assertEqual("success", safe["provider_statuses"]["tavily"])
        self.assertNotIn("secret", repr(safe).lower())
        self.assertNotIn("sensitive", repr(safe).lower())
        self.assertNotIn("http", repr(safe).lower())

    def test_source_contains_no_legacy_adapters(self):
        source = Path("src/search/simple/retrieval.py").read_text(encoding="utf-8")
        self.assertNotIn("_legacy_query", source)
        self.assertNotIn("_legacy_tier", source)
        self.assertNotIn("src.search." + "models", source)
        self.assertNotIn("src.search." + "providers", source)


if __name__ == "__main__":
    unittest.main()
