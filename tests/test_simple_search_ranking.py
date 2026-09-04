from types import SimpleNamespace
import unittest

from src.search.simple.models import SearchResult
from src.search.simple.ranking import EvidenceRanker


class FakeLLM:
    def __init__(self, content="", error=None):
        self.content = content
        self.error = error
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


def three_results() -> tuple[SearchResult, ...]:
    return (
        SearchResult("R1", "Title 1", "https://example.com/1", "Excerpt 1", "tavily", 0.5),
        SearchResult("R2", "Title 2", "https://example.com/2", "Excerpt 2", "tavily", 0.5),
        SearchResult("R3", "Title 3", "https://example.com/3", "Excerpt 3", "ddgs", 0.5),
    )


def by_id(results: tuple[SearchResult, ...], result_id: str) -> SearchResult:
    return next(r for r in results if r.result_id == result_id)


class EvidenceRankerTests(unittest.TestCase):
    def test_known_numeric_scores_sort_stably_and_remove_explicit_zero(self):
        ranked = EvidenceRanker(FakeLLM('{"scores":{"R1":0.2,"R2":0.9,"R3":0}}')).rank(
            "q", three_results(), timeout_seconds=10
        )
        self.assertEqual(("R2", "R1"), tuple(item.result_id for item in ranked.results))
        self.assertFalse(ranked.degraded)

    def test_invalid_or_missing_scores_default_to_half(self):
        ranked = EvidenceRanker(FakeLLM('{"scores":{"R1":"bad","R3":2,"unknown":1}}')).rank(
            "q", three_results(), timeout_seconds=10
        )
        self.assertEqual(1.0, by_id(ranked.results, "R3").score)
        self.assertEqual(0.5, by_id(ranked.results, "R1").score)
        self.assertEqual(0.5, by_id(ranked.results, "R2").score)

    def test_exception_preserves_provider_order_and_degrades(self):
        ranked = EvidenceRanker(FakeLLM(error=TimeoutError())).rank(
            "q", three_results(), timeout_seconds=10
        )
        self.assertEqual(("R1", "R2", "R3"), tuple(item.result_id for item in ranked.results))
        self.assertTrue(ranked.degraded)

    def test_prompt_excludes_urls_and_timeout_is_forwarded(self):
        llm = FakeLLM('{"scores":{"R1":1}}')
        EvidenceRanker(llm).rank("q", three_results(), timeout_seconds=9.5)
        self.assertNotIn("https://", repr(llm.calls[0][0]))
        self.assertEqual(9.5, llm.calls[0][1]["timeout_seconds"])


if __name__ == "__main__":
    unittest.main()
