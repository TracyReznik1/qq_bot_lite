import time
import unittest
from types import SimpleNamespace

from src.search.simple.models import RequestSource, SearchMode, SearchResult, SearchTrace
from src.search.simple.reader import OnDemandReader


class FakeFetch:
    def __init__(self, text="full text", ok=True, delay=0.0):
        self.text = text
        self.ok = ok
        self.delay = delay
        self.calls = []

    def __call__(self, url, *, timeout=None):
        self.calls.append((url, timeout))
        if self.delay > 0:
            time.sleep(self.delay)
        return SimpleNamespace(ok=self.ok, text=self.text)


def make_result(result_id: str, excerpt: str, url: str | None = None) -> SearchResult:
    return SearchResult(
        result_id=result_id,
        title="Title",
        url=url or f"https://example.com/{result_id}",
        excerpt=excerpt,
        provider="tavily",
        score=0.8,
    )


class OnDemandReaderTests(unittest.TestCase):
    def test_reads_only_missing_or_short_snippets_up_to_limit(self):
        fake_fetch = FakeFetch("fetched text")
        results = (make_result("R1", "短"), make_result("R2", "字" * 80), make_result("R3", ""))
        trace = SearchTrace("r1", RequestSource.CHAT, SearchMode.LIGHT)
        enriched = OnDemandReader(fake_fetch).enrich(
            results,
            limit=1,
            timeout_seconds=5,
            trace=trace,
        )
        self.assertEqual([("https://example.com/R1", 5)], fake_fetch.calls)
        self.assertEqual("字" * 80, enriched[1].excerpt)
        self.assertEqual(1, trace.reader_count)

    def test_failed_read_preserves_provider_snippet(self):
        failing_fetch = FakeFetch(ok=False)
        result = make_result("R1", "短摘要")
        output = OnDemandReader(failing_fetch).enrich(
            (result,),
            limit=1,
            timeout_seconds=5,
            trace=SearchTrace("r1", RequestSource.CHAT, SearchMode.LIGHT),
        )
        self.assertEqual("短摘要", output[0].excerpt)

    def test_reader_cleans_and_caps_successful_page_text(self):
        long_fetch = FakeFetch("text \x00 " + "a" * 2000)
        output = OnDemandReader(long_fetch).enrich(
            (make_result("R1", ""),),
            limit=1,
            timeout_seconds=5,
            trace=SearchTrace("r1", RequestSource.CHAT, SearchMode.LIGHT),
        )
        self.assertEqual(1500, len(output[0].excerpt))
        self.assertNotIn("\x00", output[0].excerpt)

    def test_noncooperative_reader_is_request_bounded(self):
        started = time.monotonic()
        blocking_fetch = FakeFetch(delay=1.0)
        OnDemandReader(blocking_fetch).enrich(
            (make_result("R1", "短"),),
            limit=1,
            timeout_seconds=0.05,
            trace=SearchTrace("r1", RequestSource.CHAT, SearchMode.LIGHT),
        )
        self.assertLess(time.monotonic() - started, 0.4)


if __name__ == "__main__":
    unittest.main()
