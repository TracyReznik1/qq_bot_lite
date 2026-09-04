from datetime import date
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src.search.simple.models import SearchMode, SearchQuery
from src.search.simple.providers import (
    ProviderErrorCode,
    ProviderHit,
    ProviderReadiness,
    ProviderResult,
    ProviderStatus,
    SearchProvider,
)
from src.search.simple.tavily import TavilySearchProvider
from src.search.simple.ddgs import DDGSSearchProvider


class FakeSearchProvider:
    name = "fake"

    def readiness(self) -> ProviderReadiness:
        return ProviderReadiness("fake", configured=True, available=True)

    def search(
        self,
        query: SearchQuery,
        *,
        mode: SearchMode,
        max_results: int,
        timeout_seconds: float,
    ) -> ProviderResult:
        hit = ProviderHit(
            provider=self.name,
            query_id=query.query_id,
            title="Title",
            url="https://example.com",
            snippet="Snippet",
            score=0.9,
            raw_content=None,
        )
        return ProviderResult(
            provider=self.name,
            status=ProviderStatus.SUCCESS,
            hits=(hit,),
            latency_ms=10.0,
            error_code=None,
            date_filter_normalized=False,
            parameter_retry_attempted=False,
        )


def tavily_provider(response_data=None, error=None):
    client = MagicMock()
    if error:
        client.search.side_effect = error
    else:
        client.search.return_value = response_data if response_data is not None else {"results": []}
    provider = TavilySearchProvider(api_key="test-key", proxy_url="")
    provider._client = client
    return provider, client


class SimpleSearchProviderContractTests(unittest.TestCase):
    def test_provider_records_and_protocol_contract(self):
        readiness = ProviderReadiness(provider="test", configured=True, available=True)
        self.assertEqual("test", readiness.provider)
        self.assertTrue(readiness.configured)
        self.assertTrue(readiness.available)

        hit = ProviderHit(
            provider="test",
            query_id="q1",
            title="Example",
            url="https://example.com",
            snippet="Snippet text",
            score=0.8,
            raw_content="Content",
        )
        self.assertEqual("https://example.com", hit.url)

        res = ProviderResult(
            provider="test",
            status=ProviderStatus.SUCCESS,
            hits=(hit,),
            latency_ms=12.5,
            error_code=None,
            date_filter_normalized=False,
            parameter_retry_attempted=False,
        )
        self.assertIs(ProviderStatus.SUCCESS, res.status)

        fake = FakeSearchProvider()
        self.assertTrue(isinstance(fake, SearchProvider))

        query = SearchQuery(query_id="q1", text="test query")
        search_res = fake.search(query, mode=SearchMode.LIGHT, max_results=4, timeout_seconds=8)
        self.assertIs(ProviderStatus.SUCCESS, search_res.status)
        self.assertEqual(1, len(search_res.hits))

    def test_providers_module_has_no_forbidden_legacy_imports(self):
        source = Path("src/search/simple/providers.py").read_text(encoding="utf-8")
        self.assertNotIn("src.search." + "models", source)
        self.assertNotIn("src.search." + "providers", source)


class SimpleProviderImplementationTests(unittest.TestCase):
    def test_tavily_light_forwards_basic_mode_and_timeout(self):
        provider, client = tavily_provider({"results": []})
        result = provider.search(
            SearchQuery("q1", "EDG 上海冠军赛"),
            mode=SearchMode.LIGHT,
            max_results=4,
            timeout_seconds=8,
        )
        self.assertIs(ProviderStatus.EMPTY, result.status)
        client.search.assert_called_once_with(
            "EDG 上海冠军赛",
            search_depth="basic",
            max_results=4,
            timeout=8,
            include_raw_content=False,
        )

    def test_tavily_standard_news_and_filters_use_final_query_fields(self):
        provider, client = tavily_provider({"results": []})
        provider.search(
            SearchQuery(
                "q1",
                "news",
                date_from=date(2026, 9, 1),
                date_to=date(2026, 9, 4),
                include_domains=("example.com",),
                exclude_domains=("bad.example",),
                news=True,
            ),
            mode=SearchMode.STANDARD,
            max_results=8,
            timeout_seconds=7,
        )
        kwargs = client.search.call_args.kwargs
        self.assertEqual("advanced", kwargs["search_depth"])
        self.assertTrue(kwargs["include_raw_content"])
        self.assertEqual("news", kwargs["topic"])
        self.assertEqual("2026-09-01", kwargs["start_date"])
        self.assertEqual("2026-09-04", kwargs["end_date"])
        self.assertEqual(("example.com",), kwargs["include_domains"])
        self.assertEqual(("bad.example",), kwargs["exclude_domains"])

    def test_tavily_equal_date_bounds_normalized(self):
        provider, client = tavily_provider({"results": []})
        result = provider.search(
            SearchQuery(
                "q1",
                "q",
                date_from=date(2026, 9, 1),
                date_to=date(2026, 9, 1),
            ),
            mode=SearchMode.LIGHT,
            max_results=4,
            timeout_seconds=8,
        )
        self.assertTrue(result.date_filter_normalized)
        kwargs = client.search.call_args.kwargs
        self.assertNotIn("start_date", kwargs)
        self.assertNotIn("end_date", kwargs)

    def test_tavily_readiness(self):
        unconfigured = TavilySearchProvider(api_key="", proxy_url="")
        self.assertEqual(
            ProviderReadiness("tavily", configured=False, available=False),
            unconfigured.readiness(),
        )
        configured = TavilySearchProvider(api_key="key", proxy_url="")
        self.assertEqual(
            ProviderReadiness("tavily", configured=True, available=True),
            configured.readiness(),
        )

    @patch("src.search.simple.ddgs.DDGS")
    def test_ddgs_uses_smaller_configured_or_call_timeout(self, ddgs):
        ddgs.return_value.__enter__.return_value.text.return_value = []
        provider = DDGSSearchProvider(proxy_url="", timeout_seconds=15)
        provider.search(
            SearchQuery("q1", "q"),
            mode=SearchMode.LIGHT,
            max_results=4,
            timeout_seconds=6,
        )
        ddgs.assert_called_once_with(proxy=None, timeout=6)

    def test_ddgs_readiness(self):
        provider = DDGSSearchProvider(proxy_url="", timeout_seconds=15)
        readiness = provider.readiness()
        self.assertEqual("ddgs", readiness.provider)
        self.assertTrue(readiness.configured)
        self.assertTrue(readiness.available)

    def test_new_provider_modules_never_depend_on_legacy_search(self):
        for path in (
            Path("src/search/simple/tavily.py"),
            Path("src/search/simple/ddgs.py"),
        ):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("src.search." + "models", source)
            self.assertNotIn("src.search." + "providers", source)


if __name__ == "__main__":
    unittest.main()
