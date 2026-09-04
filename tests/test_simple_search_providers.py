from pathlib import Path
import unittest

from src.search.simple.models import SearchMode, SearchQuery
from src.search.simple.providers import (
    ProviderErrorCode,
    ProviderHit,
    ProviderReadiness,
    ProviderResult,
    ProviderStatus,
    SearchProvider,
)


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
