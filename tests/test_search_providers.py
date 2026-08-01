"""Provider adapter tests: neutral Tavily/DDGS with availability fallback."""

from __future__ import annotations

import importlib
import unittest
from datetime import date, datetime, timezone
from typing import Any
from unittest import mock

from src.search.models import (
    ProviderReadiness,
    ProviderStatus,
    QueryPurpose,
    SearchQuery,
    SearchRoundKind,
    SearchTier,
    SearchFailureCode,
)


def providers_module():
    try:
        return importlib.import_module("src.search.providers")
    except ModuleNotFoundError:
        raise AssertionError("src.search.providers must exist") from None


def base_module():
    try:
        return importlib.import_module("src.search.providers.base")
    except ModuleNotFoundError:
        raise AssertionError("src.search.providers.base must exist") from None


def query(purpose=QueryPurpose.DIRECT, text="什么是光合作用", **kwargs):
    return SearchQuery(
        "q1", SearchRoundKind.INITIAL, purpose, text, **kwargs
    )


class FakeTavilyClient:
    """Records constructor kwargs and search calls."""

    def __init__(self, api_key=None, proxies=None, **kwargs):
        self.api_key = api_key
        self.proxies = proxies
        self.kwargs = kwargs
        self.calls = []
        self.response = {"results": [_tavily_hit()]}

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class FakeDDGSClient:
    def __init__(self, results=None, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.results = list(results or [])

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def text(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.results


def _tavily_hit(**overrides):
    item = {
        "title": "光合作用 - 百度百科",
        "url": "https://baike.example.com/guanghezuoyong",
        "content": "光合作用是绿色植物利用光能将二氧化碳和水转化为有机物并释放氧气的过程。",
        "score": 0.92,
        "published_date": "2024-05-01",
        "raw_content": "植物利用光能……",
    }
    item.update(overrides)
    return item


def _ddgs_hit(**overrides):
    item = {
        "title": "光合作用 - 百度百科",
        "href": "https://baike.example.com/guanghezuoyong",
        "body": "光合作用是绿色植物利用光能……",
    }
    item.update(overrides)
    return item

class TavilyAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = providers_module()
        self._stack = mock.patch.object(self.module.tavily, "TavilyClient", lambda **kw: FakeTavilyClient())

    def _adapter(self, client=None):
        self._stack.stop()
        patcher = mock.patch.object(
            self.module.tavily,
            "TavilyClient",
            lambda **kw: client if client is not None else FakeTavilyClient(),
        )
        patcher.start()
        self._stack = patcher
        return self.module.tavily.TavilySearchProvider(api_key="sk-test", proxy_url="http://proxy:8080")

    def _unavailable_adapter(self):
        self._stack.stop()
        patcher = mock.patch.object(self.module.tavily, "TavilyClient", None)
        patcher.start()
        self._stack = patcher
        return self.module.tavily.TavilySearchProvider(api_key="sk-test", proxy_url="")

    def tearDown(self) -> None:
        self._stack.stop()

    def test_basic_depth_for_light(self):
        client = FakeTavilyClient()
        adapter = self._adapter(client)
        result = adapter.search(
            query(purpose=QueryPurpose.DIRECT),
            tier=SearchTier.LIGHT,
            max_results=5,
            timeout_seconds=8.0,
        )
        self.assertEqual(result.status, ProviderStatus.SUCCESS)
        self.assertEqual(client.calls[0][1]["search_depth"], "basic")
        self.assertFalse(client.calls[0][1]["include_raw_content"])

    def test_advanced_depth_and_raw_content_for_standard(self):
        client = FakeTavilyClient()
        adapter = self._adapter(client)
        result = adapter.search(
            query(),
            tier=SearchTier.STANDARD,
            max_results=8,
            timeout_seconds=20.0,
        )
        self.assertEqual(client.calls[0][1]["search_depth"], "advanced")
        self.assertEqual(client.calls[0][1]["include_raw_content"], True)

    def test_deep_uses_advanced_and_raw_content(self):
        client = FakeTavilyClient()
        adapter = self._adapter(client)
        adapter.search(query(), tier=SearchTier.DEEP, max_results=15, timeout_seconds=40.0)
        self.assertEqual(client.calls[0][1]["search_depth"], "advanced")
        self.assertTrue(client.calls[0][1]["include_raw_content"])

    def test_maps_date_domains_and_topic(self):
        client = FakeTavilyClient()
        adapter = self._adapter(client)
        from_today = date(2026, 7, 29)
        adapter.search(
            query(
                purpose=QueryPurpose.TIME_BOUNDED,
                date_from=from_today,
                date_to=from_today,
                include_domains=("example.com",),
                exclude_domains=("badexample.net",),
            ),
            tier=SearchTier.STANDARD,
            max_results=8,
            timeout_seconds=20.0,
        )
        kwargs = client.calls[0][1]
        self.assertEqual(kwargs["start_date"], "2026-07-29")
        self.assertEqual(kwargs["end_date"], "2026-07-29")
        self.assertEqual(kwargs["include_domains"], ("example.com",))
        self.assertEqual(kwargs["exclude_domains"], ("badexample.net",))

    def test_news_topic_for_time_bounded_purpose(self):
        client = FakeTavilyClient()
        adapter = self._adapter(client)
        adapter.search(
            query(purpose=QueryPurpose.TIME_BOUNDED, date_from=date(2026, 7, 29), date_to=date(2026, 7, 29)),
            tier=SearchTier.STANDARD,
            max_results=8,
            timeout_seconds=20.0,
        )
        self.assertEqual(client.calls[0][1]["topic"], "news")

    def test_preserves_provider_metadata_without_evidence_authority(self):
        client = FakeTavilyClient()
        client.response = {"results": [_tavily_hit()]}
        adapter = self._adapter(client)
        result = adapter.search(query(), tier=SearchTier.STANDARD, max_results=8, timeout_seconds=20.0)
        hit = result.hits[0]
        self.assertEqual(hit.provider, "tavily")
        self.assertEqual(hit.title, "光合作用 - 百度百科")
        self.assertEqual(hit.score, 0.92)
        self.assertEqual(hit.raw_content, "植物利用光能……")
        self.assertIsNotNone(hit.published_at)
        # The provider layer must never assign relevance/source-relation/Evidence IDs.
        self.assertNotIn("relevance", hit.quality_flags)
        self.assertFalse(hasattr(hit, "source_relation"))
        self.assertFalse(hasattr(hit, "evidence_id"))

    def test_missing_key_is_not_configured(self):
        tavily_mod = self.module.tavily
        adapter = tavily_mod.TavilySearchProvider(api_key="", proxy_url="")
        readiness = adapter.readiness()
        self.assertFalse(readiness.configured)
        self.assertFalse(readiness.available)
        self.assertIs(readiness.reason_code, SearchFailureCode.PROVIDER_NOT_CONFIGURED)

    def test_unavailable_sdk_is_unavailable(self):
        adapter = self._unavailable_adapter()
        readiness = adapter.readiness()
        self.assertFalse(readiness.available)
        self.assertIs(readiness.reason_code, SearchFailureCode.PROVIDER_UNAVAILABLE)

    def test_exception_maps_to_error_without_message(self):
        client = FakeTavilyClient()
        client.search = mock.Mock(side_effect=RuntimeError("secret api detail"))
        adapter = self._adapter(client)
        result = adapter.search(query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=8.0)
        self.assertEqual(result.status, ProviderStatus.ERROR)
        self.assertEqual(result.hits, ())
        # No exception message leaks into the result.
        self.assertNotIn("secret", repr(result))

    def test_timeout_maps_to_timeout(self):
        client = FakeTavilyClient()
        client.search = mock.Mock(side_effect=TimeoutError("slow"))
        adapter = self._adapter(client)
        result = adapter.search(query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=8.0)
        self.assertEqual(result.status, ProviderStatus.TIMEOUT)


class DDGSAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = providers_module()
        self._stack = None

    def _adapter(self, client):
        if self._stack is not None:
            self._stack.stop()
        patcher = mock.patch.object(self.module.ddgs, "DDGS", lambda **kw: client)
        patcher.start()
        self._stack = patcher
        return self.module.ddgs.DDGSSearchProvider(proxy_url="http://proxy:8080", timeout_seconds=18.0)

    def tearDown(self) -> None:
        if self._stack is not None:
            self._stack.stop()

    def test_normalizes_fields_without_score_or_raw(self):
        client = FakeDDGSClient(results=[_ddgs_hit()])
        adapter = self._adapter(client)
        result = adapter.search(query(), tier=SearchTier.STANDARD, max_results=8, timeout_seconds=20.0)
        self.assertEqual(result.status, ProviderStatus.SUCCESS)
        hit = result.hits[0]
        self.assertEqual(hit.provider, "ddgs")
        self.assertEqual(hit.title, "光合作用 - 百度百科")
        self.assertEqual(hit.url, "https://baike.example.com/guanghezuoyong")
        self.assertIsNone(hit.score)
        self.assertIsNone(hit.raw_content)
        # DDGS hits are weak until relevance/content support is established.
        self.assertIn("availability_fallback", hit.quality_flags)

    def test_keeps_date_when_present(self):
        client = FakeDDGSClient(results=[_ddgs_hit(date="2026-07-28")])
        adapter = self._adapter(client)
        result = adapter.search(query(), tier=SearchTier.STANDARD, max_results=8, timeout_seconds=20.0)
        self.assertEqual(result.status, ProviderStatus.SUCCESS)
        self.assertIsNotNone(result.hits[0].published_at)

    def test_exception_maps_to_error(self):
        client = FakeDDGSClient()
        client.text = mock.Mock(side_effect=RuntimeError("boom"))
        adapter = self._adapter(client)
        result = adapter.search(query(), tier=SearchTier.STANDARD, max_results=8, timeout_seconds=20.0)
        self.assertEqual(result.status, ProviderStatus.ERROR)


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = providers_module()
        self.base = base_module()

    def _registry(self, tavily=None, ddgs=None):
        return self.base.ProviderRegistry(
            [tavily, ddgs] if tavily is not None and ddgs is not None else [item for item in (tavily, ddgs) if item is not None]
        )

    def test_prefers_tavily_when_configured(self):
        tavily = mock.Mock()
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        ddgs = mock.Mock()
        ddgs.name = "ddgs"
        ddgs.readiness.return_value = ProviderReadiness("ddgs", True, True, None)
        registry = self._registry(tavily, ddgs)
        self.assertEqual(registry.readiness()[0].provider, "tavily")
        self.assertIn("tavily", registry.available_providers())
        self.assertIn("ddgs", registry.available_providers())

    def test_primary_search_uses_tavily(self):
        tavily = mock.Mock()
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        tavily.search.return_value = mock.Mock(
            status=ProviderStatus.SUCCESS, hits=(mock.Mock(),), latency_ms=10
        )
        registry = self._registry(tavily, None)
        result = registry.search(query(), tier=SearchTier.STANDARD, max_results=8, timeout_seconds=20.0)
        self.assertEqual(result.status, ProviderStatus.SUCCESS)
        self.assertTrue(tavily.search.called)

    def test_no_usable_tavily_falls_back_to_ddgs(self):
        # An available-but-erroring primary triggers the DDGS availability fallback.
        tavily = mock.Mock()
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        tavily.search.return_value = mock.Mock(status=ProviderStatus.ERROR, hits=(), latency_ms=5)
        ddgs = mock.Mock()
        ddgs.name = "ddgs"
        ddgs.readiness.return_value = ProviderReadiness("ddgs", True, True, None)
        ddgs.search.return_value = mock.Mock(status=ProviderStatus.SUCCESS, hits=(mock.Mock(),), latency_ms=3)
        registry = self._registry(tavily, ddgs)
        result = registry.search(query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=8.0)
        self.assertEqual(result.status, ProviderStatus.SUCCESS)
        self.assertTrue(ddgs.search.called)
        # A fallback call is a separate ProviderAttempt but one semantic query.

    def test_unavailable_primary_is_skipped_without_invocation(self):
        tavily = mock.Mock()
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, False, SearchFailureCode.PROVIDER_UNAVAILABLE)
        tavily.search = mock.Mock(side_effect=AssertionError("must not be invoked"))
        ddgs = mock.Mock()
        ddgs.name = "ddgs"
        ddgs.readiness.return_value = ProviderReadiness("ddgs", True, True, None)
        ddgs.search.return_value = mock.Mock(status=ProviderStatus.SUCCESS, hits=(mock.Mock(),), latency_ms=3)
        registry = self._registry(tavily, ddgs)
        result = registry.search(query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=8.0)
        self.assertEqual(result.status, ProviderStatus.SUCCESS)
        self.assertEqual(len(registry.last_attempts), 1)
        self.assertFalse(tavily.search.called)

    def test_fallback_same_query_id_is_one_semantic_query(self):
        tavily = mock.Mock()
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        tavily.search.return_value = mock.Mock(status=ProviderStatus.ERROR, hits=(), latency_ms=5)
        ddgs = mock.Mock()
        ddgs.name = "ddgs"
        ddgs.readiness.return_value = ProviderReadiness("ddgs", True, True, None)
        ddgs.search.return_value = mock.Mock(status=ProviderStatus.SUCCESS, hits=(mock.Mock(),), latency_ms=3)
        registry = self._registry(tavily, ddgs)
        registry.search(query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=8.0)
        self.assertEqual(len(registry.last_attempts), 2)

    def test_no_fallback_after_usable_tavily_hits(self):
        tavily = mock.Mock()
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        tavily.search.return_value = mock.Mock(status=ProviderStatus.SUCCESS, hits=(mock.Mock(),), latency_ms=5)
        ddgs = mock.Mock()
        ddgs.name = "ddgs"
        registry = self._registry(tavily, ddgs)
        registry.search(query(), tier=SearchTier.STANDARD, max_results=8, timeout_seconds=20.0)
        self.assertFalse(ddgs.search.called)

    def test_no_provider_is_not_configured(self):
        registry = self._registry(None, None)
        readiness = registry.readiness()
        self.assertEqual(readiness, ())
        self.assertFalse(registry.available_providers())

    def test_no_provider_result_is_not_configured(self):
        registry = self._registry(None, None)
        result = registry.search(query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=8.0)
        self.assertEqual(result.status, ProviderStatus.NOT_CONFIGURED)
        self.assertEqual(result.hits, ())


class ProviderPrivacyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = providers_module()
        self._stack = None

    def _adapter(self, client):
        if self._stack is not None:
            self._stack.stop()
        patcher = mock.patch.object(self.module.tavily, "TavilyClient", lambda **kw: client)
        patcher.start()
        self._stack = patcher
        return self.module.tavily.TavilySearchProvider(api_key="sk-test", proxy_url="")

    def tearDown(self) -> None:
        if self._stack is not None:
            self._stack.stop()

    def test_docs_url_stays_relation_unknown_at_this_layer(self):
        client = FakeTavilyClient()
        client.response = {"results": [_tavily_hit(url="https://unrelated.example/docs/guide")]}
        adapter = self._adapter(client)
        result = adapter.search(query(), tier=SearchTier.STANDARD, max_results=8, timeout_seconds=20.0)
        hit = result.hits[0]
        # The provider layer preserves the URL but assigns no relation/Evidence authority.
        self.assertEqual(hit.url, "https://unrelated.example/docs/guide")
        self.assertFalse(hasattr(hit, "source_relation"))
        self.assertFalse(hasattr(hit, "evidence_id"))
