"""Provider adapter tests: neutral Tavily/DDGS with availability fallback."""

from __future__ import annotations

import importlib
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Any
from unittest import mock

from src.search.models import (
    ProviderHit,
    ProviderReadiness,
    ProviderResult,
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


def _provider_result(provider="tavily", *, status=ProviderStatus.SUCCESS, query_id="q1", latency_ms=0):
    hits = ()
    if status is ProviderStatus.SUCCESS:
        hits = (
            ProviderHit(
                provider,
                query_id,
                "result",
                f"https://example.com/{provider}/{query_id}",
                "result",
                None,
                None,
                None,
                (),
            ),
        )
    return ProviderResult(provider, status, hits, latency_ms)

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

    def test_each_call_uses_the_smaller_remaining_timeout(self):
        client = FakeDDGSClient(results=[_ddgs_hit()])
        constructor_kwargs = []
        patcher = mock.patch.object(
            self.module.ddgs,
            "DDGS",
            lambda **kwargs: constructor_kwargs.append(kwargs) or client,
        )
        if self._stack is not None:
            self._stack.stop()
        patcher.start()
        self._stack = patcher
        adapter = self.module.ddgs.DDGSSearchProvider(
            proxy_url="http://proxy:8080",
            timeout_seconds=18.0,
        )

        adapter.search(
            query(),
            tier=SearchTier.STANDARD,
            max_results=8,
            timeout_seconds=0.25,
        )

        self.assertEqual(constructor_kwargs[0]["timeout"], 0.25)


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
        tavily.search.return_value = _provider_result(latency_ms=10)
        registry = self._registry(tavily, None)
        result = registry.search(query(), tier=SearchTier.STANDARD, max_results=8, timeout_seconds=20.0)
        self.assertEqual(result.status, ProviderStatus.SUCCESS)
        self.assertTrue(tavily.search.called)

    def test_public_search_preserves_provider_result_contract(self):
        tavily = mock.Mock()
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        tavily.search.return_value = _provider_result(latency_ms=10)

        result = self._registry(tavily, None).search(
            query(), tier=SearchTier.STANDARD, max_results=8, timeout_seconds=1.0
        )

        self.assertIsInstance(result, ProviderResult)
        self.assertEqual(
            set(result.__dataclass_fields__),
            {"provider", "status", "hits", "latency_ms"},
        )

    def test_noncooperative_adapter_is_enforced_by_registry_deadline(self):
        class SlowProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                time.sleep(0.2)
                return _provider_result(query_id=search_query.query_id)

        registry = self._registry(SlowProvider(), None)
        search_with_attempts = getattr(registry, "search_with_attempts", registry.search)
        started = time.monotonic()

        outcome = search_with_attempts(
            query(), tier=SearchTier.LIGHT, max_results=1, timeout_seconds=0.05
        )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.15)
        self.assertEqual(outcome.status, ProviderStatus.TIMEOUT)
        self.assertEqual(len(outcome.attempts), 1)
        self.assertEqual(outcome.attempts[0].provider, "tavily")
        self.assertEqual(outcome.attempts[0].status, ProviderStatus.TIMEOUT)
        self.assertTrue(outcome.attempts[0].invocation_started)
        self.assertGreaterEqual(outcome.attempts[0].latency_ms, 40)
        self.assertLess(outcome.attempts[0].latency_ms, 150)

    def test_ready_calls_queued_past_deadline_are_timeouts_without_attempts(self):
        release_workers = threading.Event()
        all_workers_started = threading.Event()
        worker_lock = threading.Lock()
        started_workers = 0

        def occupy_worker():
            nonlocal started_workers
            with worker_lock:
                started_workers += 1
                if started_workers == 8:
                    all_workers_started.set()
            release_workers.wait(timeout=2.0)

        blockers = [self.base._ADAPTER_EXECUTOR.submit(occupy_worker) for _ in range(8)]
        self.assertTrue(all_workers_started.wait(timeout=1.0))

        provider = mock.Mock()
        provider.name = "tavily"
        provider.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        provider.search.side_effect = AssertionError("queued provider must never start")
        registry = self._registry(provider, None)
        started_callbacks = []
        finished_callbacks = []

        def queued_call(index):
            queued_query = SearchQuery(
                f"queued-{index}",
                SearchRoundKind.INITIAL,
                QueryPurpose.DIRECT,
                f"queued query {index}",
            )
            return registry.search_with_attempts(
                queued_query,
                tier=SearchTier.LIGHT,
                max_results=1,
                timeout_seconds=0.05,
                on_attempt_started=lambda *args: started_callbacks.append(args),
                on_attempt_finished=lambda attempt: finished_callbacks.append(attempt),
            )

        try:
            with ThreadPoolExecutor(max_workers=4) as caller_pool:
                outcomes = list(caller_pool.map(queued_call, range(4)))
        finally:
            release_workers.set()
            for blocker in blockers:
                blocker.result(timeout=1.0)

        self.assertEqual([outcome.status for outcome in outcomes], [ProviderStatus.TIMEOUT] * 4)
        self.assertEqual([outcome.attempts for outcome in outcomes], [()] * 4)
        self.assertFalse(provider.search.called)
        self.assertEqual(started_callbacks, [])
        self.assertEqual(finished_callbacks, [])

    def test_fallback_timeout_preserves_completed_primary_and_real_fallback_attempt(self):
        class Primary:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, _query, **_kwargs):
                time.sleep(0.01)
                return _provider_result(status=ProviderStatus.ERROR)

        class SlowFallback:
            name = "ddgs"

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, search_query, **_kwargs):
                time.sleep(0.2)
                return _provider_result("ddgs", query_id=search_query.query_id)

        registry = self._registry(Primary(), SlowFallback())
        search_with_attempts = getattr(registry, "search_with_attempts", registry.search)
        started = time.monotonic()

        outcome = search_with_attempts(
            query(), tier=SearchTier.LIGHT, max_results=1, timeout_seconds=0.06
        )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.16)
        self.assertEqual(outcome.status, ProviderStatus.TIMEOUT)
        self.assertEqual(
            [(attempt.provider, attempt.status) for attempt in outcome.attempts],
            [
                ("tavily", ProviderStatus.ERROR),
                ("ddgs", ProviderStatus.TIMEOUT),
            ],
        )
        self.assertGreaterEqual(outcome.attempts[0].latency_ms, 5)
        self.assertGreaterEqual(outcome.attempts[1].latency_ms, 35)
        self.assertLess(outcome.attempts[1].latency_ms, 150)

    def test_no_usable_tavily_falls_back_to_ddgs(self):
        # An available-but-erroring primary triggers the DDGS availability fallback.
        tavily = mock.Mock()
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        tavily.search.return_value = _provider_result(status=ProviderStatus.ERROR, latency_ms=5)
        ddgs = mock.Mock()
        ddgs.name = "ddgs"
        ddgs.readiness.return_value = ProviderReadiness("ddgs", True, True, None)
        ddgs.search.return_value = _provider_result("ddgs", latency_ms=3)
        registry = self._registry(tavily, ddgs)
        result = registry.search_with_attempts(
            query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=8.0
        )
        self.assertEqual(result.status, ProviderStatus.SUCCESS)
        self.assertTrue(ddgs.search.called)
        # A fallback call is a separate ProviderAttempt but one semantic query.
        self.assertIsInstance(result.attempts, tuple)
        self.assertEqual(
            [(attempt.provider, attempt.query_id) for attempt in result.attempts],
            [("tavily", "q1"), ("ddgs", "q1")],
        )

    def test_unavailable_primary_is_skipped_without_invocation(self):
        tavily = mock.Mock()
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, False, SearchFailureCode.PROVIDER_UNAVAILABLE)
        tavily.search = mock.Mock(side_effect=AssertionError("must not be invoked"))
        ddgs = mock.Mock()
        ddgs.name = "ddgs"
        ddgs.readiness.return_value = ProviderReadiness("ddgs", True, True, None)
        ddgs.search.return_value = _provider_result("ddgs", latency_ms=3)
        registry = self._registry(tavily, ddgs)
        result = registry.search_with_attempts(
            query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=8.0
        )
        self.assertEqual(result.status, ProviderStatus.SUCCESS)
        self.assertIsInstance(result.attempts, tuple)
        self.assertEqual(len(result.attempts), 1)
        self.assertFalse(tavily.search.called)

    def test_fallback_same_query_id_is_one_semantic_query(self):
        tavily = mock.Mock()
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        tavily.search.return_value = _provider_result(status=ProviderStatus.ERROR, latency_ms=5)
        ddgs = mock.Mock()
        ddgs.name = "ddgs"
        ddgs.readiness.return_value = ProviderReadiness("ddgs", True, True, None)
        ddgs.search.return_value = _provider_result("ddgs", latency_ms=3)
        registry = self._registry(tavily, ddgs)
        result = registry.search_with_attempts(
            query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=8.0
        )
        self.assertIsInstance(result.attempts, tuple)
        self.assertEqual(len(result.attempts), 2)

    def test_concurrent_calls_return_immutable_attempts_for_their_own_query(self):
        class PerQueryProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                time.sleep(0.03 if search_query.query_id == "qa" else 0.01)
                return self_result(search_query.query_id)

        def self_result(query_id):
            from src.search.models import ProviderHit, ProviderResult

            provider_hit = ProviderHit(
                "tavily",
                query_id,
                query_id,
                f"https://example.com/{query_id}",
                "result",
                None,
                None,
                None,
                (),
            )
            return ProviderResult("tavily", ProviderStatus.SUCCESS, (provider_hit,), 0)

        registry = self._registry(PerQueryProvider(), None)
        qa = SearchQuery("qa", SearchRoundKind.INITIAL, QueryPurpose.DIRECT, "A")
        qb = SearchQuery("qb", SearchRoundKind.INITIAL, QueryPurpose.DIRECT, "B")
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda item: registry.search_with_attempts(
                        item,
                        tier=SearchTier.LIGHT,
                        max_results=1,
                        timeout_seconds=1.0,
                    ),
                    (qa, qb),
                )
            )

        self.assertTrue(hasattr(results[0], "attempts"))
        self.assertTrue(hasattr(results[1], "attempts"))
        self.assertEqual(results[0].attempts[0].query_id, "qa")
        self.assertEqual(results[1].attempts[0].query_id, "qb")
        self.assertIsInstance(results[0].attempts, tuple)
        self.assertIsInstance(results[1].attempts, tuple)

    def test_attempts_keep_each_adapter_latency_and_reduced_fallback_time(self):
        observed_timeouts = []

        def primary_search(_query, **kwargs):
            observed_timeouts.append(("tavily", kwargs["timeout_seconds"]))
            time.sleep(0.03)
            from src.search.models import ProviderResult
            return ProviderResult("tavily", ProviderStatus.ERROR, (), 0)

        def fallback_search(search_query, **kwargs):
            observed_timeouts.append(("ddgs", kwargs["timeout_seconds"]))
            time.sleep(0.01)
            from src.search.models import ProviderHit, ProviderResult
            provider_hit = ProviderHit(
                "ddgs", search_query.query_id, "ok", "https://example.com/ok",
                "ok", None, None, None, ("availability_fallback",),
            )
            return ProviderResult("ddgs", ProviderStatus.SUCCESS, (provider_hit,), 0)

        tavily = mock.Mock(name="tavily")
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        tavily.search.side_effect = primary_search
        ddgs = mock.Mock(name="ddgs")
        ddgs.name = "ddgs"
        ddgs.readiness.return_value = ProviderReadiness("ddgs", True, True, None)
        ddgs.search.side_effect = fallback_search

        result = self._registry(tavily, ddgs).search_with_attempts(
            query(), tier=SearchTier.LIGHT, max_results=1, timeout_seconds=0.2
        )

        self.assertIsInstance(result.attempts, tuple)
        self.assertEqual([attempt.provider for attempt in result.attempts], ["tavily", "ddgs"])
        self.assertGreaterEqual(result.attempts[0].latency_ms, 20)
        self.assertGreaterEqual(result.attempts[1].latency_ms, 5)
        self.assertLess(observed_timeouts[1][1], observed_timeouts[0][1])
        self.assertTrue(all(attempt.invocation_started for attempt in result.attempts))

    def test_attempt_retains_invocation_start_readiness_without_completion_reprobe(self):
        class FlippingReadinessProvider:
            name = "tavily"

            def __init__(self):
                self.readiness_calls = 0

            def readiness(self):
                self.readiness_calls += 1
                if self.readiness_calls <= 2:
                    return ProviderReadiness("tavily", True, True, None)
                return ProviderReadiness(
                    "tavily",
                    True,
                    False,
                    SearchFailureCode.PROVIDER_UNAVAILABLE,
                )

            def search(self, search_query, **_kwargs):
                return _provider_result(query_id=search_query.query_id)

        provider = FlippingReadinessProvider()
        outcome = self._registry(provider, None).search_with_attempts(
            query(), tier=SearchTier.LIGHT, max_results=1, timeout_seconds=1.0
        )

        self.assertEqual(provider.readiness_calls, 2)
        self.assertEqual(len(outcome.attempts), 1)
        self.assertTrue(outcome.attempts[0].configured)
        self.assertTrue(outcome.attempts[0].available)
        self.assertTrue(outcome.attempts[0].invocation_started)

    def test_no_fallback_after_usable_tavily_hits(self):
        tavily = mock.Mock()
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        tavily.search.return_value = _provider_result(latency_ms=5)
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
