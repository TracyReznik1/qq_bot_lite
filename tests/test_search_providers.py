"""Provider adapter tests: neutral Tavily/DDGS with availability fallback."""

from __future__ import annotations

import importlib
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Any
from unittest import mock

from tavily.errors import BadRequestError

from src.search.models import (
    ProviderErrorCode,
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


class RunningBeforeInvokeExecutor:
    """Marks a Future RUNNING, then pauses before calling its wrapper."""

    def __init__(self):
        self.running = threading.Event()
        self.release = threading.Event()
        self.future = None
        self.thread = None

    def submit(self, operation):
        future = Future()

        def run():
            if not future.set_running_or_notify_cancel():
                return
            self.running.set()
            self.release.wait(timeout=2.0)
            try:
                result = operation()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        thread = threading.Thread(target=run, daemon=True)
        self.future = future
        self.thread = thread
        thread.start()
        if not self.running.wait(timeout=1.0):
            raise RuntimeError("test executor did not enter RUNNING state")
        return future

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
        adapter.search(query(), tier=SearchTier.STANDARD, max_results=15, timeout_seconds=40.0)
        self.assertEqual(client.calls[0][1]["search_depth"], "advanced")
        self.assertTrue(client.calls[0][1]["include_raw_content"])

    def test_equal_date_bounds_are_omitted_before_tavily_call(self):
        client = FakeTavilyClient()
        adapter = self._adapter(client)
        from_today = date(2026, 7, 29)
        result = adapter.search(
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
        self.assertNotIn("start_date", kwargs)
        self.assertNotIn("end_date", kwargs)
        self.assertEqual(kwargs["include_domains"], ("example.com",))
        self.assertEqual(kwargs["exclude_domains"], ("badexample.net",))
        self.assertTrue(result.date_filter_normalized)
        self.assertFalse(result.parameter_retry_attempted)

    def test_bad_request_with_dates_retries_once_without_dates(self):
        client = FakeTavilyClient()
        client.search = mock.Mock(side_effect=[
            BadRequestError("start_date and end_date invalid"),
            {"results": [_tavily_hit()]},
        ])
        adapter = self._adapter(client)

        result = adapter.search(
            query(date_from=date(2026, 1, 1), date_to=date(2026, 9, 3)),
            tier=SearchTier.LIGHT,
            max_results=4,
            timeout_seconds=8.0,
        )

        self.assertEqual(2, client.search.call_count)
        self.assertIn("start_date", client.search.call_args_list[0].kwargs)
        self.assertNotIn("start_date", client.search.call_args_list[1].kwargs)
        self.assertIs(result.status, ProviderStatus.SUCCESS)
        self.assertIs(result.error_code, ProviderErrorCode.INVALID_PARAMETERS)
        self.assertTrue(result.parameter_retry_attempted)

    def test_second_bad_request_returns_closed_parameter_error(self):
        client = FakeTavilyClient()
        client.search = mock.Mock(side_effect=[
            BadRequestError("bad dates"),
            BadRequestError("still bad"),
        ])
        adapter = self._adapter(client)

        result = adapter.search(
            query(date_from=date(2026, 1, 1), date_to=date(2026, 9, 3)),
            tier=SearchTier.LIGHT,
            max_results=4,
            timeout_seconds=8.0,
        )

        self.assertEqual(2, client.search.call_count)
        self.assertIs(result.status, ProviderStatus.ERROR)
        self.assertIs(result.error_code, ProviderErrorCode.INVALID_PARAMETERS)
        self.assertTrue(result.date_filter_normalized)
        self.assertTrue(result.parameter_retry_attempted)

    def test_parameter_retry_receives_only_remaining_timeout(self):
        client = FakeTavilyClient()
        client.search = mock.Mock(side_effect=[
            BadRequestError("bad dates"),
            {"results": [_tavily_hit()]},
        ])
        adapter = self._adapter(client)

        with mock.patch.object(self.module.tavily.time, "monotonic", side_effect=[10.0, 12.5, 12.5]):
            result = adapter.search(
                query(date_from=date(2026, 1, 1), date_to=date(2026, 9, 3)),
                tier=SearchTier.LIGHT,
                max_results=4,
                timeout_seconds=8.0,
            )

        self.assertIs(result.status, ProviderStatus.SUCCESS)
        self.assertAlmostEqual(5.5, client.search.call_args_list[1].kwargs["timeout"])

    def test_non_parameter_error_does_not_retry(self):
        client = FakeTavilyClient()
        client.search = mock.Mock(side_effect=RuntimeError("connection failed"))
        adapter = self._adapter(client)

        result = adapter.search(
            query(date_from=date(2026, 1, 1), date_to=date(2026, 9, 3)),
            tier=SearchTier.LIGHT,
            max_results=4,
            timeout_seconds=8.0,
        )

        self.assertEqual(1, client.search.call_count)
        self.assertIs(result.status, ProviderStatus.ERROR)

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
        self.assertNotIn("availability_fallback", hit.quality_flags)
        self.assertEqual(hit.quality_flags, ())

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

    def _ready_provider(self, name, *, status=ProviderStatus.SUCCESS):
        provider = mock.Mock()
        provider.name = name
        provider.readiness.return_value = ProviderReadiness(name, True, True, None)
        provider.search.return_value = _provider_result(
            name, status=status, latency_ms=3,
        )
        return provider

    def _registry(self, tavily=None, ddgs=None):
        return self.base.ProviderRegistry(
            [item for item in (ddgs, tavily) if item is not None]
        )

    def test_registry_exposes_ddgs_before_tavily(self):
        tavily = self._ready_provider("tavily")
        ddgs = self._ready_provider("ddgs")
        registry = self._registry(tavily, ddgs)
        self.assertEqual(registry.readiness()[0].provider, "ddgs")
        self.assertIn("tavily", registry.available_providers())
        self.assertIn("ddgs", registry.available_providers())

    def test_noncooperative_adapter_is_enforced_by_registry_deadline(self):
        class SlowProvider:
            name = "ddgs"

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, search_query, **_kwargs):
                time.sleep(0.2)
                return _provider_result(query_id=search_query.query_id)

        registry = self._registry(None, SlowProvider())
        started = time.monotonic()

        outcome = registry.search_provider_with_attempts(
            "ddgs", query(), tier=SearchTier.LIGHT, max_results=1, timeout_seconds=0.05
        )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.15)
        self.assertEqual(outcome.status, ProviderStatus.TIMEOUT)
        self.assertEqual(len(outcome.attempts), 1)
        self.assertEqual(outcome.attempts[0].provider, "ddgs")
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
        provider.name = "ddgs"
        provider.readiness.return_value = ProviderReadiness("ddgs", True, True, None)
        provider.search.side_effect = AssertionError("queued provider must never start")
        registry = self._registry(None, provider)
        started_callbacks = []
        finished_callbacks = []

        def queued_call(index):
            queued_query = SearchQuery(
                f"queued-{index}",
                SearchRoundKind.INITIAL,
                QueryPurpose.DIRECT,
                f"queued query {index}",
            )
            return registry.search_provider_with_attempts(
                "ddgs",
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

    def test_running_wrapper_timeout_seal_prevents_late_internal_invocation(self):
        result = self._running_before_invoke_timeout()

        self.assertEqual(result["outcome"].status, ProviderStatus.TIMEOUT)
        self.assertEqual(result["outcome"].attempts, ())
        self.assertLess(result["elapsed"], 0.15)
        self.assertEqual(result["before_release"], ((), (), ()))
        self.assertEqual(result["after_release"], result["before_release"])

    def test_invocation_start_winner_is_recorded_as_real_timeout_attempt(self):
        result = self._invocation_start_wins_timeout()

        self.assertEqual(result["outcome"].status, ProviderStatus.TIMEOUT)
        self.assertEqual(
            [(attempt.provider, attempt.status) for attempt in result["outcome"].attempts],
            [("ddgs", ProviderStatus.TIMEOUT)],
        )
        self.assertLess(result["elapsed"], 0.15)
        self.assertEqual(result["before_release"], (("q1",), ("ddgs",), ("ddgs",)))
        self.assertEqual(result["after_release"], result["before_release"])

    def test_invocation_gate_stress_preserves_both_race_winners(self):
        for iteration in range(20):
            with self.subTest(iteration=iteration, winner="timeout"):
                timeout_result = self._running_before_invoke_timeout(
                    timeout_seconds=0.005,
                )
                self.assertEqual(timeout_result["outcome"].attempts, ())
                self.assertEqual(
                    timeout_result["after_release"],
                    timeout_result["before_release"],
                )

            with self.subTest(iteration=iteration, winner="start"):
                start_result = self._invocation_start_wins_timeout(
                    timeout_seconds=0.005,
                )
                self.assertEqual(len(start_result["outcome"].attempts), 1)
                self.assertEqual(
                    start_result["after_release"],
                    start_result["before_release"],
                )

    def _running_before_invoke_timeout(self, *, timeout_seconds=0.05):
        executor = RunningBeforeInvokeExecutor()

        class RecordingProvider:
            name = "ddgs"

            def __init__(self):
                self.calls = []

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, search_query, **_kwargs):
                self.calls.append(search_query.query_id)
                return _provider_result(query_id=search_query.query_id)

        provider = RecordingProvider()
        registry = self._registry(None, provider)
        actual_remaining = registry._remaining

        def controlled_remaining(deadline):
            if threading.current_thread() is executor.thread:
                return 1.0
            return actual_remaining(deadline)

        registry._remaining = controlled_remaining
        started = []
        finished = []
        try:
            with mock.patch.object(self.base, "_ADAPTER_EXECUTOR", executor):
                began = time.monotonic()
                outcome = registry.search_provider_with_attempts(
                    "ddgs",
                    query(),
                    tier=SearchTier.LIGHT,
                    max_results=1,
                    timeout_seconds=timeout_seconds,
                    on_attempt_started=lambda provider_name, search_query, _ready, _at: (
                        started.append((provider_name, search_query.query_id))
                    ),
                    on_attempt_finished=lambda attempt: finished.append(attempt.provider),
                )
                elapsed = time.monotonic() - began
            self.assertTrue(executor.running.is_set())
            before_release = (
                tuple(provider.calls),
                tuple(started),
                tuple(finished),
            )
        finally:
            executor.release.set()
            executor.thread.join(timeout=1.0)
        self.assertFalse(executor.thread.is_alive())
        after_release = (
            tuple(provider.calls),
            tuple(started),
            tuple(finished),
        )
        return {
            "outcome": outcome,
            "elapsed": elapsed,
            "before_release": before_release,
            "after_release": after_release,
        }

    def _invocation_start_wins_timeout(self, *, timeout_seconds=0.05):
        class BlockingProvider:
            name = "ddgs"

            def __init__(self):
                self.calls = []
                self.entered = threading.Event()
                self.release = threading.Event()
                self.completed = threading.Event()

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, search_query, **_kwargs):
                self.calls.append(search_query.query_id)
                self.entered.set()
                try:
                    self.release.wait(timeout=2.0)
                    return _provider_result(query_id=search_query.query_id)
                finally:
                    self.completed.set()

        provider = BlockingProvider()
        registry = self._registry(None, provider)
        started = []
        finished = []
        began = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=1) as caller:
                result_future = caller.submit(
                    registry.search_provider_with_attempts,
                    "ddgs",
                    query(),
                    tier=SearchTier.LIGHT,
                    max_results=1,
                    timeout_seconds=timeout_seconds,
                    on_attempt_started=lambda provider_name, _query, _ready, _at: (
                        started.append(provider_name)
                    ),
                    on_attempt_finished=lambda attempt: finished.append(attempt.provider),
                )
                self.assertTrue(provider.entered.wait(timeout=1.0))
                outcome = result_future.result(timeout=1.0)
            elapsed = time.monotonic() - began
            before_release = (
                tuple(provider.calls),
                tuple(started),
                tuple(finished),
            )
        finally:
            provider.release.set()
            provider.completed.wait(timeout=1.0)
        after_release = (
            tuple(provider.calls),
            tuple(started),
            tuple(finished),
        )
        return {
            "outcome": outcome,
            "elapsed": elapsed,
            "before_release": before_release,
            "after_release": after_release,
        }

    def test_concurrent_calls_return_immutable_attempts_for_their_own_query(self):
        class PerQueryProvider:
            name = "ddgs"

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, search_query, **_kwargs):
                time.sleep(0.03 if search_query.query_id == "qa" else 0.01)
                return self_result(search_query.query_id)

        def self_result(query_id):
            from src.search.models import ProviderHit, ProviderResult

            provider_hit = ProviderHit(
                "ddgs",
                query_id,
                query_id,
                f"https://example.com/{query_id}",
                "result",
                None,
                None,
                None,
                (),
            )
            return ProviderResult("ddgs", ProviderStatus.SUCCESS, (provider_hit,), 0)

        registry = self._registry(None, PerQueryProvider())
        qa = SearchQuery("qa", SearchRoundKind.INITIAL, QueryPurpose.DIRECT, "A")
        qb = SearchQuery("qb", SearchRoundKind.INITIAL, QueryPurpose.DIRECT, "B")
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda item: registry.search_provider_with_attempts(
                        "ddgs",
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

    def test_search_provider_with_attempts_dispatches_exact_provider_without_reserve(self):
        observed = []
        tavily = self._ready_provider("tavily")
        ddgs = self._ready_provider("ddgs")

        def ddgs_search(_query, **kwargs):
            observed.append(("ddgs", kwargs["timeout_seconds"]))
            return _provider_result("ddgs", query_id=_query.query_id)

        ddgs.search.side_effect = ddgs_search
        registry = self._registry(tavily, ddgs)

        outcome = registry.search_provider_with_attempts(
            "ddgs",
            query(),
            tier=SearchTier.LIGHT,
            max_results=1,
            timeout_seconds=4.0,
        )
        self.assertEqual(outcome.status, ProviderStatus.SUCCESS)
        self.assertEqual(len(outcome.attempts), 1)
        self.assertEqual(outcome.attempts[0].provider, "ddgs")
        self.assertEqual(observed[0][0], "ddgs")
        self.assertAlmostEqual(observed[0][1], 4.0, delta=0.1)
        tavily.search.assert_not_called()

    def test_search_provider_with_attempts_returns_not_configured_for_missing_provider(self):
        tavily = self._ready_provider("tavily")
        registry = self._registry(tavily, None)

        outcome = registry.search_provider_with_attempts(
            "ddgs",
            query(),
            tier=SearchTier.LIGHT,
            max_results=1,
            timeout_seconds=4.0,
        )
        self.assertEqual(outcome.status, ProviderStatus.NOT_CONFIGURED)
        self.assertEqual(outcome.attempts, ())

    def test_attempt_retains_invocation_start_readiness_without_completion_reprobe(self):
        class FlippingReadinessProvider:
            name = "ddgs"

            def __init__(self):
                self.readiness_calls = 0

            def readiness(self):
                self.readiness_calls += 1
                if self.readiness_calls <= 3:
                    return ProviderReadiness("ddgs", True, True, None)
                return ProviderReadiness(
                    "ddgs",
                    True,
                    False,
                    SearchFailureCode.PROVIDER_UNAVAILABLE,
                )

            def search(self, search_query, **_kwargs):
                return _provider_result(query_id=search_query.query_id)

        provider = FlippingReadinessProvider()
        outcome = self._registry(None, provider).search_provider_with_attempts(
            "ddgs", query(), tier=SearchTier.LIGHT, max_results=1, timeout_seconds=1.0
        )

        self.assertEqual(provider.readiness_calls, 3)
        self.assertEqual(len(outcome.attempts), 1)
        self.assertTrue(outcome.attempts[0].configured)
        self.assertTrue(outcome.attempts[0].available)
        self.assertTrue(outcome.attempts[0].invocation_started)

    def test_no_provider_is_not_configured(self):
        registry = self._registry(None, None)
        readiness = registry.readiness()
        self.assertEqual(readiness, ())
        self.assertFalse(registry.available_providers())

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
