"""Orchestrator tests: bounded two-stage search state machine."""

from __future__ import annotations

import importlib
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from datetime import date
from unittest import mock

from src.search.models import (
    DEFAULT_TIER_BUDGETS,
    EvidenceState,
    Factuality,
    Freshness,
    ProviderReadiness,
    ProviderStatus,
    RequestSource,
    RetrievalDecision,
    RetrievalRequest,
    RiskLevel,
    SearchFailureCode,
    SearchTier,
    SkipReason,
    TriggerCode,
)
from tests.search_fakes import (
    FakeClock,
    RecordingProvider,
    StaticEvidenceJudge,
    StaticPlannerModel,
    StaticRouterAdvisor,
)


def orchestrator_module():
    try:
        return importlib.import_module("src.search.orchestrator")
    except ModuleNotFoundError:
        raise AssertionError("src.search.orchestrator must exist") from None


def search_module():
    return importlib.import_module("src.search")


def router_payload(tier="light"):
    return {
        "skip_candidate": None,
        "benefit_dimensions": ["accuracy"],
        "factuality": "factual",
        "external_fact_required": True,
        "freshness": "none",
        "risk": "low",
        "actionability": "none",
        "potential_harm": "none",
        "recommended_tier": tier,
        "trigger_codes": ["factual_default"],
    }


def skip_payload(reason="social_or_emotional"):
    return {
        "skip_candidate": {"reason": reason},
        "benefit_dimensions": [],
        "factuality": "non_factual",
        "external_fact_required": False,
        "freshness": "none",
        "risk": "low",
        "actionability": "none",
        "potential_harm": "none",
        "recommended_tier": None,
        "trigger_codes": [],
    }


def request(question="什么是光合作用", force_search=False):
    return RetrievalRequest(
        question,
        force_search=force_search,
        request_source=RequestSource.CHAT,
    )


def _make_router(payload):
    """Wrap a static advisor in the real program router."""
    from src.search.router import RetrievalBenefitRouter
    from tests.search_fakes import StaticRouterAdvisor
    return RetrievalBenefitRouter(StaticRouterAdvisor(payload))


def _make_planner():
    """Real SearchPlanner over a deterministic static model."""
    from src.search.planner import SearchPlanner
    from tests.search_fakes import StaticPlannerModel
    return SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))


class _FakeJudge:
    def __init__(self, verdicts=None, supported_topics=("光合作用", "Rust", "Go")):
        self.verdicts = verdicts or {}
        self.supported_topics = tuple(supported_topics)

    def judge(self, question, candidates):
        del question
        result = {}
        for index, candidate in enumerate(candidates, 1):
            if f"C{index}" in self.verdicts:
                result[f"C{index}"] = self.verdicts[f"C{index}"]
            else:
                result[f"C{index}"] = {
                    "relevance": "direct",
                    "source_relation": "independent",
                    "publisher_entity_match": False,
                    "ownership_basis": None,
                    "supported_topics": list(self.supported_topics),
                    "conflict_key": None,
                }
        return result


class _FakeExtractor:
    def __init__(self):
        self.calls = []

    def extract(self, hit, query, *, allow_network_read, timeout_seconds):
        del allow_network_read, timeout_seconds
        self.calls.append((hit, query))
        from src.search.models import EvidenceCandidate, ExcerptOrigin
        return EvidenceCandidate(
            hit=hit,
            document=None,
            excerpt=hit.snippet or "正文内容",
            excerpt_origin=ExcerptOrigin.PROVIDER_SNIPPET,
            extraction_status="provider_raw_content" if hit.raw_content else "search_result_snippet",
            safety_flags=(),
            content_reads_consumed=1 if hit.raw_content else 0,
        )


class _FakeProvider:
    def __init__(self, name="tavily", hits=()):
        self.name = name
        self.hits = list(hits)
        self.calls = []

    def readiness(self):
        return ProviderReadiness(self.name, True, True, None)

    def search(self, query, *, tier, max_results, timeout_seconds):
        del tier, max_results, timeout_seconds
        self.calls.append(query)
        from src.search.models import ProviderHit, ProviderResult
        if not self.hits:
            return ProviderResult(self.name, ProviderStatus.EMPTY, (), 1)
        return ProviderResult(self.name, ProviderStatus.SUCCESS, tuple(self.hits), 1)


def _hit(url="https://example.com/page", snippet="光合作用的定义内容。", raw_content=None):
    from src.search.models import ProviderHit
    return ProviderHit(
        provider="fake",
        query_id="q1",
        title="Title",
        url=url,
        snippet=snippet,
        score=1.0,
        published_at=None,
        raw_content=raw_content,
        quality_flags=(),
    )


class OrchestratorSkipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = orchestrator_module()

    def test_social_skip_returns_decision_without_provider(self):
        router = _make_router(skip_payload())
        orchestrator = self.module.SearchOrchestrator(
            router=router,
            planner=_make_planner(),
            judge=StaticEvidenceJudge({}),
            providers=(),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request("你好，今天心情有点差"))
        self.assertIs(result.decision.route, SearchTier.SKIP)
        self.assertIs(result.decision.skip_reason, SkipReason.SOCIAL_OR_EMOTIONAL)
        self.assertIsNone(result.plan)
        self.assertIsNone(result.evidence)
        trace = result.trace
        self.assertFalse(trace.orchestrator_started)
        self.assertEqual(trace.provider_attempts, ())
        self.assertEqual(trace.retrieval_round_count, 0)

    def test_user_forbid_web_skip_has_zero_provider_eligibility(self):
        router = _make_router(skip_payload("user_forbid_web"))
        orchestrator = self.module.SearchOrchestrator(
            router=router,
            planner=_make_planner(),
            judge=StaticEvidenceJudge({}),
            providers=(),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request("不要联网，只根据我贴的内容总结"))
        self.assertIs(result.decision.route, SearchTier.SKIP)
        self.assertEqual(result.trace.provider_attempts, ())
        self.assertFalse(result.trace.provider_invocation_started)


class OrchestratorLightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = orchestrator_module()

    def test_light_runs_one_round_and_no_repair(self):
        provider = _FakeProvider(hits=[_hit()])
        judge = _FakeJudge()
        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=judge,
            providers=(provider,),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request())
        self.assertIs(result.decision.route, SearchTier.LIGHT)
        self.assertIsNotNone(result.plan)
        self.assertIsNotNone(result.evidence)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.evidence.retrieval_round_count, 1)
        self.assertFalse(result.evidence.repair_plan.triggered)
        self.assertFalse(result.trace.adaptive_repair_round_started)

    def test_light_sufficient_evidence_seals(self):
        provider = _FakeProvider(hits=[_hit()])
        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(provider,),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request())
        self.assertIs(result.evidence.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIsNone(result.failure_code)


class OrchestratorStandardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = orchestrator_module()

    def _orchestrator(self, provider_hits, judge_verdicts=None, question="Rust 和 Go 的并发模型有什么区别"):
        provider = _FakeProvider(hits=provider_hits)
        from src.search.planner import _derive_required_topics
        topics = _derive_required_topics(question)
        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("standard")),
            planner=_make_planner(),
            judge=_FakeJudge(judge_verdicts or {}, supported_topics=topics),
            providers=(provider,),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        return orchestrator, provider

    def test_standard_three_initial_queries_is_one_round(self):
        orchestrator, provider = self._orchestrator([_hit()] * 3)
        result = orchestrator.run(request("Rust 和 Go 的并发模型有什么区别"))
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(result.evidence.retrieval_round_count, 1)
        self.assertEqual(result.trace.initial_query_count, 3)

    def test_standard_one_repair_makes_second_round(self):
        from src.search.models import ProviderHit
        hits = [_hit()] * 3
        provider = _FakeProvider(hits=hits)
        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("standard")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(provider,),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request("Rust 和 Go 的并发模型有什么区别"))
        self.assertLessEqual(result.evidence.retrieval_round_count, 2)
        self.assertLessEqual(
            len(provider.calls),
            DEFAULT_TIER_BUDGETS[SearchTier.STANDARD].max_total_queries,
        )

    def test_standard_never_exceeds_one_repair(self):
        from src.search.models import ProviderHit
        provider = _FakeProvider(hits=[_hit()] * 3)
        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("standard")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(provider,),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request("Rust 和 Go 的并发模型有什么区别"))
        self.assertLessEqual(result.evidence.retrieval_round_count, 2)
        self.assertLessEqual(
            len(provider.calls),
            DEFAULT_TIER_BUDGETS[SearchTier.STANDARD].max_total_queries,
        )


class OrchestratorFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = orchestrator_module()

    def test_no_provider_is_provider_not_configured(self):
        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request())
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_NOT_CONFIGURED)
        self.assertIsNone(result.evidence)
        self.assertFalse(result.trace.provider_invocation_started)

    def test_provider_failure_never_changes_route_to_skip(self):
        class FailingProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, query, *, tier, max_results, timeout_seconds):
                del query, tier, max_results, timeout_seconds
                from src.search.models import ProviderResult
                return ProviderResult("tavily", ProviderStatus.ERROR, (), 1)

        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(FailingProvider(),),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request())
        self.assertNotEqual(result.decision.route, SearchTier.SKIP)
        self.assertTrue(result.trace.provider_invocation_started)
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_UNAVAILABLE)

    def test_unreadable_content_is_content_unreadable(self):
        class EmptyProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, query, *, tier, max_results, timeout_seconds):
                del query, tier, max_results, timeout_seconds
                from src.search.models import ProviderResult, ProviderHit
                hit = ProviderHit("tavily", "q1", "t", "https://example.com/x", None, None, None, None, ())
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (hit,), 1)

        class NoContentExtractor:
            def extract(self, hit, query, *, allow_network_read, timeout_seconds):
                del query, allow_network_read, timeout_seconds
                from src.search.models import EvidenceCandidate
                if not hit.snippet and not hit.raw_content:
                    return EvidenceCandidate(hit, None, None, None, "no_content", (), 0)
                return EvidenceCandidate(hit, None, hit.snippet, None, "snippet", (), 0)

        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(EmptyProvider(),),
            extractor=NoContentExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request())
        self.assertEqual(result.failure_code, SearchFailureCode.CONTENT_UNREADABLE)


class OrchestratorDeadlineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = orchestrator_module()
        original = DEFAULT_TIER_BUDGETS[SearchTier.LIGHT]
        self.short_budget = SimpleNamespace(
            max_initial_queries=original.max_initial_queries,
            max_candidate_urls=original.max_candidate_urls,
            max_content_reads=original.max_content_reads,
            max_repair_queries=original.max_repair_queries,
            max_total_queries=original.max_total_queries,
            max_retrieval_rounds=original.max_retrieval_rounds,
            hard_timeout_seconds=0.05,
        )
        self.budgets = dict(DEFAULT_TIER_BUDGETS)
        self.budgets[SearchTier.LIGHT] = self.short_budget

    def _run_with_short_budget(self, orchestrator):
        started = time.monotonic()
        with mock.patch.object(self.module, "DEFAULT_TIER_BUDGETS", self.budgets):
            try:
                result = orchestrator.run(request())
            except TimeoutError as exc:  # hard deadline must be contained
                self.fail(f"deadline leaked TimeoutError: {exc}")
        return result, time.monotonic() - started

    def test_provider_deadline_returns_without_waiting_for_running_future(self):
        class SlowProvider(_FakeProvider):
            def search(self, query, *, tier, max_results, timeout_seconds):
                self.calls.append((query, timeout_seconds))
                time.sleep(0.25)
                return super().search(
                    query, tier=tier, max_results=max_results, timeout_seconds=timeout_seconds
                )

        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(SlowProvider(hits=[_hit(raw_content="正文")]),),
            extractor=_FakeExtractor(),
        )

        result, elapsed = self._run_with_short_budget(orchestrator)

        self.assertLess(elapsed, 0.18)
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_TIMEOUT)
        self.assertTrue(result.trace.provider_invocation_started)

    def test_planner_is_bounded_by_same_deadline_and_degrades(self):
        class SlowPlanner:
            def plan(self, *_args, **_kwargs):
                time.sleep(0.25)
                return _make_planner().plan(*_args, **_kwargs)

        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("light")),
            planner=SlowPlanner(),
            judge=_FakeJudge(),
            providers=(),
            extractor=_FakeExtractor(),
        )

        result, elapsed = self._run_with_short_budget(orchestrator)

        self.assertLess(elapsed, 0.18)
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.planning_status.value, "degraded")
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_TIMEOUT)

    def test_judge_is_bounded_by_same_deadline_and_cannot_admit(self):
        class SlowJudge:
            def judge(self, *_args, **_kwargs):
                time.sleep(0.25)
                return {"C1": {"relevance": "direct"}}

        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=SlowJudge(),
            providers=(_FakeProvider(hits=[_hit(raw_content="光合作用正文")]),),
            extractor=_FakeExtractor(),
        )

        result, elapsed = self._run_with_short_budget(orchestrator)

        self.assertLess(elapsed, 0.18)
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_TIMEOUT)
        self.assertEqual(result.evidence.evidence_items, ())

    def test_reader_is_bounded_and_cleanup_does_not_delay_return(self):
        class SlowExtractor:
            def extract(self, *_args, **_kwargs):
                time.sleep(0.25)
                return _FakeExtractor().extract(*_args, **_kwargs)

        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(_FakeProvider(hits=[_hit(raw_content="光合作用正文")]),),
            extractor=SlowExtractor(),
        )

        result, elapsed = self._run_with_short_budget(orchestrator)

        self.assertLess(elapsed, 0.18)
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_TIMEOUT)


class OrchestratorAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = orchestrator_module()

    def test_every_failed_fetch_then_snippet_attempt_consumes_aggregate_read_budget(self):
        from src.search.models import EvidenceCandidate, ExcerptOrigin, ProviderHit, ProviderResult

        class ManyHitsProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                hits = tuple(
                    ProviderHit(
                        "tavily", search_query.query_id, f"h{i}",
                        f"https://example.com/{search_query.query_id}/{i}",
                        "provider snippet", None, None, None, (),
                    )
                    for i in range(6)
                )
                return ProviderResult("tavily", ProviderStatus.SUCCESS, hits, 1)

        class FailedFetchSnippetExtractor:
            def __init__(self):
                self.calls = []

            def extract(self, provider_hit, search_query, **_kwargs):
                self.calls.append((provider_hit, search_query))
                return EvidenceCandidate(
                    provider_hit, None, provider_hit.snippet,
                    ExcerptOrigin.PROVIDER_SNIPPET,
                    "search_result_snippet_after_fetch_failure", (), 1,
                )

        extractor = FailedFetchSnippetExtractor()
        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("standard")),
            planner=_make_planner(),
            judge=_FakeJudge(supported_topics=("定义",)),
            providers=(ManyHitsProvider(),),
            extractor=extractor,
        )

        result = orchestrator.run(request("什么是光合作用"))

        self.assertEqual(len(extractor.calls), DEFAULT_TIER_BUDGETS[SearchTier.STANDARD].max_content_reads)
        self.assertEqual(result.trace.content_read_count, DEFAULT_TIER_BUDGETS[SearchTier.STANDARD].max_content_reads)

    def test_repair_counts_are_deltas_and_hit_keeps_actual_repair_query(self):
        from src.search.models import EvidenceCandidate, ExcerptOrigin, ProviderHit, ProviderResult
        from src.search.planner import _derive_required_topics

        class PerQueryProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                provider_hit = ProviderHit(
                    "tavily", search_query.query_id, search_query.query_id,
                    f"https://example.com/{search_query.query_id}",
                    "正文", None, None, "正文", (),
                )
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (provider_hit,), 1)

        class RecordingExtractor:
            def __init__(self):
                self.query_ids = []

            def extract(self, provider_hit, search_query, **_kwargs):
                self.query_ids.append(search_query.query_id)
                return EvidenceCandidate(
                    provider_hit, None, "正文", ExcerptOrigin.PROVIDER_SNIPPET,
                    "provider_raw_content", (), 1,
                )

        question = "Rust 和 Go 的并发模型有什么区别"
        topics = _derive_required_topics(question)
        extractor = RecordingExtractor()
        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("standard")),
            planner=_make_planner(),
            judge=_FakeJudge(supported_topics=()),
            providers=(PerQueryProvider(),),
            extractor=extractor,
        )

        result = orchestrator.run(request(question))

        self.assertEqual(result.trace.candidate_url_count, 4)
        self.assertEqual(result.trace.content_read_count, 4)
        self.assertEqual(extractor.query_ids[-1], "repair-1")
        self.assertEqual(
            [query_id for query_id, _purpose in result.trace.executed_queries],
            ["initial-1", "initial-2", "initial-3", "repair-1"],
        )
        self.assertEqual(result.evidence.retrieval_round_count, 2)
        self.assertTrue(result.evidence.repair_plan.triggered)
        self.assertEqual(result.evidence.attempts, result.trace.provider_attempts)
        self.assertEqual(result.evidence.gap_analysis, self.module.EvidenceAssembler(_FakeJudge()).analyze_gap(result.plan, result.evidence))


class OrchestratorTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = orchestrator_module()

    def test_skip_trace_has_complete_route_fields(self):
        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(skip_payload()),
            planner=_make_planner(),
            judge=StaticEvidenceJudge({}),
            providers=(),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request("你好，今天心情有点差"))
        trace = result.trace
        self.assertIs(trace.route, SearchTier.SKIP)
        self.assertIs(trace.skip_reason, SkipReason.SOCIAL_OR_EMOTIONAL)
        self.assertFalse(trace.orchestrator_started)
        self.assertEqual(trace.provider_attempts, ())
        self.assertEqual(trace.retrieval_round_count, 0)

    def test_light_trace_distinguishes_start_attempt_sufficient(self):
        provider = _FakeProvider(hits=[_hit()])
        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(provider,),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request())
        trace = result.trace
        self.assertTrue(trace.orchestrator_started)
        self.assertTrue(trace.provider_invocation_started)
        self.assertIs(trace.evidence_state, EvidenceState.SUFFICIENT)
        attempt = trace.to_log_dict()["provider_attempts"][0]
        self.assertEqual(attempt["query_id"], "initial-1")
        self.assertTrue(attempt["configured"])
        self.assertTrue(attempt["available"])
        self.assertTrue(attempt["invocation_started"])

    def test_request_local_redaction_audit_flows_from_plan_and_repair_to_trace(self):
        provider = _FakeProvider(hits=[_hit()] * 3)
        orchestrator = self.module.SearchOrchestrator(
            router=_make_router(router_payload("standard")),
            planner=_make_planner(),
            judge=_FakeJudge(supported_topics=()),
            providers=(provider,),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        secret = "abcdef123456"
        result = orchestrator.run(
            request(f"[CQ:image,file=x.png] data:image/png;base64,AAAA 回调签名 {secret} 什么是光合作用")
        )
        self.assertTrue(result.trace.adaptive_repair_round_started)
        self.assertEqual(result.plan.query_redaction_codes, result.trace.initial_query_redaction_codes)
        self.assertIn("cq_control_code", result.trace.initial_query_redaction_codes)
        self.assertIn("data_url", result.trace.initial_query_redaction_codes)
        self.assertIn("callback_secret", result.trace.adaptive_repair_redaction_codes)
        self.assertNotIn(secret, result.trace.initial_query_redaction_codes)
        self.assertNotIn(secret, result.trace.adaptive_repair_redaction_codes)

        unrelated = orchestrator.run(request("什么是光合作用"))
        self.assertNotIn("callback_secret", unrelated.trace.initial_query_redaction_codes)
        self.assertNotIn("callback_secret", unrelated.trace.adaptive_repair_redaction_codes)


class OrchestratorSingletonTests(unittest.TestCase):
    def test_get_search_orchestrator_returns_singleton(self):
        module = orchestrator_module()
        first = module.get_search_orchestrator()
        second = module.get_search_orchestrator()
        self.assertIs(first, second)

    def test_reset_search_orchestrator_clears_singleton(self):
        module = orchestrator_module()
        first = module.get_search_orchestrator()
        module.reset_search_orchestrator()
        second = module.get_search_orchestrator()
        self.assertIsNot(first, second)
        module.reset_search_orchestrator()


if __name__ == "__main__":
    unittest.main()
