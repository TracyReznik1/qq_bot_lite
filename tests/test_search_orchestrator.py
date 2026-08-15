"""Orchestrator tests: bounded two-stage search state machine."""

from __future__ import annotations

import importlib
import json
import threading
import time
import unittest
from dataclasses import fields, replace
from types import SimpleNamespace
from datetime import date
from unittest import mock

from src.search.models import (
    Actionability,
    DEFAULT_TIER_BUDGETS,
    EvidenceState,
    Factuality,
    Freshness,
    PotentialHarm,
    ProviderReadiness,
    ProviderStatus,
    QueryPurpose,
    RequestSource,
    RetrievalDecision,
    RetrievalRequest,
    RiskLevel,
    SearchFailureCode,
    SearchTrace,
    SearchTier,
    SkipReason,
    TriggerCode,
)
from src.search.budget import (
    DEFAULT_SEARCH_BUDGET_POLICY,
    RouteStageBudget,
    SearchBudgetPolicy,
)
from tests.search_fakes import (
    FakeClock,
    RecordingProvider,
    StaticEvidenceJudge,
    StaticPlannerModel,
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
        "factuality": "factual",
        "external_fact_required": True,
        "complexity_codes": ["comparison"] if tier == "standard" else [],
        "source_requirement": "any_relevant",
        "freshness_requirement": "not_required",
        "as_of": None,
        "date_from": None,
        "date_to": None,
        "version_constraint": None,
        "high_consequence": False,
        "warning_required": False,
        "fail_closed": False,
    }


def skip_payload(reason="social_or_emotional"):
    del reason
    return router_payload("light")


class OrchestratorRequestIdTests(unittest.TestCase):
    def test_request_ids_are_collision_resistant_uuid_hex_values(self):
        module = orchestrator_module()
        values = [module._new_request_id() for _index in range(10_000)]
        self.assertEqual(len(values), len(set(values)))
        for value in values:
            self.assertTrue(value.startswith("req-"))
            self.assertEqual(32, len(value.removeprefix("req-")))
            int(value.removeprefix("req-"), 16)

    def test_uuid_request_ids_survive_trace_safe_log_round_trip(self):
        module = orchestrator_module()
        values = [module._new_request_id() for _index in range(1_000)]
        logged = [
            SearchTrace(value, RequestSource.CHAT, SearchTier.LIGHT).to_log_dict()[
                "request_id"
            ]
            for value in values
        ]
        self.assertEqual(values, logged)
        self.assertEqual(len(logged), len(set(logged)))


def request(question="什么是光合作用", force_search=False):
    return RetrievalRequest(
        question,
        force_search=force_search,
        request_source=RequestSource.CHAT,
    )


def short_watchdog_policy(route, seconds):
    """Keep data caps unchanged while deriving a short interim watchdog."""
    count = len(fields(RouteStageBudget))
    per_stage = max(float(seconds) / float(count), 0.001)
    budgets = {
        tier: DEFAULT_SEARCH_BUDGET_POLICY.for_route(tier)
        for tier in (SearchTier.LIGHT, SearchTier.STANDARD)
    }
    budgets[route] = RouteStageBudget(
        **{
            field.name: per_stage
            for field in fields(RouteStageBudget)
        }
    )
    return SearchBudgetPolicy(budgets)


def _make_router(payload):
    """The production router is pure; payload belongs to the analyzer."""
    del payload
    from src.search.router import RetrievalBenefitRouter
    return RetrievalBenefitRouter()


class _StaticRoutingLLM:
    def __init__(self, payload):
        self.content = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content=self.content)


class _StaticRequestAnalyzer:
    """A real one-shot analyzer over a deterministic test LLM response."""

    def __init__(self, payload):
        from src.search.router import LLMRequestAnalyzer
        self._llm = _StaticRoutingLLM(payload)
        self._analyzer = LLMRequestAnalyzer(self._llm)
        self.calls = []

    def analyze(self, retrieval_request):
        self.calls.append(retrieval_request)
        return self._analyzer.analyze(retrieval_request)


def _make_request_analyzer(payload):
    return _StaticRequestAnalyzer(payload)


def _make_planner():
    """Real SearchPlanner over a deterministic static model."""
    from src.search.planner import SearchPlanner
    from tests.search_fakes import StaticPlannerModel
    return SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))


class _FakeJudge:
    def __init__(self, verdicts=None, supported_topic_ids=None):
        self.verdicts = verdicts or {}
        self.supported_topic_ids = (
            None
            if supported_topic_ids is None
            else tuple(supported_topic_ids)
        )

    def judge(self, question, candidates, *, required_topics=None):
        del question
        available_topic_ids = tuple(
            row["topic_id"]
            for row in (required_topics or ())
            if isinstance(row, dict) and isinstance(row.get("topic_id"), str)
        )
        supported_topic_ids = (
            available_topic_ids
            if self.supported_topic_ids is None
            else tuple(
                topic_id
                for topic_id in self.supported_topic_ids
                if topic_id in available_topic_ids
            )
        )
        result = {}
        for index, candidate in enumerate(candidates, 1):
            if f"C{index}" in self.verdicts:
                result[f"C{index}"] = self.verdicts[f"C{index}"]
            else:
                result[f"C{index}"] = {
                    "candidate_id": f"C{index}",
                    "source_relation": "independent",
                    "publisher_entity_match": False,
                    "ownership_basis": None,
                    "publisher": None,
                    "supported_topic_ids": list(supported_topic_ids),
                    "freshness_by_topic": {
                        topic_id: "satisfied"
                        for topic_id in supported_topic_ids
                    },
                    "conflict_key": None,
                    "conflict_value": None,
                    "conflict_relation": None,
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


class _RecordingRequestAnalyzer:
    def __init__(self, analysis):
        self.analysis = analysis
        self.calls = []

    def analyze(self, retrieval_request):
        self.calls.append(retrieval_request)
        return self.analysis


def _task3_analysis(*, skip_reason=None, complexity_codes=()):
    m = importlib.import_module("src.search.models")
    return m.RequestAnalysis(
        retrieval=m.RetrievalContext(
            must_search=skip_reason is None,
            skip_reason=skip_reason,
            factuality=(
                m.Factuality.NON_FACTUAL
                if skip_reason is not None
                else m.Factuality.FACTUAL
            ),
            external_fact_required=skip_reason is None,
            complexity_codes=complexity_codes,
            source_requirement=m.SourceRequirement.ANY_RELEVANT,
        ),
        freshness=m.FreshnessContext(
            m.FreshnessRequirement.NOT_REQUIRED,
            None,
            None,
            None,
            None,
        ),
        risk=m.RiskContext(False, False, False),
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
            request_analyzer=_make_request_analyzer(skip_payload()),
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
            request_analyzer=_make_request_analyzer(skip_payload("user_forbid_web")),
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


class OrchestratorRequestAnalysisPropagationTests(unittest.TestCase):
    """Task 3: one immutable analysis object crosses every return boundary."""

    def setUp(self) -> None:
        self.module = orchestrator_module()

    def _run(self, analysis, question, *, providers, judge):
        from src.search.router import RetrievalBenefitRouter

        analyzer = _RecordingRequestAnalyzer(analysis)
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=analyzer,
            router=RetrievalBenefitRouter(),
            planner=_make_planner(),
            judge=judge,
            providers=providers,
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request(question))
        self.assertEqual([request(question)], analyzer.calls)
        self.assertIs(analysis, result.analysis)
        return result

    def test_analysis_identity_is_preserved_for_skip_success_partial_and_failure(self):
        m = importlib.import_module("src.search.models")
        skip = self._run(
            _task3_analysis(skip_reason=m.SkipReason.SOCIAL_OR_EMOTIONAL),
            "你好，今天心情有点差",
            providers=(),
            judge=StaticEvidenceJudge({}),
        )
        self.assertIs(skip.decision.route, SearchTier.SKIP)

        success = self._run(
            _task3_analysis(),
            "什么是光合作用",
            providers=(_FakeProvider(hits=[_hit()]),),
            judge=_FakeJudge(),
        )
        self.assertIs(success.evidence.evidence_state, EvidenceState.SUFFICIENT)

        class PartialJudge:
            def judge(self, _question, candidates, *, required_topics=None, **_kwargs):
                supported = tuple(
                    row["topic_id"]
                    for row in (required_topics or ())[:1]
                )
                return {
                    f"C{index}": {
                        "candidate_id": f"C{index}",
                        "source_relation": "independent",
                        "publisher_entity_match": False,
                        "ownership_basis": None,
                        "publisher": None,
                        "supported_topic_ids": list(supported),
                        "freshness_by_topic": {
                            topic_id: "satisfied" for topic_id in supported
                        },
                        "conflict_key": None,
                        "conflict_value": None,
                        "conflict_relation": None,
                    }
                    for index, _candidate in enumerate(candidates, 1)
                }

        partial = self._run(
            _task3_analysis(
                complexity_codes=(m.RetrievalComplexityCode.COMPARISON,),
            ),
            "比较 Rust 和 Go 的并发模型",
            providers=(_FakeProvider(hits=[_hit()] * 3),),
            judge=PartialJudge(),
        )
        self.assertIs(partial.evidence.evidence_state, EvidenceState.PARTIAL)

        failure = self._run(
            _task3_analysis(),
            "什么是光合作用",
            providers=(),
            judge=StaticEvidenceJudge({}),
        )
        self.assertIs(
            failure.failure_code,
            SearchFailureCode.PROVIDER_NOT_CONFIGURED,
        )

    def test_orchestrator_routes_only_the_retrieval_context_once(self):
        m = importlib.import_module("src.search.models")
        retrieval = m.RetrievalContext(
            must_search=True,
            skip_reason=None,
            factuality=m.Factuality.FACTUAL,
            external_fact_required=True,
            complexity_codes=(),
            source_requirement=m.SourceRequirement.ANY_RELEVANT,
        )
        analysis = m.RequestAnalysis(
            retrieval,
            m.FreshnessContext(
                m.FreshnessRequirement.CURRENT,
                None,
                None,
                None,
                None,
            ),
            m.RiskContext(True, True, True),
        )

        class RecordingRouter:
            def __init__(self):
                self.contexts = []

            def decide(self, context):
                self.contexts.append(context)
                return __import__(
                    "src.search.router", fromlist=["RetrievalBenefitRouter"]
                ).RetrievalBenefitRouter().decide(context)

        analyzer = _RecordingRequestAnalyzer(analysis)
        router = RecordingRouter()
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=analyzer,
            router=router,
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )

        result = orchestrator.run(request("什么是光合作用"))

        self.assertEqual(1, len(analyzer.calls))
        self.assertEqual([retrieval], router.contexts)
        self.assertIs(result.analysis, analysis)
        self.assertIs(result.decision.route, SearchTier.LIGHT)

    def test_planner_receives_only_retrieval_and_freshness_contexts_and_trace_counts_dispatch(self):
        m = importlib.import_module("src.search.models")
        retrieval = m.RetrievalContext(
            must_search=True,
            skip_reason=None,
            factuality=m.Factuality.FACTUAL,
            external_fact_required=True,
            complexity_codes=(m.RetrievalComplexityCode.COMPARISON,),
            source_requirement=m.SourceRequirement.INDEPENDENT_CORROBORATION,
        )
        freshness = m.FreshnessContext(
            m.FreshnessRequirement.CURRENT,
            None,
            None,
            None,
            None,
        )
        analysis = m.RequestAnalysis(
            retrieval,
            freshness,
            m.RiskContext(True, True, True),
        )
        standard = m.RetrievalDecision(
            route=m.SearchTier.STANDARD, skip_reason=None, must_search=True, reason_codes=(),
        )

        class Router:
            def decide(self, context):
                self.context = context
                return standard

        class CapturingPlanner:
            def __init__(self):
                self.contexts = None

            def plan(self, retrieval_request, retrieval_decision, retrieval_context, freshness_context, **kwargs):
                self.contexts = (retrieval_context, freshness_context)
                return _make_planner().plan(
                    retrieval_request,
                    retrieval_decision,
                    retrieval_context,
                    freshness_context,
                    **kwargs,
                )

            def plan_repair(self, *args, **kwargs):
                return _make_planner().plan_repair(*args, **kwargs)

        provider = _FakeProvider(hits=[_hit()])
        router = Router()
        planner = CapturingPlanner()
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_RecordingRequestAnalyzer(analysis),
            router=router,
            planner=planner,
            judge=_FakeJudge(),
            providers=(provider,),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )

        result = orchestrator.run(request("比较 Python 并发 API"))

        self.assertIs(retrieval, router.context)
        self.assertEqual((retrieval, freshness), planner.contexts)
        self.assertEqual(len(provider.calls), len(result.trace.executed_queries))
        self.assertIs(QueryPurpose.DIRECT, result.plan.initial_queries[0].purpose)
        self.assertIn(QueryPurpose.DIRECT, [query.purpose for query in provider.calls])

    def test_watchdog_starts_at_response_started_and_anchors_maximum_request_seconds(self):
        clock = FakeClock()
        analysis = _task3_analysis()

        class DelayedAnalyzer:
            def __init__(self):
                self.calls = 0

            def analyze(self, retrieval_request, **kwargs):
                del retrieval_request, kwargs
                self.calls += 1
                clock.advance(17)
                return analysis

        class CapturingPlanner:
            def __init__(self):
                self.timeout_seconds = None

            def plan(self, retrieval_request, retrieval_decision, retrieval_context, freshness_context, **kwargs):
                self.timeout_seconds = kwargs.get("timeout_seconds")
                return _make_planner().plan(
                    retrieval_request,
                    retrieval_decision,
                    retrieval_context,
                    freshness_context,
                    **kwargs,
                )

        analyzer = DelayedAnalyzer()
        planner = CapturingPlanner()
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=analyzer,
            router=__import__("src.search.router", fromlist=["RetrievalBenefitRouter"]).RetrievalBenefitRouter(),
            planner=planner,
            judge=StaticEvidenceJudge({}),
            providers=(),
            extractor=_FakeExtractor(),
            clock=clock,
        )
        result = orchestrator.run(request("什么是光合作用"))

        self.assertEqual(1, analyzer.calls)
        self.assertEqual(
            DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.LIGHT).planner_seconds,
            planner.timeout_seconds,
        )
        self.assertEqual(17_000, result.trace.route_latency_ms)


class OrchestratorLightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = orchestrator_module()

    def test_light_runs_one_round_and_no_repair(self):
        provider = _FakeProvider(hits=[_hit()])
        judge = _FakeJudge()
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
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
            request_analyzer=_make_request_analyzer(router_payload("light")),
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
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("standard")),
            router=_make_router(router_payload("standard")),
            planner=_make_planner(),
            judge=_FakeJudge(judge_verdicts or {}),
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
            request_analyzer=_make_request_analyzer(router_payload("standard")),
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
            request_analyzer=_make_request_analyzer(router_payload("standard")),
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
            request_analyzer=_make_request_analyzer(router_payload("light")),
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
        self.assertEqual(result.trace.executed_queries, ())
        self.assertEqual(result.trace.to_log_dict()["semantic_query_count"], 0)

    def test_empty_bundle_marks_only_material_topics_missing(self):
        m = importlib.import_module("src.search.models")
        decision = m.RetrievalDecision(
            route=m.SearchTier.STANDARD, skip_reason=None, must_search=True, reason_codes=(),
        )
        search_plan = m.SearchPlan(
            decision,
            "比较并发 API",
            m.PlanningStatus.NORMAL,
            (),
            None,
            (m.SearchQuery(
                "initial-1",
                m.SearchRoundKind.INITIAL,
                m.QueryPurpose.DIRECT,
                "比较并发 API",
                query_index=1,
                target_topic_ids=("topic-2",),
            ),),
            (
                m.RequiredTopic(
                    "topic-1", "background", False,
                    m.FreshnessRequirement.NOT_REQUIRED,
                ),
                m.RequiredTopic(
                    "topic-2", "core", True,
                    m.FreshnessRequirement.NOT_REQUIRED,
                ),
            ),
            frozenset(),
            (),
            m.DEFAULT_TIER_BUDGETS[m.SearchTier.STANDARD],
        )

        bundle = self.module._empty_bundle(search_plan)

        self.assertEqual(("core",), bundle.missing_claim_topics)

    def _assert_bundle_trace_mirror(self, result):
        bundle = result.evidence
        self.assertIsNotNone(bundle)
        self.assertIs(result.trace.evidence_state, bundle.evidence_state)
        self.assertEqual(
            result.trace.citable_evidence_count,
            sum(1 for item in bundle.evidence_items if item.citable),
        )
        self.assertEqual(result.trace.provider_attempts, bundle.attempts)
        self.assertEqual(result.trace.retrieval_round_count, bundle.retrieval_round_count)
        self.assertEqual(
            len(result.trace.provider_failures),
            len(set(result.trace.provider_failures)),
        )

    def test_no_results_failure_seals_bundle_and_trace_from_same_state(self):
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(_FakeProvider(hits=()),),
            extractor=_FakeExtractor(),
        )

        result = orchestrator.run(request())

        self.assertEqual(result.failure_code, SearchFailureCode.NO_RESULTS)
        self._assert_bundle_trace_mirror(result)

    def test_unavailable_failure_seals_bundle_and_trace_from_same_state(self):
        class UnavailableProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness(
                    "tavily", True, False, SearchFailureCode.PROVIDER_UNAVAILABLE
                )

            def search(self, *_args, **_kwargs):
                raise AssertionError("unavailable adapter must not execute")

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(UnavailableProvider(),),
            extractor=_FakeExtractor(),
        )

        result = orchestrator.run(request())

        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_UNAVAILABLE)
        self._assert_bundle_trace_mirror(result)

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
            request_analyzer=_make_request_analyzer(router_payload("light")),
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

    def test_unreadable_content_becomes_insufficient_evidence_with_content_gap(self):
        class EmptyProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, query, *, tier, max_results, timeout_seconds):
                del tier, max_results, timeout_seconds
                from src.search.models import ProviderResult, ProviderHit
                hit = ProviderHit(
                    "tavily",
                    query.query_id,
                    "t",
                    "https://example.com/x",
                    None,
                    None,
                    None,
                    None,
                    (),
                )
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (hit,), 1)

        class NoContentExtractor:
            def extract(self, hit, query, *, allow_network_read, timeout_seconds):
                del query, allow_network_read, timeout_seconds
                from src.search.models import EvidenceCandidate
                if not hit.snippet and not hit.raw_content:
                    return EvidenceCandidate(hit, None, None, None, "no_content", (), 0)
                return EvidenceCandidate(hit, None, hit.snippet, None, "snippet", (), 0)

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(EmptyProvider(),),
            extractor=NoContentExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request())
        self.assertEqual(result.failure_code, SearchFailureCode.INSUFFICIENT_EVIDENCE)
        self.assertEqual(
            (importlib.import_module("src.search.models").RepairReasonCode.CONTENT_UNREADABLE,),
            result.evidence.gap_analysis.repair_reason_codes,
        )
        self._assert_bundle_trace_mirror(result)

    def test_light_zero_content_does_not_dispatch_repair(self):
        class ZeroContentProvider:
            name = "tavily"

            def __init__(self):
                self.calls = []

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                from src.search.models import ProviderHit, ProviderResult

                self.calls.append(search_query.query_id)
                hit = ProviderHit(
                    "tavily",
                    search_query.query_id,
                    "zero-content",
                    f"https://example.com/{search_query.query_id}",
                    None,
                    None,
                    None,
                    None,
                    (),
                )
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (hit,), 1)

        class ZeroContentExtractor:
            def extract(self, hit, _query, **_kwargs):
                from src.search.models import EvidenceCandidate

                return EvidenceCandidate(hit, None, None, None, "no_content", (), 1)

        provider = ZeroContentProvider()
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(supported_topic_ids=()),
            providers=(provider,),
            extractor=ZeroContentExtractor(),
            clock=FakeClock(),
        )

        result = orchestrator.run(request("什么是光合作用"))

        self.assertEqual(SearchFailureCode.INSUFFICIENT_EVIDENCE, result.failure_code)
        self.assertFalse(result.trace.repair_used)
        self.assertEqual(0, result.trace.repair_query_count)
        self.assertEqual(1, result.trace.retrieval_round_count)
        self.assertEqual(1, len(provider.calls))
        self.assertEqual(1, result.trace.candidate_url_count)
        self.assertEqual(1, result.trace.content_read_count)
        self.assertEqual(1, len(result.trace.provider_attempts))
        self.assertEqual(
            (importlib.import_module("src.search.models").RepairReasonCode.CONTENT_UNREADABLE,),
            result.evidence.gap_analysis.repair_reason_codes,
        )


class OrchestratorDeadlineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = orchestrator_module()
        self.short_policy = short_watchdog_policy(SearchTier.LIGHT, 0.05)

    def _run_with_short_watchdog(self, orchestrator):
        started = time.monotonic()
        with mock.patch.object(self.module, "DEFAULT_SEARCH_BUDGET_POLICY", self.short_policy):
            try:
                result = orchestrator.run(request())
            except TimeoutError as exc:  # hard deadline must be contained
                self.fail(f"deadline leaked TimeoutError: {exc}")
        return result, time.monotonic() - started

    def _assert_bundle_trace_mirror(self, result):
        bundle = result.evidence
        self.assertIsNotNone(bundle)
        self.assertIs(result.trace.evidence_state, bundle.evidence_state)
        self.assertEqual(
            result.trace.citable_evidence_count,
            sum(1 for item in bundle.evidence_items if item.citable),
        )
        self.assertEqual(result.trace.provider_attempts, bundle.attempts)
        self.assertEqual(result.trace.retrieval_round_count, bundle.retrieval_round_count)
        self.assertEqual(
            len(result.trace.provider_failures),
            len(set(result.trace.provider_failures)),
        )

    def test_provider_deadline_returns_without_waiting_for_running_future(self):
        class SlowProvider(_FakeProvider):
            def search(self, query, *, tier, max_results, timeout_seconds):
                self.calls.append((query, timeout_seconds))
                time.sleep(0.25)
                return super().search(
                    query, tier=tier, max_results=max_results, timeout_seconds=timeout_seconds
                )

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(SlowProvider(hits=[_hit(raw_content="正文")]),),
            extractor=_FakeExtractor(),
        )

        result, elapsed = self._run_with_short_watchdog(orchestrator)

        self.assertLess(elapsed, 0.18)
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_TIMEOUT)
        self.assertTrue(result.trace.provider_invocation_started)
        self._assert_bundle_trace_mirror(result)

    def test_adapter_queue_expiry_is_timeout_without_semantic_execution(self):
        from src.search.providers import base as provider_base

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

        blockers = [provider_base._ADAPTER_EXECUTOR.submit(occupy_worker) for _ in range(8)]
        self.assertTrue(all_workers_started.wait(timeout=1.0))

        class QueuedProvider:
            name = "tavily"

            def __init__(self):
                self.calls = []

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                self.calls.append(search_query.query_id)
                raise AssertionError("queued provider must never start")

        provider = QueuedProvider()
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(provider,),
            extractor=_FakeExtractor(),
        )

        try:
            result, elapsed = self._run_with_short_watchdog(orchestrator)
        finally:
            release_workers.set()
            for blocker in blockers:
                blocker.result(timeout=1.0)

        self.assertLess(elapsed, 0.18)
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_TIMEOUT)
        self.assertEqual(provider.calls, [])
        self.assertEqual(result.trace.provider_attempts, ())
        self.assertEqual(result.trace.executed_queries, ())
        self.assertEqual(result.trace.initial_query_count, 0)
        self.assertFalse(result.trace.provider_invocation_started)
        self.assertEqual(result.trace.to_log_dict()["semantic_query_count"], 0)

    def test_running_adapter_wrapper_cannot_mutate_after_timeout_trace_is_sealed(self):
        from concurrent.futures import Future
        from src.search.providers import base as provider_base

        class RunningBeforeInvokeExecutor:
            def __init__(self):
                self.running = threading.Event()
                self.release = threading.Event()
                self.sealed = threading.Event()
                self.thread = None

            def submit(self, operation):
                owner = self

                class SealObservedFuture(Future):
                    def cancel(self):
                        cancelled = super().cancel()
                        owner.sealed.set()
                        return cancelled

                future = SealObservedFuture()

                def run():
                    if not future.set_running_or_notify_cancel():
                        return
                    self.running.set()
                    self.release.wait(timeout=2.0)
                    try:
                        value = operation()
                    except BaseException as exc:
                        future.set_exception(exc)
                    else:
                        future.set_result(value)

                self.thread = threading.Thread(target=run, daemon=True)
                self.thread.start()
                if not self.running.wait(timeout=1.0):
                    raise RuntimeError("test executor did not enter RUNNING state")
                return future

        class RecordingProvider:
            name = "tavily"

            def __init__(self):
                self.calls = []

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                self.calls.append(search_query.query_id)
                from src.search.models import ProviderResult
                return ProviderResult("tavily", ProviderStatus.EMPTY, (), 0)

        executor = RunningBeforeInvokeExecutor()
        provider = RecordingProvider()
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(provider,),
            extractor=_FakeExtractor(),
        )
        actual_remaining = orchestrator._registry._remaining

        def controlled_remaining(deadline):
            if threading.current_thread() is executor.thread:
                return 1.0
            return actual_remaining(deadline)

        orchestrator._registry._remaining = controlled_remaining
        try:
            with mock.patch.object(provider_base, "_ADAPTER_EXECUTOR", executor):
                result, elapsed = self._run_with_short_watchdog(orchestrator)
            self.assertTrue(executor.running.is_set())
            self.assertTrue(executor.sealed.wait(timeout=1.0))
            trace_before_release = result.trace.to_log_dict()
            calls_before_release = tuple(provider.calls)
        finally:
            executor.release.set()
            executor.thread.join(timeout=1.0)

        self.assertFalse(executor.thread.is_alive())
        self.assertLess(elapsed, 0.18)
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_TIMEOUT)
        self.assertEqual(calls_before_release, ())
        self.assertEqual(tuple(provider.calls), calls_before_release)
        self.assertEqual(result.trace.provider_attempts, ())
        self.assertEqual(result.trace.executed_queries, ())
        self.assertFalse(result.trace.provider_invocation_started)
        self.assertEqual(result.trace.to_log_dict(), trace_before_release)

    def test_fallback_timeout_keeps_completed_primary_and_real_fallback_truth(self):
        class ErrorPrimary:
            name = "ddgs"

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, _query, **_kwargs):
                from src.search.models import ProviderResult
                return ProviderResult("ddgs", ProviderStatus.ERROR, (), 1)

        class SlowFallback:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                time.sleep(0.25)
                from src.search.models import ProviderHit, ProviderResult
                provider_hit = ProviderHit(
                    "tavily", search_query.query_id, "late", "https://late.example/item",
                    "late", None, None, None, (),
                )
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (provider_hit,), 1)

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(ErrorPrimary(), SlowFallback()),
            extractor=_FakeExtractor(),
        )

        result, elapsed = self._run_with_short_watchdog(orchestrator)

        self.assertLess(elapsed, 0.18)
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_TIMEOUT)
        self.assertEqual(
            [(attempt.provider, attempt.status) for attempt in result.trace.provider_attempts],
            [
                ("ddgs", ProviderStatus.ERROR),
                ("tavily", ProviderStatus.TIMEOUT),
            ],
        )
        self._assert_bundle_trace_mirror(result)

    def test_planner_is_bounded_by_same_deadline_and_degrades(self):
        class SlowPlanner:
            def plan(self, *_args, **_kwargs):
                time.sleep(0.25)
                return _make_planner().plan(*_args, **_kwargs)

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=SlowPlanner(),
            judge=_FakeJudge(),
            providers=(),
            extractor=_FakeExtractor(),
        )

        result, elapsed = self._run_with_short_watchdog(orchestrator)

        self.assertLess(elapsed, 0.18)
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.planning_status.value, "degraded")
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_NOT_CONFIGURED)

    def test_judge_is_bounded_by_same_deadline_and_cannot_admit(self):
        class SlowJudge:
            def judge(self, *_args, **_kwargs):
                time.sleep(0.25)
                return {"C1": {"source_relation": "independent"}}

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=SlowJudge(),
            providers=(_FakeProvider(hits=[_hit(raw_content="光合作用正文")]),),
            extractor=_FakeExtractor(),
        )

        result, elapsed = self._run_with_short_watchdog(orchestrator)

        self.assertLess(elapsed, 0.18)
        self.assertEqual(result.failure_code, SearchFailureCode.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.evidence.evidence_items, ())

    def test_reader_is_bounded_and_cleanup_does_not_delay_return(self):
        class SlowExtractor:
            def extract(self, *_args, **_kwargs):
                time.sleep(0.25)
                return _FakeExtractor().extract(*_args, **_kwargs)

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(_FakeProvider(hits=[_hit(raw_content="光合作用正文")]),),
            extractor=SlowExtractor(),
        )

        result, elapsed = self._run_with_short_watchdog(orchestrator)

        self.assertLess(elapsed, 0.18)
        self.assertEqual(result.failure_code, SearchFailureCode.INSUFFICIENT_EVIDENCE)

    def test_queued_fourth_query_is_never_recorded_as_executed_or_invoked(self):
        started_queries = []
        lock = threading.Lock()

        class BlockingProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                with lock:
                    started_queries.append(search_query.query_id)
                time.sleep(0.25)
                from src.search.models import ProviderResult
                return ProviderResult("tavily", ProviderStatus.EMPTY, (), 1)


        class FourQueryPlanner:
            def plan(self, *args, **kwargs):
                from src.search.models import QueryPurpose, SearchQuery, SearchRoundKind

                base = _make_planner().plan(*args, **kwargs)
                fourth = SearchQuery(
                    "initial-4",
                    SearchRoundKind.INITIAL,
                    QueryPurpose.PRIMARY,
                    "fourth query",
                    query_index=4,
                    target_topic_ids=base.initial_queries[0].target_topic_ids,
                )
                object.__setattr__(
                    base,
                    "initial_queries",
                    (*base.initial_queries, fourth),
                )
                return base

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("standard")),
            router=_make_router(router_payload("standard")),
            planner=FourQueryPlanner(),
            judge=_FakeJudge(),
            providers=(BlockingProvider(),),
            extractor=_FakeExtractor(),
        )

        with mock.patch.object(
            self.module,
            "DEFAULT_SEARCH_BUDGET_POLICY",
            short_watchdog_policy(SearchTier.STANDARD, 0.05),
        ):
            with self.assertRaisesRegex(
                AssertionError,
                "tier/direct contract",
            ):
                orchestrator.run(request("Rust 和 Go 的并发模型有什么区别"))
        time.sleep(0.27)

        self.assertEqual(started_queries, [])

    def test_repair_timeout_preserves_citable_truth_and_deduplicates_failure(self):
        class RepairBlockingProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                from src.search.models import ProviderHit, ProviderResult
                if search_query.query_id == "repair-1":
                    time.sleep(0.25)
                    return ProviderResult("tavily", ProviderStatus.EMPTY, (), 1)
                provider_hit = ProviderHit(
                    "tavily",
                    search_query.query_id,
                    search_query.query_id,
                    f"https://example.com/{search_query.query_id}",
                    "正文",
                    None,
                    None,
                    "正文",
                    (),
                )
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (provider_hit,), 1)

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("standard")),
            router=_make_router(router_payload("standard")),
            planner=_make_planner(),
            judge=_FakeJudge(supported_topic_ids=("topic-1",)),
            providers=(RepairBlockingProvider(),),
            extractor=_FakeExtractor(),
        )
        started = time.monotonic()

        with mock.patch.object(
            self.module,
            "DEFAULT_SEARCH_BUDGET_POLICY",
            short_watchdog_policy(SearchTier.STANDARD, 0.12),
        ):
            result = orchestrator.run(request("Rust 和 Go 的并发模型有什么区别"))
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.22)
        self.assertEqual(result.failure_code, SearchFailureCode.PROVIDER_TIMEOUT)
        self.assertGreater(len(result.evidence.evidence_items), 0)
        self.assertGreater(
            sum(1 for item in result.evidence.evidence_items if item.citable), 0
        )
        self.assertIn("hard_deadline_exceeded", result.evidence.limitations)
        self._assert_bundle_trace_mirror(result)


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
            request_analyzer=_make_request_analyzer(router_payload("standard")),
            router=_make_router(router_payload("standard")),
            planner=_make_planner(),
            judge=_FakeJudge(supported_topic_ids=("topic-1",)),
            providers=(ManyHitsProvider(),),
            extractor=extractor,
        )

        result = orchestrator.run(request("光合作用和呼吸作用有什么区别"))

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
            request_analyzer=_make_request_analyzer(router_payload("standard")),
            router=_make_router(router_payload("standard")),
            planner=_make_planner(),
            judge=_FakeJudge(supported_topic_ids=()),
            providers=(PerQueryProvider(),),
            extractor=extractor,
        )

        result = orchestrator.run(request(question))

        self.assertEqual(result.trace.candidate_url_count, 4)
        self.assertEqual(result.trace.content_read_count, 4)
        self.assertEqual(extractor.query_ids[-1], "repair-1")
        self.assertEqual(
            [entry.query_index for entry in result.trace.executed_queries],
            [1, 2, 3, 4],
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
            request_analyzer=_make_request_analyzer(skip_payload()),
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
            request_analyzer=_make_request_analyzer(router_payload("light")),
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
        self.assertEqual(attempt["query_index"], 1)
        self.assertTrue(attempt["configured"])
        self.assertTrue(attempt["available"])
        self.assertTrue(attempt["invocation_started"])

    def test_evidence_judge_anomalies_flow_to_the_body_free_trace(self):
        m = importlib.import_module("src.search.models")

        def row(candidate_id):
            return {
                "candidate_id": candidate_id,
                "source_relation": "independent",
                "publisher_entity_match": False,
                "ownership_basis": None,
                "publisher": None,
                "supported_topic_ids": ["topic-1"],
                "freshness_by_topic": {"topic-1": "not_required"},
                "conflict_key": None,
                "conflict_value": None,
                "conflict_relation": None,
            }

        class StaticLLM:
            def chat(self, *_args, **_kwargs):
                return SimpleNamespace(
                    content=json.dumps(
                        {"candidates": {"C1": row("C1"), "C99": row("C99")}, "gap_hints": []}
                    )
                )

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=self.module.LLMEvidenceJudge(StaticLLM()),
            providers=(_FakeProvider(hits=[_hit()]),),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )

        result = orchestrator.run(request())

        self.assertEqual(
            (m.JudgeAnomalyCode.UNKNOWN_CANDIDATE,),
            result.evidence.judge_anomaly_codes,
        )
        self.assertEqual(1, result.evidence.judge_anomaly_count)
        logged = result.trace.to_log_dict()
        self.assertEqual(["unknown_candidate"], logged["judge_anomaly_codes"])
        self.assertEqual(1, logged["judge_anomaly_count"])
        self.assertNotIn("C99", json.dumps(logged))

    def test_request_local_redaction_audit_flows_from_plan_and_repair_to_trace(self):
        provider = _FakeProvider(hits=[_hit()] * 3)
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("standard")),
            router=_make_router(router_payload("standard")),
            planner=_make_planner(),
            judge=_FakeJudge(supported_topic_ids=()),
            providers=(provider,),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        secret = "abcdef123456"
        result = orchestrator.run(
            request(
                f"[CQ:image,file=x.png] data:image/png;base64,AAAA 回调签名 "
                f"{secret} 光合作用和呼吸作用有什么区别"
            )
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

    def test_repair_stage_timings_are_non_overlapping_and_queries_serialize_flat(self):
        m = __import__("src.search.models", fromlist=["SearchPlan"])

        class ManualClock:
            def __init__(self):
                self.now = 0.0
                self.lock = threading.Lock()

            def monotonic(self):
                with self.lock:
                    return self.now

            def advance(self, seconds):
                with self.lock:
                    self.now += seconds

        clock = ManualClock()
        d = RetrievalDecision(
            route=SearchTier.STANDARD,
            skip_reason=None,
            must_search=True,
            reason_codes=(),
        )
        initial_query = m.SearchQuery(
            "initial-1", m.SearchRoundKind.INITIAL, m.QueryPurpose.DIRECT, "topic",
            None, None, (), (), 1, ("topic-1",),
        )
        repair_query = m.SearchQuery(
            "repair-1", m.SearchRoundKind.REPAIR, m.QueryPurpose.REPAIR, "topic source",
            None, None, (), (), 2, ("topic-1",),
        )
        search_plan = m.SearchPlan(
            d, "topic", m.PlanningStatus.NORMAL, (), None, (initial_query,),
            (
                m.RequiredTopic(
                    "topic-1", "topic", True,
                    m.FreshnessRequirement.NOT_REQUIRED,
                ),
            ),
            frozenset(), (), m.DEFAULT_TIER_BUDGETS[SearchTier.STANDARD],
        )

        class Router:
            def decide(self, _context):
                clock.advance(0.007)
                return d

        class Planner:
            def plan(self, *_args, **_kwargs):
                clock.advance(0.011)
                return search_plan

            def plan_repair(self, *_args, **_kwargs):
                clock.advance(0.013)
                return m.RepairPlan(
                    True,
                    (m.RepairReasonCode.MISSING_TOPIC,),
                    ("topic-1",),
                    repair_query,
                )

        class Provider:
            name = "tavily"

            def readiness(self):
                return m.ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                clock.advance(0.017)
                hit = m.ProviderHit(
                    "tavily", search_query.query_id, search_query.query_id,
                    f"https://example.com/{search_query.query_id}",
                    "topic", None, None, "topic", (),
                )
                return m.ProviderResult("tavily", m.ProviderStatus.SUCCESS, (hit,), 1)

        class Extractor:
            def extract(self, hit, _query, **_kwargs):
                clock.advance(0.019)
                return m.EvidenceCandidate(
                    hit, None, "topic", m.ExcerptOrigin.PAGE_EXTRACT,
                    "page_extract", (), 1,
                )

        class Judge:
            def __init__(self):
                self.calls = 0

            def judge(self, _question, candidates, **kwargs):
                clock.advance(0.023)
                self.calls += 1
                supported = (
                    [
                        row["topic_id"]
                        for row in kwargs.get("required_topics", ())
                    ]
                    if self.calls > 1
                    else []
                )
                return {
                    f"C{index}": {
                        "candidate_id": f"C{index}",
                        "source_relation": "independent",
                        "publisher_entity_match": False,
                        "ownership_basis": None,
                        "publisher": "Example",
                        "supported_topic_ids": supported,
                        "freshness_by_topic": {
                            topic_id: "satisfied" for topic_id in supported
                        },
                        "conflict_key": None,
                        "conflict_value": None,
                        "conflict_relation": None,
                    }
                    for index, _candidate in enumerate(candidates, 1)
                }

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_RecordingRequestAnalyzer(
                _task3_analysis(complexity_codes=(m.RetrievalComplexityCode.COMPARISON,))
            ),
            router=Router(), planner=Planner(), judge=Judge(),
            providers=(Provider(),), extractor=Extractor(), clock=clock,
        )
        result = orchestrator.run(request("topic"))
        trace = result.trace

        self.assertAlmostEqual(17, trace.initial_provider_search_latency_ms, delta=1)
        self.assertAlmostEqual(34, trace.provider_search_total_latency_ms, delta=1)
        self.assertAlmostEqual(19, trace.initial_content_read_latency_ms, delta=1)
        self.assertAlmostEqual(38, trace.content_read_total_latency_ms, delta=1)
        self.assertAlmostEqual(23, trace.initial_evidence_assembly_latency_ms, delta=1)
        self.assertAlmostEqual(46, trace.evidence_assembly_total_latency_ms, delta=1)
        self.assertAlmostEqual(72, trace.adaptive_repair_latency_ms, delta=1)
        self.assertAlmostEqual(142, trace.retrieval_pipeline_latency_ms, delta=1)
        self.assertEqual(
            [1, 2],
            [row["query_index"] for row in trace.to_log_dict()["executed_queries"]],
        )
        json.dumps(trace.to_log_dict())

    def test_finalizer_records_real_end_from_start_and_logs_exactly_once(self):
        m = __import__("src.search.models", fromlist=["SearchTrace"])
        trace = m.SearchTrace("req-1", RequestSource.CHAT, SearchTier.LIGHT)
        trace.response_started_at = 10.0
        with mock.patch.object(self.module.logger, "info") as log:
            self.module.finalize_search_trace(trace, response_finished_at=12.5)
            self.module.finalize_search_trace(trace, response_finished_at=99.0)
        self.assertEqual(12.5, trace.response_finished_at)
        self.assertEqual(2500, trace.total_response_latency_ms)
        self.assertTrue(trace.finalized)
        log.assert_called_once()


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


class OrchestratorProductionGraphTests(unittest.TestCase):
    def test_production_provider_graph_uses_only_ddgs_without_tavily_key(self):
        module = orchestrator_module()
        fake_config = SimpleNamespace(
            tavily_api_key="",
            proxy_url="",
            request_timeout=18.0,
        )
        ddgs = SimpleNamespace(name="ddgs")

        with (
            mock.patch.object(module, "config", fake_config),
            mock.patch(
                "src.services.llm_client.get_llm_client",
                return_value=SimpleNamespace(chat=mock.Mock()),
            ),
            mock.patch.object(module, "DDGSSearchProvider", return_value=ddgs),
            mock.patch.object(module, "TavilySearchProvider") as tavily_constructor,
        ):
            orchestrator = module._build_production_orchestrator()

        self.assertEqual([provider.name for provider in orchestrator._providers], ["ddgs"])
        tavily_constructor.assert_not_called()

    def test_production_provider_graph_is_ddgs_then_tavily(self):
        module = orchestrator_module()
        fake_config = SimpleNamespace(
            tavily_api_key="test-key",
            proxy_url="",
            request_timeout=18.0,
        )
        ddgs = SimpleNamespace(name="ddgs")
        tavily = SimpleNamespace(name="tavily")

        with (
            mock.patch.object(module, "config", fake_config),
            mock.patch(
                "src.services.llm_client.get_llm_client",
                return_value=SimpleNamespace(chat=mock.Mock()),
            ),
            mock.patch.object(module, "DDGSSearchProvider", return_value=ddgs),
            mock.patch.object(module, "TavilySearchProvider", return_value=tavily),
        ):
            orchestrator = module._build_production_orchestrator()

        self.assertEqual(
            [provider.name for provider in orchestrator._providers],
            ["ddgs", "tavily"],
        )


class RepairBudgetAndStopTests(unittest.TestCase):
    """Task 6: deterministic repair budgets, trace metadata, and the post-repair stop."""

    def setUp(self) -> None:
        self.module = orchestrator_module()

    def _orchestrator(self, *, providers, judge, extractor=None, planner=None, clock=None):
        return self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("standard")),
            router=_make_router(router_payload("standard")),
            planner=planner if planner is not None else _make_planner(),
            judge=judge,
            providers=providers,
            extractor=extractor if extractor is not None else _FakeExtractor(),
            clock=clock if clock is not None else FakeClock(),
        )

    @staticmethod
    def _per_query_provider():
        from src.search.models import ProviderHit, ProviderResult

        class PerQueryProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                hit = ProviderHit(
                    "tavily",
                    search_query.query_id,
                    search_query.query_id,
                    f"https://example.com/{search_query.query_id}",
                    "正文",
                    None,
                    None,
                    "正文",
                    (),
                )
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (hit,), 1)

        return PerQueryProvider()

    def test_full_capacity_repair_uses_request_wide_query_indexes(self):
        m = importlib.import_module("src.search.models")
        from src.search.models import ProviderResult

        class DdgsEmptyProvider:
            name = "ddgs"

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, _query, **_kwargs):
                return ProviderResult("ddgs", ProviderStatus.EMPTY, (), 1)

        orchestrator = self._orchestrator(
            providers=(DdgsEmptyProvider(), self._per_query_provider()),
            judge=_FakeJudge(supported_topic_ids=()),
        )
        result = orchestrator.run(request("Rust 和 Go 的并发模型有什么区别"))
        trace = result.trace

        indexes = [entry.query_index for entry in trace.executed_queries]
        self.assertEqual([1, 2, 3, 4], sorted(set(indexes)))
        self.assertLessEqual(trace.semantic_query_count, 4)
        self.assertEqual(4, trace.semantic_query_count)
        self.assertLessEqual(trace.initial_query_count, 3)
        self.assertEqual(1, trace.repair_query_count)
        self.assertLessEqual(trace.candidate_url_count, 8)
        self.assertLessEqual(trace.content_read_count, 5)
        self.assertLessEqual(trace.retrieval_round_count, 2)
        # DDGS and Tavily attempts for query index 1 are one semantic query.
        index_one_entries = [entry for entry in trace.executed_queries if entry.query_index == 1]
        self.assertEqual(2, len(index_one_entries))
        self.assertIs(trace.retrieval_stop_reason, m.RetrievalStopReason.POST_REPAIR_STOP)

    def test_provider_fallback_does_not_increment_semantic_query_count(self):
        from src.search.models import ProviderResult

        class DdgsEmptyProvider:
            name = "ddgs"

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, _query, **_kwargs):
                return ProviderResult("ddgs", ProviderStatus.EMPTY, (), 1)

        orchestrator = self._orchestrator(
            providers=(DdgsEmptyProvider(), self._per_query_provider()),
            judge=_FakeJudge(),
        )
        result = orchestrator.run(request("Rust 和 Go 的并发模型有什么区别"))
        self.assertEqual(result.trace.semantic_query_count, result.trace.initial_query_count)
        self.assertGreater(result.trace.semantic_query_count, 0)
        index_one_entries = [
            entry for entry in result.trace.executed_queries if entry.query_index == 1
        ]
        self.assertEqual(2, len(index_one_entries))

    def test_post_repair_stop_prevents_a_second_repair_dispatch(self):
        m = importlib.import_module("src.search.models")

        class CallCountingJudge(_FakeJudge):
            def __init__(self):
                super().__init__(supported_topic_ids=())
                self.calls = 0

            def judge(self, question, candidates, *, required_topics=None):
                self.calls += 1
                if self.calls == 1:
                    return _FakeJudge(supported_topic_ids=()).judge(
                        question, candidates, required_topics=required_topics
                    )
                return _FakeJudge().judge(question, candidates, required_topics=required_topics)

        judge = CallCountingJudge()
        orchestrator = self._orchestrator(
            providers=(self._per_query_provider(),),
            judge=judge,
        )
        result = orchestrator.run(request("Rust 和 Go 的并发模型有什么区别"))

        self.assertIs(result.trace.retrieval_stop_reason, m.RetrievalStopReason.POST_REPAIR_STOP)
        self.assertEqual(1, result.trace.repair_query_count)
        self.assertEqual(2, result.evidence.retrieval_round_count)
        self.assertEqual(2, judge.calls)
        self.assertTrue(result.evidence.repair_plan.triggered)
        self.assertTrue(result.trace.adaptive_repair_round_started)

    def test_repair_preserves_first_round_judge_anomalies_in_final_trace(self):
        m = importlib.import_module("src.search.models")

        def verdict(candidate_id, topic_ids=()):
            return {
                "candidate_id": candidate_id,
                "source_relation": "independent",
                "publisher_entity_match": False,
                "ownership_basis": None,
                "publisher": None,
                "supported_topic_ids": list(topic_ids),
                "freshness_by_topic": {
                    topic_id: "not_required" for topic_id in topic_ids
                },
                "conflict_key": None,
                "conflict_value": None,
                "conflict_relation": None,
            }

        class TwoRoundLLM:
            def __init__(self):
                self.calls = 0

            def chat(self, messages, **_kwargs):
                self.calls += 1
                payload = json.loads(messages[1]["content"])
                candidate_ids = [
                    item["candidate_id"] for item in payload["candidates"]
                ]
                topic_ids = [
                    item["topic_id"] for item in payload["required_topics"]
                ]
                if self.calls == 1:
                    candidates = {
                        "C1": verdict("C1"),
                        "C99": verdict("C99"),
                    }
                else:
                    candidates = {
                        candidate_id: verdict(candidate_id, topic_ids)
                        for candidate_id in candidate_ids
                    }
                return SimpleNamespace(
                    content=json.dumps({"candidates": candidates, "gap_hints": []})
                )

        llm = TwoRoundLLM()
        orchestrator = self._orchestrator(
            providers=(self._per_query_provider(),),
            judge=self.module.LLMEvidenceJudge(llm),
        )

        result = orchestrator.run(request("Rust 和 Go 的并发模型有什么区别"))

        self.assertEqual(2, llm.calls)
        self.assertTrue(result.trace.adaptive_repair_round_started)
        self.assertEqual(2, result.evidence.retrieval_round_count)
        self.assertEqual(
            (
                m.JudgeAnomalyCode.MISSING_CANDIDATE,
                m.JudgeAnomalyCode.UNKNOWN_CANDIDATE,
            ),
            result.evidence.judge_anomaly_codes,
        )
        self.assertEqual(3, result.evidence.judge_anomaly_count)
        self.assertEqual(
            result.evidence.judge_anomaly_codes,
            result.trace.judge_anomaly_codes,
        )
        self.assertEqual(3, result.trace.judge_anomaly_count)

    def test_light_route_never_dispatches_repair(self):
        m = importlib.import_module("src.search.models")
        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(supported_topic_ids=()),
            providers=(_FakeProvider(hits=[_hit()]),),
            extractor=_FakeExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request("什么是光合作用"))
        self.assertFalse(result.trace.adaptive_repair_round_started)
        self.assertEqual(0, result.trace.repair_query_count)
        self.assertIs(result.trace.retrieval_stop_reason, m.RetrievalStopReason.NO_REPAIR_BENEFIT)

    def test_unreadable_candidates_produce_content_unreadable_repair(self):
        m = importlib.import_module("src.search.models")

        d = RetrievalDecision(
            route=SearchTier.STANDARD,
            skip_reason=None,
            must_search=True,
            reason_codes=(),
        )
        topics = (
            m.RequiredTopic("topic-1", "background", False, m.FreshnessRequirement.NOT_REQUIRED),
            m.RequiredTopic("topic-2", "core", True, m.FreshnessRequirement.NOT_REQUIRED),
        )
        direct = m.SearchQuery(
            "initial-1",
            m.SearchRoundKind.INITIAL,
            m.QueryPurpose.DIRECT,
            "比较并发 API",
            query_index=1,
            target_topic_ids=("topic-2",),
        )
        search_plan = m.SearchPlan(
            d, "比较并发 API", m.PlanningStatus.NORMAL, (), None, (direct,),
            topics, frozenset({m.SourceRelation.PRIMARY, m.SourceRelation.INDEPENDENT}),
            (), m.DEFAULT_TIER_BUDGETS[SearchTier.STANDARD],
        )

        class FixedPlanPlanner:
            def __init__(self):
                self._real = _make_planner()

            def plan(self, *_args, **_kwargs):
                return search_plan

            def plan_repair(self, plan, gap, prior_fingerprints=()):
                return self._real.plan_repair(plan, gap, prior_fingerprints=prior_fingerprints)

        class UnreadableExtractor:
            def extract(self, hit, _query, **_kwargs):
                return m.EvidenceCandidate(
                    hit,
                    None,
                    hit.snippet,
                    m.ExcerptOrigin.PROVIDER_SNIPPET,
                    "search_result_snippet_after_fetch_failure",
                    (),
                    1,
                )

        orchestrator = self._orchestrator(
            providers=(self._per_query_provider(),),
            judge=_FakeJudge(supported_topic_ids=()),
            extractor=UnreadableExtractor(),
            planner=FixedPlanPlanner(),
        )
        result = orchestrator.run(request("比较并发 API"))

        self.assertTrue(result.trace.repair_used)
        self.assertEqual(
            (m.RepairReasonCode.CONTENT_UNREADABLE,),
            result.evidence.repair_plan.reason_codes,
        )
        self.assertEqual(("topic-2",), result.evidence.repair_plan.target_topic_ids)
        self.assertIs(result.trace.retrieval_stop_reason, m.RetrievalStopReason.POST_REPAIR_STOP)

    def test_standard_zero_content_runs_exactly_one_content_unreadable_repair(self):
        m = importlib.import_module("src.search.models")
        decision = RetrievalDecision(
            route=SearchTier.STANDARD,
            skip_reason=None,
            must_search=True,
            reason_codes=(),
        )
        direct = m.SearchQuery(
            "initial-1",
            m.SearchRoundKind.INITIAL,
            m.QueryPurpose.DIRECT,
            "比较并发 API",
            query_index=1,
            target_topic_ids=("topic-1",),
        )
        search_plan = m.SearchPlan(
            decision,
            "比较并发 API",
            m.PlanningStatus.NORMAL,
            (),
            None,
            (direct,),
            (
                m.RequiredTopic(
                    "topic-1",
                    "core",
                    True,
                    m.FreshnessRequirement.NOT_REQUIRED,
                ),
            ),
            frozenset({m.SourceRelation.PRIMARY, m.SourceRelation.INDEPENDENT}),
            (),
            m.DEFAULT_TIER_BUDGETS[SearchTier.STANDARD],
        )

        class FixedPlanPlanner:
            def __init__(self):
                self._real = _make_planner()

            def plan(self, *_args, **_kwargs):
                return search_plan

            def plan_repair(self, plan, gap, prior_fingerprints=()):
                return self._real.plan_repair(
                    plan,
                    gap,
                    prior_fingerprints=prior_fingerprints,
                )

        class ZeroContentProvider:
            name = "tavily"

            def __init__(self):
                self.calls = []

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                self.calls.append(search_query.query_id)
                hit = m.ProviderHit(
                    "tavily",
                    search_query.query_id,
                    "zero-content",
                    f"https://example.com/{search_query.query_id}",
                    None,
                    None,
                    None,
                    None,
                    (),
                )
                return m.ProviderResult(
                    "tavily",
                    m.ProviderStatus.SUCCESS,
                    (hit,),
                    1,
                )

        class ZeroContentExtractor:
            def extract(self, hit, _query, **_kwargs):
                return m.EvidenceCandidate(
                    hit,
                    None,
                    None,
                    None,
                    "no_content",
                    (),
                    1,
                )

        provider = ZeroContentProvider()
        orchestrator = self._orchestrator(
            providers=(provider,),
            judge=_FakeJudge(supported_topic_ids=()),
            extractor=ZeroContentExtractor(),
            planner=FixedPlanPlanner(),
        )

        result = orchestrator.run(request("比较并发 API"))

        self.assertEqual(SearchFailureCode.INSUFFICIENT_EVIDENCE, result.failure_code)
        self.assertTrue(result.trace.repair_used)
        self.assertEqual(1, result.trace.repair_query_count)
        self.assertEqual(2, result.trace.retrieval_round_count)
        self.assertEqual(["initial-1", "repair-1"], provider.calls)
        self.assertEqual(2, result.trace.candidate_url_count)
        self.assertEqual(2, result.trace.content_read_count)
        self.assertEqual(2, len(result.trace.provider_attempts))
        self.assertEqual(
            (m.RepairReasonCode.CONTENT_UNREADABLE,),
            result.evidence.repair_plan.reason_codes,
        )
        self.assertEqual(
            ("topic-1",),
            result.evidence.repair_plan.target_topic_ids,
        )
        self.assertIs(
            result.trace.retrieval_stop_reason,
            m.RetrievalStopReason.POST_REPAIR_STOP,
        )
        self.assertEqual(
            (m.RepairReasonCode.CONTENT_UNREADABLE,),
            result.evidence.gap_analysis.repair_reason_codes,
        )
        self.assertEqual(
            ("topic-1",),
            result.evidence.gap_analysis.repair_target_topic_ids,
        )
        self.assertEqual(
            (m.RepairReasonCode.CONTENT_UNREADABLE,),
            result.trace.repair_reason_codes,
        )

    def test_judge_hint_for_an_unknown_topic_does_not_repair(self):
        m = importlib.import_module("src.search.models")

        class HintingJudge(_FakeJudge):
            def judge(self, question, candidates, *, required_topics=None):
                result = super().judge(question, candidates, required_topics=required_topics)
                result["gap_hints"] = (
                    {"reason_code": "premise_mismatch", "target_topic_id": "topic-9"},
                )
                return result

        orchestrator = self._orchestrator(
            providers=(self._per_query_provider(),),
            judge=HintingJudge(),
        )
        result = orchestrator.run(request("Rust 和 Go 的并发模型有什么区别"))
        self.assertFalse(result.evidence.repair_plan.triggered)
        self.assertIs(result.trace.retrieval_stop_reason, m.RetrievalStopReason.EVIDENCE_SUFFICIENT)

    def test_exhausted_candidate_and_read_budgets_block_repair(self):
        m = importlib.import_module("src.search.models")

        class ManyHitsProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                from src.search.models import ProviderHit, ProviderResult

                hits = tuple(
                    ProviderHit(
                        "tavily",
                        search_query.query_id,
                        str(index),
                        f"https://example.com/{search_query.query_id}/{index}",
                        "正文",
                        None,
                        None,
                        "正文",
                        (),
                    )
                    for index in range(8)
                )
                return ProviderResult("tavily", ProviderStatus.SUCCESS, hits, 1)

        orchestrator = self._orchestrator(
            providers=(ManyHitsProvider(),),
            judge=_FakeJudge(supported_topic_ids=()),
        )
        result = orchestrator.run(request("Rust 和 Go 的并发模型有什么区别"))

        self.assertGreaterEqual(result.trace.candidate_url_count, 8)
        self.assertGreaterEqual(result.trace.content_read_count, 5)
        self.assertFalse(result.trace.repair_used)
        self.assertFalse(result.evidence.repair_plan.triggered)
        self.assertIs(result.trace.retrieval_stop_reason, m.RetrievalStopReason.BUDGET_EXHAUSTED)

    def test_failed_fetch_provider_content_fallback_counts_one_read(self):
        from src.search.models import EvidenceCandidate, ExcerptOrigin, ProviderHit, ProviderResult

        class ProviderRawContentProvider:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, search_query, **_kwargs):
                hit = ProviderHit(
                    "tavily",
                    search_query.query_id,
                    "title",
                    f"https://example.com/{search_query.query_id}",
                    "正文",
                    None,
                    None,
                    "可读正文",
                    (),
                )
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (hit,), 1)

        class FallbackExtractor:
            def extract(self, hit, _query, **_kwargs):
                return EvidenceCandidate(
                    hit,
                    None,
                    hit.raw_content,
                    ExcerptOrigin.PROVIDER_SNIPPET,
                    "provider_raw_content",
                    (),
                    1,
                )

        orchestrator = self.module.SearchOrchestrator(
            request_analyzer=_make_request_analyzer(router_payload("light")),
            router=_make_router(router_payload("light")),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=(ProviderRawContentProvider(),),
            extractor=FallbackExtractor(),
            clock=FakeClock(),
        )
        result = orchestrator.run(request("什么是光合作用"))
        self.assertEqual(1, result.trace.content_read_count)
        self.assertIs(result.evidence.evidence_state, EvidenceState.SUFFICIENT)


if __name__ == "__main__":
    unittest.main()
