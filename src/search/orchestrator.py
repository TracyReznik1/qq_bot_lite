"""Bounded two-stage search orchestrator with exact Trace timings."""

from __future__ import annotations

import logging
import inspect
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import replace
from threading import Lock
from typing import Any, Callable, Sequence
from uuid import uuid4

from src.config import config
from src.search.evidence import EvidenceAssembler, LLMEvidenceJudge
from src.search.extraction import SearchExtractor
from src.search.models import (
    Actionability,
    DEFAULT_TIER_BUDGETS,
    EvidenceBundle,
    EvidenceState,
    ExcerptOrigin,
    Freshness,
    FreshnessRequirement,
    PlanningStatus,
    PotentialHarm,
    ProviderAttempt,
    ProviderResult,
    ProviderStatus,
    QueryPurpose,
    QueryTraceEntry,
    RequestAnalysis,
    RepairPlan,
    RequestSource,
    RetrievalStopReason,
    RetrievalRequest,
    RiskLevel,
    SearchFailureCode,
    SearchPipelineResult,
    SearchPlan,
    SearchQuery,
    SearchRoundKind,
    SearchTier,
    SearchTrace,
    SkipReason,
    TopicFreshnessTraceEntry,
    TriggerCode,
)
from src.search.planner import SearchPlanner, _query_fingerprint
from src.search.providers import (
    DDGSSearchProvider,
    ProviderRegistry,
    TavilySearchProvider,
)
from src.search.providers.base import ProviderSearchOutcome
from src.search.router import LLMRequestAnalyzer, RetrievalBenefitRouter

logger = logging.getLogger("qq-bot")


def _new_request_id() -> str:
    return f"req-{uuid4().hex}"


class _QueryAttemptTracker:
    """Thread-safe, request-local adapter truth for one semantic query."""

    def __init__(self, query: SearchQuery) -> None:
        self.query = query
        self._lock = Lock()
        self._order: list[str] = []
        self._started: dict[str, tuple[Any, float]] = {}
        self._finished: dict[str, ProviderAttempt] = {}

    def on_started(
        self,
        provider: str,
        query: SearchQuery,
        readiness: Any,
        started_at: float,
    ) -> None:
        if query.query_id != self.query.query_id:
            return
        with self._lock:
            if provider not in self._started:
                self._order.append(provider)
                self._started[provider] = (readiness, started_at)

    def on_finished(self, attempt: ProviderAttempt) -> None:
        with self._lock:
            if attempt.provider in self._started:
                self._finished[attempt.provider] = attempt

    def snapshot(self, now: float) -> tuple[ProviderAttempt, ...]:
        with self._lock:
            attempts: list[ProviderAttempt] = []
            for provider in self._order:
                completed = self._finished.get(provider)
                if completed is not None:
                    attempts.append(completed)
                    continue
                readiness, started_at = self._started[provider]
                attempts.append(
                    ProviderAttempt(
                        provider=provider,
                        status=ProviderStatus.TIMEOUT,
                        count=1,
                        latency_ms=max(int((now - started_at) * 1000), 0),
                        query_id=self.query.query_id,
                        configured=readiness.configured,
                        available=readiness.available,
                        invocation_started=True,
                    )
                )
            return tuple(attempts)


class SearchOrchestrator:
    """Run the route -> plan -> retrieve -> evidence state machine."""

    def __init__(
        self,
        *,
        request_analyzer: Any,
        router: Any,
        planner: Any,
        judge: Any,
        providers: Sequence[Any] | None = None,
        extractor: Any = None,
        clock: Any = None,
    ) -> None:
        self._request_analyzer = request_analyzer
        self._router = router
        self._planner = planner
        self._judge = judge
        self._providers = providers if providers is not None else ()
        self._extractor = extractor if extractor is not None else SearchExtractor()
        self._clock = clock
        self._registry = ProviderRegistry(self._providers)

    def run(self, request: RetrievalRequest) -> SearchPipelineResult:
        response_started = self._monotonic()
        analysis = self._request_analyzer.analyze(request)
        if not isinstance(analysis, RequestAnalysis):
            raise TypeError("request_analyzer must return a RequestAnalysis")
        trace = SearchTrace(
            request_id=_new_request_id(),
            request_source=request.request_source,
            route=SearchTier.LIGHT,
            response_started_at=response_started,
        )
        retrieval_decision = self._router.decide(analysis.retrieval)
        decision = retrieval_decision
        trace.route = retrieval_decision.route
        trace.skip_reason = retrieval_decision.skip_reason
        trace.must_search = retrieval_decision.must_search
        trace.retrieval_reason_codes = retrieval_decision.reason_codes
        trace.route_latency_ms = self._elapsed_ms(response_started)

        if retrieval_decision.route is SearchTier.SKIP:
            if retrieval_decision.requires_clarification:
                trace.degradation_reason = SearchFailureCode.USER_FORBID_WEB
            result = SearchPipelineResult(
                decision=retrieval_decision,
                plan=None,
                evidence=None,
                trace=trace,
                failure_code=(
                    SearchFailureCode.USER_FORBID_WEB
                    if retrieval_decision.skip_reason is SkipReason.USER_FORBID_WEB
                    else None
                ),
                analysis=analysis,
            )
            return result

        # One monotonic hard deadline per tier starts before planning, so
        # planner and Evidence-judge latency count toward the retrieval budget.
        budget = DEFAULT_TIER_BUDGETS[retrieval_decision.route]
        deadline = self._monotonic() + float(budget.hard_timeout_seconds)

        trace.orchestrator_started = True
        retrieval_started = self._monotonic()
        plan_started = retrieval_started
        plan_completed, plan = self._call_until_deadline(
            self._invoke_planner,
            deadline,
            request,
            retrieval_decision,
            analysis.retrieval,
            analysis.freshness,
            deadline,
        )
        if not plan_completed or not isinstance(plan, SearchPlan):
            plan = self._degraded_plan(
                request,
                retrieval_decision,
                analysis.retrieval,
                analysis.freshness,
            )
        if (
            len(plan.initial_queries) > budget.max_initial_queries
            or not plan.initial_queries
            or plan.initial_queries[0].purpose is not QueryPurpose.DIRECT
        ):
            raise AssertionError("planner initial queries violated the tier/direct contract")
        trace.query_planning_latency_ms = self._elapsed_ms(plan_started)
        trace.initial_query_redaction_codes = plan.query_redaction_codes

        if self._expired(deadline):
            return self._timeout_result(
                decision, plan, trace, retrieval_started, analysis=analysis,
            )

        trace.provider_configured = any(
            provider.readiness().configured for provider in self._providers
        )

        initial_round_started = self._monotonic()
        trace.initial_round_started = True
        trace.retrieval_round_count = 1

        provider_results = self._run_initial_batch(
            plan,
            retrieval_decision,
            budget,
            deadline,
            trace,
        )
        trace.initial_provider_search_latency_ms = self._elapsed_ms(initial_round_started)
        trace.provider_search_total_latency_ms = trace.initial_provider_search_latency_ms

        if self._expired(deadline):
            return self._timeout_result(
                decision, plan, trace, retrieval_started, analysis=analysis,
            )

        if not provider_results or all(
            result.status in {ProviderStatus.NOT_CONFIGURED, ProviderStatus.UNAVAILABLE, ProviderStatus.ERROR, ProviderStatus.TIMEOUT, ProviderStatus.EMPTY}
            for result in provider_results
        ):
            status = next(
                (result.status for result in provider_results),
                ProviderStatus.NOT_CONFIGURED,
            )
            failure = _failure_for_status(status)
            return self._failure_result(
                decision,
                plan,
                trace,
                retrieval_started,
                failure=failure,
                bundle=None,
                limitation=_limitation_for_failure(failure),
                analysis=analysis,
            )

        content_started = self._monotonic()
        candidates, candidate_keys, reads, unreadable_query_ids = self._extract_candidates(
            plan, provider_results, budget, deadline, trace,
        )
        unreadable_topic_ids = _unreadable_topic_ids(plan, unreadable_query_ids)
        trace.initial_content_read_latency_ms = self._elapsed_ms(content_started)
        trace.content_read_total_latency_ms = trace.initial_content_read_latency_ms
        trace.candidate_url_count = len(candidate_keys)

        if self._expired(deadline):
            return self._timeout_result(
                decision, plan, trace, retrieval_started, analysis=analysis,
            )

        evidence_started = self._monotonic()
        assembled, bundle = self._call_until_deadline(
            self._assembler().assemble,
            deadline,
            plan,
            candidates,
            timeout_seconds=self._remaining(deadline),
        )
        if not assembled or not isinstance(bundle, EvidenceBundle):
            bundle = self._assembler_with_rejecting_judge().assemble(plan, candidates, timeout_seconds=0.0)
        trace.initial_evidence_assembly_latency_ms = self._elapsed_ms(evidence_started)
        trace.evidence_assembly_total_latency_ms = trace.initial_evidence_assembly_latency_ms
        trace.citable_evidence_count = sum(1 for item in bundle.evidence_items if item.citable)
        trace.evidence_state = bundle.evidence_state
        self._record_evidence_trace(trace, bundle)

        if self._expired(deadline):
            return self._timeout_result(
                decision,
                plan,
                trace,
                retrieval_started,
                bundle=bundle,
                analysis=analysis,
            )

        gap_started = self._monotonic()
        gap = self._assembler().analyze_gap(
            plan,
            bundle,
            content_unreadable_topic_ids=unreadable_topic_ids,
        )
        trace.gap_analysis_latency_ms = self._elapsed_ms(gap_started)
        initial_canonical_urls = {
            item.canonical_url for item in bundle.evidence_items if item.canonical_url
        }
        repair = RepairPlan(False, (), (), None)

        route_standard = retrieval_decision.route is SearchTier.STANDARD
        gates_pass = _repair_gates_pass(
            route_standard=route_standard,
            repair_used=trace.repair_used,
            semantic_query_count=trace.semantic_query_count,
            candidate_url_count=trace.candidate_url_count,
            content_read_count=trace.content_read_count,
            retrieval_round_count=trace.retrieval_round_count,
            remaining_seconds=self._remaining(deadline),
            budget=budget,
        )
        repair_dispatched = False
        post_repair_gap = gap
        if gates_pass and gap.repair_eligible:
            repair_started = self._monotonic()
            prior_fingerprints = tuple(
                _query_fingerprint(query.text) for query in plan.initial_queries
            )
            repair_completed, planned_repair = self._call_until_deadline(
                self._invoke_repair_planner,
                deadline,
                plan,
                gap,
                prior_fingerprints,
                deadline,
            )
            if repair_completed and isinstance(planned_repair, RepairPlan):
                repair = planned_repair
            if repair.triggered and repair.repair_query is not None:
                trace.repair_reason_codes = repair.reason_codes
                trace.repair_target_topic_ids = repair.target_topic_ids
                repair_dispatched = True
                trace.adaptive_repair_round_started = True
                trace.adaptive_repair_redaction_codes = repair.query_redaction_codes
                trace.retrieval_round_count = 2
                repair_provider_started = self._monotonic()
                repair_result = self._run_repair_query(
                    repair,
                    retrieval_decision,
                    budget,
                    deadline,
                    trace,
                )
                repair_provider_finished = self._monotonic()
                trace.provider_search_total_latency_ms += self._elapsed_ms(
                    repair_provider_started
                )
                repair_candidates, repair_keys, more_reads, _repair_unreadable = self._extract_candidates(
                    plan,
                    repair_result,
                    budget,
                    deadline,
                    trace,
                    additional_queries=(repair.repair_query,),
                    existing_candidate_keys=candidate_keys,
                    existing_read_count=reads,
                )
                repair_unreadable_topic_ids = _unreadable_topic_ids(
                    plan,
                    _repair_unreadable,
                    additional_queries=(repair.repair_query,),
                )
                unreadable_topic_ids = tuple(
                    dict.fromkeys(
                        (*unreadable_topic_ids, *repair_unreadable_topic_ids)
                    )
                )
                candidate_keys.update(repair_keys)
                reads = min(reads + more_reads, budget.max_content_reads)
                trace.candidate_url_count = len(candidate_keys)
                trace.content_read_count = reads
                trace.content_read_total_latency_ms += self._elapsed_ms(repair_provider_finished)
                repair_evidence_started = self._monotonic()
                assembled, repaired_bundle = self._call_until_deadline(
                    self._assembler().assemble,
                    deadline,
                    plan,
                    (*candidates, *repair_candidates),
                    previous=bundle,
                    timeout_seconds=self._remaining(deadline),
                )
                if assembled and isinstance(repaired_bundle, EvidenceBundle):
                    bundle = repaired_bundle
                trace.evidence_assembly_total_latency_ms += self._elapsed_ms(repair_evidence_started)
                trace.evidence_state = bundle.evidence_state
                self._record_evidence_trace(trace, bundle)
                # This is diagnostic-only: it is sealed below and deliberately
                # never feeds a second repair dispatch.
                post_repair_gap = self._assembler().analyze_gap(
                    plan,
                    bundle,
                    content_unreadable_topic_ids=unreadable_topic_ids,
                )
                trace.repair_used = True
                # A second gap never triggers another dispatch.
                trace.retrieval_stop_reason = RetrievalStopReason.POST_REPAIR_STOP
            trace.adaptive_repair_latency_ms = self._elapsed_ms(repair_started)

        if self._expired(deadline):
            return self._timeout_result(
                decision,
                plan,
                trace,
                retrieval_started,
                bundle=bundle,
                gap=post_repair_gap,
                repair=repair,
                initial_canonical_urls=initial_canonical_urls,
                analysis=analysis,
            )

        if not repair_dispatched:
            final_gap = self._assembler().analyze_gap(
                plan,
                bundle,
                content_unreadable_topic_ids=unreadable_topic_ids,
            )
            trace.retrieval_stop_reason = _repair_stop_reason(
                bundle,
                gap,
                repair,
                gates_pass,
                route_standard,
            )
        else:
            # Post-repair state is diagnostic-only: it never opens a third
            # round, even if its gap remains repair-eligible.
            final_gap = post_repair_gap
        bundle = self._finalize_bundle(
            bundle,
            trace,
            retrieval_started,
            gap=final_gap,
            repair=repair,
            initial_canonical_urls=initial_canonical_urls,
        )

        failure_code = _failure_for_state(bundle.evidence_state)
        return SearchPipelineResult(
            decision=decision,
            plan=plan,
            evidence=bundle,
            trace=trace,
            failure_code=failure_code,
            analysis=analysis,
        )

    # ── internal stages ─────────────────────────────────────────────

    def _run_initial_batch(
        self,
        plan: SearchPlan,
        decision: RetrievalDecision,
        budget: Any,
        deadline: float,
        trace: SearchTrace,
    ) -> list[ProviderResult]:
        budget_cap = min(budget.max_initial_queries, len(plan.initial_queries))
        queries = plan.initial_queries[:budget_cap]
        results = self._dispatch_queries(queries, decision, deadline, trace)
        trace.initial_query_count = len(
            {
                entry.query_index
                for entry in trace.executed_queries
                if entry.round_kind is SearchRoundKind.INITIAL
            }
        )
        return results

    def _run_repair_query(
        self,
        repair: RepairPlan,
        decision: RetrievalDecision,
        budget: Any,
        deadline: float,
        trace: SearchTrace,
    ) -> list[ProviderResult]:
        del budget
        query = repair.repair_query
        if query is None:
            return []
        result = self._dispatch_queries((query,), decision, deadline, trace)
        return result

    def _dispatch_queries(
        self,
        queries: Sequence[SearchQuery],
        decision: RetrievalDecision,
        deadline: float,
        trace: SearchTrace,
    ) -> list[ProviderResult]:
        results: list[ProviderResult] = []
        if not queries:
            return results
        executor = ThreadPoolExecutor(max_workers=min(len(queries), 4), thread_name_prefix="search-provider")
        futures: dict[Any, _QueryAttemptTracker] = {}
        outcomes: dict[_QueryAttemptTracker, ProviderSearchOutcome] = {}
        try:
            for query in queries:
                remaining = self._remaining(deadline)
                if remaining <= 0:
                    break
                tracker = _QueryAttemptTracker(query)
                future = executor.submit(
                    self._search_one,
                    query,
                    decision,
                    deadline,
                    tracker,
                )
                futures[future] = tracker
            try:
                for future in as_completed(futures, timeout=self._remaining(deadline)):
                    tracker = futures[future]
                    try:
                        outcome = future.result()
                    except Exception:
                        logger.debug(
                            "query dispatch failed for %s",
                            tracker.query.query_id,
                            exc_info=True,
                        )
                        continue
                    if isinstance(outcome, ProviderSearchOutcome):
                        outcomes[tracker] = outcome
            except FuturesTimeoutError:
                for future in futures:
                    if future.done():
                        continue
                    future.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        observed_at = self._monotonic()
        for future, tracker in futures.items():
            outcome = outcomes.get(tracker)
            if outcome is None and future.done() and not future.cancelled():
                try:
                    completed = future.result(timeout=0)
                except Exception:
                    completed = None
                if isinstance(completed, ProviderSearchOutcome):
                    outcome = completed
            attempts = outcome.attempts if outcome is not None else tracker.snapshot(observed_at)
            if attempts:
                self._record_query_attempts(trace, tracker.query, attempts)
            if outcome is not None:
                results.append(outcome.result)
            elif attempts:
                last = attempts[-1]
                results.append(
                    ProviderResult(
                        provider=last.provider,
                        status=ProviderStatus.TIMEOUT,
                        hits=(),
                        latency_ms=sum(attempt.latency_ms for attempt in attempts),
                    )
                )
        return results

    def _search_one(
        self,
        query: SearchQuery,
        decision: Any,
        deadline: float,
        tracker: _QueryAttemptTracker,
    ) -> ProviderSearchOutcome:
        max_results = max(int(config.search_max_results or 1), 1)
        return self._registry.search_with_attempts(
            query,
            tier=decision.route,
            max_results=max_results,
            timeout_seconds=min(config.request_timeout, self._remaining(deadline)),
            on_attempt_started=tracker.on_started,
            on_attempt_finished=tracker.on_finished,
        )

    def _extract_candidates(
        self,
        plan: SearchPlan,
        results: Sequence[ProviderResult],
        budget: Any,
        deadline: float,
        trace: SearchTrace,
        *,
        additional_queries: Sequence[SearchQuery] = (),
        existing_candidate_keys: set[str] | None = None,
        existing_read_count: int = 0,
    ) -> tuple[list[Any], set[str], int, set[str]]:
        hits: list[Any] = []
        for result in results:
            if result.status is ProviderStatus.SUCCESS:
                hits.extend(result.hits)
        existing_keys = set(existing_candidate_keys or ())
        new_keys: set[str] = set()
        unique_hits: list[Any] = []
        for hit in hits:
            key = str(hit.url or "").strip()
            if not key or key in existing_keys or key in new_keys:
                continue
            if len(existing_keys) + len(new_keys) >= budget.max_candidate_urls:
                break
            new_keys.add(key)
            unique_hits.append(hit)

        candidates: list[Any] = []
        attempted_query_ids: set[str] = set()
        readable_query_ids: set[str] = set()
        remaining_read_budget = max(budget.max_content_reads - existing_read_count, 0)
        read_hits = unique_hits[:remaining_read_budget]
        if not read_hits:
            return candidates, new_keys, 0, set()
        executor = ThreadPoolExecutor(
            max_workers=min(len(read_hits), 4),
            thread_name_prefix="search-reader",
        )
        futures: dict[Any, Any] = {}
        try:
            for hit in read_hits:
                remaining = self._remaining(deadline)
                if remaining <= 0:
                    break
                attempted_query_ids.add(str(hit.query_id or ""))
                future = executor.submit(
                    self._extractor.extract,
                    hit,
                    _query_for_hit(plan, hit, additional_queries=additional_queries),
                    allow_network_read=True,
                    timeout_seconds=min(config.request_timeout, remaining),
                )
                futures[future] = hit
            try:
                for future in as_completed(futures, timeout=self._remaining(deadline)):
                    try:
                        candidate = future.result()
                    except Exception:
                        logger.debug("extraction failed", exc_info=True)
                        continue
                    if candidate is not None and candidate.excerpt:
                        candidates.append(candidate)
                        if _readable_candidate(candidate):
                            readable_query_ids.add(str(candidate.hit.query_id or ""))
            except FuturesTimeoutError:
                trace.provider_failures = (*trace.provider_failures, SearchFailureCode.PROVIDER_TIMEOUT)
                for future in futures:
                    future.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        reads_attempted = len(futures)
        trace.content_read_count = existing_read_count + reads_attempted
        unreadable_query_ids = {
            query_id
            for query_id in attempted_query_ids
            if query_id and query_id not in readable_query_ids
        }
        return candidates, new_keys, reads_attempted, unreadable_query_ids

    def _invoke_planner(
        self,
        request: RetrievalRequest,
        decision: Any,
        retrieval_context: Any,
        freshness_context: Any,
        deadline: float,
    ) -> SearchPlan:
        return _call_with_supported_kwargs(
            self._planner.plan,
            request,
            decision,
            retrieval_context,
            freshness_context,
            deadline=deadline,
            timeout_seconds=self._remaining(deadline),
        )

    def _invoke_repair_planner(
        self,
        plan: SearchPlan,
        gap: Any,
        prior_fingerprints: Sequence[str],
        deadline: float,
    ) -> RepairPlan:
        return _call_with_supported_kwargs(
            self._planner.plan_repair,
            plan,
            gap,
            prior_fingerprints=prior_fingerprints,
            deadline=deadline,
            timeout_seconds=self._remaining(deadline),
        )

    def _degraded_plan(
        self,
        request: RetrievalRequest,
        decision: Any,
        retrieval_context: Any,
        freshness_context: Any,
    ) -> SearchPlan:
        class _UnavailableModel:
            def chat(self, *_args: Any, **_kwargs: Any) -> Any:
                raise TimeoutError("planner deadline expired")

        fallback = SearchPlanner(_UnavailableModel()).plan(
            request,
            decision,
            retrieval_context,
            freshness_context,
            deadline=self._monotonic(),
        )
        return replace(fallback, planning_status=PlanningStatus.DEGRADED)

    def _call_until_deadline(
        self,
        method: Callable[..., Any],
        deadline: float,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[bool, Any]:
        remaining = self._remaining(deadline)
        if remaining <= 0:
            return False, None
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search-bounded")
        future = executor.submit(method, *args, **kwargs)
        try:
            return True, future.result(timeout=remaining)
        except FuturesTimeoutError:
            future.cancel()
            return False, None
        except Exception:
            logger.debug("bounded retrieval stage failed", exc_info=True)
            return True, None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _record_query_attempts(
        self,
        trace: SearchTrace,
        query: SearchQuery,
        attempts: Sequence[ProviderAttempt],
    ) -> None:
        if not attempts:
            return
        for attempt in attempts:
            logged_attempt = replace(attempt, query_index=query.query_index)
            entry = QueryTraceEntry(
                query_index=query.query_index,
                purpose=query.purpose,
                round_kind=query.round_kind,
                provider=logged_attempt.provider,
                status=logged_attempt.status,
                latency_ms=logged_attempt.latency_ms,
            )
            if entry not in trace.executed_queries:
                trace.executed_queries = (*trace.executed_queries, entry)
            trace.provider_attempts = (*trace.provider_attempts, logged_attempt)
            if logged_attempt.invocation_started:
                trace.provider_invocation_started = True
            if logged_attempt.status is not ProviderStatus.SUCCESS:
                trace.provider_failures = (
                    *trace.provider_failures,
                    _failure_for_status(logged_attempt.status),
                )

    def _assembler_with_rejecting_judge(self) -> EvidenceAssembler:
        class _RejectingJudge:
            def judge(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
                return {}

        return EvidenceAssembler(_RejectingJudge())

    def _seal_bundle(
        self,
        bundle: EvidenceBundle,
        trace: SearchTrace,
        gap: Any,
        repair: RepairPlan,
        initial_canonical_urls: set[str],
    ) -> EvidenceBundle:
        initial_ids = tuple(
            item.evidence_id
            for item in bundle.evidence_items
            if item.canonical_url in initial_canonical_urls
        )
        return replace(
            bundle,
            request_id=trace.request_id,
            attempts=trace.provider_attempts,
            initial_evidence_ids=initial_ids,
            gap_analysis=gap,
            repair_plan=repair,
            retrieval_round_count=trace.retrieval_round_count,
        )

    def _finalize_bundle(
        self,
        bundle: EvidenceBundle,
        trace: SearchTrace,
        started: float,
        *,
        gap: Any = None,
        repair: RepairPlan | None = None,
        initial_canonical_urls: set[str] | None = None,
    ) -> EvidenceBundle:
        bundle = self._seal_bundle(
            bundle,
            trace,
            gap if gap is not None else bundle.gap_analysis,
            repair if repair is not None else bundle.repair_plan,
            initial_canonical_urls
            if initial_canonical_urls is not None
            else {item.canonical_url for item in bundle.evidence_items if item.canonical_url},
        )
        trace.provider_failures = tuple(dict.fromkeys(trace.provider_failures))
        trace.evidence_state = bundle.evidence_state
        self._record_evidence_trace(trace, bundle)
        trace.citable_evidence_count = sum(
            1 for item in bundle.evidence_items if item.citable
        )
        trace.retrieval_pipeline_latency_ms = self._elapsed_ms(started)
        return bundle

    @staticmethod
    def _record_evidence_trace(trace: SearchTrace, bundle: EvidenceBundle) -> None:
        """Project immutable Evidence into body-free Trace metadata only."""
        trace.supported_topic_ids = bundle.supported_topic_ids
        trace.missing_topic_ids = bundle.missing_topic_ids
        trace.topic_freshness = tuple(
            TopicFreshnessTraceEntry(assessment.topic_id, assessment.freshness)
            for assessment in bundle.topic_assessments
        )
        trace.judge_anomaly_codes = bundle.judge_anomaly_codes
        trace.judge_anomaly_count = bundle.judge_anomaly_count
        gap = bundle.gap_analysis
        if gap.repair_reason_codes:
            trace.repair_reason_codes = gap.repair_reason_codes
            trace.repair_target_topic_ids = gap.repair_target_topic_ids

    def _failure_result(
        self,
        decision: Any,
        plan: SearchPlan,
        trace: SearchTrace,
        started: float,
        *,
        analysis: RequestAnalysis,
        failure: SearchFailureCode,
        bundle: EvidenceBundle | None,
        limitation: str,
        gap: Any = None,
        repair: RepairPlan | None = None,
        initial_canonical_urls: set[str] | None = None,
    ) -> SearchPipelineResult:
        if failure in {
            SearchFailureCode.PROVIDER_NOT_CONFIGURED,
            SearchFailureCode.PROVIDER_UNAVAILABLE,
            SearchFailureCode.PROVIDER_TIMEOUT,
            SearchFailureCode.NO_RESULTS,
        }:
            trace.provider_failures = (*trace.provider_failures, failure)
        trace.provider_failures = tuple(dict.fromkeys(trace.provider_failures))
        trace.degradation_reason = failure

        if failure is SearchFailureCode.PROVIDER_NOT_CONFIGURED and bundle is None:
            trace.evidence_state = None
            trace.citable_evidence_count = 0
            trace.retrieval_pipeline_latency_ms = self._elapsed_ms(started)
            return SearchPipelineResult(
                decision=decision,
                plan=plan,
                evidence=None,
                trace=trace,
                failure_code=failure,
                analysis=analysis,
            )

        if bundle is None:
            bundle = _empty_bundle(
                plan,
                attempts=trace.provider_attempts,
                retrieval_round_count=trace.retrieval_round_count,
                limitation=limitation,
            )
        else:
            bundle = replace(
                bundle,
                limitations=tuple(dict.fromkeys((*bundle.limitations, limitation))),
            )
        bundle = self._finalize_bundle(
            bundle,
            trace,
            started,
            gap=gap,
            repair=repair,
            initial_canonical_urls=initial_canonical_urls,
        )
        return SearchPipelineResult(
            decision=decision,
            plan=plan,
            evidence=bundle,
            trace=trace,
            failure_code=failure,
            analysis=analysis,
        )

    def _timeout_result(
        self,
        decision: Any,
        plan: SearchPlan,
        trace: SearchTrace,
        started: float,
        *,
        analysis: RequestAnalysis,
        bundle: EvidenceBundle | None = None,
        gap: Any = None,
        repair: RepairPlan | None = None,
        initial_canonical_urls: set[str] | None = None,
    ) -> SearchPipelineResult:
        if trace.retrieval_stop_reason is None:
            trace.retrieval_stop_reason = RetrievalStopReason.BUDGET_EXHAUSTED
        return self._failure_result(
            decision,
            plan,
            trace,
            started,
            analysis=analysis,
            failure=SearchFailureCode.PROVIDER_TIMEOUT,
            bundle=bundle,
            limitation="hard_deadline_exceeded",
            gap=gap,
            repair=repair,
            initial_canonical_urls=initial_canonical_urls,
        )

    def _remaining(self, deadline: float) -> float:
        return max(deadline - self._monotonic(), 0.0)

    def _expired(self, deadline: float) -> bool:
        return self._remaining(deadline) <= 0

    def _assembler(self) -> EvidenceAssembler:
        return EvidenceAssembler(self._judge)

    def _monotonic(self) -> float:
        if self._clock is not None and hasattr(self._clock, "monotonic"):
            return self._clock.monotonic()
        return time.monotonic()

    def _elapsed_ms(self, started: float) -> int:
        return int((self._monotonic() - started) * 1000)


def _call_with_supported_kwargs(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    parameters = inspect.signature(method).parameters.values()
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    names = {parameter.name for parameter in parameters}
    supported = {
        key: value for key, value in kwargs.items() if accepts_kwargs or key in names
    }
    return method(*args, **supported)


def _readable_candidate(candidate: Any) -> bool:
    """A candidate is a readable-content acquisition only when the page or
    document was actually read; provider snippets and fetch failures are not."""
    if candidate is None:
        return False
    return (
        getattr(candidate, "excerpt_origin", None)
        in {ExcerptOrigin.PAGE_EXTRACT, ExcerptOrigin.DOCUMENT_EXTRACT}
        or getattr(candidate, "extraction_status", None) == "provider_raw_content"
    )


def _unreadable_topic_ids(
    plan: SearchPlan,
    unreadable_query_ids: Sequence[str],
    *,
    additional_queries: Sequence[SearchQuery] = (),
) -> tuple[str, ...]:
    """Map Reader-failed queries to their material target topic IDs."""
    if not unreadable_query_ids:
        return ()
    material_ids = {
        topic.topic_id for topic in plan.required_topics if topic.material
    }
    queries = {
        query.query_id: query
        for query in (*plan.initial_queries, *additional_queries)
    }
    topic_ids: list[str] = []
    for query_id in unreadable_query_ids:
        query = queries.get(query_id)
        if query is None:
            continue
        for topic_id in query.target_topic_ids:
            if topic_id in material_ids and topic_id not in topic_ids:
                topic_ids.append(topic_id)
    return tuple(topic_ids)


def _repair_gates_pass(
    *,
    route_standard: bool,
    repair_used: bool,
    semantic_query_count: int,
    candidate_url_count: int,
    content_read_count: int,
    retrieval_round_count: int,
    remaining_seconds: float,
    budget: Any,
) -> bool:
    """Deterministic, program-only pre-repair budget gate."""
    return (
        route_standard
        and not repair_used
        and semantic_query_count < budget.max_total_queries
        and candidate_url_count < budget.max_candidate_urls
        and content_read_count < budget.max_content_reads
        and retrieval_round_count < budget.max_retrieval_rounds
        and remaining_seconds > 0
    )


def _repair_stop_reason(
    bundle: EvidenceBundle,
    gap: Any,
    repair: RepairPlan,
    gates_pass: bool,
    route_standard: bool,
) -> RetrievalStopReason:
    if bundle.evidence_state is EvidenceState.SUFFICIENT:
        return RetrievalStopReason.EVIDENCE_SUFFICIENT
    if not gap.repair_eligible:
        return RetrievalStopReason.NO_REPAIR_BENEFIT
    if not route_standard:
        return RetrievalStopReason.NO_REPAIR_BENEFIT
    if not gates_pass:
        return RetrievalStopReason.BUDGET_EXHAUSTED
    if not repair.triggered:
        return RetrievalStopReason.NO_REPAIR_BENEFIT
    return RetrievalStopReason.BUDGET_EXHAUSTED


def _query_for_hit(
    plan: SearchPlan,
    hit: Any,
    *,
    additional_queries: Sequence[SearchQuery] = (),
) -> SearchQuery:
    for query in (*plan.initial_queries, *additional_queries):
        if query.query_id == hit.query_id:
            return query
    return plan.initial_queries[0]


def _failure_for_status(status: ProviderStatus) -> SearchFailureCode:
    from src.search.models import PROVIDER_STATUS_FAILURE_CODES
    return PROVIDER_STATUS_FAILURE_CODES.get(status, SearchFailureCode.PROVIDER_UNAVAILABLE)


def _failure_for_state(state: EvidenceState) -> SearchFailureCode | None:
    return {
        EvidenceState.PARTIAL: SearchFailureCode.PARTIAL_EVIDENCE,
        EvidenceState.CONFLICTING: SearchFailureCode.SOURCE_CONFLICT,
        EvidenceState.INSUFFICIENT: SearchFailureCode.INSUFFICIENT_EVIDENCE,
    }.get(state)


def _limitation_for_failure(failure: SearchFailureCode) -> str:
    return {
        SearchFailureCode.PROVIDER_NOT_CONFIGURED: "provider_not_configured",
        SearchFailureCode.PROVIDER_UNAVAILABLE: "provider_unavailable",
        SearchFailureCode.PROVIDER_TIMEOUT: "hard_deadline_exceeded",
        SearchFailureCode.NO_RESULTS: "no_results",
        SearchFailureCode.CONTENT_UNREADABLE: "content_unreadable",
    }.get(failure, "retrieval_failure")


def _empty_bundle(
    plan: SearchPlan,
    *,
    attempts: Sequence[ProviderAttempt] = (),
    retrieval_round_count: int = 1,
    limitation: str = "provider_failure",
) -> EvidenceBundle:
    from src.search.models import (
        EvidenceGapAnalysis,
        FreshnessEligibility,
        FreshnessRequirement,
        RepairPlan,
        TopicAssessment,
    )
    material_topics = tuple(topic for topic in plan.required_topics if topic.material)
    missing = tuple(topic.label for topic in material_topics)
    missing_topic_ids = tuple(topic.topic_id for topic in material_topics)
    assessments = tuple(
        TopicAssessment(
            topic.topic_id,
            FreshnessEligibility.NOT_REQUIRED
            if topic.freshness_requirement is FreshnessRequirement.NOT_REQUIRED
            else FreshnessEligibility.UNKNOWN,
            (),
        )
        for topic in material_topics
    )
    return EvidenceBundle(
        request_id="req-empty",
        decision=plan.decision,
        plan=plan,
        attempts=tuple(attempts),
        initial_evidence_ids=(),
        gap_analysis=EvidenceGapAnalysis(missing_topic_ids, (), False, (), ()),
        repair_plan=RepairPlan(False, (), (), None),
        retrieval_round_count=retrieval_round_count,
        evidence_items=(),
        evidence_state=EvidenceState.INSUFFICIENT,
        missing_claim_topics=missing,
        weak_source_topics=(),
        conflict_groups=(),
        limitations=(limitation,),
        topic_assessments=assessments,
        supported_topic_ids=(),
        missing_topic_ids=missing_topic_ids,
    )


def finalize_search_trace(trace: SearchTrace, *, response_finished_at: float) -> None:
    """Fill the end-to-end total latency and log the body-free Trace exactly
    once. The caller invokes it only after validation/rendering complete."""
    if trace.finalized:
        return
    trace.finalized = True
    trace.response_finished_at = response_finished_at
    started_at = trace.response_started_at
    if (
        isinstance(started_at, (int, float))
        and isinstance(response_finished_at, (int, float))
        and response_finished_at >= started_at
    ):
        trace.total_response_latency_ms = (
            response_finished_at - started_at
        ) * 1000.0
    try:
        logger.info("search trace final: %s", trace.to_log_dict())
    except Exception:
        logger.debug("failed to serialize search trace", exc_info=True)


# ── lazy singleton graph ────────────────────────────────────────────────

_orchestrator: SearchOrchestrator | None = None


def get_search_orchestrator() -> SearchOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = _build_production_orchestrator()
    return _orchestrator


def reset_search_orchestrator() -> None:
    global _orchestrator
    _orchestrator = None


def _build_production_orchestrator() -> SearchOrchestrator:
    from src.services.llm_client import get_llm_client

    llm = get_llm_client()
    request_analyzer = LLMRequestAnalyzer(llm)
    router = RetrievalBenefitRouter()
    planner = SearchPlanner(llm)
    judge = LLMEvidenceJudge(llm)
    providers: list[Any] = [
        DDGSSearchProvider(
            proxy_url=config.proxy_url,
            timeout_seconds=config.request_timeout,
        )
    ]
    if config.tavily_api_key:
        providers.append(
            TavilySearchProvider(
                api_key=config.tavily_api_key,
                proxy_url=config.proxy_url,
            )
        )
    return SearchOrchestrator(
        request_analyzer=request_analyzer,
        router=router,
        planner=planner,
        judge=judge,
        providers=providers,
        extractor=SearchExtractor(),
    )
