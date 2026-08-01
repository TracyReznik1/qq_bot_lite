"""Bounded two-stage search orchestrator with exact Trace timings."""

from __future__ import annotations

import logging
import inspect
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import replace
from typing import Any, Callable, Sequence

from src.config import config
from src.search.evidence import EvidenceAssembler, LLMEvidenceJudge
from src.search.extraction import SearchExtractor
from src.search.models import (
    DEFAULT_TIER_BUDGETS,
    EvidenceBundle,
    EvidenceState,
    PlanningStatus,
    ProviderAttempt,
    ProviderResult,
    ProviderStatus,
    RepairPlan,
    RequestSource,
    RetrievalRequest,
    SearchFailureCode,
    SearchPipelineResult,
    SearchPlan,
    SearchQuery,
    SearchTier,
    SearchTrace,
    SkipReason,
    TriggerCode,
)
from src.search.planner import SearchPlanner
from src.search.providers import DDGSSearchProvider, ProviderRegistry, TavilySearchProvider
from src.search.router import LLMRoutingAdvisor, RetrievalBenefitRouter

logger = logging.getLogger("qq-bot")


class SearchOrchestrator:
    """Run the route -> plan -> retrieve -> evidence state machine."""

    def __init__(
        self,
        *,
        router: Any,
        planner: Any,
        judge: Any,
        providers: Sequence[Any] | None = None,
        extractor: Any = None,
        clock: Any = None,
    ) -> None:
        self._router = router
        self._planner = planner
        self._judge = judge
        self._providers = providers if providers is not None else ()
        self._extractor = extractor if extractor is not None else SearchExtractor()
        self._clock = clock
        self._registry = ProviderRegistry(self._providers)

    def run(self, request: RetrievalRequest) -> SearchPipelineResult:
        started = self._monotonic()
        trace = SearchTrace(
            request_id=f"req-{int(started * 1000) % 1000000}",
            request_source=request.request_source,
            route=SearchTier.LIGHT,
        )
        decision = self._router.decide(request)
        trace.route = decision.route
        trace.skip_reason = decision.skip_reason
        trace.trigger_codes = decision.trigger_codes
        trace.factuality = decision.factuality
        trace.external_fact_required = decision.external_fact_required
        trace.program_minimum_tier = decision.program_minimum_tier
        trace.final_tier = decision.route
        trace.route_latency_ms = self._elapsed_ms(started)

        if decision.route is SearchTier.SKIP:
            if decision.requires_clarification:
                trace.degradation_reason = SearchFailureCode.USER_FORBID_WEB
            result = SearchPipelineResult(
                decision=decision,
                plan=None,
                evidence=None,
                trace=trace,
                failure_code=(
                    SearchFailureCode.USER_FORBID_WEB
                    if decision.skip_reason is SkipReason.USER_FORBID_WEB
                    else None
                ),
            )
            return result

        # One monotonic hard deadline per tier starts before planning, so
        # planner and Evidence-judge latency count toward the retrieval budget.
        budget = DEFAULT_TIER_BUDGETS[decision.route]
        deadline = self._monotonic() + float(budget.hard_timeout_seconds)

        trace.orchestrator_started = True
        plan_started = self._monotonic()
        plan_completed, plan = self._call_until_deadline(
            self._invoke_planner,
            deadline,
            request,
            decision,
            deadline,
        )
        if not plan_completed or not isinstance(plan, SearchPlan):
            plan = self._degraded_plan(request, decision)
        trace.query_planning_latency_ms = self._elapsed_ms(plan_started)
        trace.initial_query_count = len(plan.initial_queries)
        trace.initial_query_redaction_codes = plan.query_redaction_codes

        if self._expired(deadline):
            return self._timeout_result(decision, plan, trace, started)

        trace.provider_configured = any(
            provider.readiness().configured for provider in self._providers
        )

        initial_round_started = self._monotonic()
        trace.initial_round_started = True
        trace.retrieval_round_count = 1

        provider_results = self._run_initial_batch(
            plan,
            decision,
            budget,
            deadline,
            trace,
        )
        trace.initial_provider_search_latency_ms = self._elapsed_ms(initial_round_started)
        trace.provider_search_total_latency_ms = trace.initial_provider_search_latency_ms

        if self._expired(deadline):
            return self._timeout_result(decision, plan, trace, started)

        if not provider_results or all(
            result.status in {ProviderStatus.NOT_CONFIGURED, ProviderStatus.UNAVAILABLE, ProviderStatus.ERROR, ProviderStatus.TIMEOUT, ProviderStatus.EMPTY}
            for result in provider_results
        ):
            status = next(
                (result.status for result in provider_results),
                ProviderStatus.NOT_CONFIGURED,
            )
            failure = _failure_for_status(status)
            trace.degradation_reason = failure
            evidence = None if failure is SearchFailureCode.PROVIDER_NOT_CONFIGURED else _empty_bundle(
                plan,
                attempts=trace.provider_attempts,
                retrieval_round_count=trace.retrieval_round_count,
            )
            trace.retrieval_pipeline_latency_ms = self._elapsed_ms(started)
            return SearchPipelineResult(
                decision=decision,
                plan=plan,
                evidence=evidence,
                trace=trace,
                failure_code=failure,
            )

        content_started = self._monotonic()
        candidates, candidate_keys, reads = self._extract_candidates(
            plan, provider_results, budget, deadline, trace,
        )
        trace.initial_content_read_latency_ms = self._elapsed_ms(content_started)
        trace.content_read_total_latency_ms = trace.initial_content_read_latency_ms
        trace.candidate_url_count = len(candidate_keys)

        if self._expired(deadline):
            return self._timeout_result(decision, plan, trace, started)

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

        if self._expired(deadline):
            return self._timeout_result(decision, plan, trace, started, bundle=bundle)

        if not bundle.evidence_items and any(
            result.status is ProviderStatus.SUCCESS and result.hits
            for result in provider_results
        ):
            trace.degradation_reason = SearchFailureCode.CONTENT_UNREADABLE
            return SearchPipelineResult(
                decision=decision,
                plan=plan,
                evidence=_empty_bundle(
                    plan,
                    attempts=trace.provider_attempts,
                    retrieval_round_count=trace.retrieval_round_count,
                ),
                trace=trace,
                failure_code=SearchFailureCode.CONTENT_UNREADABLE,
            )

        gap_started = self._monotonic()
        gap = self._assembler().analyze_gap(plan, bundle)
        trace.gap_analysis_latency_ms = self._elapsed_ms(gap_started)
        initial_canonical_urls = {
            item.canonical_url for item in bundle.evidence_items if item.canonical_url
        }
        repair = RepairPlan(False, gap.repair_reason_codes, None)

        repair_already_planned = False
        if _repair_allowed(plan, decision) and gap.repair_eligible and not self._expired(deadline):
            repair_started = self._monotonic()
            repair_completed, planned_repair = self._call_until_deadline(
                self._invoke_repair_planner,
                deadline,
                plan,
                gap,
                repair_already_planned,
                deadline,
            )
            if repair_completed and isinstance(planned_repair, RepairPlan):
                repair = planned_repair
            if repair.triggered and repair.repair_query is not None:
                repair_already_planned = True
                trace.adaptive_repair_round_started = True
                trace.adaptive_repair_query = repair.repair_query
                trace.adaptive_repair_redaction_codes = repair.query_redaction_codes
                trace.retrieval_round_count = 2
                repair_result = self._run_repair_query(
                    repair,
                    decision,
                    budget,
                    deadline,
                    trace,
                )
                repair_provider_finished = self._monotonic()
                trace.provider_search_total_latency_ms += self._elapsed_ms(repair_started)
                repair_candidates, repair_keys, more_reads = self._extract_candidates(
                    plan,
                    repair_result,
                    budget,
                    deadline,
                    trace,
                    additional_queries=(repair.repair_query,),
                    existing_candidate_keys=candidate_keys,
                    existing_read_count=reads,
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
                trace.repair_used = True
            trace.adaptive_repair_latency_ms = self._elapsed_ms(repair_started)

        if self._expired(deadline):
            return self._timeout_result(
                decision,
                plan,
                trace,
                started,
                bundle=bundle,
                gap=gap,
                repair=repair,
                initial_canonical_urls=initial_canonical_urls,
            )

        final_gap = self._assembler().analyze_gap(plan, bundle)
        bundle = self._seal_bundle(
            bundle,
            trace,
            final_gap,
            repair,
            initial_canonical_urls,
        )

        trace.evidence_state = bundle.evidence_state
        trace.citable_evidence_count = sum(1 for item in bundle.evidence_items if item.citable)
        trace.retrieval_pipeline_latency_ms = self._elapsed_ms(started)

        failure_code = _failure_for_state(bundle.evidence_state)
        return SearchPipelineResult(
            decision=decision,
            plan=plan,
            evidence=bundle,
            trace=trace,
            failure_code=failure_code,
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
        return self._dispatch_queries(queries, decision, deadline, trace)

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
        result = self._dispatch_queries((query,), decision, deadline, trace, mark_repair=True)
        return result

    def _dispatch_queries(
        self,
        queries: Sequence[SearchQuery],
        decision: RetrievalDecision,
        deadline: float,
        trace: SearchTrace,
        *,
        mark_repair: bool = False,
    ) -> list[ProviderResult]:
        del mark_repair
        results: list[ProviderResult] = []
        if not queries:
            return results
        dispatch_started = self._monotonic()
        executor = ThreadPoolExecutor(max_workers=min(len(queries), 4), thread_name_prefix="search-provider")
        futures: dict[Any, SearchQuery] = {}
        has_available_provider = any(
            provider.readiness().available for provider in self._providers
        )
        try:
            for query in queries:
                remaining = self._remaining(deadline)
                if remaining <= 0:
                    break
                trace.executed_queries = (*trace.executed_queries, (query.query_id, query.purpose))
                if has_available_provider:
                    trace.provider_invocation_started = True
                future = executor.submit(
                    self._search_one,
                    query,
                    decision,
                    min(config.request_timeout, remaining),
                )
                futures[future] = query
            try:
                for future in as_completed(futures, timeout=self._remaining(deadline)):
                    query = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        logger.debug("query dispatch failed for %s", query.query_id, exc_info=True)
                        result = ProviderResult("tavily", ProviderStatus.ERROR, (), 0)
                    self._record_provider_result(trace, result)
                    results.append(result)
            except FuturesTimeoutError:
                trace.provider_failures = (*trace.provider_failures, SearchFailureCode.PROVIDER_TIMEOUT)
                for future, query in futures.items():
                    if future.done():
                        continue
                    future.cancel()
                    provider = self._first_available_provider_name()
                    timeout_attempt = ProviderAttempt(
                        provider,
                        ProviderStatus.TIMEOUT,
                        1,
                        self._elapsed_ms(dispatch_started),
                        query_id=query.query_id,
                    )
                    trace.provider_attempts = (*trace.provider_attempts, timeout_attempt)
                results.append(ProviderResult("tavily", ProviderStatus.TIMEOUT, (), 0))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return results

    def _search_one(
        self,
        query: SearchQuery,
        decision: Any,
        timeout: float,
    ) -> ProviderResult:
        max_results = max(int(config.search_max_results or 1), 1)
        return self._registry.search(
            query,
            tier=decision.route,
            max_results=max_results,
            timeout_seconds=timeout,
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
    ) -> tuple[list[Any], set[str], int]:
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
        remaining_read_budget = max(budget.max_content_reads - existing_read_count, 0)
        read_hits = unique_hits[:remaining_read_budget]
        if not read_hits:
            return candidates, new_keys, 0
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
            except FuturesTimeoutError:
                trace.provider_failures = (*trace.provider_failures, SearchFailureCode.PROVIDER_TIMEOUT)
                for future in futures:
                    future.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        reads_attempted = len(futures)
        trace.content_read_count = existing_read_count + reads_attempted
        return candidates, new_keys, reads_attempted

    def _invoke_planner(
        self,
        request: RetrievalRequest,
        decision: Any,
        deadline: float,
    ) -> SearchPlan:
        return _call_with_supported_kwargs(
            self._planner.plan,
            request,
            decision,
            deadline=deadline,
            timeout_seconds=self._remaining(deadline),
        )

    def _invoke_repair_planner(
        self,
        plan: SearchPlan,
        gap: Any,
        repair_already_planned: bool,
        deadline: float,
    ) -> RepairPlan:
        return _call_with_supported_kwargs(
            self._planner.plan_repair,
            plan,
            gap,
            repair_already_planned=repair_already_planned,
            deadline=deadline,
            timeout_seconds=self._remaining(deadline),
        )

    def _degraded_plan(self, request: RetrievalRequest, decision: Any) -> SearchPlan:
        class _UnavailableModel:
            def chat(self, *_args: Any, **_kwargs: Any) -> Any:
                raise TimeoutError("planner deadline expired")

        fallback = SearchPlanner(_UnavailableModel()).plan(
            request,
            decision,
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

    def _record_provider_result(self, trace: SearchTrace, result: Any) -> None:
        for attempt in getattr(result, "attempts", ()):
            trace.provider_attempts = (*trace.provider_attempts, attempt)
            if attempt.invocation_started:
                trace.provider_invocation_started = True
            if attempt.status is not ProviderStatus.SUCCESS:
                trace.provider_failures = (
                    *trace.provider_failures,
                    _failure_for_status(attempt.status),
                )

    def _first_available_provider_name(self) -> str:
        for provider in self._providers:
            if provider.readiness().available:
                return provider.name
        return "tavily"

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
            attempts=trace.provider_attempts,
            initial_evidence_ids=initial_ids,
            gap_analysis=gap,
            repair_plan=repair,
            retrieval_round_count=trace.retrieval_round_count,
        )

    def _timeout_result(
        self,
        decision: Any,
        plan: SearchPlan,
        trace: SearchTrace,
        started: float,
        *,
        bundle: EvidenceBundle | None = None,
        gap: Any = None,
        repair: RepairPlan | None = None,
        initial_canonical_urls: set[str] | None = None,
    ) -> SearchPipelineResult:
        trace.provider_failures = (*trace.provider_failures, SearchFailureCode.PROVIDER_TIMEOUT)
        trace.degradation_reason = SearchFailureCode.PROVIDER_TIMEOUT
        trace.retrieval_pipeline_latency_ms = self._elapsed_ms(started)
        if bundle is None:
            bundle = _empty_bundle(
                plan,
                attempts=trace.provider_attempts,
                retrieval_round_count=trace.retrieval_round_count,
                limitation="hard_deadline_exceeded",
            )
        else:
            bundle = replace(
                bundle,
                evidence_state=EvidenceState.INSUFFICIENT,
                missing_claim_topics=tuple(plan.required_topics) or ("material_claim",),
                limitations=tuple(dict.fromkeys((*bundle.limitations, "hard_deadline_exceeded"))),
            )
            bundle = self._seal_bundle(
                bundle,
                trace,
                gap if gap is not None else bundle.gap_analysis,
                repair if repair is not None else bundle.repair_plan,
                initial_canonical_urls
                if initial_canonical_urls is not None
                else {item.canonical_url for item in bundle.evidence_items if item.canonical_url},
            )
        trace.evidence_state = EvidenceState.INSUFFICIENT
        trace.citable_evidence_count = 0
        return SearchPipelineResult(
            decision=decision,
            plan=plan,
            evidence=bundle,
            trace=trace,
            failure_code=SearchFailureCode.PROVIDER_TIMEOUT,
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


def _repair_allowed(plan: SearchPlan, decision: Any) -> bool:
    del plan
    return decision.route in {SearchTier.STANDARD, SearchTier.DEEP}


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


def _empty_bundle(
    plan: SearchPlan,
    *,
    attempts: Sequence[ProviderAttempt] = (),
    retrieval_round_count: int = 1,
    limitation: str = "provider_failure",
) -> EvidenceBundle:
    from src.search.models import EvidenceGapAnalysis, RepairPlan
    return EvidenceBundle(
        request_id="req-empty",
        decision=plan.decision,
        plan=plan,
        attempts=tuple(attempts),
        initial_evidence_ids=(),
        gap_analysis=EvidenceGapAnalysis((), (), False, None, ()),
        repair_plan=RepairPlan(False, (), None),
        retrieval_round_count=retrieval_round_count,
        evidence_items=(),
        evidence_state=EvidenceState.INSUFFICIENT,
        missing_claim_topics=tuple(plan.required_topics),
        weak_source_topics=(),
        conflict_groups=(),
        limitations=(limitation,),
    )


def finalize_search_trace(trace: SearchTrace, *, response_finished_at: float) -> None:
    """Fill the end-to-end total latency and log the body-free Trace exactly
    once. The caller invokes it only after validation/rendering complete."""
    if getattr(trace, "_logged", False):
        return
    if trace.total_response_latency_ms == 0 and response_finished_at >= 0:
        trace.total_response_latency_ms = int(response_finished_at * 1000)
    try:
        logger.info("search trace final: %s", trace.to_log_dict())
    except Exception:
        logger.debug("failed to serialize search trace", exc_info=True)
    trace._logged = True


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
    from src.search.router import LLMRoutingAdvisor

    llm = get_llm_client()
    router = RetrievalBenefitRouter(LLMRoutingAdvisor(llm))
    planner = SearchPlanner(llm)
    judge = LLMEvidenceJudge(llm)
    providers: list[Any] = []
    if config.tavily_api_key:
        providers.append(
            TavilySearchProvider(
                api_key=config.tavily_api_key,
                proxy_url=config.proxy_url,
            )
        )
    providers.append(
        DDGSSearchProvider(
            proxy_url=config.proxy_url,
            timeout_seconds=config.request_timeout,
        )
    )
    return SearchOrchestrator(
        router=router,
        planner=planner,
        judge=judge,
        providers=providers,
        extractor=SearchExtractor(),
    )
