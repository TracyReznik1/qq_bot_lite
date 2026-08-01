"""Bounded two-stage search orchestrator with exact Trace timings."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    SearchRoundKind,
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
        deadline = self._monotonic() + budget.hard_timeout_seconds

        trace.orchestrator_started = True
        plan_started = self._monotonic()
        plan = self._planner.plan(request, decision, deadline=deadline)
        trace.query_planning_latency_ms = self._elapsed_ms(plan_started)
        trace.initial_query_count = len(plan.initial_queries)
        trace.executed_queries = _query_metadata(plan.initial_queries)

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
            evidence = None if failure is SearchFailureCode.PROVIDER_NOT_CONFIGURED else _empty_bundle(plan)
            return SearchPipelineResult(
                decision=decision,
                plan=plan,
                evidence=evidence,
                trace=trace,
                failure_code=failure,
            )

        content_started = self._monotonic()
        candidates, candidate_url_count, reads = self._extract_candidates(
            plan, provider_results, budget, deadline, trace,
        )
        trace.initial_content_read_latency_ms = self._elapsed_ms(content_started)
        trace.content_read_total_latency_ms = trace.initial_content_read_latency_ms
        trace.candidate_url_count = candidate_url_count

        evidence_started = self._monotonic()
        bundle = self._assembler().assemble(plan, candidates)
        trace.initial_evidence_assembly_latency_ms = self._elapsed_ms(evidence_started)
        trace.evidence_assembly_total_latency_ms = trace.initial_evidence_assembly_latency_ms
        trace.citable_evidence_count = sum(1 for item in bundle.evidence_items if item.citable)
        trace.evidence_state = bundle.evidence_state

        if not bundle.evidence_items and any(
            result.status is ProviderStatus.SUCCESS and result.hits
            for result in provider_results
        ):
            trace.degradation_reason = SearchFailureCode.CONTENT_UNREADABLE
            return SearchPipelineResult(
                decision=decision,
                plan=plan,
                evidence=_empty_bundle(plan),
                trace=trace,
                failure_code=SearchFailureCode.CONTENT_UNREADABLE,
            )

        gap_started = self._monotonic()
        gap = self._assembler().analyze_gap(plan, bundle)
        trace.gap_analysis_latency_ms = self._elapsed_ms(gap_started)

        repair_already_planned = False
        if _repair_allowed(plan, decision) and gap.repair_eligible:
            repair_started = self._monotonic()
            repair = self._planner.plan_repair(plan, gap, repair_already_planned=repair_already_planned)
            if repair.triggered and repair.repair_query is not None:
                repair_already_planned = True
                trace.adaptive_repair_round_started = True
                trace.adaptive_repair_query = (repair.repair_query.query_id, repair.repair_query.purpose)
                trace.retrieval_round_count = 2
                repair_result = self._run_repair_query(
                    repair,
                    decision,
                    budget,
                    deadline,
                    trace,
                )
                repair_candidates, more_urls, more_reads = self._extract_candidates(
                    plan,
                    repair_result,
                    budget,
                    deadline,
                    trace,
                    round_kind=SearchRoundKind.REPAIR,
                    existing_candidate_count=len(candidates),
                    existing_read_count=reads,
                )
                candidate_url_count = min(candidate_url_count + more_urls, budget.max_candidate_urls)
                reads = min(reads + more_reads, budget.max_content_reads)
                trace.candidate_url_count = candidate_url_count
                trace.content_read_count = reads
                bundle = self._assembler().assemble(plan, (*candidates, *repair_candidates))
                trace.evidence_assembly_total_latency_ms += self._elapsed_ms(repair_started)
                trace.evidence_state = bundle.evidence_state
                trace.repair_used = True
            trace.adaptive_repair_latency_ms = self._elapsed_ms(repair_started)

        trace.evidence_state = bundle.evidence_state
        trace.citable_evidence_count = sum(1 for item in bundle.evidence_items if item.citable)
        trace.retrieval_pipeline_latency_ms = self._elapsed_ms(plan_started)

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
        results: list[ProviderResult] = []
        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
            futures = {}
            for query in queries:
                if self._monotonic() >= deadline:
                    trace.provider_failures = (*trace.provider_failures, SearchFailureCode.PROVIDER_TIMEOUT)
                    break
                remaining = max(deadline - self._monotonic(), 0.0)
                future = executor.submit(
                    self._search_one,
                    query,
                    decision,
                    min(config.request_timeout, remaining),
                    trace,
                )
                futures[future] = query
            for future in as_completed(futures, timeout=max(deadline - self._monotonic(), 0.01)):
                query = futures[future]
                try:
                    result = future.result()
                except Exception:
                    logger.debug("query dispatch failed for %s", query.query_id, exc_info=True)
                    result = ProviderResult(
                        provider="tavily",
                        status=ProviderStatus.ERROR,
                        hits=(),
                        latency_ms=0,
                    )
                if mark_repair:
                    trace.executed_queries = (*trace.executed_queries, _query_metadata((query,)))
                results.append(result)
            # Cancel any pending futures without waiting past the deadline.
            executor.shutdown(wait=False, cancel_futures=True)
        return results

    def _search_one(
        self,
        query: SearchQuery,
        decision: Any,
        timeout: float,
        trace: SearchTrace,
    ) -> ProviderResult:
        max_results = max(int(config.search_max_results or 1), 1)
        started = self._monotonic()
        result = self._registry.search(
            query,
            tier=decision.route,
            max_results=max_results,
            timeout_seconds=timeout,
        )
        latency = self._elapsed_ms(started)
        for attempt in getattr(self._registry, "last_attempts", ()) or ():
            status = getattr(attempt, "status", ProviderStatus.ERROR)
            provider = getattr(attempt, "provider", "unknown")
            trace.provider_attempts = (*trace.provider_attempts, ProviderAttempt(provider, status, 1, latency))
            trace.provider_invocation_started = True
            if status is not ProviderStatus.SUCCESS:
                trace.provider_failures = (*trace.provider_failures, _failure_for_status(status))
        return result

    def _extract_candidates(
        self,
        plan: SearchPlan,
        results: Sequence[ProviderResult],
        budget: Any,
        deadline: float,
        trace: SearchTrace,
        *,
        round_kind: SearchRoundKind = SearchRoundKind.INITIAL,
        existing_candidate_count: int = 0,
        existing_read_count: int = 0,
    ) -> tuple[list[Any], int, int]:
        hits: list[Any] = []
        for result in results:
            if result.status is ProviderStatus.SUCCESS:
                hits.extend(result.hits)
        candidate_url_total = min(existing_candidate_count + len(hits), budget.max_candidate_urls)
        hits = hits[: max(budget.max_candidate_urls - existing_candidate_count, 0)]

        candidates: list[Any] = []
        reads = existing_read_count
        for hit in hits:
            if self._monotonic() >= deadline:
                break
            if reads >= budget.max_content_reads:
                break
            remaining = max(deadline - self._monotonic(), 0.0)
            try:
                candidate = self._extractor.extract(
                    hit,
                    _query_for_hit(plan, hit, round_kind=round_kind),
                    allow_network_read=True,
                    timeout_seconds=min(config.request_timeout, remaining),
                )
            except Exception:
                logger.debug("extraction failed for %s", hit.url, exc_info=True)
                continue
            if candidate is not None and candidate.excerpt:
                candidates.append(candidate)
                reads += candidate.content_reads_consumed
        trace.content_read_count = reads
        return candidates, candidate_url_total, reads

    def _assembler(self) -> EvidenceAssembler:
        return EvidenceAssembler(self._judge)

    def _monotonic(self) -> float:
        if self._clock is not None and hasattr(self._clock, "monotonic"):
            return self._clock.monotonic()
        return time.monotonic()

    def _elapsed_ms(self, started: float) -> int:
        return int((self._monotonic() - started) * 1000)


def _query_metadata(queries: Sequence[SearchQuery]) -> tuple[tuple[str, object], ...]:
    return tuple((query.query_id, query.purpose) for query in queries)


def _candidate_url_count(results: Sequence[ProviderResult]) -> int:
    return sum(len(result.hits) for result in results if result.status is ProviderStatus.SUCCESS)


def _repair_allowed(plan: SearchPlan, decision: Any) -> bool:
    del plan
    return decision.route in {SearchTier.STANDARD, SearchTier.DEEP}


def _query_for_hit(
    plan: SearchPlan,
    hit: Any,
    *,
    round_kind: SearchRoundKind = SearchRoundKind.INITIAL,
) -> SearchQuery:
    if round_kind is SearchRoundKind.REPAIR:
        if hit.query_id and hit.query_id.startswith("repair"):
            return plan.initial_queries[0]
        return plan.initial_queries[0]
    for query in plan.initial_queries:
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


def _empty_bundle(plan: SearchPlan) -> EvidenceBundle:
    from src.search.models import EvidenceGapAnalysis, RepairPlan
    return EvidenceBundle(
        request_id="req-empty",
        decision=plan.decision,
        plan=plan,
        attempts=(),
        initial_evidence_ids=(),
        gap_analysis=EvidenceGapAnalysis((), (), False, None, ()),
        repair_plan=RepairPlan(False, (), None),
        retrieval_round_count=1,
        evidence_items=(),
        evidence_state=EvidenceState.INSUFFICIENT,
        missing_claim_topics=tuple(plan.required_topics),
        weak_source_topics=(),
        conflict_groups=(),
        limitations=("provider_failure",),
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
