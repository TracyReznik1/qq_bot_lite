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
            self._finalize_trace(result.trace)
            return result

        trace.orchestrator_started = True
        plan_started = self._monotonic()
        plan = self._planner.plan(request, decision)
        trace.query_planning_latency_ms = self._elapsed_ms(plan_started)
        trace.initial_query_count = len(plan.initial_queries)
        trace.executed_queries = _query_metadata(plan.initial_queries)

        budget = DEFAULT_TIER_BUDGETS[decision.route]
        deadline = self._monotonic() + budget.hard_timeout_seconds
        trace.provider_configured = any(
            provider.readiness().available for provider in self._providers
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
            result = SearchPipelineResult(
                decision=decision,
                plan=plan,
                evidence=evidence,
                trace=trace,
                failure_code=failure,
            )
            self._finalize_trace(trace)
            return result

        content_started = self._monotonic()
        candidates = self._extract_candidates(plan, provider_results, budget, deadline, trace)
        trace.initial_content_read_latency_ms = self._elapsed_ms(content_started)
        trace.content_read_total_latency_ms = trace.initial_content_read_latency_ms
        trace.candidate_url_count = _candidate_url_count(provider_results)

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
            result = SearchPipelineResult(
                decision=decision,
                plan=plan,
                evidence=_empty_bundle(plan),
                trace=trace,
                failure_code=SearchFailureCode.CONTENT_UNREADABLE,
            )
            self._finalize_trace(trace)
            return result

        gap_started = self._monotonic()
        gap = self._assembler().analyze_gap(plan, bundle)
        trace.gap_analysis_latency_ms = self._elapsed_ms(gap_started)

        if _repair_allowed(plan, decision) and gap.repair_eligible:
            repair_started = self._monotonic()
            repair = self._planner.plan_repair(plan, gap)
            if repair.triggered and repair.repair_query is not None:
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
                repair_candidates = self._extract_candidates(
                    plan,
                    repair_result,
                    budget,
                    deadline,
                    trace,
                    round_kind=SearchRoundKind.REPAIR,
                )
                bundle = self._assembler().assemble(plan, (*candidates, *repair_candidates))
                trace.evidence_assembly_total_latency_ms += self._elapsed_ms(repair_started)
                trace.evidence_state = bundle.evidence_state
                trace.repair_used = True
            trace.adaptive_repair_latency_ms = self._elapsed_ms(repair_started)

        trace.evidence_state = bundle.evidence_state
        trace.citable_evidence_count = sum(1 for item in bundle.evidence_items if item.citable)
        trace.retrieval_pipeline_latency_ms = self._elapsed_ms(plan_started)

        failure_code = _failure_for_state(bundle.evidence_state)
        result = SearchPipelineResult(
            decision=decision,
            plan=plan,
            evidence=bundle,
            trace=trace,
            failure_code=failure_code,
        )
        self._finalize_trace(trace)
        return result

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
        for query in queries:
            if self._monotonic() >= deadline:
                trace.provider_failures = (*trace.provider_failures, SearchFailureCode.PROVIDER_TIMEOUT)
                break
            remaining = max(deadline - self._monotonic(), 0.0)
            result = self._search_one(query, decision, min(config.request_timeout, remaining), trace)
            if mark_repair:
                trace.executed_queries = (*trace.executed_queries, _query_metadata((query,)))
            results.append(result)
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
    ) -> list[Any]:
        del round_kind
        hits: list[Any] = []
        for result in results:
            if result.status is ProviderStatus.SUCCESS:
                hits.extend(result.hits)
        hits = hits[: budget.max_candidate_urls]

        candidates: list[Any] = []
        reads = 0
        for hit in hits:
            if self._monotonic() >= deadline:
                break
            if reads >= budget.max_content_reads:
                break
            remaining = max(deadline - self._monotonic(), 0.0)
            try:
                candidate = self._extractor.extract(
                    hit,
                    _query_for_hit(plan, hit),
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
        return candidates

    def _assembler(self) -> EvidenceAssembler:
        return EvidenceAssembler(self._judge)

    def _monotonic(self) -> float:
        if self._clock is not None and hasattr(self._clock, "monotonic"):
            return self._clock.monotonic()
        return time.monotonic()

    def _elapsed_ms(self, started: float) -> int:
        return int((self._monotonic() - started) * 1000)

    def _finalize_trace(self, trace: SearchTrace) -> None:
        trace.total_response_latency_ms = (
            trace.route_latency_ms
            + trace.query_planning_latency_ms
            + trace.initial_provider_search_latency_ms
            + trace.initial_content_read_latency_ms
            + trace.initial_evidence_assembly_latency_ms
            + trace.gap_analysis_latency_ms
            + trace.adaptive_repair_latency_ms
        )
        try:
            logger.info("search trace: %s", trace.to_log_dict())
        except Exception:
            logger.debug("failed to serialize search trace", exc_info=True)


def _query_metadata(queries: Sequence[SearchQuery]) -> tuple[tuple[str, object], ...]:
    return tuple((query.query_id, query.purpose) for query in queries)


def _candidate_url_count(results: Sequence[ProviderResult]) -> int:
    return sum(len(result.hits) for result in results if result.status is ProviderStatus.SUCCESS)


def _repair_allowed(plan: SearchPlan, decision: Any) -> bool:
    del plan
    return decision.route in {SearchTier.STANDARD, SearchTier.DEEP}


def _query_for_hit(plan: SearchPlan, hit: Any) -> SearchQuery:
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
    """Idempotently fill total latency and log the body-free Trace once."""
    if trace.total_response_latency_ms == 0 and response_finished_at >= 0:
        trace.total_response_latency_ms = int(response_finished_at * 1000)
    logger.info("search trace final: %s", trace.to_log_dict())


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
