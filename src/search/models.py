"""Closed data contracts for the bounded evidence-search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class SearchTier(StrEnum):
    SKIP = "skip"
    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"


class SkipReason(StrEnum):
    USER_FORBID_WEB = "user_forbid_web"
    SOCIAL_OR_EMOTIONAL = "social_or_emotional"
    CREATIVE_OR_ROLEPLAY = "creative_or_roleplay"
    PROVIDED_TEXT_TRANSFORM = "provided_text_transform"
    PROVIDED_CONTENT_SUMMARY = "provided_content_summary"
    PURE_MATH = "pure_math"
    CLOSED_LOGIC = "closed_logic"
    CLOSED_CONTEXT_ONLY = "closed_context_only"


class SearchRoundKind(StrEnum):
    INITIAL = "initial"
    REPAIR = "repair"


class EvidenceState(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"


class RequestSource(StrEnum):
    CHAT = "chat"
    COMMAND = "command"
    COMPATIBILITY = "compatibility"


class TriggerCode(StrEnum):
    EXPLICIT_NO_WEB = "explicit_no_web"
    EXPLICIT_SEARCH = "explicit_search"
    EXPLICIT_VERIFICATION = "explicit_verification"
    EXPLICIT_SOURCE_REQUEST = "explicit_source_request"
    FRESHNESS_MARKER = "freshness_marker"
    DYNAMIC_ATTRIBUTE = "dynamic_attribute"
    REGULATED_DOMAIN_FOUNDATION = "regulated_domain_foundation"
    HIGH_CONSEQUENCE_ACTION = "high_consequence_action"
    CURRENT_RULE_OR_POLICY = "current_rule_or_policy"
    CONTROVERSY_OR_CONFLICT = "controversy_or_conflict"
    EXTERNAL_FACT_EXPLANATION_OR_COMPARISON = "external_fact_explanation_or_comparison"
    RECOMMENDATION_OR_EVALUATION = "recommendation_or_evaluation"
    AMBIGUOUS_ENTITY = "ambiguous_entity"
    MULTI_HOP_COMPLEXITY = "multi_hop_complexity"
    MIXED_TASK = "mixed_task"
    FACTUAL_DEFAULT = "factual_default"
    CLASSIFIER_UNCERTAIN = "classifier_uncertain"


class BenefitDimension(StrEnum):
    ACCURACY = "accuracy"
    FRESHNESS = "freshness"
    COMPLETENESS = "completeness"
    VERIFIABILITY = "verifiability"
    DISAMBIGUATION = "disambiguation"
    RISK_CONTROL = "risk_control"


class Factuality(StrEnum):
    NON_FACTUAL = "non_factual"
    FACTUAL = "factual"
    MIXED = "mixed"
    AMBIGUOUS = "ambiguous"


class Freshness(StrEnum):
    NONE = "none"
    LOW = "low"
    HIGH = "high"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Actionability(StrEnum):
    NONE = "none"
    GENERAL = "general"
    PERSONALIZED = "personalized"


class PotentialHarm(StrEnum):
    NONE = "none"
    LOW = "low"
    HIGH = "high"


class QueryPurpose(StrEnum):
    DIRECT = "direct"
    PRIMARY = "primary"
    INDEPENDENT = "independent"
    TIME_BOUNDED = "time_bounded"
    DISAMBIGUATION = "disambiguation"
    COUNTEREVIDENCE = "counterevidence"
    REPAIR = "repair"


class PlanningStatus(StrEnum):
    NORMAL = "normal"
    DEGRADED = "degraded"


class ProviderStatus(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"


class ExcerptOrigin(StrEnum):
    PROVIDER_SNIPPET = "provider_snippet"
    PAGE_EXTRACT = "page_extract"
    DOCUMENT_EXTRACT = "document_extract"


class CandidateRelevance(StrEnum):
    DIRECT = "direct"
    CONTEXTUAL = "contextual"
    IRRELEVANT = "irrelevant"


class SourceRelation(StrEnum):
    PRIMARY = "primary"
    INDEPENDENT = "independent"
    SECONDARY = "secondary"
    COMMUNITY = "community"
    UNKNOWN = "unknown"


class SupportLabel(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"
    UNMAPPED = "unmapped"


class SearchFailureCode(StrEnum):
    PROVIDER_NOT_CONFIGURED = "provider_not_configured"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    NO_RESULTS = "no_results"
    CONTENT_UNREADABLE = "content_unreadable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PARTIAL_EVIDENCE = "partial_evidence"
    SOURCE_CONFLICT = "source_conflict"
    VALIDATION_FAILED = "validation_failed"
    USER_FORBID_WEB = "user_forbid_web"


PROVIDER_STATUS_FAILURE_CODES: Mapping[ProviderStatus, SearchFailureCode | None] = MappingProxyType(
    {
        ProviderStatus.SUCCESS: None,
        ProviderStatus.EMPTY: SearchFailureCode.NO_RESULTS,
        ProviderStatus.TIMEOUT: SearchFailureCode.PROVIDER_TIMEOUT,
        ProviderStatus.ERROR: SearchFailureCode.PROVIDER_UNAVAILABLE,
        ProviderStatus.NOT_CONFIGURED: SearchFailureCode.PROVIDER_NOT_CONFIGURED,
        ProviderStatus.UNAVAILABLE: SearchFailureCode.PROVIDER_UNAVAILABLE,
    }
)


_TIER_RANK = {
    SearchTier.SKIP: 0,
    SearchTier.LIGHT: 1,
    SearchTier.STANDARD: 2,
    SearchTier.DEEP: 3,
}


def max_tier(left: SearchTier, right: SearchTier) -> SearchTier:
    return left if _TIER_RANK[left] >= _TIER_RANK[right] else right


def _require_enum(value: Any, enum_type: type[StrEnum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be a {enum_type.__name__}")


def _require_enum_values(values: tuple[Any, ...] | frozenset[Any], enum_type: type[StrEnum], field_name: str) -> None:
    for value in values:
        _require_enum(value, enum_type, field_name)


@dataclass(frozen=True)
class RetrievalRequest:
    question: str
    force_search: bool = False
    has_images: bool = False
    request_source: RequestSource = RequestSource.CHAT

    def __post_init__(self) -> None:
        _require_enum(self.request_source, RequestSource, "request_source")


@dataclass(frozen=True)
class RetrievalDecision:
    route: SearchTier
    skip_reason: SkipReason | None
    forced_search: bool
    trigger_codes: tuple[TriggerCode, ...]
    benefit_dimensions: frozenset[BenefitDimension]
    factuality: Factuality
    external_fact_required: bool
    freshness: Freshness
    risk: RiskLevel
    actionability: Actionability
    potential_harm: PotentialHarm
    program_minimum_tier: SearchTier | None
    model_recommended_tier: SearchTier | None
    final_reason_codes: tuple[TriggerCode, ...]

    def __post_init__(self) -> None:
        _require_enum(self.route, SearchTier, "route")
        if self.skip_reason is not None:
            _require_enum(self.skip_reason, SkipReason, "skip_reason")
        _require_enum_values(self.trigger_codes, TriggerCode, "trigger_codes")
        _require_enum_values(self.benefit_dimensions, BenefitDimension, "benefit_dimensions")
        _require_enum(self.factuality, Factuality, "factuality")
        _require_enum(self.freshness, Freshness, "freshness")
        _require_enum(self.risk, RiskLevel, "risk")
        _require_enum(self.actionability, Actionability, "actionability")
        _require_enum(self.potential_harm, PotentialHarm, "potential_harm")
        _require_enum_values(self.final_reason_codes, TriggerCode, "final_reason_codes")
        if self.program_minimum_tier is not None:
            _require_enum(self.program_minimum_tier, SearchTier, "program_minimum_tier")
        if self.model_recommended_tier is not None:
            _require_enum(self.model_recommended_tier, SearchTier, "model_recommended_tier")
        if self.route is SearchTier.SKIP:
            if self.skip_reason is None:
                raise ValueError("skip route requires a skip_reason")
            if self.program_minimum_tier is not None:
                raise ValueError("skip route cannot have a program minimum tier")
        else:
            if self.skip_reason is not None:
                raise ValueError("search routes cannot have a skip_reason")
            if self.program_minimum_tier is None:
                raise ValueError("search routes require a program minimum tier")
            if self.program_minimum_tier is SearchTier.SKIP:
                raise ValueError("program minimum tier cannot be skip")
            if _TIER_RANK[self.route] < _TIER_RANK[self.program_minimum_tier]:
                raise ValueError("final route cannot be below program minimum tier")
        if self.forced_search and self.route is SearchTier.SKIP and not self.requires_clarification:
            raise ValueError("forced search cannot use skip without explicit web/search conflict")

    @property
    def requires_clarification(self) -> bool:
        return (
            self.route is SearchTier.SKIP
            and self.skip_reason is SkipReason.USER_FORBID_WEB
            and TriggerCode.EXPLICIT_NO_WEB in self.trigger_codes
            and TriggerCode.EXPLICIT_SEARCH in self.trigger_codes
        )


@dataclass(frozen=True)
class TierBudget:
    max_initial_queries: int
    max_candidate_urls: int
    max_content_reads: int
    max_repair_queries: int
    max_total_queries: int
    max_retrieval_rounds: int
    hard_timeout_seconds: int

    def __post_init__(self) -> None:
        values = self.__dict__.values()
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("tier budgets must be non-negative integers")
        if self.max_total_queries != self.max_initial_queries + self.max_repair_queries:
            raise ValueError("max_total_queries must equal initial plus repair queries")
        expected_rounds = 1 + int(self.max_repair_queries > 0)
        if self.max_retrieval_rounds != expected_rounds:
            raise ValueError("max_retrieval_rounds must match the repair budget")
        if self.hard_timeout_seconds <= 0:
            raise ValueError("hard_timeout_seconds must be positive")


DEFAULT_TIER_BUDGETS: Mapping[SearchTier, TierBudget] = MappingProxyType(
    {
        SearchTier.LIGHT: TierBudget(1, 5, 2, 0, 1, 1, 8),
        SearchTier.STANDARD: TierBudget(3, 8, 5, 1, 4, 2, 20),
        SearchTier.DEEP: TierBudget(5, 15, 8, 1, 6, 2, 40),
    }
)


@dataclass(frozen=True)
class SearchQuery:
    query_id: str
    round_kind: SearchRoundKind
    purpose: QueryPurpose
    text: str
    date_from: date | None = None
    date_to: date | None = None
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_enum(self.round_kind, SearchRoundKind, "round_kind")
        _require_enum(self.purpose, QueryPurpose, "purpose")


@dataclass(frozen=True)
class SearchPlan:
    decision: RetrievalDecision
    original_question: str
    planning_status: PlanningStatus
    entities: tuple[str, ...]
    time_window: tuple[date | None, date | None] | None
    initial_queries: tuple[SearchQuery, ...]
    required_topics: tuple[str, ...]
    required_source_relations: frozenset[SourceRelation]
    query_redaction_codes: tuple[str, ...]
    budget: TierBudget

    def __post_init__(self) -> None:
        _require_enum(self.planning_status, PlanningStatus, "planning_status")
        _require_enum_values(self.required_source_relations, SourceRelation, "required_source_relations")


@dataclass(frozen=True)
class RepairPlan:
    triggered: bool
    gap_codes: tuple[str, ...]
    repair_query: SearchQuery | None

    def __post_init__(self) -> None:
        if self.triggered != (self.repair_query is not None):
            raise ValueError("repair_query must be present exactly when repair is triggered")


@dataclass(frozen=True)
class ProviderHit:
    provider: str
    query_id: str
    title: str
    url: str
    snippet: str | None
    score: float | None
    published_at: datetime | None
    raw_content: str | None
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    query_id: str
    provider: str
    title: str
    url: str
    canonical_url: str | None
    domain: str | None
    publisher: str | None
    source_relation: SourceRelation
    source_relation_basis: str | None
    published_at: datetime | None
    retrieved_at: datetime | None
    excerpt: str | None
    excerpt_origin: ExcerptOrigin | None
    extraction_status: str
    provider_score: float | None
    relevance_score: float | None
    relevance_gate_passed: bool
    freshness_state: Freshness
    citable: bool
    safety_flags: tuple[str, ...]
    supported_topics: tuple[str, ...]
    independence_group: str | None

    def __post_init__(self) -> None:
        _require_enum(self.source_relation, SourceRelation, "source_relation")
        if self.excerpt_origin is not None:
            _require_enum(self.excerpt_origin, ExcerptOrigin, "excerpt_origin")
        _require_enum(self.freshness_state, Freshness, "freshness_state")


@dataclass(frozen=True)
class EvidenceGapAnalysis:
    missing_claim_topics: tuple[str, ...]
    weak_source_topics: tuple[str, ...]
    conflict_groups: tuple[str, ...]
    repairable: bool
    repair_reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceBundle:
    request_id: str
    decision: RetrievalDecision
    plan: SearchPlan
    attempts: tuple["ProviderAttempt", ...]
    initial_evidence_ids: tuple[str, ...]
    gap_analysis: EvidenceGapAnalysis
    repair_plan: RepairPlan
    retrieval_round_count: int
    evidence_items: tuple[EvidenceItem, ...]
    evidence_state: EvidenceState
    missing_claim_topics: tuple[str, ...]
    weak_source_topics: tuple[str, ...]
    conflict_groups: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum(self.evidence_state, EvidenceState, "evidence_state")


@dataclass(frozen=True)
class ProviderReadiness:
    provider: str
    configured: bool
    available: bool
    reason_code: SearchFailureCode | None

    def __post_init__(self) -> None:
        if self.reason_code is not None:
            _require_enum(self.reason_code, SearchFailureCode, "reason_code")


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    status: ProviderStatus
    count: int
    latency_ms: int | float

    def __post_init__(self) -> None:
        _require_enum(self.status, ProviderStatus, "status")


@dataclass
class SearchTrace:
    request_id: str
    request_source: RequestSource
    route: SearchTier
    skip_reason: SkipReason | None = None
    trigger_codes: tuple[TriggerCode, ...] = ()
    factuality: Factuality | None = None
    external_fact_required: bool = False
    program_minimum_tier: SearchTier | None = None
    final_tier: SearchTier | None = None
    orchestrator_started: bool = False
    initial_query_count: int = 0
    initial_round_started: bool = False
    adaptive_repair_round_started: bool = False
    adaptive_repair_query: SearchQuery | tuple[str, QueryPurpose] | None = None
    retrieval_round_count: int = 0
    executed_queries: tuple[SearchQuery | tuple[str, QueryPurpose], ...] = ()
    provider_configured: bool = False
    provider_attempts: tuple[ProviderAttempt | Mapping[str, Any], ...] = ()
    provider_invocation_started: bool = False
    provider_failures: tuple[SearchFailureCode, ...] = ()
    candidate_url_count: int = 0
    citable_evidence_count: int = 0
    evidence_state: EvidenceState | None = None
    repair_used: bool = False
    claim_count: int = 0
    supported_claim_count: int = 0
    citation_count: int = 0
    knowledge_fallback_used: bool = False
    degradation_reason: SearchFailureCode | None = None
    route_latency_ms: int | float = 0
    query_planning_latency_ms: int | float = 0
    initial_provider_search_latency_ms: int | float = 0
    provider_search_total_latency_ms: int | float = 0
    initial_content_read_latency_ms: int | float = 0
    content_read_total_latency_ms: int | float = 0
    initial_evidence_assembly_latency_ms: int | float = 0
    evidence_assembly_total_latency_ms: int | float = 0
    gap_analysis_latency_ms: int | float = 0
    adaptive_repair_latency_ms: int | float = 0
    answer_generation_latency_ms: int | float = 0
    structural_validation_latency_ms: int | float = 0
    semantic_validation_latency_ms: int | float = 0
    qq_render_latency_ms: int | float = 0
    retrieval_pipeline_latency_ms: int | float = 0
    total_response_latency_ms: int | float = 0
    content_read_count: int = 0

    def to_log_dict(self) -> dict[str, Any]:
        values = {
            "request_id": self.request_id,
            "request_source": self.request_source,
            "route": self.route,
            "skip_reason": self.skip_reason,
            "trigger_codes": self.trigger_codes,
            "factuality": self.factuality,
            "external_fact_required": self.external_fact_required,
            "program_minimum_tier": self.program_minimum_tier,
            "final_tier": self.final_tier,
            "orchestrator_started": self.orchestrator_started,
            "initial_query_count": self.initial_query_count,
            "initial_round_started": self.initial_round_started,
            "adaptive_repair_round_started": self.adaptive_repair_round_started,
            "adaptive_repair_query": _query_metadata(self.adaptive_repair_query),
            "retrieval_round_count": self.retrieval_round_count,
            "executed_queries": [_query_metadata(query) for query in self.executed_queries],
            "provider_configured": self.provider_configured,
            "provider_attempts": [_attempt_metadata(attempt) for attempt in self.provider_attempts],
            "provider_invocation_started": self.provider_invocation_started,
            "provider_failures": self.provider_failures,
            "candidate_url_count": self.candidate_url_count,
            "citable_evidence_count": self.citable_evidence_count,
            "evidence_state": self.evidence_state,
            "repair_used": self.repair_used,
            "claim_count": self.claim_count,
            "supported_claim_count": self.supported_claim_count,
            "citation_count": self.citation_count,
            "knowledge_fallback_used": self.knowledge_fallback_used,
            "degradation_reason": self.degradation_reason,
            "route_latency_ms": self.route_latency_ms,
            "query_planning_latency_ms": self.query_planning_latency_ms,
            "initial_provider_search_latency_ms": self.initial_provider_search_latency_ms,
            "provider_search_total_latency_ms": self.provider_search_total_latency_ms,
            "initial_content_read_latency_ms": self.initial_content_read_latency_ms,
            "content_read_total_latency_ms": self.content_read_total_latency_ms,
            "initial_evidence_assembly_latency_ms": self.initial_evidence_assembly_latency_ms,
            "evidence_assembly_total_latency_ms": self.evidence_assembly_total_latency_ms,
            "gap_analysis_latency_ms": self.gap_analysis_latency_ms,
            "adaptive_repair_latency_ms": self.adaptive_repair_latency_ms,
            "answer_generation_latency_ms": self.answer_generation_latency_ms,
            "structural_validation_latency_ms": self.structural_validation_latency_ms,
            "semantic_validation_latency_ms": self.semantic_validation_latency_ms,
            "qq_render_latency_ms": self.qq_render_latency_ms,
            "retrieval_pipeline_latency_ms": self.retrieval_pipeline_latency_ms,
            "total_response_latency_ms": self.total_response_latency_ms,
            "semantic_query_count": len({item["query_id"] for item in (_query_metadata(query) for query in self.executed_queries) if item is not None}),
            "repair_query_count": int(self.adaptive_repair_round_started),
            "content_read_count": self.content_read_count,
            "provider_attempted": self.provider_invocation_started,
            "sufficient_evidence": self.evidence_state is EvidenceState.SUFFICIENT,
        }
        return _json_safe(values)


@dataclass(frozen=True)
class SearchPipelineResult:
    decision: RetrievalDecision
    plan: SearchPlan | None
    evidence: EvidenceBundle | None
    trace: SearchTrace
    failure_code: SearchFailureCode | None = None

    def __post_init__(self) -> None:
        if self.failure_code is not None:
            _require_enum(self.failure_code, SearchFailureCode, "failure_code")


def _query_metadata(query: SearchQuery | tuple[str, QueryPurpose] | None) -> dict[str, str] | None:
    if query is None:
        return None
    if isinstance(query, SearchQuery):
        return {"query_id": query.query_id, "purpose": query.purpose.value}
    query_id, purpose = query
    _require_enum(purpose, QueryPurpose, "query purpose")
    return {"query_id": query_id, "purpose": purpose.value}


def _attempt_metadata(attempt: ProviderAttempt | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(attempt, ProviderAttempt):
        return {
            "provider": attempt.provider,
            "status": attempt.status,
            "count": attempt.count,
            "latency_ms": attempt.latency_ms,
        }
    permitted = {"provider", "status", "count", "latency_ms"}
    return {key: value for key, value in attempt.items() if key in permitted}


def _json_safe(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value
