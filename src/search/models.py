"""Closed data contracts for the bounded evidence-search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
import math


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


class RetrievalComplexityCode(StrEnum):
    MULTI_FACT = "multi_fact"
    MULTI_ENTITY = "multi_entity"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    MULTI_SOURCE_REQUIRED = "multi_source_required"
    CROSS_VERIFICATION_REQUIRED = "cross_verification_required"
    AMBIGUOUS_ENTITY = "ambiguous_entity"


class SourceRequirement(StrEnum):
    ANY_RELEVANT = "any_relevant"
    INDEPENDENT_CORROBORATION = "independent_corroboration"


class FreshnessRequirement(StrEnum):
    NOT_REQUIRED = "not_required"
    CURRENT = "current"
    AS_OF = "as_of"
    WINDOW = "window"
    VERSION = "version"


class FreshnessEligibility(StrEnum):
    NOT_REQUIRED = "not_required"
    SATISFIED = "satisfied"
    STALE = "stale"
    UNKNOWN = "unknown"


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


class RedactionCode(StrEnum):
    CQ_CONTROL_CODE = "cq_control_code"
    DATA_URL = "data_url"
    CALLBACK_SECRET = "callback_secret"
    ONE_TIME_CODE = "one_time_code"
    PASSWORD = "password"
    BANK_ACCOUNT = "bank_account"
    CARD_CVV = "card_cvv"
    HARD_SECRET = "hard_secret"
    PHONE_NUMBER = "phone_number"
    EMAIL_ADDRESS = "email_address"
    EMPTY_AFTER_REDACTION = "empty_after_redaction"
    INVALID_REDACTION_CODE = "invalid_redaction_code"


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


def _tuple(values: Any) -> tuple[Any, ...]:
    return tuple(values)


def _frozenset(values: Any) -> frozenset[Any]:
    return frozenset(values)


def _normalize_fields(instance: Any, **values: Any) -> None:
    for name, value in values.items():
        object.__setattr__(instance, name, value)


def _strings(values: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, (tuple, list, set, frozenset)):
        raise TypeError(f"{field_name} must be a collection of strings")
    result = _tuple(values)
    if any(type(value) is not str for value in result):
        raise TypeError(f"{field_name} must contain strings")
    return result


def _redaction_codes(values: Any, field_name: str) -> tuple[RedactionCode, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, (tuple, list, set, frozenset)):
        raise TypeError(f"{field_name} must be a collection of RedactionCode values")
    try:
        return tuple(RedactionCode(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only recognized redaction codes") from exc


def _safe_redaction_codes(values: Any) -> tuple[RedactionCode, ...]:
    try:
        return _redaction_codes(values, "redaction codes")
    except (TypeError, ValueError):
        return (RedactionCode.INVALID_REDACTION_CODE,)


def _records(values: Any, record_type: type[Any], field_name: str) -> tuple[Any, ...]:
    result = _tuple(values)
    if any(not isinstance(value, record_type) for value in result):
        raise TypeError(f"{field_name} must contain {record_type.__name__}")
    return result


def _require_record(value: Any, record_type: type[Any], field_name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, record_type):
        raise TypeError(f"{field_name} must be a {record_type.__name__}")


def _enum_set(values: Any, enum_type: type[StrEnum], field_name: str) -> frozenset[Any]:
    result = _frozenset(values)
    _require_enum_values(result, enum_type, field_name)
    return result


def _require_number(value: Any, field_name: str, *, non_negative: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    if non_negative and value < 0:
        raise ValueError(f"{field_name} must not be negative")


def _is_sensitive(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in ("http://", "https://", "api_key", "apikey", "token=", "qq=", "group="))


def _require_safe_metadata(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or _is_sensitive(value):
        raise ValueError(f"{field_name} must be non-sensitive metadata")


@dataclass(frozen=True)
class RetrievalRequest:
    question: str
    force_search: bool = False
    has_images: bool = False
    request_source: RequestSource = RequestSource.CHAT

    def __post_init__(self) -> None:
        _require_enum(self.request_source, RequestSource, "request_source")


@dataclass(frozen=True)
class RetrievalContext:
    must_search: bool
    skip_reason: SkipReason | None
    factuality: Factuality
    external_fact_required: bool
    complexity_codes: tuple[RetrievalComplexityCode, ...]
    source_requirement: SourceRequirement

    def __post_init__(self) -> None:
        if type(self.must_search) is not bool:
            raise TypeError("must_search must be a boolean")
        if type(self.external_fact_required) is not bool:
            raise TypeError("external_fact_required must be a boolean")
        if self.skip_reason is not None:
            _require_enum(self.skip_reason, SkipReason, "skip_reason")
        _require_enum(self.factuality, Factuality, "factuality")
        _require_enum(self.source_requirement, SourceRequirement, "source_requirement")
        complexity_codes = _tuple(self.complexity_codes)
        _require_enum_values(
            complexity_codes,
            RetrievalComplexityCode,
            "complexity_codes",
        )
        _normalize_fields(self, complexity_codes=complexity_codes)


@dataclass(frozen=True)
class FreshnessContext:
    requirement: FreshnessRequirement
    as_of: date | None
    date_from: date | None
    date_to: date | None
    version_constraint: str | None

    def __post_init__(self) -> None:
        _require_enum(self.requirement, FreshnessRequirement, "requirement")
        for field_name in ("as_of", "date_from", "date_to"):
            value = getattr(self, field_name)
            if value is not None and type(value) is not date:
                raise TypeError(f"{field_name} must be a date or None")
        if self.version_constraint is not None and type(self.version_constraint) is not str:
            raise TypeError("version_constraint must be a string or None")
        version_constraint = (
            self.version_constraint.strip()
            if self.version_constraint is not None
            else None
        )
        _normalize_fields(self, version_constraint=version_constraint)
        if self.date_from is not None and self.date_to is not None and self.date_from > self.date_to:
            raise ValueError("freshness date_from cannot exceed date_to")
        if self.requirement is FreshnessRequirement.NOT_REQUIRED and any(
            value is not None
            for value in (
                self.as_of,
                self.date_from,
                self.date_to,
                self.version_constraint,
            )
        ):
            raise ValueError("not_required freshness cannot carry a constraint")
        if (
            self.requirement is FreshnessRequirement.VERSION
            and not self.version_constraint
        ):
            raise ValueError("version freshness requires version_constraint")


@dataclass(frozen=True)
class RiskContext:
    high_consequence: bool
    warning_required: bool
    fail_closed: bool

    def __post_init__(self) -> None:
        if not all(
            type(value) is bool
            for value in (
                self.high_consequence,
                self.warning_required,
                self.fail_closed,
            )
        ):
            raise TypeError("risk flags must be booleans")
        if self.warning_required and not self.high_consequence:
            raise ValueError("warning requires a high-consequence request")


@dataclass(frozen=True)
class RequestAnalysis:
    retrieval: RetrievalContext
    freshness: FreshnessContext
    risk: RiskContext

    def __post_init__(self) -> None:
        _require_record(self.retrieval, RetrievalContext, "retrieval")
        _require_record(self.freshness, FreshnessContext, "freshness")
        _require_record(self.risk, RiskContext, "risk")


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
        _normalize_fields(self, trigger_codes=_tuple(self.trigger_codes), benefit_dimensions=_enum_set(self.benefit_dimensions, BenefitDimension, "benefit_dimensions"), final_reason_codes=_tuple(self.final_reason_codes))
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
            and self.forced_search
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
        if any(type(value) is not int or value < 0 for value in values):
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
class RequiredTopic:
    """One bounded answer topic owned by the retrieval/freshness contracts."""

    topic_id: str
    label: str
    material: bool
    freshness_requirement: FreshnessRequirement
    date_from: date | None = None
    date_to: date | None = None
    version_constraint: str | None = None
    source_requirement: SourceRequirement = SourceRequirement.ANY_RELEVANT

    def __post_init__(self) -> None:
        if not isinstance(self.topic_id, str) or not self.topic_id.strip():
            raise ValueError("topic_id must be a non-blank string")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-blank string")
        if type(self.material) is not bool:
            raise TypeError("material must be a boolean")
        _require_enum(
            self.freshness_requirement,
            FreshnessRequirement,
            "freshness_requirement",
        )
        _require_enum(
            self.source_requirement,
            SourceRequirement,
            "source_requirement",
        )
        for field_name in ("date_from", "date_to"):
            value = getattr(self, field_name)
            if value is not None and type(value) is not date:
                raise TypeError(f"{field_name} must be a date or None")
        if self.version_constraint is not None and type(self.version_constraint) is not str:
            raise TypeError("version_constraint must be a string or None")
        version_constraint = (
            self.version_constraint.strip()
            if self.version_constraint is not None
            else None
        )
        _normalize_fields(
            self,
            topic_id=self.topic_id.strip(),
            label=self.label.strip(),
            version_constraint=version_constraint,
        )
        if self.date_from is not None and self.date_to is not None and self.date_from > self.date_to:
            raise ValueError("topic date_from cannot exceed date_to")
        if self.freshness_requirement is FreshnessRequirement.NOT_REQUIRED and any(
            value is not None
            for value in (self.date_from, self.date_to, self.version_constraint)
        ):
            raise ValueError("not_required topic freshness cannot carry a constraint")
        if (
            self.freshness_requirement is FreshnessRequirement.VERSION
            and not self.version_constraint
        ):
            raise ValueError("version topic freshness requires version_constraint")


_LEGACY_REQUIRED_TOPIC_MARKER = "_task4_legacy_required_topic"


def _legacy_required_topic(topic_id: str, label: str) -> RequiredTopic:
    """Build the Task 10-removable label-only SearchPlan input shim."""
    topic = RequiredTopic(
        topic_id=topic_id,
        label=label,
        material=True,
        freshness_requirement=FreshnessRequirement.NOT_REQUIRED,
        source_requirement=SourceRequirement.ANY_RELEVANT,
    )
    # `dataclasses.replace(plan, ...)` retains topic objects.  The marker
    # therefore preserves only the explicit legacy-string constructor path;
    # newly-built RequiredTopic plans must pass sealed-query validation.
    object.__setattr__(topic, _LEGACY_REQUIRED_TOPIC_MARKER, True)
    return topic


def _has_legacy_required_topic_provenance(
    topics: tuple[RequiredTopic, ...],
) -> bool:
    return bool(topics) and all(
        getattr(topic, _LEGACY_REQUIRED_TOPIC_MARKER, False) is True
        for topic in topics
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
    query_index: int = None  # type: ignore[assignment]
    target_topic_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.query_index is None:
            query_index = None
        else:
            if type(self.query_index) is not int or self.query_index <= 0:
                raise ValueError("query_index must be a positive integer")
            query_index = self.query_index
        _normalize_fields(
            self,
            include_domains=_strings(self.include_domains, "include_domains"),
            exclude_domains=_strings(self.exclude_domains, "exclude_domains"),
            query_index=query_index,
            target_topic_ids=_strings(self.target_topic_ids, "target_topic_ids"),
        )
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
    required_topics: tuple[RequiredTopic, ...]
    required_source_relations: frozenset[SourceRelation]
    query_redaction_codes: tuple[RedactionCode, ...]
    budget: TierBudget

    def __post_init__(self) -> None:
        _require_record(self.decision, RetrievalDecision, "decision")
        _require_record(self.budget, TierBudget, "budget")
        time_window = None if self.time_window is None else _tuple(self.time_window)
        if time_window is not None and (len(time_window) != 2 or any(item is not None and not isinstance(item, date) for item in time_window)):
            raise ValueError("time_window must be a two-element date tuple")
        if isinstance(self.required_topics, (str, bytes, Mapping)) or not isinstance(
            self.required_topics,
            (tuple, list, set, frozenset),
        ):
            raise TypeError("required_topics must be a collection of RequiredTopic values")
        raw_topics = _tuple(self.required_topics)
        legacy_topic_labels = all(type(topic) is str for topic in raw_topics)
        if legacy_topic_labels:
            topic_labels = _strings(raw_topics, "required_topics")
            if len(topic_labels) > 3:
                raise ValueError("legacy required_topics cannot exceed three labels")
            if not topic_labels:
                topic_labels = (
                    str(self.original_question or "").strip() or "用户问题",
                )
            required_topics = tuple(
                _legacy_required_topic(
                    topic_id=f"topic-{index}",
                    label=label,
                )
                for index, label in enumerate(topic_labels, 1)
            )
        else:
            required_topics = _records(
                raw_topics,
                RequiredTopic,
                "required_topics",
            )
        initial_queries = _records(
            self.initial_queries,
            SearchQuery,
            "initial_queries",
        )
        legacy_query_slots = all(
            query.query_index is None and not query.target_topic_ids
            for query in initial_queries
        )
        _normalize_fields(
            self,
            entities=_strings(self.entities, "entities"),
            time_window=time_window,
            initial_queries=initial_queries,
            required_topics=required_topics,
            required_source_relations=_enum_set(
                self.required_source_relations,
                SourceRelation,
                "required_source_relations",
            ),
            query_redaction_codes=_redaction_codes(
                self.query_redaction_codes,
                "query_redaction_codes",
            ),
        )
        _require_enum(self.planning_status, PlanningStatus, "planning_status")
        _require_enum_values(self.required_source_relations, SourceRelation, "required_source_relations")
        legacy_topic_provenance = (
            legacy_topic_labels
            or _has_legacy_required_topic_provenance(required_topics)
        )
        # Task 10 removes this constructor shim.  Only a plan that originated
        # from legacy strings may retain legacy query slots through
        # `dataclasses.replace`; a newly-built RequiredTopic plan is always
        # required to carry sealed initial query metadata.
        if not (legacy_topic_provenance and legacy_query_slots):
            self._validate_structured_queries()

    def _validate_structured_queries(self) -> None:
        if not self.required_topics or len(self.required_topics) > 3:
            raise ValueError("structured plans require one to three topics")
        expected_topic_ids = tuple(
            f"topic-{index}" for index in range(1, len(self.required_topics) + 1)
        )
        if tuple(topic.topic_id for topic in self.required_topics) != expected_topic_ids:
            raise ValueError("required topic ids must be closed and sequential")
        material_topic_ids = tuple(
            topic.topic_id for topic in self.required_topics if topic.material
        )
        if not material_topic_ids:
            raise ValueError("structured plans require at least one material topic")
        if not self.initial_queries:
            raise ValueError("structured plans require an initial query")
        if len(self.initial_queries) > self.budget.max_initial_queries:
            raise ValueError("initial queries exceed the tier budget")
        if any(query.round_kind is not SearchRoundKind.INITIAL for query in self.initial_queries):
            raise ValueError("structured plan queries must be initial queries")
        if self.initial_queries[0].purpose is not QueryPurpose.DIRECT:
            raise ValueError("the first initial query must be direct")
        if any(
            query.purpose is QueryPurpose.DIRECT
            for query in self.initial_queries[1:]
        ):
            raise ValueError("only the first initial query may be direct")
        if any(
            not isinstance(query.query_id, str) or not query.query_id.strip()
            for query in self.initial_queries
        ) or len({query.query_id for query in self.initial_queries}) != len(self.initial_queries):
            raise ValueError("initial query ids must be unique and non-blank")
        expected_query_ids = tuple(
            f"initial-{index}" for index in range(1, len(self.initial_queries) + 1)
        )
        if tuple(query.query_id for query in self.initial_queries) != expected_query_ids:
            raise ValueError("initial query ids must match their sealed order")
        expected_query_indexes = tuple(range(1, len(self.initial_queries) + 1))
        if tuple(query.query_index for query in self.initial_queries) != expected_query_indexes:
            raise ValueError("initial query indexes must be unique and sequential")
        direct = self.initial_queries[0]
        if tuple(direct.target_topic_ids) != material_topic_ids:
            raise ValueError("direct query must target every material topic")
        material_topic_id_set = set(material_topic_ids)
        for query in self.initial_queries[1:]:
            target_topic_ids = tuple(query.target_topic_ids)
            target_topic_id_set = set(target_topic_ids)
            if (
                not target_topic_ids
                or len(target_topic_id_set) != len(target_topic_ids)
                or not target_topic_id_set.issubset(material_topic_id_set)
                or tuple(
                    topic_id
                    for topic_id in material_topic_ids
                    if topic_id in target_topic_id_set
                ) != target_topic_ids
            ):
                raise ValueError("supplemental queries must target known material topics")


@dataclass(frozen=True)
class RepairPlan:
    triggered: bool
    gap_codes: tuple[str, ...]
    repair_query: SearchQuery | None
    query_redaction_codes: tuple[RedactionCode, ...] = ()

    def __post_init__(self) -> None:
        _normalize_fields(
            self,
            gap_codes=_strings(self.gap_codes, "gap_codes"),
            query_redaction_codes=_redaction_codes(self.query_redaction_codes, "query_redaction_codes"),
        )
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

    def __post_init__(self) -> None:
        _normalize_fields(self, quality_flags=_strings(self.quality_flags, "quality_flags"))


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
    conflict_key: str | None = None
    conflict_value: str | None = None
    conflict_relation: str | None = None

    def __post_init__(self) -> None:
        _normalize_fields(self, safety_flags=_strings(self.safety_flags, "safety_flags"), supported_topics=_strings(self.supported_topics, "supported_topics"))
        _require_enum(self.source_relation, SourceRelation, "source_relation")
        if self.excerpt_origin is not None:
            _require_enum(self.excerpt_origin, ExcerptOrigin, "excerpt_origin")
        _require_enum(self.freshness_state, Freshness, "freshness_state")
        if self.conflict_relation not in {None, "contradicts", "claims_supersession"}:
            raise ValueError("conflict_relation must be a closed conflict relation")


@dataclass(frozen=True)
class EvidenceConflictMember:
    evidence_id: str
    value: str
    published_at: datetime | None
    relation: str

    def __post_init__(self) -> None:
        _require_safe_metadata(self.evidence_id, "evidence_id")
        if not str(self.value or "").strip():
            raise ValueError("conflict member value is required")
        if self.relation not in {"contradicts", "claims_supersession"}:
            raise ValueError("unknown conflict member relation")


@dataclass(frozen=True)
class EvidenceConflict:
    conflict_id: str
    conflict_key: str
    members: tuple[EvidenceConflictMember, ...]

    def __post_init__(self) -> None:
        _require_safe_metadata(self.conflict_id, "conflict_id")
        if not str(self.conflict_key or "").strip():
            raise ValueError("conflict_key is required")
        _normalize_fields(
            self,
            members=_records(self.members, EvidenceConflictMember, "members"),
        )
        if len(self.members) < 2:
            raise ValueError("a conflict requires at least two members")
        if len({member.evidence_id for member in self.members}) < 2:
            raise ValueError("a conflict requires at least two distinct evidence ids")
        if len({str(member.value).strip() for member in self.members}) < 2:
            raise ValueError("a conflict requires at least two distinct asserted values")


@dataclass(frozen=True)
class EvidenceGapAnalysis:
    missing_claim_topics: tuple[str, ...]
    conflict_group_ids: tuple[str, ...]
    repair_eligible: bool
    repair_purpose: str | None
    repair_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _normalize_fields(self, missing_claim_topics=_strings(self.missing_claim_topics, "missing_claim_topics"), conflict_group_ids=_strings(self.conflict_group_ids, "conflict_group_ids"), repair_reason_codes=_strings(self.repair_reason_codes, "repair_reason_codes"))
        if self.repair_eligible and (not self.repair_purpose or not self.repair_reason_codes):
            raise ValueError("eligible repair requires a purpose and reason code")
        if not self.repair_eligible and self.repair_purpose is not None:
            raise ValueError("ineligible repair cannot have a repair purpose")


@dataclass(frozen=True)
class TopicAssessment:
    topic_id: str
    freshness: FreshnessEligibility
    supporting_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.topic_id, str) or not self.topic_id.strip():
            raise ValueError("topic_id must be a non-blank string")
        _require_enum(self.freshness, FreshnessEligibility, "freshness")
        supporting_evidence_ids = _strings(
            self.supporting_evidence_ids,
            "supporting_evidence_ids",
        )
        if any(not evidence_id.strip() for evidence_id in supporting_evidence_ids):
            raise ValueError("supporting_evidence_ids must be non-blank")
        if len(set(supporting_evidence_ids)) != len(supporting_evidence_ids):
            raise ValueError("supporting_evidence_ids must be unique")
        _normalize_fields(
            self,
            topic_id=self.topic_id.strip(),
            supporting_evidence_ids=supporting_evidence_ids,
        )


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
    conflicts: tuple[EvidenceConflict, ...] = ()
    topic_assessments: tuple[TopicAssessment, ...] = field(kw_only=True)
    supported_topic_ids: tuple[str, ...] = field(kw_only=True)
    missing_topic_ids: tuple[str, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        _require_record(self.decision, RetrievalDecision, "decision")
        _require_record(self.plan, SearchPlan, "plan")
        _require_record(self.gap_analysis, EvidenceGapAnalysis, "gap_analysis")
        _require_record(self.repair_plan, RepairPlan, "repair_plan")
        _normalize_fields(self, attempts=_records(self.attempts, ProviderAttempt, "attempts"), initial_evidence_ids=_strings(self.initial_evidence_ids, "initial_evidence_ids"), evidence_items=_records(self.evidence_items, EvidenceItem, "evidence_items"), missing_claim_topics=_strings(self.missing_claim_topics, "missing_claim_topics"), weak_source_topics=_strings(self.weak_source_topics, "weak_source_topics"), conflict_groups=_strings(self.conflict_groups, "conflict_groups"), limitations=_strings(self.limitations, "limitations"), conflicts=_records(self.conflicts, EvidenceConflict, "conflicts"), topic_assessments=_records(self.topic_assessments, TopicAssessment, "topic_assessments"), supported_topic_ids=_strings(self.supported_topic_ids, "supported_topic_ids"), missing_topic_ids=_strings(self.missing_topic_ids, "missing_topic_ids"))
        _require_enum(self.evidence_state, EvidenceState, "evidence_state")
        material_topic_ids = tuple(
            topic.topic_id for topic in self.plan.required_topics if topic.material
        )
        material_topics = tuple(
            topic for topic in self.plan.required_topics if topic.material
        )
        assessment_ids = tuple(
            assessment.topic_id for assessment in self.topic_assessments
        )
        if assessment_ids != material_topic_ids:
            raise ValueError("topic assessments must cover material topics in plan order")
        assessment_by_id = {
            assessment.topic_id: assessment for assessment in self.topic_assessments
        }
        expected_supported = tuple(
            topic_id
            for topic_id in material_topic_ids
            if assessment_by_id[topic_id].supporting_evidence_ids
        )
        expected_missing = tuple(
            topic_id for topic_id in material_topic_ids if topic_id not in expected_supported
        )
        if self.supported_topic_ids != expected_supported:
            raise ValueError("supported topic ids must match topic assessments")
        if self.missing_topic_ids != expected_missing:
            raise ValueError("missing topic ids must match topic assessments")
        expected_missing_claim_topics = tuple(
            topic.label
            for topic in material_topics
            if topic.topic_id in expected_missing
        )
        if self.missing_claim_topics != expected_missing_claim_topics:
            raise ValueError("missing claim topics must project missing topic ids")
        for topic in material_topics:
            assessment = assessment_by_id[topic.topic_id]
            if topic.freshness_requirement is FreshnessRequirement.NOT_REQUIRED:
                if assessment.freshness is not FreshnessEligibility.NOT_REQUIRED:
                    raise ValueError("not-required topics require not-required freshness")
            elif assessment.freshness is FreshnessEligibility.NOT_REQUIRED:
                raise ValueError("freshness-constrained topics cannot be not-required")
        if any(
            assessment.freshness in {
                FreshnessEligibility.STALE,
                FreshnessEligibility.UNKNOWN,
            }
            and assessment.supporting_evidence_ids
            for assessment in self.topic_assessments
        ):
            raise ValueError("stale or unknown topics cannot have supporting evidence")
        evidence_by_id = {item.evidence_id: item for item in self.evidence_items}
        if len(evidence_by_id) != len(self.evidence_items):
            raise ValueError("bundle evidence ids must be unique")
        conflict_ids = tuple(conflict.conflict_id for conflict in self.conflicts)
        if len(set(conflict_ids)) != len(conflict_ids):
            raise ValueError("bundle conflict ids must be unique")
        if self.conflict_groups != conflict_ids:
            raise ValueError("conflict groups must match conflicts in order")
        for conflict in self.conflicts:
            for member in conflict.members:
                item = evidence_by_id.get(member.evidence_id)
                if item is None:
                    raise ValueError("conflict members must reference bundle evidence")
                if not item.citable or not item.relevance_gate_passed:
                    raise ValueError("conflict members require citable relevant evidence")
        for topic in material_topics:
            assessment = assessment_by_id[topic.topic_id]
            for evidence_id in assessment.supporting_evidence_ids:
                item = evidence_by_id.get(evidence_id)
                if item is None:
                    raise ValueError("topic assessments must reference bundle evidence")
                if not item.citable or not item.relevance_gate_passed:
                    raise ValueError("topic support requires citable relevant evidence")
                if topic.label not in item.supported_topics:
                    raise ValueError("topic support must match the legacy label projection")
        if self.conflicts:
            expected_state = EvidenceState.CONFLICTING
        elif not self.missing_topic_ids:
            expected_state = EvidenceState.SUFFICIENT
        elif self.supported_topic_ids:
            expected_state = EvidenceState.PARTIAL
        else:
            expected_state = EvidenceState.INSUFFICIENT
        if self.evidence_state is not expected_state:
            raise ValueError("evidence state must match deterministic topic priority")


@dataclass(frozen=True)
class ProviderReadiness:
    provider: str
    configured: bool
    available: bool
    reason_code: SearchFailureCode | None

    def __post_init__(self) -> None:
        if self.reason_code is not None:
            _require_enum(self.reason_code, SearchFailureCode, "reason_code")
        if not self.configured and (self.available or self.reason_code is not SearchFailureCode.PROVIDER_NOT_CONFIGURED):
            raise ValueError("unconfigured providers require provider_not_configured")
        if self.configured and not self.available and self.reason_code is not SearchFailureCode.PROVIDER_UNAVAILABLE:
            raise ValueError("configured unavailable providers require provider_unavailable")
        if self.available and not self.configured:
            raise ValueError("unconfigured providers cannot be available")
        if self.available and self.reason_code is not None:
            raise ValueError("available providers cannot have a failure reason")
        if not self.available and self.reason_code is None:
            raise ValueError("unavailable providers require a failure reason")


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    status: ProviderStatus
    count: int
    latency_ms: int | float
    query_id: str = ""
    configured: bool = True
    available: bool = True
    invocation_started: bool = True

    def __post_init__(self) -> None:
        _require_enum(self.status, ProviderStatus, "status")
        _require_safe_metadata(self.provider, "provider")
        if self.query_id:
            _require_safe_metadata(self.query_id, "query_id")
        if type(self.count) is not int or self.count < 0:
            raise ValueError("count must be a non-negative integer")
        _require_number(self.latency_ms, "latency_ms")
        if not all(type(value) is bool for value in (self.configured, self.available, self.invocation_started)):
            raise TypeError("provider readiness and invocation fields must be booleans")


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    status: ProviderStatus
    hits: tuple[ProviderHit, ...]
    latency_ms: int | float

    def __post_init__(self) -> None:
        _normalize_fields(self, hits=_records(self.hits, ProviderHit, "hits"))
        _require_safe_metadata(self.provider, "provider")
        _require_enum(self.status, ProviderStatus, "status")
        _require_number(self.latency_ms, "latency_ms")
        if self.status is ProviderStatus.SUCCESS and not self.hits:
            raise ValueError("successful provider result requires hits")
        if self.status is not ProviderStatus.SUCCESS and self.hits:
            raise ValueError("non-success provider result cannot contain hits")


@dataclass(frozen=True)
class FetchedDocument:
    requested_url: str
    final_url: str | None
    content_type: str | None
    title: str | None
    excerpt: str | None
    fetch_status: str
    untrusted_content_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        _normalize_fields(self, untrusted_content_flags=_strings(self.untrusted_content_flags, "untrusted_content_flags"))


@dataclass(frozen=True)
class EvidenceCandidate:
    hit: ProviderHit
    document: FetchedDocument | None
    excerpt: str | None
    excerpt_origin: ExcerptOrigin | None
    extraction_status: str
    safety_flags: tuple[str, ...]
    content_reads_consumed: int

    def __post_init__(self) -> None:
        _require_record(self.hit, ProviderHit, "hit")
        _require_record(self.document, FetchedDocument, "document", optional=True)
        _normalize_fields(self, safety_flags=_strings(self.safety_flags, "safety_flags"))
        if self.excerpt_origin is not None:
            _require_enum(self.excerpt_origin, ExcerptOrigin, "excerpt_origin")
        if type(self.content_reads_consumed) is not int or self.content_reads_consumed not in (0, 1):
            raise ValueError("content_reads_consumed must be 0 or 1")


@dataclass(frozen=True)
class Claim:
    claim_id: str
    block_id: str
    text: str
    material: bool
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _normalize_fields(self, evidence_ids=_strings(self.evidence_ids, "evidence_ids"))


@dataclass(frozen=True)
class AnswerBlock:
    block_id: str
    kind: str
    text: str
    claim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _normalize_fields(self, claim_ids=_strings(self.claim_ids, "claim_ids"))


@dataclass(frozen=True)
class GroundedDraft:
    answer_blocks: tuple[AnswerBlock, ...]
    claims: tuple[Claim, ...]
    limitations: tuple[str, ...]
    conflict_summary: tuple[str, ...]
    used_knowledge_fallback: bool

    def __post_init__(self) -> None:
        _normalize_fields(self, answer_blocks=_records(self.answer_blocks, AnswerBlock, "answer_blocks"), claims=_records(self.claims, Claim, "claims"), limitations=_strings(self.limitations, "limitations"), conflict_summary=_strings(self.conflict_summary, "conflict_summary"))


@dataclass(frozen=True)
class ValidationReport:
    draft: GroundedDraft
    retained_blocks: tuple[AnswerBlock, ...]
    retained_claims: tuple[Claim, ...]
    removed_block_ids: tuple[str, ...]
    claim_labels: Mapping[str, SupportLabel]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_record(self.draft, GroundedDraft, "draft")
        labels = dict(self.claim_labels)
        if any(type(key) is not str or not isinstance(value, SupportLabel) for key, value in labels.items()):
            raise TypeError("claim_labels must map strings to SupportLabel")
        _normalize_fields(self, retained_blocks=_records(self.retained_blocks, AnswerBlock, "retained_blocks"), retained_claims=_records(self.retained_claims, Claim, "retained_claims"), removed_block_ids=_strings(self.removed_block_ids, "removed_block_ids"), claim_labels=MappingProxyType(labels), limitations=_strings(self.limitations, "limitations"))
        retained_ids = {block.block_id for block in self.retained_blocks}
        overlap = retained_ids.intersection(self.removed_block_ids)
        if overlap:
            raise ValueError("retained and removed block sets must be disjoint")


@dataclass(frozen=True)
class RenderedReply:
    text: str
    chunks: tuple[str, ...]
    used_evidence_ids: tuple[str, ...]
    shown_source_urls: tuple[str, ...]
    degradation_disclosures: tuple[str, ...]

    def __post_init__(self) -> None:
        _normalize_fields(self, chunks=_strings(self.chunks, "chunks"), used_evidence_ids=_strings(self.used_evidence_ids, "used_evidence_ids"), shown_source_urls=_strings(self.shown_source_urls, "shown_source_urls"), degradation_disclosures=_strings(self.degradation_disclosures, "degradation_disclosures"))


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
    provider_attempts: tuple[ProviderAttempt, ...] = ()
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
    initial_query_redaction_codes: tuple[RedactionCode, ...] = ()
    adaptive_repair_redaction_codes: tuple[RedactionCode, ...] = ()
    response_started_at: float | None = field(default=None, repr=False, compare=False)
    response_finished_at: float | None = field(default=None, repr=False, compare=False)
    finalized: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.trigger_codes = _tuple(self.trigger_codes)
        self.executed_queries = _tuple(self.executed_queries)
        self.provider_attempts = _tuple(self.provider_attempts)
        self.provider_failures = _tuple(self.provider_failures)
        self.initial_query_redaction_codes = _redaction_codes(
            self.initial_query_redaction_codes, "initial_query_redaction_codes"
        )
        self.adaptive_repair_redaction_codes = _redaction_codes(
            self.adaptive_repair_redaction_codes, "adaptive_repair_redaction_codes"
        )
        for attempt in self.provider_attempts:
            if not isinstance(attempt, ProviderAttempt):
                raise TypeError("provider_attempts must contain ProviderAttempt values")

    def to_log_dict(self) -> dict[str, Any]:
        adaptive_repair = _query_metadata(self.adaptive_repair_query)
        executed_metadata = tuple(
            metadata
            for query in self.executed_queries
            if (metadata := _query_metadata(query)) is not None
        )
        values = {
            "request_id": _safe_log_identifier(self.request_id),
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
            "adaptive_repair_query": adaptive_repair,
            "initial_query_redaction_codes": _safe_redaction_codes(self.initial_query_redaction_codes),
            "adaptive_repair_redaction_codes": _safe_redaction_codes(self.adaptive_repair_redaction_codes),
            "retrieval_round_count": self.retrieval_round_count,
            "executed_queries": executed_metadata,
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
            "semantic_query_count": len({item["query_id"] for item in executed_metadata}),
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
    # Task 10 retains old direct constructors; production must always supply
    # the immutable analysis created at the beginning of the request.
    analysis: RequestAnalysis | None = None

    def __post_init__(self) -> None:
        _require_record(self.analysis, RequestAnalysis, "analysis", optional=True)
        if self.failure_code is not None:
            _require_enum(self.failure_code, SearchFailureCode, "failure_code")
        if self.decision.route is SearchTier.SKIP:
            if self.plan is not None or self.evidence is not None:
                raise ValueError("skip results cannot include a plan or evidence")
            if self.failure_code is not None and not (
                self.failure_code is SearchFailureCode.USER_FORBID_WEB
                and self.decision.skip_reason is SkipReason.USER_FORBID_WEB
            ):
                raise ValueError("skip results only permit user_forbid_web failure")
            return
        if self.plan is None:
            raise ValueError("search results require a plan")
        if self.evidence is None:
            if self.failure_code is not SearchFailureCode.PROVIDER_NOT_CONFIGURED:
                raise ValueError("evidence-free search results require a pre-evidence failure")
            return
        state_failures = {
            EvidenceState.SUFFICIENT: None,
            EvidenceState.PARTIAL: SearchFailureCode.PARTIAL_EVIDENCE,
            EvidenceState.CONFLICTING: SearchFailureCode.SOURCE_CONFLICT,
            EvidenceState.INSUFFICIENT: SearchFailureCode.INSUFFICIENT_EVIDENCE,
        }
        evidence_state = self.evidence.evidence_state
        allowed = {state_failures[evidence_state]}
        if evidence_state in {
            EvidenceState.SUFFICIENT,
            EvidenceState.PARTIAL,
            EvidenceState.CONFLICTING,
        }:
            allowed.add(SearchFailureCode.VALIDATION_FAILED)
        if "hard_deadline_exceeded" in getattr(self.evidence, "limitations", ()):
            allowed.add(SearchFailureCode.PROVIDER_TIMEOUT)
        if evidence_state is EvidenceState.INSUFFICIENT:
            allowed |= {SearchFailureCode.PROVIDER_UNAVAILABLE, SearchFailureCode.PROVIDER_TIMEOUT, SearchFailureCode.NO_RESULTS, SearchFailureCode.CONTENT_UNREADABLE}
        if self.failure_code not in allowed:
            raise ValueError("failure_code must match evidence state")


def _query_metadata(query: Any) -> dict[str, str] | None:
    if query is None:
        return None
    if isinstance(query, SearchQuery):
        return {"query_id": _safe_log_identifier(query.query_id), "purpose": query.purpose.value}
    if not isinstance(query, tuple) or len(query) != 2:
        return None
    query_id, purpose = query
    if not isinstance(query_id, str) or not isinstance(purpose, QueryPurpose):
        return None
    return {"query_id": _safe_log_identifier(query_id), "purpose": purpose.value}


def _attempt_metadata(attempt: ProviderAttempt) -> dict[str, Any]:
    if not isinstance(attempt, ProviderAttempt):
        raise TypeError("provider attempts must be ProviderAttempt")
    provider = attempt.provider if attempt.provider in {"tavily", "ddgs"} else "[redacted]"
    return {
        "provider": provider,
        "status": attempt.status,
        "count": attempt.count,
        "latency_ms": attempt.latency_ms,
        "query_id": _safe_log_identifier(attempt.query_id),
        "configured": attempt.configured,
        "available": attempt.available,
        "invocation_started": attempt.invocation_started,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("log values must be finite")
        return value
    raise TypeError(f"unsupported log value: {type(value).__name__}")


def _safe_log_identifier(value: Any) -> str:
    import re
    if not isinstance(value, str) or not re.fullmatch(r"(?:req-(?:[0-9]+|[0-9a-f]{32})|initial-[0-9]+|repair-[0-9]+|[Qq][0-9]+|[0-9a-fA-F]{16}|[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", value):
        return "[redacted]"
    return value
