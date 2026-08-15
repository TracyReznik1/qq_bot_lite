"""Closed data contracts for the bounded evidence-search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
import math
import re


class SearchTier(StrEnum):
    SKIP = "skip"
    LIGHT = "light"
    STANDARD = "standard"


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


class QueryOutcomeStatus(StrEnum):
    RESOLVED = "resolved"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class RetrievalBatchState(StrEnum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ALL_FAILED = "all_failed"


class ExcerptOrigin(StrEnum):
    PROVIDER_SNIPPET = "provider_snippet"
    PAGE_EXTRACT = "page_extract"
    DOCUMENT_EXTRACT = "document_extract"


class JudgeBatchStatus(StrEnum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"


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


class RepairReasonCode(StrEnum):
    MISSING_TOPIC = "missing_topic"
    STALE_EVIDENCE = "stale_evidence"
    SOURCE_CONFLICT = "source_conflict"
    ENTITY_AMBIGUITY = "entity_ambiguity"
    PREMISE_MISMATCH = "premise_mismatch"
    SOURCE_QUALITY_GAP = "source_quality_gap"
    CONTENT_UNREADABLE = "content_unreadable"


class JudgeAnomalyCode(StrEnum):
    """Closed, body-free parse anomalies from one Evidence Judge response."""

    MISSING_CANDIDATE = "missing_candidate"
    UNKNOWN_CANDIDATE = "unknown_candidate"
    MALFORMED_CANDIDATE = "malformed_candidate"
    DUPLICATE_CANDIDATE = "duplicate_candidate"


class RetrievalStopReason(StrEnum):
    EVIDENCE_SUFFICIENT = "evidence_sufficient"
    NO_REPAIR_BENEFIT = "no_repair_benefit"
    BUDGET_EXHAUSTED = "budget_exhausted"
    POST_REPAIR_STOP = "post_repair_stop"


class AnswerCertainty(StrEnum):
    VERIFIED = "verified"
    LIMITED = "limited"
    CONFLICTING = "conflicting"
    UNVERIFIED = "unverified"


class AnswerGenerationMode(StrEnum):
    PLAIN = "plain"
    GROUNDED = "grounded"
    FIXED = "fixed"


class AllowedClaimScope(StrEnum):
    ALL_SUPPORTED = "all_supported"
    SUPPORTED_SUBSET = "supported_subset"
    SUPPORTED_SUBSET_WITH_CONFLICTS = "supported_subset_with_conflicts"
    CONFLICT_DESCRIPTION_ONLY = "conflict_description_only"
    NO_EXTERNAL_FACTUAL_CLAIMS = "no_external_factual_claims"


class DisclosureCode(StrEnum):
    ONLINE_VERIFICATION_FAILED = "online_verification_failed"
    PARTIAL_EVIDENCE = "partial_evidence"
    SOURCE_CONFLICT = "source_conflict"
    VALIDATION_UNAVAILABLE = "validation_unavailable"
    VALIDATION_FAILED = "validation_failed"
    USER_FORBID_WEB = "user_forbid_web"


class WarningCode(StrEnum):
    HIGH_CONSEQUENCE = "high_consequence"


class ValidatorRequirement(StrEnum):
    NORMAL = "normal"
    FAIL_CLOSED = "fail_closed"


class ValidatorStatus(StrEnum):
    PASSED = "passed"
    FILTERED = "filtered"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"


class RenderOutcome(StrEnum):
    ANSWER = "answer"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    FAILURE = "failure"
    VALIDATION_FAILURE = "validation_failure"


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


def _gap_hint_pairs(values: Any) -> tuple[tuple[str, str], ...]:
    if values is None:
        return ()
    if isinstance(values, dict):
        values = (values,)
    if not isinstance(values, (tuple, list, set, frozenset)):
        raise TypeError("gap_hints must be a collection of reason/topic pairs")
    pairs: list[tuple[str, str]] = []
    for raw in values:
        if isinstance(raw, dict):
            raw = (raw.get("reason_code"), raw.get("target_topic_id"))
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise TypeError("gap_hints must contain reason/topic pairs")
        reason_code, target_topic_id = raw
        if (
            type(reason_code) is not str
            or type(target_topic_id) is not str
            or not reason_code.strip()
            or not target_topic_id.strip()
        ):
            raise ValueError("gap_hints must contain non-blank reason/topic pairs")
        pairs.append((reason_code.strip(), target_topic_id.strip()))
    return tuple(dict.fromkeys(pairs))


def _redaction_codes(values: Any, field_name: str) -> tuple[RedactionCode, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, (tuple, list, set, frozenset)):
        raise TypeError(f"{field_name} must be a collection of RedactionCode values")
    try:
        return tuple(RedactionCode(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only recognized redaction codes") from exc


def _repair_reason_codes(values: Any, field_name: str) -> tuple[RepairReasonCode, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, (tuple, list, set, frozenset)):
        raise TypeError(f"{field_name} must be a collection of RepairReasonCode values")
    codes = _tuple(values)
    _require_enum_values(codes, RepairReasonCode, field_name)
    return codes


def _trace_enum_values(
    values: Any,
    enum_type: type[StrEnum],
    field_name: str,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values,
        (tuple, list, set, frozenset),
    ):
        raise TypeError(f"{field_name} must be a collection of closed enum values")
    result = _tuple(values)
    _require_enum_values(result, enum_type, field_name)
    return result


def _validate_judge_anomalies(
    codes: tuple[JudgeAnomalyCode, ...],
    count: Any,
) -> None:
    if type(count) is not int or count < 0 or count > 8:
        raise ValueError("judge_anomaly_count must be an integer from zero through eight")
    if len(set(codes)) != len(codes):
        raise ValueError("judge_anomaly_codes must be unique")
    if count < len(codes):
        raise ValueError("judge_anomaly_count cannot be smaller than its code count")


def _trace_metadata_ids(values: Any, field_name: str) -> tuple[str, ...]:
    result = _topic_ids(values, field_name)
    for value in result:
        _require_safe_metadata(value, field_name)
    return tuple(dict.fromkeys(result))


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


def _require_opaque_topic_id(value: str, field_name: str = "topic_id") -> None:
    if not isinstance(value, str) or re.fullmatch(r"topic-[1-9][0-9]*", value) is None:
        raise ValueError(f"{field_name} must be an opaque topic-N identifier")


def _topic_ids(values: Any, field_name: str) -> tuple[str, ...]:
    result = _strings(values, field_name)
    for value in result:
        _require_opaque_topic_id(value, field_name)
    return result


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
    must_search: bool
    reason_codes: tuple[RetrievalComplexityCode, ...]

    def __post_init__(self) -> None:
        _normalize_fields(self, reason_codes=_tuple(self.reason_codes))
        _require_enum(self.route, SearchTier, "route")
        if type(self.must_search) is not bool:
            raise TypeError("must_search must be a boolean")
        if self.skip_reason is not None:
            _require_enum(self.skip_reason, SkipReason, "skip_reason")
        _require_enum_values(self.reason_codes, RetrievalComplexityCode, "reason_codes")
        if self.route is SearchTier.SKIP:
            if self.skip_reason is None:
                raise ValueError("skip route requires a skip_reason")
        elif self.skip_reason is not None:
            raise ValueError("search routes cannot have a skip_reason")

    @property
    def requires_clarification(self) -> bool:
        return (
            self.route is SearchTier.SKIP
            and self.skip_reason is SkipReason.USER_FORBID_WEB
            and self.must_search
        )


@dataclass(frozen=True)
class TierBudget:
    max_initial_queries: int
    max_candidate_urls: int
    max_content_reads: int
    max_repair_queries: int
    max_total_queries: int
    max_retrieval_rounds: int

    def __post_init__(self) -> None:
        values = self.__dict__.values()
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("tier budgets must be non-negative integers")
        if self.max_total_queries != self.max_initial_queries + self.max_repair_queries:
            raise ValueError("max_total_queries must equal initial plus repair queries")
        expected_rounds = 1 + int(self.max_repair_queries > 0)
        if self.max_retrieval_rounds != expected_rounds:
            raise ValueError("max_retrieval_rounds must match the repair budget")


DEFAULT_TIER_BUDGETS: Mapping[SearchTier, TierBudget] = MappingProxyType(
    {
        SearchTier.LIGHT: TierBudget(1, 5, 2, 0, 1, 1),
        SearchTier.STANDARD: TierBudget(3, 8, 5, 1, 4, 2),
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
        _require_opaque_topic_id(self.topic_id)
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
        required_topics = _records(
            _tuple(self.required_topics),
            RequiredTopic,
            "required_topics",
        )
        initial_queries = _records(
            self.initial_queries,
            SearchQuery,
            "initial_queries",
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
    reason_codes: tuple[RepairReasonCode, ...]
    target_topic_ids: tuple[str, ...]
    repair_query: SearchQuery | None
    query_redaction_codes: tuple[RedactionCode, ...] = ()

    def __post_init__(self) -> None:
        _normalize_fields(
            self,
            reason_codes=_repair_reason_codes(self.reason_codes, "reason_codes"),
            target_topic_ids=_topic_ids(self.target_topic_ids, "target_topic_ids"),
            query_redaction_codes=_redaction_codes(self.query_redaction_codes, "query_redaction_codes"),
        )
        if type(self.triggered) is not bool:
            raise TypeError("triggered must be a boolean")
        if self.triggered != (self.repair_query is not None):
            raise ValueError("repair_query must be present exactly when repair is triggered")
        if self.triggered:
            if not (self.reason_codes and self.target_topic_ids):
                raise ValueError("triggered repair requires non-empty reason codes and target topic ids")
        elif self.reason_codes or self.target_topic_ids:
            raise ValueError("untriggered repair cannot carry reason codes or target topic ids")


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
class JudgeVerdict:
    candidate_id: str
    supported_topic_ids: tuple[str, ...]
    freshness_by_topic: Mapping[str, FreshnessEligibility] = ()
    source_relation: SourceRelation = SourceRelation.UNKNOWN
    publisher_match: bool = False
    ownership_basis: str | None = None
    publisher: str | None = None
    publication_date: str | None = None
    conflict_key: str | None = None
    conflict_value: str | None = None
    conflict_relation: str | None = None

    def __post_init__(self) -> None:
        _normalize_fields(
            self,
            supported_topic_ids=_strings(self.supported_topic_ids, "supported_topic_ids"),
        )
        _require_enum(self.source_relation, SourceRelation, "source_relation")


@dataclass(frozen=True)
class JudgeBatchResult:
    rows: Mapping[str, JudgeVerdict]
    status: JudgeBatchStatus
    anomaly_codes: tuple[JudgeAnomalyCode, ...] = ()
    anomaly_count: int = 0
    gap_hints: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_enum(self.status, JudgeBatchStatus, "status")


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
    freshness_state: Freshness
    citable: bool
    safety_flags: tuple[str, ...]
    supported_topic_ids: tuple[str, ...]
    independence_group: str | None
    conflict_key: str | None = None
    conflict_value: str | None = None
    conflict_relation: str | None = None

    def __post_init__(self) -> None:
        _normalize_fields(
            self,
            safety_flags=_strings(self.safety_flags, "safety_flags"),
            supported_topic_ids=_strings(self.supported_topic_ids, "supported_topic_ids"),
        )
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
    topic_ids: tuple[str, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        _require_safe_metadata(self.conflict_id, "conflict_id")
        if not str(self.conflict_key or "").strip():
            raise ValueError("conflict_key is required")
        _normalize_fields(
            self,
            members=_records(self.members, EvidenceConflictMember, "members"),
            topic_ids=_topic_ids(self.topic_ids, "topic_ids"),
        )
        if len(self.members) < 2:
            raise ValueError("a conflict requires at least two members")
        if len({member.evidence_id for member in self.members}) < 2:
            raise ValueError("a conflict requires at least two distinct evidence ids")
        if len({str(member.value).strip() for member in self.members}) < 2:
            raise ValueError("a conflict requires at least two distinct asserted values")
        if not self.topic_ids or any(not topic_id.strip() for topic_id in self.topic_ids):
            raise ValueError("a conflict requires non-blank topic ids")
        if len(set(self.topic_ids)) != len(self.topic_ids):
            raise ValueError("conflict topic ids must be unique")


@dataclass(frozen=True)
class EvidenceGapAnalysis:
    missing_topic_ids: tuple[str, ...]
    conflict_group_ids: tuple[str, ...]
    repair_eligible: bool
    repair_reason_codes: tuple[RepairReasonCode, ...]
    repair_target_topic_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _normalize_fields(
            self,
            missing_topic_ids=_topic_ids(self.missing_topic_ids, "missing_topic_ids"),
            conflict_group_ids=_strings(self.conflict_group_ids, "conflict_group_ids"),
            repair_reason_codes=_repair_reason_codes(self.repair_reason_codes, "repair_reason_codes"),
            repair_target_topic_ids=_topic_ids(self.repair_target_topic_ids, "repair_target_topic_ids"),
        )
        if type(self.repair_eligible) is not bool:
            raise TypeError("repair_eligible must be a boolean")
        if self.repair_eligible:
            if not (self.repair_reason_codes and self.repair_target_topic_ids):
                raise ValueError("eligible repair requires non-empty reason codes and target topic ids")
        elif self.repair_reason_codes or self.repair_target_topic_ids:
            raise ValueError("ineligible repair cannot carry reason codes or target topic ids")


@dataclass(frozen=True)
class TopicAssessment:
    topic_id: str
    freshness: FreshnessEligibility
    supporting_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_opaque_topic_id(self.topic_id)
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
    gap_hints: tuple[tuple[str, str], ...] = field(kw_only=True, default=())
    judge_anomaly_codes: tuple[JudgeAnomalyCode, ...] = field(kw_only=True, default=())
    judge_anomaly_count: int = field(kw_only=True, default=0)

    def __post_init__(self) -> None:
        _require_record(self.decision, RetrievalDecision, "decision")
        _require_record(self.plan, SearchPlan, "plan")
        _require_record(self.gap_analysis, EvidenceGapAnalysis, "gap_analysis")
        _require_record(self.repair_plan, RepairPlan, "repair_plan")
        _normalize_fields(self, attempts=_records(self.attempts, ProviderAttempt, "attempts"), initial_evidence_ids=_strings(self.initial_evidence_ids, "initial_evidence_ids"), evidence_items=_records(self.evidence_items, EvidenceItem, "evidence_items"), missing_claim_topics=_strings(self.missing_claim_topics, "missing_claim_topics"), weak_source_topics=_strings(self.weak_source_topics, "weak_source_topics"), conflict_groups=_strings(self.conflict_groups, "conflict_groups"), limitations=_strings(self.limitations, "limitations"), conflicts=_records(self.conflicts, EvidenceConflict, "conflicts"), topic_assessments=_records(self.topic_assessments, TopicAssessment, "topic_assessments"), supported_topic_ids=_strings(self.supported_topic_ids, "supported_topic_ids"), missing_topic_ids=_strings(self.missing_topic_ids, "missing_topic_ids"), gap_hints=_gap_hint_pairs(self.gap_hints), judge_anomaly_codes=_trace_enum_values(self.judge_anomaly_codes, JudgeAnomalyCode, "judge_anomaly_codes"))
        _require_enum(self.evidence_state, EvidenceState, "evidence_state")
        _validate_judge_anomalies(
            self.judge_anomaly_codes,
            self.judge_anomaly_count,
        )
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
            if any(topic_id not in material_topic_ids for topic_id in conflict.topic_ids):
                raise ValueError("conflict topic ids must be material plan topic ids")
            if any(topic_id not in self.missing_topic_ids for topic_id in conflict.topic_ids):
                raise ValueError("conflict topic ids must be unresolved material topics")
            for member in conflict.members:
                item = evidence_by_id.get(member.evidence_id)
                if item is None:
                    raise ValueError("conflict members must reference bundle evidence")
                if not item.citable:
                    raise ValueError("conflict members require citable evidence")
            for topic_id in conflict.topic_ids:
                topic = next(
                    topic for topic in material_topics if topic.topic_id == topic_id
                )
                supporting_member_ids = {
                    member.evidence_id
                    for member in conflict.members
                    if topic.topic_id in evidence_by_id[member.evidence_id].supported_topic_ids
                }
                if len(supporting_member_ids) < 2:
                    raise ValueError(
                        "conflict topic must be supported by at least two conflict members"
                    )
        for topic in material_topics:
            assessment = assessment_by_id[topic.topic_id]
            for evidence_id in assessment.supporting_evidence_ids:
                item = evidence_by_id.get(evidence_id)
                if item is None:
                    raise ValueError("topic assessments must reference bundle evidence")
                if not item.citable:
                    raise ValueError("topic support requires citable evidence")
                if topic.topic_id not in item.supported_topic_ids:
                    raise ValueError("topic support must match the supported topic ids")
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
    query_index: int = 0
    configured: bool = True
    available: bool = True
    invocation_started: bool = True

    def __post_init__(self) -> None:
        _require_enum(self.status, ProviderStatus, "status")
        _require_safe_metadata(self.provider, "provider")
        if self.query_id:
            _require_safe_metadata(self.query_id, "query_id")
        if type(self.query_index) is not int or self.query_index < 0:
            raise ValueError("query_index must be a non-negative integer")
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
class QueryOutcome:
    query: SearchQuery
    status: QueryOutcomeStatus
    hits: tuple[ProviderHit, ...] = ()
    attempts: tuple[ProviderAttempt, ...] = ()
    readiness_failure: SearchFailureCode | None = None

    def __post_init__(self) -> None:
        _require_record(self.query, SearchQuery, "query")
        _require_enum(self.status, QueryOutcomeStatus, "status")
        _normalize_fields(
            self,
            hits=_records(self.hits, ProviderHit, "hits"),
            attempts=_records(self.attempts, ProviderAttempt, "attempts"),
        )
        if self.readiness_failure is not None:
            _require_enum(self.readiness_failure, SearchFailureCode, "readiness_failure")
            if self.readiness_failure not in {
                SearchFailureCode.PROVIDER_NOT_CONFIGURED,
                SearchFailureCode.PROVIDER_UNAVAILABLE,
            }:
                raise ValueError("readiness_failure must be PROVIDER_NOT_CONFIGURED or PROVIDER_UNAVAILABLE")
        if self.status is QueryOutcomeStatus.RESOLVED:
            if not self.hits:
                raise ValueError("resolved QueryOutcome requires hits")
        else:
            if self.hits:
                raise ValueError("non-resolved QueryOutcome cannot have hits")

    @property
    def query_index(self) -> int:
        return self.query.query_index

    @property
    def round_kind(self) -> SearchRoundKind:
        return self.query.round_kind

    @property
    def resolved(self) -> bool:
        return self.status is QueryOutcomeStatus.RESOLVED


@dataclass(frozen=True)
class QueryBatchResult:
    outcomes: tuple[QueryOutcome, ...]
    state: RetrievalBatchState

    def __post_init__(self) -> None:
        outcomes = _records(self.outcomes, QueryOutcome, "outcomes")
        _require_enum(self.state, RetrievalBatchState, "state")
        if not outcomes:
            _normalize_fields(self, outcomes=outcomes)
            if self.state is not RetrievalBatchState.ALL_FAILED:
                raise ValueError("Empty QueryBatchResult must have ALL_FAILED state")
            return
        indexes = tuple(outcome.query_index for outcome in outcomes)
        if len(set(indexes)) != len(indexes):
            raise ValueError("QueryBatchResult outcomes must have unique query_index values")
        if indexes != tuple(sorted(indexes)):
            raise ValueError("QueryBatchResult outcomes must be sorted by query_index")
        _normalize_fields(self, outcomes=outcomes)

        resolved_count = sum(outcome.resolved for outcome in outcomes)
        expected_state = (
            RetrievalBatchState.SUCCESS
            if resolved_count == len(outcomes)
            else RetrievalBatchState.PARTIAL_SUCCESS
            if resolved_count > 0
            else RetrievalBatchState.ALL_FAILED
        )
        if self.state is not expected_state:
            raise ValueError(f"QueryBatchResult state {self.state} does not match outcome resolution {expected_state}")

    @property
    def total_count(self) -> int:
        return len(self.outcomes)

    @property
    def total_query_count(self) -> int:
        return len(self.outcomes)

    @property
    def resolved_count(self) -> int:
        return sum(outcome.resolved for outcome in self.outcomes)

    @property
    def resolved_query_count(self) -> int:
        return self.resolved_count

    @property
    def unresolved_count(self) -> int:
        return sum(not outcome.resolved for outcome in self.outcomes)

    @property
    def unresolved_query_count(self) -> int:
        return self.unresolved_count


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


class ReadOutcomeStatus(StrEnum):
    READABLE = "readable"
    UNREADABLE = "unreadable"
    TIMEOUT = "timeout"
    UNSAFE_URL = "unsafe_url"
    UNSUPPORTED_TYPE = "unsupported_type"


@dataclass(frozen=True)
class ReadOutcome:
    hit: ProviderHit
    status: ReadOutcomeStatus
    candidate: EvidenceCandidate | None
    read_attempted: bool
    latency_ms: int | float = 0

    def __post_init__(self) -> None:
        _require_record(self.hit, ProviderHit, "hit")
        _require_enum(self.status, ReadOutcomeStatus, "status")
        if self.candidate is not None:
            _require_record(self.candidate, EvidenceCandidate, "candidate")
        if type(self.read_attempted) is not bool:
            raise TypeError("read_attempted must be a boolean")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be a finite non-negative number")

        if self.status is ReadOutcomeStatus.READABLE:
            if self.candidate is None or not self.candidate.excerpt:
                raise ValueError("READABLE read outcome requires a candidate with an excerpt")
        elif self.status is ReadOutcomeStatus.UNSAFE_URL:
            if self.candidate is not None:
                raise ValueError("UNSAFE_URL read outcome cannot carry a candidate")

    @property
    def readable(self) -> bool:
        return self.status is ReadOutcomeStatus.READABLE


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
    status: ValidatorStatus = ValidatorStatus.PASSED
    effective_certainty: AnswerCertainty = AnswerCertainty.VERIFIED
    effective_claim_scope: AllowedClaimScope = AllowedClaimScope.ALL_SUPPORTED

    def __post_init__(self) -> None:
        _require_record(self.draft, GroundedDraft, "draft")
        _require_enum(self.status, ValidatorStatus, "status")
        _require_enum(self.effective_certainty, AnswerCertainty, "effective_certainty")
        _require_enum(self.effective_claim_scope, AllowedClaimScope, "effective_claim_scope")
        labels = dict(self.claim_labels)
        if any(type(key) is not str or not isinstance(value, SupportLabel) for key, value in labels.items()):
            raise TypeError("claim_labels must map strings to SupportLabel")
        _normalize_fields(self, retained_blocks=_records(self.retained_blocks, AnswerBlock, "retained_blocks"), retained_claims=_records(self.retained_claims, Claim, "retained_claims"), removed_block_ids=_strings(self.removed_block_ids, "removed_block_ids"), claim_labels=MappingProxyType(labels), limitations=_strings(self.limitations, "limitations"))
        retained_ids = {block.block_id for block in self.retained_blocks}
        overlap = retained_ids.intersection(self.removed_block_ids)
        if overlap:
            raise ValueError("retained and removed block sets must be disjoint")


@dataclass(frozen=True)
class RenderState:
    outcome: RenderOutcome
    visible_blocks: tuple[AnswerBlock, ...]
    visible_claims: tuple[Claim, ...]
    citation_map: Mapping[str, int]
    used_sources: tuple[EvidenceItem, ...]
    conflict_groups: tuple[EvidenceConflict, ...]
    disclosure_codes: tuple[DisclosureCode, ...]
    warning_codes: tuple[WarningCode, ...]

    def __post_init__(self) -> None:
        _require_enum(self.outcome, RenderOutcome, "outcome")
        citation_map = dict(self.citation_map)
        if any(
            type(key) is not str or type(value) is not int or value <= 0
            for key, value in citation_map.items()
        ):
            raise TypeError("citation_map must map strings to positive integers")
        _normalize_fields(
            self,
            visible_blocks=_records(self.visible_blocks, AnswerBlock, "visible_blocks"),
            visible_claims=_records(self.visible_claims, Claim, "visible_claims"),
            citation_map=MappingProxyType(citation_map),
            used_sources=_records(self.used_sources, EvidenceItem, "used_sources"),
            conflict_groups=_records(self.conflict_groups, EvidenceConflict, "conflict_groups"),
            disclosure_codes=_tuple(self.disclosure_codes),
            warning_codes=_tuple(self.warning_codes),
        )
        _require_enum_values(self.disclosure_codes, DisclosureCode, "disclosure_codes")
        _require_enum_values(self.warning_codes, WarningCode, "warning_codes")
        _validate_render_closure(self)


def _validate_render_closure(state: RenderState) -> None:
    """Require deterministic citations and sources to close over the view."""
    block_by_id = {block.block_id: block for block in state.visible_blocks}
    if len(block_by_id) != len(state.visible_blocks):
        raise ValueError("visible block ids must be unique")
    claims_by_id = {claim.claim_id: claim for claim in state.visible_claims}
    if len(claims_by_id) != len(state.visible_claims):
        raise ValueError("visible claim ids must be unique")
    for claim in state.visible_claims:
        block = block_by_id.get(claim.block_id)
        if block is None or claim.claim_id not in block.claim_ids:
            raise ValueError("visible claims must belong to visible blocks")

    source_by_id = {item.evidence_id: item for item in state.used_sources}
    if len(source_by_id) != len(state.used_sources):
        raise ValueError("used source evidence ids must be unique")
    if any(
        not item.citable or not isinstance(item.url, str) or not item.url.startswith(("http://", "https://"))
        for item in state.used_sources
    ):
        raise ValueError("used sources must be citable HTTP evidence")

    citation_ids = set(state.citation_map)
    source_ids = set(source_by_id)
    if citation_ids != source_ids:
        raise ValueError("citation map and used sources must reference the same evidence")
    numbers = tuple(state.citation_map.values())
    if len(set(numbers)) != len(numbers) or set(numbers) != set(range(1, len(numbers) + 1)):
        raise ValueError("citation numbers must be unique and contiguous")

    required_ids = {
        evidence_id
        for claim in state.visible_claims
        for evidence_id in claim.evidence_ids
    }
    for conflict in state.conflict_groups:
        required_ids.update(member.evidence_id for member in conflict.members)
    if required_ids != citation_ids:
        raise ValueError("citations must close exactly over visible claims and conflicts")


@dataclass(frozen=True)
class RenderedReply:
    text: str
    chunks: tuple[str, ...]
    used_evidence_ids: tuple[str, ...]
    shown_source_urls: tuple[str, ...]
    degradation_disclosures: tuple[str, ...]

    def __post_init__(self) -> None:
        _normalize_fields(self, chunks=_strings(self.chunks, "chunks"), used_evidence_ids=_strings(self.used_evidence_ids, "used_evidence_ids"), shown_source_urls=_strings(self.shown_source_urls, "shown_source_urls"), degradation_disclosures=_strings(self.degradation_disclosures, "degradation_disclosures"))


@dataclass(frozen=True)
class QueryTraceEntry:
    """Body-free metadata for one provider attempt of one semantic query."""

    query_index: int
    purpose: QueryPurpose
    round_kind: SearchRoundKind
    provider: str
    status: ProviderStatus
    latency_ms: int | float

    def __post_init__(self) -> None:
        if type(self.query_index) is not int or self.query_index <= 0:
            raise ValueError("query_index must be a positive integer")
        _require_enum(self.purpose, QueryPurpose, "purpose")
        _require_enum(self.round_kind, SearchRoundKind, "round_kind")
        _require_enum(self.status, ProviderStatus, "status")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-blank string")
        _require_number(self.latency_ms, "latency_ms")
        _normalize_fields(self, provider=self.provider.strip())


@dataclass(frozen=True)
class TopicFreshnessTraceEntry:
    """Body-free, opaque topic identifier plus its closed freshness result."""

    topic_id: str
    freshness: FreshnessEligibility

    def __post_init__(self) -> None:
        _require_opaque_topic_id(self.topic_id)
        _require_enum(self.freshness, FreshnessEligibility, "freshness")


@dataclass
class SearchTrace:
    request_id: str
    request_source: RequestSource
    route: SearchTier
    skip_reason: SkipReason | None = None
    orchestrator_started: bool = False
    initial_query_count: int = 0
    initial_round_started: bool = False
    adaptive_repair_round_started: bool = False
    retrieval_round_count: int = 0
    executed_queries: tuple[QueryTraceEntry, ...] = ()
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
    retrieval_stop_reason: RetrievalStopReason | None = None
    must_search: bool = False
    retrieval_reason_codes: tuple[RetrievalComplexityCode, ...] = ()
    repair_reason_codes: tuple[RepairReasonCode, ...] = ()
    repair_target_topic_ids: tuple[str, ...] = ()
    supported_topic_ids: tuple[str, ...] = ()
    missing_topic_ids: tuple[str, ...] = ()
    topic_freshness: tuple[TopicFreshnessTraceEntry, ...] = ()
    judge_anomaly_codes: tuple[JudgeAnomalyCode, ...] = ()
    judge_anomaly_count: int = 0
    answer_generation_mode: AnswerGenerationMode | None = None
    answer_certainty: AnswerCertainty | None = None
    answer_claim_scope: AllowedClaimScope | None = None
    answer_disclosure_codes: tuple[DisclosureCode, ...] = ()
    answer_warning_codes: tuple[WarningCode, ...] = ()
    validator_status: ValidatorStatus | None = None
    validator_retained_claim_count: int = 0
    validator_removed_block_count: int = 0
    render_outcome: RenderOutcome | None = None
    render_citation_count: int = 0
    render_source_count: int = 0
    response_started_at: float | None = field(default=None, repr=False, compare=False)
    response_finished_at: float | None = field(default=None, repr=False, compare=False)
    finalized: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.executed_queries = _tuple(self.executed_queries)
        if any(
            not isinstance(entry, QueryTraceEntry)
            for entry in self.executed_queries
        ):
            raise TypeError("executed_queries must contain QueryTraceEntry values")
        self.provider_attempts = _tuple(self.provider_attempts)
        self.provider_failures = _tuple(self.provider_failures)
        self.initial_query_redaction_codes = _redaction_codes(
            self.initial_query_redaction_codes, "initial_query_redaction_codes"
        )
        self.adaptive_repair_redaction_codes = _redaction_codes(
            self.adaptive_repair_redaction_codes, "adaptive_repair_redaction_codes"
        )
        if type(self.must_search) is not bool:
            raise TypeError("must_search must be a boolean")
        self.retrieval_reason_codes = _trace_enum_values(
            self.retrieval_reason_codes,
            RetrievalComplexityCode,
            "retrieval_reason_codes",
        )
        self.repair_reason_codes = _trace_enum_values(
            self.repair_reason_codes,
            RepairReasonCode,
            "repair_reason_codes",
        )
        self.repair_target_topic_ids = _trace_metadata_ids(
            self.repair_target_topic_ids,
            "repair_target_topic_ids",
        )
        self.supported_topic_ids = _trace_metadata_ids(
            self.supported_topic_ids,
            "supported_topic_ids",
        )
        self.missing_topic_ids = _trace_metadata_ids(
            self.missing_topic_ids,
            "missing_topic_ids",
        )
        self.topic_freshness = _records(
            self.topic_freshness,
            TopicFreshnessTraceEntry,
            "topic_freshness",
        )
        self.judge_anomaly_codes = _trace_enum_values(
            self.judge_anomaly_codes,
            JudgeAnomalyCode,
            "judge_anomaly_codes",
        )
        _validate_judge_anomalies(
            self.judge_anomaly_codes,
            self.judge_anomaly_count,
        )
        for enum_value, enum_type, field_name in (
            (self.answer_generation_mode, AnswerGenerationMode, "answer_generation_mode"),
            (self.answer_certainty, AnswerCertainty, "answer_certainty"),
            (self.answer_claim_scope, AllowedClaimScope, "answer_claim_scope"),
            (self.validator_status, ValidatorStatus, "validator_status"),
            (self.render_outcome, RenderOutcome, "render_outcome"),
        ):
            if enum_value is not None:
                _require_enum(enum_value, enum_type, field_name)
        self.answer_disclosure_codes = _trace_enum_values(
            self.answer_disclosure_codes,
            DisclosureCode,
            "answer_disclosure_codes",
        )
        self.answer_warning_codes = _trace_enum_values(
            self.answer_warning_codes,
            WarningCode,
            "answer_warning_codes",
        )
        for field_name in (
            "validator_retained_claim_count",
            "validator_removed_block_count",
            "render_citation_count",
            "render_source_count",
            "judge_anomaly_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for attempt in self.provider_attempts:
            if not isinstance(attempt, ProviderAttempt):
                raise TypeError("provider_attempts must contain ProviderAttempt values")
        if self.retrieval_stop_reason is not None:
            _require_enum(
                self.retrieval_stop_reason,
                RetrievalStopReason,
                "retrieval_stop_reason",
            )

    @property
    def semantic_query_count(self) -> int:
        return len({entry.query_index for entry in self.executed_queries})

    @property
    def repair_query_count(self) -> int:
        return len(
            {
                entry.query_index
                for entry in self.executed_queries
                if entry.round_kind is SearchRoundKind.REPAIR
            }
        )

    def to_log_dict(self) -> dict[str, Any]:
        try:
            judge_anomaly_codes = _trace_enum_values(
                self.judge_anomaly_codes,
                JudgeAnomalyCode,
                "judge_anomaly_codes",
            )
            _validate_judge_anomalies(
                judge_anomaly_codes,
                self.judge_anomaly_count,
            )
            judge_anomaly_count = self.judge_anomaly_count
        except (TypeError, ValueError):
            judge_anomaly_codes = ()
            judge_anomaly_count = 0
        executed_metadata = tuple(
            {
                "query_index": entry.query_index,
                "purpose": entry.purpose,
                "round_kind": entry.round_kind,
                "provider": (
                    entry.provider
                    if entry.provider in {"tavily", "ddgs"}
                    else "[redacted]"
                ),
                "status": entry.status,
                "latency_ms": entry.latency_ms,
            }
            for entry in self.executed_queries
        )
        values = {
            "request_id": _safe_log_identifier(self.request_id),
            "request_source": self.request_source,
            "route": self.route,
            "skip_reason": self.skip_reason,
            "orchestrator_started": self.orchestrator_started,
            "initial_query_count": self.initial_query_count,
            "initial_round_started": self.initial_round_started,
            "adaptive_repair_round_started": self.adaptive_repair_round_started,
            "initial_query_redaction_codes": _safe_redaction_codes(self.initial_query_redaction_codes),
            "adaptive_repair_redaction_codes": _safe_redaction_codes(self.adaptive_repair_redaction_codes),
            "retrieval_round_count": self.retrieval_round_count,
            "executed_queries": executed_metadata,
            "retrieval_stop_reason": self.retrieval_stop_reason,
            "must_search": self.must_search,
            "retrieval_reason_codes": self.retrieval_reason_codes,
            "repair_reason_codes": self.repair_reason_codes,
            "repair_target_topic_ids": self.repair_target_topic_ids,
            "supported_topic_ids": self.supported_topic_ids,
            "missing_topic_ids": self.missing_topic_ids,
            "topic_freshness": tuple(
                {
                    "topic_id": entry.topic_id,
                    "freshness": entry.freshness,
                }
                for entry in self.topic_freshness
            ),
            "judge_anomaly_codes": judge_anomaly_codes,
            "judge_anomaly_count": judge_anomaly_count,
            "answer_generation_mode": self.answer_generation_mode,
            "answer_certainty": self.answer_certainty,
            "answer_claim_scope": self.answer_claim_scope,
            "answer_disclosure_codes": self.answer_disclosure_codes,
            "answer_warning_codes": self.answer_warning_codes,
            "validator_status": self.validator_status,
            "validator_retained_claim_count": self.validator_retained_claim_count,
            "validator_removed_block_count": self.validator_removed_block_count,
            "render_outcome": self.render_outcome,
            "render_citation_count": self.render_citation_count,
            "render_source_count": self.render_source_count,
            "finalized": self.finalized,
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
            "semantic_query_count": self.semantic_query_count,
            "repair_query_count": self.repair_query_count,
            "content_read_count": self.content_read_count,
            "provider_attempted": self.provider_invocation_started,
            "sufficient_evidence": self.evidence_state is EvidenceState.SUFFICIENT,
        }
        return _json_safe(values)


@dataclass(frozen=True)
class AnswerState:
    """Closed, transient answer/render policy over immutable search state.

    This is a pure chat-dispatch output: it never enters retrieval decisions and
    is not consumed by Router, Planner, Evidence, Validator semantics or Renderer
    beyond what Task 8/9 wire explicitly.
    """

    evidence_state: EvidenceState | None
    generation_mode: AnswerGenerationMode
    certainty: AnswerCertainty
    allowed_claim_scope: AllowedClaimScope
    disclosure_codes: tuple[DisclosureCode, ...]
    warning_codes: tuple[WarningCode, ...]
    validator_requirement: ValidatorRequirement

    def __post_init__(self) -> None:
        if self.evidence_state is not None:
            _require_enum(self.evidence_state, EvidenceState, "evidence_state")
        _require_enum(self.generation_mode, AnswerGenerationMode, "generation_mode")
        _require_enum(self.certainty, AnswerCertainty, "certainty")
        _require_enum(self.allowed_claim_scope, AllowedClaimScope, "allowed_claim_scope")
        _require_enum(self.validator_requirement, ValidatorRequirement, "validator_requirement")
        _normalize_fields(
            self,
            disclosure_codes=_tuple(self.disclosure_codes),
            warning_codes=_tuple(self.warning_codes),
        )
        _require_enum_values(self.disclosure_codes, DisclosureCode, "disclosure_codes")
        _require_enum_values(self.warning_codes, WarningCode, "warning_codes")


@dataclass(frozen=True)
class SearchPipelineResult:
    decision: RetrievalDecision
    plan: SearchPlan | None
    evidence: EvidenceBundle | None
    trace: SearchTrace
    failure_code: SearchFailureCode | None = None
    analysis: RequestAnalysis = field(kw_only=True)

    def __post_init__(self) -> None:
        _require_record(self.analysis, RequestAnalysis, "analysis")
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


def _attempt_metadata(attempt: ProviderAttempt) -> dict[str, Any]:
    if not isinstance(attempt, ProviderAttempt):
        raise TypeError("provider attempts must be ProviderAttempt")
    provider = attempt.provider if attempt.provider in {"tavily", "ddgs"} else "[redacted]"
    return {
        "provider": provider,
        "status": attempt.status,
        "count": attempt.count,
        "latency_ms": attempt.latency_ms,
        "query_index": attempt.query_index,
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
