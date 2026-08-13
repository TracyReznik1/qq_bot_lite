"""Evidence-search evaluation CLI.

The evaluator keeps deterministic Trace acceptance separate from model-quality
measurements.  Human labels are read only from case/audit files and are joined
to predictions or traces by ``case_id``; embedded labels in traces are ignored.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sys
import unicodedata
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "eval" / "search" / "cases.jsonl"
PROVIDER_RECORDINGS_PATH = ROOT / "eval" / "search" / "provider_recordings.jsonl"
MODEL_PREDICTIONS_PATH = ROOT / "eval" / "search" / "model_predictions.jsonl"

CATEGORY_QUOTAS = {
    "no_benefit": 20,
    "stable_fact": 20,
    "explanation_comparison": 25,
    "dynamic_fact": 20,
    "regulated_controversy": 15,
    "explicit_search": 10,
    "ambiguous_mixed": 10,
    "failure_partial_conflict": 20,
}

TIERS = ("light", "standard")
ROUTES = ("skip", *TIERS)
SKIP_REASONS = {
    "user_forbid_web",
    "social_or_emotional",
    "creative_or_roleplay",
    "provided_text_transform",
    "provided_content_summary",
    "pure_math",
    "closed_logic",
    "closed_context_only",
}
SKIP_REASON_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "user_forbid_web": {
        "required_triggers": {"explicit_no_web"},
        "allowed_triggers": {
            "explicit_no_web", "explicit_search", "high_consequence_action",
        },
        "factuality": "ambiguous",
        "allowed_degradations": {None, "user_forbid_web"},
    },
    "provided_text_transform": {
        "required_triggers": set(), "allowed_triggers": set(),
        "factuality": "mixed", "allowed_degradations": {None},
    },
    "provided_content_summary": {
        "required_triggers": set(), "allowed_triggers": set(),
        "factuality": "mixed", "allowed_degradations": {None},
    },
    **{
        reason: {
            "required_triggers": set(), "allowed_triggers": set(),
            "factuality": "non_factual", "allowed_degradations": {None},
        }
        for reason in (
            "social_or_emotional", "creative_or_roleplay", "pure_math",
            "closed_logic", "closed_context_only",
        )
    },
}
ACTIONABILITY = {"none", "general", "personalized"}
POTENTIAL_HARM = {"none", "low", "high"}
QUERY_PURPOSES = {
    "direct", "primary", "independent", "time_bounded", "disambiguation",
    "counterevidence", "repair",
}
SOURCE_RELATIONS = {"primary", "independent", "secondary", "community", "unknown"}
EXPECTED_OUTCOMES = {"skip", "grounded_answer", "degraded", "conflict"}
LABEL_STATUSES = {"reviewed", "unreviewed"}
PREDICTION_COMPONENTS = {
    "router", "planner", "relevance", "claim_discovery", "semantic_support",
}
QUALITY_COMPONENTS = ("claim_discovery", "semantic_support", "relevance")
SUPPORT_LABELS = {"supported", "partial", "conflict", "unsupported", "unmapped"}
RELEVANCE_LABELS = {
    "relevant", "irrelevant", "direct", "contextual", "admitted", "excluded",
    "pass", "fail",
}
DISCOVERY_LABELS = {"present", "absent"}
EVIDENCE_STATES = {"sufficient", "partial", "conflicting", "insufficient"}
PROVIDER_STATUSES = {"success", "empty", "timeout", "error", "not_configured", "unavailable"}
FAILURE_CODES = {
    "provider_not_configured", "provider_unavailable", "provider_timeout",
    "no_results", "content_unreadable", "insufficient_evidence",
    "partial_evidence", "source_conflict", "validation_failed", "user_forbid_web",
}

CASE_FIELDS = {
    "case_id", "category", "question", "allow_skip", "skip_reason",
    "minimum_tier", "external_fact_required", "actionability", "potential_harm",
    "expected_query_purposes", "expected_initial_query_min",
    "expected_initial_query_max", "expected_max_rounds", "material_claim_spans",
    "acceptable_source_relations", "semantic_labels", "expected_outcome",
    "fixture_id", "label_status", "reviewed_by", "reviewed_at",
}
CASE_OPTIONAL_FIELDS = {"dynamic", "high_consequence", "expected_final_tier", "acceptable_final_tiers"}
RECORDING_FIELDS = {
    "fixture_id", "case_id", "provider", "query_text", "title", "url",
    "excerpt", "expected_fetch_status",
}
PREDICTION_COMMON_FIELDS = {
    "case_id", "component", "model", "model_version", "prompt_schema_version",
    "run_timestamp",
}
PREDICTION_FIELDS = {
    "router": PREDICTION_COMMON_FIELDS | {"predicted_tier"},
    "planner": PREDICTION_COMMON_FIELDS | {
        "predicted_query_purposes", "predicted_initial_query_count", "predicted_repair_used",
    },
    "claim_discovery": PREDICTION_COMMON_FIELDS | {"predictions"},
    "semantic_support": PREDICTION_COMMON_FIELDS | {"predictions"},
    "relevance": PREDICTION_COMMON_FIELDS | {"predictions"},
}
HUMAN_LABEL_FIELDS = {
    "question", "allow_skip", "skip_reason", "minimum_tier",
    "external_fact_required", "actionability", "potential_harm",
    "expected_query_purposes", "expected_initial_query_min",
    "expected_initial_query_max", "expected_max_rounds", "material_claim_spans",
    "acceptable_source_relations", "semantic_labels", "expected_outcome",
    "label_status", "reviewed_by", "reviewed_at", "d_factual",
}

TIER_BUDGETS: Mapping[str, Mapping[str, int]] = {
    "light": {
        "initial_query_count": 1,
        "candidate_url_count": 5,
        "content_read_count": 2,
        "semantic_query_count": 1,
        "repair_query_count": 0,
        "retrieval_round_count": 1,
        "hard_timeout_ms": 8_000,
    },
    "standard": {
        "initial_query_count": 3,
        "candidate_url_count": 8,
        "content_read_count": 5,
        "semantic_query_count": 4,
        "repair_query_count": 1,
        "retrieval_round_count": 2,
        "hard_timeout_ms": 20_000,
    },
}

LATENCY_FIELDS = (
    "route_latency_ms",
    "query_planning_latency_ms",
    "initial_provider_search_latency_ms",
    "provider_search_total_latency_ms",
    "initial_content_read_latency_ms",
    "content_read_total_latency_ms",
    "initial_evidence_assembly_latency_ms",
    "evidence_assembly_total_latency_ms",
    "gap_analysis_latency_ms",
    "adaptive_repair_latency_ms",
    "answer_generation_latency_ms",
    "structural_validation_latency_ms",
    "semantic_validation_latency_ms",
    "qq_render_latency_ms",
    "retrieval_pipeline_latency_ms",
    "total_response_latency_ms",
)

REQUEST_SOURCES = {"chat", "command", "compatibility"}
FACTUALITIES = {"non_factual", "factual", "mixed", "ambiguous"}
TRIGGER_CODES = {
    "explicit_no_web", "explicit_search", "explicit_verification",
    "explicit_source_request", "freshness_marker", "dynamic_attribute",
    "regulated_domain_foundation", "high_consequence_action",
    "current_rule_or_policy", "controversy_or_conflict",
    "external_fact_explanation_or_comparison", "recommendation_or_evaluation",
    "ambiguous_entity", "multi_hop_complexity", "mixed_task", "factual_default",
    "classifier_uncertain",
}
REDACTION_CODES = {
    "cq_control_code", "data_url", "callback_secret", "one_time_code",
    "password", "bank_account", "card_cvv", "hard_secret", "phone_number",
    "email_address", "empty_after_redaction", "invalid_redaction_code",
}
PROVIDER_NAMES = {"tavily", "ddgs", "[redacted]"}
QUERY_FIELDS = {"query_id", "purpose"}
PROVIDER_ATTEMPT_FIELDS = {
    "provider", "status", "count", "latency_ms", "query_id", "configured",
    "available", "invocation_started",
}
TRACE_FIELDS = {
    "request_id", "request_source", "route", "skip_reason",
    "orchestrator_started", "initial_query_count", "initial_round_started",
    "adaptive_repair_round_started",
    "initial_query_redaction_codes", "adaptive_repair_redaction_codes",
    "retrieval_round_count", "executed_queries", "provider_configured",
    "provider_attempts", "provider_invocation_started", "provider_failures",
    "candidate_url_count", "citable_evidence_count", "evidence_state",
    "repair_used", "claim_count", "supported_claim_count", "citation_count",
    "knowledge_fallback_used", "degradation_reason", "content_read_count",
    "provider_attempted", "sufficient_evidence", "semantic_query_count",
    "repair_query_count", *LATENCY_FIELDS,
}
AUDIT_FIELDS = {
    "case_id", "request_id", "category", "allow_skip", "skip_reason",
    "external_fact_required", "explicit_search", "dynamic", "high_consequence",
    "minimum_tier", "acceptable_final_tiers", "label_status", "reviewed_by",
    "reviewed_at", "claims", "evidence", "used_evidence_ids",
    "shown_source_urls", "missing_claim_topics", "conflict_groups",
    "rendered_disclosures", "stages_started",
}
CLAIM_AUDIT_FIELDS = {
    "claim_id", "material", "retained", "support_label", "evidence_ids",
    "topic_ids", "partial_topic_ids", "conflict_group_ids", "disclosure_codes",
}
EVIDENCE_AUDIT_FIELDS = {"evidence_id", "final_url", "relevance", "citable"}
CONFLICT_AUDIT_FIELDS = {"group_id", "member_evidence_ids"}
DISCLOSURE_CODES = {
    "partial_evidence", "source_conflict", "verification_failed",
    "dynamic_unverified", "provider_not_configured", "provider_unavailable",
    "provider_timeout", "no_results", "content_unreadable",
    "insufficient_evidence", "semantic_validation_unavailable",
    "user_forbid_web", "explicit_search_failure", "single_source_authority",
}
STAGE_TO_LATENCY = {
    "route": "route_latency_ms",
    "query_planning": "query_planning_latency_ms",
    "initial_provider_search": "initial_provider_search_latency_ms",
    "provider_search_total": "provider_search_total_latency_ms",
    "initial_content_read": "initial_content_read_latency_ms",
    "content_read_total": "content_read_total_latency_ms",
    "initial_evidence_assembly": "initial_evidence_assembly_latency_ms",
    "evidence_assembly_total": "evidence_assembly_total_latency_ms",
    "gap_analysis": "gap_analysis_latency_ms",
    "adaptive_repair": "adaptive_repair_latency_ms",
    "answer_generation": "answer_generation_latency_ms",
    "structural_validation": "structural_validation_latency_ms",
    "semantic_validation": "semantic_validation_latency_ms",
    "qq_render": "qq_render_latency_ms",
    "retrieval_pipeline": "retrieval_pipeline_latency_ms",
    "total_response": "total_response_latency_ms",
}
RUN_MANIFEST_FIELDS = {
    "schema_version", "run_id", "provenance", "data_source", "fixture_derived",
    "case_set_sha256", "recordings_sha256", "predictions_sha256",
    "run_timestamp", "attestation",
}
SAMPLE_MANIFEST_FIELDS = {
    "schema_version", "sample_id", "provenance", "fixture_derived", "collected_at",
    "traces_sha256", "audits_sha256", "attestation",
}
ATTESTATION_FIELDS = {"algorithm", "key_id", "signature"}
KNOWN_FIXTURE_PREDICTION_HASHES = {
    "597a2d237bbb02a12ef72b1b9d438a70f52e3c44a6885e98c20ceda423e5e1e0",
}
KNOWN_FIXTURE_MODEL_IDENTITIES = {"fixture-baseline"}

_SECRET_MARKERS = ("AIza", "sk-", "qq:1000")


# ── metric primitives ──────────────────────────────────────────────────

def prf1(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def tier_metrics(
    confusion: Mapping[tuple[str, str], int],
    *,
    labels: Iterable[str] = TIERS,
) -> dict[str, float]:
    labels = tuple(labels)
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = confusion.get((label, label), 0)
        fp = sum(count for (pred, true), count in confusion.items() if pred == label and true != label)
        fn = sum(count for (pred, true), count in confusion.items() if true == label and pred != label)
        per_label[label] = prf1(tp, fp, fn)
    result: dict[str, float] = {}
    for metric in ("precision", "recall", "f1"):
        result[f"macro_{metric}"] = (
            sum(per_label[label][metric] for label in labels) / len(labels)
            if labels else 0.0
        )
    for label in labels:
        for metric, value in per_label[label].items():
            result[f"{label}_{metric}"] = value
    return result


def _binary_quality(pairs: Sequence[tuple[str, str]], positive: set[str]) -> dict[str, Any]:
    if not pairs:
        return {
            "evaluable": False, "sample_count": 0, "precision": None,
            "recall": None, "f1": None,
        }
    tp = sum(expected in positive and predicted in positive for expected, predicted in pairs)
    fp = sum(expected not in positive and predicted in positive for expected, predicted in pairs)
    fn = sum(expected in positive and predicted not in positive for expected, predicted in pairs)
    return {"evaluable": True, "sample_count": len(pairs), **prf1(tp, fp, fn)}


def _multiclass_quality(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    if not pairs:
        return {
            "evaluable": False, "sample_count": 0, "macro_precision": None,
            "macro_recall": None, "macro_f1": None, "per_label": {},
        }
    labels = tuple(sorted({label for pair in pairs for label in pair}))
    confusion = Counter((predicted, expected) for expected, predicted in pairs)
    metrics = tier_metrics(confusion, labels=labels)
    per_label = {
        label: {
            metric: metrics[f"{label}_{metric}"]
            for metric in ("precision", "recall", "f1")
        }
        for label in labels
    }
    return {
        "evaluable": True,
        "sample_count": len(pairs),
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "per_label": per_label,
    }


def _nearest_rank(values: Sequence[float], percentile: float) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def latency_percentiles(
    traces: Iterable[Any],
    audits_by_request_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, float | int | bool | None]]:
    rows = tuple(traces)
    result: dict[str, dict[str, float | int | bool | None]] = {}
    for field in LATENCY_FIELDS:
        stage = next(name for name, latency in STAGE_TO_LATENCY.items() if latency == field)
        values = [
            value for row in rows
            if (
                audits_by_request_id is None
                or stage in (
                    audits_by_request_id.get(str(_field(row, "request_id")), {}).get(
                        "stages_started", []
                    )
                )
            )
            if _is_nonnegative_number(value := _field(row, field))
        ]
        result[field] = {
            "evaluable": bool(values),
            "sample_count": len(values),
            "p50": _nearest_rank(values, 0.50),
            "p95": _nearest_rank(values, 0.95),
            "p99": _nearest_rank(values, 0.99),
        }
    return result


# ── generic helpers ────────────────────────────────────────────────────

def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return raw if isinstance(raw, str) else ""


def _route(trace: Any) -> str:
    return _enum_text(_field(trace, "route"))


def _is_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _counter(trace: Any, name: str) -> int:
    value = _field(trace, name, 0)
    return value if _is_int(value) else 0


def _is_nonnegative_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_string_list(value: Any, *, allow_empty: bool = True) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(_is_nonempty_string(item) for item in value)
    )


def _closed(value: Any, allowed: set[str] | tuple[str, ...]) -> bool:
    return isinstance(value, str) and value in allowed


def _valid_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _valid_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and _canonical_hostname(value) is not None
    )


def _canonical_hostname(value: str) -> str | None:
    try:
        host = urlsplit(value).hostname
    except (TypeError, ValueError):
        return None
    if not isinstance(host, str) or not host:
        return None
    normalized = unicodedata.normalize("NFKC", host).casefold()
    if not normalized or any(
        unicodedata.category(character) in {"Cc", "Cf"}
        for character in normalized
    ):
        return None
    try:
        canonical = normalized.encode("idna").decode("ascii").casefold()
    except UnicodeError:
        return None
    if canonical.endswith("."):
        canonical = canonical[:-1]
    if not canonical or canonical.endswith("."):
        return None
    return canonical


def _canonical_identity(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized or any(
        unicodedata.category(character) in {"Cc", "Cf"}
        for character in normalized
    ):
        return None
    return normalized.casefold()


def _normalized_question(value: str) -> str:
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", value)
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _secret_errors(rows: Iterable[Mapping[str, Any]], artifact: str) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows, 1):
        payload = json.dumps(row, ensure_ascii=False)
        for marker in _SECRET_MARKERS:
            if marker in payload:
                errors.append(f"{artifact} row {index} contains forbidden secret marker {marker}")
    return errors


def _load_jsonl_checked(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"{path}: file does not exist"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"{path}: could not read file ({type(exc).__name__})"]
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path}:{line_number}: JSONL row must be an object")
            continue
        rows.append(value)
    return rows, errors


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows, errors = _load_jsonl_checked(path)
    if errors:
        raise ValueError("; ".join(errors))
    return rows


def _duplicate_values(values: Iterable[Any]) -> list[Any]:
    hashable_values: list[Any] = []
    for value in values:
        try:
            hash(value)
        except TypeError:
            continue
        hashable_values.append(value)
    counts = Counter(hashable_values)
    return [value for value, count in counts.items() if value and count > 1]


def _artifact_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _attestation_payload(manifest: Mapping[str, Any]) -> bytes:
    unsigned = dict(manifest)
    attestation = manifest.get("attestation")
    if isinstance(attestation, Mapping):
        unsigned["attestation"] = {
            key: value for key, value in attestation.items() if key != "signature"
        }
    return json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _validate_trusted_attestation(
    manifest: Mapping[str, Any] | None,
    trusted_verifier_key: bytes | bytearray | None,
    prefix: str,
) -> list[str]:
    if not isinstance(manifest, Mapping):
        return [f"{prefix} trusted attestation cannot be verified"]
    attestation = manifest.get("attestation")
    if not isinstance(attestation, Mapping):
        return [f"{prefix} trusted attestation is required"]
    errors: list[str] = []
    missing = sorted(ATTESTATION_FIELDS - set(attestation))
    unexpected = sorted(set(attestation) - ATTESTATION_FIELDS)
    if missing:
        errors.append(f"{prefix} attestation missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"{prefix} attestation unexpected fields: {', '.join(unexpected)}")
    if attestation.get("algorithm") != "hmac-sha256":
        errors.append(f"{prefix} attestation invalid algorithm")
    if not _normalized_identifier(attestation.get("key_id")):
        errors.append(f"{prefix} attestation invalid key_id")
    signature = attestation.get("signature")
    if not isinstance(signature, str) or len(signature) != 64 or any(
        character not in "0123456789abcdef" for character in signature
    ):
        errors.append(f"{prefix} attestation signature must be lowercase SHA-256 hex")
    if not isinstance(trusted_verifier_key, (bytes, bytearray)) or len(trusted_verifier_key) < 32:
        errors.append(f"{prefix} trusted verifier key is required and must be at least 32 bytes")
        return errors
    try:
        expected = hmac.new(
            bytes(trusted_verifier_key), _attestation_payload(manifest), hashlib.sha256,
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        errors.append(f"{prefix} attestation payload is invalid ({type(exc).__name__})")
        return errors
    if isinstance(signature, str) and not hmac.compare_digest(signature, expected):
        errors.append(f"{prefix} attestation signature verification failed")
    return errors


def _fixture_evidence_urls(audits: Sequence[Mapping[str, Any]]) -> list[str]:
    found: list[str] = []
    for audit in audits:
        evidence = audit.get("evidence") if isinstance(audit, Mapping) else None
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            url = item.get("final_url")
            if not isinstance(url, str):
                continue
            host = _canonical_hostname(url) or ""
            if (
                "fixture" in host
                or host in {"example", "example.com", "example.org", "example.net"}
                or host.endswith(".example")
                or host.endswith(".test")
            ):
                found.append(url)
    return found


def _normalized_identifier(value: Any) -> bool:
    if not _is_nonempty_string(value):
        return False
    normalized = unicodedata.normalize("NFKC", value).strip()
    return (
        value == normalized
        and value.casefold() != "unreviewed"
        and all(
            not character.isspace()
            and unicodedata.category(character) not in {"Cc", "Cf"}
            for character in value
        )
    )


def _validate_run_manifest(
    manifest: Mapping[str, Any] | None,
    cases: Sequence[Mapping[str, Any]],
    recordings: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    trusted_verifier_key: bytes | bytearray | None = None,
) -> list[str]:
    if not isinstance(manifest, Mapping):
        return ["run manifest is required"]
    errors: list[str] = []
    missing = sorted(RUN_MANIFEST_FIELDS - set(manifest))
    unexpected = sorted(set(manifest) - RUN_MANIFEST_FIELDS)
    if missing:
        errors.append(f"run manifest missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"run manifest unexpected fields: {', '.join(unexpected)}")
    if manifest.get("schema_version") != "search-eval-run-v1":
        errors.append("run manifest invalid schema_version")
    if not _normalized_identifier(manifest.get("run_id")):
        errors.append("run manifest invalid run_id")
    if manifest.get("provenance") not in {"independent_model_run", "fixture_baseline"}:
        errors.append("run manifest invalid provenance")
    if manifest.get("data_source") not in {"reviewed_cases", "synthetic_provider_fixtures"}:
        errors.append("run manifest invalid data_source")
    if type(manifest.get("fixture_derived")) is not bool:
        errors.append("run manifest fixture_derived must be boolean")
    if not _valid_timestamp(manifest.get("run_timestamp")):
        errors.append("run manifest invalid run_timestamp")
    if manifest.get("case_set_sha256") != _artifact_sha256(cases):
        errors.append("run manifest case_set_sha256 does not bind cases")
    if manifest.get("recordings_sha256") != _artifact_sha256(recordings):
        errors.append("run manifest recordings_sha256 does not bind provider recordings")
    if manifest.get("predictions_sha256") != _artifact_sha256(predictions):
        errors.append("run manifest predictions_sha256 does not bind predictions")
    fixture = manifest.get("fixture_derived") is True
    if manifest.get("provenance") == "fixture_baseline" and not fixture:
        errors.append("fixture_baseline provenance requires fixture_derived=true")
    if manifest.get("data_source") == "synthetic_provider_fixtures" and not fixture:
        errors.append("synthetic fixture data source requires fixture_derived=true")
    if manifest.get("provenance") == "independent_model_run" and fixture:
        errors.append("independent_model_run cannot be fixture-derived")
    if manifest.get("provenance") == "independent_model_run" and manifest.get("data_source") != "reviewed_cases":
        errors.append("independent_model_run requires reviewed_cases data source")
    if manifest.get("provenance") == "fixture_baseline" and manifest.get("data_source") != "synthetic_provider_fixtures":
        errors.append("fixture_baseline requires synthetic_provider_fixtures data source")
    if any(row.get("run_timestamp") != manifest.get("run_timestamp") for row in predictions):
        errors.append("run manifest run_timestamp does not match every prediction")
    run_signatures = {
        (row.get("model"), row.get("model_version"), row.get("prompt_schema_version"))
        for row in predictions
    }
    if len(run_signatures) > 1:
        errors.append("run manifest binds predictions from multiple model runs")
    errors.extend(_validate_trusted_attestation(manifest, trusted_verifier_key, "run manifest"))
    return errors


def _validate_sample_manifest(
    manifest: Mapping[str, Any] | None,
    traces: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    trusted_verifier_key: bytes | bytearray | None = None,
) -> list[str]:
    if not isinstance(manifest, Mapping):
        return ["controlled sample manifest is required"]
    errors: list[str] = []
    missing = sorted(SAMPLE_MANIFEST_FIELDS - set(manifest))
    unexpected = sorted(set(manifest) - SAMPLE_MANIFEST_FIELDS)
    if missing:
        errors.append(f"sample manifest missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"sample manifest unexpected fields: {', '.join(unexpected)}")
    if manifest.get("schema_version") != "search-trace-sample-v1":
        errors.append("sample manifest invalid schema_version")
    if not _normalized_identifier(manifest.get("sample_id")):
        errors.append("sample manifest invalid sample_id")
    if manifest.get("provenance") not in {"controlled_production", "synthetic_fixture"}:
        errors.append("sample manifest invalid provenance")
    if type(manifest.get("fixture_derived")) is not bool:
        errors.append("sample manifest fixture_derived must be boolean")
    if not _valid_timestamp(manifest.get("collected_at")):
        errors.append("sample manifest invalid collected_at")
    if manifest.get("traces_sha256") != _artifact_sha256(traces):
        errors.append("sample manifest traces_sha256 does not bind traces")
    if manifest.get("audits_sha256") != _artifact_sha256(audits):
        errors.append("sample manifest audits_sha256 does not bind audits")
    if manifest.get("provenance") == "controlled_production" and manifest.get("fixture_derived") is not False:
        errors.append("controlled_production must not be fixture-derived")
    if manifest.get("provenance") == "synthetic_fixture" and manifest.get("fixture_derived") is not True:
        errors.append("synthetic_fixture requires fixture_derived=true")
    errors.extend(_validate_trusted_attestation(manifest, trusted_verifier_key, "sample manifest"))
    return errors


# ── integrity ──────────────────────────────────────────────────────────

def _validate_case(case: Mapping[str, Any], index: int) -> list[str]:
    prefix = f"case row {index}"
    errors: list[str] = []
    missing = sorted(CASE_FIELDS - set(case))
    unexpected = sorted(set(case) - CASE_FIELDS - CASE_OPTIONAL_FIELDS)
    if missing:
        errors.append(f"{prefix} missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"{prefix} unexpected fields: {', '.join(unexpected)}")
    case_id = case.get("case_id")
    display = case_id if _is_nonempty_string(case_id) else f"row-{index}"
    if not _is_nonempty_string(case_id):
        errors.append(f"{prefix} missing case_id")
    if not _closed(case.get("category"), set(CATEGORY_QUOTAS)):
        errors.append(f"case {display} invalid category")
    if not _is_nonempty_string(case.get("question")):
        errors.append(f"case {display} missing question")
    if type(case.get("allow_skip")) is not bool:
        errors.append(f"case {display} allow_skip must be boolean")
    if case.get("skip_reason") is not None and not _closed(case.get("skip_reason"), SKIP_REASONS):
        errors.append(f"case {display} invalid skip_reason")
    if case.get("minimum_tier") is not None and not _closed(case.get("minimum_tier"), TIERS):
        errors.append(f"case {display} invalid minimum_tier")
    for name in ("dynamic", "high_consequence"):
        if name in case and type(case.get(name)) is not bool:
            errors.append(f"case {display} {name} must be boolean")
    expected_tier = case.get("expected_final_tier")
    acceptable_tiers = case.get("acceptable_final_tiers")
    if expected_tier is not None and not _closed(expected_tier, ROUTES):
        errors.append(f"case {display} invalid expected_final_tier")
    if acceptable_tiers is not None and (
        not _is_string_list(acceptable_tiers, allow_empty=False)
        or any(not _closed(tier, ROUTES) for tier in acceptable_tiers or ())
        or len(set(acceptable_tiers or ())) != len(acceptable_tiers or ())
    ):
        errors.append(f"case {display} invalid acceptable_final_tiers")
    if expected_tier is not None and acceptable_tiers is not None:
        errors.append(f"case {display} cannot set both expected_final_tier and acceptable_final_tiers")
    if type(case.get("external_fact_required")) is not bool:
        errors.append(f"case {display} external_fact_required must be boolean")
    if not _closed(case.get("actionability"), ACTIONABILITY):
        errors.append(f"case {display} invalid actionability")
    if not _closed(case.get("potential_harm"), POTENTIAL_HARM):
        errors.append(f"case {display} invalid potential_harm")
    purposes = case.get("expected_query_purposes")
    if not _is_string_list(purposes) or any(not _closed(item, QUERY_PURPOSES) for item in purposes or ()):
        errors.append(f"case {display} invalid expected_query_purposes")
    for name in ("expected_initial_query_min", "expected_initial_query_max", "expected_max_rounds"):
        if not _is_int(case.get(name)):
            errors.append(f"case {display} {name} must be a non-negative integer")
    minimum = case.get("expected_initial_query_min")
    maximum = case.get("expected_initial_query_max")
    if _is_int(minimum) and _is_int(maximum) and minimum > maximum:
        errors.append(f"case {display} initial query minimum exceeds maximum")
    if not _is_string_list(case.get("material_claim_spans")):
        errors.append(f"case {display} invalid material_claim_spans")
    relations = case.get("acceptable_source_relations")
    if not _is_string_list(relations, allow_empty=False) or any(not _closed(item, SOURCE_RELATIONS) for item in relations or ()):
        errors.append(f"case {display} invalid acceptable_source_relations")
    if not _closed(case.get("expected_outcome"), EXPECTED_OUTCOMES):
        errors.append(f"case {display} invalid expected_outcome")
    if not _is_nonempty_string(case.get("fixture_id")):
        errors.append(f"case {display} missing fixture_id")
    if not _closed(case.get("label_status"), LABEL_STATUSES):
        errors.append(f"case {display} invalid label_status")

    reviewed = (
        case.get("label_status") == "reviewed"
        and _normalized_identifier(case.get("reviewed_by"))
        and _valid_date(case.get("reviewed_at"))
    )
    if not reviewed:
        errors.append(f"case {display} owner review is incomplete")
    elif not _valid_date(case.get("reviewed_at")):
        errors.append(f"case {display} invalid reviewed_at date")
    if case.get("label_status") == "reviewed" and case.get("reviewed_at") is not None and not _valid_date(case.get("reviewed_at")):
        date_error = f"case {display} invalid reviewed_at date"
        if date_error not in errors:
            errors.append(date_error)

    if case.get("allow_skip") is True:
        if case.get("skip_reason") is None:
            errors.append(f"case {display} allow_skip requires skip_reason")
        if case.get("minimum_tier") is not None:
            errors.append(f"case {display} skippable case cannot set minimum_tier")
    elif case.get("skip_reason") is not None:
        errors.append(f"case {display} non-skippable case cannot set skip_reason")

    labels = case.get("semantic_labels")
    if not isinstance(labels, list):
        errors.append(f"case {display} semantic_labels must be a list")
    else:
        identities: list[tuple[Any, Any]] = []
        for label_index, label in enumerate(labels, 1):
            if not isinstance(label, dict):
                errors.append(f"case {display} semantic label {label_index} must be an object")
                continue
            unexpected_label_fields = sorted(set(label) - {"label_id", "component", "expected"})
            missing_label_fields = sorted({"label_id", "component", "expected"} - set(label))
            if missing_label_fields:
                errors.append(
                    f"case {display} semantic label {label_index} missing fields: {', '.join(missing_label_fields)}"
                )
            if unexpected_label_fields:
                errors.append(
                    f"case {display} semantic label {label_index} unexpected fields: {', '.join(unexpected_label_fields)}"
                )
            label_id = label.get("label_id")
            component = label.get("component")
            expected = label.get("expected")
            if not _is_nonempty_string(label_id):
                errors.append(f"case {display} semantic label {label_index} missing label_id")
            if not _closed(component, QUALITY_COMPONENTS):
                errors.append(f"case {display} semantic label {label_index} invalid component")
            allowed = {
                "claim_discovery": DISCOVERY_LABELS,
                "semantic_support": SUPPORT_LABELS,
                "relevance": RELEVANCE_LABELS,
            }.get(component, set())
            if expected not in allowed:
                errors.append(f"case {display} semantic label {label_index} invalid expected label")
            identities.append((component, label_id))
        if _duplicate_values(identities):
            errors.append(f"case {display} duplicate semantic label_id")
    return errors


def _validate_recording(recording: Mapping[str, Any], index: int, case_ids: set[str]) -> list[str]:
    prefix = f"recording row {index}"
    errors: list[str] = []
    missing = sorted(RECORDING_FIELDS - set(recording))
    unexpected = sorted(set(recording) - RECORDING_FIELDS)
    if missing:
        errors.append(f"{prefix} missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"{prefix} unexpected fields: {', '.join(unexpected)}")
    case_id = recording.get("case_id")
    if not _is_nonempty_string(case_id):
        errors.append(f"{prefix} missing case_id")
    elif case_id not in case_ids:
        errors.append(f"{prefix} unknown case_id {case_id}")
    for name in ("fixture_id", "provider", "query_text", "title", "excerpt"):
        if not _is_nonempty_string(recording.get(name)):
            errors.append(f"{prefix} missing {name}")
    if not _valid_http_url(recording.get("url")):
        errors.append(f"{prefix} invalid url")
    if not _closed(recording.get("expected_fetch_status"), PROVIDER_STATUSES | {"unreadable"}):
        errors.append(f"{prefix} invalid expected_fetch_status")
    return errors


def _prediction_items(prediction: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = prediction.get("predictions")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if _is_nonempty_string(prediction.get("label_id")):
        return [{
            "label_id": prediction.get("label_id"),
            "predicted": prediction.get("predicted"),
        }]
    return []


def _recursive_human_label_fields(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in HUMAN_LABEL_FIELDS or key == "expected":
                found.append(child_path)
            found.extend(_recursive_human_label_fields(nested, child_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_recursive_human_label_fields(nested, f"{path}[{index}]"))
    return found


def _validate_prediction(prediction: Mapping[str, Any], index: int, case_ids: set[str]) -> list[str]:
    prefix = f"prediction row {index}"
    errors: list[str] = []
    component = prediction.get("component")
    if _closed(component, QUALITY_COMPONENTS) and "label_id" in prediction:
        allowed_fields = PREDICTION_COMMON_FIELDS | {"label_id", "predicted"}
    else:
        allowed_fields = PREDICTION_FIELDS.get(component, PREDICTION_COMMON_FIELDS) if isinstance(component, str) else PREDICTION_COMMON_FIELDS
    missing = sorted(allowed_fields - set(prediction))
    unexpected = sorted(set(prediction) - allowed_fields)
    if missing:
        errors.append(f"{prefix} missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"{prefix} unexpected fields: {', '.join(unexpected)}")
    leaked = sorted(set(prediction).intersection(HUMAN_LABEL_FIELDS))
    if leaked:
        errors.append(f"{prefix} contains human-label field: {', '.join(leaked)}")
    recursive_leaks = sorted(
        path for path in _recursive_human_label_fields(prediction)
        if "." in path or "[" in path
    )
    if recursive_leaks:
        errors.append(
            f"{prefix} contains recursive human-label field: {', '.join(recursive_leaks)}"
        )
    case_id = prediction.get("case_id")
    if not _is_nonempty_string(case_id):
        errors.append(f"{prefix} missing case_id")
    elif case_id not in case_ids:
        errors.append(f"{prefix} unknown case_id {case_id}")
    if not _closed(component, PREDICTION_COMPONENTS):
        errors.append(f"{prefix} invalid component")
    for name in ("model", "model_version", "prompt_schema_version"):
        if not _normalized_identifier(prediction.get(name)):
            errors.append(f"{prefix} missing {name}")
    if not _valid_timestamp(prediction.get("run_timestamp")):
        errors.append(f"{prefix} invalid run_timestamp")
    if component == "router" and not _closed(prediction.get("predicted_tier"), ROUTES):
        errors.append(f"{prefix} invalid predicted_tier")
    if component == "planner":
        purposes = prediction.get("predicted_query_purposes")
        if not _is_string_list(purposes) or any(not _closed(value, QUERY_PURPOSES) for value in purposes or ()):
            errors.append(f"{prefix} invalid predicted_query_purposes")
        if not _is_int(prediction.get("predicted_initial_query_count")):
            errors.append(f"{prefix} predicted_initial_query_count must be a non-negative integer")
        if type(prediction.get("predicted_repair_used")) is not bool:
            errors.append(f"{prefix} predicted_repair_used must be boolean")
    if _closed(component, QUALITY_COMPONENTS):
        if "predictions" in prediction and not isinstance(prediction.get("predictions"), list):
            errors.append(f"{prefix} predictions must be a list")
        if isinstance(prediction.get("predictions"), list):
            for item_index, item in enumerate(prediction["predictions"], 1):
                if not isinstance(item, Mapping):
                    errors.append(f"{prefix} prediction {item_index} must be an object")
        items = _prediction_items(prediction)
        if not items:
            errors.append(f"{prefix} quality prediction requires predictions")
        seen: set[str] = set()
        allowed = {
            "claim_discovery": DISCOVERY_LABELS,
            "semantic_support": SUPPORT_LABELS,
            "relevance": RELEVANCE_LABELS,
        }[component]
        for item_index, item in enumerate(items, 1):
            missing_item = sorted({"label_id", "predicted"} - set(item))
            unexpected_item = sorted(set(item) - {"label_id", "predicted"})
            if missing_item:
                errors.append(f"{prefix} prediction {item_index} missing fields: {', '.join(missing_item)}")
            if unexpected_item:
                errors.append(f"{prefix} prediction {item_index} unexpected fields: {', '.join(unexpected_item)}")
            label_id = item.get("label_id")
            predicted = item.get("predicted", item.get("predicted_label"))
            if not _is_nonempty_string(label_id):
                errors.append(f"{prefix} prediction {item_index} missing label_id")
            elif label_id in seen:
                errors.append(f"{prefix} duplicate prediction label_id {label_id}")
            else:
                seen.add(label_id)
            if predicted not in allowed:
                errors.append(f"{prefix} prediction {item_index} invalid predicted label")
    return errors


def _validate_case_prediction_artifacts(
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for index, case in enumerate(cases, 1):
        errors.extend(_validate_case(case, index))
    case_ids = {
        case.get("case_id") for case in cases if _is_nonempty_string(case.get("case_id"))
    }
    for case_id in _duplicate_values(case.get("case_id") for case in cases):
        errors.append(f"duplicate case_id {case_id}")
    for index, prediction in enumerate(predictions, 1):
        errors.extend(_validate_prediction(prediction, index, case_ids))

    singleton_keys = [
        (row.get("case_id"), row.get("component"))
        for row in predictions
        if _is_nonempty_string(row.get("case_id"))
        and not _closed(row.get("component"), QUALITY_COMPONENTS)
    ]
    for case_id, component in _duplicate_values(singleton_keys):
        errors.append(f"duplicate prediction for case_id {case_id} component {component}")
    quality_keys = [
        (row.get("case_id"), row.get("component"), item.get("label_id"))
        for row in predictions
        if _is_nonempty_string(row.get("case_id")) and _closed(row.get("component"), QUALITY_COMPONENTS)
        for item in _prediction_items(row)
        if _is_nonempty_string(item.get("label_id"))
    ]
    for case_id, component, label_id in _duplicate_values(quality_keys):
        errors.append(
            f"duplicate prediction for case_id {case_id} component {component} label_id {label_id}"
        )
    human_keys = {
        (case.get("case_id"), label.get("component"), label.get("label_id"))
        for case in cases
        for label in (case.get("semantic_labels") or ())
        if isinstance(label, Mapping)
        and _is_nonempty_string(case.get("case_id"))
        and _closed(label.get("component"), QUALITY_COMPONENTS)
        and _is_nonempty_string(label.get("label_id"))
    }
    predicted_keys = set(quality_keys)
    for case_id, component, label_id in sorted(human_keys - predicted_keys):
        errors.append(f"missing quality prediction for {case_id}/{component}/{label_id}")
    for case_id, component, label_id in sorted(predicted_keys - human_keys):
        errors.append(f"quality prediction has no external human label {case_id}/{component}/{label_id}")
    router_ids = {
        row.get("case_id") for row in predictions
        if row.get("component") == "router" and _is_nonempty_string(row.get("case_id"))
    }
    for case_id in sorted(case_ids - router_ids):
        errors.append(f"missing router prediction for case_id {case_id}")
    return errors


def _validate_integrity_impl(
    cases: Sequence[Mapping[str, Any]],
    recordings: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    expected_case_count: int = 140,
    category_quotas: Mapping[str, int] = CATEGORY_QUOTAS,
) -> list[str]:
    errors: list[str] = []
    if len(cases) != expected_case_count:
        errors.append(f"case count {len(cases)} != {expected_case_count}")
    categories = Counter(case.get("category") for case in cases)
    for category, expected in category_quotas.items():
        if categories[category] != expected:
            errors.append(f"category {category} count {categories[category]} != {expected}")
    for index, case in enumerate(cases, 1):
        errors.extend(_validate_case(case, index))

    case_ids = {case.get("case_id") for case in cases if _is_nonempty_string(case.get("case_id"))}
    duplicates = _duplicate_values(case.get("case_id") for case in cases)
    for case_id in duplicates:
        errors.append(f"duplicate case_id {case_id}")
    normalized_questions = [
        _normalized_question(case["question"])
        for case in cases if _is_nonempty_string(case.get("question"))
    ]
    for normalized in _duplicate_values(normalized_questions):
        errors.append(f"duplicate normalized question {normalized}")

    for index, recording in enumerate(recordings, 1):
        errors.extend(_validate_recording(recording, index, case_ids))
    fixture_ids = [recording.get("fixture_id") for recording in recordings]
    for fixture_id in _duplicate_values(fixture_ids):
        errors.append(f"duplicate recording fixture_id {fixture_id}")
    recording_by_fixture = {
        recording.get("fixture_id"): recording
        for recording in recordings if _is_nonempty_string(recording.get("fixture_id"))
    }
    for case in cases:
        if case.get("expected_outcome") == "skip":
            continue
        fixture_id = case.get("fixture_id")
        recording = recording_by_fixture.get(fixture_id)
        if recording is None:
            errors.append(f"case {case.get('case_id')} missing fixture_id reference {fixture_id}")
        elif recording.get("case_id") != case.get("case_id"):
            errors.append(f"case {case.get('case_id')} fixture_id {fixture_id} references another case")

    for index, prediction in enumerate(predictions, 1):
        errors.extend(_validate_prediction(prediction, index, case_ids))
    singleton_prediction_keys = [
        (prediction.get("case_id"), prediction.get("component"))
        for prediction in predictions
        if _is_nonempty_string(prediction.get("case_id"))
        and not _closed(prediction.get("component"), QUALITY_COMPONENTS)
    ]
    for case_id, component in _duplicate_values(singleton_prediction_keys):
        errors.append(f"duplicate prediction for case_id {case_id} component {component}")
    quality_prediction_keys = [
        (prediction.get("case_id"), prediction.get("component"), item.get("label_id"))
        for prediction in predictions
        if _is_nonempty_string(prediction.get("case_id"))
        and _closed(prediction.get("component"), QUALITY_COMPONENTS)
        for item in _prediction_items(prediction)
        if _is_nonempty_string(item.get("label_id"))
    ]
    for case_id, component, label_id in _duplicate_values(quality_prediction_keys):
        errors.append(
            f"duplicate prediction for case_id {case_id} component {component} label_id {label_id}"
        )
    human_quality_keys = {
        (case.get("case_id"), label.get("component"), label.get("label_id"))
        for case in cases
        for label in (case.get("semantic_labels") or ())
        if isinstance(label, Mapping)
        and _is_nonempty_string(case.get("case_id"))
        and _closed(label.get("component"), QUALITY_COMPONENTS)
        and _is_nonempty_string(label.get("label_id"))
    }
    predicted_quality_keys = set(quality_prediction_keys)
    for case_id, component, label_id in sorted(human_quality_keys - predicted_quality_keys):
        errors.append(f"missing quality prediction for {case_id}/{component}/{label_id}")
    for case_id, component, label_id in sorted(predicted_quality_keys - human_quality_keys):
        errors.append(f"quality prediction has no external human label {case_id}/{component}/{label_id}")
    router_case_ids = {
        prediction.get("case_id") for prediction in predictions
        if prediction.get("component") == "router" and _is_nonempty_string(prediction.get("case_id"))
    }
    for case_id in sorted(case_ids - router_case_ids):
        errors.append(f"missing router prediction for case_id {case_id}")

    errors.extend(_secret_errors(cases, "case"))
    errors.extend(_secret_errors(recordings, "recording"))
    errors.extend(_secret_errors(predictions, "prediction"))
    return errors


def validate_integrity(
    cases: Sequence[Mapping[str, Any]],
    recordings: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    expected_case_count: int = 140,
    category_quotas: Mapping[str, int] = CATEGORY_QUOTAS,
) -> list[str]:
    try:
        return _validate_integrity_impl(
            cases, recordings, predictions,
            expected_case_count=expected_case_count,
            category_quotas=category_quotas,
        )
    except Exception as exc:
        return [f"controlled integrity validation error ({type(exc).__name__})"]


def collect_integrity_errors(
    cases_path: Path = CASES_PATH,
    recordings_path: Path = PROVIDER_RECORDINGS_PATH,
    predictions_path: Path = MODEL_PREDICTIONS_PATH,
) -> list[str]:
    cases, case_errors = _load_jsonl_checked(cases_path)
    recordings, recording_errors = _load_jsonl_checked(recordings_path)
    predictions, prediction_errors = _load_jsonl_checked(predictions_path)
    load_errors = [*case_errors, *recording_errors, *prediction_errors]
    if load_errors:
        return load_errors
    return validate_integrity(cases, recordings, predictions)


def integrity() -> int:
    errors = collect_integrity_errors()
    for error in errors:
        print(f"integrity error: {error}")
    print(f"integrity: errors={len(errors)}")
    return 1 if errors else 0


# ── offline model-quality metrics ─────────────────────────────────────

def _case_groups(case: Mapping[str, Any]) -> set[str]:
    groups = {"overall"}
    if case.get("dynamic") is True:
        groups.add("dynamic")
    if case.get("high_consequence") is True:
        groups.add("high_consequence")
    return groups


def quality_metrics(
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    labels: dict[tuple[str, str, str], tuple[str, set[str]]] = {}
    errors: list[str] = []
    for case in cases:
        case_id = case.get("case_id")
        if not _is_nonempty_string(case_id):
            continue
        for label in case.get("semantic_labels") or ():
            if not isinstance(label, Mapping):
                continue
            component = label.get("component")
            label_id = label.get("label_id")
            expected = label.get("expected")
            if not _closed(component, QUALITY_COMPONENTS) or not _is_nonempty_string(label_id):
                continue
            key = (case_id, component, label_id)
            if key in labels:
                errors.append(f"duplicate human semantic label {case_id}/{component}/{label_id}")
                continue
            labels[key] = (expected, _case_groups(case))

    predicted: dict[tuple[str, str, str], str] = {}
    for row in predictions:
        component = row.get("component")
        case_id = row.get("case_id")
        if not _closed(component, QUALITY_COMPONENTS) or not _is_nonempty_string(case_id):
            continue
        for item in _prediction_items(row):
            label_id = item.get("label_id")
            value = item.get("predicted", item.get("predicted_label"))
            if not _is_nonempty_string(label_id):
                continue
            key = (case_id, component, label_id)
            if key in predicted:
                errors.append(f"duplicate model prediction {case_id}/{component}/{label_id}")
                continue
            predicted[key] = value

    grouped_pairs: dict[str, dict[str, list[tuple[str, str]]]] = {
        component: {group: [] for group in ("overall", "dynamic", "high_consequence")}
        for component in QUALITY_COMPONENTS
    }
    missing_by_component = Counter()
    for key, (expected, groups) in labels.items():
        case_id, component, label_id = key
        if key not in predicted:
            missing_by_component[component] += 1
            errors.append(f"missing {component} prediction for {case_id}/{label_id}")
            continue
        pair = (expected, predicted[key])
        for group in groups:
            grouped_pairs[component][group].append(pair)
    for case_id, component, label_id in sorted(set(predicted) - set(labels)):
        errors.append(f"prediction has no external human label {case_id}/{component}/{label_id}")

    result: dict[str, Any] = {}
    for component in QUALITY_COMPONENTS:
        result[component] = {}
        for group, pairs in grouped_pairs[component].items():
            if component == "semantic_support":
                metric = _multiclass_quality(pairs)
            elif component == "claim_discovery":
                metric = _binary_quality(pairs, {"present"})
            else:
                metric = _binary_quality(pairs, {"relevant", "direct", "contextual", "admitted", "pass"})
            metric["label_count"] = sum(
                1 for (_case_id, label_component, _label_id), (_expected, groups) in labels.items()
                if label_component == component and group in groups
            )
            metric["missing_prediction_count"] = metric["label_count"] - metric["sample_count"]
            flag = group if group in {"dynamic", "high_consequence"} else None
            missing_group_labels = sum(
                1
                for case in cases
                if flag is not None
                and any(
                    isinstance(label, Mapping) and label.get("component") == component
                    for label in (case.get("semantic_labels") or ())
                )
                and type(case.get(flag)) is not bool
            )
            metric["missing_subgroup_label_count"] = missing_group_labels
            if missing_group_labels:
                metric["evaluable"] = False
                for name in ("precision", "recall", "f1", "macro_precision", "macro_recall", "macro_f1"):
                    if name in metric:
                        metric[name] = None
            result[component][group] = metric
    result["errors"] = errors
    return result


def in_d_factual(case: Mapping[str, Any]) -> bool:
    if case.get("skip_reason") == "user_forbid_web" or case.get("allow_skip") is True:
        return False
    external = case.get("d_factual")
    if type(external) is bool:
        return external
    return case.get("external_fact_required") is not False


_TIER_RANK = {"skip": 0, "light": 1, "standard": 2}


def _router_pairs(
    cases: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[Mapping[str, Any], str]], list[str]]:
    failures: list[str] = []
    router_rows: dict[str, Mapping[str, Any]] = {}
    for row in predictions:
        if row.get("component") != "router" or not _is_nonempty_string(row.get("case_id")):
            continue
        case_id = row["case_id"]
        if case_id in router_rows:
            failures.append(f"duplicate router prediction for case_id {case_id}")
            continue
        router_rows[case_id] = row
    pairs: list[tuple[Mapping[str, Any], str]] = []
    for case in cases:
        case_id = case.get("case_id")
        if not _is_nonempty_string(case_id):
            continue
        row = router_rows.get(case_id)
        if row is None:
            failures.append(f"missing router prediction for case_id {case_id}")
            continue
        predicted = row.get("predicted_tier")
        if not _closed(predicted, ROUTES):
            failures.append(f"invalid router prediction for case_id {case_id}")
            continue
        pairs.append((case, predicted))
    return pairs, failures


def routing_quality_metrics(
    cases: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pairs, errors = _router_pairs(cases, predictions)
    confusion: Counter[tuple[str, str]] = Counter()
    floor_violations = 0
    target_samples = 0
    acceptable_matches = 0
    missing_targets = 0
    for case, predicted in pairs:
        minimum = case.get("minimum_tier") or "skip"
        if _TIER_RANK[predicted] < _TIER_RANK.get(minimum, 0):
            floor_violations += 1
        exact = case.get("expected_final_tier")
        acceptable = case.get("acceptable_final_tiers")
        if _closed(exact, ROUTES):
            targets = (exact,)
        elif isinstance(acceptable, list) and acceptable and all(_closed(tier, ROUTES) for tier in acceptable):
            targets = tuple(acceptable)
        else:
            missing_targets += 1
            continue
        target_samples += 1
        if predicted in targets:
            acceptable_matches += 1
            confusion[(predicted, predicted)] += 1
        else:
            confusion[(predicted, targets[0])] += 1
    metrics = tier_metrics(confusion) if target_samples else {
        "macro_precision": None, "macro_recall": None, "macro_f1": None,
    }
    return {
        "router_sample_count": len(pairs),
        "minimum_tier_violations": floor_violations,
        "tier_target_evaluable": target_samples > 0 and missing_targets == 0,
        "tier_target_sample_count": target_samples,
        "missing_tier_target_count": missing_targets,
        "acceptable_tier_matches": acceptable_matches,
        "tier_metrics": metrics,
        "errors": errors,
    }


def _evaluate_offline_impl(
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    provider_recordings: Sequence[Mapping[str, Any]] | None = None,
    run_manifest: Mapping[str, Any] | None = None,
    trusted_verifier_key: bytes | bytearray | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if provider_recordings is None:
        artifact_errors = _validate_case_prediction_artifacts(cases, predictions)
        artifact_errors.append("provider recordings are required for offline evaluation")
        recordings: Sequence[Mapping[str, Any]] = ()
    else:
        recordings = provider_recordings
        observed_quotas = Counter(
            case.get("category") for case in cases
            if isinstance(case, Mapping) and isinstance(case.get("category"), str)
        )
        artifact_errors = _validate_integrity_impl(
            cases, recordings, predictions,
            expected_case_count=len(cases), category_quotas=observed_quotas,
        )
    if artifact_errors:
        failures.append("offline case/prediction integrity errors")
    manifest_errors = _validate_run_manifest(
        run_manifest, cases, recordings, predictions, trusted_verifier_key,
    )
    failures.extend(manifest_errors)
    routing = routing_quality_metrics(cases, predictions)
    failures.extend(routing["errors"])
    route_pairs, _ = _router_pairs(cases, predictions)
    tier = routing["tier_metrics"]
    mandatory = [(case, prediction) for case, prediction in route_pairs if in_d_factual(case)]
    explicit = [
        (case, prediction) for case, prediction in route_pairs
        if case.get("category") == "explicit_search" and case.get("skip_reason") != "user_forbid_web"
    ]
    legal_skip = [
        (case, prediction) for case, prediction in route_pairs
        if case.get("allow_skip") and case.get("external_fact_required") is False
    ]

    def route_rate(rows: Sequence[tuple[Mapping[str, Any], str]]) -> dict[str, Any]:
        numerator = sum(prediction != "skip" for _case, prediction in rows)
        denominator = len(rows)
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator if denominator else None,
        }

    mandatory_rate = route_rate(mandatory)
    explicit_rate = route_rate(explicit)
    meaningless_numerator = sum(prediction != "skip" for _case, prediction in legal_skip)
    meaningless = {
        "numerator": meaningless_numerator,
        "denominator": len(legal_skip),
        "rate": meaningless_numerator / len(legal_skip) if legal_skip else None,
    }

    quality = quality_metrics(cases, predictions)
    failures.extend(quality["errors"])
    if not route_pairs:
        failures.append("router zero samples")
    if routing["minimum_tier_violations"]:
        failures.append("minimum tier safety-floor violations")
    if not routing["tier_target_evaluable"]:
        failures.append("reviewed final-tier targets missing or zero samples")
    elif tier["macro_f1"] < 0.90:
        failures.append("tier macro F1 below 0.90")
    if mandatory_rate["denominator"] == 0:
        failures.append("mandatory route zero samples")
    elif mandatory_rate["rate"] < 1.0:
        failures.append("mandatory search route rate below 1.00")
    if explicit_rate["denominator"] == 0:
        failures.append("explicit search route zero samples")
    elif explicit_rate["rate"] < 1.0:
        failures.append("explicit search route rate below 1.00")
    if meaningless["denominator"] == 0:
        failures.append("legal non-factual skip zero samples")
    elif meaningless["rate"] > 0.10:
        failures.append("legal non-factual meaningless-search rate above 0.10")

    for component in QUALITY_COMPONENTS:
        for group in ("overall", "dynamic", "high_consequence"):
            metric = quality[component][group]
            if not metric["evaluable"]:
                failures.append(f"{component} {group} zero samples")
                continue
            threshold = 0.97 if group != "overall" else 0.95
            names = ("macro_precision", "macro_recall", "macro_f1") if component == "semantic_support" else ("precision", "recall", "f1")
            if any(metric[name] < threshold for name in names):
                failures.append(f"{component} {group} PRF below {threshold:.2f}")

    manifest_valid = not manifest_errors
    trusted_attestation_verified = not _validate_trusted_attestation(
        run_manifest, trusted_verifier_key, "run manifest",
    )
    prediction_hash = _artifact_sha256(predictions)
    fixture_models = {
        identity
        for row in predictions if isinstance(row, Mapping)
        if (identity := _canonical_identity(row.get("model"))) is not None
    }.intersection(KNOWN_FIXTURE_MODEL_IDENTITIES)
    known_fixture_predictions = prediction_hash in KNOWN_FIXTURE_PREDICTION_HASHES
    fixture_baseline = bool(
        fixture_models or known_fixture_predictions or (
            isinstance(run_manifest, Mapping) and (
            run_manifest.get("provenance") == "fixture_baseline"
            or run_manifest.get("fixture_derived") is True
            or run_manifest.get("data_source") == "synthetic_provider_fixtures"
            )
        )
    )
    if fixture_models:
        failures.append("fixture model identity is diagnostic and non-certifying")
    if known_fixture_predictions:
        failures.append("known fixture prediction hash is diagnostic and non-certifying")
    if fixture_baseline:
        failures.append("fixture baseline is diagnostic and non-certifying")
    owner_reviewed = all(
        case.get("label_status") == "reviewed"
        and _normalized_identifier(case.get("reviewed_by"))
        and _valid_date(case.get("reviewed_at"))
        for case in cases
    ) and bool(cases)
    if not owner_reviewed:
        failures.append("human owner review is incomplete")

    failures = list(dict.fromkeys(failures))
    return {
        "mode": "offline",
        "certifying": not failures,
        "artifact_class": (
            "fixture_baseline" if fixture_baseline
            else "independent_predictions" if manifest_valid
            else "unverified_provenance"
        ),
        "case_count": len(cases),
        "router_sample_count": len(route_pairs),
        "tier_metrics": tier,
        "routing_quality": routing,
        "errors": artifact_errors,
        "run_manifest_valid": manifest_valid,
        "trusted_attestation_verified": trusted_attestation_verified,
        "mandatory_search_route_rate": mandatory_rate,
        "explicit_search_route_rate": explicit_rate,
        "legal_non_factual_meaningless_search_rate": meaningless,
        "quality": quality,
        "failures": failures,
    }


def evaluate_offline(
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    provider_recordings: Sequence[Mapping[str, Any]] | None = None,
    run_manifest: Mapping[str, Any] | None = None,
    trusted_verifier_key: bytes | bytearray | None = None,
) -> dict[str, Any]:
    try:
        return _evaluate_offline_impl(
            cases, predictions, provider_recordings=provider_recordings,
            run_manifest=run_manifest,
            trusted_verifier_key=trusted_verifier_key,
        )
    except Exception as exc:
        return {
            "mode": "offline", "certifying": False,
            "artifact_class": "unverified_provenance",
            "trusted_attestation_verified": False,
            "errors": [f"controlled offline validation error ({type(exc).__name__})"],
            "failures": ["offline validation could not safely evaluate the artifacts"],
        }


def _trusted_verifier_key_from_env(name: str | None) -> tuple[bytes | None, list[str]]:
    if not _normalized_identifier(name):
        return None, ["trusted verifier environment variable name is required"]
    value = os.environ.get(name)
    if value is None:
        return None, [f"trusted verifier environment variable {name} is not set"]
    encoded = value.encode("utf-8")
    if len(encoded) < 32:
        return None, ["trusted verifier key must be at least 32 bytes"]
    return encoded, []


def offline(
    cases_path: Path | None = None,
    recordings_path: Path | None = None,
    predictions_path: Path | None = None,
    manifest_path: Path | None = None,
    verifier_key_env: str | None = None,
) -> int:
    custom = any(value is not None for value in (
        cases_path, recordings_path, predictions_path, manifest_path,
        verifier_key_env,
    ))
    if custom and (
        cases_path is None or recordings_path is None or predictions_path is None
        or manifest_path is None or verifier_key_env is None
    ):
        report = {
            "mode": "offline", "certifying": False,
            "errors": [
                "independent offline evaluation requires --cases, --recordings, "
                "--predictions, --manifest, and --verifier-key-env"
            ],
            "failures": ["incomplete independent offline CLI inputs"],
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 1
    resolved_cases = cases_path if cases_path is not None else CASES_PATH
    resolved_recordings = (
        recordings_path if recordings_path is not None else PROVIDER_RECORDINGS_PATH
    )
    resolved_predictions = predictions_path if predictions_path is not None else MODEL_PREDICTIONS_PATH
    cases, case_errors = _load_jsonl_checked(resolved_cases)
    recordings, recording_errors = _load_jsonl_checked(resolved_recordings)
    predictions, prediction_errors = _load_jsonl_checked(resolved_predictions)
    manifest: Mapping[str, Any] | None = None
    manifest_errors: list[str] = []
    trusted_key: bytes | None = None
    key_errors: list[str] = []
    if custom:
        manifest, manifest_errors = _load_json_object_checked(manifest_path)
        trusted_key, key_errors = _trusted_verifier_key_from_env(verifier_key_env)
    load_errors = [
        *case_errors, *recording_errors, *prediction_errors,
        *manifest_errors, *key_errors,
    ]
    if load_errors:
        report = {
            "mode": "offline", "certifying": False, "errors": load_errors,
            "failures": ["offline input or verifier configuration errors"],
        }
    else:
        if not custom:
            manifest = {
                "schema_version": "search-eval-run-v1",
                "run_id": "checked-in-fixture-baseline-v1",
                "provenance": "fixture_baseline",
                "data_source": "synthetic_provider_fixtures",
                "fixture_derived": True,
                "case_set_sha256": _artifact_sha256(cases),
                "recordings_sha256": _artifact_sha256(recordings),
                "predictions_sha256": _artifact_sha256(predictions),
                "run_timestamp": "2026-07-29T00:00:00Z",
                "attestation": {
                    "algorithm": "none", "key_id": "untrusted-fixture",
                    "signature": "",
                },
            }
        report = evaluate_offline(
            cases, predictions, provider_recordings=recordings,
            run_manifest=manifest,
            trusted_verifier_key=trusted_key,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["certifying"] else 1


# ── trace metrics and acceptance ───────────────────────────────────────

def _semantic_query_count(trace: Any) -> int:
    executed = _field(trace, "executed_queries", ()) or ()
    query_ids: set[str] = set()
    for query in executed:
        if isinstance(query, Mapping):
            query_id = query.get("query_id")
        elif isinstance(query, tuple) and len(query) == 2:
            query_id = getattr(query[0], "query_id", query[0])
        else:
            query_id = getattr(query, "query_id", None)
        if _is_nonempty_string(query_id):
            query_ids.add(query_id)
    if isinstance(trace, Mapping) and "executed_queries" in trace:
        return len(query_ids)
    value = _field(trace, "semantic_query_count")
    return value if _is_int(value) else 0


def _repair_query_count(trace: Any) -> int:
    executed = _field(trace, "executed_queries", ()) or ()
    query_ids: set[str] = set()
    for query in executed:
        purpose = query.get("purpose") if isinstance(query, Mapping) else getattr(query, "purpose", None)
        query_id = query.get("query_id") if isinstance(query, Mapping) else getattr(query, "query_id", None)
        if _enum_text(purpose) == "repair" and _is_nonempty_string(query_id):
            query_ids.add(query_id)
    if isinstance(trace, Mapping) and "executed_queries" in trace:
        return len(query_ids)
    value = _field(trace, "repair_query_count")
    return value if _is_int(value) else int(bool(_field(trace, "adaptive_repair_round_started")))


def _initial_query_count(trace: Any) -> int:
    executed = _field(trace, "executed_queries", ()) or ()
    query_ids: set[str] = set()
    for query in executed:
        purpose = query.get("purpose") if isinstance(query, Mapping) else getattr(query, "purpose", None)
        query_id = query.get("query_id") if isinstance(query, Mapping) else getattr(query, "query_id", None)
        if _enum_text(purpose) != "repair" and _is_nonempty_string(query_id):
            query_ids.add(query_id)
    if isinstance(trace, Mapping) and "executed_queries" in trace:
        return len(query_ids)
    return _counter(trace, "initial_query_count")


def _retrieval_round_count(trace: Any) -> int:
    if isinstance(trace, Mapping) and "executed_queries" in trace:
        return int(bool(_field(trace, "initial_round_started"))) + int(
            bool(_field(trace, "adaptive_repair_round_started"))
        )
    return _counter(trace, "retrieval_round_count")


def budget_violations(traces: Iterable[Any]) -> dict[str, int]:
    names = (
        "initial_query_count", "candidate_url_count", "content_read_count",
        "semantic_query_count", "repair_query_count", "retrieval_round_count",
        "hard_timeout",
    )
    violations = {name: 0 for name in names}
    for trace in traces:
        route = _route(trace)
        budget = TIER_BUDGETS.get(route)
        if budget is None:
            continue
        observed = {
            "initial_query_count": _initial_query_count(trace),
            "candidate_url_count": _counter(trace, "candidate_url_count"),
            "content_read_count": _counter(trace, "content_read_count"),
            "semantic_query_count": _semantic_query_count(trace),
            "repair_query_count": _repair_query_count(trace),
            "retrieval_round_count": _retrieval_round_count(trace),
        }
        for name, value in observed.items():
            if value > budget[name]:
                violations[name] += 1
        latency = _field(trace, "retrieval_pipeline_latency_ms")
        if _is_nonnegative_number(latency) and latency > budget["hard_timeout_ms"]:
            violations["hard_timeout"] += 1
    return violations


def initial_batch_round_count(traces: Iterable[Any]) -> int:
    return sum(1 for trace in traces if bool(_field(trace, "initial_round_started")))


def repair_round_count(traces: Iterable[Any]) -> int:
    return sum(1 for trace in traces if bool(_field(trace, "adaptive_repair_round_started")))


def _no_attempt_readiness_failure(trace: Mapping[str, Any]) -> str | None:
    attempts = trace.get("provider_attempts")
    if (
        _route(trace) not in TIERS
        or trace.get("orchestrator_started") is not True
        or trace.get("provider_invocation_started") is True
        or trace.get("provider_attempted") is True
        or not isinstance(attempts, list)
        or attempts
    ):
        return None
    if trace.get("provider_configured") is False:
        return "provider_not_configured"
    if trace.get("provider_configured") is True and (
        trace.get("degradation_reason") == "provider_unavailable"
        or (
            isinstance(trace.get("provider_failures"), list)
            and "provider_unavailable" in trace.get("provider_failures", [])
        )
    ):
        return "provider_unavailable"
    return None


def _coherent_no_attempt_readiness_failure(
    trace: Mapping[str, Any], audit: Mapping[str, Any],
) -> bool:
    code = _no_attempt_readiness_failure(trace)
    if code is None:
        return False
    failures = trace.get("provider_failures")
    disclosures = audit.get("rendered_disclosures")
    return (
        isinstance(failures, list)
        and set(failures) == {code}
        and trace.get("degradation_reason") == code
        and trace.get("evidence_state") in {None, "insufficient"}
        and isinstance(disclosures, list)
        and code in disclosures
    )


def deterministic_invariant_violations(
    traces: Iterable[Any],
    audits_by_request_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    violations = {
        "skip_with_provider_attempt": 0,
        "search_without_orchestrator": 0,
        "unsupported_claim_or_citation": 0,
        "supported_claim_count_exceeds_claim_count": 0,
        "citation_without_citable_evidence": 0,
        "citation_count_exceeds_citable_evidence_count": 0,
        "knowledge_fallback_with_citation": 0,
        "sufficient_without_citable_evidence": 0,
        "failure_state_mismatch": 0,
        "evidence_without_provider_attempt": 0,
        "provider_configuration_contradiction": 0,
        "audit_count_mismatch": 0,
        "claim_evidence_mapping": 0,
        "invalid_evidence_url": 0,
        "citable_without_relevance": 0,
        "used_evidence_mapping": 0,
        "shown_source_mapping": 0,
        "partial_without_missing_topics_or_disclosure": 0,
        "retained_claim_on_missing_topic": 0,
        "conflict_without_members_or_disclosure": 0,
        "failure_without_disclosure": 0,
        "dynamic_unsupported_conclusion": 0,
        "retained_unsupported_claim": 0,
        "final_tier_below_audit_floor": 0,
        "final_tier_outside_reviewed_target": 0,
        "evidence_without_successful_provider": 0,
        "provider_failure_mismatch": 0,
        "unconfigured_provider_failure_mismatch": 0,
        "retained_claim_evidence_admission": 0,
        "retained_claim_rendering_contract": 0,
        "retained_claim_partial_structure": 0,
        "retained_claim_conflict_structure": 0,
        "provider_readiness_failure_mismatch": 0,
    }
    allowed_failures = {
        "sufficient": {None, "validation_failed", "provider_timeout"},
        "partial": {"partial_evidence", "validation_failed", "provider_timeout"},
        "conflicting": {"source_conflict", "validation_failed", "provider_timeout"},
        "insufficient": {
            "insufficient_evidence", "provider_not_configured", "provider_unavailable",
            "provider_timeout", "no_results", "content_unreadable",
        },
    }
    for trace in traces:
        route = _route(trace)
        attempts = tuple(
            attempt for attempt in (_field(trace, "provider_attempts", ()) or ())
            if isinstance(attempt, Mapping)
        )
        provider_attempted = bool(
            _field(trace, "provider_invocation_started", _field(trace, "provider_attempted", False))
        ) or any(
            isinstance(attempt, Mapping) and attempt.get("invocation_started") is True
            for attempt in attempts
        )
        successful_provider = any(
            attempt.get("invocation_started") is True and attempt.get("status") == "success"
            for attempt in attempts
        )
        orchestrator_started = bool(_field(trace, "orchestrator_started"))
        evidence_state = _enum_text(_field(trace, "evidence_state")) or None
        degradation = _enum_text(_field(trace, "degradation_reason")) or None
        claim_count = _counter(trace, "claim_count")
        supported_count = _counter(trace, "supported_claim_count")
        citations = _counter(trace, "citation_count")
        citable = _counter(trace, "citable_evidence_count")
        if route == "skip" and provider_attempted:
            violations["skip_with_provider_attempt"] += 1
        if route in TIERS and not orchestrator_started:
            violations["search_without_orchestrator"] += 1
        if evidence_state in {None, "insufficient"} and (claim_count or citations):
            violations["unsupported_claim_or_citation"] += 1
        if supported_count > claim_count:
            violations["supported_claim_count_exceeds_claim_count"] += 1
        if citations and not citable:
            violations["citation_without_citable_evidence"] += 1
        if citations > citable:
            violations["citation_count_exceeds_citable_evidence_count"] += 1
        if citations and bool(_field(trace, "knowledge_fallback_used")):
            violations["knowledge_fallback_with_citation"] += 1
        if evidence_state == "sufficient" and citable == 0:
            violations["sufficient_without_citable_evidence"] += 1
        if evidence_state in allowed_failures and degradation not in allowed_failures[evidence_state]:
            violations["failure_state_mismatch"] += 1
        if route == "skip" and degradation not in {None, "user_forbid_web"}:
            violations["failure_state_mismatch"] += 1
        if (
            evidence_state in {"sufficient", "partial", "conflicting"}
            or claim_count or citable or citations
        ) and not provider_attempted:
            violations["evidence_without_provider_attempt"] += 1
        if (not bool(_field(trace, "provider_configured"))) and (
            provider_attempted
            or evidence_state in {"sufficient", "partial", "conflicting"}
            or claim_count or citable or citations
        ):
            violations["provider_configuration_contradiction"] += 1
        if (
            evidence_state in {"sufficient", "partial", "conflicting"}
            or claim_count or citable or citations
        ) and not successful_provider:
            violations["evidence_without_successful_provider"] += 1
        failure_for_status = {
            "empty": "no_results",
            "timeout": "provider_timeout",
            "error": "provider_unavailable",
            "not_configured": "provider_not_configured",
            "unavailable": "provider_unavailable",
        }
        expected_provider_failures = {
            failure_for_status[attempt.get("status")]
            for attempt in attempts
            if attempt.get("status") in failure_for_status
        }
        readiness_failure = (
            _no_attempt_readiness_failure(trace)
            if isinstance(trace, Mapping) else None
        )
        if readiness_failure is not None:
            expected_provider_failures.add(readiness_failure)
        actual_provider_failures = {
            _enum_text(value) for value in (_field(trace, "provider_failures", ()) or ())
        }
        if expected_provider_failures != actual_provider_failures:
            violations["provider_failure_mismatch"] += 1

        audit = None
        if audits_by_request_id is not None:
            audit = audits_by_request_id.get(str(_field(trace, "request_id", "")))
        if not isinstance(audit, Mapping):
            continue
        claims = audit.get("claims") if isinstance(audit.get("claims"), list) else []
        evidence = audit.get("evidence") if isinstance(audit.get("evidence"), list) else []
        used_ids = audit.get("used_evidence_ids") if isinstance(audit.get("used_evidence_ids"), list) else []
        shown_urls = audit.get("shown_source_urls") if isinstance(audit.get("shown_source_urls"), list) else []
        missing_topics = set(audit.get("missing_claim_topics") or ())
        disclosures = set(audit.get("rendered_disclosures") or ())
        if (
            route in TIERS
            and audit.get("external_fact_required") is True
            and audit.get("allow_skip") is False
            and orchestrator_started
            and readiness_failure in {
                "provider_not_configured", "provider_unavailable",
            }
            and not _coherent_no_attempt_readiness_failure(trace, audit)
        ):
            violations["provider_readiness_failure_mismatch"] += 1
            if readiness_failure == "provider_not_configured":
                violations["unconfigured_provider_failure_mismatch"] += 1
        evidence_by_id = {
            item.get("evidence_id"): item
            for item in evidence if isinstance(item, Mapping) and _is_nonempty_string(item.get("evidence_id"))
        }
        conflict_group_by_id = {
            group.get("group_id"): group
            for group in (
                audit.get("conflict_groups")
                if isinstance(audit.get("conflict_groups"), list) else []
            )
            if isinstance(group, Mapping)
            and _is_nonempty_string(group.get("group_id"))
        }
        retained_supported = sum(
            1 for claim in claims
            if isinstance(claim, Mapping)
            and claim.get("retained") is True
            and claim.get("support_label") == "supported"
        )
        retained_material_edge_ids: set[str] = set()
        retained_edge_violation = False
        retained_partial = False
        retained_conflict = False
        if (
            claim_count != len(claims)
            or supported_count != retained_supported
            or citable != sum(item.get("citable") is True for item in evidence if isinstance(item, Mapping))
            or citations != len(used_ids)
        ):
            violations["audit_count_mismatch"] += 1
        for claim in claims:
            if not isinstance(claim, Mapping):
                continue
            mapped = claim.get("evidence_ids") if isinstance(claim.get("evidence_ids"), list) else []
            claim_topics = {
                topic for topic in (claim.get("topic_ids") or ())
                if isinstance(topic, str)
            }
            partial_topics = {
                topic for topic in (claim.get("partial_topic_ids") or ())
                if isinstance(topic, str)
            }
            conflict_group_ids = {
                group_id for group_id in (claim.get("conflict_group_ids") or ())
                if isinstance(group_id, str)
            }
            claim_disclosures = {
                code for code in (claim.get("disclosure_codes") or ())
                if isinstance(code, str)
            }
            if any(item not in evidence_by_id for item in mapped) or (
                claim.get("retained") is True
                and claim.get("material") is True
                and claim.get("support_label") in {"supported", "partial", "conflict"}
                and not mapped
            ):
                violations["claim_evidence_mapping"] += 1
            if claim.get("retained") is True and claim.get("material") is True:
                retained_material_edge_ids.update(
                    item for item in mapped if isinstance(item, str)
                )
                if (
                    not mapped
                    or any(
                        item not in evidence_by_id
                        or evidence_by_id[item].get("citable") is not True
                        or evidence_by_id[item].get("relevance") not in {"direct", "relevant"}
                        or item not in used_ids
                        for item in mapped
                    )
                ):
                    retained_edge_violation = True
                retained_partial = retained_partial or claim.get("support_label") == "partial"
                retained_conflict = retained_conflict or claim.get("support_label") == "conflict"
            if (
                claim.get("retained") is True
                and claim.get("support_label") == "supported"
                and missing_topics.intersection(claim_topics)
            ):
                violations["retained_claim_on_missing_topic"] += 1
            if claim.get("retained") is True and claim.get("support_label") in {"unsupported", "unmapped"}:
                violations["retained_unsupported_claim"] += 1
            if (
                (audit.get("dynamic") is True or audit.get("high_consequence") is True)
                and claim.get("retained") is True
                and claim.get("material") is True
                and claim.get("support_label") != "supported"
            ):
                violations["dynamic_unsupported_conclusion"] += 1
            if (
                claim.get("retained") is True
                and claim.get("material") is True
                and claim.get("support_label") == "partial"
                and not (
                    partial_topics
                    and partial_topics.issubset(claim_topics)
                    and partial_topics.issubset(missing_topics)
                    and not conflict_group_ids
                    and "partial_evidence" in claim_disclosures
                    and "partial_evidence" in disclosures
                )
            ):
                violations["retained_claim_partial_structure"] += 1
            if (
                claim.get("retained") is True
                and claim.get("material") is True
                and claim.get("support_label") == "conflict"
            ):
                referenced_groups = [
                    conflict_group_by_id.get(group_id)
                    for group_id in conflict_group_ids
                ]
                if not (
                    conflict_group_ids
                    and not partial_topics
                    and all(isinstance(group, Mapping) for group in referenced_groups)
                    and all(
                        len(group.get("member_evidence_ids") or ()) >= 2
                        and set(group.get("member_evidence_ids") or ()).issubset(set(mapped))
                        for group in referenced_groups
                        if isinstance(group, Mapping)
                    )
                    and "source_conflict" in claim_disclosures
                    and "source_conflict" in disclosures
                ):
                    violations["retained_claim_conflict_structure"] += 1
        if retained_material_edge_ids != set(used_ids):
            retained_edge_violation = True
        if retained_edge_violation:
            violations["retained_claim_evidence_admission"] += 1
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            if not _valid_http_url(item.get("final_url")):
                violations["invalid_evidence_url"] += 1
            if item.get("citable") is True and item.get("relevance") in {
                "irrelevant", "excluded", "fail", None,
            }:
                violations["citable_without_relevance"] += 1
        if any(item not in evidence_by_id or evidence_by_id[item].get("citable") is not True for item in used_ids):
            violations["used_evidence_mapping"] += 1
        expected_urls = [evidence_by_id[item].get("final_url") for item in used_ids if item in evidence_by_id]
        if shown_urls != expected_urls:
            violations["shown_source_mapping"] += 1
        if evidence_state == "partial" and (
            not missing_topics or "partial_evidence" not in disclosures
        ):
            violations["partial_without_missing_topics_or_disclosure"] += 1
        if evidence_state == "conflicting":
            groups = audit.get("conflict_groups") if isinstance(audit.get("conflict_groups"), list) else []
            valid_group = any(
                isinstance(group, Mapping)
                and len(group.get("member_evidence_ids") or ()) >= 2
                and all(member in evidence_by_id for member in group.get("member_evidence_ids") or ())
                for group in groups
            )
            if not valid_group or "source_conflict" not in disclosures:
                violations["conflict_without_members_or_disclosure"] += 1
        if retained_partial and not (
            evidence_state == "partial"
            and degradation == "partial_evidence"
            and bool(missing_topics)
            and "partial_evidence" in disclosures
        ):
            violations["retained_claim_rendering_contract"] += 1
        if retained_conflict:
            groups = audit.get("conflict_groups") if isinstance(audit.get("conflict_groups"), list) else []
            conflict_members = {
                member
                for group in groups if isinstance(group, Mapping)
                for member in (group.get("member_evidence_ids") or ())
                if isinstance(member, str)
            }
            if not (
                evidence_state == "conflicting"
                and degradation == "source_conflict"
                and "source_conflict" in disclosures
                and len(conflict_members.intersection(retained_material_edge_ids)) >= 2
            ):
                violations["retained_claim_rendering_contract"] += 1
        disclosure_for_failure = {
            "partial_evidence": "partial_evidence",
            "source_conflict": "source_conflict",
            "validation_failed": "verification_failed",
            "provider_not_configured": "provider_not_configured",
            "provider_unavailable": "provider_unavailable",
            "provider_timeout": "provider_timeout",
            "no_results": "no_results",
            "content_unreadable": "content_unreadable",
            "insufficient_evidence": "insufficient_evidence",
            "user_forbid_web": "user_forbid_web",
        }
        if degradation in disclosure_for_failure and disclosure_for_failure[degradation] not in disclosures:
            violations["failure_without_disclosure"] += 1
        final_tier = route
        minimum_tier = audit.get("minimum_tier") or "skip"
        if _TIER_RANK.get(final_tier, -1) < _TIER_RANK.get(minimum_tier, 0):
            violations["final_tier_below_audit_floor"] += 1
        acceptable_tiers = audit.get("acceptable_final_tiers")
        if isinstance(acceptable_tiers, list) and acceptable_tiers and final_tier not in acceptable_tiers:
            violations["final_tier_outside_reviewed_target"] += 1
    return violations


def structural_violations(traces: Iterable[Any]) -> int:
    values = deterministic_invariant_violations(traces)
    return values["skip_with_provider_attempt"] + values["search_without_orchestrator"]


def _validate_trace(trace: Mapping[str, Any], index: int) -> list[str]:
    prefix = f"trace row {index}"
    errors: list[str] = []
    missing = sorted(TRACE_FIELDS - set(trace))
    unexpected = sorted(set(trace) - TRACE_FIELDS)
    if missing:
        errors.append(f"{prefix} missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"{prefix} unexpected fields: {', '.join(unexpected)}")
    if not _normalized_identifier(trace.get("request_id")):
        errors.append(f"{prefix} invalid request_id")
    if not _closed(trace.get("request_source"), REQUEST_SOURCES):
        errors.append(f"{prefix} invalid request_source")
    route = trace.get("route")
    if not _closed(route, ROUTES):
        errors.append(f"{prefix} invalid route")
    skip_reason = trace.get("skip_reason")
    if skip_reason is not None and not _closed(skip_reason, SKIP_REASONS):
        errors.append(f"{prefix} invalid skip_reason")
    if route == "skip" and not _closed(trace.get("skip_reason"), SKIP_REASONS):
        errors.append(f"{prefix} skip route requires a closed skip_reason")
    if _closed(route, TIERS) and trace.get("skip_reason") is not None:
        errors.append(f"{prefix} search route cannot set skip_reason")
    if route == "skip" and _closed(skip_reason, SKIP_REASONS):
        contract = SKIP_REASON_CONTRACTS[skip_reason]
        skip_contract_valid = (
            trace.get("provider_configured") is False
            and trace.get("provider_failures") == []
            and trace.get("evidence_state") is None
            and trace.get("degradation_reason") in contract["allowed_degradations"]
            and trace.get("knowledge_fallback_used") is False
        )
        if not skip_contract_valid:
            errors.append(f"{prefix} skip contract mismatch for {skip_reason}")
    for name in (
        "orchestrator_started", "initial_round_started",
        "adaptive_repair_round_started", "provider_configured",
        "provider_invocation_started", "knowledge_fallback_used", "repair_used",
        "provider_attempted", "sufficient_evidence",
    ):
        if type(trace.get(name)) is not bool:
            errors.append(f"{prefix} {name} must be boolean")
    for name in (
        "initial_query_count", "retrieval_round_count", "candidate_url_count",
        "content_read_count", "semantic_query_count", "repair_query_count",
        "citable_evidence_count", "claim_count", "supported_claim_count", "citation_count",
    ):
        if not _is_int(trace.get(name)):
            errors.append(f"{prefix} {name} must be a non-negative integer")
    for name in ("initial_query_redaction_codes", "adaptive_repair_redaction_codes"):
        values = trace.get(name)
        if not _is_string_list(values) or any(not _closed(value, REDACTION_CODES) for value in values or ()):
            errors.append(f"{prefix} invalid {name}")
    executed = trace.get("executed_queries")
    query_purposes: dict[str, str] = {}
    if not isinstance(executed, list):
        errors.append(f"{prefix} executed_queries must be a list")
    else:
        for query_index, query in enumerate(executed, 1):
            query_prefix = f"{prefix} executed query {query_index}"
            if not isinstance(query, Mapping):
                errors.append(f"{query_prefix} must be an object")
                continue
            query_missing = sorted(QUERY_FIELDS - set(query))
            query_unexpected = sorted(set(query) - QUERY_FIELDS)
            if query_missing:
                errors.append(f"{query_prefix} missing fields: {', '.join(query_missing)}")
            if query_unexpected:
                errors.append(f"{query_prefix} unexpected fields: {', '.join(query_unexpected)}")
            query_id = query.get("query_id")
            purpose = query.get("purpose")
            if not _normalized_identifier(query_id):
                errors.append(f"{query_prefix} invalid query_id")
            if not _closed(purpose, QUERY_PURPOSES):
                errors.append(f"{query_prefix} invalid purpose")
            if (
                _is_nonempty_string(query_id)
                and query_id in query_purposes
                and query_purposes[query_id] != purpose
            ):
                errors.append(f"{query_prefix} query_id has conflicting purposes")
            if _is_nonempty_string(query_id) and isinstance(purpose, str):
                query_purposes[query_id] = purpose
    attempts = trace.get("provider_attempts")
    if not isinstance(attempts, list):
        errors.append(f"{prefix} provider_attempts must be a list")
    else:
        for attempt_index, attempt in enumerate(attempts, 1):
            if not isinstance(attempt, dict):
                errors.append(f"{prefix} provider attempt {attempt_index} must be an object")
                continue
            attempt_prefix = f"{prefix} provider attempt {attempt_index}"
            attempt_missing = sorted(PROVIDER_ATTEMPT_FIELDS - set(attempt))
            attempt_unexpected = sorted(set(attempt) - PROVIDER_ATTEMPT_FIELDS)
            if attempt_missing:
                errors.append(f"{attempt_prefix} missing fields: {', '.join(attempt_missing)}")
            if attempt_unexpected:
                errors.append(f"{attempt_prefix} unexpected fields: {', '.join(attempt_unexpected)}")
            if not _closed(attempt.get("provider"), PROVIDER_NAMES):
                errors.append(f"{attempt_prefix} invalid provider")
            if not _closed(attempt.get("status"), PROVIDER_STATUSES):
                errors.append(f"{prefix} provider attempt {attempt_index} invalid status")
            if not _is_int(attempt.get("count")):
                errors.append(f"{attempt_prefix} count must be a non-negative integer")
            if not _is_nonnegative_number(attempt.get("latency_ms")):
                errors.append(f"{prefix} provider attempt {attempt_index} invalid latency_ms")
            if not _normalized_identifier(attempt.get("query_id")):
                errors.append(f"{attempt_prefix} invalid query_id")
            for name in ("configured", "available", "invocation_started"):
                if type(attempt.get(name)) is not bool:
                    errors.append(f"{attempt_prefix} {name} must be boolean")
            configured = attempt.get("configured")
            available = attempt.get("available")
            invoked = attempt.get("invocation_started")
            status = attempt.get("status")
            if configured is False and (available is not False or status != "not_configured"):
                errors.append(f"{attempt_prefix} impossible unconfigured state")
            if configured is True and available is False and status != "unavailable":
                errors.append(f"{attempt_prefix} impossible unavailable state")
            if available is True and _closed(status, {"not_configured", "unavailable"}):
                errors.append(f"{attempt_prefix} impossible available status")
            if invoked is True and attempt.get("count") == 0:
                errors.append(f"{attempt_prefix} invoked attempt requires positive count")
            count = attempt.get("count")
            if invoked is False and not (count is None or count == 0):
                errors.append(f"{attempt_prefix} non-invoked attempt must have zero count")
    failures = trace.get("provider_failures")
    if not isinstance(failures, list) or any(not _closed(value, FAILURE_CODES) for value in failures):
        errors.append(f"{prefix} invalid provider_failures")
    if trace.get("evidence_state") is not None and not _closed(trace.get("evidence_state"), EVIDENCE_STATES):
        errors.append(f"{prefix} invalid evidence_state")
    if trace.get("degradation_reason") is not None and not _closed(trace.get("degradation_reason"), FAILURE_CODES):
        errors.append(f"{prefix} invalid degradation_reason")
    for name in LATENCY_FIELDS:
        if not _is_nonnegative_number(trace.get(name)):
            errors.append(f"{prefix} {name} must be a finite non-negative number")
    if isinstance(executed, list):
        derived = {
            "initial_query_count": _initial_query_count(trace),
            "semantic_query_count": _semantic_query_count(trace),
            "repair_query_count": _repair_query_count(trace),
            "retrieval_round_count": _retrieval_round_count(trace),
        }
        for name, value in derived.items():
            if trace.get(name) != value:
                errors.append(f"{prefix} {name} disagrees with derived value {value}")
    repair_started = trace.get("adaptive_repair_round_started") is True
    if repair_started != (trace.get("repair_used") is True):
        errors.append(f"{prefix} repair_used disagrees with adaptive repair round")
    if repair_started and _repair_query_count(trace) != 1:
        errors.append(f"{prefix} adaptive repair round requires exactly one repair query")
    if not repair_started and trace.get("adaptive_repair_latency_ms") != 0:
        errors.append(f"{prefix} non-started adaptive repair requires zero latency")
    if trace.get("provider_attempted") != trace.get("provider_invocation_started"):
        errors.append(f"{prefix} provider_attempted disagrees with provider_invocation_started")
    if trace.get("sufficient_evidence") != (trace.get("evidence_state") == "sufficient"):
        errors.append(f"{prefix} sufficient_evidence disagrees with evidence_state")
    if isinstance(attempts, list) and trace.get("provider_invocation_started") != any(
        isinstance(item, Mapping) and item.get("invocation_started") is True for item in attempts
    ):
        errors.append(f"{prefix} provider_invocation_started disagrees with provider attempts")
    if isinstance(attempts, list) and isinstance(executed, list):
        executed_ids = {
            item.get("query_id") for item in executed
            if isinstance(item, Mapping) and _is_nonempty_string(item.get("query_id"))
        }
        for attempt_index, attempt in enumerate(attempts, 1):
            if (
                isinstance(attempt, Mapping)
                and (
                    not _is_nonempty_string(attempt.get("query_id"))
                    or attempt.get("query_id") not in executed_ids
                )
            ):
                errors.append(
                    f"{prefix} provider attempt {attempt_index} query_id is not an executed query"
                )
    if route == "skip" and any(
        bool(trace.get(name)) for name in (
            "orchestrator_started", "initial_round_started", "adaptive_repair_round_started",
            "provider_invocation_started", "provider_attempted", "sufficient_evidence",
        )
    ):
        errors.append(f"{prefix} skip route has impossible execution state")
    if route == "skip" and any(
        _counter(trace, name) > 0 for name in (
            "initial_query_count", "retrieval_round_count", "candidate_url_count",
            "content_read_count", "semantic_query_count", "repair_query_count",
            "citable_evidence_count", "claim_count", "supported_claim_count",
            "citation_count",
        )
    ):
        errors.append(f"{prefix} skip route has nonzero execution counters")
    return errors


def _validate_audit(audit: Mapping[str, Any], index: int) -> list[str]:
    prefix = f"audit row {index}"
    errors: list[str] = []
    missing = sorted(AUDIT_FIELDS - set(audit))
    unexpected = sorted(set(audit) - AUDIT_FIELDS)
    if missing:
        errors.append(f"{prefix} missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"{prefix} unexpected fields: {', '.join(unexpected)}")
    for name in ("case_id", "request_id"):
        if not _normalized_identifier(audit.get(name)):
            errors.append(f"{prefix} invalid {name}")
    if not _closed(audit.get("category"), set(CATEGORY_QUOTAS)):
        errors.append(f"{prefix} invalid category")
    for name in ("allow_skip", "external_fact_required", "explicit_search", "dynamic", "high_consequence"):
        if type(audit.get(name)) is not bool:
            errors.append(f"{prefix} {name} must be boolean")
    if audit.get("skip_reason") is not None and not _closed(audit.get("skip_reason"), SKIP_REASONS):
        errors.append(f"{prefix} invalid skip_reason")
    if audit.get("minimum_tier") is not None and not _closed(audit.get("minimum_tier"), TIERS):
        errors.append(f"{prefix} invalid minimum_tier")
    acceptable = audit.get("acceptable_final_tiers")
    if not _is_string_list(acceptable, allow_empty=False) or any(not _closed(value, ROUTES) for value in acceptable or ()) or len(set(acceptable or ())) != len(acceptable or ()):
        errors.append(f"{prefix} invalid acceptable_final_tiers")
    if not _closed(audit.get("label_status"), LABEL_STATUSES):
        errors.append(f"{prefix} invalid label_status")
    if not (
        audit.get("label_status") == "reviewed"
        and _normalized_identifier(audit.get("reviewed_by"))
        and _valid_date(audit.get("reviewed_at"))
    ):
        errors.append(f"{prefix} owner review is incomplete")
    if audit.get("allow_skip") is True:
        if audit.get("skip_reason") is None or audit.get("minimum_tier") is not None:
            errors.append(f"{prefix} invalid skip label state")
        if acceptable != ["skip"]:
            errors.append(f"{prefix} skip label requires acceptable_final_tiers=['skip']")
    elif audit.get("skip_reason") is not None:
        errors.append(f"{prefix} non-skippable row cannot set skip_reason")
    if audit.get("external_fact_required") is True and audit.get("allow_skip") is False:
        minimum = audit.get("minimum_tier")
        if not _closed(minimum, TIERS):
            errors.append(f"{prefix} factual non-skip row requires minimum_tier")
        if isinstance(acceptable, list) and any(not _closed(tier, TIERS) for tier in acceptable):
            errors.append(f"{prefix} factual non-skip targets cannot include skip")
        if _closed(minimum, TIERS) and isinstance(acceptable, list) and any(
            not _closed(tier, TIERS) or _TIER_RANK[tier] < _TIER_RANK[minimum]
            for tier in acceptable
        ):
            errors.append(f"{prefix} acceptable_final_tiers fall below minimum_tier")

    claims = audit.get("claims")
    claim_ids: list[Any] = []
    if not isinstance(claims, list):
        errors.append(f"{prefix} claims must be a list")
    else:
        for item_index, claim in enumerate(claims, 1):
            item_prefix = f"{prefix} claim {item_index}"
            if not isinstance(claim, Mapping):
                errors.append(f"{item_prefix} must be an object")
                continue
            item_missing = sorted(CLAIM_AUDIT_FIELDS - set(claim))
            item_unexpected = sorted(set(claim) - CLAIM_AUDIT_FIELDS)
            if item_missing:
                errors.append(f"{item_prefix} missing fields: {', '.join(item_missing)}")
            if item_unexpected:
                errors.append(f"{item_prefix} unexpected fields: {', '.join(item_unexpected)}")
            if not _normalized_identifier(claim.get("claim_id")):
                errors.append(f"{item_prefix} invalid claim_id")
            claim_ids.append(claim.get("claim_id"))
            for name in ("material", "retained"):
                if type(claim.get(name)) is not bool:
                    errors.append(f"{item_prefix} {name} must be boolean")
            if not _closed(claim.get("support_label"), SUPPORT_LABELS):
                errors.append(f"{item_prefix} invalid support_label")
            for name in (
                "evidence_ids", "topic_ids", "partial_topic_ids",
                "conflict_group_ids",
            ):
                if not _is_string_list(claim.get(name)) or len(set(claim.get(name) or ())) != len(claim.get(name) or ()):
                    errors.append(f"{item_prefix} invalid {name}")
            claim_disclosures = claim.get("disclosure_codes")
            if (
                not _is_string_list(claim_disclosures)
                or any(
                    not _closed(value, DISCLOSURE_CODES)
                    for value in claim_disclosures or ()
                )
                or len(set(claim_disclosures or ())) != len(claim_disclosures or ())
            ):
                errors.append(f"{item_prefix} invalid disclosure_codes")
            support = claim.get("support_label")
            partial_topics = claim.get("partial_topic_ids")
            conflict_refs = claim.get("conflict_group_ids")
            if support == "partial":
                if (
                    not _is_string_list(partial_topics, allow_empty=False)
                    or conflict_refs != []
                    or not isinstance(claim_disclosures, list)
                    or "partial_evidence" not in claim_disclosures
                ):
                    errors.append(f"{item_prefix} partial claim requires only partial_topic_ids")
            elif support == "conflict":
                if (
                    not _is_string_list(conflict_refs, allow_empty=False)
                    or partial_topics != []
                    or not isinstance(claim_disclosures, list)
                    or "source_conflict" not in claim_disclosures
                ):
                    errors.append(f"{item_prefix} conflict claim requires only conflict_group_ids")
            elif support in SUPPORT_LABELS and (
                partial_topics != [] or conflict_refs != [] or claim_disclosures != []
            ):
                errors.append(f"{item_prefix} non-partial/conflict claim cannot reference partial/conflict structure")
    if _duplicate_values(claim_ids):
        errors.append(f"{prefix} duplicate claim_id")

    evidence = audit.get("evidence")
    evidence_ids: list[Any] = []
    if not isinstance(evidence, list):
        errors.append(f"{prefix} evidence must be a list")
    else:
        for item_index, item in enumerate(evidence, 1):
            item_prefix = f"{prefix} evidence {item_index}"
            if not isinstance(item, Mapping):
                errors.append(f"{item_prefix} must be an object")
                continue
            item_missing = sorted(EVIDENCE_AUDIT_FIELDS - set(item))
            item_unexpected = sorted(set(item) - EVIDENCE_AUDIT_FIELDS)
            if item_missing:
                errors.append(f"{item_prefix} missing fields: {', '.join(item_missing)}")
            if item_unexpected:
                errors.append(f"{item_prefix} unexpected fields: {', '.join(item_unexpected)}")
            if not _normalized_identifier(item.get("evidence_id")):
                errors.append(f"{item_prefix} invalid evidence_id")
            evidence_ids.append(item.get("evidence_id"))
            if not _valid_http_url(item.get("final_url")):
                errors.append(f"{item_prefix} invalid final_url")
            if not _closed(item.get("relevance"), RELEVANCE_LABELS):
                errors.append(f"{item_prefix} invalid relevance")
            if type(item.get("citable")) is not bool:
                errors.append(f"{item_prefix} citable must be boolean")
    if _duplicate_values(evidence_ids):
        errors.append(f"{prefix} duplicate evidence_id")

    for name in ("used_evidence_ids", "shown_source_urls", "missing_claim_topics"):
        values = audit.get(name)
        if not _is_string_list(values) or len(set(values or ())) != len(values or ()):
            errors.append(f"{prefix} invalid {name}")
    for url in audit.get("shown_source_urls") or ():
        if not _valid_http_url(url):
            errors.append(f"{prefix} invalid shown source URL")
    groups = audit.get("conflict_groups")
    group_ids: list[Any] = []
    if not isinstance(groups, list):
        errors.append(f"{prefix} conflict_groups must be a list")
    else:
        for item_index, group in enumerate(groups, 1):
            item_prefix = f"{prefix} conflict group {item_index}"
            if not isinstance(group, Mapping):
                errors.append(f"{item_prefix} must be an object")
                continue
            item_missing = sorted(CONFLICT_AUDIT_FIELDS - set(group))
            item_unexpected = sorted(set(group) - CONFLICT_AUDIT_FIELDS)
            if item_missing:
                errors.append(f"{item_prefix} missing fields: {', '.join(item_missing)}")
            if item_unexpected:
                errors.append(f"{item_prefix} unexpected fields: {', '.join(item_unexpected)}")
            if not _normalized_identifier(group.get("group_id")):
                errors.append(f"{item_prefix} invalid group_id")
            group_ids.append(group.get("group_id"))
            if not _is_string_list(group.get("member_evidence_ids"), allow_empty=False) or len(set(group.get("member_evidence_ids") or ())) != len(group.get("member_evidence_ids") or ()):
                errors.append(f"{item_prefix} invalid member_evidence_ids")
    if _duplicate_values(group_ids):
        errors.append(f"{prefix} duplicate conflict group_id")
    disclosures = audit.get("rendered_disclosures")
    if not _is_string_list(disclosures) or any(not _closed(value, DISCLOSURE_CODES) for value in disclosures or ()) or len(set(disclosures or ())) != len(disclosures or ()):
        errors.append(f"{prefix} invalid rendered_disclosures")
    stages = audit.get("stages_started")
    if not _is_string_list(stages, allow_empty=False) or any(not _closed(value, set(STAGE_TO_LATENCY)) for value in stages or ()) or len(set(stages or ())) != len(stages or ()):
        errors.append(f"{prefix} invalid stages_started")
    return errors


def _validate_trace_audit_pair(
    trace: Mapping[str, Any], audit: Mapping[str, Any], index: int,
) -> list[str]:
    prefix = f"joined row {index}"
    errors: list[str] = []
    if trace.get("skip_reason") != audit.get("skip_reason"):
        errors.append(f"{prefix} trace/audit skip_reason mismatch")
    if audit.get("skip_reason") == "user_forbid_web":
        if trace.get("route") != "skip":
            errors.append(f"{prefix} explicit no-web audit requires matching skip trigger")
        if trace.get("degradation_reason") not in {None, "user_forbid_web"}:
            errors.append(f"{prefix} explicit no-web trace has wrong degradation_reason")
    stages = {
        value for value in (audit.get("stages_started") or ())
        if isinstance(value, str)
    } if isinstance(audit.get("stages_started"), list) else set()
    repair_stage_started = "adaptive_repair" in stages
    if repair_stage_started != (trace.get("adaptive_repair_round_started") is True):
        errors.append(f"{prefix} adaptive_repair stage disagrees with trace repair state")
    for stage, latency_field in STAGE_TO_LATENCY.items():
        latency = trace.get(latency_field)
        if _is_nonnegative_number(latency) and latency > 0 and stage not in stages:
            errors.append(
                f"{prefix} positive {latency_field} requires stage {stage}"
            )
    required_stages = {"route", "total_response", "qq_render"}
    if trace.get("orchestrator_started") is True:
        required_stages.update({"query_planning", "retrieval_pipeline"})
    if trace.get("initial_round_started") is True:
        required_stages.update({"initial_provider_search", "provider_search_total"})
    if trace.get("provider_invocation_started") is True:
        required_stages.update({"initial_provider_search", "provider_search_total"})
    if _counter(trace, "content_read_count") > 0:
        required_stages.update({"initial_content_read", "content_read_total"})
    if trace.get("evidence_state") is not None or _counter(trace, "citable_evidence_count") > 0:
        required_stages.update({
            "initial_evidence_assembly", "evidence_assembly_total", "gap_analysis",
        })
    if trace.get("adaptive_repair_round_started") is True:
        required_stages.add("adaptive_repair")
    if trace.get("evidence_state") is not None or _counter(trace, "claim_count") > 0:
        required_stages.update({
            "answer_generation", "structural_validation", "semantic_validation",
        })
    for stage in sorted(required_stages - stages):
        errors.append(f"{prefix} execution facts require stage {stage}")
    return errors


def _rate(rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any] | None]], predicate: Any) -> dict[str, Any]:
    numerator = sum(trace is not None and predicate(trace) for _label, trace in rows)
    denominator = len(rows)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _external_label_is_explicit(label: Mapping[str, Any]) -> bool:
    return label.get("explicit_search") is True or label.get("category") == "explicit_search"


def _evaluate_traces_impl(
    trace_rows: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    *,
    sample_manifest: Mapping[str, Any] | None = None,
    trusted_verifier_key: bytes | bytearray | None = None,
) -> dict[str, Any]:
    errors: list[str] = _validate_sample_manifest(
        sample_manifest, trace_rows, audits, trusted_verifier_key,
    )
    for index, trace in enumerate(trace_rows, 1):
        errors.extend(_validate_trace(trace, index))
    for index, audit in enumerate(audits, 1):
        errors.extend(_validate_audit(audit, index))

    audit_case_ids = [audit.get("case_id") for audit in audits]
    audit_request_ids = [audit.get("request_id") for audit in audits]
    trace_request_ids = [trace.get("request_id") for trace in trace_rows]
    for case_id in _duplicate_values(audit_case_ids):
        errors.append(f"duplicate audit case_id {case_id}")
    for request_id in _duplicate_values(audit_request_ids):
        errors.append(f"duplicate audit request_id {request_id}")
    for request_id in _duplicate_values(trace_request_ids):
        errors.append(f"duplicate trace request_id {request_id}")
    audit_by_request_id = {
        audit["request_id"]: audit for audit in audits if _normalized_identifier(audit.get("request_id"))
    }
    trace_by_request_id: dict[str, Mapping[str, Any]] = {}
    for trace in trace_rows:
        request_id = trace.get("request_id")
        if not _normalized_identifier(request_id):
            continue
        if request_id not in audit_by_request_id:
            errors.append(f"unknown trace request_id {request_id}")
            continue
        trace_by_request_id.setdefault(request_id, trace)
    for request_id, audit in audit_by_request_id.items():
        if request_id not in trace_by_request_id:
            errors.append(f"missing trace for case_id {audit.get('case_id')} request_id {request_id}")
    for pair_index, (request_id, audit) in enumerate(audit_by_request_id.items(), 1):
        trace = trace_by_request_id.get(request_id)
        if trace is not None:
            errors.extend(_validate_trace_audit_pair(trace, audit, pair_index))

    exclusions = {"explicit_no_web": 0, "legal_closed_context": 0}
    d_factual_rows: list[tuple[Mapping[str, Any], Mapping[str, Any] | None]] = []
    explicit_rows: list[tuple[Mapping[str, Any], Mapping[str, Any] | None]] = []
    no_web_rows: list[tuple[Mapping[str, Any], Mapping[str, Any] | None]] = []
    for request_id, audit in audit_by_request_id.items():
        trace = trace_by_request_id.get(request_id)
        if audit.get("skip_reason") == "user_forbid_web":
            exclusions["explicit_no_web"] += 1
            no_web_rows.append((audit, trace))
            continue
        if audit.get("allow_skip") is True:
            exclusions["legal_closed_context"] += 1
        if audit.get("external_fact_required") is True and audit.get("allow_skip") is not True:
            d_factual_rows.append((audit, trace))
        if audit.get("explicit_search") is True or audit.get("category") == "explicit_search":
            explicit_rows.append((audit, trace))

    routed = lambda trace: _route(trace) in TIERS
    orchestrated = lambda trace: bool(_field(trace, "orchestrator_started"))
    attempted = lambda trace: bool(_field(trace, "provider_invocation_started", _field(trace, "provider_attempted", False)))
    sufficient = lambda trace: bool(_field(trace, "sufficient_evidence", _field(trace, "evidence_state") == "sufficient"))
    configured_explicit = [
        (label, trace) for label, trace in explicit_rows
        if trace is not None and bool(trace.get("provider_configured"))
        and _no_attempt_readiness_failure(trace) != "provider_unavailable"
    ]
    configured_factual = [
        (label, trace) for label, trace in d_factual_rows
        if trace is not None and bool(trace.get("provider_configured"))
        and _no_attempt_readiness_failure(trace) != "provider_unavailable"
    ]
    provider_execution_accounted_numerator = sum(
        trace is not None and (
            attempted(trace)
            or _coherent_no_attempt_readiness_failure(trace, audit)
        )
        for audit, trace in d_factual_rows
    )
    provider_execution_accounted_rate = {
        "numerator": provider_execution_accounted_numerator,
        "denominator": len(d_factual_rows),
        "rate": (
            provider_execution_accounted_numerator / len(d_factual_rows)
            if d_factual_rows else None
        ),
    }
    rates = {
        "route_coverage": _rate(d_factual_rows, routed),
        "orchestrator_start_rate": _rate(d_factual_rows, orchestrated),
        "provider_attempt_rate": _rate(d_factual_rows, attempted),
        "provider_attempt_rate_configured": _rate(configured_factual, attempted),
        "sufficient_evidence_rate": _rate(d_factual_rows, sufficient),
        "provider_execution_accounted_rate": provider_execution_accounted_rate,
        "explicit_search_route_rate": _rate(explicit_rows, routed),
        "explicit_search_orchestrator_start_rate": _rate(explicit_rows, orchestrated),
        "explicit_search_provider_attempt_rate_configured": _rate(configured_explicit, attempted),
        "explicit_no_web_zero_provider_rate": _rate(
            no_web_rows,
            lambda trace: not bool(trace.get("provider_invocation_started")) and not bool(trace.get("provider_attempts")),
        ),
    }

    conditional_rates = {
        "orchestrated_per_routed": _rate(
            [(audit, trace) for audit, trace in d_factual_rows if trace is not None and routed(trace)],
            orchestrated,
        ),
        "attempted_per_orchestrated": _rate(
            [(audit, trace) for audit, trace in d_factual_rows if trace is not None and orchestrated(trace)],
            attempted,
        ),
        "sufficient_per_attempted": _rate(
            [(audit, trace) for audit, trace in d_factual_rows if trace is not None and attempted(trace)],
            sufficient,
        ),
    }

    skip_reason_breakdown: dict[str, dict[str, Any]] = {}
    for reason in sorted(SKIP_REASONS):
        rows = [
            (audit, trace) for request_id, audit in audit_by_request_id.items()
            if audit.get("skip_reason") == reason
            for trace in [trace_by_request_id.get(request_id)]
        ]
        zero = _rate(rows, lambda trace: not bool(trace.get("provider_invocation_started")) and not bool(trace.get("provider_attempts")))
        skip_reason_breakdown[reason] = {"count": len(rows), "zero_provider": zero}

    execution_failures = {"provider_not_configured": 0, "provider_unavailable": 0}
    joined_traces = list(trace_by_request_id.values())
    outcome_counts = {code: 0 for code in sorted(FAILURE_CODES)}
    outcome_counts.update({state: 0 for state in sorted(EVIDENCE_STATES)})
    for trace in joined_traces:
        codes = {
            _enum_text(_field(trace, "degradation_reason")),
            *(_enum_text(value) for value in (_field(trace, "provider_failures", ()) or ())),
        }
        statuses = {
            _enum_text(attempt.get("status"))
            for attempt in (_field(trace, "provider_attempts", ()) or ())
            if isinstance(attempt, Mapping)
        }
        if "provider_not_configured" in codes or "not_configured" in statuses:
            execution_failures["provider_not_configured"] += 1
        if "provider_unavailable" in codes or "unavailable" in statuses:
            execution_failures["provider_unavailable"] += 1
        for code in codes - {""}:
            if code in outcome_counts:
                outcome_counts[code] += 1
        state = trace.get("evidence_state")
        if state in EVIDENCE_STATES:
            outcome_counts[state] += 1

    budgets = budget_violations(joined_traces)
    invariants = deterministic_invariant_violations(joined_traces, audit_by_request_id)
    audit_schema_valid = not any(error.startswith("audit row") for error in errors)
    trace_schema_valid = not any(error.startswith("trace row") for error in errors)
    join_complete = len(trace_by_request_id) == len(audit_by_request_id) == len(trace_rows)
    deterministic_evaluation = {
        "evaluable": audit_schema_valid and trace_schema_valid and join_complete,
        "joined_sample_count": len(trace_by_request_id),
    }
    latencies = latency_percentiles(joined_traces, audit_by_request_id)
    per_tier_retrieval: dict[str, dict[str, Any]] = {}
    for tier in TIERS:
        values = [
            trace["retrieval_pipeline_latency_ms"] for trace in joined_traces
            if trace.get("route") == tier
            and "retrieval_pipeline" in audit_by_request_id.get(str(trace.get("request_id")), {}).get("stages_started", [])
            and _is_nonnegative_number(trace.get("retrieval_pipeline_latency_ms"))
        ]
        per_tier_retrieval[tier] = {
            "evaluable": bool(values),
            "sample_count": len(values),
            "p50": _nearest_rank(values, 0.50),
            "p95": _nearest_rank(values, 0.95),
            "p99": _nearest_rank(values, 0.99),
        }

    failures: list[str] = []
    if errors:
        failures.append("trace or label integrity errors")
    if rates["route_coverage"]["denominator"] == 0:
        failures.append("D_factual zero samples")
    elif rates["route_coverage"]["rate"] < 0.98:
        failures.append("D_factual route coverage below 0.98")
    if rates["orchestrator_start_rate"]["denominator"] and rates["orchestrator_start_rate"]["rate"] < 1.0:
        failures.append("mandatory orchestrator start rate below 1.00")
    if rates["explicit_search_route_rate"]["denominator"] == 0:
        failures.append("explicit search zero samples")
    elif rates["explicit_search_route_rate"]["rate"] < 1.0:
        failures.append("explicit search route rate below 1.00")
    if rates["explicit_search_orchestrator_start_rate"]["denominator"] and rates["explicit_search_orchestrator_start_rate"]["rate"] < 1.0:
        failures.append("explicit search orchestrator start rate below 1.00")
    configured_rate = rates["explicit_search_provider_attempt_rate_configured"]
    if configured_rate["denominator"] and configured_rate["rate"] < 1.0:
        failures.append("configured explicit search provider attempt rate below 1.00")
    factual_configured_rate = rates["provider_attempt_rate_configured"]
    if factual_configured_rate["denominator"] and factual_configured_rate["rate"] < 1.0:
        failures.append("configured provider attempt rate below 1.00")
    accounted_rate = rates["provider_execution_accounted_rate"]
    if accounted_rate["denominator"] == 0 or accounted_rate["rate"] < 1.0:
        failures.append("provider execution accounting rate below 1.00")
    retrieval_p95_limits = {"light": 6_000, "standard": 15_000}
    for tier, limit in retrieval_p95_limits.items():
        p95 = per_tier_retrieval[tier]["p95"]
        if p95 is None:
            failures.append(f"{tier} latency zero samples")
        elif p95 > limit:
            failures.append(f"{tier} retrieval P95 above {limit} ms")
    for field, summary in latencies.items():
        if summary["sample_count"] == 0:
            failures.append(f"{field} zero samples")
    if any(budgets.values()):
        failures.append("budget or hard-timeout violations")
    if not deterministic_evaluation["evaluable"]:
        failures.append("deterministic audit is non-evaluable")
    if any(invariants.values()):
        failures.append("deterministic citation/failure invariant violations")
    if not isinstance(sample_manifest, Mapping):
        failures.append("controlled sample manifest is required")
    elif sample_manifest.get("provenance") != "controlled_production" or sample_manifest.get("fixture_derived") is not False:
        failures.append("trace sample is fixture-derived or uncontrolled")
    fixture_urls = _fixture_evidence_urls(audits)
    if fixture_urls:
        failures.append("fixture/example evidence URL is diagnostic and non-certifying")
    trusted_attestation_verified = not _validate_trusted_attestation(
        sample_manifest, trusted_verifier_key, "sample manifest",
    )

    return {
        "mode": "traces",
        "certifying": not failures,
        "joined_case_count": len(trace_by_request_id),
        "sample_provenance": dict(sample_manifest) if isinstance(sample_manifest, Mapping) else None,
        "trusted_attestation_verified": trusted_attestation_verified,
        "errors": errors,
        "exclusions": exclusions,
        "rates": rates,
        "conditional_rates": conditional_rates,
        "skip_reason_breakdown": skip_reason_breakdown,
        "outcome_counts": outcome_counts,
        "execution_failures": execution_failures,
        "budget_violations": budgets,
        "deterministic_invariant_violations": invariants,
        "deterministic_evaluation": deterministic_evaluation,
        "initial_batch_count": initial_batch_round_count(joined_traces),
        "repair_round_count": repair_round_count(joined_traces),
        "latencies_ms": latencies,
        "per_tier_retrieval_pipeline_latency_ms": per_tier_retrieval,
        "failures": failures,
    }


def evaluate_traces(
    trace_rows: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    *,
    sample_manifest: Mapping[str, Any] | None = None,
    trusted_verifier_key: bytes | bytearray | None = None,
) -> dict[str, Any]:
    try:
        return _evaluate_traces_impl(
            trace_rows, audits, sample_manifest=sample_manifest,
            trusted_verifier_key=trusted_verifier_key,
        )
    except Exception as exc:
        return {
            "mode": "traces", "certifying": False,
            "trusted_attestation_verified": False,
            "deterministic_evaluation": {"evaluable": False, "joined_sample_count": 0},
            "errors": [f"controlled trace validation error ({type(exc).__name__})"],
            "failures": ["trace validation could not safely evaluate the artifacts"],
        }


def _load_json_object_checked(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"{path}: file does not exist"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, [f"{path}: could not read file ({type(exc).__name__})"]
    except json.JSONDecodeError as exc:
        return None, [f"{path}: invalid JSON ({exc.msg})"]
    if not isinstance(value, dict):
        return None, [f"{path}: JSON value must be an object"]
    return value, []


def traces(
    traces_path: Path,
    labels_path: Path,
    manifest_path: Path | None = None,
    verifier_key_env: str | None = None,
) -> int:
    trace_rows, trace_errors = _load_jsonl_checked(traces_path)
    labels, label_errors = _load_jsonl_checked(labels_path)
    manifest, manifest_errors = (
        _load_json_object_checked(manifest_path)
        if manifest_path is not None
        else (None, ["controlled sample manifest is required"])
    )
    trusted_key, key_errors = _trusted_verifier_key_from_env(verifier_key_env)
    load_errors = [*trace_errors, *label_errors, *manifest_errors, *key_errors]
    if load_errors:
        report = {
            "mode": "traces", "certifying": False, "errors": load_errors,
            "failures": ["input JSONL integrity errors"],
        }
    else:
        report = evaluate_traces(
            trace_rows, labels, sample_manifest=manifest,
            trusted_verifier_key=trusted_key,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["certifying"] else 1


def online(limit: int) -> int:
    print(json.dumps({
        "mode": "online",
        "status": "not run",
        "certifying": False,
        "limit": limit,
        "reason": "not authorized; explicit user authorization and credentials are required",
    }, sort_keys=True))
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="evidence-search evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("integrity")
    offline_parser = sub.add_parser("offline")
    offline_parser.add_argument("--cases", type=Path)
    offline_parser.add_argument("--recordings", type=Path)
    offline_parser.add_argument("--predictions", type=Path)
    offline_parser.add_argument("--manifest", type=Path)
    offline_parser.add_argument("--verifier-key-env")
    traces_parser = sub.add_parser("traces")
    traces_parser.add_argument("--traces", required=True, type=Path)
    traces_parser.add_argument("--labels", required=True, type=Path)
    traces_parser.add_argument("--manifest", type=Path)
    traces_parser.add_argument("--verifier-key-env")
    online_parser = sub.add_parser("online")
    online_parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    if args.command == "integrity":
        return integrity()
    if args.command == "offline":
        return offline(
            args.cases, args.recordings, args.predictions, args.manifest,
            args.verifier_key_env,
        )
    if args.command == "traces":
        return traces(
            args.traces, args.labels, args.manifest, args.verifier_key_env,
        )
    if args.command == "online":
        return online(args.limit)
    return 2


if __name__ == "__main__":
    sys.exit(main())
