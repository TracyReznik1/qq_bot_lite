"""Evidence-search evaluation CLI.

The evaluator keeps deterministic Trace acceptance separate from model-quality
measurements.  Human labels are read only from case/audit files and are joined
to predictions or traces by ``case_id``; embedded labels in traces are ignored.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime
import json
import math
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

TIERS = ("light", "standard", "deep")
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
ACTIONABILITY = {"none", "general", "personalized"}
POTENTIAL_HARM = {"none", "low", "medium", "high"}
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
RECORDING_FIELDS = {
    "fixture_id", "case_id", "provider", "query_text", "title", "url",
    "excerpt", "expected_fetch_status",
}
PREDICTION_COMMON_FIELDS = {
    "case_id", "component", "model", "model_version", "prompt_schema_version",
    "run_timestamp",
}
PREDICTION_ALLOWED_FIELDS = PREDICTION_COMMON_FIELDS | {
    "predicted_tier", "predictions", "label_id", "predicted",
    "predicted_label", "predicted_query_purposes", "predicted_initial_query_count",
    "predicted_repair_used",
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
    "deep": {
        "initial_query_count": 5,
        "candidate_url_count": 15,
        "content_read_count": 8,
        "semantic_query_count": 6,
        "repair_query_count": 1,
        "retrieval_round_count": 2,
        "hard_timeout_ms": 40_000,
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


def latency_percentiles(traces: Iterable[Any]) -> dict[str, dict[str, float | int | None]]:
    rows = tuple(traces)
    result: dict[str, dict[str, float | int | None]] = {}
    for field in LATENCY_FIELDS:
        values = [
            value for row in rows
            if _is_nonnegative_number(value := _field(row, field))
        ]
        result[field] = {
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
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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
    counts = Counter(values)
    return [value for value, count in counts.items() if value and count > 1]


# ── integrity ──────────────────────────────────────────────────────────

def _validate_case(case: Mapping[str, Any], index: int) -> list[str]:
    prefix = f"case row {index}"
    errors: list[str] = []
    missing = sorted(CASE_FIELDS - set(case))
    unexpected = sorted(set(case) - CASE_FIELDS)
    if missing:
        errors.append(f"{prefix} missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"{prefix} unexpected fields: {', '.join(unexpected)}")
    case_id = case.get("case_id")
    display = case_id if _is_nonempty_string(case_id) else f"row-{index}"
    if not _is_nonempty_string(case_id):
        errors.append(f"{prefix} missing case_id")
    if case.get("category") not in CATEGORY_QUOTAS:
        errors.append(f"case {display} invalid category")
    if not _is_nonempty_string(case.get("question")):
        errors.append(f"case {display} missing question")
    if type(case.get("allow_skip")) is not bool:
        errors.append(f"case {display} allow_skip must be boolean")
    if case.get("skip_reason") is not None and case.get("skip_reason") not in SKIP_REASONS:
        errors.append(f"case {display} invalid skip_reason")
    if case.get("minimum_tier") is not None and case.get("minimum_tier") not in TIERS:
        errors.append(f"case {display} invalid minimum_tier")
    if type(case.get("external_fact_required")) is not bool:
        errors.append(f"case {display} external_fact_required must be boolean")
    if case.get("actionability") not in ACTIONABILITY:
        errors.append(f"case {display} invalid actionability")
    if case.get("potential_harm") not in POTENTIAL_HARM:
        errors.append(f"case {display} invalid potential_harm")
    purposes = case.get("expected_query_purposes")
    if not _is_string_list(purposes) or any(item not in QUERY_PURPOSES for item in purposes or ()):
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
    if not _is_string_list(relations, allow_empty=False) or any(item not in SOURCE_RELATIONS for item in relations or ()):
        errors.append(f"case {display} invalid acceptable_source_relations")
    if case.get("expected_outcome") not in EXPECTED_OUTCOMES:
        errors.append(f"case {display} invalid expected_outcome")
    if not _is_nonempty_string(case.get("fixture_id")):
        errors.append(f"case {display} missing fixture_id")
    if case.get("label_status") not in LABEL_STATUSES:
        errors.append(f"case {display} invalid label_status")

    reviewed = (
        case.get("label_status") == "reviewed"
        and _is_nonempty_string(case.get("reviewed_by"))
        and case.get("reviewed_by") != "unreviewed"
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
            label_id = label.get("label_id")
            component = label.get("component")
            expected = label.get("expected")
            if not _is_nonempty_string(label_id):
                errors.append(f"case {display} semantic label {label_index} missing label_id")
            if component not in QUALITY_COMPONENTS:
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
    if recording.get("expected_fetch_status") not in PROVIDER_STATUSES | {"unreadable"}:
        errors.append(f"{prefix} invalid expected_fetch_status")
    return errors


def _prediction_items(prediction: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = prediction.get("predictions")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if _is_nonempty_string(prediction.get("label_id")):
        return [{
            "label_id": prediction.get("label_id"),
            "predicted": prediction.get("predicted", prediction.get("predicted_label")),
        }]
    return []


def _validate_prediction(prediction: Mapping[str, Any], index: int, case_ids: set[str]) -> list[str]:
    prefix = f"prediction row {index}"
    errors: list[str] = []
    missing = sorted(PREDICTION_COMMON_FIELDS - set(prediction))
    unexpected = sorted(set(prediction) - PREDICTION_ALLOWED_FIELDS)
    if missing:
        errors.append(f"{prefix} missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"{prefix} unexpected fields: {', '.join(unexpected)}")
    leaked = sorted(set(prediction).intersection(HUMAN_LABEL_FIELDS))
    if leaked:
        errors.append(f"{prefix} contains human-label field: {', '.join(leaked)}")
    case_id = prediction.get("case_id")
    if not _is_nonempty_string(case_id):
        errors.append(f"{prefix} missing case_id")
    elif case_id not in case_ids:
        errors.append(f"{prefix} unknown case_id {case_id}")
    component = prediction.get("component")
    if component not in PREDICTION_COMPONENTS:
        errors.append(f"{prefix} invalid component")
    for name in ("model", "model_version", "prompt_schema_version"):
        if not _is_nonempty_string(prediction.get(name)):
            errors.append(f"{prefix} missing {name}")
    if not _valid_timestamp(prediction.get("run_timestamp")):
        errors.append(f"{prefix} invalid run_timestamp")
    if component == "router" and prediction.get("predicted_tier") not in ROUTES:
        errors.append(f"{prefix} invalid predicted_tier")
    if component in QUALITY_COMPONENTS:
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


def validate_integrity(
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
        and prediction.get("component") not in QUALITY_COMPONENTS
    ]
    for case_id, component in _duplicate_values(singleton_prediction_keys):
        errors.append(f"duplicate prediction for case_id {case_id} component {component}")
    quality_prediction_keys = [
        (prediction.get("case_id"), prediction.get("component"), item.get("label_id"))
        for prediction in predictions
        if _is_nonempty_string(prediction.get("case_id"))
        and prediction.get("component") in QUALITY_COMPONENTS
        for item in _prediction_items(prediction)
        if _is_nonempty_string(item.get("label_id"))
    ]
    for case_id, component, label_id in _duplicate_values(quality_prediction_keys):
        errors.append(
            f"duplicate prediction for case_id {case_id} component {component} label_id {label_id}"
        )
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
    if case.get("category") == "dynamic_fact" or case.get("dynamic") is True:
        groups.add("dynamic")
    if (
        case.get("potential_harm") == "high"
        or case.get("actionability") == "personalized"
        or case.get("high_consequence") is True
    ):
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
            if component not in QUALITY_COMPONENTS or not _is_nonempty_string(label_id):
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
        if component not in QUALITY_COMPONENTS or not _is_nonempty_string(case_id):
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


def evaluate_offline(
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
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

    confusion: Counter[tuple[str, str]] = Counter()
    route_pairs: list[tuple[Mapping[str, Any], str]] = []
    for case in cases:
        case_id = case.get("case_id")
        if not _is_nonempty_string(case_id):
            continue
        row = router_rows.get(case_id)
        if row is None:
            failures.append(f"missing router prediction for case_id {case_id}")
            continue
        expected = case.get("minimum_tier") or "skip"
        predicted = row.get("predicted_tier")
        if predicted not in ROUTES:
            failures.append(f"invalid router prediction for case_id {case_id}")
            continue
        confusion[(predicted, expected)] += 1
        route_pairs.append((case, predicted))

    tier = tier_metrics(confusion)
    mandatory = [(case, prediction) for case, prediction in route_pairs if in_d_factual(case)]
    explicit = [(case, prediction) for case, prediction in route_pairs if case.get("category") == "explicit_search"]
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

    fixture_baseline = any(row.get("model") == "fixture-baseline" for row in predictions)
    if fixture_baseline:
        failures.append("fixture baseline is diagnostic and non-certifying")
    owner_reviewed = all(
        case.get("label_status") == "reviewed"
        and _is_nonempty_string(case.get("reviewed_by"))
        and case.get("reviewed_by") != "unreviewed"
        and _valid_date(case.get("reviewed_at"))
        for case in cases
    ) and bool(cases)
    if not owner_reviewed:
        failures.append("human owner review is incomplete")

    failures = list(dict.fromkeys(failures))
    return {
        "mode": "offline",
        "certifying": not failures,
        "artifact_class": "fixture_baseline" if fixture_baseline else "independent_predictions",
        "case_count": len(cases),
        "router_sample_count": len(route_pairs),
        "tier_metrics": tier,
        "mandatory_search_route_rate": mandatory_rate,
        "explicit_search_route_rate": explicit_rate,
        "legal_non_factual_meaningless_search_rate": meaningless,
        "quality": quality,
        "failures": failures,
    }


def offline() -> int:
    cases, case_errors = _load_jsonl_checked(CASES_PATH)
    predictions, prediction_errors = _load_jsonl_checked(MODEL_PREDICTIONS_PATH)
    load_errors = [*case_errors, *prediction_errors]
    if load_errors:
        report = {"mode": "offline", "certifying": False, "failures": load_errors}
    else:
        report = evaluate_offline(cases, predictions)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["certifying"] else 1


# ── trace metrics and acceptance ───────────────────────────────────────

def _semantic_query_count(trace: Any) -> int:
    value = _field(trace, "semantic_query_count")
    if _is_int(value):
        return value
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
    return len(query_ids)


def _repair_query_count(trace: Any) -> int:
    value = _field(trace, "repair_query_count")
    if _is_int(value):
        return value
    return int(bool(_field(trace, "adaptive_repair_round_started")))


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
            "initial_query_count": _counter(trace, "initial_query_count"),
            "candidate_url_count": _counter(trace, "candidate_url_count"),
            "content_read_count": _counter(trace, "content_read_count"),
            "semantic_query_count": _semantic_query_count(trace),
            "repair_query_count": _repair_query_count(trace),
            "retrieval_round_count": _counter(trace, "retrieval_round_count"),
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


def deterministic_invariant_violations(traces: Iterable[Any]) -> dict[str, int]:
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
        provider_attempted = bool(
            _field(trace, "provider_invocation_started", _field(trace, "provider_attempted", False))
        ) or bool(_field(trace, "provider_attempts", ()))
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
    return violations


def structural_violations(traces: Iterable[Any]) -> int:
    values = deterministic_invariant_violations(traces)
    return values["skip_with_provider_attempt"] + values["search_without_orchestrator"]


def _validate_trace(trace: Mapping[str, Any], index: int) -> list[str]:
    prefix = f"trace row {index}"
    errors: list[str] = []
    case_id = trace.get("case_id")
    if not _is_nonempty_string(case_id):
        errors.append(f"{prefix} missing case_id")
    route = trace.get("route")
    if route not in ROUTES:
        errors.append(f"{prefix} invalid route")
    skip_reason = trace.get("skip_reason")
    if skip_reason is not None and skip_reason not in SKIP_REASONS:
        errors.append(f"{prefix} invalid skip_reason")
    for name in (
        "orchestrator_started", "initial_round_started",
        "adaptive_repair_round_started", "provider_configured",
        "provider_invocation_started", "knowledge_fallback_used",
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
    attempts = trace.get("provider_attempts")
    if not isinstance(attempts, list):
        errors.append(f"{prefix} provider_attempts must be a list")
    else:
        for attempt_index, attempt in enumerate(attempts, 1):
            if not isinstance(attempt, dict):
                errors.append(f"{prefix} provider attempt {attempt_index} must be an object")
                continue
            if attempt.get("status") not in PROVIDER_STATUSES:
                errors.append(f"{prefix} provider attempt {attempt_index} invalid status")
            if not _is_nonnegative_number(attempt.get("latency_ms")):
                errors.append(f"{prefix} provider attempt {attempt_index} invalid latency_ms")
    failures = trace.get("provider_failures")
    if not isinstance(failures, list) or any(value not in FAILURE_CODES for value in failures):
        errors.append(f"{prefix} invalid provider_failures")
    if trace.get("evidence_state") is not None and trace.get("evidence_state") not in EVIDENCE_STATES:
        errors.append(f"{prefix} invalid evidence_state")
    if trace.get("degradation_reason") is not None and trace.get("degradation_reason") not in FAILURE_CODES:
        errors.append(f"{prefix} invalid degradation_reason")
    for name in LATENCY_FIELDS:
        if not _is_nonnegative_number(trace.get(name)):
            errors.append(f"{prefix} {name} must be a finite non-negative number")
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


def evaluate_traces(
    trace_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    for index, trace in enumerate(trace_rows, 1):
        errors.extend(_validate_trace(trace, index))
    label_ids = [label.get("case_id") for label in labels]
    trace_ids = [trace.get("case_id") for trace in trace_rows]
    for case_id in _duplicate_values(label_ids):
        errors.append(f"duplicate label case_id {case_id}")
    for case_id in _duplicate_values(trace_ids):
        errors.append(f"duplicate trace case_id {case_id}")
    for index, label in enumerate(labels, 1):
        if not _is_nonempty_string(label.get("case_id")):
            errors.append(f"label row {index} missing case_id")
        if "d_factual" in label and type(label.get("d_factual")) is not bool:
            errors.append(f"label row {index} d_factual must be boolean")
        reviewed = (
            label.get("label_status") == "reviewed"
            and _is_nonempty_string(label.get("reviewed_by"))
            and label.get("reviewed_by") != "unreviewed"
            and _valid_date(label.get("reviewed_at"))
        )
        if not reviewed:
            errors.append(f"label row {index} owner review is incomplete")

    label_by_id = {
        label["case_id"]: label for label in labels if _is_nonempty_string(label.get("case_id"))
    }
    trace_by_id: dict[str, Mapping[str, Any]] = {}
    for trace in trace_rows:
        case_id = trace.get("case_id")
        if not _is_nonempty_string(case_id):
            continue
        if case_id not in label_by_id:
            errors.append(f"unknown trace case_id {case_id}")
            continue
        trace_by_id.setdefault(case_id, trace)
    for case_id in label_by_id:
        if case_id not in trace_by_id:
            errors.append(f"missing trace for case_id {case_id}")

    exclusions = {"explicit_no_web": 0, "legal_closed_context": 0}
    d_factual_rows: list[tuple[Mapping[str, Any], Mapping[str, Any] | None]] = []
    explicit_rows: list[tuple[Mapping[str, Any], Mapping[str, Any] | None]] = []
    for case_id, label in label_by_id.items():
        trace = trace_by_id.get(case_id)
        if label.get("skip_reason") == "user_forbid_web":
            exclusions["explicit_no_web"] += 1
        elif label.get("allow_skip") is True:
            exclusions["legal_closed_context"] += 1
        if in_d_factual(label):
            d_factual_rows.append((label, trace))
        if _external_label_is_explicit(label):
            explicit_rows.append((label, trace))

    routed = lambda trace: _route(trace) in TIERS
    orchestrated = lambda trace: bool(_field(trace, "orchestrator_started"))
    attempted = lambda trace: bool(_field(trace, "provider_invocation_started", _field(trace, "provider_attempted", False)))
    sufficient = lambda trace: bool(_field(trace, "sufficient_evidence", _field(trace, "evidence_state") == "sufficient"))
    configured_explicit = [
        (label, trace) for label, trace in explicit_rows
        if trace is not None and bool(trace.get("provider_configured"))
    ]
    configured_factual = [
        (label, trace) for label, trace in d_factual_rows
        if trace is not None and bool(trace.get("provider_configured"))
    ]
    rates = {
        "route_coverage": _rate(d_factual_rows, routed),
        "orchestrator_start_rate": _rate(d_factual_rows, orchestrated),
        "provider_attempt_rate": _rate(d_factual_rows, attempted),
        "provider_attempt_rate_configured": _rate(configured_factual, attempted),
        "sufficient_evidence_rate": _rate(d_factual_rows, sufficient),
        "explicit_search_route_rate": _rate(explicit_rows, routed),
        "explicit_search_orchestrator_start_rate": _rate(explicit_rows, orchestrated),
        "explicit_search_provider_attempt_rate_configured": _rate(configured_explicit, attempted),
    }

    execution_failures = {"provider_not_configured": 0, "provider_unavailable": 0}
    joined_traces = list(trace_by_id.values())
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

    budgets = budget_violations(joined_traces)
    invariants = deterministic_invariant_violations(joined_traces)
    latencies = latency_percentiles(joined_traces)
    per_tier_retrieval: dict[str, dict[str, Any]] = {}
    for tier in TIERS:
        values = [
            trace["retrieval_pipeline_latency_ms"] for trace in joined_traces
            if trace.get("route") == tier and _is_nonnegative_number(trace.get("retrieval_pipeline_latency_ms"))
        ]
        per_tier_retrieval[tier] = {
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
    retrieval_p95_limits = {"light": 6_000, "standard": 15_000, "deep": 30_000}
    for tier, limit in retrieval_p95_limits.items():
        p95 = per_tier_retrieval[tier]["p95"]
        if p95 is not None and p95 > limit:
            failures.append(f"{tier} retrieval P95 above {limit} ms")
    if any(budgets.values()):
        failures.append("budget or hard-timeout violations")
    if any(invariants.values()):
        failures.append("deterministic citation/failure invariant violations")

    return {
        "mode": "traces",
        "certifying": not failures,
        "joined_case_count": len(trace_by_id),
        "errors": errors,
        "exclusions": exclusions,
        "rates": rates,
        "execution_failures": execution_failures,
        "budget_violations": budgets,
        "deterministic_invariant_violations": invariants,
        "initial_batch_count": initial_batch_round_count(joined_traces),
        "repair_round_count": repair_round_count(joined_traces),
        "latencies_ms": latencies,
        "per_tier_retrieval_pipeline_latency_ms": per_tier_retrieval,
        "failures": failures,
    }


def traces(traces_path: Path, labels_path: Path) -> int:
    trace_rows, trace_errors = _load_jsonl_checked(traces_path)
    labels, label_errors = _load_jsonl_checked(labels_path)
    load_errors = [*trace_errors, *label_errors]
    if load_errors:
        report = {
            "mode": "traces", "certifying": False, "errors": load_errors,
            "failures": ["input JSONL integrity errors"],
        }
    else:
        report = evaluate_traces(trace_rows, labels)
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
    sub.add_parser("offline")
    traces_parser = sub.add_parser("traces")
    traces_parser.add_argument("--traces", required=True, type=Path)
    traces_parser.add_argument("--labels", required=True, type=Path)
    online_parser = sub.add_parser("online")
    online_parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    if args.command == "integrity":
        return integrity()
    if args.command == "offline":
        return offline()
    if args.command == "traces":
        return traces(args.traces, args.labels)
    if args.command == "online":
        return online(args.limit)
    return 2


if __name__ == "__main__":
    sys.exit(main())
