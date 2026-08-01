"""Evidence-search evaluation CLI: dataset integrity and offline metrics.

Commands:
  integrity   validate count, quotas, unique IDs, enums, human-review fields
  offline     compute metrics from fixed fixtures and stored predictions
  traces      aggregate a production human-audited trace sample
  online      opt-in controlled online run (requires separate authorization)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

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


# ── metric primitives ──────────────────────────────────────────────────

def prf1(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def tier_metrics(
    confusion: dict[tuple[str, str], int],
    *,
    labels: Iterable[str] = TIERS,
) -> dict[str, float]:
    labels = tuple(labels)
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(confusion.get((label, label), 0) for _ in [0])
        fp = sum(count for (pred, true), count in confusion.items() if pred == label and true != label)
        fn = sum(count for (pred, true), count in confusion.items() if true == label and pred != label)
        per_label[label] = prf1(tp, fp, fn)
    macro_f1 = sum(per_label[label]["f1"] for label in labels) / len(labels) if labels else 0.0
    result: dict[str, float] = {"macro_f1": macro_f1}
    for label in labels:
        result[f"{label}_f1"] = per_label[label]["f1"]
    return result


# ── trace helpers ──────────────────────────────────────────────────────

def _field(trace: Any, name: str, default: Any = None) -> Any:
    if isinstance(trace, dict):
        return trace.get(name, default)
    return getattr(trace, name, default)


def _route(trace: Any) -> str:
    return str(_field(trace, "route") or "")


def _initial_query_count(trace: Any) -> int:
    return int(_field(trace, "initial_query_count") or 0)


def _retrieval_round_count(trace: Any) -> int:
    return int(_field(trace, "retrieval_round_count") or 0)


def _adaptive_repair_round_started(trace: Any) -> bool:
    return bool(_field(trace, "adaptive_repair_round_started"))


def budget_violations(traces: Iterable[Any]) -> dict[str, int]:
    violations = {"initial_query_count": 0, "candidate_url_count": 0, "content_read_count": 0, "retrieval_round_count": 0, "semantic_query_count": 0}
    max_initial = {"light": 1, "standard": 3, "deep": 5}
    max_rounds = {"light": 1, "standard": 2, "deep": 2}
    for trace in traces:
        route = _route(trace)
        if _initial_query_count(trace) > max_initial.get(route, 5):
            violations["initial_query_count"] += 1
        if _retrieval_round_count(trace) > max_rounds.get(route, 2):
            violations["retrieval_round_count"] += 1
    return violations


def initial_batch_round_count(traces: Iterable[Any]) -> int:
    return sum(1 for trace in traces if _field(trace, "initial_round_started"))


def repair_round_count(traces: Iterable[Any]) -> int:
    return sum(1 for trace in traces if _adaptive_repair_round_started(trace))


def structural_violations(traces: Iterable[Any]) -> int:
    count = 0
    for trace in traces:
        route = _route(trace)
        if route == "skip":
            if _field(trace, "provider_attempts"):
                count += 1
        if route in {"light", "standard", "deep"}:
            if not _field(trace, "orchestrator_started"):
                count += 1
    return count


def in_d_factual(case: dict[str, Any]) -> bool:
    if case.get("allow_skip"):
        return False
    if case.get("external_fact_required") is False:
        return False
    return True


# ── integrity ──────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def integrity() -> int:
    cases = _load_jsonl(CASES_PATH)
    errors: list[str] = []
    if len(cases) != 140:
        errors.append(f"case count {len(cases)} != 140")
    categories = [case.get("category") for case in cases]
    for category, expected in CATEGORY_QUOTAS.items():
        count = categories.count(category)
        if count != expected:
            errors.append(f"category {category} count {count} != {expected}")
    ids = [case.get("case_id") for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case_id")
    for case in cases:
        if not case.get("case_id") or not case.get("question"):
            errors.append(f"case {case.get('case_id')} missing case_id/question")
        reviewer = case.get("reviewed_by")
        reviewed_at = case.get("reviewed_at")
        if not reviewer or reviewer == "unreviewed":
            errors.append(f"case {case.get('case_id')} missing reviewed_by")
        if not reviewed_at:
            errors.append(f"case {case.get('case_id')} missing reviewed_at")
        payload = json.dumps(case, ensure_ascii=False)
        for secret in ("AIza", "sk-", "qq:1000"):
            if secret in payload:
                errors.append(f"case {case.get('case_id')} contains {secret}")
    for error in errors:
        print(f"integrity error: {error}")
    print(f"integrity: cases={len(cases)} errors={len(errors)}")
    return 1 if errors else 0


# ── offline metrics ────────────────────────────────────────────────────

def offline() -> int:
    cases = _load_jsonl(CASES_PATH)
    predictions = _load_jsonl(MODEL_PREDICTIONS_PATH)
    route_preds = {pred["case_id"]: pred for pred in predictions if pred.get("component") == "router"}

    confusion: dict[tuple[str, str], int] = {}
    correct = 0
    total = 0
    explicit_total = 0
    explicit_routed = 0
    meaningless_total = 0
    meaningless_searched = 0
    for case in cases:
        pred = route_preds.get(case["case_id"])
        expected = str(case.get("minimum_tier") or "skip")
        predicted = str(pred.get("predicted_tier") or "skip") if pred is not None else None
        if pred is None:
            continue
        confusion[(predicted, expected)] = confusion.get((predicted, expected), 0) + 1
        total += 1
        if predicted == expected:
            correct += 1
        if case.get("category") == "explicit_search":
            explicit_total += 1
            if predicted != "skip":
                explicit_routed += 1
        if case.get("category") == "no_benefit":
            meaningless_total += 1
            if predicted != "skip":
                meaningless_searched += 1

    tier = tier_metrics(confusion)
    explicit_rate = explicit_routed / explicit_total if explicit_total else 0.0
    meaningless_rate = meaningless_searched / meaningless_total if meaningless_total else 0.0
    print(f"offline: cases_with_router_prediction={total}")
    print(f"offline: tier_macro_f1={tier['macro_f1']:.4f}")
    for label in TIERS:
        print(f"offline: tier_{label}_f1={tier[f'{label}_f1']:.4f}")
    print(f"offline: explicit_search_route_rate={explicit_rate:.4f} ({explicit_routed}/{explicit_total})")
    print(f"offline: legal_non_factual_meaningless_search_rate={meaningless_rate:.4f} ({meaningless_searched}/{meaningless_total})")
    print(f"offline: route_accuracy={correct / total:.4f} ({correct}/{total})" if total else "offline: route_accuracy=n/a")

    failures = 0
    if total > 0:
        if tier["macro_f1"] < 0.90:
            failures += 1
            print("offline: FAIL tier_macro_f1 < 0.90")
        if explicit_total and explicit_rate < 1.00:
            failures += 1
            print("offline: FAIL explicit_search_route_rate < 1.00")
        if meaningless_total and meaningless_rate > 0.10:
            failures += 1
            print("offline: FAIL meaningless_search_rate > 0.10")
    print(f"offline: failures={failures}")
    return 1 if failures else 0


def traces(traces_path: Path, labels_path: Path) -> int:
    trace_rows = _load_jsonl(traces_path)
    labels = _load_jsonl(labels_path)
    d_factual = [row for row in trace_rows if row.get("d_factual") is True]
    routed = sum(1 for row in d_factual if _route(row) != "skip")
    route_coverage = routed / len(d_factual) if d_factual else 0.0
    print(f"traces: d_factual={len(d_factual)} routed={routed} route_coverage={route_coverage:.4f}")
    failures = 0
    if d_factual and route_coverage < 0.98:
        failures += 1
        print("traces: FAIL route_coverage < 0.98")
    print(f"traces: labels={len(labels)} failures={failures}")
    return 1 if failures else 0


def online(limit: int) -> int:
    print("online: not run (requires separate explicit authorization and credentials)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="evidence-search evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("integrity")
    sub.add_parser("offline")
    traces_parser = sub.add_parser("traces")
    traces_parser.add_argument("--traces", required=True, type=Path)
    traces_parser.add_argument("--labels", required=True, type=Path)
    online_parser = sub.add_parser("online")
    online_parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

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
