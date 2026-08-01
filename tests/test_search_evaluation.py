"""Evaluation harness tests: dataset integrity and metric correctness."""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

from src.search.models import (
    RequestSource,
    SearchTier,
    SearchTrace,
    ProviderAttempt,
    ProviderStatus,
)


def evaluate_tool():
    try:
        return importlib.import_module("tools.evaluate_search")
    except ModuleNotFoundError:
        raise AssertionError("tools.evaluate_search must exist") from None


def _make_case(case_id, category, **overrides):
    case = {
        "case_id": case_id,
        "category": category,
        "question": "什么是光合作用",
        "allow_skip": False,
        "skip_reason": None,
        "minimum_tier": "light",
        "external_fact_required": True,
        "actionability": "none",
        "potential_harm": "none",
        "expected_query_purposes": ["direct"],
        "expected_initial_query_min": 1,
        "expected_initial_query_max": 1,
        "expected_max_rounds": 1,
        "material_claim_spans": ["光合作用的定义"],
        "acceptable_source_relations": ["primary", "independent", "unknown"],
        "semantic_labels": [],
        "expected_outcome": "grounded_answer",
        "fixture_id": "stable-fact-001",
        "label_status": "reviewed",
        "reviewed_by": "project_owner",
        "reviewed_at": "2026-07-29",
    }
    case.update(overrides)
    return case


class IntegrityMetricTests(unittest.TestCase):
    """Small hand-calculable confusion matrices."""

    def test_tier_precision_recall_f1(self):
        tool = evaluate_tool()
        confusion = {
            ("light", "light"): 8,
            ("light", "standard"): 1,
            ("standard", "standard"): 7,
            ("standard", "light"): 2,
            ("standard", "deep"): 1,
            ("deep", "deep"): 6,
            ("deep", "standard"): 1,
        }
        metrics = tool.tier_metrics(confusion, labels=("light", "standard", "deep"))
        self.assertAlmostEqual(metrics["macro_f1"], 0.81203, places=4)
        self.assertAlmostEqual(metrics["light_f1"], 0.84211, places=4)
        self.assertAlmostEqual(metrics["standard_f1"], 0.73684, places=4)
        self.assertAlmostEqual(metrics["deep_f1"], 0.85714, places=4)

    def test_precision_recall_f1_formula(self):
        tool = evaluate_tool()
        # tp=5, fp=1, fn=2
        self.assertAlmostEqual(tool.prf1(5, 1, 2)["precision"], 5 / 6)
        self.assertAlmostEqual(tool.prf1(5, 1, 2)["recall"], 5 / 7)
        expected_f1 = 2 * (5 / 6) * (5 / 7) / ((5 / 6) + (5 / 7))
        self.assertAlmostEqual(tool.prf1(5, 1, 2)["f1"], expected_f1)

    def test_budget_violation_count(self):
        tool = evaluate_tool()
        traces = [
            _trace_with(route="standard", initial_query_count=3),
            _trace_with(route="standard", initial_query_count=5),
            _trace_with(route="deep", retrieval_round_count=3),
        ]
        violations = tool.budget_violations(traces)
        self.assertEqual(violations["initial_query_count"], 1)
        self.assertEqual(violations["retrieval_round_count"], 1)

    def test_round_accounting_initial_vs_repair(self):
        tool = evaluate_tool()
        traces = [
            _trace_with(route="standard", initial_query_count=3, retrieval_round_count=1, adaptive_repair_round_started=False),
            _trace_with(route="standard", initial_query_count=3, retrieval_round_count=2, adaptive_repair_round_started=True),
            _trace_with(route="deep", initial_query_count=5, retrieval_round_count=2, adaptive_repair_round_started=True),
        ]
        self.assertEqual(tool.initial_batch_round_count(traces), 3)
        self.assertEqual(tool.repair_round_count(traces), 2)

    def test_structural_invariant_violations_are_zero_when_clean(self):
        tool = evaluate_tool()
        traces = [_trace_with(route="standard", retrieval_round_count=2)]
        self.assertEqual(tool.structural_violations(traces), 0)

    def test_d_factual_rate_denominator_excludes_skip(self):
        tool = evaluate_tool()
        cases = [
            _make_case("skip-1", "no_benefit", allow_skip=True, skip_reason="social_or_emotional", minimum_tier=None, external_fact_required=False),
            _make_case("fact-1", "stable_fact", minimum_tier="light"),
        ]
        d_factual = [case for case in cases if tool.in_d_factual(case)]
        self.assertEqual(len(d_factual), 1)
        self.assertEqual(d_factual[0]["case_id"], "fact-1")


def _trace_with(route="light", **overrides):
    trace = SearchTrace("req-1", RequestSource.CHAT, SearchTier(route))
    trace.orchestrator_started = route != "skip"
    trace.initial_round_started = route != "skip"
    trace.retrieval_round_count = 1 if route != "skip" else 0
    for key, value in overrides.items():
        setattr(trace, key, value)
    return trace


class DatasetQuotaTests(unittest.TestCase):
    """The exact 140-row quota table from the plan."""

    def test_dataset_exists_with_exact_quota(self):
        root = Path(__file__).resolve().parents[1]
        cases_path = root / "eval" / "search" / "cases.jsonl"
        if not cases_path.exists():
            self.skipTest("cases.jsonl not authored yet")
        rows = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 140)
        categories = [row["category"] for row in rows]
        quota = {
            "no_benefit": 20,
            "stable_fact": 20,
            "explanation_comparison": 25,
            "dynamic_fact": 20,
            "regulated_controversy": 15,
            "explicit_search": 10,
            "ambiguous_mixed": 10,
            "failure_partial_conflict": 20,
        }
        for category, expected in quota.items():
            with self.subTest(category=category):
                self.assertEqual(categories.count(category), expected, f"category {category} count")

    def test_ids_unique_and_no_secrets(self):
        root = Path(__file__).resolve().parents[1]
        cases_path = root / "eval" / "search" / "cases.jsonl"
        if not cases_path.exists():
            self.skipTest("cases.jsonl not authored yet")
        rows = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [row["case_id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        for row in rows:
            self.assertNotIn("AIza", str(row))
            self.assertNotIn("sk-", str(row))
            self.assertNotIn("qq:1000", str(row))


if __name__ == "__main__":
    unittest.main()
