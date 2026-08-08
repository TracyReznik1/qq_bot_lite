"""Evaluation harness tests: dataset integrity and metric correctness."""

from __future__ import annotations

import importlib
import contextlib
import io
import json
import tempfile
import unittest
from unittest import mock
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


def _prediction(case_id, component="router", **overrides):
    prediction = {
        "case_id": case_id,
        "component": component,
        "model": "independent-test-model",
        "model_version": "1.0",
        "prompt_schema_version": "retrieval-v1",
        "run_timestamp": "2026-07-29T00:00:00Z",
    }
    prediction.update(overrides)
    return prediction


def _recording(case_id, fixture_id=None, **overrides):
    recording = {
        "fixture_id": fixture_id or case_id,
        "case_id": case_id,
        "provider": "synthetic",
        "query_text": "fixture query",
        "title": "Fixture title",
        "url": f"https://fixtures.example/{case_id}",
        "excerpt": "fixture excerpt",
        "expected_fetch_status": "success",
    }
    recording.update(overrides)
    return recording


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
            _trace_with(
                route="standard",
                initial_query_count=5,
                candidate_url_count=9,
                content_read_count=6,
                retrieval_round_count=3,
                adaptive_repair_round_started=True,
                retrieval_pipeline_latency_ms=20_001,
            ),
            {
                "route": "light",
                "initial_query_count": 1,
                "candidate_url_count": 5,
                "content_read_count": 2,
                "retrieval_round_count": 1,
                "semantic_query_count": 2,
                "repair_query_count": 1,
                "retrieval_pipeline_latency_ms": 8_001,
            },
        ]
        violations = tool.budget_violations(traces)
        self.assertEqual(violations["initial_query_count"], 1)
        self.assertEqual(violations["retrieval_round_count"], 1)
        self.assertEqual(violations["candidate_url_count"], 1)
        self.assertEqual(violations["content_read_count"], 1)
        self.assertEqual(violations["semantic_query_count"], 1)
        self.assertEqual(violations["repair_query_count"], 1)
        self.assertEqual(violations["hard_timeout"], 2)

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


class IntegrityContractTests(unittest.TestCase):
    def test_one_case_schema_accepts_external_fixture_and_separate_prediction(self):
        tool = evaluate_tool()
        case = _make_case("fact-1", "stable_fact", fixture_id="fixture-1")
        errors = tool.validate_integrity(
            [case],
            [_recording("fact-1", "fixture-1")],
            [_prediction("fact-1", predicted_tier="light")],
            expected_case_count=1,
            category_quotas={"stable_fact": 1},
        )
        self.assertEqual([], errors)

    def test_schema_rejects_enums_dates_fixture_references_and_prediction_label_leaks(self):
        tool = evaluate_tool()
        case = _make_case(
            "fact-1",
            "stable_fact",
            minimum_tier="turbo",
            reviewed_at="29/07/2026",
            fixture_id="missing-fixture",
        )
        prediction = _prediction(
            "fact-1",
            predicted_tier="light",
            reviewed_by="prediction-must-not-contain-labels",
        )
        errors = tool.validate_integrity(
            [case], [], [prediction], expected_case_count=1,
            category_quotas={"stable_fact": 1},
        )
        joined = "\n".join(errors)
        self.assertIn("minimum_tier", joined)
        self.assertIn("reviewed_at", joined)
        self.assertIn("missing fixture_id", joined)
        self.assertIn("human-label field", joined)

    def test_duplicate_and_missing_case_ids_are_rejected_across_artifacts(self):
        tool = evaluate_tool()
        case = _make_case("fact-1", "stable_fact")
        errors = tool.validate_integrity(
            [case, dict(case)],
            [_recording("fact-1"), _recording("unknown-case")],
            [
                _prediction("fact-1", predicted_tier="light"),
                _prediction("fact-1", predicted_tier="standard"),
                _prediction("", predicted_tier="light"),
            ],
            expected_case_count=2,
            category_quotas={"stable_fact": 2},
        )
        joined = "\n".join(errors)
        self.assertIn("duplicate case_id", joined)
        self.assertIn("unknown case_id", joined)
        self.assertIn("missing case_id", joined)
        self.assertIn("duplicate prediction", joined)

    def test_repository_integrity_failures_are_only_the_real_owner_review_gate(self):
        tool = evaluate_tool()
        errors = tool.collect_integrity_errors()
        self.assertEqual(140, len(errors))
        self.assertTrue(all("owner review" in error for error in errors))

    def test_integrity_rejects_case_without_router_prediction(self):
        tool = evaluate_tool()
        errors = tool.validate_integrity(
            [_make_case("fact-1", "stable_fact")],
            [_recording("fact-1", "stable-fact-001")],
            [],
            expected_case_count=1,
            category_quotas={"stable_fact": 1},
        )
        self.assertIn("missing router prediction for case_id fact-1", errors)

    def test_quality_predictions_may_use_one_atomic_row_per_label_id(self):
        tool = evaluate_tool()
        case = _make_case(
            "fact-1", "stable_fact",
            semantic_labels=[
                {"label_id": "claim-1", "component": "claim_discovery", "expected": "present"},
                {"label_id": "claim-2", "component": "claim_discovery", "expected": "absent"},
            ],
        )
        predictions = [
            _prediction("fact-1", predicted_tier="light"),
            _prediction("fact-1", "claim_discovery", label_id="claim-1", predicted="present"),
            _prediction("fact-1", "claim_discovery", label_id="claim-2", predicted="absent"),
        ]
        errors = tool.validate_integrity(
            [case], [_recording("fact-1", "stable-fact-001")], predictions,
            expected_case_count=1, category_quotas={"stable_fact": 1},
        )
        self.assertEqual([], errors)


class ModelQualityContractTests(unittest.TestCase):
    def test_quality_metrics_join_label_ids_and_compute_three_components(self):
        tool = evaluate_tool()
        cases = [
            _make_case(
                "dynamic-1",
                "dynamic_fact",
                minimum_tier="deep",
                potential_harm="high",
                semantic_labels=[
                    {"label_id": "claim-a", "component": "claim_discovery", "expected": "present"},
                    {"label_id": "claim-b", "component": "claim_discovery", "expected": "absent"},
                    {"label_id": "support-a", "component": "semantic_support", "expected": "supported"},
                    {"label_id": "support-b", "component": "semantic_support", "expected": "unsupported"},
                    {"label_id": "candidate-a", "component": "relevance", "expected": "relevant"},
                    {"label_id": "candidate-b", "component": "relevance", "expected": "irrelevant"},
                ],
            )
        ]
        predictions = [
            _prediction(
                "dynamic-1", "claim_discovery",
                predictions=[
                    {"label_id": "claim-a", "predicted": "present"},
                    {"label_id": "claim-b", "predicted": "present"},
                ],
            ),
            _prediction(
                "dynamic-1", "semantic_support",
                predictions=[
                    {"label_id": "support-a", "predicted": "supported"},
                    {"label_id": "support-b", "predicted": "unsupported"},
                ],
            ),
            _prediction(
                "dynamic-1", "relevance",
                predictions=[
                    {"label_id": "candidate-a", "predicted": "relevant"},
                    {"label_id": "candidate-b", "predicted": "irrelevant"},
                ],
            ),
        ]
        quality = tool.quality_metrics(cases, predictions)
        self.assertEqual(2, quality["claim_discovery"]["overall"]["sample_count"])
        self.assertAlmostEqual(0.5, quality["claim_discovery"]["overall"]["precision"])
        self.assertAlmostEqual(2 / 3, quality["claim_discovery"]["overall"]["f1"])
        self.assertAlmostEqual(1.0, quality["semantic_support"]["overall"]["macro_f1"])
        self.assertAlmostEqual(1.0, quality["relevance"]["overall"]["f1"])
        self.assertEqual(2, quality["claim_discovery"]["dynamic"]["sample_count"])
        self.assertEqual(2, quality["claim_discovery"]["high_consequence"]["sample_count"])

    def test_zero_sample_groups_are_non_evaluable_not_perfect(self):
        tool = evaluate_tool()
        quality = tool.quality_metrics([_make_case("fact-1", "stable_fact")], [])
        for component in ("claim_discovery", "semantic_support", "relevance"):
            for group in ("overall", "dynamic", "high_consequence"):
                metric = quality[component][group]
                self.assertFalse(metric["evaluable"])
                self.assertEqual(0, metric["sample_count"])
                self.assertIsNone(metric.get("f1", metric.get("macro_f1")))

    def test_offline_missing_predictions_and_fixture_baseline_are_non_certifying(self):
        tool = evaluate_tool()
        cases = [_make_case("fact-1", "stable_fact")]
        missing = tool.evaluate_offline(cases, [])
        self.assertFalse(missing["certifying"])
        self.assertTrue(any("missing router prediction" in item for item in missing["failures"]))
        self.assertTrue(any("zero samples" in item for item in missing["failures"]))

        baseline = tool.evaluate_offline(
            cases,
            [_prediction("fact-1", model="fixture-baseline", predicted_tier="light")],
        )
        self.assertFalse(baseline["certifying"])
        self.assertIn("fixture baseline", "\n".join(baseline["failures"]).lower())


def _trace_row(case_id, route="light", **overrides):
    searched = route != "skip"
    row = {
        "case_id": case_id,
        "request_id": f"request-{case_id}",
        "request_source": "chat",
        "route": route,
        "skip_reason": None,
        "orchestrator_started": searched,
        "initial_query_count": 1 if searched else 0,
        "initial_round_started": searched,
        "adaptive_repair_round_started": False,
        "retrieval_round_count": 1 if searched else 0,
        "provider_configured": searched,
        "provider_attempts": ([{
            "provider": "tavily", "status": "success", "count": 1,
            "latency_ms": 5, "query_id": "q-1", "configured": True,
            "available": True, "invocation_started": True,
        }] if searched else []),
        "provider_invocation_started": searched,
        "provider_failures": [],
        "candidate_url_count": 1 if searched else 0,
        "content_read_count": 1 if searched else 0,
        "semantic_query_count": 1 if searched else 0,
        "repair_query_count": 0,
        "citable_evidence_count": 1 if searched else 0,
        "evidence_state": "sufficient" if searched else None,
        "claim_count": 1 if searched else 0,
        "supported_claim_count": 1 if searched else 0,
        "citation_count": 1 if searched else 0,
        "knowledge_fallback_used": False,
        "degradation_reason": None,
    }
    for field in (
        "route_latency_ms", "query_planning_latency_ms",
        "initial_provider_search_latency_ms", "provider_search_total_latency_ms",
        "initial_content_read_latency_ms", "content_read_total_latency_ms",
        "initial_evidence_assembly_latency_ms", "evidence_assembly_total_latency_ms",
        "gap_analysis_latency_ms", "adaptive_repair_latency_ms",
        "answer_generation_latency_ms", "structural_validation_latency_ms",
        "semantic_validation_latency_ms", "qq_render_latency_ms",
        "retrieval_pipeline_latency_ms", "total_response_latency_ms",
    ):
        row[field] = 10 if searched else 0
    row.update(overrides)
    return row


class TraceAcceptanceContractTests(unittest.TestCase):
    def test_external_case_id_labels_control_d_factual_not_embedded_trace_label(self):
        tool = evaluate_tool()
        labels = [_make_case("fact-1", "stable_fact")]
        trace = _trace_row("fact-1", route="skip", d_factual=False)
        report = tool.evaluate_traces([trace], labels)
        self.assertEqual(1, report["rates"]["route_coverage"]["denominator"])
        self.assertEqual(0, report["rates"]["route_coverage"]["numerator"])
        self.assertEqual(0.0, report["rates"]["route_coverage"]["rate"])

    def test_explicit_no_web_is_excluded_even_if_external_d_factual_flag_is_wrong(self):
        tool = evaluate_tool()
        label = _make_case(
            "no-web-1", "explicit_search", allow_skip=True,
            skip_reason="user_forbid_web", minimum_tier=None,
            external_fact_required=True, d_factual=True,
        )
        report = tool.evaluate_traces(
            [_trace_row("no-web-1", route="skip", skip_reason="user_forbid_web")],
            [label],
        )
        self.assertEqual(0, report["rates"]["route_coverage"]["denominator"])
        self.assertEqual(1, report["exclusions"]["explicit_no_web"])

    def test_trace_rates_track_exclusions_and_provider_failures_separately(self):
        tool = evaluate_tool()
        labels = [
            _make_case("fact-1", "stable_fact"),
            _make_case("explicit-1", "explicit_search", minimum_tier="standard"),
            _make_case(
                "no-web-1", "no_benefit", allow_skip=True,
                skip_reason="user_forbid_web", minimum_tier=None,
                external_fact_required=True, expected_outcome="skip",
            ),
            _make_case(
                "closed-1", "no_benefit", allow_skip=True,
                skip_reason="closed_context_only", minimum_tier=None,
                external_fact_required=False, expected_outcome="skip",
            ),
        ]
        traces = [
            _trace_row("fact-1", degradation_reason="provider_unavailable", provider_attempts=[], provider_invocation_started=False, citable_evidence_count=0, evidence_state="insufficient", claim_count=0, supported_claim_count=0, citation_count=0),
            _trace_row("explicit-1", route="standard"),
            _trace_row("no-web-1", route="skip", skip_reason="user_forbid_web"),
            _trace_row("closed-1", route="skip", skip_reason="closed_context_only"),
        ]
        report = tool.evaluate_traces(traces, labels)
        self.assertEqual({"explicit_no_web": 1, "legal_closed_context": 1}, report["exclusions"])
        self.assertEqual(2, report["rates"]["route_coverage"]["denominator"])
        self.assertEqual(1, report["rates"]["provider_attempt_rate"]["numerator"])
        self.assertEqual(1, report["execution_failures"]["provider_unavailable"])
        self.assertEqual(1, report["rates"]["explicit_search_orchestrator_start_rate"]["denominator"])

    def test_trace_join_rejects_unknown_duplicate_and_missing_case_ids(self):
        tool = evaluate_tool()
        labels = [_make_case("fact-1", "stable_fact"), _make_case("fact-2", "stable_fact")]
        report = tool.evaluate_traces(
            [_trace_row("fact-1"), _trace_row("fact-1"), _trace_row("unknown")],
            labels,
        )
        joined = "\n".join(report["errors"])
        self.assertIn("duplicate trace case_id", joined)
        self.assertIn("unknown trace case_id", joined)
        self.assertIn("missing trace for case_id fact-2", joined)

    def test_human_audit_labels_require_real_reviewer_and_iso_date(self):
        tool = evaluate_tool()
        unreviewed = _make_case(
            "fact-1", "stable_fact", label_status="unreviewed",
            reviewed_by="unreviewed", reviewed_at=None,
        )
        report = tool.evaluate_traces([_trace_row("fact-1")], [unreviewed])
        self.assertIn("owner review", "\n".join(report["errors"]))

    def test_trace_schema_and_deterministic_failure_invariants_are_counted(self):
        tool = evaluate_tool()
        malformed = _trace_row(
            "fact-1",
            route="warp",
            provider_attempts="not-a-list",
            route_latency_ms=-1,
        )
        report = tool.evaluate_traces([malformed], [_make_case("fact-1", "stable_fact")])
        self.assertTrue(report["errors"])

        violations = tool.deterministic_invariant_violations([
            _trace_row(
                "fact-2", evidence_state="insufficient", citable_evidence_count=0,
                claim_count=1, supported_claim_count=2, citation_count=1,
                knowledge_fallback_used=True,
            ),
            _trace_row(
                "skip-1", route="skip", provider_invocation_started=True,
                provider_attempts=[{"provider": "tavily"}], citation_count=1,
            ),
        ])
        self.assertEqual(2, violations["unsupported_claim_or_citation"])
        self.assertEqual(1, violations["supported_claim_count_exceeds_claim_count"])
        self.assertEqual(1, violations["knowledge_fallback_with_citation"])
        self.assertEqual(1, violations["skip_with_provider_attempt"])

    def test_all_stage_latencies_report_p50_p95_p99_and_hard_timeouts(self):
        tool = evaluate_tool()
        traces = []
        for index, value in enumerate((10, 20, 30, 40), 1):
            traces.append(_trace_row(
                f"fact-{index}",
                **{field: value for field in tool.LATENCY_FIELDS},
            ))
        summary = tool.latency_percentiles(traces)
        self.assertEqual(set(tool.LATENCY_FIELDS), set(summary))
        for field in tool.LATENCY_FIELDS:
            self.assertEqual({"sample_count": 4, "p50": 20, "p95": 40, "p99": 40}, summary[field])

    def test_configured_provider_rate_and_retrieval_p95_threshold_are_acceptance_gates(self):
        tool = evaluate_tool()
        trace = _trace_row(
            "fact-1", provider_configured=True, provider_invocation_started=False,
            provider_attempts=[], retrieval_pipeline_latency_ms=7_000,
        )
        report = tool.evaluate_traces([trace], [_make_case("fact-1", "stable_fact")])
        configured = report["rates"]["provider_attempt_rate_configured"]
        self.assertEqual({"numerator": 0, "denominator": 1, "rate": 0.0}, configured)
        failures = "\n".join(report["failures"])
        self.assertIn("configured provider attempt rate below 1.00", failures)
        self.assertIn("light retrieval P95 above 6000 ms", failures)


class CliNegativePathTests(unittest.TestCase):
    def test_unauthorised_online_mode_is_not_run_and_returns_nonzero(self):
        tool = evaluate_tool()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = tool.main(["online", "--limit", "1"])
        self.assertNotEqual(0, status)
        self.assertIn("not run", output.getvalue().lower())
        self.assertIn("not authorized", output.getvalue().lower())

    def test_traces_cli_malformed_json_is_controlled_nonzero_not_traceback(self):
        tool = evaluate_tool()
        with tempfile.TemporaryDirectory() as temp_dir:
            traces_path = Path(temp_dir) / "traces.jsonl"
            labels_path = Path(temp_dir) / "labels.jsonl"
            traces_path.write_text("{broken json\n", encoding="utf-8")
            labels_path.write_text(json.dumps(_make_case("fact-1", "stable_fact")), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = tool.traces(traces_path, labels_path)
        self.assertNotEqual(0, status)
        self.assertIn("invalid JSON", output.getvalue())

    def test_offline_cli_returns_nonzero_for_missing_predictions_and_zero_samples(self):
        tool = evaluate_tool()
        with tempfile.TemporaryDirectory() as temp_dir:
            cases_path = Path(temp_dir) / "cases.jsonl"
            predictions_path = Path(temp_dir) / "predictions.jsonl"
            cases_path.write_text(json.dumps(_make_case("fact-1", "stable_fact")), encoding="utf-8")
            predictions_path.write_text("", encoding="utf-8")
            output = io.StringIO()
            with (
                mock.patch.object(tool, "CASES_PATH", cases_path),
                mock.patch.object(tool, "MODEL_PREDICTIONS_PATH", predictions_path),
                contextlib.redirect_stdout(output),
            ):
                status = tool.offline()
        self.assertNotEqual(0, status)
        report = json.loads(output.getvalue())
        self.assertFalse(report["certifying"])
        self.assertIn("missing router prediction", "\n".join(report["failures"]))
        self.assertIn("zero samples", "\n".join(report["failures"]))


if __name__ == "__main__":
    unittest.main()
