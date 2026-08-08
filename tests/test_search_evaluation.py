"""Evaluation harness tests: dataset integrity and metric correctness."""

from __future__ import annotations

import importlib
import contextlib
import hashlib
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


def _artifact_sha(rows):
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_manifest(cases, predictions, **overrides):
    manifest = {
        "schema_version": "search-eval-run-v1",
        "run_id": "independent-run-001",
        "provenance": "independent_model_run",
        "data_source": "reviewed_cases",
        "fixture_derived": False,
        "case_set_sha256": _artifact_sha(cases),
        "predictions_sha256": _artifact_sha(predictions),
        "run_timestamp": "2026-07-29T00:00:00Z",
    }
    manifest.update(overrides)
    return manifest


def _sample_manifest(traces, audits, **overrides):
    manifest = {
        "schema_version": "search-trace-sample-v1",
        "sample_id": "controlled-production-001",
        "provenance": "controlled_production",
        "fixture_derived": False,
        "collected_at": "2026-07-29T00:00:00Z",
        "traces_sha256": _artifact_sha(traces),
        "audits_sha256": _artifact_sha(audits),
    }
    manifest.update(overrides)
    return manifest


def _raw_trace_row(case_id, route="light", **overrides):
    searched = route != "skip"
    request_id = f"request-{case_id}"
    trace = {
        "request_id": request_id,
        "request_source": "chat",
        "route": route,
        "skip_reason": None,
        "trigger_codes": ["factual_default"] if searched else [],
        "factuality": "factual" if searched else "non_factual",
        "external_fact_required": searched,
        "program_minimum_tier": route if searched else None,
        "final_tier": route,
        "orchestrator_started": searched,
        "initial_query_count": 1 if searched else 0,
        "initial_round_started": searched,
        "adaptive_repair_round_started": False,
        "adaptive_repair_query": None,
        "initial_query_redaction_codes": [],
        "adaptive_repair_redaction_codes": [],
        "retrieval_round_count": 1 if searched else 0,
        "executed_queries": ([{"query_id": "q-1", "purpose": "direct"}] if searched else []),
        "provider_configured": searched,
        "provider_attempts": ([{
            "provider": "tavily", "status": "success", "count": 1,
            "latency_ms": 5, "query_id": "q-1", "configured": True,
            "available": True, "invocation_started": True,
        }] if searched else []),
        "provider_invocation_started": searched,
        "provider_failures": [],
        "candidate_url_count": 1 if searched else 0,
        "citable_evidence_count": 1 if searched else 0,
        "evidence_state": "sufficient" if searched else None,
        "repair_used": False,
        "claim_count": 1 if searched else 0,
        "supported_claim_count": 1 if searched else 0,
        "citation_count": 1 if searched else 0,
        "knowledge_fallback_used": False,
        "degradation_reason": None,
        "content_read_count": 1 if searched else 0,
        "provider_attempted": searched,
        "sufficient_evidence": searched,
        "semantic_query_count": 1 if searched else 0,
        "repair_query_count": 0,
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
        trace[field] = 10 if searched else (10 if field in {"route_latency_ms", "total_response_latency_ms"} else 0)
    trace.update(overrides)
    return trace


def _audit_row(case_id, *, route="light", **overrides):
    searched = route != "skip"
    url = f"https://fixtures.example/{case_id}"
    audit = {
        "case_id": case_id,
        "request_id": f"request-{case_id}",
        "category": "stable_fact" if searched else "no_benefit",
        "allow_skip": not searched,
        "skip_reason": None if searched else "pure_math",
        "external_fact_required": searched,
        "explicit_search": False,
        "dynamic": False,
        "high_consequence": False,
        "minimum_tier": route if searched else None,
        "acceptable_final_tiers": [route],
        "label_status": "reviewed",
        "reviewed_by": "owner-001",
        "reviewed_at": "2026-07-29",
        "claims": ([{
            "claim_id": "C1", "material": True, "retained": True,
            "support_label": "supported", "evidence_ids": ["E1"],
            "topic_ids": ["topic-1"],
        }] if searched else []),
        "evidence": ([{
            "evidence_id": "E1", "final_url": url, "relevance": "direct",
            "citable": True,
        }] if searched else []),
        "used_evidence_ids": (["E1"] if searched else []),
        "shown_source_urls": ([url] if searched else []),
        "missing_claim_topics": [],
        "conflict_groups": [],
        "rendered_disclosures": [],
        "stages_started": ([
            "route", "query_planning", "initial_provider_search",
            "provider_search_total", "initial_content_read", "content_read_total",
            "initial_evidence_assembly", "evidence_assembly_total", "gap_analysis",
            "answer_generation", "structural_validation", "semantic_validation",
            "qq_render", "retrieval_pipeline", "total_response",
        ] if searched else ["route", "qq_render", "total_response"]),
    }
    audit.update(overrides)
    return audit


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
        self.assertEqual(142, len(errors))
        self.assertEqual(140, sum("owner review" in error for error in errors))
        self.assertEqual(2, sum("invalid potential_harm" in error for error in errors))

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
                dynamic=True,
                high_consequence=True,
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

        predictions = [_prediction("fact-1", model="fixture-baseline", predicted_tier="light")]
        baseline = tool.evaluate_offline(
            cases, predictions,
            run_manifest=_run_manifest(
                cases, predictions, provenance="fixture_baseline",
                data_source="synthetic_provider_fixtures", fixture_derived=True,
            ),
        )
        self.assertFalse(baseline["certifying"])
        self.assertIn("fixture baseline", "\n".join(baseline["failures"]).lower())


def _trace_row(case_id, route="light", **overrides):
    return _raw_trace_row(case_id, route=route, **overrides)


class TraceAcceptanceContractTests(unittest.TestCase):
    def test_external_case_id_labels_control_d_factual_not_embedded_trace_label(self):
        tool = evaluate_tool()
        audits = [_audit_row("fact-1")]
        trace = _trace_row("fact-1", route="skip")
        report = tool.evaluate_traces(
            [trace], audits, sample_manifest=_sample_manifest([trace], audits),
        )
        self.assertEqual(1, report["rates"]["route_coverage"]["denominator"])
        self.assertEqual(0, report["rates"]["route_coverage"]["numerator"])
        self.assertEqual(0.0, report["rates"]["route_coverage"]["rate"])

    def test_explicit_no_web_is_excluded_even_if_external_d_factual_flag_is_wrong(self):
        tool = evaluate_tool()
        audit = _audit_row(
            "no-web-1", route="skip", category="explicit_search",
            skip_reason="user_forbid_web", external_fact_required=True,
            explicit_search=True,
        )
        trace = _trace_row(
            "no-web-1", route="skip", skip_reason="user_forbid_web",
            degradation_reason="user_forbid_web",
        )
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        self.assertEqual(0, report["rates"]["route_coverage"]["denominator"])
        self.assertEqual(1, report["exclusions"]["explicit_no_web"])

    def test_trace_rates_track_exclusions_and_provider_failures_separately(self):
        tool = evaluate_tool()
        audits = [
            _audit_row(
                "fact-1", claims=[], evidence=[], used_evidence_ids=[],
                shown_source_urls=[], rendered_disclosures=["provider_unavailable"],
            ),
            _audit_row("explicit-1", route="standard", category="explicit_search", explicit_search=True),
            _audit_row(
                "no-web-1", route="skip", skip_reason="user_forbid_web",
                external_fact_required=True,
            ),
            _audit_row("closed-1", route="skip", skip_reason="closed_context_only"),
        ]
        traces = [
            _trace_row(
                "fact-1", degradation_reason="provider_unavailable",
                provider_configured=False, provider_attempts=[],
                provider_invocation_started=False, provider_attempted=False,
                citable_evidence_count=0, evidence_state="insufficient",
                sufficient_evidence=False, claim_count=0,
                supported_claim_count=0, citation_count=0,
            ),
            _trace_row("explicit-1", route="standard"),
            _trace_row("no-web-1", route="skip", skip_reason="user_forbid_web", degradation_reason="user_forbid_web"),
            _trace_row("closed-1", route="skip", skip_reason="closed_context_only"),
        ]
        report = tool.evaluate_traces(
            traces, audits, sample_manifest=_sample_manifest(traces, audits),
        )
        self.assertEqual({"explicit_no_web": 1, "legal_closed_context": 1}, report["exclusions"])
        self.assertEqual(2, report["rates"]["route_coverage"]["denominator"])
        self.assertEqual(1, report["rates"]["provider_attempt_rate"]["numerator"])
        self.assertEqual(1, report["execution_failures"]["provider_unavailable"])
        self.assertEqual(1, report["rates"]["explicit_search_orchestrator_start_rate"]["denominator"])

    def test_trace_join_rejects_unknown_duplicate_and_missing_case_ids(self):
        tool = evaluate_tool()
        audits = [_audit_row("fact-1"), _audit_row("fact-2")]
        traces = [_trace_row("fact-1"), _trace_row("fact-1"), _trace_row("unknown")]
        report = tool.evaluate_traces(
            traces, audits, sample_manifest=_sample_manifest(traces, audits),
        )
        joined = "\n".join(report["errors"])
        self.assertIn("duplicate trace request_id", joined)
        self.assertIn("unknown trace request_id", joined)
        self.assertIn("missing trace for case_id fact-2", joined)

    def test_human_audit_labels_require_real_reviewer_and_iso_date(self):
        tool = evaluate_tool()
        unreviewed = _audit_row(
            "fact-1", label_status="unreviewed",
            reviewed_by="unreviewed", reviewed_at=None,
        )
        trace = _trace_row("fact-1")
        report = tool.evaluate_traces(
            [trace], [unreviewed], sample_manifest=_sample_manifest([trace], [unreviewed]),
        )
        self.assertIn("owner review", "\n".join(report["errors"]))

    def test_trace_schema_and_deterministic_failure_invariants_are_counted(self):
        tool = evaluate_tool()
        malformed = _trace_row(
            "fact-1",
            route="warp",
            provider_attempts="not-a-list",
            route_latency_ms=-1,
        )
        audit = _audit_row("fact-1")
        report = tool.evaluate_traces(
            [malformed], [audit], sample_manifest=_sample_manifest([malformed], [audit]),
        )
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
            self.assertEqual(
                {"evaluable": True, "sample_count": 4, "p50": 20, "p95": 40, "p99": 40},
                summary[field],
            )

    def test_configured_provider_rate_and_retrieval_p95_threshold_are_acceptance_gates(self):
        tool = evaluate_tool()
        trace = _trace_row(
            "fact-1", provider_configured=True, provider_invocation_started=False,
            provider_attempted=False, provider_attempts=[], retrieval_pipeline_latency_ms=7_000,
        )
        audit = _audit_row("fact-1")
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
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

    def test_traces_cli_requires_a_bound_controlled_sample_manifest(self):
        tool = evaluate_tool()
        trace = _raw_trace_row("manifest-required")
        audit = _audit_row("manifest-required")
        with tempfile.TemporaryDirectory() as temp_dir:
            traces_path = Path(temp_dir) / "traces.jsonl"
            labels_path = Path(temp_dir) / "audit.jsonl"
            traces_path.write_text(json.dumps(trace), encoding="utf-8")
            labels_path.write_text(json.dumps(audit), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = tool.traces(traces_path, labels_path)
        self.assertNotEqual(0, status)
        report = json.loads(output.getvalue())
        self.assertFalse(report["certifying"])
        self.assertIn("controlled sample manifest is required", "\n".join(report["errors"]))

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


class StrictReviewRegressionTests(unittest.TestCase):
    def test_i1_component_schemas_and_recursive_leakage_cover_planner_rows(self):
        tool = evaluate_tool()
        case = _make_case("planner", "stable_fact")
        recording = _recording("planner", case["fixture_id"])
        predictions = [
            _prediction("planner", predicted_tier="light"),
            _prediction(
                "planner", "planner",
                predicted_query_purposes=[{"expected": "copied-label"}],
            ),
        ]
        errors = tool.validate_integrity(
            [case], [recording], predictions, expected_case_count=1,
            category_quotas={"stable_fact": 1},
        )
        joined = "\n".join(errors)
        self.assertIn("missing fields", joined)
        self.assertIn("recursive human-label field", joined)

    def test_c3_manifest_provenance_source_and_run_timestamp_are_bound(self):
        tool = evaluate_tool()
        cases = [_make_case("manifest", "stable_fact")]
        predictions = [_prediction("manifest", predicted_tier="light")]
        manifest = _run_manifest(
            cases, predictions, provenance="independent_model_run",
            data_source="synthetic_provider_fixtures",
            run_timestamp="2026-07-30T00:00:00Z",
        )
        report = tool.evaluate_offline(cases, predictions, run_manifest=manifest)
        failures = "\n".join(report["failures"])
        self.assertIn("independent_model_run requires reviewed_cases", failures)
        self.assertIn("run_timestamp does not match", failures)

    def test_c3_offline_scoring_revalidates_closed_case_and_prediction_schemas(self):
        tool = evaluate_tool()
        cases = [_make_case("offline-schema", "stable_fact")]
        predictions = [_prediction(
            "offline-schema", predicted_tier="light", reviewed_by="copied-label",
        )]
        report = tool.evaluate_offline(
            cases, predictions, run_manifest=_run_manifest(cases, predictions),
        )
        self.assertFalse(report["certifying"])
        errors = "\n".join(report["errors"])
        self.assertIn("unexpected fields", errors)
        self.assertIn("human-label field", errors)

    def test_c1_attempt_queries_and_skip_execution_counters_are_cross_checked(self):
        tool = evaluate_tool()
        trace = _raw_trace_row("cross")
        trace["provider_attempts"][0]["query_id"] = "not-executed"
        audit = _audit_row("cross")
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        self.assertIn("query_id is not an executed query", "\n".join(report["errors"]))

        skipped = _raw_trace_row("skip-cross", route="skip", candidate_url_count=1)
        skip_audit = _audit_row("skip-cross", route="skip")
        report = tool.evaluate_traces(
            [skipped], [skip_audit],
            sample_manifest=_sample_manifest([skipped], [skip_audit]),
        )
        self.assertIn("skip route has nonzero execution counters", "\n".join(report["errors"]))

    def test_c2_retained_unsupported_claim_is_a_deterministic_violation(self):
        tool = evaluate_tool()
        trace = _raw_trace_row("unsupported", supported_claim_count=0)
        audit = _audit_row("unsupported")
        audit["claims"][0]["support_label"] = "unsupported"
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        self.assertEqual(
            1,
            report["deterministic_invariant_violations"]["retained_unsupported_claim"],
        )

    def test_c2_provider_outcomes_must_support_evidence_and_failure_codes(self):
        tool = evaluate_tool()
        trace = _raw_trace_row("provider-outcome")
        trace["provider_attempts"][0]["status"] = "timeout"
        trace["provider_failures"] = []
        audit = _audit_row("provider-outcome")
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        violations = report["deterministic_invariant_violations"]
        self.assertEqual(1, violations["evidence_without_successful_provider"])
        self.assertEqual(1, violations["provider_failure_mismatch"])

    def test_c1_external_audit_schema_rejects_unknown_enums_and_wrong_types(self):
        tool = evaluate_tool()
        trace = _raw_trace_row("bad-label")
        audit = _audit_row(
            "bad-label", allow_skip="yes", skip_reason="bogus",
            external_fact_required="yes", unexpected="field",
        )
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        self.assertFalse(report["certifying"])
        errors = "\n".join(report["errors"])
        self.assertIn("allow_skip", errors)
        self.assertIn("skip_reason", errors)
        self.assertIn("external_fact_required", errors)
        self.assertIn("unexpected", errors)

    def test_c1_trace_and_provider_attempt_schemas_are_exact_and_recursive(self):
        tool = evaluate_tool()
        trace = _raw_trace_row("bad-trace")
        trace.pop("request_source")
        trace["provider_attempts"] = [{"status": "success", "latency_ms": 1}]
        audit = _audit_row("bad-trace")
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        self.assertFalse(report["certifying"])
        errors = "\n".join(report["errors"])
        self.assertIn("request_source", errors)
        self.assertIn("provider attempt", errors)
        self.assertIn("missing fields", errors)

    def test_c1_raw_trace_joins_to_audited_case_via_request_id(self):
        tool = evaluate_tool()
        trace = _raw_trace_row("joined")
        audit = _audit_row("joined")
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        self.assertEqual(1, report["joined_case_count"])
        self.assertFalse(any("missing trace" in error or "unknown trace" in error for error in report["errors"]))

    def test_c1_query_and_repair_counts_are_derived_and_mismatches_rejected(self):
        tool = evaluate_tool()
        trace = _raw_trace_row(
            "budget", route="light", semantic_query_count=1,
            repair_query_count=0, adaptive_repair_round_started=True,
            repair_used=True, retrieval_round_count=1,
            adaptive_repair_query={"query_id": "repair-1", "purpose": "repair"},
        )
        trace["executed_queries"] = [
            {"query_id": f"q-{index}", "purpose": "direct"} for index in range(10)
        ] + [{"query_id": "repair-1", "purpose": "repair"}]
        audit = _audit_row("budget")
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        self.assertFalse(report["certifying"])
        self.assertEqual(1, report["budget_violations"]["semantic_query_count"])
        self.assertEqual(1, report["budget_violations"]["repair_query_count"])
        errors = "\n".join(report["errors"])
        self.assertIn("semantic_query_count disagrees", errors)
        self.assertIn("repair_query_count disagrees", errors)

    def test_c2_impossible_sufficient_state_without_provider_attempt_is_rejected(self):
        tool = evaluate_tool()
        trace = _raw_trace_row(
            "impossible", provider_configured=False,
            provider_invocation_started=False, provider_attempted=False,
            provider_attempts=[],
        )
        audit = _audit_row("impossible")
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        self.assertFalse(report["certifying"])
        self.assertEqual(
            1,
            report["deterministic_invariant_violations"]["evidence_without_provider_attempt"],
        )

    def test_c2_partial_and_conflict_require_complete_enrichment_and_disclosures(self):
        tool = evaluate_tool()
        rows = []
        audits = []
        for state, reason in (("partial", "partial_evidence"), ("conflicting", "source_conflict")):
            trace = _raw_trace_row(
                state, evidence_state=state, degradation_reason=reason,
                sufficient_evidence=False, claim_count=0,
                supported_claim_count=0, citation_count=0,
            )
            audit = _audit_row(
                state, claims=[], used_evidence_ids=[], shown_source_urls=[],
                rendered_disclosures=[], missing_claim_topics=[], conflict_groups=[],
            )
            rows.append(trace)
            audits.append(audit)
        report = tool.evaluate_traces(
            rows, audits, sample_manifest=_sample_manifest(rows, audits),
        )
        self.assertFalse(report["certifying"])
        violations = report["deterministic_invariant_violations"]
        self.assertEqual(1, violations["partial_without_missing_topics_or_disclosure"])
        self.assertEqual(1, violations["conflict_without_members_or_disclosure"])

    def test_c2_missing_deterministic_audit_fields_are_non_evaluable(self):
        tool = evaluate_tool()
        trace = _raw_trace_row("missing-audit")
        audit = _audit_row("missing-audit")
        audit.pop("used_evidence_ids")
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        self.assertFalse(report["certifying"])
        self.assertFalse(report["deterministic_evaluation"]["evaluable"])

    def test_c3_spoofed_model_name_cannot_replace_closed_bound_run_manifest(self):
        tool = evaluate_tool()
        cases = [_make_case("fact-1", "stable_fact")]
        predictions = [_prediction("fact-1", model="fixture-baseline ", predicted_tier="light")]
        report = tool.evaluate_offline(cases, predictions)
        self.assertFalse(report["certifying"])
        self.assertNotEqual("independent_predictions", report["artifact_class"])
        self.assertIn("run manifest", "\n".join(report["failures"]))

        bad_manifest = _run_manifest(
            cases, predictions, provenance="fixture_baseline", fixture_derived=True,
            data_source="synthetic_provider_fixtures",
        )
        report = tool.evaluate_offline(cases, predictions, run_manifest=bad_manifest)
        self.assertFalse(report["certifying"])
        self.assertEqual("fixture_baseline", report["artifact_class"])

    def test_i1_integrity_enforces_quality_coverage_unknown_ids_and_recursive_separation(self):
        tool = evaluate_tool()
        case = _make_case("labels", "stable_fact", semantic_labels=[
            {"label_id": "c1", "component": "claim_discovery", "expected": "present"},
        ])
        recording = _recording("labels", case["fixture_id"])
        router = _prediction("labels", predicted_tier="light")

        missing = tool.validate_integrity(
            [case], [recording], [router], expected_case_count=1,
            category_quotas={"stable_fact": 1},
        )
        self.assertIn("missing quality prediction", "\n".join(missing))

        unknown = tool.validate_integrity(
            [case], [recording], [router, _prediction(
                "labels", "claim_discovery", label_id="unknown", predicted="present",
            )], expected_case_count=1, category_quotas={"stable_fact": 1},
        )
        self.assertIn("no external human label", "\n".join(unknown))

        leaked = tool.validate_integrity(
            [case], [recording], [router, _prediction(
                "labels", "claim_discovery", predictions=[{
                    "label_id": "c1", "predicted": "present",
                    "expected": "present", "reviewed_by": "copied",
                }],
            )], expected_case_count=1, category_quotas={"stable_fact": 1},
        )
        self.assertIn("unexpected fields", "\n".join(leaked))

    def test_i2_potential_harm_medium_is_invalid_and_repository_reports_two_enum_errors(self):
        tool = evaluate_tool()
        case = _make_case("harm", "stable_fact", potential_harm="medium")
        errors = tool.validate_integrity(
            [case], [_recording("harm", case["fixture_id"])],
            [_prediction("harm", predicted_tier="light")],
            expected_case_count=1, category_quotas={"stable_fact": 1},
        )
        self.assertIn("case harm invalid potential_harm", errors)
        repository_errors = tool.collect_integrity_errors()
        self.assertEqual(2, sum("invalid potential_harm" in error for error in repository_errors))

    def test_i3_no_web_precedes_explicit_denominator_and_reports_breakdowns(self):
        tool = evaluate_tool()
        trace = _raw_trace_row(
            "no-web", route="skip", skip_reason="user_forbid_web",
            trigger_codes=["explicit_no_web", "explicit_search"],
            degradation_reason="user_forbid_web",
        )
        audit = _audit_row(
            "no-web", route="skip", category="explicit_search",
            explicit_search=True, skip_reason="user_forbid_web",
        )
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        self.assertEqual(0, report["rates"]["explicit_search_route_rate"]["denominator"])
        self.assertEqual(
            {"numerator": 1, "denominator": 1, "rate": 1.0},
            report["rates"]["explicit_no_web_zero_provider_rate"],
        )
        self.assertEqual(1, report["skip_reason_breakdown"]["user_forbid_web"]["count"])
        for name in ("orchestrated_per_routed", "attempted_per_orchestrated", "sufficient_per_attempted"):
            self.assertIn(name, report["conditional_rates"])
        for name in ("provider_timeout", "no_results", "partial", "conflicting", "insufficient"):
            self.assertIn(name, report["outcome_counts"])
        self.assertFalse(any("explicit search route rate" in failure for failure in report["failures"]))

    def test_i4_tier_floor_is_separate_from_reviewed_acceptable_final_tiers(self):
        tool = evaluate_tool()
        case = _make_case(
            "promotion", "stable_fact", minimum_tier="light",
            acceptable_final_tiers=["light", "standard"],
            dynamic=False, high_consequence=False,
        )
        prediction = _prediction("promotion", predicted_tier="standard")
        metrics = tool.routing_quality_metrics([case], [prediction])
        self.assertEqual(0, metrics["minimum_tier_violations"])
        self.assertEqual(1, metrics["tier_target_sample_count"])
        self.assertEqual(1, metrics["acceptable_tier_matches"])

        missing = tool.routing_quality_metrics(
            [_make_case("missing", "stable_fact")],
            [_prediction("missing", predicted_tier="light")],
        )
        self.assertFalse(missing["tier_target_evaluable"])
        self.assertEqual(1, missing["missing_tier_target_count"])

    def test_i5_controlled_provenance_tier_and_stage_zero_samples_are_noncertifying(self):
        tool = evaluate_tool()
        trace = _raw_trace_row("light-only")
        audit = _audit_row(
            "light-only",
            stages_started=["route", "total_response"],
        )
        report = tool.evaluate_traces(
            [trace], [audit], sample_manifest=_sample_manifest([trace], [audit]),
        )
        self.assertFalse(report["certifying"])
        self.assertIn("standard latency zero samples", report["failures"])
        self.assertIn("deep latency zero samples", report["failures"])
        self.assertEqual(0, report["latencies_ms"]["provider_search_total_latency_ms"]["sample_count"])
        self.assertFalse(report["latencies_ms"]["provider_search_total_latency_ms"]["evaluable"])

        bad_manifest = _sample_manifest([trace], [audit], traces_sha256="0" * 64)
        bad = tool.evaluate_traces([trace], [audit], sample_manifest=bad_manifest)
        self.assertFalse(bad["certifying"])
        self.assertIn("traces_sha256", "\n".join(bad["errors"]))


if __name__ == "__main__":
    unittest.main()
