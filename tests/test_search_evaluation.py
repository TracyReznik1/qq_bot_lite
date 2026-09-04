import json
import os
import tempfile
import unittest
from unittest import mock

from tools.evaluate_search import evaluate_rows, evaluate_traces_file, main, run_smoke


def row(**overrides):
    base = {
        "request_id": "req-1",
        "source": "chat",
        "mode": "light",
        "query_count": 1,
        "provider_statuses": {"tavily": "success"},
        "candidate_count": 5,
        "reader_count": 1,
        "planner_degraded": False,
        "ranker_degraded": False,
        "answer_degraded": False,
        "output_kind": "model_answer",
        "stage_latency_ms": {"planner": 10.0},
    }
    base.update(overrides)
    return base


class DeterministicSearchEvaluationTests(unittest.TestCase):
    def test_entry_point_mode_invariants(self):
        report = evaluate_rows([
            row(source="chat", mode="standard", query_count=1),
            row(source="command", mode="light", query_count=1),
            row(source="compatibility", mode="light", query_count=1),
        ])
        self.assertEqual(1, report["violations"]["chat_not_light"])
        self.assertEqual(2, report["violations"]["standard_source_not_standard"])

    def test_mode_query_caps(self):
        report = evaluate_rows([
            row(source="chat", mode="light", query_count=2),
            row(source="command", mode="standard", query_count=4),
        ])
        self.assertEqual(2, report["violations"]["query_cap_exceeded"])

    def test_trace_rejects_sensitive_or_open_ended_fields(self):
        report = evaluate_rows([dict(row(), query="secret", url="https://example.com", exception="body")])
        self.assertEqual(1, report["violations"]["unsafe_trace_record"])

    def test_provider_success_and_degradation_rates(self):
        report = evaluate_rows([
            row(planner_degraded=True, ranker_degraded=False, answer_degraded=True, provider_statuses={"tavily": "success", "ddgs": "error"}),
            row(planner_degraded=False, ranker_degraded=True, answer_degraded=False, provider_statuses={"tavily": "timeout"}),
        ])
        self.assertEqual(0.5, report["rates"]["planner_degraded_rate"])
        self.assertEqual(0.5, report["rates"]["ranker_degraded_rate"])
        self.assertEqual(0.5, report["rates"]["answer_degraded_rate"])
        self.assertAlmostEqual(1 / 3, report["rates"]["provider_success_rate"])

    def test_evaluate_traces_file_and_cli(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".jsonl") as tf:
            tf.write(json.dumps(row()) + "\n")
            tf.write("malformed-json\n")
            path = tf.name
        try:
            report = evaluate_traces_file(path)
            self.assertEqual(2, report["total_traces"])
            self.assertEqual(1, report["violations"]["unsafe_trace_record"])
            exit_code = main(["traces", path])
            self.assertEqual(0, exit_code)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_smoke_requires_explicit_authorization(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            exit_code = run_smoke()
            self.assertEqual(1, exit_code)


if __name__ == "__main__":
    unittest.main()
