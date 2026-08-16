import importlib
import json
import unittest
from pathlib import Path


def runner_module():
    try:
        return importlib.import_module("tools.run_search_blind_acceptance")
    except ImportError as exc:
        raise AssertionError("tools.run_search_blind_acceptance must exist") from exc


def _case(**overrides):
    case = {
        "case_id": "blind-01",
        "category": "current_single_fact",
        "question": "一个从未在仓库出现过的全新实体事实问题",
        "expected_route": "light",
        "fault_profile": "none",
    }
    case.update(overrides)
    return case


def _sealed_file(cases, sealed_at="2099-01-01T00:00:00Z"):
    return {"sealed_at": sealed_at, "cases": cases}


class RunnerSchemaTests(unittest.TestCase):
    def test_valid_case_file_is_accepted(self):
        runner = runner_module()
        obj = _sealed_file([_case()])
        cases, errors = runner.validate_case_file(
            obj, sealed_after="2026-01-01T00:00:00Z", repo_texts={}
        )
        self.assertEqual([], errors)
        self.assertEqual(1, len(cases))

    def test_non_object_case_file_is_rejected(self):
        runner = runner_module()
        cases, errors = runner.validate_case_file([], repo_texts={})
        self.assertEqual([], cases)
        self.assertIn("JSON object", "\n".join(errors))

    def test_missing_sealed_at_is_rejected(self):
        runner = runner_module()
        obj = {"cases": [_case()]}
        _, errors = runner.validate_case_file(obj, repo_texts={})
        self.assertTrue(any("sealed_at" in error for error in errors))

    def test_empty_cases_is_rejected(self):
        runner = runner_module()
        obj = _sealed_file([])
        _, errors = runner.validate_case_file(obj, repo_texts={})
        self.assertTrue(any("non-empty" in error for error in errors))


class RunnerRejectionTests(unittest.TestCase):
    def test_duplicate_case_id_is_rejected(self):
        runner = runner_module()
        obj = _sealed_file([_case(), _case()])
        _, errors = runner.validate_case_file(obj, repo_texts={})
        self.assertTrue(any("duplicate case_id" in error for error in errors))

    def test_unknown_category_is_rejected(self):
        runner = runner_module()
        obj = _sealed_file([_case(category="made_up_category")])
        _, errors = runner.validate_case_file(obj, repo_texts={})
        self.assertTrue(any("unknown category" in error for error in errors))

    def test_unknown_fault_profile_is_rejected(self):
        runner = runner_module()
        obj = _sealed_file([_case(fault_profile="made_up_fault")])
        _, errors = runner.validate_case_file(obj, repo_texts={})
        self.assertTrue(any("unknown fault_profile" in error for error in errors))

    def test_normal_category_requires_none_fault_profile(self):
        runner = runner_module()
        obj = _sealed_file(
            [_case(category="current_single_fact", fault_profile="reader_partial_completion")]
        )
        _, errors = runner.validate_case_file(obj, repo_texts={})
        self.assertTrue(any("fault_profile none" in error for error in errors))

    def test_failure_injection_category_requires_a_fault_profile(self):
        runner = runner_module()
        obj = _sealed_file(
            [_case(category="judge_row_partial_failure", fault_profile="none")]
        )
        _, errors = runner.validate_case_file(obj, repo_texts={})
        self.assertTrue(any("fault profile" in error for error in errors))

    def test_schema_marker_question_is_rejected(self):
        runner = runner_module()
        obj = _sealed_file([_case(question="reviewer-supplied after implementation")])
        _, errors = runner.validate_case_file(obj, repo_texts={})
        self.assertTrue(any("schema marker" in error for error in errors))

    def test_pre_implementation_timestamp_is_rejected(self):
        runner = runner_module()
        obj = _sealed_file([_case()], sealed_at="2020-01-01T00:00:00Z")
        _, errors = runner.validate_case_file(
            obj, sealed_after="2026-01-01T00:00:00Z", repo_texts={}
        )
        self.assertTrue(any("final implementation commit" in error for error in errors))

    def test_repository_text_collision_is_rejected(self):
        runner = runner_module()
        question = "这个精确的问题文本已经出现在仓库里"
        texts = {Path("docs/example.md"): runner.normalize_question(f"前言 {question} 结尾")}
        obj = _sealed_file([_case(question=question)])
        _, errors = runner.validate_case_file(obj, repo_texts=texts)
        self.assertTrue(any("matches repository text" in error for error in errors))

    def test_question_not_in_repository_is_accepted(self):
        runner = runner_module()
        question = "一个确定不在仓库文本里的问题"
        texts = {Path("docs/example.md"): "完全无关的内容"}
        self.assertFalse(runner.question_collides(question, texts=texts))


class RunnerReportTests(unittest.TestCase):
    def test_report_is_body_free(self):
        runner = runner_module()
        question = "私密问题文本不应出现在报告里"
        cases = [_case(question=question)]
        report = runner.body_free_report(
            cases, status="not run", certifying=False, errors=[]
        )
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(question, serialized)

    def test_report_serializes_to_json_with_closed_shape(self):
        runner = runner_module()
        cases = [_case()]
        report = runner.body_free_report(
            cases, status="not run", certifying=False, errors=[]
        )
        loaded = json.loads(json.dumps(report, ensure_ascii=False))
        self.assertEqual("not run", loaded["status"])
        self.assertFalse(loaded["certifying"])
        self.assertEqual(1, loaded["case_count"])
        row = loaded["cases"][0]
        self.assertEqual("blind-01", row["case_id"])
        self.assertNotIn("question", row)
        self.assertNotIn("answer", row)


if __name__ == "__main__":
    unittest.main()
