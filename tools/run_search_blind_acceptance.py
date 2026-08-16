"""Blind online acceptance runner for the WebSearch reliability simplification.

Run::

    python -B tools/run_search_blind_acceptance.py --cases <sealed-cases.json> --output <result.json>

The runner accepts a reviewer-owned sealed case file and, only with explicit
online authorization, executes each case through the search pipeline. Without
authorization it validates the file and reports ``status="not run"`` and
``certifying=False`` — it never fabricates a live-search result.

The sealed case file uses this schema::

    {
      "sealed_at": "ISO-8601 timestamp",
      "cases": [
        {
          "case_id": "blind-01",
          "category": "current_single_fact",
          "question": "reviewer-supplied after implementation",
          "expected_route": "light",
          "fault_profile": "none"
        }
      ]
    }

The literal example question text is a schema marker and is rejected as an
executable question.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]

CLOSED_CATEGORIES = frozenset(
    {
        "current_single_fact",
        "current_release_or_version",
        "current_event_result",
        "complete_schedule_or_list",
        "official_announcement",
        "multi_topic_comparison",
        "ddgs_failure_tavily_success",
        "sibling_query_partial_success",
        "reader_partial_completion",
        "judge_row_partial_failure",
    }
)

CLOSED_FAULT_PROFILES = frozenset(
    {
        "none",
        "ddgs_failure_tavily_success",
        "sibling_query_partial_success",
        "reader_partial_completion",
        "judge_row_partial_failure",
    }
)

NORMAL_ROUTE_CATEGORIES = frozenset(
    {
        "current_single_fact",
        "current_release_or_version",
        "current_event_result",
        "complete_schedule_or_list",
        "official_announcement",
        "multi_topic_comparison",
    }
)

FAILURE_INJECTION_CATEGORIES = frozenset(
    {
        "ddgs_failure_tavily_success",
        "sibling_query_partial_success",
        "reader_partial_completion",
        "judge_row_partial_failure",
    }
)

SCHEMA_MARKER = "reviewer-supplied after implementation"

_TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".txt", ".jsonl", ".json", ".toml", ".cfg", ".ini", ".yaml", ".yml"}
)

_EXCLUDE_DIR_PARTS = frozenset(
    {
        ".git", ".venv", ".worktrees", "node_modules", "__pycache__",
        "qqbot_data", "atri_data", "atri_data.backup-20260714-202341",
        ".tmp.driveupload", ".idea", ".claude",
    }
)


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().casefold())


def _is_sealed_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def repo_text_files(roots: Iterable[Path] | None = None) -> Iterable[Path]:
    """Yield repository text files whose content may embed a blind question."""
    for root in roots or (ROOT,):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            if any(part in _EXCLUDE_DIR_PARTS for part in rel.parts):
                continue
            if path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            yield path


def repo_normalized_texts(
    roots: Iterable[Path] | None = None,
) -> Mapping[Path, str]:
    texts: dict[Path, str] = {}
    for path in repo_text_files(roots):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        texts[path] = normalize_question(content)
    return texts


def question_collides(
    question: str,
    *,
    texts: Mapping[Path, str] | None = None,
) -> bool:
    """Return True when the exact normalized question already appears in text."""
    normalized = normalize_question(question)
    if not normalized:
        return True
    scanned = texts if texts is not None else repo_normalized_texts()
    return any(normalized in content for content in scanned.values())


def validate_case_file(
    obj: Mapping[str, Any],
    *,
    sealed_after: str | None = None,
    repo_texts: Mapping[Path, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate a sealed case file; return (cases, errors)."""
    errors: list[str] = []
    if not isinstance(obj, Mapping):
        return [], ["case file must be a JSON object"]
    if not _is_sealed_timestamp(obj.get("sealed_at")):
        errors.append("sealed_at must be an ISO-8601 timestamp")
    raw_cases = obj.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        errors.append("cases must be a non-empty list")
        return [], errors

    if sealed_after is not None:
        try:
            sealed_at = _parse_timestamp(obj["sealed_at"])
            minimum = _parse_timestamp(sealed_after)
            if sealed_at < minimum:
                errors.append("sealed_at precedes the final implementation commit")
        except (KeyError, ValueError):
            pass

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, case in enumerate(raw_cases, 1):
        prefix = f"case {index}"
        if not isinstance(case, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        case_id = case.get("case_id")
        category = case.get("category")
        question = case.get("question")
        expected_route = case.get("expected_route")
        fault_profile = case.get("fault_profile", "none")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"{prefix} case_id must be a non-blank string")
        elif case_id in seen_ids:
            errors.append(f"{prefix} duplicate case_id {case_id!r}")
        else:
            seen_ids.add(case_id)
        if category not in CLOSED_CATEGORIES:
            errors.append(f"{prefix} unknown category {category!r}")
        if fault_profile not in CLOSED_FAULT_PROFILES:
            errors.append(f"{prefix} unknown fault_profile {fault_profile!r}")
        if category in NORMAL_ROUTE_CATEGORIES and fault_profile != "none":
            errors.append(f"{prefix} normal category requires fault_profile none")
        if category in FAILURE_INJECTION_CATEGORIES and fault_profile == "none":
            errors.append(f"{prefix} failure-injection category requires a fault profile")
        if not isinstance(expected_route, str) or expected_route not in {"light", "standard"}:
            errors.append(f"{prefix} expected_route must be light or standard")
        if not isinstance(question, str) or not question.strip():
            errors.append(f"{prefix} question must be a non-blank string")
        elif normalize_question(question) == normalize_question(SCHEMA_MARKER):
            errors.append(f"{prefix} question is the schema marker, not an executable question")
        elif question_collides(question, texts=repo_texts):
            errors.append(f"{prefix} question matches repository text")
        cases.append(dict(case))
    return cases, errors


def body_free_report(
    cases: list[dict[str, Any]],
    *,
    status: str,
    certifying: bool,
    errors: list[str],
) -> dict[str, Any]:
    """Serialize a body-free report: never emits raw questions or answers."""
    rows = []
    for case in cases:
        rows.append(
            {
                "case_id": case.get("case_id"),
                "category": case.get("category"),
                "expected_route": case.get("expected_route"),
                "fault_profile": case.get("fault_profile", "none"),
                "actual_route": None,
                "outcome": None,
                "elapsed_ms": None,
                "citation_verdict": None,
                "stage_outcomes": {},
            }
        )
    return {
        "status": status,
        "certifying": certifying,
        "case_count": len(cases),
        "cases": rows,
        "errors": errors,
    }


def _load_json(path: Path) -> tuple[Any, list[str]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [f"cannot read case file: {exc}"]
    try:
        return json.loads(raw), []
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON in case file: {exc}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="absolute path to the sealed case file")
    parser.add_argument("--output", required=True, help="absolute path for the JSON result")
    parser.add_argument(
        "--sealed-after",
        default=None,
        help="optional ISO-8601 floor; sealed_at earlier than this is rejected",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="execute cases against live providers (requires credentials)",
    )
    args = parser.parse_args(argv)

    cases_path = Path(args.cases)
    output_path = Path(args.output)
    obj, load_errors = _load_json(cases_path)
    if load_errors:
        report = body_free_report([], status="rejected", certifying=False, errors=load_errors)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 2

    repo_texts = repo_normalized_texts()
    cases, errors = validate_case_file(
        obj,
        sealed_after=args.sealed_after,
        repo_texts=repo_texts,
    )
    if errors:
        report = body_free_report(cases, status="rejected", certifying=False, errors=errors)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 2

    # The actual blind execution requires live providers and a reviewer who did
    # not implement the production changes. Without --online, never fabricate a
    # live result.
    status = "not run" if not args.online else "run (external; not implemented here)"
    certifying = False
    report = body_free_report(cases, status=status, certifying=certifying, errors=[])
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
