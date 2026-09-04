"""Deterministic search evaluation and smoke runner."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.search.simple.models import (
    OutputKind,
    RequestSource,
    SearchMode,
    SearchRequest,
)

ALLOWED_TRACE_KEYS = frozenset({
    "request_id",
    "source",
    "mode",
    "query_count",
    "provider_statuses",
    "candidate_count",
    "reader_count",
    "planner_degraded",
    "ranker_degraded",
    "answer_degraded",
    "output_kind",
    "stage_latency_ms",
})

ALLOWED_OUTPUT_KINDS = frozenset({k.value for k in OutputKind})
ALLOWED_SOURCES = frozenset({RequestSource.CHAT.value, RequestSource.COMMAND.value, RequestSource.COMPATIBILITY.value})
ALLOWED_MODES = frozenset({SearchMode.LIGHT.value, SearchMode.STANDARD.value})
CLOSED_STATUSES = frozenset({
    "success", "empty", "timeout", "error", "not_configured", "unavailable"
})


def evaluate_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, Any]:
    total_traces = 0
    violations: Counter[str] = Counter()
    output_kinds: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    modes: Counter[str] = Counter()

    total_planner_degraded = 0
    total_ranker_degraded = 0
    total_answer_degraded = 0

    provider_attempts = 0
    provider_successes = 0

    for raw in rows:
        if not isinstance(raw, Mapping):
            violations["malformed_trace"] += 1
            continue

        total_traces += 1
        keys = set(raw.keys())
        if not keys.issubset(ALLOWED_TRACE_KEYS):
            violations["unsafe_trace_record"] += 1

        source = str(raw.get("source", ""))
        mode = str(raw.get("mode", ""))
        query_count = raw.get("query_count")

        if source not in ALLOWED_SOURCES or mode not in ALLOWED_MODES:
            violations["malformed_entry_point"] += 1

        if source == "chat" and mode != "light":
            violations["chat_not_light"] += 1
        elif source in ("command", "compatibility") and mode != "standard":
            violations["standard_source_not_standard"] += 1

        if not isinstance(query_count, int):
            violations["malformed_query_count"] += 1
        else:
            if mode == "light" and query_count != 1:
                violations["query_cap_exceeded"] += 1
            elif mode == "standard" and not (1 <= query_count <= 3):
                violations["query_cap_exceeded"] += 1

        output_kind = str(raw.get("output_kind", ""))
        if output_kind not in ALLOWED_OUTPUT_KINDS:
            violations["malformed_output_kind"] += 1
        else:
            output_kinds[output_kind] += 1

        sources[source] += 1
        modes[mode] += 1

        if bool(raw.get("planner_degraded")):
            total_planner_degraded += 1
        if bool(raw.get("ranker_degraded")):
            total_ranker_degraded += 1
        if bool(raw.get("answer_degraded")):
            total_answer_degraded += 1

        provider_statuses = raw.get("provider_statuses")
        if isinstance(provider_statuses, Mapping):
            for p, st in provider_statuses.items():
                provider_attempts += 1
                st_str = str(st)
                if st_str not in CLOSED_STATUSES:
                    violations["malformed_provider_status"] += 1
                elif st_str == "success":
                    provider_successes += 1
        elif provider_statuses is not None:
            violations["malformed_provider_statuses"] += 1

    return {
        "total_traces": total_traces,
        "violations": dict(violations),
        "rates": {
            "planner_degraded_rate": (total_planner_degraded / total_traces) if total_traces else 0.0,
            "ranker_degraded_rate": (total_ranker_degraded / total_traces) if total_traces else 0.0,
            "answer_degraded_rate": (total_answer_degraded / total_traces) if total_traces else 0.0,
            "provider_success_rate": (provider_successes / provider_attempts) if provider_attempts else 0.0,
        },
        "output_kinds": dict(output_kinds),
        "sources": dict(sources),
        "modes": dict(modes),
    }


def evaluate_traces_file(path: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except Exception:
                rows.append({"__malformed_json__": True})
    return evaluate_rows(rows)


def run_smoke() -> int:
    auth = os.environ.get("QQBOT_ALLOW_LIVE_SEARCH_SMOKE")
    if auth != "1":
        sys.stderr.write("Set QQBOT_ALLOW_LIVE_SEARCH_SMOKE=1 to authorize live provider calls.\n")
        return 1

    from src.search.simple.factory import get_simple_search_pipeline
    pipeline = get_simple_search_pipeline()

    chat_req = SearchRequest(
        mode=SearchMode.LIGHT,
        text="今日科技要闻",
        source=RequestSource.CHAT,
    )
    chat_outcome = pipeline.run(chat_req)
    print(json.dumps(chat_outcome.trace.to_safe_dict(), ensure_ascii=False))

    cmd_req = SearchRequest(
        mode=SearchMode.STANDARD,
        text="今日天气预报",
        source=RequestSource.COMMAND,
    )
    cmd_outcome = pipeline.run(cmd_req)
    print(json.dumps(cmd_outcome.trace.to_safe_dict(), ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic search evaluator and smoke runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    traces_parser = subparsers.add_parser("traces", help="Evaluate search trace JSONL file")
    traces_parser.add_argument("path", help="Path to traces.jsonl")

    subparsers.add_parser("smoke", help="Run authorized live search smoke test")

    args = parser.parse_args(argv)
    if args.command == "traces":
        report = evaluate_traces_file(args.path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    elif args.command == "smoke":
        return run_smoke()
    return 1


if __name__ == "__main__":
    sys.exit(main())
