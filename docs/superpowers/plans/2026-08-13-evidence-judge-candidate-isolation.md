# Evidence Judge Candidate Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Evidence Judge parsing candidate-level fail-closed without weakening admission or retrieval budgets.

**Architecture:** Keep the existing batch LLM call and strict per-row verdict parser. Replace the batch candidate-ID equality gate with duplicate-aware JSON decoding and independent expected-row parsing; return a small parse result carrying valid rows plus closed anomaly metadata for SearchTrace.

**Tech Stack:** Python 3.13, dataclasses, json object-pairs hooks, unittest.

---

### Task 1: Candidate-level parser contract

**Files:**
- Modify: `tests/test_search_evidence.py`
- Modify: `src/search/evidence.py`

- [ ] Add six parser regression tests: complete five rows; partial valid rows;
  valid plus unknown ID; valid plus malformed expected row; duplicate expected
  ID; and empty candidates.
- [ ] Run `python -B -m unittest tests.test_search_evidence.EvidenceJudgeSchemaTests -v`
  and verify the partial/unknown/duplicate cases fail against the current batch gate.
- [ ] Add duplicate-aware JSON object parsing. Keep the top-level exact key set,
  discard unknown rows, reject only duplicated expected IDs, and feed every
  expected row independently through `_parse_verdict`.
- [ ] Strengthen `_JUDGE_SYSTEM_PROMPT` to require exactly one row per supplied
  candidate ID and forbid omitted, merged, duplicated, or invented IDs.
- [ ] Re-run the targeted class and the full `tests.test_search_evidence` module.

### Task 2: Privacy-safe anomaly trace

**Files:**
- Modify: `tests/test_search_models.py`
- Modify: `tests/test_search_orchestrator.py`
- Modify: `src/search/models.py`
- Modify: `src/search/evidence.py`
- Modify: `src/search/orchestrator.py`
- Modify: `tools/evaluate_search.py`
- Modify: `tests/test_search_evaluation.py`

- [ ] Add RED tests requiring closed anomaly codes/counts, total serialization,
  evaluator schema acceptance, and absence of raw candidate IDs or Judge text.
- [ ] Add a closed `JudgeAnomalyCode` enum and bounded Trace count/code fields.
- [ ] Propagate parser anomalies through the immutable Evidence bundle into
  SearchTrace; do not change Evidence admission state computation.
- [ ] Update evaluator schema/validation for the new closed metadata only.
- [ ] Run the focused models/evidence/orchestrator/evaluation tests.

### Task 3: Acceptance

**Files:**
- No new production files.

- [ ] Run `python -B -m unittest discover -s tests -t . -q`.
- [ ] Run `python -B -m compileall -q src tests tools` and `git diff --check`.
- [ ] Verify `git diff --name-only -- eval/search` is empty.
- [ ] With explicit network approval, repeat the live dynamic-query pipeline
  probe and require at least one valid Judge row to survive when other rows are
  omitted or malformed.
- [ ] Request independent code review, address findings, and create one scoped
  commit only after all gates pass.
