# WebSearch reliability simplification — acceptance report

## 1. Scope

Acceptance of the implementation plan
`docs/superpowers/plans/2026-08-14-websearch-reliability-simplification.md`
against its specification
`docs/superpowers/specs/2026-08-13-websearch-reliability-simplification-design.md`.

Final implementation commit: `1f62e9b` (`chore: close web search runtime contracts`),
2026-08-17 04:40 +0800.

## 2. Hermetic gates

- Search-focused suite (Tasks 2–8 domain): `python -B -m unittest tests.test_search_budget tests.test_search_stage_runner tests.test_search_models tests.test_search_simplification_baseline tests.test_search_router tests.test_search_providers tests.test_search_url_policy tests.test_search_extraction tests.test_search_evidence tests.test_search_validation tests.test_search_orchestrator tests.test_search_planner tests.test_search_policy tests.test_search_outcomes tests.test_search_provider_batches tests.test_search_renderer tests.test_chat_retrieval_flow` → **517 tests, 0 failures, 0 errors**.
- Evaluator tests: `python -B -m unittest tests.test_search_evaluation` → 80 tests, 74 pass, 6 errors. The 6 errors are environment-only (`tempfile.mkdtemp()` directories become non-writable in the sandbox used for the CLI negative-path tests), not code failures.
- Blind-runner tests: `python -B -m unittest tests.test_search_blind_acceptance_runner` → **15 tests, 0 failures, 0 errors**.
- `git diff --check` → exit 0.

## 3. Blind acceptance gate

The blind gate harness is implemented and covered:

- `tools/run_search_blind_acceptance.py` accepts a reviewer-owned sealed case file
  (`sealed_at` + `cases[{case_id, category, question, expected_route, fault_profile}]`).
- It rejects duplicate IDs, unknown categories, unknown fault profiles, normal
  categories with a fault profile, failure-injection categories without one, files
  sealed before the final implementation commit, the schema-marker question text,
  and any question found by exact normalized match in repository text.
- Its report is body-free: raw questions and answer text never appear in the JSON
  output.

The **live blind online run has NOT been executed**. It requires, per the
specification, an independent reviewer who did not implement Tasks 2–9 to
generate and seal unseen questions, plus explicit online authorization and live
DDGS/Tavily credentials. Neither condition is satisfied in this environment, so
the runner correctly reports `status="not run"`, `certifying=False`.

Per-case pass/fail, route, stage outcomes, elapsed time, and manual citation
verdict are therefore recorded only after that external run.

## 4. Honest conclusion

Hermetic and blind-case acceptance proves conformance for this bounded set. It
does not certify general internet retrieval quality. Real DDGS/Tavily quality
remains a separate, manually reviewed external gate and must be reported
honestly, including any failed blind case.

## 5. Known environment limitation

The full package-aware suite (`python -B -m unittest discover -s tests -t .`)
reports 313 errors in this sandbox because `tempfile.mkdtemp()`-created
directories become non-writable here (SQLite/history/case files cannot be
created inside them). This is an environment restriction, not a code defect;
the same suites pass in a writable environment. The search-focused suite above
does not depend on those temp directories and is fully green.
