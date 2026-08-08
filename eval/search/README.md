# Evidence-Search Evaluation Set

This directory holds a 140-case synthetic, privacy-safe evaluation dataset for the
retrieval-benefit search pipeline.

## Files

- `cases.jsonl` — 140 labelled cases across 8 categories with exact quotas.
- `provider_recordings.jsonl` — synthetic fixture recordings referenced by `fixture_id`.
- `model_predictions.jsonl` — offline component predictions (router, planner,
  relevance, claim discovery, semantic support). Predictions never overwrite human
  labels, which live only in `cases.jsonl`.

## Category quotas

| Category | Count |
|---|---:|
| no_benefit | 20 |
| stable_fact | 20 |
| explanation_comparison | 25 |
| dynamic_fact | 20 |
| regulated_controversy | 15 |
| explicit_search | 10 |
| ambiguous_mixed | 10 |
| failure_partial_conflict | 20 |
| **Total** | **140** |

## Review state

Every row requires a real `reviewed_by` / `reviewed_at`. The reserved value
`unreviewed` is rejected by the integrity check. Until the project owner has
reviewed all 140 rows, the offline acceptance thresholds cannot be certified.

The checked-in `model_predictions.jsonl` is a router-only
`fixture-baseline`. It is useful for exercising deterministic plumbing, but it
is not an independent model run and can never certify model quality. The CLI
therefore returns nonzero while the owner review, independent predictions, or
semantic samples are missing. A zero-sample group is reported as
`evaluable=false` with null PRF values; it is never reported as a perfect score.

## Integrity and separation contracts

`integrity` validates every required case/recording/prediction field, closed
enums, ISO review dates, timezone-qualified prediction timestamps, exact case
quotas, unique IDs, normalized duplicate questions, HTTP(S) fixture URLs, and
fixture-to-case references. Every non-skip case must reference its own provider
fixture. It also requires exactly one router prediction per case.

Human labels stay only in `cases.jsonl`. Prediction rows containing fields such
as `minimum_tier`, `semantic_labels`, `reviewed_by`, `reviewed_at`, or any other
expected/human-label field are rejected. Predictions and recordings with an
unknown or missing `case_id` are rejected; duplicate `(case_id, component)`
prediction rows are rejected.

Semantic human labels use a closed, joinable shape inside a case:

```json
{"label_id":"claim-1","component":"claim_discovery","expected":"present"}
{"label_id":"claim-1","component":"semantic_support","expected":"supported"}
{"label_id":"candidate-1","component":"relevance","expected":"relevant"}
```

The corresponding component prediction row stores predictions separately:

```json
{
  "case_id": "dynamic-001",
  "component": "semantic_support",
  "model": "model-name",
  "model_version": "version",
  "prompt_schema_version": "retrieval-v1",
  "run_timestamp": "2026-07-29T00:00:00Z",
  "predictions": [
    {"label_id": "claim-1", "predicted": "supported"}
  ]
}
```

Closed values are `present|absent` for claim discovery,
`supported|partial|conflict|unsupported|unmapped` for semantic support, and
`relevant|irrelevant|direct|contextual|admitted|excluded|pass|fail` for the
relevance gate. Missing predictions, predictions without an external label,
and duplicate label joins make the run non-certifying.

## Trace acceptance

`traces` joins every trace to an externally reviewed label row by `case_id`.
It ignores any embedded `d_factual` or expected label in a trace. Explicit
no-web and legal closed-context rows are excluded before `D_factual` is formed,
and both exclusion counts are reported. Provider-not-configured and
provider-unavailable rows remain execution failures; they are never rewritten
as route skips.

The JSON report includes numerator, denominator, and rate for factual route
coverage, orchestrator start, provider attempt, sufficient Evidence, and the
explicit-search subgroups. Provider attempt is also reported on the configured
subset. Candidate URL, content read, semantic query, repair query, retrieval
round, and hard-timeout violations use the immutable tier budgets:

| tier | candidates | reads | semantic queries | repair queries | rounds | hard timeout |
|---|---:|---:|---:|---:|---:|---:|
| light | 5 | 2 | 1 | 0 | 1 | 8 s |
| standard | 8 | 5 | 4 | 1 | 2 | 20 s |
| deep | 15 | 8 | 6 | 1 | 2 | 40 s |

Citation/failure invariants are counted separately from model metrics. They
cover skip/provider contradictions, missing orchestration, claims or citations
without sufficient support, impossible supported-claim counts, citations
without citable Evidence, knowledge-fallback citations, and evidence/failure
state mismatches.

Every Trace latency field is reported with nearest-rank P50/P95/P99, including
route, planning, initial/total provider search, initial/total content read,
initial/total Evidence assembly, gap analysis, adaptive repair, answer,
structural validation, semantic validation, QQ rendering, retrieval pipeline,
and total response latency. Retrieval P95 is evaluated separately by tier
(`light <= 6 s`, `standard <= 15 s`, `deep <= 30 s`); answer, validation, and
render time are not folded into it.

## Commands

```powershell
python tools/evaluate_search.py integrity
python tools/evaluate_search.py offline
python tools/evaluate_search.py traces --traces path\to\search-traces.jsonl --labels path\to\human-audit.jsonl
python tools/evaluate_search.py online --limit 10
```

Online provider runs are opt-in and never part of `unittest`.
Without separate user authorization and credentials, `online` reports
`status="not run"`, `certifying=false`, and exits nonzero. The current command
does not perform provider access.
