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

The checked-in `model_predictions.jsonl` is a router-only fixture baseline. The
offline command binds it to a closed `search-eval-run-v1` manifest with
`provenance=fixture_baseline`, `data_source=synthetic_provider_fixtures`, and
`fixture_derived=true`. It is useful for exercising deterministic plumbing, but
it is not an independent model run and can never certify model quality. A model
name is not provenance. An independent run needs a manifest whose hashes bind
the exact case and prediction arrays and whose run timestamp matches every
prediction row.

The CLI returns nonzero while owner review, reviewed final-tier targets,
independent predictions, or semantic samples are missing. A zero-sample or
missing-reviewed-subgroup group is reported as `evaluable=false` with null PRF
values; it is never reported as a perfect score.

## Integrity and separation contracts

`integrity` validates every required case/recording/prediction field, closed
enums, ISO review dates, timezone-qualified prediction timestamps, exact case
quotas, unique IDs, normalized duplicate questions, HTTP(S) fixture URLs, and
fixture-to-case references. Every non-skip case must reference its own provider
fixture. It also requires exactly one router prediction per case and a one-to-one
quality prediction for every semantic label. `potential_harm` is exactly
`none|low|high`; the two currently checked-in `medium` rows remain honest
integrity errors pending owner remediation.

Cases may carry the closed reviewed subgroup flags `dynamic` and
`high_consequence`, plus either `expected_final_tier` or a nonempty
`acceptable_final_tiers`. The minimum tier is only an ordinal safety floor;
upward promotion is not automatically an error. Macro tier quality is
non-evaluable when the separate reviewed final-tier target is absent.

Human labels stay only in `cases.jsonl`. Prediction rows containing fields such
as `minimum_tier`, `semantic_labels`, `reviewed_by`, `reviewed_at`, or any other
expected/human-label field at any nesting depth are rejected. Router, planner,
and each semantic component use component-specific exact schemas; unknown
fields are rejected. Predictions and recordings with an
unknown or missing `case_id` are rejected; duplicate `(case_id, component)`
singleton prediction rows and duplicate `(case_id, component, label_id)` quality
predictions are rejected.

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

`traces` accepts raw current `SearchTrace.to_log_dict()` rows. Those rows contain
`request_id`, not `case_id`. A separate, closed human-audit/enrichment JSONL row
provides the reviewed `case_id -> request_id` join. The evaluator rejects
unknown/missing/duplicate joins and does not accept embedded expected labels in
a Trace.

The audit schema requires route truth and reviewer fields plus per-claim support
and Evidence IDs, each Evidence final HTTP(S) URL/relevance/citable verdict,
used Evidence IDs, shown source URLs, missing topics, conflict-member groups,
rendered disclosure codes, and the stages that actually started. Missing or
malformed deterministic fields make the sample non-evaluable and
non-certifying. The raw Trace and every nested provider attempt are also closed
schemas. Query, repair, and retrieval-round counts are derived from validated
query metadata; a contradictory serialized counter is an integrity error.

Trace mode also requires a closed `search-trace-sample-v1` manifest. Its hashes
bind the exact raw Trace and audit arrays. Only
`provenance=controlled_production` with `fixture_derived=false` can certify;
synthetic fixtures remain diagnostic.

Explicit no-web rows are excluded before both `D_factual` and explicit-search
denominators are formed. Legal closed-context rows are excluded before
`D_factual`. The report separately shows both exclusion counts, the no-web
zero-provider rate, and a closed per-`skip_reason` breakdown.
Provider-not-configured and provider-unavailable rows remain execution failures;
they are never rewritten as route skips.

The JSON report includes numerator, denominator, and rate for factual route
coverage, orchestrator start, provider attempt, sufficient Evidence, and the
explicit-search subgroups. Provider attempt is also reported on the configured
subset. It separately reports orchestrated/routed, attempted/orchestrated, and
sufficient/attempted conversion rates, and timeout, no-result, partial,
conflicting, and insufficient outcomes. Candidate URL, content read, semantic
query, repair query, retrieval
round, and hard-timeout violations use the immutable tier budgets:

| tier | candidates | reads | semantic queries | repair queries | rounds | hard timeout |
|---|---:|---:|---:|---:|---:|---:|
| light | 5 | 2 | 1 | 0 | 1 | 8 s |
| standard | 8 | 5 | 4 | 1 | 2 | 20 s |
| deep | 15 | 8 | 6 | 1 | 2 | 40 s |

Citation/failure invariants are counted separately from model metrics. They
cover provider/evidence contradictions, Claim-to-Evidence-to-final-URL mapping,
used-versus-shown sources, relevance admission, count reconciliation, partial
missing topics, retained unsupported or missing-topic claims, conflict members,
dynamic unsupported conclusions, and required failure/conflict disclosures.

Every Trace latency field is reported with nearest-rank P50/P95/P99, including
route, planning, initial/total provider search, initial/total content read,
initial/total Evidence assembly, gap analysis, adaptive repair, answer,
structural validation, semantic validation, QQ rendering, retrieval pipeline,
and total response latency. Each percentile uses only audit rows where that
stage actually started, so a not-run zero is excluded; route and total response
have their own declared denominators. Every required stage and tier with zero
samples is non-evaluable and non-certifying. Retrieval P95 is evaluated separately by final tier
(`light <= 6 s`, `standard <= 15 s`, `deep <= 30 s`); answer, validation, and
render time are not folded into it.

## Commands

```powershell
python tools/evaluate_search.py integrity
python tools/evaluate_search.py offline
python tools/evaluate_search.py traces --traces path\to\search-traces.jsonl --labels path\to\human-audit.jsonl --manifest path\to\sample-manifest.json
python tools/evaluate_search.py online --limit 10
```

Online provider runs are opt-in and never part of `unittest`.
Without separate user authorization and credentials, `online` reports
`status="not run"`, `certifying=false`, and exits nonzero. The current command
does not perform provider access.
