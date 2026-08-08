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
argument-free offline command classifies it as an untrusted fixture diagnostic.
It is useful for exercising deterministic plumbing, but it is not an independent
model run and can never certify model quality. The evaluator blacklists the
checked-in fixture prediction hash and known fixture model identities even when
a caller re-labels a manifest as independent. A model name, a hash, or an
artifact-authored boolean is not provenance.

A certifying independent run needs a closed `search-eval-run-v1` manifest whose
hashes bind the exact canonical case, provider-recording, and prediction arrays,
whose timestamp matches every prediction, and whose `attestation` is a valid
HMAC-SHA256 over the canonical manifest including immutable
`attestation.algorithm` and `attestation.key_id` metadata but excluding only
`attestation.signature`. Canonical JSON is UTF-8,
key-sorted, compact (`separators=(",", ":")`), and preserves non-ASCII text.
The closed attestation object contains only `algorithm="hmac-sha256"`, `key_id`,
and a lowercase 64-character `signature`. The verifier secret must be supplied
separately through the callable API or a named CLI environment variable; it is
never read from the manifest or printed. Missing/short/wrong verifier secrets,
invalid signatures, post-signing metadata changes, and unexpected attestation
fields are non-certifying.

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

Certifying offline API and CLI runs must supply provider recordings separately.
The verified manifest includes `recordings_sha256`, and offline scoring runs the
same full case/recording/prediction integrity path. Missing recordings,
normalized duplicate questions, duplicate fixture IDs, or a case-to-recording
reference mismatch are non-certifying.

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

`traces` accepts raw current `SearchTrace.to_log_dict()` rows. Production request
IDs are lowercase `req-` plus 32 UUID hex characters, and the safe-log boundary
preserves that exact join identifier. Those rows contain `request_id`, not
`case_id`. A separate, closed human-audit/enrichment JSONL row
provides the reviewed `case_id -> request_id` join. The evaluator rejects
unknown/missing/duplicate joins and does not accept embedded expected labels in
a Trace.

The audit schema requires route truth and reviewer fields plus per-claim support,
Evidence IDs, `partial_topic_ids`, `conflict_group_ids`, and owned
`disclosure_codes`; each Evidence has a
final HTTP(S) URL/relevance/citable verdict,
used Evidence IDs, shown source URLs, missing topics, conflict-member groups,
rendered disclosure codes, and the stages that actually started. Missing or
malformed deterministic fields make the sample non-evaluable and
non-certifying. The raw Trace and every nested provider attempt are also closed
schemas. Query, repair, and retrieval-round counts are derived from validated
query metadata; a contradictory serialized counter is an integrity error.

Trace mode also requires a closed `search-trace-sample-v1` manifest. Its hashes
bind the exact raw Trace and audit arrays, and it uses the same externally
verified HMAC attestation contract. Only `provenance=controlled_production` with
`fixture_derived=false` can certify; synthetic fixtures and evidence URLs on
fixture/example/test hosts remain diagnostic regardless of manifest claims.
Fixture identities containing Unicode control/format characters are rejected.
Evidence hosts use Unicode normalization, IDNA/case folding, and DNS terminal
root-dot removal before reserved-host classification.

The current production transition is exact: `final_tier == route`. Tier budgets
are always selected from `route`. For every joined row, the trace must also
reconcile with the external audit's skip reason, external-fact requirement,
program minimum tier, explicit/no-web trigger, and acceptable final-tier target.
Factual non-skip audits require both a minimum tier and one or more acceptable
non-skip targets at or above that floor.

Explicit no-web rows are excluded before both `D_factual` and explicit-search
denominators are formed. Legal closed-context rows are excluded before
`D_factual`. The report separately shows both exclusion counts, the no-web
zero-provider rate, and a closed per-`skip_reason` breakdown.
Provider-not-configured and provider-unavailable rows remain execution failures;
they are never rewritten as route skips. A routed, orchestrated required-search
row with no configured provider must record the `provider_not_configured`
degradation, deduplicated provider failure code, matching disclosure, no attempt,
and a non-success evidence outcome. `provider_execution_accounted_rate` gates
the whole applicable factual population, so declaring a provider unconfigured
cannot remove a row from acceptance.
The configured-but-unavailable/no-attempt state must instead record
`provider_unavailable`, a non-success outcome, and its matching disclosure.
Either readiness state counts as execution-accounted, is excluded from the
invocation-eligible configured-attempt denominator, and remains separately
visible in failure/outcome metrics. Missing or mismatched failure codes,
degradations, and disclosures are violations.

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
Every retained material claim must map only to existing direct/relevant,
citable, used, cited, and shown Evidence edges; unrelated good Evidence cannot
admit a bad claim edge. Any retained partial/conflicting claim forces the
corresponding non-definitive evidence state, degradation structure, missing-topic
or conflict membership, and rendered disclosure, independent of subgroup.
Partial topics must belong to that exact claim. Every conflict group referenced
by a claim must exist and contain two or more Evidence IDs mapped to that same
claim; unrelated audit-global structure cannot satisfy either contract.

Every Trace latency field is reported with nearest-rank P50/P95/P99, including
route, planning, initial/total provider search, initial/total content read,
initial/total Evidence assembly, gap analysis, adaptive repair, answer,
structural validation, semantic validation, QQ rendering, retrieval pipeline,
and total response latency. Each percentile uses only audit rows where that
stage actually started, so a not-run zero is excluded; a zero is included when
the stage did start. Positive latency and downstream execution facts are
cross-checked against `stages_started`, preventing a slow executed stage from
being omitted. Route and total response have their own declared denominators.
`adaptive_repair` is bidirectional: stage membership equals Trace repair state,
and the state, single repair query, derived/serialized counts, and latency must
agree before inclusion.
Every required stage and tier with zero samples is non-evaluable and
non-certifying. Retrieval P95 is evaluated separately by route
(`light <= 6 s`, `standard <= 15 s`, `deep <= 30 s`); answer, validation, and
render time are not folded into it.

## Commands

```powershell
python tools/evaluate_search.py integrity
python tools/evaluate_search.py offline
python tools/evaluate_search.py offline --cases path\to\reviewed-cases.jsonl --recordings path\to\provider-recordings.jsonl --predictions path\to\independent-predictions.jsonl --manifest path\to\signed-run-manifest.json --verifier-key-env SEARCH_EVAL_VERIFIER_KEY
python tools/evaluate_search.py traces --traces path\to\search-traces.jsonl --labels path\to\human-audit.jsonl --manifest path\to\signed-sample-manifest.json --verifier-key-env SEARCH_EVAL_VERIFIER_KEY
python tools/evaluate_search.py online --limit 10
```

The named environment variable above is illustrative; choose a deployment-owned
name and inject its secret outside the artifact directory. The argument-free
offline command is deliberately non-certifying. Independent offline mode
requires all five explicit options. Trace mode requires the signed manifest and
verifier option. Any absent input, invalid JSON/schema/hash/signature, fixture
re-label, missing prediction, or zero required sample exits nonzero.

Skip traces follow a closed per-reason table. Non-web-forbidden skips have no
search triggers, provider failures, or degradation; provided-text transforms
and summaries are `mixed`, while other closed tasks are `non_factual`.
`user_forbid_web` is `ambiguous`, requires `explicit_no_web`, and permits only
the explicit-search-conflict/high-consequence companion triggers and its
matching optional degradation. Every skip has
`external_fact_required=false`, no program tier/evidence/provider readiness,
and zero execution.

Online provider runs are opt-in and never part of `unittest`.
Without separate user authorization and credentials, `online` reports
`status="not run"`, `certifying=false`, and exits nonzero. The current command
does not perform provider access.
