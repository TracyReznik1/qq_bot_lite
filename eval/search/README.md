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

## Commands

```powershell
python tools/evaluate_search.py integrity
python tools/evaluate_search.py offline
python tools/evaluate_search.py traces --traces path\to\search-traces.jsonl --labels path\to\human-audit.jsonl
python tools/evaluate_search.py online --limit 10
```

Online provider runs are opt-in and never part of `unittest`.
