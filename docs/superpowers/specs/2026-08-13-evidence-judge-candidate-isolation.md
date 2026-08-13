# Evidence Judge candidate isolation

## Goal

Prevent one omitted, malformed, duplicated, or unknown Judge row from clearing
other valid candidate judgements in the same batch.

## Contract

- The prompt requires exactly one judgement for every supplied candidate ID.
- Parsing is candidate-level fail-closed:
  - a returned expected ID with one valid row is retained;
  - a missing expected ID is rejected only for that candidate;
  - a malformed expected row is rejected only for that candidate;
  - an unknown ID is discarded and recorded as a closed trace anomaly;
  - duplicate JSON candidate keys reject that candidate deterministically;
  - an empty candidate object yields no admitted judgement.
- Top-level schema remains closed: only `candidates` and `gap_hints` are valid.
- Gap hints remain individually closed and may not invent topic IDs.
- No change to retrieval tiers, budgets, Provider order/fallback, Reader, or
  Evidence admission rules.

## Trace

SearchTrace records only a closed anomaly code and count. It does not record
raw candidate IDs, Judge output, queries, URLs, titles, or Evidence content.

## Verification

Tests cover complete 5/5 output, partial valid output, valid plus unknown ID,
valid plus malformed expected row, duplicate expected ID, and an empty result.
The full hermetic suite and a separately authorized live pipeline probe must
pass before merge.
