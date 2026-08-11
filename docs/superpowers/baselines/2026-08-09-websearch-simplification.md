# WebSearch Simplification Baseline

## Frozen implementation and retained contracts

- Baseline implementation: `8abaa8f`
- Frozen caps: light `1/5/2/0/1/1/8`; standard `3/8/5/1/4/2/20`
- Retained: DDGS-first/Tavily conditional fallback, absolute deadline, Reader,
  Evidence relevance gate, Claim/Citation validation, and body-free Trace.

## Intentional migration surface

- Intentionally changed: operational deep, risk/freshness tier floors,
  explicit-search standard floor, and deep failure/validation branches.

## Verification record and external gates

- Hermetic package-aware suite: `842` tests passed with
  `python -B -m unittest discover -s tests -t . -v` on 2026-08-11.
- Commit baseline: `9379da5`.
- External gates are not certified: 140 owner-review rows are incomplete, two
  rows use illegal `potential_harm=medium`, and online mode remains not run.

This is a behavior baseline, not a real online quality certification. It records
the current hermetic reliability contract without claiming live DDGS/Tavily
retrieval quality.
