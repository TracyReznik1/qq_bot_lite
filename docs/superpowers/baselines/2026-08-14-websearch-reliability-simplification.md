# WebSearch reliability simplification baseline

- Code baseline: `98685ec`
- Package-aware hermetic suite: `956 tests, 0 failures, 0 errors, 20.854s`
- Preserved caps: light `1/5/2/0/1`; standard `3 initial, 1 repair, 4 total, 8 URL, 5 Read, 2 rounds`.
- Preserved flow: DDGS first, Tavily fallback, Reader, one Judge batch per round, Evidence, Answer Policy, Validator, Renderer.
- Known failure reproduction A: one completed sibling Query can be discarded when another sibling consumes the shared request deadline.
- Known failure reproduction B: a Judge support row can be rejected by the retired `relevance` field even when `supported_topic_ids` is valid.
- This artifact is a diagnostic baseline, not a quality certificate. Live DDGS/Tavily quality remains unverified.
- Stage-focused command: `python -B -m unittest tests.test_search_providers tests.test_search_orchestrator tests.test_search_evidence -v`.
- Stage-focused suites: `173 tests, 0 failures, 0 errors, 2.406s`.
