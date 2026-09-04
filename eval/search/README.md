# Deterministic Search Evaluation & Safe Traces

This directory documents the evaluation and verification model for the deterministic search pipeline in `qqbot_lite`.

## Core Invariants

1. **Caller-Owned Deterministic Modes:**
   - **Ordinary Chat:** Always selects `LIGHT` mode with exactly 1 query. Never displays source URLs to QQ users.
   - **`/search` Command:** Always selects `STANDARD` mode with 1 to 3 queries. Displays up to 3 source URLs.
   - **`/skip` Command:** Always selects `SKIP` mode. Forwards text and images to plain multimodal chat, invokes no search dependencies, and emits no search trace.
   - **Multimodal Everywhere:** Pure text, pure image, and text+image are supported identically across all modes.

2. **Provider Hierarchy & Fallback:**
   - Tavily is the primary provider, called concurrently for planned queries.
   - DDGS is the secondary provider, receiving only queries that Tavily failed to resolve, returned empty, timed out, errored, or produced only invalid URLs.
   - If both providers fail or return no usable results, the pipeline degrades gracefully to clean failure messages without fabricating online evidence.

3. **Privacy-Safe Search Traces:**
   - Every search pipeline run produces a `SearchTrace` that can be serialized via `to_safe_dict()`.
   - Safe traces contain only closed metadata:
     - `request_id`: Request identifier string
     - `source`: Request source (`chat`, `command`, `compatibility`)
     - `mode`: Search mode (`light`, `standard`)
     - `query_count`: Number of queries executed
     - `provider_statuses`: Status per provider (`success`, `empty`, `timeout`, `error`, `not_configured`, `unavailable`)
     - `candidate_count`: Total retrieved URLs
     - `reader_count`: Pages fetched for snippet expansion
     - `planner_degraded`: Whether query planning fell back to default query
     - `ranker_degraded`: Whether ranking fell back to provider order
     - `answer_degraded`: Whether answering fell back to deterministic summary
     - `output_kind`: Result type (`plain`, `model_answer`, `summary_fallback`, `search_failure`)
     - `stage_latency_ms`: Execution latency breakdown in milliseconds
   - Safe traces strictly omit raw query strings, target URLs, fetched page bodies, evidence excerpts, image bytes, and credentials.

## Evaluation Tool

`tools/evaluate_search.py` validates trace collections against mode and safety invariants:

```bash
# Evaluate a JSONL file of safe traces:
python tools/evaluate_search.py traces path/to/traces.jsonl

# Run an authorized live smoke test:
QQBOT_ALLOW_LIVE_SEARCH_SMOKE=1 python tools/evaluate_search.py smoke
```

### Evaluator Metrics & Invariant Checks

- **Entry Point Mode Invariants:** Verifies that `chat` always produces `light`, while `command` and `compatibility` always produce `standard`.
- **Query Caps:** Flags query count violations (`light == 1`, `1 <= standard <= 3`).
- **Trace Hygiene:** Flags any row containing unsafe, open-ended fields outside the safe trace schema.
- **Provider Success & Degradation Rates:** Computes overall provider success rate and planner/ranker/answer degradation frequencies.
