# Deterministic Search Modes Design

**Date:** 2026-09-04

## Goal

Replace model-selected search routing with caller-owned deterministic modes while rewriting the search runtime as a small independent pipeline:

- ordinary chat always uses `LIGHT` search;
- `/search` always uses `STANDARD` search;
- `/skip` is the only user-facing route that selects `SKIP` and bypasses search;
- text, image-only, and text-plus-image inputs are supported by all three chat paths;
- the model may generate search queries but may never choose or change the search mode.

## Scope

The implementation will be a clean rewrite under `src/search/simple/`. Existing search state-machine code may be read to understand provider APIs, URL safety, page fetching, and integration behavior, but its routing, evidence, claim, repair, freshness, risk, and validation abstractions will not be extended or reused in the new design.

After the new runtime is integrated and verified, the old runtime and obsolete tests will be deleted atomically. No compatibility aliases for old routing or evidence types will remain.

## Routing Invariant

The entry point owns the mode:

| Entry point | Mode | Query count | Sources shown |
|---|---|---:|---|
| Ordinary chat | `LIGHT` | exactly 1 | no |
| `/search` | `STANDARD` | 1–3 | up to 3 |
| `/skip` | `SKIP` | 0 | no |

No model response contains a `mode` field. No regex, heuristic, risk policy, freshness rule, or fallback may change these mappings.

`/skip` without content returns a usage message. `/skip` with text, images, or both uses normal multimodal chat generation without constructing or invoking the search pipeline.

## Request and Command Contracts

`SearchRequest` carries an explicit `mode: SearchMode` and normalized text plus image inputs. It does not use `force_search` to infer policy.

`generate_reply` receives an explicit mode from its caller:

- normal message dispatch passes `SearchMode.LIGHT`;
- `/search` passes `SearchMode.STANDARD`;
- `/skip` passes `SearchMode.SKIP`.

The generic slash router remains syntax-only. The command registry adds `skip`, and help/unknown-command output lists it. Search-like commands must receive downloaded image inputs, so main message dispatch will no longer discard images merely because a message is a command.

## Fixed-Mode Query Planning

A new `QueryPlanner` receives the already-selected mode, user text, and optional images.

- `LIGHT` returns exactly one normalized, non-empty query.
- `STANDARD` returns one to three normalized, deduplicated queries.
- `SKIP` never calls the planner.

For text-only requests, the planner asks the configured chat model for JSON containing only `queries`. For image-only and text-plus-image requests, it sends multimodal content so the generated query can reflect visible subjects, text, products, places, or events.

The parser accepts the first valid balanced JSON object, ignores unknown fields, normalizes whitespace, removes duplicates, and enforces caller-owned count limits. A returned `mode` field is ignored.

Fallbacks are deterministic:

- when text is present, use the normalized user text as the first query;
- for image-only input, use `识别并查找图片中的主体、事件或内容`;
- planner exceptions, malformed output, empty output, and timeouts set `planner_degraded=True` but never change mode.

## Search Pipeline

The rewritten pipeline has focused components:

1. **Models** — request, fixed plan, result, trace, failure, and response records.
2. **Query planner** — query generation only; no route selection.
3. **Providers** — final simple provider protocol and records.
4. **Retrieval** — concurrent Tavily-first execution, DDGS fallback for unresolved queries, URL safety, canonical deduplication, and mode-specific caps.
5. **Reader** — bounded concurrent page fetches only for missing or short snippets.
6. **Ranker** — one tolerant relevance-scoring call; stable deterministic degradation.
7. **Answerer** — natural Simplified Chinese answer from title/excerpt evidence; deterministic summary fallback.
8. **Renderer** — QQ length bounds, warnings, and command-only source display.
9. **Factory** — production dependency construction and reset hook.

Provider and HTTP calls receive real configured timeouts. Executors are request-local and shut down without waiting indefinitely.

## Data Flow

### Ordinary chat

1. Main dispatch downloads attached images.
2. Chat service creates a `LIGHT` request.
3. Query planner generates exactly one text or multimodal query.
4. Retrieval, optional reading, and ranking produce evidence.
5. Answerer receives conversation context and body-bounded evidence without URLs.
6. Renderer returns a natural answer without source URLs.
7. The final reply is appended to history.

### `/search`

The flow is the same, except the mode is `STANDARD`, the planner may return up to three queries, reading/retrieval limits are larger, and rendering appends at most three canonical sources.

### `/skip`

1. Empty content and no images return usage guidance.
2. Main dispatch downloads attached images.
3. Chat service invokes plain multimodal generation directly.
4. No planner, provider, reader, ranker, or search answerer is constructed or called.
5. The final reply is appended to history using the original command text as the user-history entry.

## Failure Behavior

Every failure is bounded and yields a non-empty reply:

- planner failure: original text or the fixed image-only query;
- Tavily failure or unusable URLs: DDGS receives only unresolved queries;
- both providers fail or no usable result: fixed search-unavailable/no-results message;
- reader failure: retain provider snippets;
- ranker failure: preserve provider order and add `信息可能不完整。`;
- answer failure: deterministic ranked title/excerpt summary;
- unexpected pipeline or chat-dispatch exception: fixed temporary-unavailable message.

Safe traces contain modes, counts, closed status values, degradation flags, output kind, and latency only. They never contain query text, URLs, page bodies, model evidence, image bytes, credentials, or exception messages.

## Images

Normal chat and `/search` include images in query planning and answer generation. Image-only input therefore still performs the required search:

- ordinary image-only message → one generated LIGHT query;
- image-only `/search` → one to three generated STANDARD queries.

`/skip` includes images only in plain multimodal answer generation and performs no search.

If multimodal query planning fails, the mode remains fixed and the image-only fallback query is used.

## Migration Strategy

The rewrite proceeds in three phases:

1. Build and test the complete independent `src/search/simple/` runtime without altering the old state machine.
2. Switch chat, `/search`, `/skip`, image dispatch, and the compatibility search service to the new runtime.
3. In one atomic cleanup task, migrate providers to final simple contracts, remove the old runtime/tests, update exports, and run the complete suite plus static forbidden-symbol scans.

The partially implemented model-routing `RoutePlanner` and its tests are obsolete. They will be replaced by the fixed-mode `QueryPlanner`; existing simple contracts and retrieval code may be rewritten wherever they conflict with this specification.

## Testing

Tests must prove the routing invariant at component and black-box levels:

- every ordinary text, image-only, and text-plus-image chat request selects `LIGHT`;
- every `/search` request selects `STANDARD`;
- every `/skip` request selects `SKIP` and invokes zero search dependencies;
- model output cannot change the caller-selected mode;
- LIGHT is capped at one query and STANDARD at three;
- planner failures preserve mode and use deterministic fallbacks;
- Tavily invalid/empty/failed results correctly trigger DDGS;
- provider, reader, ranker, and answer timeouts are forwarded and bounded;
- normal replies hide URLs while `/search` shows at most three;
- image inputs reach multimodal query planning and answering;
- history stores the original slash-command text and final reply;
- old search runtime files and forbidden concepts are absent after cleanup.

Final verification includes the complete hermetic suite, `compileall`, `git diff --check`, forbidden-import scans, mocked text/image routing probes, and authorized live text and image smoke tests when credentials are available.

## Documentation and Configuration

Documentation will describe deterministic routing rather than adaptive/model-selected routing. It will include `/skip`, image behavior, source visibility, provider fallback, degradation behavior, and all active timeout variables.

Because search mode selection no longer has a model call, the planner timeout is specifically a **query-planning timeout**, not a routing timeout. Configuration and names should reflect that distinction; obsolete route-policy settings and descriptions are removed.
