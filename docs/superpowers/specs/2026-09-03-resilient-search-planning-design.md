# Resilient Search Planning Design

## Goal

Prevent recoverable search-planning mistakes from turning successful provider searches into a generic online-verification failure, without adding a second LLM call or exposing citations and source details in QQ replies.

## Scope

This change covers:

- clearer instructions and a closed output format for the existing single request-analysis/planning LLM call;
- separation of event/fact time from source publication time;
- deterministic provider-parameter validation;
- one Tavily retry without invalid date filters before DDGS fallback;
- cautious use of relevant provider snippets when page extraction is unavailable;
- accurate user-facing failure wording and backend Trace diagnostics.

It does not add another LLM review pass, parallel provider execution, visible citations, or a general natural-language date parser in application code.

## Architecture

The existing retrieval pipeline remains Tavily-first. The request-analysis LLM produces concise search keywords and structured time intent using a stricter prompt. Application code treats that output as a proposal: it validates only closed-schema types, date ordering, and provider API invariants rather than attempting to reinterpret arbitrary language.

Search planning distinguishes the time described by the requested fact or event from the publication date of a web page. Provider publication-date filters are emitted only when the user explicitly asks for material published in a particular period. Event-time expressions remain in the search keywords.

If Tavily rejects a request because of date parameters, the orchestrator retries that query once with publication-date filters removed. DDGS remains the fallback only when the normalized Tavily attempt also fails or yields no usable URLs.

## Request-Analysis Contract

The existing LLM request-analysis output gains or clarifies these fields:

- `search_keywords`: concise keywords preserving the core entity, event, version, and relevant time expression;
- `time_scope`: one of `none`, `today`, `recent`, `year`, or `explicit_range`;
- `time_scope_text`: the exact normalized time expression to retain in the query, or `null`;
- `publication_date_from`: ISO date or `null`;
- `publication_date_to`: ISO date or `null`.

The prompt must explain that publication-date bounds describe when sources were published, not when the event happened or when a fact is valid. It includes at least these examples:

- “今年参加上海冠军赛的队伍” keeps the year/event term in `search_keywords` and leaves both publication dates null.
- “今天发布了哪些新闻” may use a publication-time constraint.
- “截至今天有哪些队伍晋级” treats today as the fact cutoff and does not require every source to be published today.
- A named model such as “Gemini 3.8 Flash” remains unchanged unless retrieved evidence—not planning uncertainty—shows that the name is wrong.

Invalid or incomplete optional time fields degrade to no publication-date filter. They must not invalidate otherwise usable search keywords or cause the code to invent a today-only range.

## Provider Parameter Normalization

Before invoking Tavily, code applies these deterministic checks:

1. Parse publication bounds only from valid ISO dates accepted by the closed contract.
2. Reject reversed ranges where the start is after the end.
3. Do not send equal `start_date` and `end_date` values because Tavily rejects them.
4. Preserve the relevant date expression in `search_keywords` when an exact-day filter is omitted.
5. Never synthesize a same-day range merely because freshness is `current`.

These checks enforce API correctness, not natural-language semantics.

## Tavily Retry and DDGS Fallback

Each semantic query follows this sequence:

1. Invoke Tavily with the validated plan.
2. If Tavily reports a parameter-related error and date bounds were supplied, retry Tavily once for that query with both publication-date bounds removed.
3. Accept usable Tavily URLs from the retry normally.
4. Fall back to DDGS only if Tavily is unavailable, times out, errors after normalization, returns no results, or returns no valid public URLs.

The retry remains inside the Tavily stage budget. It does not add another LLM call. Non-parameter Tavily errors continue directly to DDGS so transient or unrelated failures do not cause redundant Tavily calls.

Because the current provider adapter collapses all exceptions into `ProviderStatus.ERROR`, it must retain a closed, non-sensitive reason code sufficient to distinguish invalid parameters from other provider errors. Raw response bodies, API keys, query result bodies, and exception strings must not enter Trace.

## Snippet Degradation

A provider result snippet may become low-confidence evidence when page extraction is unavailable, provided that:

- it has a valid public final/search URL;
- it directly supports a required topic according to the existing evidence judge;
- it is not empty, truncated below the existing minimum usefulness threshold, or marked unsafe;
- freshness is supported by explicit snippet/title text or by the judge when structured publication metadata is absent;
- source-quality requirements are still met, including independent corroboration when the plan requires it.

A snippet produced after fetch failure and an ordinary provider snippet follow the same explicit low-confidence path rather than being rejected solely because full-page extraction failed. The Evidence item records its origin and degradation status. Conflicting or indirect snippets do not become sufficient evidence.

Low-confidence evidence may support a qualified answer. It does not allow unsupported claims, bypass claim validation, or turn an unknown entity into a claimed typo.

## Entity and Premise Handling

Failure to retrieve sufficient evidence is not proof that the user misspelled an entity. The system may suggest a corrected or nearby name only when retrieved evidence explicitly establishes that relationship. Otherwise it must preserve the user's entity wording and report limited evidence without guessing.

For a real entity such as “Gemini 3.8 Flash”, provider errors, inaccessible pages, or sparse results must not produce an unsupported “you may mean another model” correction.

## User-Visible Outcomes

Replies continue to hide citation numbers, source titles, URLs, and source lists.

The visible outcome distinguishes:

- **Provider outage:** all providers were unavailable or failed to connect; explain that online search is temporarily unavailable.
- **Limited evidence:** search ran but only part of the answer is supported; answer the supported part and naturally state that the result may be incomplete.
- **No supporting evidence:** state that reliable information could not yet be confirmed, without claiming a network outage.
- **Evidence-backed premise mismatch:** explain the likely naming or premise issue and offer the evidence-supported nearby interpretation.
- **Recovered parameter error:** answer normally; the recovery appears only in Trace.

## Trace and Privacy

Trace records closed metadata only:

- whether publication dates were omitted or normalized;
- whether a no-date Tavily retry was attempted and its status;
- whether provider-snippet degradation was used;
- the terminal category: provider connectivity, provider parameter rejection, empty results, content unreadable, or insufficient evidence.

Existing backend evidence mappings, validation results, used evidence IDs, shown source URLs, and provider attempts remain available for diagnostics. Visible reply text and chunks remain source-free. Trace must not contain raw exception messages or sensitive provider configuration.

## Testing Strategy

Use TDD with focused unit tests followed by retrieval-flow integration tests.

### Request analysis and planning

- “今年参加上海冠军赛的队伍” retains the year/event wording and emits no source-publication date bounds.
- “截至今天有哪些队伍晋级” emits no same-day publication filter.
- “今天发布了哪些新闻” retains valid publication-time intent without creating a Tavily-invalid equal range.
- malformed optional time fields degrade to no date filter while preserving keywords.
- no path silently invents a today-only range from `current`.

### Provider and orchestration

- equal Tavily date bounds are removed before invocation.
- a Tavily parameter rejection with date bounds triggers exactly one no-date Tavily retry.
- successful retry prevents DDGS invocation.
- failed retry falls back to DDGS.
- non-parameter errors do not receive the normalization retry.
- all retry work remains subject to the existing Tavily stage/watchdog budget.

### Evidence

- a directly relevant provider snippet can support a low-confidence answer after page-fetch failure.
- explicit date text can satisfy freshness when structured `published_at` is absent.
- indirect, unsafe, empty, stale, or conflicting snippets remain insufficient.
- independent corroboration requirements still apply.
- retrieval failure alone cannot produce an entity correction.

### Rendering and Trace

- provider outage, limited evidence, no evidence, and evidence-backed premise mismatch have distinct natural-language outcomes.
- recovered parameter errors are not shown to the user.
- no reply contains citation numbers, source labels, titles, or URLs.
- Trace exposes only closed diagnostic codes and retains backend evidence metadata.

Finally, run the complete unittest suite, `python -m compileall -q src tests run_bot.py`, and `git diff --check`.

## Compatibility and Migration

No stored chat or memory migration is required. Existing plans lacking the optional time fields remain valid and behave as unbounded publication-date searches. Provider ordering, DDGS stage budgets, hidden-citation behavior, and backend evidence retention remain unchanged.
