# Resilient Search Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent invalid date planning and recoverable provider/read failures from appearing as generic networking failures, without adding an LLM review call or restoring visible citations.

**Architecture:** Keep the existing request analyzer and planner calls, but tighten their closed prompt contracts so event time is not confused with source publication time. Add deterministic Tavily parameter normalization and one bounded no-date retry, then admit directly relevant provider snippets through an explicitly low-confidence evidence path. Preserve all evidence metadata in Trace while mapping provider outages, insufficient evidence, and evidence-backed premise mismatches to distinct source-free QQ messages.

**Tech Stack:** Python 3, dataclasses/enums, Tavily Python SDK, DDGS, `unittest`, existing search pipeline and Trace models.

---

## Working-tree safety

The implementation starts on local `main`, whose pre-existing uncommitted changes include several files this plan must also edit (`src/search/models.py`, `src/search/router.py`, `src/search/planner.py`, `src/search/evidence.py`, and their tests). Before every task, inspect `git diff -- <target files>`. Preserve those changes, use targeted edits, stage only the feature hunks/files explicitly listed for that task, and verify staged content with `git diff --cached`. Never add `.tmp.driveupload/`, `docs/plans/`, `websearch-simplification-report.md`, or unrelated modified files.

## File structure

- Modify `src/search/models.py`: closed time-intent, provider diagnostic, disclosure, and Trace metadata types.
- Modify `src/search/router.py`: single-call request-analysis schema and prompt examples.
- Modify `src/search/planner.py`: stop synthesizing today-only publication bounds; normalize model-generated query dates.
- Modify `src/search/providers/tavily.py`: validate Tavily date bounds and perform one bounded parameter-recovery retry.
- Modify `src/search/orchestrator.py`: preserve Tavily recovery metadata, fallback behavior, and terminal diagnostics.
- Modify `src/search/evidence.py`: controlled low-confidence provider-snippet admission and freshness handling.
- Modify `src/search/policy.py`: map failure classes and evidence-backed premise mismatch to distinct disclosures.
- Modify `src/search/renderer.py`: natural source-free text for the new disclosures.
- Modify `tests/test_search_models.py`, `tests/test_search_router.py`, `tests/test_search_planner.py`, `tests/test_search_providers.py`, `tests/test_search_provider_batches.py`, `tests/test_search_orchestrator.py`, `tests/test_search_evidence.py`, `tests/test_search_policy.py`, `tests/test_search_renderer.py`, and `tests/test_chat_retrieval_flow.py`: focused and end-to-end regression coverage.
- Modify `README.md` and `tests/test_readme_guide.py`: document recovery and low-confidence behavior.

### Task 1: Make time semantics explicit in the existing LLM contracts

**Files:**
- Modify: `src/search/models.py`
- Modify: `src/search/router.py`
- Modify: `src/search/planner.py`
- Test: `tests/test_search_models.py`
- Test: `tests/test_search_router.py`
- Test: `tests/test_search_planner.py`

- [ ] **Step 1: Write failing model and parser tests for the closed time-intent fields**

Add tests that require a closed enum and preserve publication bounds separately from event/fact time:

```python
def test_advisor_parses_search_time_contract(self):
    payload = advisor_payload(
        search_keywords="2026 无畏契约 上海冠军赛 CN 晋级队伍",
        time_scope="year",
        time_scope_text="2026年",
        publication_date_from=None,
        publication_date_to=None,
    )
    parsed = self.module.parse_advisor_json(json.dumps(payload, ensure_ascii=False))
    self.assertEqual("2026 无畏契约 上海冠军赛 CN 晋级队伍", parsed["search_keywords"])
    self.assertIs(parsed["time_scope"], self.module.SearchTimeScope.YEAR)
    self.assertEqual("2026年", parsed["time_scope_text"])
    self.assertIsNone(parsed["publication_date_from"])
    self.assertIsNone(parsed["publication_date_to"])


def test_advisor_rejects_unknown_time_scope(self):
    payload = advisor_payload(time_scope="sometimes")
    self.assertEqual({}, self.module.parse_advisor_json(json.dumps(payload)))
```

Add `RetrievalContext` tests asserting `SearchTimeScope` type validation, whitespace normalization for `time_scope_text`, and rejection of reversed publication ranges.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m unittest tests.test_search_models tests.test_search_router -v
```

Expected: FAIL because `SearchTimeScope` and the new closed fields do not exist.

- [ ] **Step 3: Add the closed model and propagate it through request analysis**

Add to `src/search/models.py`:

```python
class SearchTimeScope(StrEnum):
    NONE = "none"
    TODAY = "today"
    RECENT = "recent"
    YEAR = "year"
    EXPLICIT_RANGE = "explicit_range"
```

Extend `RetrievalContext` with defaults so existing callers remain compatible:

```python
search_keywords: str | None = None
time_scope: SearchTimeScope = SearchTimeScope.NONE
time_scope_text: str | None = None
publication_date_from: date | None = None
publication_date_to: date | None = None
```

Validate the enum/date types, normalize blank `time_scope_text` to `None`, and reject only reversed publication ranges. Extend `_RequestClassification`, `_REQUEST_ANALYSIS_ALLOWED_FIELDS`, `_REQUEST_ANALYSIS_ENUM_FIELDS`, `_normalized_advisor_output`, `_validated_request_classification`, and `_build_request_analysis` in `src/search/router.py` to carry the fields without adding another model invocation. Keep the fields optional at the parser boundary for compatibility with cached/test advisor output: omitted fields normalize to `SearchTimeScope.NONE` and `None` values. Add a test proving the legacy closed payload still parses.

- [ ] **Step 4: Replace the request-analysis prompt with explicit publication-time rules and examples**

Make `ROUTING_SYSTEM_PROMPT` require:

```json
{
  "search_keywords": "concise entity/event/version keywords",
  "time_scope": "none|today|recent|year|explicit_range",
  "time_scope_text": null,
  "publication_date_from": null,
  "publication_date_to": null
}
```

Include these normative examples in the prompt:

```text
“今年参加上海冠军赛的队伍”: time_scope=year, keep the year/event in
search_keywords, publication dates=null because the event occurs this year.
“今天发布了哪些新闻”: time_scope=today; this explicitly concerns source
publication time.
“截至今天有哪些队伍晋级”: keep the cutoff in search_keywords; publication
dates=null because sources need not have been published today.
Never alter a named entity merely because it is unfamiliar.
```

Retain all existing factuality, complexity, freshness, source, and risk fields.

- [ ] **Step 5: Write failing planner tests for no invented today-only range**

Replace the old expectation in `test_current_bounds_do_not_add_queries_and_bound_the_direct_query` and add regressions:

```python
def test_current_freshness_does_not_invent_today_publication_bounds(self):
    plan = planner(today_provider=lambda: date(2026, 9, 3)).plan(
        request("截至今天有哪些队伍晋级"),
        light_decision(),
        retrieval_context(search_keywords="截至 2026-09-03 晋级队伍"),
        freshness_context(FreshnessRequirement.CURRENT),
    )
    self.assertIsNone(plan.initial_queries[0].date_from)
    self.assertIsNone(plan.initial_queries[0].date_to)
    self.assertIn("2026-09-03", plan.initial_queries[0].text)


def test_event_year_does_not_become_publication_window(self):
    context = retrieval_context(
        search_keywords="2026 无畏契约 上海冠军赛 CN 晋级队伍",
        time_scope=SearchTimeScope.YEAR,
        time_scope_text="2026年",
    )
    plan = planner().plan(request("今年参加上海冠军赛的队伍"), light_decision(), context, no_freshness())
    self.assertEqual((None, None), (plan.initial_queries[0].date_from, plan.initial_queries[0].date_to))
```

- [ ] **Step 6: Run planner tests and verify RED**

Run:

```bash
python -m unittest tests.test_search_planner -v
```

Expected: FAIL because `_effective_freshness_context` still converts `CURRENT` to equal bounds.

- [ ] **Step 7: Stop synthesizing source-publication dates and normalize planner output**

Change `_effective_freshness_context` so an unconstrained `CURRENT` context remains unconstrained:

```python
if context.requirement is FreshnessRequirement.CURRENT:
    return context
```

Update `PLANNER_SYSTEM_PROMPT` to repeat the event-time/publication-time distinction. When constructing executable queries, source publication bounds come only from validated `RetrievalContext.publication_date_from/to` or explicitly valid planner output. If optional model output is malformed, reversed, or equal, preserve the query text and set both executable bounds to `None`; never degrade the whole plan.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_search_models tests.test_search_router tests.test_search_planner -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit only Task 1 changes**

```bash
git add -p -- src/search/models.py src/search/router.py src/search/planner.py tests/test_search_models.py tests/test_search_router.py tests/test_search_planner.py
git diff --cached --check
git diff --cached
git commit -m "fix: separate search event and publication time"
```

### Task 2: Normalize Tavily parameters and retry one recoverable rejection

**Files:**
- Modify: `src/search/models.py`
- Modify: `src/search/providers/tavily.py`
- Test: `tests/test_search_models.py`
- Test: `tests/test_search_providers.py`

- [ ] **Step 1: Write failing ProviderResult diagnostic tests**

Add tests for closed, body-free diagnostics:

```python
def test_provider_result_accepts_closed_parameter_recovery_metadata(self):
    result = m.ProviderResult(
        "tavily", m.ProviderStatus.SUCCESS, (hit(),), 3,
        error_code=m.ProviderErrorCode.INVALID_PARAMETERS,
        date_filter_normalized=True,
        parameter_retry_attempted=True,
    )
    self.assertIs(result.error_code, m.ProviderErrorCode.INVALID_PARAMETERS)
    self.assertTrue(result.date_filter_normalized)
    self.assertTrue(result.parameter_retry_attempted)
```

Also assert arbitrary strings are rejected and all three fields default to no diagnostic for existing constructors.

- [ ] **Step 2: Run model tests and verify RED**

Run:

```bash
python -m unittest tests.test_search_models -v
```

Expected: FAIL because `ProviderErrorCode` and recovery metadata are absent.

- [ ] **Step 3: Add closed provider diagnostics**

Add:

```python
class ProviderErrorCode(StrEnum):
    INVALID_PARAMETERS = "invalid_parameters"
    CONNECTION = "connection"
    UNKNOWN = "unknown"
```

Extend `ProviderResult` with:

```python
error_code: ProviderErrorCode | None = None
date_filter_normalized: bool = False
parameter_retry_attempted: bool = False
```

Validate enum/boolean types. Permit `INVALID_PARAMETERS` on a successful result only when recovery was attempted, while never storing exception text or response bodies.

- [ ] **Step 4: Write failing Tavily adapter tests**

Using a fake Tavily client, add:

```python
def test_equal_date_bounds_are_omitted_before_tavily_call(self):
    provider, client = tavily_provider_with_fake_client(success_response())
    result = provider.search(query(date_from=D, date_to=D), tier=SearchTier.LIGHT, max_results=4, timeout_seconds=8)
    self.assertNotIn("start_date", client.calls[0].kwargs)
    self.assertNotIn("end_date", client.calls[0].kwargs)
    self.assertTrue(result.date_filter_normalized)
    self.assertFalse(result.parameter_retry_attempted)


def test_bad_request_with_dates_retries_once_without_dates(self):
    provider, client = tavily_provider_with_fake_client(BadRequestError("bad dates"), success_response())
    result = provider.search(ranged_query(), tier=SearchTier.LIGHT, max_results=4, timeout_seconds=8)
    self.assertEqual(2, len(client.calls))
    self.assertIn("start_date", client.calls[0].kwargs)
    self.assertNotIn("start_date", client.calls[1].kwargs)
    self.assertIs(result.status, ProviderStatus.SUCCESS)
    self.assertIs(result.error_code, ProviderErrorCode.INVALID_PARAMETERS)
    self.assertTrue(result.parameter_retry_attempted)
```

Also test that a second `BadRequestError` returns `ERROR`, a non-parameter exception does not retry, and the retry receives only the remaining stage timeout.

- [ ] **Step 5: Run provider tests and verify RED**

Run:

```bash
python -m unittest tests.test_search_providers -v
```

Expected: FAIL because Tavily passes equal bounds and collapses all exceptions into undifferentiated `ERROR`.

- [ ] **Step 6: Implement bounded normalization and retry**

In `src/search/providers/tavily.py`, import `BadRequestError` alongside `TavilyClient`, measure elapsed time with `time.monotonic`, and centralize one SDK invocation. Before the first invocation remove both bounds when they are equal. On `BadRequestError` with any submitted date bound, calculate remaining timeout and retry exactly once after removing both dates. Return closed metadata on both recovered and terminal results. Keep `TimeoutError` as `TIMEOUT`; classify connection exceptions without including their messages.

The core flow should be equivalent to:

```python
params, normalized = _normalized_tavily_params(query, tier, max_results, timeout_seconds)
try:
    response = client.search(query.text, **params)
except BadRequestError:
    if not _has_date_bounds(params):
        return _error(ProviderErrorCode.INVALID_PARAMETERS, normalized, False)
    retry_params = _without_date_bounds(params, remaining_timeout)
    try:
        response = client.search(query.text, **retry_params)
    except BadRequestError:
        return _error(ProviderErrorCode.INVALID_PARAMETERS, True, True)
    recovered_error = ProviderErrorCode.INVALID_PARAMETERS
```

- [ ] **Step 7: Run focused tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_search_models tests.test_search_providers -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add -p -- src/search/models.py src/search/providers/tavily.py tests/test_search_models.py tests/test_search_providers.py
git diff --cached --check
git diff --cached
git commit -m "fix: recover invalid Tavily date parameters"
```

### Task 3: Preserve recovery truth through fallback and Trace

**Files:**
- Modify: `src/search/models.py`
- Modify: `src/search/orchestrator.py`
- Test: `tests/test_search_provider_batches.py`
- Test: `tests/test_search_orchestrator.py`

- [ ] **Step 1: Write failing orchestration tests**

Add fake provider results proving recovered Tavily prevents DDGS, terminal parameter failure reaches DDGS, and semantic query counts remain unchanged:

```python
def test_recovered_tavily_parameter_error_does_not_fall_back(self):
    tavily = provider_returning(ProviderResult(
        "tavily", ProviderStatus.SUCCESS, (hit("tavily"),), 2,
        error_code=ProviderErrorCode.INVALID_PARAMETERS,
        date_filter_normalized=True,
        parameter_retry_attempted=True,
    ))
    ddgs = recording_provider("ddgs")
    result = run_orchestrator(tavily, ddgs)
    self.assertEqual([], ddgs.calls)
    self.assertEqual(1, result.trace.semantic_query_count)
    self.assertEqual(1, result.trace.tavily_parameter_retry_count)
    self.assertEqual(1, result.trace.date_filter_normalized_count)


def test_terminal_tavily_parameter_error_falls_back_to_ddgs(self):
    tavily = provider_returning(ProviderResult(
        "tavily", ProviderStatus.ERROR, (), 2,
        error_code=ProviderErrorCode.INVALID_PARAMETERS,
        parameter_retry_attempted=True,
    ))
    ddgs = provider_returning(success("ddgs"))
    result = run_orchestrator(tavily, ddgs)
    self.assertEqual(1, len(ddgs.calls))
    self.assertEqual(1, result.trace.semantic_query_count)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m unittest tests.test_search_provider_batches tests.test_search_orchestrator -v
```

Expected: FAIL because provider recovery metadata is not copied into Trace.

- [ ] **Step 3: Add bounded Trace fields and aggregation**

Extend `SearchTrace` with integer/boolean metadata:

```python
date_filter_normalized_count: int = 0
tavily_parameter_retry_count: int = 0
snippet_degradation_used: bool = False
terminal_search_category: SearchTerminalCategory | None = None
```

Define `SearchTerminalCategory` as `provider_connectivity`, `provider_parameters`, `empty_results`, `content_unreadable`, or `insufficient_evidence`. Validate the fields and serialize only these closed values in `to_log_dict()`.

Update `_run_provider_round`/attempt recording so every completed Tavily `ProviderResult` contributes its recovery counters. A successful recovered result resolves the query exactly like ordinary Tavily success; a terminal failure enters the existing DDGS unresolved set. Do not count internal Tavily retry as another semantic query or DDGS stage.

- [ ] **Step 4: Derive terminal categories without leaking exceptions**

At pipeline finalization, derive the category from provider statuses, `ProviderErrorCode`, read outcomes, and final evidence state. Parameter recovery that ultimately succeeds leaves `terminal_search_category=None`; raw exception strings must never enter `SearchTrace`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_search_provider_batches tests.test_search_orchestrator -v
```

Expected: all tests pass, including existing Tavily-first, DDGS fallback, timeout, watchdog, and query-count contracts.

- [ ] **Step 6: Commit Task 3**

```bash
git add -p -- src/search/models.py src/search/orchestrator.py tests/test_search_provider_batches.py tests/test_search_orchestrator.py
git diff --cached --check
git diff --cached
git commit -m "feat: trace Tavily parameter recovery"
```

### Task 4: Admit useful provider snippets as low-confidence evidence

**Files:**
- Modify: `src/search/evidence.py`
- Modify: `src/search/orchestrator.py`
- Test: `tests/test_search_evidence.py`
- Test: `tests/test_search_orchestrator.py`

- [ ] **Step 1: Replace the blanket snippet rejection with failing qualification tests**

Change the old `test_failed_fetch_snippet_is_not_citable_even_when_current_is_satisfied` contract and add:

```python
def test_relevant_failed_fetch_snippet_is_low_confidence_citable(self):
    weak = failed_fetch_candidate(
        content="2026上海全球冠军赛CN四支队伍为JDG、TYL、EDG和XLG。",
        url="https://news.example/valorant",
    )
    judge = StaticEvidenceJudge({"C1": topic_judge_ok(
        "C1", freshness_by_topic={"topic-1": "satisfied"}
    )})
    bundle = EvidenceAssembler(judge).assemble(current_topic_plan("CN参赛队伍"), (weak,))
    self.assertTrue(bundle.evidence_items[0].citable)
    self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
    self.assertEqual("search_result_snippet_after_fetch_failure", bundle.evidence_items[0].extraction_status)


def test_missing_structured_date_can_use_judged_snippet_freshness(self):
    weak = failed_fetch_candidate(content="2026-08-22 四支晋级队伍已经确定", published=None)
    bundle = assemble_with_freshness(weak, judged="satisfied")
    self.assertIs(bundle.topic_assessments[0].freshness, FreshnessEligibility.SATISFIED)
```

Retain/add negative tests for empty/very short snippets, unsafe URLs, unsupported topics, judged stale/unknown freshness, conflicts, and unmet independent corroboration.

- [ ] **Step 2: Run evidence tests and verify RED**

Run:

```bash
python -m unittest tests.test_search_evidence -v
```

Expected: FAIL because `_citable` categorically rejects `search_result_snippet_after_fetch_failure` and date-bounded freshness requires structured `published_at`.

- [ ] **Step 3: Implement the narrow low-confidence gate**

Replace the categorical check with a helper that requires a non-empty normalized excerpt of at least 24 characters, a valid public URL, no safety flags, and a provider-snippet extraction status. Define `_MIN_USEFUL_SNIPPET_CHARS = 24` next to the extraction constants:

```python
def _usable_provider_snippet(candidate: EvidenceCandidate) -> bool:
    excerpt = " ".join(str(candidate.excerpt or "").split())
    return (
        len(excerpt) >= _MIN_USEFUL_SNIPPET_CHARS
        and canonicalize_public_http_url(_final_url_of(candidate)) is not None
        and not candidate.safety_flags
        and candidate.extraction_status in {
            "search_result_snippet",
            "search_result_snippet_after_fetch_failure",
        }
    )
```

`_citable` returns true for page/document extracts as before and for snippets passing this gate. Topic relevance remains exclusively controlled by the existing judge; source requirements remain in `_source_satisfying_evidence_ids`.

- [ ] **Step 4: Permit judged freshness only when structured metadata is absent**

In `_freshness_for_topic`, keep deterministic date checks when `published_at` exists. When it is absent, permit only `FreshnessEligibility.SATISFIED` returned by the judge for a useful snippet; preserve `STALE` and `UNKNOWN` unchanged. This does not infer dates in code and does not let an unsupported candidate into Evidence.

- [ ] **Step 5: Set snippet degradation Trace metadata**

After final Evidence assembly, set `trace.snippet_degradation_used` when any retained/citable `EvidenceItem.extraction_status` is `search_result_snippet` or `search_result_snippet_after_fetch_failure`. Do not expose this flag in visible text.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_search_evidence tests.test_search_orchestrator -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add -p -- src/search/evidence.py src/search/orchestrator.py tests/test_search_evidence.py tests/test_search_orchestrator.py
git diff --cached --check
git diff --cached
git commit -m "feat: allow qualified search snippet evidence"
```

### Task 5: Distinguish outages, limited evidence, and premise mismatches

**Files:**
- Modify: `src/search/models.py`
- Modify: `src/search/policy.py`
- Modify: `src/search/renderer.py`
- Test: `tests/test_search_policy.py`
- Test: `tests/test_search_renderer.py`
- Test: `tests/test_chat_retrieval_flow.py`

- [ ] **Step 1: Write failing policy and renderer tests**

Add closed disclosure codes and expected source-free messages:

```python
def test_provider_outage_has_network_specific_disclosure(self):
    state = decide_answer_state(analysis(), insufficient_bundle(), SearchFailureCode.PROVIDER_UNAVAILABLE)
    self.assertEqual((DisclosureCode.SEARCH_UNAVAILABLE,), state.disclosure_codes)


def test_insufficient_evidence_does_not_claim_network_failure(self):
    state = decide_answer_state(analysis(), insufficient_bundle(), SearchFailureCode.INSUFFICIENT_EVIDENCE)
    self.assertEqual((DisclosureCode.NO_SUPPORTING_EVIDENCE,), state.disclosure_codes)


def test_evidence_backed_premise_mismatch_has_distinct_disclosure(self):
    bundle = insufficient_bundle(repair_reasons=(RepairReasonCode.PREMISE_MISMATCH,))
    state = decide_answer_state(analysis(), bundle, SearchFailureCode.INSUFFICIENT_EVIDENCE)
    self.assertEqual((DisclosureCode.PREMISE_MISMATCH,), state.disclosure_codes)
```

Renderer assertions:

```python
self.assertIn("暂时无法连接在线搜索服务", outage.text)
self.assertIn("暂未找到足以确认结论的信息", insufficient.text)
self.assertIn("检索信息与问题中的名称或前提不一致", mismatch.text)
for reply in (outage, insufficient, mismatch):
    self.assertNotRegex(reply.text, r"\[\d+\]|https?://|来源[：:]")
```

- [ ] **Step 2: Run policy/renderer tests and verify RED**

Run:

```bash
python -m unittest tests.test_search_policy tests.test_search_renderer -v
```

Expected: FAIL because all ordinary failures currently map to `ONLINE_VERIFICATION_FAILED`.

- [ ] **Step 3: Add deterministic disclosure mapping**

Add `DisclosureCode.SEARCH_UNAVAILABLE`, `NO_SUPPORTING_EVIDENCE`, and `PREMISE_MISMATCH`. In `_failure_disclosure`, map `PROVIDER_NOT_CONFIGURED`, `PROVIDER_UNAVAILABLE`, and `PROVIDER_TIMEOUT` to `SEARCH_UNAVAILABLE`; map ordinary no-results/content/evidence failures to `NO_SUPPORTING_EVIDENCE`. Before that generic mapping, let `decide_answer_state` select `PREMISE_MISMATCH` only when the Evidence gap analysis contains an existing judge-validated `PREMISE_MISMATCH` reason.

Do not infer a typo from `NO_RESULTS`, provider errors, judge unavailability, or missing evidence. A nearby interpretation may appear only in a grounded, validated answer when retained Evidence supports it; the fixed mismatch disclosure itself makes no guessed correction.

- [ ] **Step 4: Add natural hidden-source renderer text**

Use:

```python
_SEARCH_UNAVAILABLE = "在线搜索服务暂时不可用，请稍后再试。"
_NO_SUPPORTING_EVIDENCE = "我完成了搜索，但暂未找到足以确认结论的信息。"
_PREMISE_MISMATCH = "检索到的信息与问题中的名称或前提不一致，请确认表述后再试。"
```

Keep the hidden-citation source metadata behavior unchanged.

- [ ] **Step 5: Add chat-flow regressions for the two reported questions**

Using fakes rather than live network calls, add end-to-end cases:

```python
def test_event_year_query_recovers_and_answers_without_sources(self):
    reply, trace = run_search_chat("无畏契约CN赛区今年去上海冠军赛的是哪几个队伍啊", recovered_tavily_bundle())
    self.assertIn("JDG", reply)
    self.assertNotIn("无法完成在线核验", reply)
    self.assertNotRegex(reply, r"\[\d+\]|https?://|来源[：:]")
    self.assertEqual(1, trace.tavily_parameter_retry_count)


def test_known_model_name_is_not_corrected_on_sparse_results(self):
    reply, trace = run_search_chat("Gemini 3.8 Flash什么时候发布的", insufficient_without_premise_hint())
    self.assertNotIn("你可能是指", reply)
    self.assertIn("暂未找到足以确认结论的信息", reply)
```

Also assert partial supported content still answers naturally with the existing partial disclosure and backend `used_evidence_ids`/`shown_source_urls` remain populated but absent from `text` and `chunks`.

- [ ] **Step 6: Run focused integration tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_search_policy tests.test_search_renderer tests.test_chat_retrieval_flow -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add -p -- src/search/models.py src/search/policy.py src/search/renderer.py tests/test_search_policy.py tests/test_search_renderer.py tests/test_chat_retrieval_flow.py
git diff --cached --check
git diff --cached
git commit -m "feat: clarify search failure outcomes"
```

### Task 6: Document and verify the complete behavior

**Files:**
- Modify: `README.md`
- Modify: `tests/test_readme_guide.py`

- [ ] **Step 1: Write the failing README contract test**

Add assertions requiring documentation of single-call planning, date recovery, and snippet degradation while preserving Tavily-first and hidden-source wording:

```python
def test_readme_describes_resilient_search_recovery(self):
    text = README.read_text(encoding="utf-8")
    self.assertIn("事件时间", text)
    self.assertIn("网页发布时间", text)
    self.assertIn("移除日期过滤后重试一次 Tavily", text)
    self.assertIn("搜索摘要", text)
    self.assertIn("低置信度", text)
    self.assertIn("不会额外调用 LLM 复核", text)
```

- [ ] **Step 2: Run the README test and verify RED**

Run:

```bash
python -m unittest tests.test_readme_guide -v
```

Expected: FAIL because the new recovery policy is not documented.

- [ ] **Step 3: Update README**

Document that event/fact time remains in keywords, publication filters are only used for publication-time requests, Tavily invalid date filters are normalized/retried once inside the existing budget, DDGS remains fallback, qualified snippets can support low-confidence answers, and citations/URLs remain backend-only. State explicitly that this uses prompt guidance plus deterministic API validation and adds no LLM review call.

- [ ] **Step 4: Run README and all focused search tests**

Run:

```bash
python -m unittest \
  tests.test_search_models \
  tests.test_search_router \
  tests.test_search_planner \
  tests.test_search_providers \
  tests.test_search_provider_batches \
  tests.test_search_orchestrator \
  tests.test_search_evidence \
  tests.test_search_policy \
  tests.test_search_renderer \
  tests.test_chat_retrieval_flow \
  tests.test_readme_guide -v
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit documentation**

```bash
git add -- README.md tests/test_readme_guide.py
git diff --cached --check
git diff --cached
git commit -m "docs: describe resilient search recovery"
```

- [ ] **Step 6: Run complete verification**

Run:

```bash
python -m unittest discover -s tests -t . -v
python -m compileall -q src tests run_bot.py
git diff --check
git status --short --branch
```

Expected: the entire unittest suite passes, compileall emits no errors, `git diff --check` emits no errors, and `git status` shows only the pre-existing unrelated modifications/untracked paths plus no uncommitted feature changes.

- [ ] **Step 7: Inspect final commits without integrating unrelated work**

```bash
git log -7 --oneline
git show --stat --oneline HEAD~5..HEAD
git status --short
```

Expected: six focused implementation/documentation commits after the design/plan commits; no unrelated path appears in their stats.
