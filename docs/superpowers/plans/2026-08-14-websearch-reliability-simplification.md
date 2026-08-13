# QQ Bot WebSearch Reliability Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让现有 WebSearch 在独立阶段预算内保留所有已完成的检索成果，以 DDGS 优先、Tavily 独立预算回退，并由唯一的 `supported_topic_ids` 合同稳定地产生 Evidence、回答和来源。

**Architecture:** 保留现有线性架构和 `skip / light / standard`，不增加 Agent、LLM 调用、搜索层级、第三轮或 Answer-to-Search 回环。新增一个纯预算策略和一组封闭的 Query/Read/Judge 状态，删除共享 rolling deadline、Provider 内部隐式预算争抢和 Judge relevance 兼容链；所有结果由现有 Orchestrator、Evidence、Policy、Validator、Renderer 单向消费。

**Tech Stack:** Python 3.11+、`dataclasses`、`StrEnum`、`ThreadPoolExecutor`、现有同步 LLM/Provider 接口、`unittest`、PowerShell、Git。

---

## Global execution rules

- Baseline is commit `98685ec`; the fresh package-aware suite is `956/956` green on 2026-08-14.
- Preserve the user-owned untracked paths `.tmp.driveupload/` and `websearch-simplification-report.md` exactly; never stage, delete, move, or rewrite them.
- Execute every task as RED → minimal GREEN → focused regression → independent spec review → independent quality review → commit. Do not combine commits across tasks.
- Do not change `eval/search/*.jsonl`, fabricate reviewer labels, or describe hermetic results as live-search quality certification.
- Do not delete an existing test method or material case to obtain green. Migrate fixtures only when a deliberately removed schema makes the old fixture invalid.
- Do not add a Risk, Freshness, Checklist, Gap, Repair Benefit, Pivot, Stop, Relevance, or Renderer LLM. Keep one Request Analysis call, one Planner call only for standard initial planning, one Candidate Judge batch call per retrieval round, and the existing Answer/Validator calls.
- Keep light at `1 Query / 5 candidate URLs / 2 Read attempts / 1 round / 0 Repair`.
- Keep standard at `<=3 initial Queries including Direct / <=1 Repair Query / <=4 total Queries / 8 candidate URLs / 5 Read attempts / <=2 rounds`.
- Provider fallback is not a semantic Query, Read, round, or Repair. It never increments those caps.
- Risk and Freshness cannot change tier, Provider, Query count, Reader count, Repair eligibility, or timeout.
- Renderer remains a deterministic view. Validator can only remove or downgrade; neither may start Retrieval.
- Exact blind online questions are intentionally absent from this plan. They must be generated and sealed only after implementation, by a reviewer who did not implement the production changes.

## File responsibility map

| File | Final responsibility |
|---|---|
| `src/search/budget.py` | The only stage-budget table and pure `maximum_request_seconds(route)` calculation. No I/O and no business decisions. |
| `src/search/stage_runner.py` | One fixed-capacity timeout wrapper for non-Provider stages; timed-out work cannot mutate request state. |
| `src/search/models.py` | Closed immutable Query, batch, Read, Judge, Evidence, Trace, and response contracts. |
| `src/search/outcomes.py` | Pure Query aggregation, round-robin hit selection, Read summary, and final retrieval failure mapping. |
| `src/search/router.py` | One Request Analysis call plus deterministic validation of applicable retrieval complexity; no timeout sharing. |
| `src/search/planner.py` | Direct plus at most two supplemental initial Queries and one deterministic Repair Query. |
| `src/search/providers/base.py` | One explicitly named Provider attempt for one Query; no internal DDGS/Tavily fallback or reserve. |
| `src/search/providers/ddgs.py` | DDGS adapter only. |
| `src/search/providers/tavily.py` | Tavily adapter only. |
| `src/search/url_policy.py` | One public-HTTP URL safety, canonicalization, and duplicate policy shared before counting and before fetch. |
| `src/services/url_fetch_service.py` | Safe redirect-aware document fetch using `url_policy`; no second URL policy. |
| `src/search/extraction.py` | Convert one candidate URL into one closed Read outcome, retaining conservative snippet fallback. |
| `src/search/orchestrator.py` | Linear stage execution, independent stage timeouts, partial completion, one Repair, and post-Repair stop. |
| `src/search/evidence.py` | One Judge batch, candidate-isolated parsing, program-derived admission/Freshness/Source/Conflict/Sufficiency/Gap. |
| `src/search/policy.py` | Immutable Evidence/failure/risk to Answer/Render policy. No search calls. |
| `src/search/validation.py` | Bounded claim discovery and semantic validation; only filters/downgrades. |
| `src/search/renderer.py` | Deterministic QQ formatting only. |
| `src/chat/chat_service.py` | Stage-bounded Answer/Validator/Renderer wiring and final Trace finalization. |
| `tools/evaluate_search.py` | Closed body-free Trace validation and non-certifying safety gates. |
| `README.md`, `eval/search/README.md` | The final production contract and the separate external quality gate. |

---

### Task 1: Freeze the current baseline and failure reproductions

**Files:**
- Create: `docs/superpowers/baselines/2026-08-14-websearch-reliability-simplification.md`

- [ ] **Step 1: Record the clean baseline without encoding broken behavior as expected**

Create the baseline document with this exact content, updating only the test duration from the fresh command output:

```markdown
# WebSearch reliability simplification baseline

- Code baseline: `98685ec`
- Package-aware hermetic suite: `956 tests, 0 failures, 0 errors`
- Preserved caps: light `1/5/2/0/1`; standard `3 initial, 1 repair, 4 total, 8 URL, 5 Read, 2 rounds`.
- Preserved flow: DDGS first, Tavily fallback, Reader, one Judge batch per round, Evidence, Answer Policy, Validator, Renderer.
- Known failure reproduction A: one completed sibling Query can be discarded when another sibling consumes the shared request deadline.
- Known failure reproduction B: a Judge support row can be rejected by the retired `relevance` field even when `supported_topic_ids` is valid.
- This artifact is a diagnostic baseline, not a quality certificate. Live DDGS/Tavily quality remains unverified.
```

- [ ] **Step 2: Run the current stage-focused suites and record diagnostics**

Run:

```powershell
python -B -m unittest tests.test_search_providers tests.test_search_orchestrator tests.test_search_evidence -v
```

Expected: the existing suites pass. Record their exact counts in the baseline document. Do not add an expected-failure marker for the known bugs; Tasks 4 and 6 add the production regressions immediately before their fixes.

- [ ] **Step 3: Commit the green baseline artifact**

```powershell
git add docs/superpowers/baselines/2026-08-14-websearch-reliability-simplification.md
git commit -m "docs: record web search reliability baseline"
```

---

### Task 2: Add the single stage-budget policy and derived watchdog

**Files:**
- Create: `src/search/budget.py`
- Create: `src/search/stage_runner.py`
- Modify: `src/search/models.py:616-648`
- Create: `tests/test_search_budget.py`
- Create: `tests/test_search_stage_runner.py`
- Modify: `tests/test_search_simplification_baseline.py`

- [ ] **Step 1: Write RED tests for exact stage budgets and mechanical maxima**

Create `tests/test_search_budget.py`:

```python
import unittest
from dataclasses import replace

from src.search.budget import DEFAULT_SEARCH_BUDGET_POLICY
from src.search.models import SearchTier


class SearchBudgetPolicyTests(unittest.TestCase):
    def test_light_stages_are_independent_and_watchdog_is_derived(self):
        budget = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.LIGHT)
        self.assertEqual(3, budget.analysis_route_seconds)
        self.assertEqual(6, budget.initial_ddgs_seconds)
        self.assertEqual(6, budget.initial_tavily_seconds)
        self.assertEqual(4, budget.initial_reader_seconds)
        self.assertEqual(4, budget.initial_judge_seconds)
        self.assertEqual(0, budget.planner_seconds)
        self.assertEqual(0, budget.repair_ddgs_seconds)
        self.assertEqual(34, DEFAULT_SEARCH_BUDGET_POLICY.maximum_request_seconds(SearchTier.LIGHT))

    def test_standard_stages_and_watchdog_are_derived(self):
        budget = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.STANDARD)
        self.assertEqual((4, 8, 8, 6, 5), (
            budget.planner_seconds,
            budget.initial_ddgs_seconds,
            budget.initial_tavily_seconds,
            budget.initial_reader_seconds,
            budget.initial_judge_seconds,
        ))
        self.assertEqual((1, 2, 5, 5, 3, 4), (
            budget.gap_seconds,
            budget.repair_planner_seconds,
            budget.repair_ddgs_seconds,
            budget.repair_tavily_seconds,
            budget.repair_reader_seconds,
            budget.repair_judge_seconds,
        ))
        self.assertEqual(65, DEFAULT_SEARCH_BUDGET_POLICY.maximum_request_seconds(SearchTier.STANDARD))

    def test_watchdog_changes_when_a_stage_changes_without_a_second_constant(self):
        original = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.LIGHT)
        changed = replace(original, initial_reader_seconds=original.initial_reader_seconds + 2)
        self.assertEqual(
            DEFAULT_SEARCH_BUDGET_POLICY.maximum_for_budget(original) + 2,
            DEFAULT_SEARCH_BUDGET_POLICY.maximum_for_budget(changed),
        )
```

- [ ] **Step 2: Run the budget tests to verify RED**

Run:

```powershell
python -B -m unittest tests.test_search_budget -v
```

Expected: import failure because `src.search.budget` does not exist.

- [ ] **Step 3: Implement the closed pure budget policy**

Create `src/search/budget.py` with a frozen `RouteStageBudget` containing exactly these numeric fields:

```python
analysis_route_seconds
planner_seconds
initial_ddgs_seconds
initial_tavily_seconds
initial_reader_seconds
initial_judge_seconds
gap_seconds
repair_planner_seconds
repair_ddgs_seconds
repair_tavily_seconds
repair_reader_seconds
repair_judge_seconds
answer_seconds
validator_seconds
renderer_seconds
scheduling_margin_seconds
```

Validate every value as a finite non-negative `int` or `float`. Implement `SearchBudgetPolicy` as an immutable mapping and the only maximum calculation:

```python
def maximum_for_budget(self, budget: RouteStageBudget) -> float:
    return sum(getattr(budget, field.name) for field in fields(RouteStageBudget))

def maximum_request_seconds(self, route: SearchTier) -> float:
    return self.maximum_for_budget(self.for_route(route))
```

Use the frozen table from the specification and set the named scheduling margin to `2` seconds for both routes. Reject `SearchTier.SKIP` in `for_route` and return `0` from `maximum_request_seconds(SearchTier.SKIP)`.

- [ ] **Step 4: Remove timeout from the request-level data-cap contract**

Delete `hard_timeout_seconds` from `TierBudget` and keep only Query/URL/Read/Repair/Round caps. Update `DEFAULT_TIER_BUDGETS` without moving time from the retired 8/20-second shared deadline into any data cap.

Update `tests/test_search_simplification_baseline.py` to assert named data-cap fields instead of tuple position or timeout:

```python
self.assertEqual(1, light.max_initial_queries)
self.assertEqual(5, light.max_candidate_urls)
self.assertEqual(2, light.max_content_reads)
self.assertEqual(0, light.max_repair_queries)
self.assertEqual(3, standard.max_initial_queries)
self.assertEqual(8, standard.max_candidate_urls)
self.assertEqual(5, standard.max_content_reads)
self.assertEqual(1, standard.max_repair_queries)
```

- [ ] **Step 5: Add one fixed-capacity non-Provider stage runner**

Create `src/search/stage_runner.py` with one module-level `ThreadPoolExecutor(max_workers=8, thread_name_prefix="search-stage")` and:

```python
@dataclass(frozen=True)
class StageCallResult:
    completed: bool
    value: object | None


def run_stage(call, *, timeout_seconds: float) -> StageCallResult:
    if not callable(call):
        raise TypeError("call must be callable")
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    future = _STAGE_EXECUTOR.submit(call)
    try:
        return StageCallResult(True, future.result(timeout=float(timeout_seconds)))
    except FuturesTimeoutError:
        future.cancel()
        return StageCallResult(False, None)
```

The callable must return immutable/request-local output and must not mutate Trace or shared request collections. Exceptions other than timeout propagate to the stage owner, which maps them to its own closed failure state.

Add tests proving a completed value is retained, a queued call is cancelled, a running timeout returns promptly, and late completion cannot mutate caller-owned state because the callable receives no such state.

- [ ] **Step 6: Run focused tests and commit**

Run:

```powershell
python -B -m unittest tests.test_search_budget tests.test_search_stage_runner tests.test_search_models tests.test_search_simplification_baseline -v
python -B -m compileall -q src tests
git diff --check
```

Expected: all tests pass; compile and diff checks exit `0`.

```powershell
git add src/search/budget.py src/search/stage_runner.py src/search/models.py tests/test_search_budget.py tests/test_search_stage_runner.py tests/test_search_simplification_baseline.py
git commit -m "refactor: define independent search stage budgets"
```

---

### Task 3: Make Router complexity deterministic without expanding the lexicon

**Files:**
- Modify: `src/search/router.py:1229-1585`
- Modify: `tests/test_search_router.py`
- Modify: `tests/test_search_orchestrator.py`

- [ ] **Step 1: Add table-driven RED tests for model overclassification**

Add a test that feeds a complete valid analyzer payload whose model claims `multi_entity`, then verifies deterministic applicability:

```python
def test_model_complexity_codes_require_question_structure(self):
    cases = (
        ("Nova V4 Pro 正式版什么时候发布？", SearchTier.LIGHT),
        ("Aster CN赛区上一场谁赢了？", SearchTier.LIGHT),
        ("列出 Aster CN赛区季后赛完整赛程", SearchTier.STANDARD),
        ("比较 Nova Pro 和 Orbit Max 的续航", SearchTier.STANDARD),
        ("推荐两款适合出差的轻薄本并解释理由", SearchTier.STANDARD),
        ("请用两个独立来源核验 Nova 的发布日期", SearchTier.STANDARD),
    )
    for question, expected in cases:
        with self.subTest(question=question):
            analysis = analyze(question, advisor_payload(complexity_codes=["multi_entity"]))
            self.assertIs(expected, RetrievalBenefitRouter().decide(analysis.retrieval).route)
```

These are hermetic classification strings, not blind online acceptance questions.

- [ ] **Step 2: Verify RED and preserve existing Risk/Freshness assertions**

Run:

```powershell
python -B -m unittest tests.test_search_router -v
```

Expected: at least the first two new subcases fail as `STANDARD`. Existing risk and freshness metadata tests must remain present and green.

- [ ] **Step 3: Implement deterministic applicability as a pure merge**

Replace the current acceptance of arbitrary model complexity codes with the following bounded merge. It deliberately ignores a standalone model `MULTI_ENTITY` label; structural comparison/list/multi-fact forms already make such questions standard without guessing entity boundaries:

```python
def _applicable_complexity_codes(question, model_codes):
    deterministic = []
    lowered = question.casefold()
    comparison = any(marker in lowered for marker in (*_COMPARISON_MARKERS, "比较", "比一比", "区别"))
    recommendation = any(marker in lowered for marker in _RECOMMENDATION_MARKERS)
    complete_scope = any(marker in lowered for marker in (
        "完整赛程", "全部赛程", "完整日程", "时间表", "全部列出", "完整列表",
        "逐一", "分别", "complete schedule", "list all",
    ))
    multiple_fact_slots = len(re.findall(r"(?:什么时候|多少|哪天|谁|哪里|如何|what|when|where|who)", lowered)) >= 2
    ambiguity = any(marker in lowered for marker in (
        "可能指", "同名", "具体哪个", "哪个版本的", "ambiguous",
    ))
    if comparison:
        deterministic.append(RetrievalComplexityCode.COMPARISON)
    if recommendation:
        deterministic.append(RetrievalComplexityCode.RECOMMENDATION)
    if complete_scope or multiple_fact_slots:
        deterministic.append(RetrievalComplexityCode.MULTI_FACT)
    deterministic.extend(_explicit_source_complexity(question))
    if RetrievalComplexityCode.AMBIGUOUS_ENTITY in model_codes and ambiguity:
        deterministic.append(RetrievalComplexityCode.AMBIGUOUS_ENTITY)
    return _dedupe_complexity_codes(deterministic)
```

`_explicit_source_complexity` reuses the existing explicit multi-source and cross-verification markers and returns only those two codes. Treat `V4 Pro`, `CN赛区`, a dotted version, and an organization suffix as neutral tokens; do not add the example product names to production constants.

- [ ] **Step 4: Assert Risk and Freshness do not affect the route**

Add a same-retrieval-context test that varies `RiskContext` and `FreshnessContext` but receives the same route. Assert Request Analyzer remains one LLM call and Router remains a pure `decide(RetrievalContext)` call.

- [ ] **Step 5: Run focused suites and commit**

```powershell
python -B -m unittest tests.test_search_router tests.test_search_orchestrator -v
python -B -m compileall -q src tests
git diff --check
git add src/search/router.py tests/test_search_router.py tests/test_search_orchestrator.py
git commit -m "fix: validate retrieval complexity deterministically"
```

Expected: all focused tests pass and no Router source line reads `RiskContext`, `FreshnessContext`, or a safety warning code.

---

### Task 4: Separate DDGS and Tavily into independent Query batches

**Files:**
- Modify: `src/search/models.py:138-220,1160-1240,1411-1700`
- Create: `src/search/outcomes.py`
- Create: `src/search/url_policy.py`
- Modify: `src/search/providers/base.py`
- Modify: `src/services/url_fetch_service.py:138-180,228-335`
- Modify: `src/search/orchestrator.py:118-590`
- Modify: `tests/test_search_models.py`
- Modify: `tests/test_search_providers.py`
- Modify: `tests/test_search_orchestrator.py`
- Create: `tests/test_search_url_policy.py`

- [ ] **Step 1: Add RED model tests for Query and batch outcomes**

Define tests for these exact contracts:

```python
QueryOutcomeStatus = resolved | empty | timeout | error | unavailable
RetrievalBatchState = success | partial_success | all_failed
```

`QueryOutcome` must contain its internal `query: SearchQuery`, final status, safe hits, ordered Provider attempts, and an optional readiness failure limited to `PROVIDER_NOT_CONFIGURED`/`PROVIDER_UNAVAILABLE`. `query_index` and `round_kind` are derived properties. The raw Query exists only in request-local runtime state and is never serialized to Trace. A resolved outcome requires at least one hit; every non-resolved outcome requires zero hits. `QueryBatchResult` derives state and resolved/unresolved counts from a non-empty, unique, query-index-ordered tuple.

Run the new model tests and expect import failures before implementation.

- [ ] **Step 2: Implement pure aggregation in `src/search/outcomes.py`**

Implement exactly one aggregate function:

```python
def aggregate_query_outcomes(outcomes: Sequence[QueryOutcome]) -> QueryBatchResult:
    ordered = tuple(sorted(outcomes, key=lambda item: item.query_index))
    resolved = sum(item.status is QueryOutcomeStatus.RESOLVED for item in ordered)
    state = (
        RetrievalBatchState.SUCCESS if resolved == len(ordered)
        else RetrievalBatchState.PARTIAL_SUCCESS if resolved
        else RetrievalBatchState.ALL_FAILED
    )
    return QueryBatchResult(ordered, state)
```

Do not add a second aggregation implementation in Orchestrator or Trace.

- [ ] **Step 3: Add the one public-HTTP URL policy before Query resolution**

Create `src/search/url_policy.py` with `UrlDecision`, `evaluate_public_http_url`, and `canonicalize_public_http_url`. Move the exact scheme, hostname, private/local IP, DNS, default-port, fragment, and redirect rules from `url_fetch_service` into this module. Test public host, private literal, DNS-to-private, localhost, unsupported scheme, invalid port, default port, fragment removal, and Unicode/IDNA host normalization.

Provider hit filtering runs under the named Provider batch deadline: submit URL decisions to the same bounded batch executor, harvest allowed decisions before the stage deadline, and retain only canonical allowed hits. A Query is `RESOLVED` only when at least one allowed hit remains. An all-URL-invalid DDGS outcome is unresolved and therefore eligible for Tavily. `url_fetch_service` reuses the same policy before every redirect; it does not keep a second URL validator.

- [ ] **Step 4: Replace ProviderRegistry's implicit fallback with one named-provider call**

Add:

```python
def search_provider_with_attempts(
    self,
    provider_name: str,
    query: SearchQuery,
    *,
    tier: SearchTier,
    max_results: int,
    timeout_seconds: float,
    on_attempt_started=None,
    on_attempt_finished=None,
) -> ProviderSearchOutcome:
```

This method invokes only the named Provider. Delete `_TAVILY_FALLBACK_RESERVE_SECONDS`, `_primary_deadline`, and all deadline subtraction inside `ProviderRegistry`. The passed timeout is the Provider stage's full timeout. Keep the fixed executor, queued-call cancellation, attempt sealing, and non-cooperative adapter protection.

- [ ] **Step 5: Write RED provider and Orchestrator tests for fallback and sibling isolation**

Add tests asserting:

```python
# DDGS succeeds: Tavily is never called.
# DDGS empty/timeout/error/unavailable: only that Query enters Tavily.
# Tavily receives exactly budget.initial_tavily_seconds, not DDGS remainder.
# A queued call that never starts creates no ProviderAttempt.
# A started timed-out call creates exactly one timeout ProviderAttempt.
```

Use captured `timeout_seconds` arguments rather than wall-clock equality.

Add `OrchestratorIndependentStageRegressionTests.test_completed_sibling_query_survives_another_query_timeout`: Query 1 returns an allowed hit immediately, Query 2 times out, Reader/Judge support Query 1, and the final result retains Evidence with `retrieval_batch_state=partial_success` instead of `PROVIDER_TIMEOUT`.

- [ ] **Step 6: Implement two bounded concurrent batches in Orchestrator**

Implement `_run_provider_round(queries, *, ddgs_seconds, tavily_seconds, trace)` with this sequence:

```python
ddgs = _run_named_provider_batch("ddgs", all_queries, ddgs_seconds)
unresolved = tuple(outcome.query for outcome in ddgs if not outcome.resolved)
tavily = _run_named_provider_batch("tavily", unresolved, tavily_seconds)
final = _merge_provider_outcomes(ddgs, tavily)
return aggregate_query_outcomes(final)
```

Each named batch creates one bounded executor, submits all eligible Queries, harvests all completed futures at its own deadline, marks only unfinished Queries as timed out, and shuts down with `wait=False, cancel_futures=True`. Never check a request-wide remaining value.

- [ ] **Step 7: Preserve partial success and only fail on `all_failed`**

Replace the current `all(status in failures)` request shortcut with a single check of `batch.state`. `SUCCESS` and `PARTIAL_SUCCESS` both continue to Candidate selection. Update Trace from `QueryBatchResult`; do not infer batch state again from Provider failures.

Add focused assertions:

```python
self.assertIs(RetrievalBatchState.PARTIAL_SUCCESS, batch.state)
self.assertEqual((1,), tuple(outcome.query_index for outcome in batch.outcomes if outcome.resolved))
self.assertNotEqual(SearchFailureCode.PROVIDER_TIMEOUT, result.failure_code)
```

- [ ] **Step 8: Run focused suites and commit**

```powershell
python -B -m unittest tests.test_search_models tests.test_search_url_policy tests.test_search_providers tests.test_search_orchestrator -v
python -B -m compileall -q src tests
git diff --check
git add src/search/models.py src/search/outcomes.py src/search/url_policy.py src/search/providers/base.py src/services/url_fetch_service.py src/search/orchestrator.py tests/test_search_models.py tests/test_search_url_policy.py tests/test_search_providers.py tests/test_search_orchestrator.py
git commit -m "refactor: separate provider query batches"
```

Expected: the sibling-success RED from Task 1 is green, DDGS-first remains green, and no Provider fallback reserve symbol remains.

---

### Task 5: Centralize URL policy, round-robin candidates, and retain partial Reader completion

**Files:**
- Modify: `src/search/models.py:877-940,1210-1240`
- Modify: `src/search/outcomes.py`
- Modify: `src/search/extraction.py`
- Modify: `src/search/orchestrator.py:592-660`
- Modify: `src/search/evidence.py:468-505`
- Modify: `tests/test_search_extraction.py`
- Modify: `tests/test_search_orchestrator.py`

- [ ] **Step 1: Add RED tests for pre-cap URL safety and round-robin selection**

Test a batch where Query 1 returns many hits, Query 2 returns one hit, URLs contain duplicates/fragments/default ports, and one URL resolves to a private address. Assert the selected order alternates Query 1 then Query 2 before returning to Query 1, and unsafe/duplicate URLs do not consume the cap.

```python
self.assertEqual(
    ("q1-a", "q2-a", "q1-b"),
    tuple(hit.title for hit in select_candidate_hits(batch, max_urls=3, validator=fake_public)),
)
```

- [ ] **Step 2: Use the one URL policy for capped round-robin selection**

Import the Task 4 policy; do not define another URL canonicalizer or validator. Implement the pure selection shape:

```python
def round_robin_hits(batch: QueryBatchResult, *, max_urls: int) -> tuple:
    per_query = [deque(outcome.hits) for outcome in batch.outcomes if outcome.resolved]
    selected = []
    while per_query and len(selected) < max_urls:
        next_round = []
        for queue in per_query:
            if queue and len(selected) < max_urls:
                selected.append(queue.popleft())
            if queue:
                next_round.append(queue)
        per_query = next_round
    return tuple(selected)
```

The hits are already public/canonical/unique per Query outcome. Deduplicate across Query/provider outcomes by canonical URL before appending; a duplicate consumes no slot, and the loop continues to the next hit. Evidence imports `canonicalize_public_http_url`; delete its local `_canonical_url`.

- [ ] **Step 3: Add closed Read outcomes**

Add `ReadOutcomeStatus` with `readable`, `unreadable`, `timeout`, `unsafe_url`, `unsupported_type`, and a frozen `ReadOutcome` containing the original hit, optional Candidate, and `read_attempted` boolean. Validate that `readable` requires a Candidate with usable page/provider raw content. A failed-fetch snippet may accompany a non-readable status as a conservative Candidate but remains non-citable.

- [ ] **Step 4: Change SearchExtractor to return one ReadOutcome**

Rename the public operation to `read`. Keep a temporary test-only adapter only inside fixture helpers, not production. Map fetch results deterministically:

```python
success with excerpt       -> READABLE
timeout                    -> TIMEOUT
unsafe_url                 -> UNSAFE_URL
unsupported content/scheme -> UNSUPPORTED_TYPE
other/no content           -> UNREADABLE
```

Provider raw content is `READABLE`. A snippet after failed fetch is carried as Candidate metadata but its Evidence admission remains non-citable.

- [ ] **Step 5: Harvest Reader partial completion under its independent budget**

Run all selected hits in one bounded Reader executor using `initial_reader_seconds` or `repair_reader_seconds`. Retain every completed `READABLE` result and every conservative snippet Candidate. Mark only unfinished futures `TIMEOUT`. Do not turn one timeout into `provider_timeout` and do not call `remaining(global_deadline)`.

Add tests for one readable URL plus one timeout, all unreadable URLs, unsafe URLs before cap, and Read-attempt counts shared across initial/Repair.

- [ ] **Step 6: Run focused suites and commit**

```powershell
python -B -m unittest tests.test_search_url_policy tests.test_search_extraction tests.test_search_orchestrator tests.test_search_evidence -v
python -B -m compileall -q src tests
git diff --check
git add src/search/models.py src/search/outcomes.py src/search/extraction.py src/search/orchestrator.py src/search/evidence.py tests/test_search_extraction.py tests/test_search_orchestrator.py
git commit -m "refactor: preserve partial reader results"
```

Expected: URL cap and Read cap remain unchanged; `rg "def _canonical_url" src/search src/services` finds only the central policy implementation.

---

### Task 6: Make supported_topic_ids the sole Judge support contract

**Files:**
- Modify: `src/search/models.py:183-205,890-1160`
- Modify: `src/search/evidence.py:1-820,1020-1310`
- Modify: `src/search/__init__.py`
- Modify: `src/chat/chat_service.py:192-235`
- Modify: `src/search/validation.py:400-470`
- Modify: `tests/test_search_models.py`
- Modify: `tests/test_search_evidence.py`
- Modify: `tests/test_search_validation.py`
- Modify: `tests/test_chat_retrieval_flow.py`

- [ ] **Step 1: Add the support-only Judge RED and complete closed-row matrix**

First add this explicit RED to `EvidenceJudgeSchemaTests`:

```python
def test_supported_topic_ids_are_the_only_direct_support_signal(self):
    row = topic_judge_ok("C1", supported_topic_ids=("topic-1",))
    row.pop("relevance", None)
    parsed = self._judge(
        json.dumps({"candidates": {"C1": row}, "gap_hints": []}),
        candidate_count=1,
    )
    self.assertEqual(("topic-1",), tuple(parsed["C1"]["supported_topic_ids"]))
```

Run this method alone and expect the row to be absent under the retired schema. Then add tests for:

```text
complete supported row without relevance       -> retained
empty supported_topic_ids with empty freshness -> valid negative
missing expected row                            -> only that Candidate fails
malformed expected row                          -> only that Candidate fails
duplicate expected ID                           -> only that Candidate fails
unknown ID                                      -> discarded anomaly only
valid row plus invalid sibling                  -> valid row retained
damaged root / empty LLM / call exception       -> judge_unavailable
all complete negative rows                      -> judge_completed, no support
valid root with every expected row malformed    -> judge_completed, anomalies, no support
```

Assert the LLM mock is called once for the whole Candidate batch.

- [ ] **Step 2: Add closed Judge batch contracts**

Add:

```python
class JudgeBatchStatus(StrEnum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"

@dataclass(frozen=True)
class JudgeBatchResult:
    rows: Mapping[str, JudgeVerdict]
    status: JudgeBatchStatus
    anomaly_codes: tuple
    anomaly_count: int
```

`JudgeVerdict` contains exactly Candidate ID, `supported_topic_ids`, per-supported-topic Freshness, Source/Publisher fields, and optional complete Conflict triple. It has no relevance field and cannot carry Evidence State, Repair, or certainty. `JudgeBatchResult` also carries the existing closed `gap_hints` tuple because `entity_ambiguity` and `premise_mismatch` Repair reasons remain part of the frozen design; hints never admit Evidence and never create support edges.

- [ ] **Step 3: Replace the Prompt and parser atomically**

Remove `relevance` from `_JUDGE_SYSTEM_PROMPT`, `_VERDICT_KEYS`, `_parse_verdict`, fallback rows, and fixtures. State in the Prompt that every supplied Candidate must appear exactly once, but keep parser candidate-level fail-closed. Determine batch status as:

```python
root/call failure or empty LLM response -> UNAVAILABLE
valid closed root, even with zero valid Candidate rows -> COMPLETED
```

A closed root whose expected rows are all missing/malformed/duplicated is `COMPLETED` with Candidate anomalies and no support; it is not a call/root failure. A fully valid batch whose rows all have empty `supported_topic_ids` is also `COMPLETED`. This preserves candidate-level fail-closed semantics and keeps overall Judge failure distinct from Candidate output failure.

- [ ] **Step 4: Remove runtime relevance and label support compatibility immediately**

Delete `CandidateRelevance`, `_relevance_score`, `EvidenceItem.relevance_score`, `EvidenceItem.relevance_gate_passed`, and `EvidenceItem.supported_topics`. Replace the last field with opaque `supported_topic_ids` and validate it against the plan's known material IDs during bundle construction.

Admission begins only when a parsed Judge verdict explicitly names at least one material topic. Publisher/primary metadata cannot create an edge from an empty tuple.

- [ ] **Step 5: Migrate answer/validation payloads without a second support representation**

Build human-readable labels only at serialization time from `bundle.plan.required_topics`:

```python
topic_labels = {
    topic.topic_id: topic.label for topic in evidence.plan.required_topics
}
```

The grounded payload may show both ID and label, but stored Evidence uses only IDs. Validation checks missing IDs and projects labels locally when it must scan visible text.

- [ ] **Step 6: Prove Evidence admission parity and Judge distinction**

Keep all existing Freshness, Source Requirement, independence, Conflict, material-topic, and failed-fetch tests. Add explicit assertions that:

```python
primary + empty support          -> no admitted edge
stale named support              -> missing topic
two independence groups          -> corroborated
one independence group           -> source_quality_gap
judge unavailable                -> distinct from complete negative
```

- [ ] **Step 7: Run focused suites and commit**

```powershell
python -B -m unittest tests.test_search_models tests.test_search_evidence tests.test_search_validation tests.test_chat_retrieval_flow -v
python -B -m compileall -q src tests
git diff --check
rg -n "CandidateRelevance|relevance_gate_passed|relevance_score|\"relevance\"" src/search src/chat
```

Expected: tests pass and the final `rg` has no runtime hit.

```powershell
git add src/search/models.py src/search/evidence.py src/search/__init__.py src/chat/chat_service.py src/search/validation.py tests/test_search_models.py tests/test_search_evidence.py tests/test_search_validation.py tests/test_chat_retrieval_flow.py
git commit -m "refactor: make topic support the judge contract"
```

---

### Task 7: Integrate stage outcomes into Evidence, one Repair, and one failure mapping

**Files:**
- Modify: `src/search/outcomes.py`
- Modify: `src/search/models.py:1024-1160,1411-1700,1760-1848`
- Modify: `src/search/evidence.py:519-1345`
- Modify: `src/search/orchestrator.py:260-460,760-1140`
- Modify: `src/search/policy.py`
- Modify: `tests/test_search_evidence.py`
- Modify: `tests/test_search_orchestrator.py`
- Modify: `tests/test_search_policy.py`

- [ ] **Step 1: Add RED tests for the frozen failure matrix**

Add one table-driven test with these inputs and results:

```python
cases = (
    ("all_empty", None, None, SearchFailureCode.NO_RESULTS),
    ("all_not_configured", None, None, SearchFailureCode.PROVIDER_NOT_CONFIGURED),
    ("all_timeout", None, None, SearchFailureCode.PROVIDER_TIMEOUT),
    ("all_error_or_unavailable", None, None, SearchFailureCode.PROVIDER_UNAVAILABLE),
    ("partial_success", "all_unreadable", None, SearchFailureCode.CONTENT_UNREADABLE),
    ("success", "readable", "judge_unavailable", SearchFailureCode.JUDGE_UNAVAILABLE),
    ("success", "readable", EvidenceState.INSUFFICIENT, SearchFailureCode.INSUFFICIENT_EVIDENCE),
    ("success", "readable", EvidenceState.PARTIAL, SearchFailureCode.PARTIAL_EVIDENCE),
    ("success", "readable", EvidenceState.CONFLICTING, SearchFailureCode.SOURCE_CONFLICT),
    ("success", "readable", EvidenceState.SUFFICIENT, None),
)
```

Add `JUDGE_UNAVAILABLE` to `SearchFailureCode` and `JUDGE_UNAVAILABLE` to `DisclosureCode` rather than aliasing Judge failure to Provider or Evidence failure.

- [ ] **Step 2: Implement the single final retrieval failure function**

In `src/search/outcomes.py`, implement one `final_search_failure` function. It receives the full `QueryBatchResult`, Read summary, Judge status, and Evidence state in stage order. When the batch is `ALL_FAILED`, derive the Provider-layer code deterministically: all `EMPTY` → `NO_RESULTS`; all readiness failures with no invocation → `PROVIDER_NOT_CONFIGURED` or `PROVIDER_UNAVAILABLE`; otherwise any timeout → `PROVIDER_TIMEOUT`; remaining error/unavailable outcomes → `PROVIDER_UNAVAILABLE`. It must never overwrite an earlier completed success with an unrelated sibling failure. Delete `_failure_for_status`, `_failure_for_state`, and duplicate failure tables after callers migrate.

- [ ] **Step 3: Keep Evidence state entirely program-derived**

Use one priority calculation in Evidence:

```python
if material_conflicts:
    state = EvidenceState.CONFLICTING
elif all_material_topics_supported:
    state = EvidenceState.SUFFICIENT
elif meaningful_supported_material_subset:
    state = EvidenceState.PARTIAL
else:
    state = EvidenceState.INSUFFICIENT
```

Judge status and Judge metadata cannot assign this enum. Conflict removes only the affected topic edge. Preserve the constructor invariants that reject a bundle whose stored state disagrees with this calculation.

- [ ] **Step 4: Give every standard Repair stage its independent timeout**

Repair executes only when deterministic Gap says eligible and request-level Query/URL/Read capacity remains. Use `repair_planner_seconds`, `repair_ddgs_seconds`, `repair_tavily_seconds`, `repair_reader_seconds`, and `repair_judge_seconds`; do not reuse initial stage deadlines or reset caps.

After the post-Repair Judge, set `POST_REPAIR_STOP` unconditionally. No second Gap may dispatch a third round.

- [ ] **Step 5: Add Repair and failure isolation regressions**

Assert:

```text
light never Repairs
standard repairs at most once
Provider fallback is not Repair
initial Judge failure cannot Repair
post-Repair insufficient state cannot Repair again
one unreadable initial round may trigger one content_unreadable Repair
Repair counters are request-wide deltas
partial Provider and Reader success continue to Evidence
```

- [ ] **Step 6: Centralize disclosure mapping in Answer Policy**

Map each final `SearchFailureCode` to one disclosure code in one table. Add the fixed text for `DisclosureCode.JUDGE_UNAVAILABLE` in the existing deterministic disclosure-template table: `已找到网页内容，但暂时无法可靠判断其是否直接支持问题。` Successful ordinary search has no status/success disclosure. High-consequence warnings continue to come only from `risk_context` through Answer Policy; do not derive them from domain names in Renderer.

- [ ] **Step 7: Run focused suites and commit**

```powershell
python -B -m unittest tests.test_search_evidence tests.test_search_orchestrator tests.test_search_policy -v
python -B -m compileall -q src tests
git diff --check
git add src/search/outcomes.py src/search/models.py src/search/evidence.py src/search/orchestrator.py src/search/policy.py tests/test_search_evidence.py tests/test_search_orchestrator.py tests/test_search_policy.py
git commit -m "refactor: derive repair and failure from stage state"
```

---

### Task 8: Apply independent budgets to Analysis, Planner, Answer, Validator, and Renderer

**Files:**
- Modify: `src/search/router.py:1229-1265`
- Modify: `src/search/planner.py:389-740`
- Modify: `src/search/stage_runner.py`
- Modify: `src/search/orchestrator.py:141-760,914-950`
- Modify: `src/search/validation.py:90-155,590-725`
- Modify: `src/chat/chat_service.py:271-525`
- Modify: `tests/test_search_router.py`
- Modify: `tests/test_search_planner.py`
- Modify: `tests/test_search_orchestrator.py`
- Modify: `tests/test_search_validation.py`
- Modify: `tests/test_chat_retrieval_flow.py`

- [ ] **Step 1: Write RED timeout-capture tests for every non-Provider stage**

Use fake clients and a fake monotonic clock to capture exact arguments:

```python
analysis LLM receives 3
standard Planner receives 4
light Answer receives 4
standard Answer receives 4
Validator receives one 4-second local stage budget
Renderer is measured against 1 second
```

Also assert a slow Analysis does not reduce Planner, and a slow Provider does not reduce Reader, Judge, Answer, or Validator.

- [ ] **Step 2: Pass explicit timeout_seconds through Analysis and Planner**

Change `LLMRequestAnalyzer.analyze(request, *, timeout_seconds)` and its LLM call to use the full Analysis budget. Remove `deadline` from Planner public/private signatures and use only the full Planner stage timeout. Delete Planner's `deadline - time.monotonic()` calculation.

Run Analysis, Planner, Candidate Judge, Gap, Repair Planner, Answer, Validator, and Renderer through the Task 2 `run_stage` wrapper. Each worker receives immutable input and returns a complete local value; only the caller writes Trace after a completed `StageCallResult`. A timed-out worker must not receive or mutate Trace, Candidate lists, EvidenceBundle, or chat history.

- [ ] **Step 3: Remove the rolling request deadline from Orchestrator**

Delete `_remaining`, `_expired`, `_call_until_deadline`, `hard_deadline_exceeded`, and every `deadline` parameter used as a stage timeout. Each executor or dependency call receives the stage budget selected once from `DEFAULT_SEARCH_BUDGET_POLICY.for_route(route)`.

The derived watchdog is anchored to `response_started_at` only:

```python
watchdog_deadline = (
    response_started_at
    + DEFAULT_SEARCH_BUDGET_POLICY.maximum_request_seconds(route)
)
```

Check it only before starting a legal next stage and when finalizing. If it expires, preserve sealed Query/Read/Evidence results and return a conservative watchdog disclosure; never rewrite a partial success as Provider timeout.

- [ ] **Step 4: Bound Answer and Validator without adding calls**

Pass `answer_seconds` to the existing answer LLM. Add `timeout_seconds` to `validate_and_filter`; inside that one Validator stage, claim discovery receives the current local remainder and semantic verification receives the remainder after discovery. This is local Validator accounting, not cross-stage borrowing. `validate_and_filter` returns a frozen `ValidationStageResult(report, structural_latency_ms, semantic_latency_ms)` so its worker never writes Trace; `chat_service` copies those timings only after completion.

Do not add a retry. On timeout/unavailable, use the existing Validator fail-closed downgrade.

- [ ] **Step 5: Keep Renderer deterministic and independently bounded**

Renderer receives no Risk, Freshness, Evidence admission, or semantic objects beyond final `RenderState`. In `chat_service`, call `run_stage(lambda: render_search_reply(render_state, qq_limit=_qq_limit()), timeout_seconds=renderer_seconds)`. The worker receives only immutable `RenderState`, cannot mutate Trace, and is never retried. Add `RenderOutcome.TIMEOUT`; on timeout, return the fixed local text `回复格式化超时，请稍后重试。`, record that closed outcome with zero citations/sources, and finalize Trace. This is the existing Renderer stage made enforceable, not a new Agent or LLM call.

- [ ] **Step 6: Add watchdog derivation and early-stop tests**

Assert that light and standard watchdog values equal the pure budget sums, no separate `LIGHT_WATCHDOG`/`STANDARD_WATCHDOG` symbol exists, unused Tavily/Repair stages do not execute, and sufficient Evidence immediately enters Answer even though the maximum contains later legal stages.

- [ ] **Step 7: Run focused and full suites, then commit**

```powershell
python -B -m unittest tests.test_search_budget tests.test_search_router tests.test_search_planner tests.test_search_orchestrator tests.test_search_validation tests.test_chat_retrieval_flow -v
python -B -m unittest discover -s tests -t . -q
python -B -m compileall -q src tests
git diff --check
rg -n "hard_timeout_seconds|hard_deadline_exceeded|_remaining\(|_expired\(|deadline=" src/search src/chat
```

Expected: all tests pass; the final search finds no retired request-wide rolling-deadline path.

```powershell
git add src/search/router.py src/search/planner.py src/search/stage_runner.py src/search/orchestrator.py src/search/validation.py src/chat/chat_service.py tests/test_search_router.py tests/test_search_planner.py tests/test_search_orchestrator.py tests/test_search_validation.py tests/test_chat_retrieval_flow.py
git commit -m "refactor: isolate every web search stage budget"
```

---

### Task 9: Close Trace/evaluator contracts and delete transitional state

**Files:**
- Modify: `src/search/models.py:1411-1700`
- Modify: `src/search/orchestrator.py`
- Modify: `src/chat/chat_service.py`
- Modify: `tools/evaluate_search.py`
- Modify: `tests/test_search_models.py`
- Modify: `tests/test_search_orchestrator.py`
- Modify: `tests/test_chat_retrieval_flow.py`
- Modify: `tests/test_search_evaluation.py`
- Modify: `README.md`
- Modify: `eval/search/README.md`

- [ ] **Step 1: Write RED Trace tests for closed body-free stage state**

Require Trace to serialize only:

```text
stage status and latency
Query index/purpose/round/final Query outcome/hit count
Provider attempt provider/status/latency
batch state and resolved/unresolved counts
Read outcome counts
Judge batch status and anomaly codes/count
opaque supported/missing topic IDs and Evidence state
final failure/disclosure/validation/render states
```

Add negative tests for raw Query, URL, title, Candidate ID, page body, Judge text, and answer text in any metadata field.

- [ ] **Step 2: Make Query and stage states the only Trace source**

Serialize Query outcomes directly from `QueryBatchResult`, Read counts from the Read summary, and Judge status from `JudgeBatchResult`. Do not reconstruct them from ProviderAttempt multiplicity or failure strings. Retain exact Provider/Query tuple consistency checks in evaluator.

- [ ] **Step 3: Update evaluator closed schemas and invariants**

Require finalized production traces and applicable Answer/Validator/Render layers. Validate:

```text
stage latency <= stage budget plus documented scheduler tolerance
total response <= maximum_request_seconds(route) plus clock tolerance
fallback does not increment semantic_query_count
Repair implies standard, repair_count=1, rounds=2, POST_REPAIR_STOP
opaque topic IDs only
body-free fields only
Judge completed-negative differs from Judge unavailable
partial_success may certify when Evidence and rendering contracts pass
```

Do not modify evaluation JSONL data.

- [ ] **Step 4: Delete obsolete compatibility and duplicate branches**

Run and eliminate production hits for:

```text
CandidateRelevance
relevance_gate_passed
relevance_score
Provider fallback reserve
shared request remaining/deadline
hard_deadline_exceeded
legacy supported topic labels
duplicate failure mappings
retired trace query_id shape
```

If a hit is a migration test proving rejection, keep it only in the test and name it clearly.

- [ ] **Step 5: Update documentation truthfully**

Document independent stage budgets, derived `34s`/`65s` safety maxima, DDGS batch then Tavily unresolved-only batch, one Repair, candidate-level Judge isolation, and successful answer/source output without a status banner. State that these maxima are watchdog ceilings, not expected latency targets.

- [ ] **Step 6: Run evaluator, full suite, static checks, and commit**

```powershell
python -B -m unittest tests.test_search_models tests.test_search_orchestrator tests.test_chat_retrieval_flow tests.test_search_evaluation -v
python -B -m unittest discover -s tests -t . -q
python -B -m compileall -q src tests tools
git diff --check
git diff --name-only -- eval/search
```

Expected: all tests pass, compile/diff checks exit `0`, and the last command prints no JSONL path.

```powershell
git add src/search/models.py src/search/orchestrator.py src/chat/chat_service.py tools/evaluate_search.py tests/test_search_models.py tests/test_search_orchestrator.py tests/test_chat_retrieval_flow.py tests/test_search_evaluation.py README.md eval/search/README.md
git commit -m "chore: close web search runtime contracts"
```

---

### Task 10: Run adversarial, blind online, and independent acceptance gates

**Files:**
- Create: `tools/run_search_blind_acceptance.py`
- Create: `tests/test_search_blind_acceptance_runner.py`
- Create after execution: `docs/superpowers/reports/2026-08-14-websearch-reliability-acceptance.md`

- [ ] **Step 1: Create a runner that accepts a reviewer-owned sealed case file**

The runner accepts `--cases <absolute-json-path>` and `--output <absolute-json-path>`. The input schema is:

```json
{
  "sealed_at": "ISO-8601 timestamp",
  "cases": [
    {
      "case_id": "blind-01",
      "category": "current_single_fact",
      "question": "reviewer-supplied after implementation",
      "expected_route": "light",
      "fault_profile": "none"
    }
  ]
}
```

The literal example text above is a schema marker and must be rejected as an executable question. The runner rejects duplicate IDs, unknown categories, unknown fault profiles, files sealed before the final implementation commit, and questions found by exact normalized match in repository text.

- [ ] **Step 2: Add runner tests without adding real blind questions**

Test schema validation, duplicate rejection, pre-implementation timestamp rejection, repository-text collision rejection, body-free Trace capture, and JSON output. Use synthetic sentinel values explicitly rejected by the runner; do not check in any acceptance question.

- [ ] **Step 3: Have an independent reviewer generate and seal the cases**

After all implementation commits exist, a reviewer who did not implement Tasks 2–9 creates the external JSON in a temporary directory. It contains new entities and facts across these categories:

```text
current_single_fact
current_release_or_version
current_event_result
complete_schedule_or_list
official_announcement
multi_topic_comparison
ddgs_failure_tavily_success
sibling_query_partial_success
reader_partial_completion
judge_row_partial_failure
```

The implementer must not see the exact questions before the file is sealed. Fault profiles are deterministic test injection for the last four failure-isolation categories; normal current-fact categories use real providers.

- [ ] **Step 4: Run the blind gate with explicit online authorization**

Run from the final implementation commit:

```powershell
python -B tools/run_search_blind_acceptance.py --cases <reviewer-owned-absolute-path> --output <temporary-absolute-result-path>
```

Expected: every case finishes within the derived route maximum; ordinary successful replies contain an answer and actual source URLs but no status banner; partial/insufficient cases remain conservative. A failed case fails the gate and is reported; it is never relabeled as passing.

- [ ] **Step 5: Manually review Evidence and citations**

For every case, the independent reviewer records:

```text
whether each visible factual claim is supported by its cited Evidence
whether source identity and URL are real
whether route matches retrieval complexity
whether DDGS/Tavily/Reader/Judge stage state matches the injected or observed condition
whether warnings appear only when Answer Policy emitted a high-consequence warning
```

- [ ] **Step 6: Run final hermetic and immutable-data checks**

```powershell
python -B -m unittest discover -s tests -t . -q
python -B -m compileall -q src tests tools
git diff --check
git diff --name-only -- eval/search
git status --short
```

Expected: full suite green; no evaluation JSONL diff; only the acceptance report and runner changes are in scope, while `.tmp.driveupload/` and `websearch-simplification-report.md` remain untracked and untouched.

- [ ] **Step 7: Write the acceptance report and commit the harness/report**

The report must list exact commit, test counts, blind categories, per-case pass/fail, route, stage outcomes, elapsed time, manual citation verdict, reviewer identity, and the statement:

```text
Hermetic and blind-case acceptance proves conformance for this bounded set. It does not certify general internet retrieval quality.
```

```powershell
git add tools/run_search_blind_acceptance.py tests/test_search_blind_acceptance_runner.py docs/superpowers/reports/2026-08-14-websearch-reliability-acceptance.md
git commit -m "test: add blind web search acceptance gate"
```

---

## Final independent review checklist

Before merging, run two independent read-only reviews against the complete branch:

1. **Specification review:** map every section of `docs/superpowers/specs/2026-08-13-websearch-reliability-simplification-design.md` to a committed test and production path.
2. **Quality review:** search for duplicated deadline, aggregation, support, failure, and feedback-loop logic; reproduce new adversarial cases not written by implementers.
3. **Merge gate:** merge only with Critical/Important/Minor = `0/0/0`, a clean tracked worktree, fresh full hermetic green, accepted blind gate, unchanged evaluation JSONL, and no online quality claim broader than the reviewed cases.

## Spec coverage index

| Frozen requirement | Implemented by |
|---|---|
| Independent stage budgets and derived watchdog | Tasks 2, 4, 5, 8 |
| Risk/Freshness/explicit search do not raise tier | Task 3 plus retained Router tests |
| DDGS-first, unresolved-only Tavily, sibling partial success | Task 4 |
| Round-robin safe URL candidates and Reader partial harvest | Task 5 |
| Judge support-only schema and candidate isolation | Task 6 |
| Program-derived Freshness/Source/Conflict/Sufficiency | Tasks 6–7 |
| Standard-only single Repair and absolute post-Repair stop | Task 7 |
| One failure/disclosure mapping and pure Renderer | Tasks 7–8 |
| Body-free Trace and evaluator closure | Task 9 |
| No new Agent/LLM/tier/loop and cleanup of old state | Tasks 6, 8, 9 |
| New, unseen acceptance questions and manual citation review | Task 10 |
