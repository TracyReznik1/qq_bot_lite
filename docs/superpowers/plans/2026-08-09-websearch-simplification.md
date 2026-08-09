# QQ Bot WebSearch Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 WebSearch 精简为只含 `skip / light / standard` 的有限自适应证据搜索，并使检索复杂度、Evidence 时效要求和回答风险保持单向、互不反向控制。

**Architecture:** 保留 DDGS-first/Tavily fallback、Reader、Evidence admission、Claim/Citation、绝对 deadline 和并发封印；把现有一次早期分析拆成三个逻辑 context，但不增加 LLM 调用。Router 只消费 retrieval context，standard 在请求级20秒总预算内最多 Repair 一次；搜索结束后，一个小型纯函数 policy 模块把 immutable Evidence、freshness 和 risk 映射为 answer/render state，Validator 只能缩小输出，Renderer 只格式化。

**Tech Stack:** Python 3.11+、`dataclasses`、`StrEnum`、现有同步 LLM/Provider 抽象、`unittest`、PowerShell、Git。

---

## Global Constraints

- 不执行真实 DDGS/Tavily 请求；所有测试必须 hermetic。
- 不修改 `eval/search/*.jsonl`，不填写 `reviewed_by/reviewed_at`，不伪造独立预测或质量认证。
- 不新增 Risk/Freshness/Checklist/Gap/Repair Benefit/Pivot/Stop/Renderer LLM。
- 不新增第三轮、Repair-of-Repair、light→standard 运行时升级、Answer/Validator→Search 回环。
- standard 保持现有硬上限 `3 initial / 1 repair / 8 URL / 5 reads / 2 rounds / 20s`；不得吸收旧 deep 的 `5 / 15 / 8 / 40s` 能力。
- direct query 是首轮 semantic query 之一：standard 最多 `1 direct + 2 supplemental`，不是 `1 + 3`。
- light 保持 `1 query / 5 URL / 2 reads / 1 round / 8s / 0 repair`。
- Freshness 按 material topic 评估；迁移期 `Freshness.HIGH` 或任何“当前/高时效”信号不得升 tier 或增加预算。
- Risk 可以在请求早期识别，但不得进入 Router、Planner query 数量、Provider、Reader、Repair 或 deadline 决策。
- Evidence state 在 Search Pipeline 完成后不可变；Validator 只记录自己对草稿的处理结果。
- Renderer 不读取 Search Tier、Risk 或 Freshness，不执行语义判断。
- Provider fallback 是同一 semantic query 的容错，不增加 query/round/repair 计数。
- Trace 只记录 Query 元数据，禁止 raw query、用户正文、Evidence 正文、完整 URL 或 QQ 标识。
- 每个任务都先产生可解释的 RED，再做最小 GREEN；不得以修改测试期望掩盖生产回归。
- 每个任务完成后先做独立规格复核，再做代码质量复核；发现阻断项则在同一任务内新增回归并修复后再提交。

## File Responsibility Map

| 文件 | 最终职责 |
|---|---|
| `src/search/models.py` | 封闭枚举和不可变数据合同；最终不含 operational deep。 |
| `src/search/router.py` | 一次请求分析的严格解析和确定性复杂度路由；Risk/Freshness 只产出 context，不参与 tier。 |
| `src/search/planner.py` | direct + 最多2个 standard 补充 Query、最多3个 material topic、日期/版本约束、唯一 Repair Query。 |
| `src/search/providers/base.py` | DDGS-first、Tavily 条件 fallback、共享绝对 deadline；不含 deep reserve。 |
| `src/search/extraction.py` | 每 URL 一次正文获取动作和读取计数；保持现有安全边界。 |
| `src/search/evidence.py` | relevance admission、topic-level freshness、sufficiency、conflict 和 Gap 信号。 |
| `src/search/orchestrator.py` | 请求级预算、两轮上限、Gap 聚合、唯一 Repair、stop reason、body-free Trace。 |
| `src/search/policy.py` | 新增的唯一生产模块；无 I/O、无 LLM、无搜索，只把状态映射成 answer/render state。 |
| `src/search/validation.py` | 结构/语义校验，返回独立 validator status，只能过滤。 |
| `src/search/renderer.py` | 纯确定性 View：模板、Citation、Source、Conflict、Warning、QQ 分段。 |
| `src/chat/chat_service.py` | 串接 orchestrator→policy→draft→validator→render；不自行重算 Evidence/Risk。 |
| `tools/evaluate_search.py` | 新三档 schema、预算/Trace 不变量和非认证 gate。 |
| `eval/search/README.md`, `README.md` | 当前生产合同和外部门槛；旧 deep 文档标记为历史。 |

---

### Task 1: Freeze the Existing Reliability Baseline

**Files:**
- Create: `docs/superpowers/baselines/2026-08-09-websearch-simplification.md`
- Create: `tests/test_search_simplification_baseline.py`

- [ ] **Step 1: Run the package-aware baseline without network access**

Run:

```powershell
python -B -m unittest discover -s tests -t . -v
```

Expected: `840` tests pass on `8abaa8f`/documentation-only descendants; no external HTTP guard fires. If the count differs because another committed test was added, record the exact current count and require zero failures/errors.

- [ ] **Step 2: Add invariant characterization tests that survive the migration**

Create `tests/test_search_simplification_baseline.py` with tests named exactly:

```python
import inspect
import unittest
from unittest.mock import Mock

from src.search.models import (
    DEFAULT_TIER_BUDGETS,
    ProviderReadiness,
    ProviderStatus,
    SearchTier,
)
from src.search.orchestrator import _repair_allowed
from src.search.providers.base import ProviderRegistry
from tests.test_search_providers import _provider_result, query


class SearchSimplificationBaselineTests(unittest.TestCase):
    def test_light_and_standard_budget_caps_are_the_frozen_caps(self):
        light = DEFAULT_TIER_BUDGETS[SearchTier.LIGHT]
        standard = DEFAULT_TIER_BUDGETS[SearchTier.STANDARD]
        self.assertEqual((1, 5, 2, 0, 1, 1, 8), tuple(light.__dict__.values()))
        self.assertEqual((3, 8, 5, 1, 4, 2, 20), tuple(standard.__dict__.values()))

    def test_provider_registry_remains_ddgs_first_with_conditional_fallback(self):
        ddgs = Mock(name="ddgs")
        ddgs.name = "ddgs"
        ddgs.readiness.return_value = ProviderReadiness("ddgs", True, True, None)
        ddgs.search.return_value = _provider_result("ddgs")
        tavily = Mock(name="tavily")
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        tavily.search.return_value = _provider_result("tavily")
        registry = ProviderRegistry((tavily, ddgs))

        result = registry.search(query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=1.0)

        self.assertEqual("ddgs", result.provider)
        ddgs.search.assert_called_once()
        tavily.search.assert_not_called()

        ddgs.reset_mock()
        tavily.reset_mock()
        ddgs.search.return_value = _provider_result("ddgs", status=ProviderStatus.ERROR)
        registry.search(query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=1.0)
        ddgs.search.assert_called_once()
        tavily.search.assert_called_once()

    def test_repair_gate_is_a_program_function_not_an_llm_stage(self):
        self.assertFalse(inspect.iscoroutinefunction(_repair_allowed))
```

These tests deliberately characterize only retained reliability contracts; do not encode current deep behavior as an invariant.

- [ ] **Step 3: Run the new baseline test**

Run:

```powershell
python -B -m unittest tests.test_search_simplification_baseline -v
```

Expected: 3 tests pass.

- [ ] **Step 4: Record the baseline and intentional migration surface**

Create `docs/superpowers/baselines/2026-08-09-websearch-simplification.md` containing:

```markdown
# WebSearch Simplification Baseline

- Baseline implementation: `8abaa8f`
- Frozen caps: light `1/5/2/0/1/1/8`; standard `3/8/5/1/4/2/20`
- Retained: DDGS-first/Tavily conditional fallback, absolute deadline, Reader, Evidence relevance gate, Claim/Citation validation, body-free Trace.
- Intentionally changed: operational deep, risk/freshness tier floors, explicit-search standard floor, deep failure/validation branches.
- External gates are not certified: 140 owner-review rows are incomplete, two rows use illegal `potential_harm=medium`, and online mode remains not run.
```

- [ ] **Step 5: Commit the baseline**

```powershell
git add docs/superpowers/baselines/2026-08-09-websearch-simplification.md tests/test_search_simplification_baseline.py
git commit -m "test: freeze web search simplification baseline"
```

---

### Task 2: Stop Producing Operational `deep` Before Removing Its Schema

**Files:**
- Modify: `src/search/router.py:1061-1493`
- Modify: `src/search/orchestrator.py:127-370`
- Test: `tests/test_search_router.py`
- Test: `tests/test_search_orchestrator.py`

- [ ] **Step 1: Write RED tests for the migration boundary**

Add tests asserting:

```python
def test_production_router_never_emits_deep_during_migration(self):
    questions = (
        "北京今天有什么新闻？",
        "我发烧39度，该吃多少布洛芬？",
        "比较 Rust 和 Go 的并发模型并给出来源",
    )
    for question in questions:
        with self.subTest(question=question):
            self.assertIn(decide(question, NEUTRAL).route, {SearchTier.LIGHT, SearchTier.STANDARD})


def test_malformed_advisor_cannot_restore_deep(self):
    decision = decide("北京今天有什么新闻？", {"recommended_tier": "deep"})
    self.assertIsNot(decision.route, SearchTier.DEEP)
```

In `tests/test_search_orchestrator.py`, add this method to `OrchestratorLightTests`, which already initializes `self.module`:

```python
def test_production_trace_never_serializes_deep_route(self):
    provider = _FakeProvider(hits=[_hit()])
    orchestrator = self.module.SearchOrchestrator(
        router=_make_router(router_payload("deep")),
        planner=_make_planner(),
        judge=_FakeJudge(),
        providers=(provider,),
        extractor=_FakeExtractor(),
        clock=FakeClock(),
    )
    result = orchestrator.run(request("北京今天有什么新闻？"))
    self.assertNotEqual("deep", result.trace.to_log_dict()["route"])
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -B -m unittest tests.test_search_router tests.test_search_orchestrator -v
```

Expected: new tests fail because the current floor and advisor schema can still select `SearchTier.DEEP`.

- [ ] **Step 3: Cap the transitional production route without deleting models yet**

In `src/search/router.py`, make the accepted advisor tier closed to `light|standard`, change every program floor that currently returns `DEEP` to `STANDARD`, and use this transitional finalizer:

```python
def _operational_tier(tier: SearchTier | None) -> SearchTier | None:
    if tier is SearchTier.DEEP:
        return SearchTier.STANDARD
    return tier
```

Apply `_operational_tier()` to both the model recommendation and program floor before `max_tier()`. Keep risk/freshness metadata intact for now so this task changes resource behavior only. Do not change the standard budget.

This is a bounded migration exception: until Task 3 lands, former freshness/risk floors may temporarily reach standard, but never old deep budgets. Task 3 must delete those floors completely, and Task 10 must remove `_operational_tier()`; do not release or merge the branch at the Task 2 checkpoint.

- [ ] **Step 4: Add an orchestrator invariant at the production boundary**

Immediately after the production decision is returned:

```python
if decision.route is SearchTier.DEEP:
    raise RuntimeError("production router emitted retired deep route")
```

Tests that directly construct legacy deep records may continue until Task 10; production `run()` paths may not.

- [ ] **Step 5: Run focused and compatibility tests**

Run:

```powershell
python -B -m unittest tests.test_search_router tests.test_search_orchestrator tests.test_search_renderer tests.test_search_validation -v
```

Expected: all tests pass after migrating old deep expectations to transitional standard only where they exercise production routing. Tests for deep-only helper behavior remain unchanged until the behavior is removed in later tasks.

- [ ] **Step 6: Commit operational deep deactivation**

```powershell
git add src/search/router.py src/search/orchestrator.py tests/test_search_router.py tests/test_search_orchestrator.py tests/test_search_renderer.py tests/test_search_validation.py
git commit -m "refactor: stop emitting operational deep searches"
```

---

### Task 3: Split One Request Analysis into Retrieval, Freshness, and Risk Contexts

**Files:**
- Modify: `src/search/models.py:49-107,300-368,890-934`
- Modify: `src/search/router.py:1061-1410`
- Modify: `src/search/orchestrator.py:106-370,930-970`
- Test: `tests/test_search_models.py`
- Test: `tests/test_search_router.py`
- Test: `tests/test_search_orchestrator.py`

- [ ] **Step 1: Add RED data-contract tests**

Add tests for one analysis object and two independent routes:

```python
def test_request_analysis_keeps_retrieval_freshness_and_risk_separate(self):
    analysis = RequestAnalysis(
        retrieval=RetrievalContext(
            must_search=True,
            skip_reason=None,
            factuality=Factuality.FACTUAL,
            external_fact_required=True,
            complexity_codes=(),
            source_requirement=SourceRequirement.ANY_RELEVANT,
        ),
        freshness=FreshnessContext(
            requirement=FreshnessRequirement.CURRENT,
            as_of=None,
            date_from=None,
            date_to=None,
            version_constraint=None,
        ),
        risk=RiskContext(high_consequence=True, warning_required=True, fail_closed=True),
    )
    self.assertEqual((), analysis.retrieval.complexity_codes)
    self.assertTrue(analysis.risk.warning_required)


def test_same_retrieval_context_has_same_tier_for_different_risk_and_freshness(self):
    retrieval = RetrievalContext(
        must_search=True,
        skip_reason=None,
        factuality=Factuality.FACTUAL,
        external_fact_required=True,
        complexity_codes=(),
        source_requirement=SourceRequirement.ANY_RELEVANT,
    )
    stable = RequestAnalysis(retrieval, no_freshness(), no_risk())
    current_risky = RequestAnalysis(retrieval, current_freshness(), high_risk())
    self.assertEqual(
        self.router.decide(stable.retrieval),
        self.router.decide(current_risky.retrieval),
    )
```

Add a `SearchPipelineResult` test asserting `result.analysis` is preserved on skip, success, partial and failure results.

- [ ] **Step 2: Run RED**

Run:

```powershell
python -B -m unittest tests.test_search_models tests.test_search_router tests.test_search_orchestrator -v
```

Expected: imports or constructors fail because the context types and `SearchPipelineResult.analysis` do not exist.

- [ ] **Step 3: Add the minimal closed context contracts**

In `src/search/models.py`, add:

```python
class RetrievalComplexityCode(StrEnum):
    MULTI_FACT = "multi_fact"
    MULTI_ENTITY = "multi_entity"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    MULTI_SOURCE_REQUIRED = "multi_source_required"
    CROSS_VERIFICATION_REQUIRED = "cross_verification_required"
    AMBIGUOUS_ENTITY = "ambiguous_entity"


class SourceRequirement(StrEnum):
    ANY_RELEVANT = "any_relevant"
    INDEPENDENT_CORROBORATION = "independent_corroboration"


class FreshnessRequirement(StrEnum):
    NOT_REQUIRED = "not_required"
    CURRENT = "current"
    AS_OF = "as_of"
    WINDOW = "window"
    VERSION = "version"


@dataclass(frozen=True)
class RetrievalContext:
    must_search: bool
    skip_reason: SkipReason | None
    factuality: Factuality
    external_fact_required: bool
    complexity_codes: tuple[RetrievalComplexityCode, ...]
    source_requirement: SourceRequirement


@dataclass(frozen=True)
class FreshnessContext:
    requirement: FreshnessRequirement
    as_of: date | None
    date_from: date | None
    date_to: date | None
    version_constraint: str | None


@dataclass(frozen=True)
class RiskContext:
    high_consequence: bool
    warning_required: bool
    fail_closed: bool


@dataclass(frozen=True)
class RequestAnalysis:
    retrieval: RetrievalContext
    freshness: FreshnessContext
    risk: RiskContext
```

Validation rules:

```python
if self.date_from is not None and self.date_to is not None and self.date_from > self.date_to:
    raise ValueError("freshness date_from cannot exceed date_to")
if self.requirement is FreshnessRequirement.NOT_REQUIRED and any(
    value is not None for value in (self.as_of, self.date_from, self.date_to, self.version_constraint)
):
    raise ValueError("not_required freshness cannot carry a constraint")
if self.requirement is FreshnessRequirement.VERSION and not str(self.version_constraint or "").strip():
    raise ValueError("version freshness requires version_constraint")
if self.warning_required and not self.high_consequence:
    raise ValueError("warning requires a high-consequence request")
```

- [ ] **Step 4: Replace tier recommendation with one strict request-analysis payload**

Keep one LLM call. Rename `LLMRoutingAdvisor` to `LLMRequestAnalyzer`, temporarily export `LLMRoutingAdvisor = LLMRequestAnalyzer` for callers until Task 10, and parse exactly:

```json
{
  "factuality": "non_factual|factual|mixed|ambiguous",
  "external_fact_required": true,
  "complexity_codes": ["multi_fact|multi_entity|comparison|recommendation|multi_source_required|cross_verification_required|ambiguous_entity"],
  "source_requirement": "any_relevant|independent_corroboration",
  "freshness_requirement": "not_required|current|as_of|window|version",
  "as_of": null,
  "date_from": null,
  "date_to": null,
  "version_constraint": null,
  "high_consequence": false,
  "warning_required": false,
  "fail_closed": false
}
```

Do not include `recommended_tier`, model confidence, or a new model stage. Merge deterministic explicit-search, closed-task, relative-date and high-consequence rules into the same `RequestAnalysis` after strict parsing.

Deterministically extract an explicit user version token with the existing bounded version grammar. When present, override model freshness with `FreshnessRequirement.VERSION` and preserve the exact normalized token in `version_constraint`; malformed or missing model fields cannot erase it.

- [ ] **Step 5: Make Router a pure retrieval-context consumer**

Replace the public decision signature with:

```python
class RetrievalBenefitRouter:
    def decide(self, context: RetrievalContext) -> RetrievalDecision:
        if context.skip_reason is SkipReason.USER_FORBID_WEB:
            return _skip_from_context(context)
        if context.skip_reason is not None and not context.must_search:
            return _skip_from_context(context)
        standard_reasons = {
            RetrievalComplexityCode.MULTI_FACT,
            RetrievalComplexityCode.MULTI_ENTITY,
            RetrievalComplexityCode.COMPARISON,
            RetrievalComplexityCode.RECOMMENDATION,
            RetrievalComplexityCode.MULTI_SOURCE_REQUIRED,
            RetrievalComplexityCode.CROSS_VERIFICATION_REQUIRED,
            RetrievalComplexityCode.AMBIGUOUS_ENTITY,
        }
        route = (
            SearchTier.STANDARD
            if standard_reasons.intersection(context.complexity_codes)
            or context.source_requirement is SourceRequirement.INDEPENDENT_CORROBORATION
            else SearchTier.LIGHT
        )
        return _search_from_context(context, route)
```

Explicit `/search`, “请搜索” and “提供来源” set `must_search=True` but do not add a complexity code. “多个来源”“独立来源”“交叉核验” add the appropriate standard code.

Plain “请提供来源” leaves `source_requirement=ANY_RELEVANT`. Only explicit multiple/independent/cross-verification language sets `INDEPENDENT_CORROBORATION`, which also adds the standard complexity code. The model may not silently raise this requirement when the user did not request it.

`USER_FORBID_WEB` remains a hard no-provider constraint even when the same request also contains an explicit-search signal. In that conflict, preserve `must_search=True` plus `skip_reason=USER_FORBID_WEB`; the decision remains `skip`, and `requires_clarification` is derived from those two fields without a separate LLM decision.

- [ ] **Step 6: Wire analysis once in Orchestrator**

`SearchOrchestrator` receives `request_analyzer` and `router`. At the start of `run()`:

```python
analysis = self._request_analyzer.analyze(request)
decision = self._router.decide(analysis.retrieval)
```

Add `analysis: RequestAnalysis` to `SearchPipelineResult` and propagate the exact same immutable object through every return path. Start the retrieval deadline only after `decision` exists and before Planner begins.

- [ ] **Step 7: Prove Risk/Freshness cannot affect tier**

Add table tests covering:

```python
cases = (
    ("FDA 是什么机构？", False, "not_required", SearchTier.LIGHT),
    ("布洛芬说明书标注的成人单次剂量是多少？", True, "not_required", SearchTier.LIGHT),
    ("北京今天气温是多少？", False, "current", SearchTier.LIGHT),
    ("比较两款药的适应症和副作用", False, "not_required", SearchTier.STANDARD),
    ("根据我的症状比较两种药并建议今晚服用哪个", True, "current", SearchTier.STANDARD),
    ("请搜索光合作用定义并给出处", False, "not_required", SearchTier.LIGHT),
    ("请用两个独立来源核验光合作用定义", False, "not_required", SearchTier.STANDARD),
)
```

Assert route and context separately; the high-consequence case must still have `warning_required=True`.

- [ ] **Step 8: Run focused tests and commit**

Run:

```powershell
python -B -m unittest tests.test_search_models tests.test_search_router tests.test_search_orchestrator tests.test_chat_retrieval_flow -v
```

Expected: all pass and the request analyzer is called exactly once per request.

```powershell
git add src/search/models.py src/search/router.py src/search/orchestrator.py tests/test_search_models.py tests/test_search_router.py tests/test_search_orchestrator.py tests/test_chat_retrieval_flow.py
git commit -m "refactor: separate retrieval freshness and risk contexts"
```

---

### Task 4: Make Planner Enforce Direct-Query and Topic Contracts

**Files:**
- Modify: `src/search/models.py:402-458`
- Modify: `src/search/planner.py:296-880`
- Modify: `src/search/orchestrator.py:564-606`
- Test: `tests/test_search_models.py`
- Test: `tests/test_search_planner.py`
- Test: `tests/test_search_orchestrator.py`

- [ ] **Step 1: Write RED tests for direct-query accounting**

Add exact tests:

```python
def test_standard_direct_query_consumes_one_of_three_initial_slots(self):
    planner = planner_returning_three_supplemental_queries()
    plan = planner.plan(
        request("比较 Rust 和 Go"),
        standard_decision(),
        standard_retrieval_context(),
        no_freshness(),
    )
    self.assertLessEqual(len(plan.initial_queries), 3)
    self.assertIs(plan.initial_queries[0].purpose, QueryPurpose.DIRECT)
    self.assertLessEqual(sum(q.purpose is not QueryPurpose.DIRECT for q in plan.initial_queries), 2)


def test_standard_may_use_only_direct_query(self):
    plan = planner_returning_no_supplements().plan(
        request("Python 当前稳定版是什么？"),
        standard_decision(),
        standard_retrieval_context(),
        current_freshness(),
    )
    self.assertEqual(1, len(plan.initial_queries))


def test_freshness_changes_query_bounds_not_query_count(self):
    stable = self.plan_with(no_freshness())
    current = self.plan_with(current_freshness())
    self.assertEqual(len(stable.initial_queries), len(current.initial_queries))
    self.assertIsNotNone(current.initial_queries[0].date_from)


def test_model_cannot_drop_request_version_constraint_from_material_topic(self):
    plan = planner_declaring_topic_not_required().plan(
        request("比较 Python 3.13 的两个并发 API"),
        standard_decision(),
        standard_retrieval_context(),
        version_freshness("3.13"),
    )
    self.assertTrue(all(t.freshness_requirement is FreshnessRequirement.VERSION for t in plan.required_topics))
    self.assertTrue(all(t.version_constraint == "3.13" for t in plan.required_topics))
```

- [ ] **Step 2: Write RED model tests for bounded material topics**

Define the final topic record:

```python
@dataclass(frozen=True)
class RequiredTopic:
    topic_id: str
    label: str
    material: bool
    freshness_requirement: FreshnessRequirement
    date_from: date | None = None
    date_to: date | None = None
    version_constraint: str | None = None
    source_requirement: SourceRequirement = SourceRequirement.ANY_RELEVANT
```

Test that IDs are `topic-1` through `topic-3`, labels are non-empty, at most three are accepted, and `material=False` topics cannot be Repair targets.

Also reject a `SearchPlan` with no material topic. The deterministic light topic and the standard Planner fallback guarantee at least one material topic on every search route.

- [ ] **Step 3: Run RED**

Run:

```powershell
python -B -m unittest tests.test_search_planner tests.test_search_models -v
```

Expected: the current model-query parser can admit the direct query plus three model queries, and `RequiredTopic` does not exist.

- [ ] **Step 4: Change the standard Planner payload to supplements, not total queries**

The Planner prompt and strict parser must use:

```json
{
  "supplemental_queries": [
    {"purpose": "primary", "text": "Rust 官方并发模型", "target_topic_ids": ["topic-1"], "date_from": null, "date_to": null}
  ],
  "required_topics": [
    {"label": "Rust 与 Go 的并发模型差异", "material": true, "freshness_requirement": "not_required", "version_constraint": null, "source_requirement": "any_relevant"}
  ]
}
```

Cap supplemental queries before assigning IDs:

```python
supplement_limit = max(plan_budget.max_initial_queries - 1, 0)
supplements = tuple(parsed_supplements[:supplement_limit])
queries = _assign_initial_query_ids((direct_query, *supplements))
```

Deduplicate against the direct query before slicing. The model cannot produce or replace the direct query.

Add `query_index: int` and `target_topic_ids: tuple[str, ...]` to `SearchQuery`. The direct query targets every material topic. Each supplement must target one or more known material topic IDs; reject unknown/empty targets. These IDs let Reader failures map back to a deterministic Repair target without logging topic text.

Assign a request-wide `query_index` inside `_assign_initial_query_ids()` when final query slots are sealed: direct is 1, supplements are 2 and 3, and a Repair query is `len(plan.initial_queries) + 1`. Provider fallback entries reuse the same index as their semantic query; no dispatch or fallback path may renumber it.

The Planner copies `RetrievalContext.source_requirement` onto every material topic. Whenever `FreshnessContext.requirement` is not `NOT_REQUIRED`, it deterministically overwrites every material topic with that requirement, date bounds and `version_constraint`; model output can never lower, remove or replace the user-derived freshness/version constraint. This conservative rule applies to all material topics in the request. A model output may not lower or raise the user-derived source requirement.

- [ ] **Step 5: Apply Freshness to existing Query slots only**

Add:

```python
def _apply_freshness_bounds(
    query: SearchQuery,
    context: FreshnessContext,
) -> SearchQuery:
    if context.requirement is FreshnessRequirement.NOT_REQUIRED:
        return query
    return replace(query, date_from=context.date_from, date_to=context.date_to)
```

For “今天/昨天/as-of/explicit window”, derive exact dates deterministically before the Planner call. For vague “当前/最新”, use the existing current-date time window conservatively. Do not synthesize a separate time-bounded query and do not consume an extra slot.

For `FreshnessRequirement.VERSION`, keep the original direct query unchanged because it already contains the user's version scope; require every supplemental query to contain the exact normalized `version_constraint` token. Reject and drop a supplement that omits it instead of adding another Query.

- [ ] **Step 6: Keep light free of a Planner checklist call**

Light continues to bypass the model. It creates exactly one implicit material topic record for Evidence bookkeeping:

```python
RequiredTopic(
    topic_id="topic-1",
    label=_short_original(request.question),
    material=True,
    freshness_requirement=freshness.requirement,
    date_from=freshness.date_from,
    date_to=freshness.date_to,
    version_constraint=freshness.version_constraint,
    source_requirement=retrieval.source_requirement,
)
```

This is not an additional Checklist stage or LLM call.

- [ ] **Step 7: Update Orchestrator planner calls and counters**

Use the exact signature `plan(request, decision, retrieval_context, freshness_context)`. Pass `analysis.retrieval` and `analysis.freshness` into both normal and degraded planner paths; never pass `analysis.risk` into Planner. Assert:

```python
len(plan.initial_queries) <= plan.budget.max_initial_queries
plan.initial_queries[0].purpose is QueryPurpose.DIRECT
```

The trace `initial_query_count` must equal the actual number dispatched, including direct.

- [ ] **Step 8: Run focused suites and commit**

Run:

```powershell
python -B -m unittest tests.test_search_planner tests.test_search_models tests.test_search_orchestrator -v
```

Expected: all pass; every standard fixture executes at most three first-round semantic queries.

```powershell
git add src/search/models.py src/search/planner.py src/search/orchestrator.py tests/test_search_models.py tests/test_search_planner.py tests/test_search_orchestrator.py
git commit -m "refactor: bound direct queries and material topics"
```

---

### Task 5: Compute Topic-Level Freshness and Deterministic Sufficiency

**Files:**
- Modify: `src/search/models.py:480-590`
- Modify: `src/search/evidence.py:87-228,287-528,744-815`
- Test: `tests/test_search_models.py`
- Test: `tests/test_search_evidence.py`

- [ ] **Step 1: Write RED topic-freshness tests**

Add the approved mixed-topic example:

```python
def test_two_fresh_topics_and_one_stale_topic_are_partial(self):
    plan = topic_plan(
        topic("topic-1", "A", FreshnessRequirement.CURRENT),
        topic("topic-2", "B", FreshnessRequirement.CURRENT),
        topic("topic-3", "C", FreshnessRequirement.CURRENT),
    )
    bundle = assemble(
        plan,
        evidence_for("topic-1", published="2026-08-09"),
        evidence_for("topic-2", published="2026-08-09"),
        evidence_for("topic-3", published="2025-01-01"),
    )
    self.assertEqual(EvidenceState.PARTIAL, bundle.evidence_state)
    self.assertEqual(("topic-1", "topic-2"), bundle.supported_topic_ids)
    self.assertEqual(("topic-3",), bundle.missing_topic_ids)
    self.assertEqual(FreshnessEligibility.STALE, bundle.topic_assessments[2].freshness)
```

Add negative tests for unknown timestamp, irrelevant edge topic, conflict precedence, and `SUFFICIENT + stale/unknown` constructor rejection.

Conflict precedence applies only to material topic conflicts: two disagreeing Evidence items that support no material required topic cannot change the bundle to `CONFLICTING`.

Add source-requirement and version cases:

```python
def test_one_relevant_source_can_satisfy_default_topic(self):
    self.assertEqual(EvidenceState.SUFFICIENT, assemble(default_topic(), one_relevant_source()).evidence_state)


def test_one_source_cannot_satisfy_explicit_independent_corroboration(self):
    bundle = assemble(independent_topic(), one_relevant_source())
    self.assertEqual(EvidenceState.INSUFFICIENT, bundle.evidence_state)
    self.assertIn("topic-1", bundle.missing_topic_ids)


def test_two_independent_groups_satisfy_independent_corroboration(self):
    bundle = assemble(independent_topic(), source(group="g1"), source(group="g2"))
    self.assertEqual(EvidenceState.SUFFICIENT, bundle.evidence_state)
```

For version-scoped topics, add matching, mismatching and absent-version tests expecting `SATISFIED`, `STALE` and `UNKNOWN` respectively.

- [ ] **Step 2: Run RED**

Run:

```powershell
python -B -m unittest tests.test_search_evidence tests.test_search_models -v
```

Expected: missing enums/fields and the current `_freshness_state()` always returning `Freshness.NONE` cause failures.

- [ ] **Step 3: Add the final topic assessment contracts**

In `src/search/models.py`:

```python
class FreshnessEligibility(StrEnum):
    NOT_REQUIRED = "not_required"
    SATISFIED = "satisfied"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TopicAssessment:
    topic_id: str
    freshness: FreshnessEligibility
    supporting_evidence_ids: tuple[str, ...]
```


Add these fields to the existing frozen `EvidenceBundle` dataclass:

```python
    topic_assessments: tuple[TopicAssessment, ...]
    supported_topic_ids: tuple[str, ...]
    missing_topic_ids: tuple[str, ...]
```

Reject a bundle when `evidence_state is SUFFICIENT` and any material topic assessment is `STALE` or `UNKNOWN`.

- [ ] **Step 4: Change Evidence Judge topic output to closed topic IDs**

The judge receives topic IDs and labels, and each candidate verdict uses:

```json
{
  "candidates": {
    "C1": {
      "candidate_id": "C1",
      "relevance": "direct",
      "supported_topic_ids": ["topic-1"],
      "freshness_by_topic": {"topic-1": "satisfied"}
    }
  },
  "gap_hints": []
}
```

Reject unknown topic IDs, unknown freshness strings, missing candidate IDs, partial rows and non-direct relevance. Keep relevance as the admission gate before any first-party ranking.

- [ ] **Step 5: Implement deterministic freshness overrides**

Use this precedence for every material topic/evidence pair:

```python
def _freshness_for_topic(topic, item, judged_status):
    if topic.freshness_requirement is FreshnessRequirement.NOT_REQUIRED:
        return FreshnessEligibility.NOT_REQUIRED
    published = item.published_at.date() if item.published_at is not None else None
    if topic.date_from is not None or topic.date_to is not None:
        if published is None:
            return FreshnessEligibility.UNKNOWN
        if topic.date_from is not None and published < topic.date_from:
            return FreshnessEligibility.STALE
        if topic.date_to is not None and published > topic.date_to:
            return FreshnessEligibility.UNKNOWN
        return FreshnessEligibility.SATISFIED
    if topic.freshness_requirement is FreshnessRequirement.VERSION:
        corpus = f"{item.title}\n{item.excerpt}".casefold()
        required = str(topic.version_constraint or "").strip().casefold()
        if required and required in corpus:
            return FreshnessEligibility.SATISFIED
        if _contains_explicit_version_token(corpus):
            return FreshnessEligibility.STALE
        return FreshnessEligibility.UNKNOWN
    if judged_status is FreshnessEligibility.SATISFIED:
        return judged_status
    if judged_status is FreshnessEligibility.STALE:
        return judged_status
    return FreshnessEligibility.UNKNOWN
```

Only `NOT_REQUIRED` and `SATISFIED` evidence may support a topic.

`_contains_explicit_version_token()` uses the existing bounded version-token grammar and never guesses semantic equivalence between different versions. The Evidence Judge may label a version, but deterministic literal mismatch/absence overrides any model claim of `satisfied`.

- [ ] **Step 6: Implement the approved state priority**

Replace tier/risk-based strong-support branches with topic/source requirements and compute:

```python
if important_conflicts:
    state = EvidenceState.CONFLICTING
elif all(material_topic_supported(topic) for topic in plan.required_topics):
    state = EvidenceState.SUFFICIENT
elif any(material_topic_supported(topic) for topic in plan.required_topics):
    state = EvidenceState.PARTIAL
else:
    state = EvidenceState.INSUFFICIENT
```

Remove `_requires_strong_support()` checks based on deep, freshness tier or risk. Explicit independent-source requirements remain a topic/source sufficiency condition.

For `ANY_RELEVANT`, one citable freshness-eligible Evidence item is enough. For `INDEPENDENT_CORROBORATION`, require at least two eligible Evidence items whose `independence_group` values are both non-empty and different. First-party identity only ranks already-relevant candidates and does not bypass either rule.

- [ ] **Step 7: Run Evidence suites and commit**

Run:

```powershell
python -B -m unittest tests.test_search_evidence tests.test_search_models tests.test_search_validation -v
```

Expected: all pass; risk changes do not change admitted Evidence, and stale/unknown topics cannot produce SUFFICIENT.

```powershell
git add src/search/models.py src/search/evidence.py tests/test_search_models.py tests/test_search_evidence.py tests/test_search_validation.py
git commit -m "feat: evaluate evidence freshness by material topic"
```

---

### Task 6: Unify Standard Repair and Enforce the Post-Repair Stop

**Files:**
- Modify: `src/search/models.py:31-34,443-589,756-886`
- Modify: `src/search/evidence.py:373-395`
- Modify: `src/search/planner.py:510-583`
- Modify: `src/search/orchestrator.py:127-370,371-680,796-891`
- Test: `tests/test_search_models.py`
- Test: `tests/test_search_planner.py`
- Test: `tests/test_search_evidence.py`
- Test: `tests/test_search_orchestrator.py`

- [ ] **Step 1: Write RED tests for all seven repair reasons**

Use a table:

```python
cases = (
    (RepairReasonCode.MISSING_TOPIC, "topic-2"),
    (RepairReasonCode.STALE_EVIDENCE, "topic-2"),
    (RepairReasonCode.SOURCE_CONFLICT, "topic-2"),
    (RepairReasonCode.ENTITY_AMBIGUITY, "topic-2"),
    (RepairReasonCode.PREMISE_MISMATCH, "topic-2"),
    (RepairReasonCode.SOURCE_QUALITY_GAP, "topic-2"),
    (RepairReasonCode.CONTENT_UNREADABLE, "topic-2"),
)
```

For each case, assert standard can create at most one distinct repair query when budget remains. Add symmetric negatives for light, no target, no distinct query, exhausted URL/read/query/time budget, and already-repaired state.

- [ ] **Step 2: Add RED end-to-end budget tests**

Assert:

```python
self.assertLessEqual(trace.semantic_query_count, 4)
self.assertLessEqual(trace.initial_query_count, 3)
self.assertLessEqual(trace.repair_query_count, 1)
self.assertLessEqual(trace.candidate_url_count, 8)
self.assertLessEqual(trace.content_read_count, 5)
self.assertLessEqual(trace.retrieval_round_count, 2)
```

Also assert direct counts as one, Provider fallback does not increment semantic query count, and a second gap after repair records `post_repair_stop` without dispatch.

Add the full-cap case `1 direct + 2 supplemental + 1 repair`: request-wide query indexes must be `(1, 2, 3, 4)` and the derived semantic query count must equal 4. DDGS and Tavily attempts for index 1 still count as one semantic query.

Add a Reader accounting case where one URL fetch fails and the same candidate falls back to provider content; assert `content_read_count == 1`, not 0 or 2.

- [ ] **Step 3: Run RED**

Run:

```powershell
python -B -m unittest tests.test_search_planner tests.test_search_evidence tests.test_search_orchestrator tests.test_search_models -v
```

Expected: only missing/conflict strings currently exist; target topics and stop reason are absent.

- [ ] **Step 4: Add closed Repair and stop contracts**

In `models.py`:

```python
class RepairReasonCode(StrEnum):
    MISSING_TOPIC = "missing_topic"
    STALE_EVIDENCE = "stale_evidence"
    SOURCE_CONFLICT = "source_conflict"
    ENTITY_AMBIGUITY = "entity_ambiguity"
    PREMISE_MISMATCH = "premise_mismatch"
    SOURCE_QUALITY_GAP = "source_quality_gap"
    CONTENT_UNREADABLE = "content_unreadable"


class RetrievalStopReason(StrEnum):
    EVIDENCE_SUFFICIENT = "evidence_sufficient"
    NO_REPAIR_BENEFIT = "no_repair_benefit"
    BUDGET_EXHAUSTED = "budget_exhausted"
    POST_REPAIR_STOP = "post_repair_stop"


@dataclass(frozen=True)
class EvidenceGapAnalysis:
    missing_topic_ids: tuple[str, ...]
    conflict_group_ids: tuple[str, ...]
    repair_eligible: bool
    repair_reason_codes: tuple[RepairReasonCode, ...]
    repair_target_topic_ids: tuple[str, ...]


@dataclass(frozen=True)
class RepairPlan:
    triggered: bool
    reason_codes: tuple[RepairReasonCode, ...]
    target_topic_ids: tuple[str, ...]
    repair_query: SearchQuery | None
    query_redaction_codes: tuple[RedactionCode, ...] = ()
```

Require non-empty reasons/targets exactly when `triggered=True`.

Extend the existing Evidence Judge result with closed `gap_reason_codes` and `gap_target_topic_ids`; reject unknown reason codes or topic IDs. These are metadata from the existing Judge call, not a new call. Orchestrator only considers a hint when the referenced material topic remains missing or conflicting after first-round Evidence assembly. Do not add parallel `planning_gap_*` fields to `SearchPlan`.

Use this exact extension to the existing Judge response:

```json
{
  "candidates": {
    "C1": {"candidate_id": "C1", "relevance": "direct", "supported_topic_ids": ["topic-1"], "freshness_by_topic": {"topic-1": "satisfied"}}
  },
  "gap_hints": [
    {"reason_code": "entity_ambiguity", "target_topic_id": "topic-1"}
  ]
}
```

Only `entity_ambiguity` and `premise_mismatch` may originate from `gap_hints`. Every hint must reference a known material topic that remains unsupported after assembly; otherwise discard it.

- [ ] **Step 5: Aggregate gaps without a new Agent**

Use this producer/target table; no reason may enter the plan by any other path:

| reason | producer | target rule |
|---|---|---|
| `missing_topic` | EvidenceAssembler | every material `missing_topic_id` not already assigned a more specific reason |
| `stale_evidence` | topic assessment | material topic with final freshness `STALE` or `UNKNOWN` because required current/version evidence is not eligible |
| `source_conflict` | Evidence conflict analysis | material topic ID attached to an unresolved structured conflict |
| `entity_ambiguity` | strict Judge `gap_hints` | hinted known material topic, only while unsupported |
| `premise_mismatch` | strict Judge `gap_hints` | hinted known material topic, only while unsupported |
| `source_quality_gap` | Evidence source sufficiency | material topic with relevant Evidence but unmet `INDEPENDENT_CORROBORATION` |
| `content_unreadable` | Orchestrator from Reader results | material target IDs of a query whose candidates all failed readable-content acquisition |

`SearchQuery.target_topic_ids` supplies the Reader→topic mapping. Orchestrator deduplicates reasons and targets in declaration/topic order. If no material target survives, `repair_eligible=False` and stop reason is `NO_REPAIR_BENEFIT`.

Do not create a Gap LLM or a new state machine.

Add an end-to-end test whose fake Reader makes all candidates for `topic-2` unreadable. It must produce `CONTENT_UNREADABLE`, target only `topic-2`, dispatch one `repair-1`, and stop after the repaired Evidence evaluation. Add a second fake Judge test that emits `premise_mismatch` for an unknown topic and assert no Repair occurs.

- [ ] **Step 6: Build exactly one constraint-preserving repair query**

Implement in `SearchPlanner.plan_repair()`:

```python
def plan_repair(self, plan, gap, prior_fingerprints):
    if not gap.repair_eligible or not gap.repair_target_topic_ids:
        return RepairPlan(False, (), (), None)
    target = _topic_by_id(plan, gap.repair_target_topic_ids[0])
    text = _repair_text(plan.original_question, target, gap.repair_reason_codes)
    cleaned = self._clean_repair_text(text, plan.original_question)
    if _query_fingerprint(cleaned) in prior_fingerprints:
        return RepairPlan(False, (), (), None)
    query = SearchQuery(
        query_id="repair-1",
        query_index=len(plan.initial_queries) + 1,
        round_kind=SearchRoundKind.REPAIR,
        purpose=QueryPurpose.REPAIR,
        text=cleaned,
        target_topic_ids=(target.topic_id,),
        date_from=target.date_from,
        date_to=target.date_to,
        include_domains=(),
        exclude_domains=(),
    )
    return RepairPlan(True, gap.repair_reason_codes, gap.repair_target_topic_ids, query)
```

The Repair execution path preserves this assigned `query_index`; it never calls the initial-query ID/index allocator again.

`_repair_text()` must include the still-valid original entity/version/region/scope anchor and a reason-specific target phrase; it must not discard the direct-query constraints.

- [ ] **Step 7: Make budget and stop checks deterministic**

Before repair dispatch, require all of:

```python
decision.route is SearchTier.STANDARD
and not trace.repair_used
and trace.semantic_query_count < 4
and trace.candidate_url_count < 8
and trace.content_read_count < 5
and trace.retrieval_round_count < 2
and remaining_seconds > 0
and repair_plan.triggered
```

After the second Evidence assembly, set `RetrievalStopReason.POST_REPAIR_STOP` unconditionally and never call `analyze_gap()` to trigger another dispatch.

- [ ] **Step 8: Replace raw Query objects in Trace with metadata records**

Add:

```python
@dataclass(frozen=True)
class QueryTraceEntry:
    query_index: int
    purpose: QueryPurpose
    round_kind: SearchRoundKind
    provider: str
    status: ProviderStatus
    latency_ms: int | float
```

`SearchTrace.executed_queries` becomes `tuple[QueryTraceEntry, ...]`; remove raw `SearchQuery` from Trace. The same semantic query may have DDGS and Tavily entries, but semantic count is the number of unique `query_index` values. Add read-only `semantic_query_count` and `repair_query_count` properties derived from unique query indexes; do not store or log query text.

The `query_index` is monotonically increasing across the entire request, not reset by round. Repair uses `len(plan.initial_queries) + 1`; `(round_kind, ordinal)` is not used as a substitute key.

- [ ] **Step 9: Run repair/deadline/provider regression suites**

Run:

```powershell
python -B -m unittest tests.test_search_orchestrator tests.test_search_planner tests.test_search_evidence tests.test_search_providers tests.test_search_extraction tests.test_search_models -v
```

Expected: all pass, including fixed-pool race sealing and no late Trace mutation.

- [ ] **Step 10: Commit the unified repair contract**

```powershell
git add src/search/models.py src/search/evidence.py src/search/planner.py src/search/orchestrator.py tests/test_search_models.py tests/test_search_planner.py tests/test_search_evidence.py tests/test_search_orchestrator.py
git commit -m "refactor: unify bounded standard repair"
```

---

### Task 7: Add a Pure Answer Policy over Immutable Search State

**Files:**
- Create: `src/search/policy.py`
- Modify: `src/search/models.py:168-186,688-753,890-934`
- Modify: `src/chat/chat_service.py:273-453`
- Test: `tests/test_search_policy.py`
- Test: `tests/test_chat_retrieval_flow.py`

- [ ] **Step 1: Write the RED failure-matrix tests**

Create `tests/test_search_policy.py` with a 4×2 matrix over Evidence state and `warning_required`:

```python
EXPECTED = {
    EvidenceState.SUFFICIENT: (AnswerCertainty.VERIFIED, AllowedClaimScope.ALL_SUPPORTED),
    EvidenceState.PARTIAL: (AnswerCertainty.LIMITED, AllowedClaimScope.SUPPORTED_SUBSET),
    EvidenceState.CONFLICTING: (
        AnswerCertainty.CONFLICTING,
        AllowedClaimScope.SUPPORTED_SUBSET_WITH_CONFLICTS,
    ),
    EvidenceState.INSUFFICIENT: (
        AnswerCertainty.UNVERIFIED,
        AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS,
    ),
}
```

For conflict covering every core topic, expect `CONFLICT_DESCRIPTION_ONLY`. For warning-required cases expect exactly `(WarningCode.HIGH_CONSEQUENCE,)`; ordinary cases expect no warning.

- [ ] **Step 2: Add RED chat tests for no memory fallback**

Test Provider unavailable, timeout, no result and unreadable cases. For external factual requests with INSUFFICIENT Evidence, assert the answer LLM is not called to invent facts, the result contains a disclosure code, and high-consequence output contains no dose/action/legal conclusion.

- [ ] **Step 3: Run RED**

Run:

```powershell
python -B -m unittest tests.test_search_policy tests.test_chat_retrieval_flow -v
```

Expected: the policy module/types do not exist and current chat failure behavior still branches on deep.

- [ ] **Step 4: Add minimal answer-state contracts**

In `models.py`:

```python
class AnswerCertainty(StrEnum):
    VERIFIED = "verified"
    LIMITED = "limited"
    CONFLICTING = "conflicting"
    UNVERIFIED = "unverified"


class AnswerGenerationMode(StrEnum):
    PLAIN = "plain"
    GROUNDED = "grounded"
    FIXED = "fixed"


class AllowedClaimScope(StrEnum):
    ALL_SUPPORTED = "all_supported"
    SUPPORTED_SUBSET = "supported_subset"
    SUPPORTED_SUBSET_WITH_CONFLICTS = "supported_subset_with_conflicts"
    CONFLICT_DESCRIPTION_ONLY = "conflict_description_only"
    NO_EXTERNAL_FACTUAL_CLAIMS = "no_external_factual_claims"


class DisclosureCode(StrEnum):
    ONLINE_VERIFICATION_FAILED = "online_verification_failed"
    PARTIAL_EVIDENCE = "partial_evidence"
    SOURCE_CONFLICT = "source_conflict"
    VALIDATION_UNAVAILABLE = "validation_unavailable"
    VALIDATION_FAILED = "validation_failed"
    USER_FORBID_WEB = "user_forbid_web"


class WarningCode(StrEnum):
    HIGH_CONSEQUENCE = "high_consequence"


class ValidatorRequirement(StrEnum):
    NORMAL = "normal"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True)
class AnswerState:
    evidence_state: EvidenceState | None
    generation_mode: AnswerGenerationMode
    certainty: AnswerCertainty
    allowed_claim_scope: AllowedClaimScope
    disclosure_codes: tuple[DisclosureCode, ...]
    warning_codes: tuple[WarningCode, ...]
    validator_requirement: ValidatorRequirement
```

`AnswerGenerationMode` is a transient, closed chat-dispatch output that replaces the existing skip/grounded/failure branching. It is not consumed by Router, Planner, Evidence, Validator semantics or Renderer, does not enter retrieval decisions, and is not a new Agent/state machine.

- [ ] **Step 5: Implement one pure mapping function**

Create `src/search/policy.py`:

```python
def decide_answer_state(
    analysis: RequestAnalysis,
    evidence: EvidenceBundle | None,
    failure_code: SearchFailureCode | None,
) -> AnswerState:
    state = evidence.evidence_state if evidence is not None else None
    warnings = (
        (WarningCode.HIGH_CONSEQUENCE,)
        if analysis.risk.warning_required
        else ()
    )
    validator_requirement = (
        ValidatorRequirement.FAIL_CLOSED
        if analysis.risk.fail_closed
        or analysis.freshness.requirement is FreshnessRequirement.CURRENT
        else ValidatorRequirement.NORMAL
    )
    if analysis.retrieval.skip_reason in {
        SkipReason.SOCIAL_OR_EMOTIONAL,
        SkipReason.CREATIVE_OR_ROLEPLAY,
        SkipReason.PROVIDED_TEXT_TRANSFORM,
        SkipReason.PROVIDED_CONTENT_SUMMARY,
        SkipReason.PURE_MATH,
        SkipReason.CLOSED_LOGIC,
        SkipReason.CLOSED_CONTEXT_ONLY,
    }:
        return AnswerState(None, AnswerGenerationMode.PLAIN, AnswerCertainty.UNVERIFIED, AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS, (), warnings, validator_requirement)
    if analysis.retrieval.skip_reason is SkipReason.USER_FORBID_WEB:
        return AnswerState(None, AnswerGenerationMode.FIXED, AnswerCertainty.UNVERIFIED, AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS, (DisclosureCode.USER_FORBID_WEB,), warnings, validator_requirement)
    if state is EvidenceState.SUFFICIENT:
        return AnswerState(state, AnswerGenerationMode.GROUNDED, AnswerCertainty.VERIFIED, AllowedClaimScope.ALL_SUPPORTED, (), warnings, validator_requirement)
    if state is EvidenceState.PARTIAL:
        return AnswerState(state, AnswerGenerationMode.GROUNDED, AnswerCertainty.LIMITED, AllowedClaimScope.SUPPORTED_SUBSET, (DisclosureCode.PARTIAL_EVIDENCE,), warnings, validator_requirement)
    if state is EvidenceState.CONFLICTING:
        scope = _conflict_scope(evidence)
        return AnswerState(state, AnswerGenerationMode.GROUNDED, AnswerCertainty.CONFLICTING, scope, (DisclosureCode.SOURCE_CONFLICT,), warnings, validator_requirement)
    return AnswerState(state, AnswerGenerationMode.FIXED, AnswerCertainty.UNVERIFIED, AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS, (_failure_disclosure(failure_code),), warnings, validator_requirement)
```

`_conflict_scope()` returns `SUPPORTED_SUBSET_WITH_CONFLICTS` when any material topic remains unconflicted and supported; otherwise `CONFLICT_DESCRIPTION_ONLY`.

- [ ] **Step 6: Replace tier-based chat failure behavior**

Delete `if decision.route is SearchTier.DEEP`. Chat calls `decide_answer_state()` after the Search Pipeline and:

- generates a grounded draft only when scope permits supported claims;
- invokes the ordinary non-search model only when `generation_mode is PLAIN` for a closed non-factual task;
- uses a fixed failure body when scope is `NO_EXTERNAL_FACTUAL_CLAIMS`;
- never invokes the ordinary answer model to fill external facts after insufficient Evidence;
- preserves the same `EvidenceBundle` object and `evidence_state` before and after policy/answer work.

- [ ] **Step 7: Run policy/chat tests and commit**

Run:

```powershell
python -B -m unittest tests.test_search_policy tests.test_chat_retrieval_flow tests.test_identity_configuration -v
```

Expected: all pass; no branch in `chat_service.py` tests `SearchTier.DEEP`.

```powershell
git add src/search/policy.py src/search/models.py src/chat/chat_service.py tests/test_search_policy.py tests/test_chat_retrieval_flow.py
git commit -m "feat: map evidence into bounded answer policy"
```

---

### Task 8: Separate Validator Status and Build Render State without Rewriting Evidence

**Files:**
- Modify: `src/search/models.py:688-753`
- Modify: `src/search/validation.py:239-772`
- Modify: `src/search/policy.py`
- Test: `tests/test_search_validation.py`
- Test: `tests/test_search_policy.py`

- [ ] **Step 1: Write RED validator-status tests**

Add tests for `passed / filtered / unavailable / malformed`. Each test saves `original_state = bundle.evidence_state`, runs validation, and asserts:

```python
self.assertIs(bundle.evidence_state, original_state)
self.assertLessEqual(len(report.retained_blocks), len(draft.answer_blocks))
self.assertLessEqual(len(report.retained_claims), len(draft.claims))
```

Add fail-closed tests for high-consequence/current fact verifier unavailability and ordinary-stable tests that keep only structurally complete, citable and hidden-fact-free blocks.

Add `SUFFICIENT + FILTERED` and `SUFFICIENT + UNAVAILABLE` cases. They must preserve `bundle.evidence_state is SUFFICIENT` while producing an effective certainty no higher than `LIMITED`; if no factual block survives, effective scope must be `NO_EXTERNAL_FACTUAL_CLAIMS` and certainty `UNVERIFIED`.

- [ ] **Step 2: Write RED render-state ownership tests**

Assert `build_render_state()` creates citation numbers only for retained claims, includes only actually used sources, preserves structured conflict groups, and carries policy-decided warnings/disclosures without examining question text.

- [ ] **Step 3: Run RED**

Run:

```powershell
python -B -m unittest tests.test_search_validation tests.test_search_policy -v
```

Expected: no validator status or render state exists.

- [ ] **Step 4: Add final validator/render contracts**

In `models.py`:

```python
class ValidatorStatus(StrEnum):
    PASSED = "passed"
    FILTERED = "filtered"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"


class RenderOutcome(StrEnum):
    ANSWER = "answer"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    FAILURE = "failure"
    VALIDATION_FAILURE = "validation_failure"


@dataclass(frozen=True)
class ValidationReport:
    status: ValidatorStatus
    effective_certainty: AnswerCertainty
    effective_claim_scope: AllowedClaimScope
    draft: GroundedDraft
    retained_blocks: tuple[AnswerBlock, ...]
    retained_claims: tuple[Claim, ...]
    removed_block_ids: tuple[str, ...]
    claim_labels: Mapping[str, SupportLabel]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class RenderState:
    outcome: RenderOutcome
    visible_blocks: tuple[AnswerBlock, ...]
    visible_claims: tuple[Claim, ...]
    citation_map: Mapping[str, int]
    used_sources: tuple[EvidenceItem, ...]
    conflict_groups: tuple[EvidenceConflict, ...]
    disclosure_codes: tuple[DisclosureCode, ...]
    warning_codes: tuple[WarningCode, ...]
```

- [ ] **Step 5: Make validator transitions explicit and monotone**

Change the function signature to the following; the keyword-only arguments retain the current production dependencies and do not add stages:

```python
def validate_and_filter(
    draft: GroundedDraft,
    bundle: EvidenceBundle,
    answer_state: AnswerState,
    *,
    claim_discoverer: Any,
    semantic_verifier: Any,
    trace: SearchTrace | None = None,
    clock: Any = None,
) -> ValidationReport:
```

The implementation assigns the following closed status:

```python
status = ValidatorStatus.PASSED
if structural_or_semantic_removals:
    status = ValidatorStatus.FILTERED
if verifier_unavailable:
    status = ValidatorStatus.UNAVAILABLE
if draft_malformed:
    status = ValidatorStatus.MALFORMED
```

Compute `effective_certainty` and `effective_claim_scope` monotonically:

```python
effective_certainty = answer_state.certainty
effective_claim_scope = answer_state.allowed_claim_scope
if status in {ValidatorStatus.FILTERED, ValidatorStatus.UNAVAILABLE}:
    effective_certainty = min_certainty(effective_certainty, AnswerCertainty.LIMITED)
if not retained_factual_or_inference_blocks:
    effective_certainty = AnswerCertainty.UNVERIFIED
    effective_claim_scope = AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS
if status is ValidatorStatus.MALFORMED:
    effective_certainty = AnswerCertainty.UNVERIFIED
    effective_claim_scope = AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS
```

`min_certainty()` uses the closed order `UNVERIFIED < CONFLICTING/LIMITED < VERIFIED`; it may never return a certainty higher than the input answer state. Record both policy certainty and effective certainty in Trace without modifying Evidence state.

For `FAIL_CLOSED`, unavailable/malformed retains no factual or inference blocks. For `NORMAL`, unavailable may retain only blocks that pass all deterministic structural, Evidence-ID, URL, ownership and hidden-span checks. Never mutate or replace `bundle.evidence_state`.

- [ ] **Step 6: Move model-status text cleaning before render state**

Move `_strip_program_owned_search_disclosures()` and professional-warning span removal out of `renderer.py` into a deterministic validation helper:

```python
def sanitize_visible_block_text(text: str) -> str:
    text = _remove_program_status_spans(text)
    text = _remove_program_warning_spans(text)
    return text.strip()
```

Apply it before constructing `RenderState.visible_blocks`; do not use domain semantics.

- [ ] **Step 7: Build citations and sources from retained claims only**

In `policy.py`, `build_render_state(answer_state, validation, evidence)` must:

1. select visible blocks allowed by `validation.effective_claim_scope`; retain `answer_state.allowed_claim_scope` only as the original policy decision for Trace/audit;
2. collect Evidence IDs from visible retained claims;
3. assign stable citation numbers in first-use order;
4. include only matching citable `EvidenceItem` values;
5. include only relevant structured conflicts;
6. carry policy disclosure/warning codes unchanged while using `validation.effective_certainty` and `validation.effective_claim_scope` to determine visible scope and render outcome.

- [ ] **Step 8: Run validation/policy suites and commit**

Run:

```powershell
python -B -m unittest tests.test_search_validation tests.test_search_policy tests.test_search_models -v
```

Expected: all pass and evidence-state identity assertions remain true.

```powershell
git add src/search/models.py src/search/validation.py src/search/policy.py tests/test_search_validation.py tests/test_search_policy.py tests/test_search_models.py
git commit -m "refactor: separate validation from immutable evidence"
```

---

### Task 9: Make Renderer a Pure View and Rewire Chat and `/search`

**Files:**
- Modify: `src/search/renderer.py:96-780`
- Modify: `src/chat/chat_service.py:273-459`
- Modify: `src/commands/search.py:5-12`
- Test: `tests/test_search_renderer.py`
- Test: `tests/test_chat_retrieval_flow.py`
- Test: `tests/test_command_renderer.py`
- Test: `tests/test_identity_configuration.py`
- Test: `tests/test_product_scope.py`

- [ ] **Step 1: Write RED pure-renderer tests**

Call Renderer with only `RenderState`:

```python
reply = render_search_reply(render_state, qq_limit=1700)
```

Test:

- success renders answer and used sources with no “搜索成功/检索完成” banner;
- partial/conflict/failure templates follow disclosure codes;
- warning code renders the fixed warning exactly once;
- no warning code renders none, even for medical-looking block text;
- citations never suspend and unused sources never appear;
- long source title+URL remains atomic under QQ splitting.

- [ ] **Step 2: Add a static RED ownership test**

```python
def test_renderer_does_not_import_or_inspect_semantic_search_state(self):
    source = Path("src/search/renderer.py").read_text(encoding="utf-8")
    for forbidden in (
        "SearchTier",
        "RiskContext",
        "FreshnessContext",
        "EvidenceState",
        "HIGH_CONSEQUENCE_ACTION",
        "medical",
        "legal",
        "financial",
    ):
        self.assertNotIn(forbidden, source)
```

- [ ] **Step 3: Run RED**

Run:

```powershell
python -B -m unittest tests.test_search_renderer tests.test_chat_retrieval_flow tests.test_command_renderer tests.test_identity_configuration tests.test_product_scope -v
```

Expected: current Renderer still accepts `SearchPipelineResult`, inspects tier/risk/evidence and owns semantic text cleanup.

- [ ] **Step 4: Replace Renderer input with `RenderState`**

Final signature:

```python
def render_search_reply(state: RenderState, *, qq_limit: int) -> RenderedReply:
    body = _render_blocks(state.visible_blocks, state.visible_claims, state.citation_map)
    conflicts = _render_conflicts(state.conflict_groups, state.citation_map)
    disclosures = _render_disclosure_codes(state.disclosure_codes)
    warnings = _render_warning_codes(state.warning_codes)
    sources = _render_sources(state.used_sources, state.citation_map)
    return _finish_render(body, conflicts, disclosures, warnings, sources, qq_limit)
```

Delete `_is_high_consequence`, tier/evidence decision branches and natural-language warning/status detection. Renderer may choose templates only from closed enum codes.

- [ ] **Step 5: Rewire normal chat and `/search` through one flow**

`chat_service.generate_reply()` performs exactly:

```python
result = orchestrator.run(request)
answer_state = decide_answer_state(result.analysis, result.evidence, result.failure_code)
draft = _generate_allowed_draft(answer_state, result)
validation = _validate_allowed_draft(draft, answer_state, result)
render_state = build_render_state(answer_state, validation, result.evidence)
reply = render_search_reply(render_state, qq_limit=_qq_limit())
```

`/search` only sets `force_search=True`; it must not add a status message or choose standard by itself. A successful `/search` outputs the answer and actual sources directly.

- [ ] **Step 6: Verify warning and disclosure boundaries**

Add controls:

- ordinary successful search: no accuracy/professional warning;
- ordinary failed search: online verification failure disclosure only;
- high-consequence success/failure: one warning;
- closed social/creative skip: no search warning;
- explicit no-web high consequence: zero Provider/answer-model calls, fixed no-web limitation and one warning;
- model text containing duplicate warning/status text is sanitized before Renderer and appears zero/one times as policy requires.

Use call-counting fakes to assert end-to-end LLM ceilings remain at or below the existing maxima: light ≤5 calls, standard without Repair ≤6, standard with Repair ≤7. The early analysis counts once; Repair may rerun the existing Evidence Judge once but introduces no new stage.

```python
for case, maximum in (
    (light_success_case(), 5),
    (standard_no_repair_case(), 6),
    (standard_one_repair_case(), 7),
):
    with self.subTest(case=case.name):
        self.assertLessEqual(run_with_counting_llms(case).total_llm_calls, maximum)
```

These are failing acceptance assertions, not report-only metrics.

- [ ] **Step 7: Run renderer/chat/command suites and commit**

Run:

```powershell
python -B -m unittest tests.test_search_renderer tests.test_chat_retrieval_flow tests.test_command_renderer tests.test_identity_configuration tests.test_product_scope -v
```

Expected: all pass; normal chat and `/search` share the same evidence/policy/validation/rendering path.

```powershell
git add src/search/renderer.py src/chat/chat_service.py src/commands/search.py tests/test_search_renderer.py tests/test_chat_retrieval_flow.py tests/test_command_renderer.py tests/test_identity_configuration.py tests/test_product_scope.py
git commit -m "refactor: render closed web search view state"
```

---

### Task 10: Remove Deep and Transitional Duplicate State

**Files:**
- Modify: `src/search/__init__.py`
- Modify: `src/search/models.py`
- Modify: `src/search/router.py`
- Modify: `src/search/planner.py`
- Modify: `src/search/providers/base.py`
- Modify: `src/search/evidence.py`
- Modify: `src/search/orchestrator.py`
- Modify: `src/search/validation.py`
- Modify: `src/search/renderer.py`
- Modify: `src/chat/chat_service.py`
- Modify: `src/services/search_service.py`
- Modify: `tests/search_fakes.py`
- Modify: all `tests/test_search_*.py` and `tests/test_chat_retrieval_flow.py` fixtures that still construct deep or legacy decision fields
- Modify: `tests/test_identity_configuration.py`
- Modify: `tests/test_main_image_flow.py`
- Modify: `tests/test_multimodal_chat.py`

- [ ] **Step 1: Add RED closed-schema tests**

Update `tests/test_search_models.py` to require:

```python
self.assertEqual({"skip", "light", "standard"}, {item.value for item in SearchTier})
self.assertEqual({SearchTier.LIGHT, SearchTier.STANDARD}, set(DEFAULT_TIER_BUDGETS))
```

Add constructor tests proving final `RetrievalDecision` has only retrieval-owned fields:

```python
expected = {
    "route",
    "skip_reason",
    "must_search",
    "reason_codes",
}
self.assertEqual(expected, set(RetrievalDecision.__dataclass_fields__))
```

Add an import/export test that imports `src.search`, asserts the final public types are available, and asserts `RiskLevel`, `PotentialHarm`, `Freshness`, `max_tier` and `SearchTier.DEEP` are not exported.

- [ ] **Step 2: Run RED and capture remaining operational references**

Run:

```powershell
rg -n "SearchTier\.DEEP|\bdeep\b|Freshness\.HIGH|model_recommended_tier|program_minimum_tier" src tests tools README.md eval/search/README.md
python -B -m unittest tests.test_search_models -v
```

Expected: references and closed-schema tests fail.

- [ ] **Step 3: Delete final deep behavior and compatibility shims**

Remove:

- `SearchTier.DEEP`, rank and default budget;
- `_operational_tier()` transitional shim;
- deep Planner query/fallback branches and `_deep_location_hint`;
- deep Provider fallback reserve;
- deep Evidence admission/source requirements;
- deep repair/failure/validation branches;
- prompt support for model-recommended tier;
- `LLMRoutingAdvisor` compatibility alias;
- tests and fixtures whose only purpose is deep behavior.

Do not replace them with standard-only branches. Standard keeps exactly the frozen budget.

- [ ] **Step 4: Remove duplicated routing-owned risk/freshness fields**

Final `RetrievalDecision`:

```python
@dataclass(frozen=True)
class RetrievalDecision:
    route: SearchTier
    skip_reason: SkipReason | None
    must_search: bool
    reason_codes: tuple[RetrievalComplexityCode, ...]
```

Delete legacy `freshness`, `risk`, `actionability`, `potential_harm`, `program_minimum_tier`, `model_recommended_tier`, mixed `trigger_codes` and `benefit_dimensions` when no remaining caller needs them. Risk/Freshness live only in `RequestAnalysis`; Trace stores closed metadata derived from the three contexts without raw text.

- [ ] **Step 5: Close the final Trace schema**

Final Trace must include:

```text
must_search, route, retrieval_reason_codes,
query_trace_entries, provider_attempts,
initial/repair/total query counts,
candidate/read/round counters, repair reason/target IDs,
retrieval_stop_reason,
evidence_state, supported/missing topic IDs, topic freshness states,
answer certainty/scope/disclosure/warning codes,
validator status and retained/removed counts,
render outcome/citation/source counts,
stage latencies and finalization state
```

No raw `SearchQuery`, raw topic label, user text, Evidence text or URL may enter `to_log_dict()`.

- [ ] **Step 6: Update all fixtures to the final constructors**

Replace positional construction with keyword construction. Preserve explicit tests for malformed schema, immutability, total serialization, UUID request IDs, Provider attempts and late-deadline sealing.

- [ ] **Step 7: Run all search-facing tests**

Run:

```powershell
python -B -m unittest tests.test_search_models tests.test_search_router tests.test_search_planner tests.test_search_providers tests.test_search_extraction tests.test_search_evidence tests.test_search_orchestrator tests.test_search_validation tests.test_search_policy tests.test_search_renderer tests.test_chat_retrieval_flow tests.test_command_renderer tests.test_identity_configuration tests.test_main_image_flow tests.test_multimodal_chat tests.test_product_scope -v
```

Expected: all pass.

- [ ] **Step 8: Run static cleanup checks**

Run:

```powershell
rg -n "SearchTier\.DEEP|Freshness\.HIGH|recommended_tier.*deep|if .*tier.*deep|route.*deep" src tests
rg -n "render.*(medical|legal|financial|freshness|risk)|HIGH_CONSEQUENCE_ACTION" src/search/renderer.py
rg -n "plan_repair|SearchOrchestrator|ProviderRegistry" src/search/policy.py src/search/validation.py src/search/renderer.py
```

Expected: all commands exit 1 with no matches, except historical prose outside `src tests` is allowed.

- [ ] **Step 9: Commit schema cleanup**

```powershell
git diff --name-only HEAD | Sort-Object
git add src/search/__init__.py src/search/models.py src/search/router.py src/search/planner.py src/search/providers/base.py src/search/evidence.py src/search/orchestrator.py src/search/validation.py src/search/renderer.py src/chat/chat_service.py src/services/search_service.py tests/search_fakes.py tests/test_search_models.py tests/test_search_router.py tests/test_search_planner.py tests/test_search_providers.py tests/test_search_extraction.py tests/test_search_evidence.py tests/test_search_orchestrator.py tests/test_search_validation.py tests/test_search_policy.py tests/test_search_renderer.py tests/test_chat_retrieval_flow.py tests/test_command_renderer.py tests/test_identity_configuration.py tests/test_main_image_flow.py tests/test_multimodal_chat.py tests/test_product_scope.py
git commit -m "refactor: remove legacy deep search state"
```

Before staging, require `git diff --name-only HEAD` to be a subset of the explicit Task 10 list above. Stop and investigate any extra path instead of staging it.

---

### Task 11: Update Evaluator, Documentation, and Final Acceptance Gates

**Files:**
- Modify: `tools/evaluate_search.py`
- Modify: `tests/test_search_evaluation.py`
- Modify: `eval/search/README.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-29-retrieval-benefit-search-design.md`
- Modify: `docs/superpowers/plans/2026-08-08-ddgs-first-search-routing-and-disclosures.md`
- Create: `.superpowers/sdd/2026-08-09-websearch-simplification/final-acceptance-report.md` (ignored review artifact)

- [ ] **Step 1: Write RED evaluator tests for the new tier and budget contract**

Add tests that:

- accept only `skip/light/standard` in new production traces;
- reject light with more than 1 query, 5 URLs, 2 reads, 1 round, 8 seconds or any repair;
- reject standard with initial>3, repair>1, total>4, URLs>8, reads>5, rounds>2 or retrieval>20 seconds;
- verify direct is included in initial/total count;
- reject `SUFFICIENT` with material stale/unknown topic;
- reject light repair and any post-repair third-round signal;
- require immutable Evidence state and separate validator status;
- require Query metadata and reject raw query text;
- keep fixture/unsigned/unreviewed artifacts non-certifying.

- [ ] **Step 2: Run evaluator RED**

Run:

```powershell
python -B -m unittest tests.test_search_evaluation -v
```

Expected: old tier labels/budget tables and Trace schema cause failures.

- [ ] **Step 3: Update evaluator without editing the dataset JSONL files**

Change closed runtime labels to `("light", "standard")`; derive query counts from `QueryTraceEntry.query_index`; validate stage/attempt/counter consistency. Treat legacy checked-in `deep` rows as migration/integrity errors until the owner supplies a reviewed new artifact—do not rewrite their labels.

Keep these non-certification behaviors:

```text
integrity: nonzero while owner review or schema errors exist
offline fixture: nonzero and certifying=false
online without explicit authorization: exit 2, status=not run
```

- [ ] **Step 4: Update current documentation and mark history explicitly**

`README.md` and `eval/search/README.md` must state:

- DDGS-first, Tavily only on DDGS failure;
- skip/light/standard only;
- exact hard caps, including direct query inside standard initial<=3;
- topic-level freshness;
- standard-only one Repair and unconditional stop;
- Risk affects answer policy, not search depth;
- successful search displays answer+used sources with no status banner;
- ordinary success has no professional/accuracy warning;
- failed online verification has an explicit failure disclosure;
- high-consequence warning is policy-decided and emitted once;
- 140 rows/online provider quality remain external gates.

Add a banner to the 2026-07-29 and 2026-08-08 documents saying they are historical baselines and that the 2026-08-09 approved spec supersedes operational deep/Tavily-order statements. Do not delete historical reasoning.

- [ ] **Step 5: Run the complete hermetic verification**

Run:

```powershell
python -B -m unittest discover -s tests -t . -v
python -B -m compileall -q src tests
git diff --check
```

Expected: zero failures/errors, compile exit 0, diff check exit 0, and no external HTTP guard fires.

- [ ] **Step 6: Run deterministic invariant probes**

Run:

```powershell
rg -n "SearchTier\.DEEP|Freshness\.HIGH|recommended_tier.*deep|if .*tier.*deep|route.*deep" src tests tools
rg -n "risk_context|warning_required|high_consequence" src/search/planner.py src/search/providers src/search/extraction.py
rg -n "SearchOrchestrator|plan_repair|ProviderRegistry" src/search/policy.py src/search/validation.py src/search/renderer.py
rg -n "raw_query|query_text|original_question|evidence_body" src/search/models.py
```

Expected: the first three searches return no prohibited matches. The final privacy probe may find internal model fields outside Trace serialization; manually confirm none are emitted by `SearchTrace.to_log_dict()`.

- [ ] **Step 7: Run evaluation gates honestly**

Run:

```powershell
python -B tools/evaluate_search.py integrity
python -B tools/evaluate_search.py offline
python -B tools/evaluate_search.py online
```

Expected:

- `integrity` remains nonzero for genuine owner-review/schema migration issues;
- `offline` remains non-certifying for the fixture baseline;
- `online` returns exit 2 / not run and makes no Provider call.

Do not mark these external gates passed.

- [ ] **Step 8: Write the final acceptance report**

Record:

- commit range;
- exact test counts and commands;
- static cleanup results;
- light/standard measured cap probes;
- LLM call counts for light, standard/no-repair, standard/repair compared with baseline;
- production branch/HEAD and clean status;
- unchanged `eval/search/*.jsonl` hashes;
- external gates separately as not certified.

Reject the branch if any operational deep path, light repair, stale SUFFICIENT, policy→search call, Renderer semantic decision or raw Query Trace remains.

- [ ] **Step 9: Commit evaluator and docs**

```powershell
git add tools/evaluate_search.py tests/test_search_evaluation.py eval/search/README.md README.md docs/superpowers/specs/2026-07-29-retrieval-benefit-search-design.md docs/superpowers/plans/2026-08-08-ddgs-first-search-routing-and-disclosures.md
git commit -m "docs: finalize simplified web search contract"
```

- [ ] **Step 10: Request final independent review before merge**

Use `superpowers:requesting-code-review` against the complete implementation range. The reviewer must verify the approved spec, all RED/GREEN evidence, cumulative diff, full hermetic suite and the external-gate wording. Do not merge until Critical and Important findings are zero.

---

## Execution Order and Review Checkpoints

Execute Tasks 1–11 strictly in order because later schemas depend on earlier migration boundaries. After each task:

1. run the task-specific tests;
2. inspect `git diff --check`;
3. run a fresh specification-compliance review;
4. run a separate code-quality review;
5. fix blocking findings with new RED tests;
6. create one scoped non-amended commit;
7. confirm tracked worktree cleanliness before starting the next task.

Task 11 is the only point at which documentation may claim the new runtime contract is implemented. It must still describe human review and controlled online quality as external, unpassed gates.

## Specification Coverage Matrix

| Approved spec area | Implementing tasks |
|---|---|
| Responsibilities and one-way contexts (§4–§6) | Tasks 2–3, 7–9 |
| Hard request budgets and direct-query accounting (§7) | Tasks 1, 4, 6, 11 |
| Planner/checklist without extra LLM (§8) | Task 4 |
| Topic freshness and sufficiency priority (§9) | Task 5 |
| Seven closed Repair reasons and post-repair stop (§10) | Task 6 |
| Failure matrix and conflict-preserving claim scope (§11) | Task 7 |
| Validator monotonicity and immutable Evidence (§12) | Task 8 |
| Pure deterministic Renderer (§13) | Task 9 |
| Body-free layered Trace (§14) | Tasks 6, 8, 10–11 |
| No added LLM stage and call ceilings (§15) | Tasks 3–4, 6, 9, 11 |
| Staged migration and final deletion (§16–§17) | Tasks 1–3, 10 |
| Test matrix and acceptance/external gates (§18–§19) | Tasks 1–11, especially Task 11 |
