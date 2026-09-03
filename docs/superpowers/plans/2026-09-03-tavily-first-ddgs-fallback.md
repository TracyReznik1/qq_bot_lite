# Tavily-First Search Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让每个搜索查询优先使用 Tavily、仅在未解决时回退 DDGS，并把所有 DDGS 阶段超时提高到 30 秒。

**Architecture:** 保留 `SearchOrchestrator` 现有的两批查询级调度，将首批和回退批次的 Provider 名称与局部预算交换。Provider 返回的 hit 只有通过公网 HTTP(S) URL 规范化后才算解决查询，确保无效 URL 也会触发 DDGS；证据、Trace 和失败聚合结构不变。

**Tech Stack:** Python 3.11、Flask、`unittest`、DDGS、Tavily

---

## File Map

- Modify: `src/search/orchestrator.py` — Tavily-first、unresolved-only DDGS 调度和有效 URL 判定。
- Modify: `src/search/budget.py` — 三个 DDGS 阶段预算统一为 30 秒。
- Modify: `tests/test_search_provider_batches.py` — Provider 顺序、条件回退、兄弟查询隔离与 repair 顺序回归测试。
- Modify: `tests/test_search_budget.py` — 精确预算与派生 watchdog 测试。
- Modify: `tests/test_readme_guide.py` — 当前 Provider 契约文档测试。
- Modify: `README.md` — 用户可见 Provider 顺序、阶段预算和 watchdog。

历史 baseline/spec/plan 保持不变，它们记录的是过去版本的冻结契约。

### Task 1: Tavily-first 查询级回退

**Files:**
- Modify: `tests/test_search_provider_batches.py`
- Modify: `src/search/orchestrator.py`

- [ ] **Step 1: 将首选 Provider 测试改成 Tavily-first，并先覆盖成功短路**

将测试模块 docstring 改为：

```python
"""Unit tests for independent Tavily-first and DDGS fallback query batches."""
```

将 `test_ddgs_resolves_all_queries_tavily_not_invoked` 改为：

```python
def test_tavily_resolves_all_queries_ddgs_not_invoked(self):
    tavily_calls = []
    ddgs_calls = []

    class MockTavily:
        name = "tavily"
        def readiness(self):
            return ProviderReadiness("tavily", True, True, None)
        def search(self, query, **kwargs):
            tavily_calls.append(query.query_id)
            hit = _hit(
                url=f"https://tavily.example.com/{query.query_id}",
                query_id=query.query_id,
                provider="tavily",
            )
            return ProviderResult("tavily", ProviderStatus.SUCCESS, (hit,), 1)

    class MockDDGS:
        name = "ddgs"
        def readiness(self):
            return ProviderReadiness("ddgs", True, True, None)
        def search(self, query, **kwargs):
            ddgs_calls.append(query.query_id)
            return ProviderResult("ddgs", ProviderStatus.SUCCESS, (), 1)

    orchestrator = self._make_orchestrator((MockDDGS(), MockTavily()))
    result = orchestrator.run(_request())

    self.assertTrue(tavily_calls)
    self.assertEqual([], ddgs_calls)
    self.assertIs(result.evidence.evidence_state, EvidenceState.SUFFICIENT)
    self.assertEqual("tavily", result.trace.provider_attempts[0].provider)
```

- [ ] **Step 2: 运行测试，确认因当前 DDGS-first 行为失败**

Run:

```bash
python -m unittest tests.test_search_provider_batches.ProviderBatchOrchestrationTests.test_tavily_resolves_all_queries_ddgs_not_invoked -v
```

Expected: FAIL，`ddgs_calls` 非空或 `tavily_calls` 为空，证明测试捕获当前顺序。

- [ ] **Step 3: 增加未解决查询、无效 URL 和 repair 回退测试**

将第二个测试改名为 `test_tavily_fails_one_query_only_unresolved_query_falls_back_to_ddgs`，让 Tavily 对 `initial-1` 返回成功 hit、对其他查询返回 `EMPTY`，让 DDGS 返回成功 hit，并断言：

```python
self.assertIn("initial-1", tavily_calls)
self.assertNotIn("initial-1", ddgs_calls)
self.assertEqual(set(tavily_calls) - {"initial-1"}, set(ddgs_calls))
```

增加无效 URL 测试：

```python
def test_tavily_invalid_urls_fall_back_to_ddgs(self):
    ddgs_calls = []

    class MockTavily:
        name = "tavily"
        def readiness(self):
            return ProviderReadiness("tavily", True, True, None)
        def search(self, query, **kwargs):
            invalid = _hit(url="http://127.0.0.1/private", query_id=query.query_id, provider="tavily")
            return ProviderResult("tavily", ProviderStatus.SUCCESS, (invalid,), 1)

    class MockDDGS:
        name = "ddgs"
        def readiness(self):
            return ProviderReadiness("ddgs", True, True, None)
        def search(self, query, **kwargs):
            ddgs_calls.append(query.query_id)
            hit = _hit(url=f"https://example.com/{query.query_id}", query_id=query.query_id, provider="ddgs")
            return ProviderResult("ddgs", ProviderStatus.SUCCESS, (hit,), 1)

    orchestrator = self._make_orchestrator((MockTavily(), MockDDGS()))
    orchestrator.run(_request())
    self.assertTrue(ddgs_calls)
```

增加失败状态覆盖：

```python
def test_each_unresolved_tavily_status_falls_back_to_ddgs(self):
    statuses = (
        ProviderStatus.EMPTY,
        ProviderStatus.TIMEOUT,
        ProviderStatus.ERROR,
        ProviderStatus.UNAVAILABLE,
        ProviderStatus.NOT_CONFIGURED,
    )
    for tavily_status in statuses:
        with self.subTest(status=tavily_status):
            ddgs_calls = []

            class MockTavily:
                name = "tavily"
                def readiness(self):
                    return ProviderReadiness(
                        "tavily",
                        tavily_status is not ProviderStatus.NOT_CONFIGURED,
                        tavily_status not in {ProviderStatus.NOT_CONFIGURED, ProviderStatus.UNAVAILABLE},
                        None,
                    )
                def search(self, query, **kwargs):
                    return ProviderResult("tavily", tavily_status, (), 1)

            class MockDDGS:
                name = "ddgs"
                def readiness(self):
                    return ProviderReadiness("ddgs", True, True, None)
                def search(self, query, **kwargs):
                    ddgs_calls.append(query.query_id)
                    hit = _hit(url=f"https://example.com/{query.query_id}", query_id=query.query_id, provider="ddgs")
                    return ProviderResult("ddgs", ProviderStatus.SUCCESS, (hit,), 1)

            self._make_orchestrator((MockTavily(), MockDDGS())).run(_request())
            self.assertTrue(ddgs_calls)
```

增加 repair 顺序的直接单元测试，并在 imports 中加入 `QueryPurpose`、`SearchQuery`、`SearchRoundKind`、`SearchTrace`：

```python
def test_repair_round_uses_tavily_before_ddgs(self):
    calls = []

    class MockTavily:
        name = "tavily"
        def readiness(self):
            return ProviderReadiness("tavily", True, True, None)
        def search(self, query, **kwargs):
            calls.append(("tavily", query.query_id))
            hit = _hit(url="https://tavily.example.com/repair", query_id=query.query_id, provider="tavily")
            return ProviderResult("tavily", ProviderStatus.SUCCESS, (hit,), 1)

    class MockDDGS:
        name = "ddgs"
        def readiness(self):
            return ProviderReadiness("ddgs", True, True, None)
        def search(self, query, **kwargs):
            calls.append(("ddgs", query.query_id))
            return ProviderResult("ddgs", ProviderStatus.EMPTY, (), 1)

    orchestrator = self._make_orchestrator((MockDDGS(), MockTavily()))
    query = SearchQuery(
        query_id="repair-1",
        query_index=1,
        round_kind=SearchRoundKind.REPAIR,
        purpose=QueryPurpose.REPAIR,
        text="补充",
        target_topic_ids=("topic-1",),
    )
    trace = SearchTrace("req-repair", RequestSource.CHAT, SearchTier.STANDARD)
    orchestrator._run_provider_round(
        (query,), SearchTier.STANDARD, SearchRoundKind.REPAIR, trace
    )

    self.assertEqual([("tavily", "repair-1")], calls)
```

同步第三个 sibling 测试：Tavily 为 `initial-1` 返回 `https://example.com/q1` 的成功 hit，其余查询超时；DDGS 只处理其余查询并超时，最终证据仍包含 `https://example.com/q1`。

- [ ] **Step 4: 运行整个 Provider 批次测试并确认新增测试失败**

Run:

```bash
python -m unittest tests.test_search_provider_batches -v
```

Expected: FAIL，失败集中在当前 Provider 顺序以及 Tavily 成功但 URL 无效时未回退。

- [ ] **Step 5: 最小实现 Tavily-first 和有效 URL 判定**

在 `SearchOrchestrator._run_provider_round` 中：

1. 首批 tracker/outcome 改为 Tavily；使用 `initial_tavily_seconds` 或 `repair_tavily_seconds`。
2. 对结果 hit 先执行：

```python
hits = tuple(
    hit
    for hit in outcome.result.hits
    if canonicalize_public_http_url(hit.url)
)
```

只有 `status is ProviderStatus.SUCCESS and hits` 才标记 `RESOLVED`。
3. 将所有未解决查询的第二批改为 DDGS，并使用 `initial_ddgs_seconds` 或 `repair_ddgs_seconds`。
4. 第二批结果同样过滤无效 URL。
5. 合并 attempts 时保持 `(tavily_attempts, ddgs_attempts)` 顺序。
6. readiness failure、QueryOutcome 和 Trace 的状态映射保持现有逻辑，不改变数据结构。

同时把方法内注释更新为：

```python
# Batch 1: Tavily for all queries concurrently
# Batch 2: DDGS ONLY for unresolved queries concurrently
```

- [ ] **Step 6: 运行 Provider 批次和编排器回归测试**

Run:

```bash
python -m unittest tests.test_search_provider_batches tests.test_search_orchestrator -v
```

Expected: PASS。若 `tests.test_search_orchestrator` 中存在旧顺序断言，只将断言改为 Tavily-first，不放宽尝试次数、失败码或 sibling isolation 要求。

- [ ] **Step 7: 提交 Provider 调度改动**

```bash
git add src/search/orchestrator.py tests/test_search_provider_batches.py tests/test_search_orchestrator.py
git commit -m "feat: prefer Tavily with DDGS fallback"
```

仅当 `tests/test_search_orchestrator.py` 实际因旧顺序断言而修改时才加入该文件。

### Task 2: DDGS 30 秒独立预算

**Files:**
- Modify: `tests/test_search_budget.py`
- Modify: `src/search/budget.py`

- [ ] **Step 1: 先修改精确预算测试**

在 light 期望中改为：

```python
"initial_ddgs_seconds": 30,
```

并将 watchdog 断言改为：

```python
self.assertEqual(58, DEFAULT_SEARCH_BUDGET_POLICY.maximum_request_seconds(SearchTier.LIGHT))
```

在 standard 期望中改为：

```python
"initial_ddgs_seconds": 30,
"repair_ddgs_seconds": 30,
```

并将 watchdog 断言改为：

```python
self.assertEqual(112, DEFAULT_SEARCH_BUDGET_POLICY.maximum_request_seconds(SearchTier.STANDARD))
```

新增明确契约测试：

```python
def test_every_active_ddgs_stage_has_thirty_second_budget(self):
    light = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.LIGHT)
    standard = DEFAULT_SEARCH_BUDGET_POLICY.for_route(SearchTier.STANDARD)

    self.assertEqual(30, light.initial_ddgs_seconds)
    self.assertEqual(30, standard.initial_ddgs_seconds)
    self.assertEqual(30, standard.repair_ddgs_seconds)
```

- [ ] **Step 2: 运行预算测试并确认 RED**

Run:

```bash
python -m unittest tests.test_search_budget.SearchBudgetPolicyTests -v
```

Expected: FAIL，实际 DDGS 值仍为 6/8/5，watchdog 仍为 34/65。

- [ ] **Step 3: 修改默认预算**

在 `DEFAULT_SEARCH_BUDGET_POLICY` 中设置：

```python
# light
initial_ddgs_seconds=30,

# standard
initial_ddgs_seconds=30,
repair_ddgs_seconds=30,
```

其余字段不变。

- [ ] **Step 4: 运行预算和搜索测试**

Run:

```bash
python -m unittest tests.test_search_budget tests.test_search_provider_batches tests.test_search_orchestrator -v
```

Expected: PASS。

- [ ] **Step 5: 提交预算改动**

```bash
git add src/search/budget.py tests/test_search_budget.py
git commit -m "feat: raise DDGS stage budgets to thirty seconds"
```

### Task 3: 同步用户文档契约

**Files:**
- Modify: `tests/test_readme_guide.py`
- Modify: `README.md`

- [ ] **Step 1: 先将 README 契约测试改为 Tavily-first**

把 `test_readme_documents_ddgs_primary_tavily_fallback` 改为：

```python
def test_readme_documents_tavily_primary_ddgs_fallback(self):
    self.assertIn("Tavily 是主搜索提供者", self.readme)
    self.assertIn("DDGS", self.readme)
    self.assertIn("回退", self.readme)
    self.assertNotIn("DDGS 是主搜索提供者", self.readme)
    self.assertIn("DDGS 的阶段超时统一为 30 秒", self.readme)
    self.assertIn("`light` 约 58 秒、`standard` 约 112 秒", self.readme)
    self.assertIn(
        "返回明确的在线检索失败披露（不同路径会说明在线检索未完成或无法完成在线核验）",
        self.readme,
    )
    self.assertIn("在线检索未完成", self.readme)
    self.assertIn("无法完成在线核验", self.readme)
```

- [ ] **Step 2: 运行 README 测试并确认 RED**

Run:

```bash
python -m unittest tests.test_readme_guide.ReadmeGuideTests.test_readme_documents_tavily_primary_ddgs_fallback -v
```

Expected: FAIL，README 仍描述 DDGS-first 和旧 watchdog。

- [ ] **Step 3: 更新 README 搜索说明**

将 `TAVILY_API_KEY` 行改为明确说明：Tavily 是主搜索 Provider；未配置、不可用、报错、超时、无结果或没有有效 URL 时，DDGS 回退；两个 Provider 均失败时保持现有失败披露。

将检索阶段说明改为：

```text
Tavily 先以一个并发批次执行全部初始查询；只有未完成（未配置、不可用、空、超时、错误或 URL 无效）的查询才进入独立的 DDGS 批次，已由 Tavily 解决的查询不会重复请求 DDGS。DDGS 的阶段超时统一为 30 秒。
```

将 watchdog 数值改为：

```text
`light` 约 58 秒、`standard` 约 112 秒
```

保留“安全上限，不是预期耗时”和 repair 最多一次的说明。

- [ ] **Step 4: 运行文档及相关专项测试**

Run:

```bash
python -m unittest tests.test_readme_guide tests.test_search_budget tests.test_search_provider_batches -v
```

Expected: PASS。

- [ ] **Step 5: 提交文档同步**

```bash
git add README.md tests/test_readme_guide.py
git commit -m "docs: describe Tavily-first search fallback"
```

### Task 4: 完整验证

**Files:**
- No production changes expected

- [ ] **Step 1: 检查差异和空白错误**

```bash
git diff --check

git status --short
```

Expected: `git diff --check` 无输出；状态中只保留进入任务前已有且不属于本功能的未提交改动。

- [ ] **Step 2: 运行完整测试套件**

```bash
python -m unittest discover -s tests -t . -v
```

Expected: 全部测试 PASS。冻结历史 baseline 测试继续验证旧文档，不修改旧 baseline 内容。

- [ ] **Step 3: 运行语法编译检查**

```bash
python -m compileall -q src tests run_bot.py
```

Expected: exit code 0，无输出。

- [ ] **Step 4: 核对最终差异范围**

```bash
git diff HEAD~3 --stat

git log -4 --oneline
```

Expected: 功能提交只涉及 File Map 中列出的运行时代码、测试和 README；不包含 `.env`、运行时数据、`.tmp.driveupload/` 或用户原有未提交改动。
