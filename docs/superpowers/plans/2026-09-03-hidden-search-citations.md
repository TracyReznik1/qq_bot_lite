# Hidden Search Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搜索回答不向 QQ 用户展示编号、来源标题、URL 或来源列表，同时完整保留后台证据映射和 Trace 元数据。

**Architecture:** 只改变 `src/search/renderer.py` 的展示投影。策略层、验证层和 `RenderState` 继续维护 Claim→Evidence→来源闭包；渲染器从这些来源生成 `RenderedReply.used_evidence_ids` 与 `shown_source_urls`，但不把它们拼进 `text` 或 `chunks`。

**Tech Stack:** Python 3.11、`unittest`

---

## File Map

- Modify: `src/search/renderer.py` — 输出自然语言，隐藏所有依据展示，保留后台元数据。
- Modify: `tests/test_search_renderer.py` — 渲染层可见文本和后台元数据测试。
- Modify: `tests/test_chat_retrieval_flow.py` — 端到端搜索回复与 Trace 契约测试。
- Modify: `tests/test_readme_guide.py` — README 用户行为契约。
- Modify: `README.md` — 说明证据仅在后台保留。

### Task 1: 隐藏渲染层引用并保留元数据

**Files:**
- Modify: `tests/test_search_renderer.py`
- Modify: `src/search/renderer.py`

- [ ] **Step 1: 将成功回答测试改为自然语言输出契约**

把 `test_success_renders_answer_and_sources_without_banner` 改名为 `test_success_hides_citations_and_sources_but_keeps_backend_metadata`，断言：

```python
self.assertEqual("版本是3.2", rendered.text)
for hidden in ("[1]", "来源：", "Source A", "https://a.example.com/page"):
    self.assertNotIn(hidden, rendered.text)
self.assertEqual(("E1",), rendered.used_evidence_ids)
self.assertEqual(("https://a.example.com/page",), rendered.shown_source_urls)
```

保留原有“不显示搜索成功 banner”断言。

- [ ] **Step 2: 将多证据、冲突和长 URL 测试改成隐藏展示契约**

将 `test_citations_never_suspend` 改为：

```python
def test_citations_never_appear_in_visible_text(self):
    # 使用现有 E1 state 构造
    rendered = render_search_reply(state, qq_limit=1700)
    self.assertEqual("正文", rendered.text)
    self.assertNotIn("[1]", rendered.text)
    self.assertEqual(("E1",), rendered.used_evidence_ids)
```

将 same-URL 测试断言改为：

```python
self.assertEqual("正文", rendered.text)
self.assertNotIn("[1]", rendered.text)
self.assertNotIn("[2]", rendered.text)
self.assertNotIn(shared_url, rendered.text)
self.assertEqual(("E1", "E2"), rendered.used_evidence_ids)
self.assertEqual((shared_url, shared_url), rendered.shown_source_urls)
```

将 conflict 测试改名为 `test_conflict_renders_natural_claims_without_source_details`，断言：

```python
self.assertIn("来源之间存在未解决差异", rendered.text)
self.assertIn("3.2", rendered.text)
self.assertIn("3.3", rendered.text)
for hidden in ("[1]", "[2]", "Source A", "Source B", "https://a.example.com", "https://b.example.com", "来源："):
    self.assertNotIn(hidden, rendered.text)
self.assertEqual(("E1", "E2"), rendered.used_evidence_ids)
self.assertEqual(("https://a.example.com", "https://b.example.com"), rendered.shown_source_urls)
```

将 long-URL 测试断言改为：

```python
self.assertEqual(("正文",), rendered.chunks)
self.assertNotIn(url, rendered.text)
self.assertEqual((url,), rendered.shown_source_urls)
```

- [ ] **Step 3: 运行渲染测试并确认 RED**

Run:

```bash
python -m unittest tests.test_search_renderer -v
```

Expected: FAIL；当前文本仍包含 `[n]`、来源标题、URL 和“来源：”。

- [ ] **Step 4: 最小修改渲染投影**

在 `src/search/renderer.py`：

1. `_render_blocks` 不再遍历 Claim/citation map，直接连接 block 文本：

```python
def _render_blocks(blocks: Sequence[AnswerBlock]) -> str:
    return "\n".join(block.text for block in blocks).strip()
```

2. `_render_conflicts` 只输出去重后的 member value：

```python
def _render_conflicts(conflicts: Sequence[EvidenceConflict]) -> list[str]:
    sections = []
    for conflict in conflicts:
        values = list(dict.fromkeys(member.value for member in conflict.members if member.value))
        if values:
            sections.append(
                f"冲突点（{conflict.conflict_key}）：\n"
                + "\n".join(f"- {value}" for value in values)
            )
    return sections
```

3. 将 `_render_sources` 改为只计算后台元数据：

```python
def _source_metadata(
    sources: Sequence[EvidenceItem],
    citation_map: Mapping[str, int],
) -> tuple[list[str], list[str]]:
    used_ids = []
    shown_urls = []
    for item in sources:
        if citation_map.get(item.evidence_id) is None or not item.url:
            continue
        used_ids.append(item.evidence_id)
        shown_urls.append(item.url)
    return used_ids, shown_urls
```

4. `render_search_reply` 调用 `_render_blocks(state.visible_blocks)`、`_render_conflicts(state.conflict_groups)` 和 `_source_metadata(...)`。
5. `_finish_render` 删除 `sources` 参数及追加“来源：”的分支；仍把 `used_ids`、`shown_urls` 写入 `RenderedReply`。
6. 删除不再使用的 `_bounded_source_title`；保留通用 `split_qq_reply` 对外部文本中来源段落的兼容拆分能力。

- [ ] **Step 5: 运行渲染与模型契约测试**

Run:

```bash
python -m unittest tests.test_search_renderer tests.test_search_models tests.test_search_policy -v
```

Expected: PASS；`RenderState` 的引用闭包校验仍保持严格。

- [ ] **Step 6: 提交渲染改动**

```bash
git add src/search/renderer.py tests/test_search_renderer.py
git commit -m "feat: hide search citations from replies"
```

### Task 2: 更新端到端搜索回复断言

**Files:**
- Modify: `tests/test_chat_retrieval_flow.py`

- [ ] **Step 1: 修改成功路径的可见文本断言**

在 `test_normal_and_force_search_success_emit_no_status_banner` 和 `test_normal_and_force_search_remove_model_success_statuses` 中，把：

```python
self.assertIn("来源：", reply)
```

改为：

```python
self.assertNotIn("来源：", reply)
self.assertNotRegex(reply, r"\[\d+\]")
self.assertNotIn("https://", reply)
```

在 `test_grounded_path_populates_answer_validation_render_counts_and_timestamps` 中，把：

```python
self.assertIn("版本是3.2[1]", reply)
```

改为：

```python
self.assertIn("版本是3.2", reply)
self.assertNotIn("[1]", reply)
self.assertNotIn("来源：", reply)
```

保留 `trace.citation_count == 1`、`render_citation_count` 和 `render_source_count` 的后台计数断言。

在 `test_partial_bundle_answers_supported_only` 中把 URL 可见断言改为：

```python
self.assertNotIn("https://a.example.com", reply)
self.assertNotRegex(reply, r"\[\d+\]")
```

把 `test_conflict_bundle_shows_sources` 改名为 `test_conflict_bundle_hides_sources`，并把两个 URL 可见断言改为：

```python
self.assertIn("来源之间存在未解决差异", reply)
self.assertIn("3.2", reply)
self.assertIn("3.3", reply)
self.assertNotIn("https://a.example.com", reply)
self.assertNotIn("https://b.example.com", reply)
self.assertNotRegex(reply, r"\[\d+\]")
```

- [ ] **Step 2: 运行端到端测试**

Run:

```bash
python -m unittest tests.test_chat_retrieval_flow -v
```

Expected: PASS；如果有其他断言仍要求搜索成功回复展示来源，只把用户可见断言改为隐藏，不改变 evidence payload、验证或 Trace 断言。

- [ ] **Step 3: 提交集成测试更新**

```bash
git add tests/test_chat_retrieval_flow.py
git commit -m "test: enforce natural search replies"
```

### Task 3: 同步 README

**Files:**
- Modify: `tests/test_readme_guide.py`
- Modify: `README.md`

- [ ] **Step 1: 先增加 README 契约断言**

在 `test_readme_documents_tavily_primary_ddgs_fallback` 中加入：

```python
self.assertIn("不会向 QQ 用户展示引用编号、来源标题或 URL", self.readme)
self.assertIn("证据映射仅保留在后台校验与 Trace 中", self.readme)
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
python -m unittest tests.test_readme_guide.ReadmeGuideTests.test_readme_documents_tavily_primary_ddgs_fallback -v
```

Expected: FAIL；README 尚未包含后台保留说明。

- [ ] **Step 3: 修改 README 搜索成功说明**

将“搜索成功时，普通聊天和 `/search` 都直接输出答案与实际使用的来源”改为：

```text
搜索成功时，普通聊天和 `/search` 都直接输出自然语言答案，不会向 QQ 用户展示引用编号、来源标题或 URL；证据映射仅保留在后台校验与 Trace 中。
```

后续 banner、风险提示和失败披露说明保持不变。

- [ ] **Step 4: 运行文档及相关测试**

Run:

```bash
python -m unittest tests.test_readme_guide tests.test_search_renderer tests.test_chat_retrieval_flow -v
```

Expected: PASS。

- [ ] **Step 5: 提交文档同步**

```bash
git add README.md tests/test_readme_guide.py
git commit -m "docs: describe backend-only search evidence"
```

### Task 4: 完整验证

**Files:**
- No production changes expected

- [ ] **Step 1: 检查差异**

```bash
git diff --check
git status --short --branch
```

Expected: 无空白错误；仅保留进入任务前已有的未提交改动。

- [ ] **Step 2: 运行完整测试**

```bash
python -m unittest discover -s tests -t . -v
```

Expected: 全部 PASS。

- [ ] **Step 3: 运行编译检查**

```bash
python -m compileall -q src tests run_bot.py
```

Expected: exit code 0，无输出。

- [ ] **Step 4: 核对范围**

```bash
git log -5 --oneline
git status --short --branch
```

Expected: 本功能只提交 File Map 中的文件，不提交 `.env`、运行数据、`.tmp.driveupload/` 或原有未提交改动。
