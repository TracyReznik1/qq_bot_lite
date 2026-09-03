# 搜索依据仅后台保留设计

## 目标

搜索回答继续使用完整的证据化检索、声明验证和引用映射，但面向 QQ 用户的回复表现为自然语言，不展示任何证据编号、来源标题、来源 URL 或来源列表。

## 用户可见行为

搜索成功、部分成功和来源冲突回答均遵循以下规则：

- 正文末尾不追加 `[1]`、`[2]` 等引用编号。
- 不输出“来源：”章节。
- 不输出证据来源标题或 URL。
- 冲突回答可以说明“来源之间存在未解决差异”，并自然列出不同说法，但不标识各说法对应的来源或编号。
- 现有部分证据披露、冲突披露、搜索失败披露和高风险警告保持不变。

## 后台行为

本次不改变证据和验证合同：

- `RenderState.citation_map` 继续由策略层构建。
- Claim 继续携带 `evidence_ids` 并接受结构与语义验证。
- `RenderState.used_sources` 继续保留实际使用的 Evidence。
- `RenderedReply.used_evidence_ids` 与 `shown_source_urls` 继续记录后台元数据。
- `SearchTrace` 的引用数、来源数及其他 body-free 指标继续根据后台元数据统计。

因此，隐藏仅发生在最终展示层，不影响回答是否有证据支持，也不降低后台审计能力。

## 实现边界

修改 `src/search/renderer.py`：

1. `_render_blocks` 仅拼接通过验证的 block 文本，不追加引用编号。
2. `_render_conflicts` 不输出来源标题和引用编号，只输出去重后的冲突说法。
3. `_finish_render` 不把来源列表追加到用户可见 `text`，但仍接收并写入 `used_ids` 与 `shown_urls`。
4. 保留现有来源元数据计算，以免改变 Trace 和审计合同。

不修改 Router、Planner、Provider、Evidence、Validator 或搜索预算。

## 测试

采用测试先行：

- 普通搜索回答正文不含 `[n]`。
- 多证据回答不含任何引用编号、来源标题、URL 或“来源：”章节。
- 冲突回答保留不同说法，但不含来源标题、编号和 URL。
- `RenderedReply.used_evidence_ids` 与 `shown_source_urls` 仍等于实际后台使用的证据和 URL。
- 长 URL 不再作为用户可见独立分片，但后台仍记录。
- 搜索失败披露、风险警告、普通非搜索回复拆分行为不回归。

完成专项测试后运行完整 unittest 和 compileall。

## 文档

README 中搜索成功行为改为：回答不会向 QQ 用户展示引用编号或来源列表，证据映射仅保留在后台校验与 Trace 中。历史设计和 baseline 文档不回写。
