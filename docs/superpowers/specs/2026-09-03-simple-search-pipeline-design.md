# QQBot 简化搜索管线设计

## 1. 背景

当前网页搜索虽然对外只有 `skip / light / standard` 三种路由，但内部仍包含请求风险分类、主题级 freshness、来源关系、Evidence 状态机、Gap Analysis、Repair、结构化 GroundedDraft、Claim Discovery 和 Semantic Verifier。普通事实问题最多会经过五次 LLM 调用，standard 路径还会增加 Planner 和第二次 Judge。

这些机制提高了形式上的证据约束，却也引入了大量独立失败点。已复现的真实问题中，检索结果已经达到 `sufficient`，回答仍因四秒超时或没有返回严格 JSON 而失败。当前超时封装还不能终止底层线程，可能让已超时调用继续占用共享 worker。

本项目只是 QQ 聊天机器人，不需要按医疗、法律、金融或其他高危领域区分回答策略。本次重构以稳定回答、容易维护和可预测降级为优先目标。

## 2. 目标

1. 搜索只保留 `skip`、`light`、`standard` 三种结果，其中 light 和 standard 是仅有的两种搜索模式。
2. `/search` 始终使用 standard；普通聊天由一次合并的 LLM 调用选择路由并生成搜索词。
3. 删除风险领域、fail-closed、主题级 freshness、Repair 和回答后语义验证。
4. Evidence Judge 只产生宽松的相关性分数，不再承担完整证据状态判定。
5. 回答模型直接生成自然语言，不再要求 GroundedDraft、Claim 或 Evidence ID JSON。
6. 任一外部阶段失败时尽量使用已经取得的结果继续回答，异常不得逃出 `generate_reply()`。
7. 普通聊天隐藏来源；显式 `/search` 展示最多三个来源。
8. 保留 Provider 回退、URL 安全、资源上限、按需 Reader、Trace 和确定性降级。

## 3. 非目标

- 不为医疗、法律、金融或安全问题提供独立策略。
- 不执行逐句事实核验或 Claim 到 Evidence 的精确映射。
- 不要求来源独立交叉印证。
- 不把网页发布日期作为证据准入条件。
- 不保留第二轮自适应修复搜索。
- 不在迁移期间同时向真实 Provider 执行新旧两套搜索。

## 4. 总体架构

迁移期间在 `src/search/simple/` 建立新管线，复用现有 Provider、公共 HTTP URL 安全策略和网页提取能力，不复用旧搜索状态机。

```text
用户请求
  ↓
RoutePlanner（一次 LLM）
  ├─ skip
  ├─ light：一个查询
  └─ standard：最多三个查询
  ↓
SearchPipeline
  ↓
Tavily → 仅未解决查询回退 DDGS
  ↓
摘要不足时 Reader 按需读取一至两页
  ↓
EvidenceRanker（一次 LLM，返回 0～1 分数）
  ↓
AnswerGenerator（自然语言）
  ↓
Renderer
  ├─ 普通聊天：隐藏来源
  └─ /search：展示最多三个来源
```

公共调用方只通过 `src.search` 导出的入口和结果类型访问新管线。`simple` 是迁移和内部实现边界；切换后调用方不得继续引用旧状态类型。

## 5. 最小数据契约

### 5.1 SearchMode

封闭值：

- `skip`：不进行网页搜索；
- `light`：一个查询；
- `standard`：最多三个查询。

### 5.2 SearchPlan

只包含：

- `mode`；
- 清理和去重后的查询列表；
- 是否来自确定性规划降级。

它不包含 Risk、Freshness、RequiredTopic、SourceRequirement、Repair 或 Claim 相关字段。

### 5.3 SearchResult

每条候选只保留：

- 内部结果 ID；
- 标题；
- 经 URL policy 验证的最终 URL；
- Provider 摘要或按需读取的摘录；
- Provider 名称；
- 可选的相关性分数。

### 5.4 SearchResponse

最终结果包含：

- 回答文本；
- 最多三个可展示来源；
- 是否使用规划、Judge 或 Answer 降级；
- 精简 Trace。

### 5.5 SearchTrace

仅记录：

- 最终模式和查询数量；
- Tavily/DDGS 的成功、空结果、超时或错误状态；
- 候选数量和 Reader 次数；
- Planner、Judge、Answer 是否降级；
- 各阶段耗时；
- 最终输出类型：模型回答、摘要降级或搜索失败。

Trace 不记录原始查询、网页正文、URL 或模型响应正文。

## 6. 路由与查询规划

### 6.1 合并调用

RoutePlanner 只调用一次 LLM，同时返回模式和查询：

```json
{
  "mode": "skip|light|standard",
  "queries": ["查询1", "查询2"]
}
```

解析采用宽松规则：

- 可从 Markdown 代码块或前后附带文字中提取 JSON；
- 未知字段忽略；
- 非法 mode、空查询和非字符串查询忽略；
- 查询标准化、去重并限制长度；
- light 最多保留一个查询；
- standard 最多保留三个查询；
- `/search` 无条件覆盖为 standard；
- 如果解析后没有可用查询，使用原问题。

规划器不再输出风险、医疗/法律领域、freshness、发布日期、来源类型或 required topics。用户问题中的“最新”“今年”“截至今天”等时间信息原样保留在搜索词里。

### 6.2 确定性降级

合并调用超时、报错或无法解析时：

- `/search`：standard，使用原问题作为查询；
- 普通聊天中明显的闲聊、创作、文本改写和纯数学：skip；
- 其他普通问题：light，使用原问题作为查询。

分类器故障不得阻断搜索或普通回复。

## 7. Provider 与 Reader

### 7.1 查询资源上限

| 模式 | 查询数 | 去重候选 | Reader 上限 |
|---|---:|---:|---:|
| light | 1 | 5 | 1 |
| standard | 3 | 8 | 2 |

standard 查询并行执行，不再保留第二轮 Repair。

### 7.2 Provider 顺序

1. 对本轮全部查询并行调用 Tavily；
2. 只对 Tavily 未解决的查询并行回退 DDGS；
3. 合并结果并按规范 URL 去重。

保留 Tavily 日期参数规范化及参数错误后的预算内一次无日期重试，以兼容旧调用方或未来显式参数；新 RoutePlanner 默认不产生日期过滤参数。

### 7.3 URL 准入

候选必须通过现有公共 HTTP/HTTPS URL policy。拒绝本地地址、私有网络、非 HTTP scheme、无效主机和无法规范化的 URL。URL 安全是硬门槛，不因降级而绕过。

### 7.4 按需 Reader

优先使用 Provider 标题和摘要。摘要去除空白后少于 80 个字符时视为过短；只有摘要为空或过短时，才按照 Provider 原始排名读取页面：light 最多一页，standard 最多两页。

Reader 超时、拒绝访问或提取失败时保留 Provider 标题和已有摘要，不改变整个请求状态。Reader 返回的正文摘录最多保留 1500 个字符。禁止为所有候选默认读取正文。

## 8. 相关性排序

EvidenceRanker 一次批量评估全部候选，只返回 `0～1` 分数：

```json
{
  "scores": {
    "R1": 0.92,
    "R2": 0.55,
    "R3": 0.10
  }
}
```

解析规则：

- 接受代码块和 JSON 前后文字；
- 未知结果 ID 忽略；
- 缺失或非法分数按 `0.5` 处理；
- 超出范围的有限数值限制到 `0～1`；
- 单条非法记录不影响其他记录；
- 相同分数保持 Provider 原始顺序；
- 没有解析出任何合法分数时，整体视为 Judge 降级并保持 Provider 原始顺序。

分数仅用于排序，不设证据合格阈值。只删除 URL 无效、标题和摘要同时为空，或 Judge 明确给出零分的候选。

Judge 超时、报错或没有解析出任何合法分数时继续使用搜索结果，并在最终回复中添加一次：

> 信息可能不完整。

## 9. 回答与渲染

### 9.1 自然语言回答

AnswerGenerator 接收：

- 用户问题；
- 排名前列的标题和摘要或 Reader 摘录；
- 对话历史；
- 角色设定。

模型直接输出自然语言。系统提示要求基于提供的信息回答、区分确定事实和不确定判断、不要声称执行了未提供的能力，但不要求 JSON、Claim ID、Evidence ID 或逐句引用。

传给 AnswerGenerator 的证据正文不包含 URL。来源链接由 Renderer 独立处理，避免普通聊天正文泄露 URL，也避免依赖模型正确生成链接。

### 9.2 来源显示

- 普通聊天：不展示来源标题和 URL；
- `/search`：在回答末尾展示相关性最高的最多三个来源标题和 URL。

来源只能来自通过 URL policy 的候选，不接受模型生成的链接。

### 9.3 Answer 降级

AnswerGenerator 超时、报错或返回空白时，Renderer 确定性整理排名靠前的搜索结果：

```text
根据搜索结果：
1. 标题：摘要
2. 标题：摘要
```

摘要需要清理控制字符、限制单条及总长度，并服从 QQ 回复长度上限。普通聊天不展示 URL；`/search` 仍附最多三个来源。

不再返回“回答未能通过证据核验”，也不执行 Claim Discovery、Semantic Verifier 或 GroundedDraft 解析。

### 9.4 搜索失败

只有以下情况视为搜索失败：

- 所有 Provider 均不可用或超时；
- 没有任何通过 URL policy 的候选；
- 所有候选的标题和摘要均为空，且 Reader 也没有取得内容。

日期缺失、来源不独立、主题覆盖不完整或 Judge 格式异常不构成搜索失败。

## 10. 超时与错误边界

建议默认超时：

| 阶段 | 超时 | 降级行为 |
|---|---:|---|
| RoutePlanner | 8 秒 | 确定性规划 |
| Tavily | 8 秒 | 回退 DDGS |
| DDGS | 15 秒 | 保留其他成功查询 |
| Reader | 5 秒 | 使用 Provider 摘要 |
| EvidenceRanker | 10 秒 | 原始顺序并提示不完整 |
| AnswerGenerator | 20 秒 | 标题与摘要模板 |

这些值应集中定义，并可通过现有配置模式覆盖。

不再使用外层线程作为网络调用的主要超时手段。所有 LLM、Provider 和 Reader 调用必须把剩余超时传到底层 HTTP 客户端。并行只用于独立查询和页面读取；调用超时后不得继续占用共享搜索 worker。

每个阶段只捕获自己能够转换的异常，并返回明确的降级状态。顶层管线提供最后一道异常边界，确保异常不会传播出 `generate_reply()`。

## 11. 迁移方案

采用旁路实现、切换后删除的方式：

1. 为已确认行为增加新管线黑盒测试；
2. 在 `src/search/simple/` 实现最小数据契约和 RoutePlanner；
3. 接入现有 Tavily、DDGS、URL policy 和基础 Reader；
4. 实现宽松 EvidenceRanker、自然语言 AnswerGenerator 和 Renderer；
5. 使用离线 Provider fixtures 验证整条管线；
6. 将普通聊天和 `/search` 统一切换到新入口；
7. 执行少量真实查询验收，不同时调用新旧 Provider；
8. 删除旧状态机、旧导出和只验证旧契约的测试；
9. 更新 README、评估工具和配置说明；
10. 执行全量测试、compileall 和 diff 检查。

迁移完成前保留旧实现以便对照和回退；入口切换完成后不长期维护双管线。

## 12. 验收标准

至少覆盖以下行为：

1. `/search` 始终使用 standard；
2. light 只执行一个查询；
3. standard 最多并行执行三个查询；
4. RoutePlanner 异常时按确定性规则降级；
5. Tavily 未解决查询才回退 DDGS；
6. 只有短摘要触发 Reader，且不超过模式上限；
7. Judge 部分格式错误不影响其他结果排序；
8. Judge 失败仍能回答，并提示信息可能不完整；
9. Answer 超时或空回复时输出标题摘要模板；
10. 普通聊天不显示 URL；
11. `/search` 展示最多三个安全来源；
12. 日期缺失不会单独导致证据不足；
13. 不再按医疗、法律、金融或高危类型改变搜索和回答流程；
14. 任一阶段异常均不会逃出 `generate_reply()`；
15. 新搜索生产代码中不存在 Repair、Claim Discovery、Semantic Verifier 或 GroundedDraft 运行路径；
16. 旧管线删除后，全量测试、`compileall` 和 `git diff --check` 通过。
