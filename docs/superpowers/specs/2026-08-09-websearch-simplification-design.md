# QQ Bot WebSearch 精简设计：有限自适应证据搜索

**日期：** 2026-08-09  
**状态：** 已完成四部分设计讨论，等待项目所有者复核正式规格  
**调查基线：** `codex/ddgs-first-search@8abaa8f`  
**适用范围：** 普通聊天、显式 `/search`、检索路由、查询规划、Provider、正文读取、Evidence、一次 Repair、回答策略、Validator、引用与 QQ 渲染

---

## 1. 设计结论

本轮采用“职责拆分式精简”，不重写现有 WebSearch，也不建设 Deep Research Agent。

保留已经通过大量 hermetic 测试的可靠骨架：

- 检索收益路由；
- DDGS 主提供者与 Tavily 条件回退；
- 请求级绝对截止时间和并发封印；
- Provider 候选与正文读取分层；
- Evidence admission、来源归因和冲突结构；
- Claim–Evidence 绑定；
- 程序生成 Citation、URL 和来源清单；
- 搜索失败的显式降级；
- 最多一次由 Evidence Gap 驱动的 Repair；
- body-free SearchTrace。

删除或解耦历史复杂度：

- 删除运行时 `deep`，只保留 `skip / light / standard`；
- 风险不再控制 Search Tier、预算、Query 或 Repair；
- Freshness 不再控制 Search Tier 或预算；
- 显式搜索只禁止 `skip`，不自动进入 `standard`；
- light 不允许运行时升级为 standard；
- Pivot / Backtrack 只作为一次 Repair 的原因，不建立独立状态机；
- Answer Policy 和 Validator 不得重新开启检索；
- Renderer 退化为确定性 View 层，不执行语义判断；
- 不新增 Risk、Freshness、Checklist、Gap、Pivot、Stop 等 LLM 或 Agent。

最终要明确保持三个互不替代的维度：

```text
Search Tier = 预计检索有多复杂
Freshness   = Evidence 需要多新
Risk        = 最终回答需要多谨慎
```

---

## 2. 当前代码事实与调整原因

当前 `8abaa8f` 的真实调用链是：

```text
普通聊天 / /search
→ SearchOrchestrator
→ RetrievalBenefitRouter
→ SearchPlanner
→ ProviderRegistry（DDGS → 条件 Tavily）
→ SearchExtractor
→ Evidence Judge / EvidenceAssembler
→ Evidence Gap / 可选一次 Repair
→ Grounded Answer
→ Claim Discovery / Semantic Validation
→ 程序化 Citation / QQ Renderer
```

当前实现已经具备一次真正的反馈式 Repair：第一轮 Evidence 完成后，系统根据缺失主题或来源冲突决定是否执行一个不同 Query。它不是把首轮多个 Query 错算成多轮。

但现有 `deep` 同时承担了多种不应耦合的职责：

- 动态事实和 `Freshness.HIGH` 的程序升档；
- 高后果请求的搜索升档；
- 5 个初始 Query、15 个候选、8 次读取和40秒预算；
- deep 专用时间限定和来源 Query；
- Provider fallback reserve；
- 强正文 Evidence admission；
- 搜索失败和 semantic validator 不可用时的保守策略。

这使“检索复杂度、证据时效、现实风险、回答失败策略”被同一个 tier 表达。删除 `deep` 不能采用全局 `deep → standard` 替换，否则 standard 会变成改名后的 deep。

调查还确认：

- `required_topics / missing_topics` 已经是轻量 Checklist 雏形；
- 当前 Gap 主要只有 `missing_topic / source_conflict`，不足以表达 stale、实体歧义和错误前提；
- Evidence freshness 还没有成为完整的 sufficiency 条件；
- 一次 standard/deep 搜索回答最多可能包含早期分析、Planner、Evidence Judge、Answer、Claim Discovery、Semantic Validator 等多次 LLM 调用；
- 当前140条评测尚未人工复核，离线 fixture 只能证明评测管线可运行，不能证明真实搜索质量；
- 当前 online evaluator 是安全占位入口，不会执行真实 Provider 请求。

因此本轮只做职责迁移和状态精简，不顺带扩建研究能力或宣称真实搜索质量已经认证。

---

## 3. 目标与非目标

### 3.1 目标

- Search Tier 只表达预期检索复杂度。
- 事实型请求继续保持积极检索，合法无收益任务才允许 skip。
- light 保持快速、单轮和不可升级。
- standard 在固定总预算内按 Evidence 状态最多执行一次 Repair。
- Risk、Freshness、Evidence、Answer、Validation 和 Rendering 保持单向数据流。
- Evidence 充分时提前停止；证据不足时诚实表达，不回退模型记忆补事实。
- 运行时概念、tier-specific 分支、重复字段和失去行为作用的状态总体减少。
- 保留 Provider、Reader、Evidence、Citation、deadline 和隐私方面的已有可靠性。

### 3.2 非目标

- 不建设 `/research` 或 Deep Research 模式。
- 不增加第三轮搜索或 Repair-of-Repair。
- 不增加动态 tier escalation。
- 不新增 Checklist/Pivot/Backtrack/Stop Agent。
- 不新增 Risk/Freshness/Gap/Repair Benefit LLM。
- 不在本轮强行合并 Claim Discovery 与 Semantic Validator。
- 不重写 Provider、Reader、Citation 或并发截止时间核心。
- 不用 LOC 作为唯一复杂度指标。
- 不伪造人工评测标签、独立预测、在线结果或质量认证。

---

## 4. 不可违反的程序不变量

1. 运行时 Search Tier 只有 `skip / light / standard`。
2. 合法 skip 仍使用封闭理由；模型知识、自信、常识或记忆不能成为 skip 理由。
3. 显式搜索只设置 `must_search/search_required`，不得直接提升 standard。
4. Risk 不得影响 Search Tier、Query、候选、读取、轮次、Repair 或时间预算；Freshness 只能约束 Query 的日期、版本或时间范围以及 Evidence 合格性，不得改变 tier、Query 数量或其他预算。
5. light 永远单轮、无 Repair、不可运行时升级。
6. standard 最多一次 Evidence-driven Repair，第二轮后无条件停止。
7. Provider fallback 是同一语义 Query 内的容错，不是 Repair，也不是新检索轮次。
8. 所有检索预算在整个检索请求内共享，Repair 不重新获得预算。
9. Search Result 只是候选，不能直接成为可引用 Evidence。
10. 查询相关性是进入 Evidence 的前置门槛；第一方身份不能越过相关性门槛。
11. 动态事实在不满足 Freshness 时不能输出确定结论。
12. 搜索失败或 Evidence 不足时不能静默回退模型记忆补外部事实。
13. “没有找到证据”不能推导成“事实不存在”。
14. Evidence state 在 Search Pipeline 结束后不可变。
15. Answer Policy 和 Validator 只能降低确定性或缩小可回答范围，不能修改 Evidence、提高确定性或重新搜索。
16. Renderer 不判断风险、Freshness、Evidence sufficiency 或 Repair，只格式化 render_state。
17. Citation 必须对应本次请求实际取得、被正文使用的 Evidence 和真实 URL。
18. 来源冲突不得静默合并；无争议的受支持事实可以与冲突说明共同展示。
19. Trace 不记录用户正文、原始 Query 或 Evidence 正文。
20. 本轮不得增加新的 LLM 阶段。

---

## 5. 职责边界与逻辑数据流

### 5.1 早期 Request Analysis

早期分析一次产生三个逻辑视图：

```text
Request Analysis
├─ retrieval_context
├─ freshness_context
└─ risk_context
```

这只是职责拆分，不要求新建三个模块、三个 class 或三次 LLM 调用。优先复用现有一次分类输出和确定性规则。

### 5.2 retrieval_context

只包含搜索决策需要的信息：

- `must_search`
- 合法 `skip_reason`
- 是否需要外部事实
- 单事实或多事实
- 单实体或多实体
- 比较、推荐、综合判断
- 多来源、独立来源或交叉核验要求
- 其他可审计的检索复杂度信号

只有 retrieval_context 可以进入 RetrievalBenefitRouter。

### 5.3 freshness_context

逻辑上包含：

- 是否要求当前或指定时间事实
- 用户指定日期、版本、时间窗或 as-of 边界
- Evidence 允许的时间范围
- 最终 freshness 状态：`not_required / satisfied / stale / unknown`

它可以进入 Planner、Evidence freshness evaluation、Answer Policy 和 Validator，但不能进入 Router 或改变预算。

### 5.4 risk_context

逻辑上包含：

- 是否可能影响现实行动
- 是否为高后果回答
- 是否要求 fail-closed
- `warning_required`
- 其他 Answer/Validator 所需的封闭风险元数据

Risk 可以在请求早期识别，但必须从 RetrievalBenefitRouter 的决策权中彻底剥离。领域名本身不能机械产生 warning；例如稳定机构定义不等于高后果行动建议。

### 5.5 单向数据流

```text
retrieval_context
→ Retrieval Router
→ skip / light / standard

freshness_context
→ Planner
→ Evidence freshness evaluation

Search Pipeline
→ immutable evidence_state

evidence_state + risk_context + freshness_context + search/failure state
→ Answer Policy
→ answer_state

answer_state + Evidence
→ Draft Generation
→ Validator
→ validator_status + render_state

render_state
→ Renderer
→ QQ 文本
```

一旦 Search Pipeline 结束，Answer Policy、Validator 和 Renderer 均无权重新打开检索。

`evidence_state / answer_state / render_state` 是职责边界概念，不强制实现为新框架或 class。

---

## 6. Retrieval Router

Router 最终只回答：

1. 是否存在合法、明确、可审计的无检索收益理由？
2. 如果必须搜索，问题是单一快速核实还是完整证据搜索？

输出只有：

```text
skip
light
standard
```

### 6.1 skip

继续适用于纯闲聊、创作、翻译、改写、用户材料总结、纯数学、封闭逻辑和仅当前上下文任务。混合外部事实请求不能使用这些理由跳过搜索。

### 6.2 light

适用于单一、明确、容易通过一次检索核实的事实。它可以是稳定事实，也可以是简单当前事实。要求来源不自动改变其复杂度。

### 6.3 standard

适用于：

- 多事实或多子问题
- 多实体
- 比较
- 推荐或综合判断
- 明确要求多个来源、独立来源、交叉验证或多渠道核验
- 第一轮预期可能存在多个 Evidence 缺口

### 6.4 明确移除的升档因素

以下因素不能单独进入 standard：

- medical / legal / financial / safety 等领域标签
- high_stakes / high_consequence
- 迁移期间现有的 `Freshness.HIGH` 或任何“当前/高时效”信号
- `/search`
- “请搜索”“联网查一下”
- “请提供来源/出处”

显式搜索和来源请求只形成 must_search，并要求最终展示实际使用 Evidence 的引用。它们不扩大预算。

---

## 7. 请求级 Budget Contract

预算是 cap，不是 target。Evidence 已充分时必须提前停止。

单一绝对截止时间在 Router 输出 tier 后、Planner 开始前启动，覆盖：

```text
Planner
→ Provider
→ Reader
→ Evidence Evaluation
→ Gap Analysis
→ 可选 Repair
→ 最终 Evidence State
```

Answer、Validator 和 Renderer 单独计时，不占用检索预算，也不能触发搜索。

### 7.1 light 硬上限

```text
semantic_query_count       <= 1
unique_candidate_url_count <= 5
read_attempt_count         <= 2
round_count                <= 1
elapsed_retrieval_time     <= 8s
repair_count                = 0
```

### 7.2 standard 硬上限

```text
initial_semantic_query_count <= 3
repair_semantic_query_count  <= 1
total_semantic_query_count   <= 4
unique_candidate_url_count   <= 8
read_attempt_count           <= 5
round_count                  <= 2
elapsed_retrieval_time       <= 20s
repair_count                 <= 1
```

由用户原问题形成的 direct query 本身就是一个 semantic query，计入首轮最多3个 Query 的预算。Planner 最多只能在该 direct query 之外再生成2个补充 Query，不能实现成“原始 Query + 3个 Planner Query”。Planner 可以少生成；Evidence 预期简单时，standard 也可以只执行 direct query。Repair 仍最多1个，因此整个 standard 请求最多4个 semantic Query。

### 7.3 共享和计数规则

- 首轮与 Repair 共享全部 URL、读取和时间预算。
- Repair 不重新获得预算。
- DDGS/Tavily fallback reserve 包含在对应8秒和20秒截止时间内。
- Provider fallback 不增加语义 Query 或 round。
- 同一规范化最终 URL 的重复 hit 不重复占候选预算。
- 每次对一个候选 URL 发起完整正文获取动作即消耗一次读取预算。
- 该动作成功、失败、超时或最终降级使用 provider 内容，均不返还预算，也不重复计数。
- deadline 后不得启动 Provider、Reader 或 Repair，也不得发生晚 Trace 变更。

---

## 8. Planner 与轻量 Evidence Checklist

### 8.1 light

- 不建立 Checklist。
- 使用一个保留原问题语义的直接 Query。
- Freshness 可以增加日期、版本或时间窗约束，但不能增加 Query 数量。
- Evidence 不足时直接进入保守回答或无法确认，不允许 Repair。

### 8.2 standard

在现有 Planner 调用内同时生成：

- 首轮 direct query 之外最多2个补充 Query；direct query 已占用首轮 Query 预算中的1个名额
- 最多3个有限、可验证的 `required_topics`
- 必要来源关系或时间约束

不增加 Checklist LLM。原始问题持续作为语义锚点，模型不能删除或替换用户的实体、版本、时间、地域和范围条件。

---

## 9. Evidence Sufficiency Contract

保留四种状态：

- `SUFFICIENT`
- `PARTIAL`
- `CONFLICTING`
- `INSUFFICIENT`

确定性优先级：

```text
存在未解决的重要冲突
→ CONFLICTING

否则全部重要 required_topics 满足
→ SUFFICIENT

否则至少一个用户实际关心的重要主题获得有效支持，
且足以形成有意义、可独立陈述的局部回答
→ PARTIAL

否则
→ INSUFFICIENT
```

零散、边缘或不影响最终回答的 Evidence 不能把 INSUFFICIENT 提升为 PARTIAL。

### 9.1 Freshness

- `not_required`：没有材料性时效要求。
- `satisfied`：Evidence 满足指定日期、版本或当前性要求。
- `stale`：存在资料，但对当前材料性主题过旧。
- `unknown`：无法可靠确定资料时间或有效期。

Freshness eligibility 至少落实到每个 material required_topic，而不是只维护一个模糊的请求级结论。每个重要 topic 分别得到 `not_required / satisfied / stale / unknown`；只有 freshness 合格的 topic 才能进入 supported_topics。过旧或时间未知的 topic 进入 missing_topics，并记录 `stale_evidence`。

例如：

```text
A → satisfied
B → satisfied
C → stale

supported_topics = [A, B]
missing_topics   = [C]
evidence_state   = PARTIAL
repair_reason    = stale_evidence
repair_target    = C
```

这不要求新增 Freshness Engine；现有 Planner/Evidence 数据流只需能够表达 topic 与 freshness eligibility 的关系。

Freshness 不生成第五种 Evidence state。过旧或时间未知的 Evidence 不能支持对应当前主题，并形成 `missing_topics + stale_evidence`。

`SUFFICIENT + material stale/unknown` 是非法组合，必须由模型不变量和测试拒绝。

### 9.2 来源和冲突

- 第一方身份不能替代相关性和 Claim 支持。
- 用户没有要求多个来源时，Citation 要求不自动产生独立来源门槛。
- 用户明确要求独立或交叉核验时，source requirement 才成为 sufficiency 条件。
- CONFLICTING 优先于 PARTIAL，但不抹除无争议的受支持事实。

---

## 10. Repair Contract

Repair 只允许 standard 使用，并且必须同时满足：

- 第一轮 Evidence Evaluation 已完成；
- Evidence 不是 SUFFICIENT；
- 存在封闭 repair reason；
- 尚未执行 Repair；
- 仍有 Query、URL、读取和时间预算；
- 存在明确 target topic；
- 可以形成与首轮不同、保留有效原约束的 Query。

封闭原因：

- `missing_topic`
- `stale_evidence`
- `source_conflict`
- `entity_ambiguity`
- `premise_mismatch`
- `source_quality_gap`
- `content_unreadable`

原因不要求全部由 EvidenceAssembler 发现：

- Reader 可产生 `content_unreadable`；
- Freshness evaluation 可产生 `stale_evidence`；
- Evidence 可产生 `missing_topic / source_conflict`；
- Planner/Judge 可产生 `entity_ambiguity / premise_mismatch`；
- 来源评价可产生 `source_quality_gap`。

Orchestrator 只聚合为：

```text
repair_reason_codes
repair_target_topics
repair_query
```

### 10.1 Repair 收益的确定性推导

不新增 Repair Benefit LLM。

例如：

- missing topic 且可形成针对性不同 Query → 可 Repair；
- stale evidence 且可增加时间/版本/官方来源约束 → 可 Repair；
- conflict 且可指向冲突事实和更原始来源 → 可 Repair；
- unreadable 且存在替代来源方向 → 可 Repair；
- 无剩余预算、无法形成不同 Query 或没有合理目标 → 停止。

### 10.2 Pivot / Backtrack

Pivot / Backtrack 仅表现为 `premise_mismatch / entity_ambiguity / source_quality_gap` 等 Repair 原因，不建立独立状态机。

Repair Query 必须保留原问题中仍有效的实体、版本、时间、地域和范围限制。

### 10.3 Post-Repair Stop

第二轮完成后必须：

```text
重新读取
→ 重新映射 Evidence
→ 重新计算 Sufficiency
→ 无条件结束 Search Pipeline
→ 进入 Answer Policy
```

禁止第三轮、Repair-of-Repair、Validator-driven Search 或风险驱动补搜。

---

## 11. Answer Policy 与 Failure Matrix

Answer Policy 只消费状态，不修改 Search Tier、Evidence 或检索生命周期。

逻辑 answer_state 表达：

```text
certainty
allowed_claim_scope
disclosure_codes
warning_codes
validator_requirement
```

`certainty`：

- `verified`
- `limited`
- `conflicting`
- `unverified`

`allowed_claim_scope`：

- `all_supported`
- `supported_subset`
- `supported_subset_with_conflicts`
- `conflict_description_only`
- `no_external_factual_claims`

### 11.1 Failure Matrix

| Evidence | 普通请求 | `warning_required=true` 的高后果请求 |
|---|---|---|
| SUFFICIENT | 展示所有通过 Evidence 和 Validator 的 Claim；不显示专业风险警告 | 只展示受支持内容并遵循行动边界；恰好一次固定警告 |
| PARTIAL | 只回答有意义的受支持子集；明确其他部分未确认 | 只回答受支持子集，不补行动结论；incomplete disclosure + 一次警告 |
| CONFLICTING | 正常陈述无争议的受支持事实并单独展示冲突；核心问题完全被冲突覆盖时只描述冲突 | 同左，但不给行动性裁决；一次警告 |
| INSUFFICIENT | 不使用模型记忆生成外部事实；明确无法在线确认 | 不输出具体建议、剂量、责任或行动结论；失败披露 + 一次警告 |

警告由 risk_context 和 Answer Policy 决定，不由领域名称或 Renderer 决定。

搜索或读取失败先转换为 Evidence/Failure metadata，再由矩阵决定回答：

- Provider 未配置或均不可用 → INSUFFICIENT；
- Provider 超时/错误且有部分有效 Evidence → 按 Evidence 计算 PARTIAL/CONFLICTING；
- 无相关结果 → INSUFFICIENT，不等于事实不存在；
- content unreadable → standard 在有剩余预算时可 Repair；
- 预算耗尽 → 立即停止并使用现有 Evidence state；
- Repair 后仍不足 → 按最终状态回答，不再搜索。

---

## 12. Validator Contract

Validator 接收：

```text
draft
Evidence
answer_state
```

状态只有：

- `passed`
- `filtered`
- `unavailable`
- `malformed`

Validator 可以：

- 验证 Claim–Evidence；
- 删除 unsupported claim/block；
- 检测隐藏事实、Citation 错配和结构错误；
- 按 validator requirement 采取普通或 fail-closed 策略；
- 降低 certainty 和缩小 visible blocks。

Validator 不可以：

- 添加 Claim；
- 创造 URL/Citation；
- 修改 Evidence 或 evidence_state；
- 提高 certainty；
- 搜索、Repair 或要求模型重新搜索。

处理规则：

- passed：按 answer_state 输出。
- filtered：保留通过块，必要时降低 answer certainty；原 evidence_state 不变。
- unavailable：高后果或高时效当前事实采用 fail-closed；普通稳定事实只可保留结构完整、映射真实 Evidence、通过全部确定性校验且无隐藏事实的块，并显示验证未完成披露。
- malformed：不触发重写或搜索，进入固定验证失败 render state；需要的高危 warning 仍保留。

Evidence state 表达搜索阶段得到了什么，Validator status 表达草稿成功利用了多少 Evidence。两者不能互相覆盖。

---

## 13. render_state 与 Renderer

逻辑 render_state 包含：

```text
outcome
visible_blocks
citation_map
used_sources
conflict_groups
disclosure_codes
warning_codes
```

Renderer 只能：

- 选择确定性模板；
- 编号和渲染 Citation；
- 输出实际使用的来源；
- 输出 Conflict、Partial、Failure disclosure；
- 输出已经决定的 warning；
- 执行 QQ 长度分段。

Renderer 不得：

- 判断医疗、法律、金融或其他风险领域；
- 判断 Freshness；
- 重新计算 Evidence state；
- 决定是否 Repair；
- 根据自然语言自行增删 warning；
- 输出“搜索成功”“检索完成”等内部状态。

模型自行生成的状态横幅或风险提示必须在进入 render_state 前由 Validator/结构化清理处理。

---

## 14. Trace 与隐私

Trace 分层记录 Retrieval、Evidence、Answer、Validation 和 Rendering。

### 14.1 Retrieval

- must_search
- route
- retrieval complexity reason codes
- Query metadata
- Provider attempts
- URL/read/round/budget counters
- Repair reason、target topics 和唯一 Query metadata
- stop reason：`evidence_sufficient / no_repair_benefit / budget_exhausted / post_repair_stop`

Query trace 只允许记录：

- `query_index`
- `query_purpose` 或封闭 reason code
- `round`
- `provider`
- `success/failure`
- `latency`
- 可选的不可逆 fingerprint

禁止记录 raw query text。

### 14.2 Evidence

- immutable evidence_state
- supported/missing topics
- freshness result
- conflict group 数量
- Provider/Reader failure metadata

### 14.3 Answer / Validation / Rendering

- answer certainty、allowed scope、disclosure/warning codes
- validator status 和 retained/removed block/claim 数量
- render outcome、Citation 和 used-source 数量

Trace 禁止记录用户正文、Evidence 正文、完整 URL、私密记忆、QQ 标识或可逆敏感内容。

---

## 15. LLM 调用原则

本轮禁止新增：

- Risk LLM
- Freshness LLM
- Checklist LLM
- Gap LLM
- Repair Benefit LLM
- Pivot/Backtrack LLM
- Stop LLM
- Renderer LLM

保留现有必要职责：

- 一次早期请求分析；
- standard 的现有 Planner；
- Evidence Judge；
- Grounded Answer；
- Claim Discovery；
- Semantic Validation。

Repair 后候选 Evidence 已变化，允许复用现有 Evidence Judge 再判断一次。

本轮不强行合并 Claim Discovery 和 Semantic Validator。合并必须由人工标注评测证明 Precision/Recall 不下降，作为独立后续任务。

light、standard 无 Repair、standard 有 Repair 三条路径的最大 LLM 调用数不得高于当前实现。Context、Sufficiency、Gap、Repair Benefit、Stop 和 Rendering 都不能新增调用。

---

## 16. 分阶段迁移

### 阶段一：建立行为基线

固定现有路由、预算、Repair、失败、Citation、deadline、warning 和 Trace 行为；标记哪些是不变量，哪些是本轮有意改变的 deep 行为。

### 阶段二：停用 deep 行为

生产 Router 不再输出 deep；停用 deep budget、Query、fallback reserve、Evidence 和失败分支。旧兼容字段可以暂时保留，但生产 Trace 不得再产生 deep。

### 阶段三：解耦 Router

一次早期分析形成三个逻辑 context；Router 只消费 retrieval_context；Risk、Freshness 和显式搜索不再升档。

### 阶段四：统一 standard Repair

接入 required/supported/missing topics、统一 reason/target/query，light 永不 Repair，第二轮后无条件停止。

### 阶段五：迁移 Answer/Failure Policy

删除 `if tier == deep: more conservative`，使用 Evidence、Freshness、Risk 和 Validator 状态生成 answer/render state；Renderer 清除语义决策。

### 阶段六：删除历史垃圾

最后删除失去行为作用的 enum、字段、Prompt、Trace、配置、fixture、兼容逻辑和 dead branch。

不要在迁移开始时大规模重构 Models。先迁移行为，再清 schema。

---

## 17. 删除与保留清单

### 17.1 最终删除

- `SearchTier.DEEP`
- deep rank/budget/provider reserve
- deep Planner Prompt、fallback Query 和 purpose 分支
- deep Evidence admission/repair/failure/validation 分支
- `model_recommended_tier=deep`
- 只用于风险或 Freshness 升档的 floor/trigger 字段
- RetrievalDecision 中已迁入 Risk/Freshness Context 的重复字段
- Trace/evaluator/fixture/README 中的 operational deep
- 无调用方兼容逻辑和 dead branch

### 17.2 保留

- 封闭 skip reasons 和事实默认搜索原则
- DDGS-first/Tavily conditional fallback
- deadline/race/attempt truth
- Provider/Reader/Evidence 分层
- relevance gate、来源归因、Freshness 和 conflict
- Claim discovery、semantic validator、Citation verifier
- 失败显式披露
- body-free Trace
- 最多一次 Repair

---

## 18. 测试矩阵

### 18.1 Router

- 纯聊天/创作/翻译/数学为 skip。
- 显式搜索只保证非 skip。
- 单一明确事实和单一来源请求可为 light。
- 多事实、多实体、比较、推荐、交叉核验为 standard。
- 迁移期间现有 `Freshness.HIGH` 或任何“当前/高时效”信号均不得提升 tier；最终实现应删除已经失去行为作用的 HIGH tier-control 字段。
- warning_required 不改变 tier。
- 同一 retrieval_context 在不同 Risk/Freshness 下得到相同 tier。

### 18.2 Budget

light：

```text
semantic_query_count       <= 1
unique_candidate_url_count <= 5
read_attempt_count         <= 2
round_count                <= 1
elapsed_retrieval_time     <= 8s
repair_count                = 0
```

standard：

```text
initial_semantic_query_count <= 3
repair_semantic_query_count  <= 1
total_semantic_query_count   <= 4
unique_candidate_url_count   <= 8
read_attempt_count           <= 5
round_count                  <= 2
elapsed_retrieval_time       <= 20s
repair_count                 <= 1
```

首轮计数必须包含由用户原问题形成的 direct query，因此首轮最多是1个 direct query 加2个补充 Query；Repair 最多再增加1个不同 Query。

另需验证提前停止、预算共享、单 URL 单次读取计数、fallback 非新 Query/round、deadline 后无晚调用。

### 18.3 Evidence 与 Repair

- 状态优先级固定。
- 边缘 Evidence 不能产生 PARTIAL。
- material stale/unknown 不能产生 SUFFICIENT。
- 无争议事实与冲突事实可以共同展示。
- light 永不 Repair。
- standard 只有明确 Gap 才 Repair。
- 七种 reason 均有正负测试。
- 无预算、无不同 Query、无 target 时不 Repair。
- Repair 保留有效原约束。
- 第二轮后永不搜索。

### 18.4 Answer、Validator、Renderer

- Evidence state 在 Answer/Validator 后保持不变。
- Validator 只能降低 certainty 和缩小 visible blocks。
- supported_subset_with_conflicts 保留无争议内容。
- warning 依赖 warning_required，不依赖领域名。
- 高后果或高时效下 validator unavailable fail-closed。
- 普通稳定事实只保留确定性校验通过的块。
- Renderer 不读取 Risk/Freshness/Search Tier。
- 成功无状态横幅；Warning 恰好一次；只显示 used sources；Citation 不悬空。

### 18.5 回归

- Provider deadline/reserve/race sealing
- Reader HTML/PDF/redirect/oversize/failure
- prompt injection 边界
- URL validation/canonicalization
- Claim–Evidence/citation/conflict
- 普通聊天与 `/search` 共用管线
- package-aware hermetic 全量测试
- external HTTP guard

---

## 19. 验收标准

### 19.1 结构

- Runtime tier 只有 skip/light/standard。
- 生产代码不存在 operational deep branch。
- Risk/Freshness 不被 Router 消费。
- 只有 standard 可以 Repair。
- 不存在第三轮或 Answer/Validator → Search 回环。
- Renderer 不含风险领域、Freshness 或 Evidence 语义判断。
- 没有新增 Agent、状态机或 LLM 阶段。

### 19.2 复杂度

运行时概念、operational branch、enum/state、tier-specific logic、重复条件和双向依赖应总体下降。

生产代码净增加不是自动失败；如果净增加，必须说明增加部分用于明确职责或可靠性，而不是新增框架复杂度。LOC 只作为辅助指标。

### 19.3 验证

- 搜索聚焦套件通过。
- Router/Planner/Evidence/Orchestrator/Validation/Renderer/Chat 集成通过。
- `python -B -m unittest discover -s tests -t . -v` 通过。
- `python -B -m compileall -q src tests` 通过。
- `git diff --check` 通过。
- deep/risk/freshness/反向搜索静态检查通过。
- 测试期间无真实外部 HTTP。
- `eval/search/*.jsonl` 未经所有者复核不得伪造修改。

### 19.4 外部门槛

当前140条评测均未完成真实人工复核，另有两条非法 `potential_harm=medium`。fixture baseline、绿色 unit tests 和空 semantic sample 不能作为真实质量证明。

真实 DDGS/Tavily 测试必须：

- 获得单独明确授权；
- 使用公开、非敏感固定用例；
- 不发送 QQ 标识、历史、记忆、图片或密钥；
- 保存脱敏 Trace 和时间戳；
- 由人工审查 Query、Evidence、Claim、Citation 和失败披露。

本轮可以证明架构符合规格、hermetic 行为正确且既有可靠性不变量保留；不能仅据此声称真实互联网检索质量达到目标。

---

## 20. 规格复核后的下一步

只有本规格经项目所有者确认后，才使用 writing-plans 生成逐文件、逐接口、逐测试、逐提交的实施计划。

规格复核前不得：

- 修改生产代码；
- 固化最终 class/字段名称；
- 全局删除 deep；
- 修改评测标签；
- 执行真实在线搜索。
