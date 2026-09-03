# Tavily 优先与 DDGS 兜底设计

## 目标

调整证据化网页搜索的 Provider 顺序：每个语义查询优先使用 Tavily，只有 Tavily 未解决该查询时才调用 DDGS。所有 DDGS 搜索批次的独立阶段超时统一提高到 30 秒。

## 范围

本次只调整 Provider 调度顺序、DDGS 阶段预算、对应测试与用户文档。不增加环境变量，不改变搜索分档、查询规划、证据组装、回答验证、失败披露或 SearchTrace 数据结构。

## Provider 调度

检索轮次继续采用现有的两批查询级回退机制：

1. 将当前轮次的全部查询并发提交给 Tavily。
2. Tavily 返回有效结果的查询视为已解决，不再调用 DDGS。
3. Tavily 未配置、不可用、报错、超时、无结果或未产生有效 URL 时，仅把对应未解决查询并发提交给 DDGS。
4. 一个查询失败不丢弃同批其他已完成查询的结果。
5. 初始检索轮次与唯一一次 repair 轮次遵循相同顺序。

因此，在未配置 `TAVILY_API_KEY` 时，查询会快速经过 Tavily 的 `NOT_CONFIGURED` 状态，再由 DDGS 兜底，DDGS 仍可独立维持搜索能力。

## 时间预算

`src/search/budget.py` 中所有 DDGS Provider 阶段预算设为 30 秒：

- light：`initial_ddgs_seconds=30`
- standard：`initial_ddgs_seconds=30`
- standard repair：`repair_ddgs_seconds=30`

Tavily 和其他阶段预算保持不变。总请求 watchdog 继续由各阶段预算之和自动推导，不另设固定常量。由此 light 上限由约 34 秒增加为约 58 秒，standard 上限由约 65 秒增加为约 112 秒。这些数值是非协作依赖的安全上限，不是预期响应时间；阶段完成后仍立即进入下一阶段。

## 可观测性与失败语义

保留现有查询级 `ProviderAttempt`、`QueryTraceEntry`、Provider readiness 和失败码。Trace 中的尝试顺序应反映 Tavily 在前、DDGS 在后。最终失败聚合与面向用户的在线检索失败披露保持不变。

## 测试

采用测试先行，覆盖：

- Tavily 成功时不调用 DDGS。
- Tavily 为空、超时、报错、不可用或未配置时调用 DDGS。
- 多查询批次中，仅 Tavily 未解决的查询进入 DDGS。
- 初始轮次和 repair 轮次均使用 Tavily → DDGS 顺序。
- light、standard 初始以及 standard repair 的 DDGS 阶段预算均为 30 秒。
- README 的 Provider 顺序和 watchdog 数值与运行时代码一致。

完成专项测试后运行完整 unittest 与 compileall。

## 兼容性

现有 `.env` 无需修改。`TAVILY_API_KEY` 仍是可选项；未配置时自动使用 DDGS。`SEARCH_MAX_RESULTS`、模型链、OneBot 接口和持久化数据格式均不受影响。
