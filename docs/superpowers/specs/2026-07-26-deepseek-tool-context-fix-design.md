# DeepSeek 工具上下文与最终总结修复设计

## 1. 目标

修复模型链实现中的三个相关问题：

1. DeepSeek V4 默认思考模式产生工具调用后，下一次请求缺少必须回传的 `reasoning_content`。
2. 达到工具调用轮数上限后的最终总结请求仍允许模型继续调用工具。
3. `_model_supports_tools()` 通过模型名是否包含 `reasoner` 或 `r1` 推断工具能力，可能误伤用户自行配置的模型。

修复后应继续保留：

- `CHAT_MODELS` 的顺序和回退行为。
- Gemini 原生 `generateContent`、thought signature 和函数调用 ID 往返。
- DeepSeek OpenAI 兼容接口。
- 本地聊天历史、图片理解、网页搜索、记忆和并发行为。
- provider context 只在当前工具循环中短暂存在，不持久化。

## 2. 已确认的根因

### 2.1 DeepSeek 思考上下文丢失

`DeepSeekClient.chat()` 当前只把响应的 `content` 和 `tool_calls` 转换为 `ChatResponse`。DeepSeek V4 默认启用思考模式；当响应包含工具调用时，官方协议要求后续请求在 assistant 消息中完整回传同一次响应的 `reasoning_content`。

由于 `ChatResponse` 没有携带 DeepSeek 的该字段，`chat_service.build_tool_messages()` 创建的临时 assistant 消息也无法恢复它，导致下一次 DeepSeek 工具请求可能返回 HTTP 400。

### 2.2 最终总结仍使用自动工具选择

达到 `MAX_TOOL_CALL_ROUNDS` 后，当前代码为了保留协议所需的工具声明而再次传入 `tools`，但没有设置 `tool_choice`。Gemini 和 DeepSeek 均将此解释为自动选择，模型仍可能返回新的工具调用；调用方却只读取文本内容，因而产生空回复或通用兜底文本。

### 2.3 模型名启发式判断过宽

工具能力判断对任何包含 `reasoner` 或 `r1` 的模型名返回不支持工具。该规则既不属于用户配置格式，也不能代表当前提供商能力，并会覆盖用户明确填写的模型。

## 3. 方案

### 3.1 使用现有 provider context 传递 DeepSeek 协议状态

继续使用 `ChatResponse.provider_context`，不新增持久化结构。

DeepSeek 响应包含工具调用且 `reasoning_content` 是字符串时，返回：

```python
provider_context={
    "provider": "deepseek",
    "reasoning_content": reasoning_content,
}
```

如果没有工具调用，不需要保留 `reasoning_content`。

`chat_service.build_tool_messages()` 已把 provider context 深拷贝到当前工具循环的临时 assistant 消息中，因此不改变它的存储边界。

### 3.2 DeepSeek 请求按提供商恢复私有字段

`DeepSeekClient.chat()` 清理每条内部消息时：

- 总是删除 `_provider_context`，禁止私有字段进入远端请求。
- 如果 `_provider_context.provider == "deepseek"` 且 `reasoning_content` 是字符串，则把它恢复为同一条 assistant 消息的公开 OpenAI 兼容字段 `reasoning_content`。
- Gemini provider context 只被删除，不转换。
- 不修改 user、tool、system 消息，也不伪造缺失的 reasoning content。

这样 DeepSeek 工具调用可以继续其思考上下文，Gemini 元数据也不会泄漏给 DeepSeek。

### 3.3 最终总结明确禁止继续调用工具

最终总结调用改为同时传入：

```python
tools=tools
tool_choice="none"
```

保留 `tools` 是为了让原生 Gemini 和 DeepSeek 能正确理解历史中的函数调用与结果；`tool_choice="none"` 确保本次只生成总结文本。

### 3.4 删除模型名子串判断

删除：

```python
if "reasoner" in key or "r1" in key:
    return False
```

保留已有的明确能力表，例如已确认不支持工具的 `gemma-4-26b-a4b-it`。其他用户配置模型默认尝试工具请求，并由实际 API 响应和模型链回退处理不支持情况。

## 4. 数据流

DeepSeek 工具回合的数据流：

```text
DeepSeek response
  ├─ content
  ├─ tool_calls
  └─ reasoning_content
          ↓
ChatResponse.provider_context
          ↓
临时 assistant._provider_context
          ↓
DeepSeekClient 恢复 assistant.reasoning_content
          ↓
下一次 DeepSeek 请求
```

如果模型链切换到 Gemini，Gemini 客户端只识别 `provider == "gemini"` 的上下文；DeepSeek 上下文不会变成 Gemini 原生 Part。

## 5. 错误处理与边界

- `reasoning_content` 缺失或不是字符串时不恢复，不自行构造内容。
- provider context 只附着在当前 `generate_reply()` 的局部消息列表。
- 最终文本仍只把用户文本占位符和助手回答写入历史。
- DeepSeek 空 API Key 防护保持不变。
- 不增加思考模式环境变量，不改变 `.env` 或 `.env.example`。
- 不调用真实模型进行自动化测试，避免泄露 Key 和产生费用。

## 6. 测试设计

先写测试并确认失败，再修改生产代码。

### 6.1 DeepSeek reasoning context

- DeepSeek 返回 `reasoning_content + tool_calls` 时，`ChatResponse.provider_context` 正确保留该字段。
- 下一次 DeepSeek 请求把上下文恢复到对应 assistant 消息。
- 请求 JSON 不包含 `_provider_context`。
- Gemini provider context 不会转换为 DeepSeek `reasoning_content`。
- 无工具调用的普通文本响应不保留 DeepSeek reasoning context。

### 6.2 最终总结

- 前两轮均返回工具调用时，第三次调用仍带 `tools`。
- 第三次调用明确设置 `tool_choice="none"`。
- 第三次文本回答正常返回，不执行第四轮工具。

### 6.3 工具能力判断

- 明确能力表中的不支持模型仍返回 `False`。
- 名称包含 `reasoner` 或 `r1` 的其他模型不再被子串规则跳过。
- DeepSeek V4 模型继续允许工具调用。

### 6.4 回归

- Gemini thought signature、函数 ID 和跨提供商上下文测试继续通过。
- 完整 unittest、编译、补丁格式和旧配置扫描全部通过。

## 7. 验收标准

- DeepSeek 工具回合完整回传同一次响应的 `reasoning_content`。
- DeepSeek 请求不包含 `_provider_context` 或 Gemini 原生上下文。
- 最终总结请求不能再次产生工具调用。
- 工具能力不再由 `reasoner`/`r1` 子串推断。
- 所有自动化测试通过。
- 工作区只包含预期修复文件，没有 `.env`、本地数据或无关改动。
