# `.env` 对话模型链配置设计

**日期：** 2026-07-23
**状态：** 已确认，等待实施计划
**目标：** 让用户只通过一个 `CHAT_MODELS` 环境变量选择对话主模型和回退模型，在启动阶段严格发现配置错误，并将 Gemini 从 OpenAI 兼容接口迁移到原生 `generateContent` API。

## 1. 背景

项目目前用以下多组变量共同描述模型链：

- `GEMINI_MODEL`
- `DEEPSEEK_MODEL`
- `LLM_PROVIDER`
- `LLM_PRIMARY_PROVIDER`
- `LLM_PRIMARY_MODEL`
- `LLM_FALLBACK_1_PROVIDER` / `LLM_FALLBACK_1_MODEL`
- `LLM_FALLBACK_2_PROVIDER` / `LLM_FALLBACK_2_MODEL`
- `LLM_FALLBACK_3_PROVIDER` / `LLM_FALLBACK_3_MODEL`

这些变量可以配置主模型和回退模型，但入口较多，用户需要理解变量优先级、默认模型和固定回退槽位。新设计将模型名称和顺序统一到一个变量中。

Gemini 当前通过 Google 的 OpenAI 兼容聊天端点调用。项目已经自行保存会话历史，并需要在 Gemini 与 DeepSeek 之间共享同一份上下文和执行回退，因此本次同时迁移到 Gemini 原生、无状态的 `generateContent` REST API。

## 2. 已确认的产品决策

1. 支持一个主模型和任意数量的回退模型。
2. 只支持项目现有的 `gemini` 和 `deepseek` 提供商。
3. 使用单行 `CHAT_MODELS` 配置模型链。
4. 删除旧模型选择变量，不提供兼容读取。
5. 配置错误时拒绝启动，不静默采用默认模型。
6. 模型链中出现的每个提供商都必须配置对应 API Key。
7. 模型的工具调用和图片输入能力由运行时自动处理，不要求用户声明。
8. Gemini 使用原生 `generateContent` REST API，不使用 OpenAI 兼容接口或 Interactions API。
9. 本地聊天历史是唯一权威会话状态；Gemini 请求不使用服务端 Interaction ID。

## 3. 用户配置

完整示例：

```dotenv
GEMINI_API_KEY=你的_Gemini_Key
GEMINI_URL=https://generativelanguage.googleapis.com/v1

DEEPSEEK_API_KEY=你的_DeepSeek_Key
DEEPSEEK_URL=https://api.deepseek.com/chat/completions

# 从左到右：主模型、回退模型
CHAT_MODELS=gemini:gemini-3.1-flash-lite,deepseek:deepseek-v4-flash
```

只使用一个模型：

```dotenv
CHAT_MODELS=gemini:gemini-3.1-flash-lite
```

同一提供商使用多个模型：

```dotenv
CHAT_MODELS=gemini:gemini-3.1-flash-lite,gemini:gemma-4-26b-a4b-it
```

跨提供商回退：

```dotenv
CHAT_MODELS=gemini:gemini-3.1-flash-lite,deepseek:deepseek-v4-flash
```

第一个有效项目是主模型，后续项目依次作为回退模型。实际尝试顺序必须与配置顺序一致。

## 4. 语法与校验规则

### 4.1 解析规则

- 使用英文逗号分隔模型项目。
- 每个项目使用第一个英文冒号分隔提供商和模型名。
- 提供商去除首尾空格并转为小写。
- 模型名只去除首尾空格，其他字符和大小写保持原样。
- 因为只在第一个冒号处分隔，所以模型名内部允许包含冒号。
- 模型名不能包含英文逗号；逗号固定作为模型项目分隔符。
- 重复的“提供商 + 模型名”组合只保留第一次出现的位置。

例如：

```dotenv
CHAT_MODELS= Gemini : gemini-3.1-flash-lite , DEEPSEEK : deepseek-v4-flash
```

解析为：

```text
1. gemini / gemini-3.1-flash-lite
2. deepseek / deepseek-v4-flash
```

### 4.2 启动失败条件

以下情况抛出 `ModelConfigurationError` 并停止启动：

- `CHAT_MODELS` 缺失或为空。
- 出现空项目，例如连续逗号或末尾多余逗号。
- 项目没有冒号。
- 提供商为空。
- 模型名为空。
- 提供商不是 `gemini` 或 `deepseek`。
- 模型链使用 Gemini，但 `GEMINI_API_KEY` 缺失或只有空白。
- 模型链使用 DeepSeek，但 `DEEPSEEK_API_KEY` 缺失或只有空白。

校验只验证本地可确认的信息，不在启动时请求模型服务。因此，模型名是否真实存在、账号是否有权限以及模型是否支持特定能力，仍由实际 API 调用结果确定。

当模型链包含 Gemini 时，`GEMINI_URL` 必须是 Gemini Developer API 基础地址。默认值为：

```text
https://generativelanguage.googleapis.com/v1
```

客户端在该基础地址后追加 `/models/{model}:generateContent`。旧的 `/v1beta/openai/chat/completions` 地址不再兼容；若检测到 `/openai/` 路径，应在启动阶段明确提示用户迁移。

### 4.3 错误信息

错误信息必须：

- 使用中文说明出错变量和模型项目位置。
- 给出合法格式示例。
- 不输出 API Key、Token 或完整环境变量内容。
- 以非零进程退出码结束。

示例：

```text
模型配置错误：CHAT_MODELS 第 2 项缺少模型名：deepseek:
正确格式：gemini:模型名,deepseek:模型名
```

## 5. Gemini API 选择

### 5.1 采用 `generateContent`

虽然 Google 自 2026 年 6 月起推荐新项目使用 Interactions API，原有 `generateContent` API 仍受完整支持。当前项目采用 `generateContent`，原因是：

- 项目已在本地按 QQ 会话保存和裁剪历史。
- 同一会话可能从 Gemini 回退到 DeepSeek，两个提供商必须使用同一份本地上下文。
- `/reset`、历史持久化和记忆模块都以本地会话为边界。
- 当前仅需要聊天、图片理解和一个客户端搜索函数，不需要后台 Agent 或复杂执行步骤。
- `generateContent.contents` 与项目现有“每次发送完整上下文”架构直接对应。

Interactions API 的 `previous_interaction_id` 只能供 Gemini 使用，无法供 DeepSeek 回退使用。如果同时维护本地历史和 Interaction ID，会引入两套状态。若改用 `store=false`，Interactions 又要求客户端保存并回传模型产生的完整 Steps，包括工具调用相关步骤，复杂度高于 `generateContent`，且不能发挥服务端状态的主要优势。

官方参考：

- <https://ai.google.dev/gemini-api/docs/interactions-overview>
- <https://ai.google.dev/api/generate-content>
- <https://ai.google.dev/gemini-api/docs/function-calling>

### 5.2 使用原生 REST 而不是 SDK

项目继续通过现有 HTTP 请求工具调用 Gemini 原生 REST API，不新增 Google GenAI SDK 依赖，原因是：

- 现有代理、超时和 HTTP 错误处理可以直接复用。
- DeepSeek 与 Gemini 仍通过统一客户端接口接入回退链。
- 避免 SDK 默认 API 版本和依赖升级改变现有行为。
- 请求和响应转换可以被独立单元测试。

默认使用稳定的 `v1` 基础地址。若将来某个预览模型或功能只支持 `v1beta`，用户可以通过 `GEMINI_URL` 明确切换基础版本。

## 6. 内部架构

### 6.1 `src/model_config.py`

新增独立模块，包含：

- `ConfiguredModel`：不可变的提供商和模型名数据结构。
- `ModelConfigurationError`：模型配置专用异常。
- 纯解析与校验函数：接收原始 `CHAT_MODELS` 和所需 Key 信息，返回不可变模型列表。
- 支持的提供商及其 API Key 变量映射。

该模块不调用模型服务，也不依赖具体 LLM 客户端，便于独立测试。

建议的数据结构：

```python
@dataclass(frozen=True)
class ConfiguredModel:
    provider: str
    model: str
```

### 6.2 `src/config.py`

`Config`：

- 保留 `GEMINI_API_KEY`、`GEMINI_URL`、`DEEPSEEK_API_KEY` 和 `DEEPSEEK_URL`。
- 新增解析后的 `chat_models` 字段。
- 删除 `gemini_model`、`deepseek_model` 及全部旧 `LLM_*` 模型链字段。
- 在全局配置初始化期间完成严格校验。

### 6.3 `src/services/llm_client.py`

`_build_chain()`：

- 直接遍历 `config.chat_models`。
- 将每个 `ConfiguredModel` 转换为现有 `LLMModelSpec`。
- 继续通过现有能力注册表设置 `supports_tools`。
- 不再推导默认模型，不再读取固定 fallback 槽位，也不再负责去重。

`FallbackLLMClient` 的以下行为保持不变：

- 主模型失败后尝试下一个模型。
- 缺少工具能力的已知模型在工具请求中提前跳过。
- 网络错误、限流、服务端错误和无效响应触发回退。
- 图片请求在当前模型失败后继续尝试模型链。
- 全部文本模型失败时返回统一错误。
- 全部图片模型失败时返回图片识别专用错误。

### 6.4 `src/services/gemini_client.py`

`GeminiClient` 保持现有统一 `chat()` 方法签名，对调用方继续接受项目内部的标准消息和工具格式，但在客户端边界转换为 Gemini 原生结构。

请求地址：

```text
POST {GEMINI_URL}/models/{model}:generateContent
```

插入路径前必须将模型名编码为单个 URL 路径组件，再追加 `:generateContent` 方法后缀，避免模型名中的冒号或其他特殊字符破坏请求路径。

认证方式：

```text
x-goog-api-key: GEMINI_API_KEY
```

不再使用 Bearer Token，也不在 JSON 请求体中发送 `model` 字段。

#### 消息转换

- `system` 消息合并为 `systemInstruction.parts[].text`。
- `user` 消息转换为 `contents[].role = "user"`。
- `assistant` 消息转换为 `contents[].role = "model"`。
- 普通文本转换为 `parts[].text`。
- OpenAI 风格的图片 data URL 解码为 `inlineData.mimeType` 和 `inlineData.data`。
- 既有聊天历史继续由项目本地生成并在每次请求中完整发送。

#### 工具转换

- 内部 `tools[].function` 转换为 Gemini `tools[].functionDeclarations[]`。
- 内部 `tool_choice="auto"` 转换为 Gemini `toolConfig.functionCallingConfig.mode = "AUTO"`。
- Gemini 返回的 `functionCall` Part 转换回项目现有 `ChatResponse.tool_calls` 结构。
- 内部 assistant tool call 与 `role="tool"` 结果转换为 Gemini 的 `functionCall` 和 `functionResponse` Parts。
- 如果 Gemini 原生函数调用没有项目内部需要的 call ID，适配器生成只在当前本地调用链内使用的稳定 ID；回传 Gemini 时按原生协议使用函数名和结果，不把该内部 ID 当作远端会话状态。

#### 生成参数和响应

- `temperature` 写入 `generationConfig.temperature`。
- `max_tokens` 写入 `generationConfig.maxOutputTokens`。
- 聚合首个有效 Candidate 中所有文本 Parts。
- 将所有函数调用 Parts 转换为内部工具调用列表。
- 安全拦截、空 Candidate、无文本且无函数调用等结构化异常转换为明确的运行时错误，使外层模型链可以继续回退。
- HTTP 429、5xx、连接失败和超时继续使用现有回退判断。

### 6.5 `run_bot.py`

启动入口捕获 `ModelConfigurationError`：

- 向标准错误输出简短中文配置错误。
- 返回非零退出码。
- 不启动 Flask 服务。
- 保持正常导入和启动路径不变。

由于当前全局 `Config` 在导入 `src.main` 时初始化，`run_bot.py` 需要在导入主模块的边界捕获该专用异常。

### 6.6 健康检查

`/health` 增加不含秘密的模型链信息，例如：

```json
{
  "chat_models": [
    {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
    {"provider": "deepseek", "model": "deepseek-v4-flash"}
  ]
}
```

健康检查不得返回 API Key。

## 7. 删除与保留的环境变量

### 7.1 保留

- `GEMINI_API_KEY`
- `GEMINI_URL`，含义调整为 Gemini Developer API 基础地址，默认 `https://generativelanguage.googleapis.com/v1`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_URL`

### 7.2 新增

- `CHAT_MODELS`

### 7.3 删除

- `GEMINI_MODEL`
- `DEEPSEEK_MODEL`
- `LLM_PROVIDER`
- `LLM_PRIMARY_PROVIDER`
- `LLM_PRIMARY_MODEL`
- `LLM_FALLBACK_1_PROVIDER`
- `LLM_FALLBACK_1_MODEL`
- `LLM_FALLBACK_2_PROVIDER`
- `LLM_FALLBACK_2_MODEL`
- `LLM_FALLBACK_3_PROVIDER`
- `LLM_FALLBACK_3_MODEL`

旧变量即使仍留在用户 `.env` 中也不会被读取；若没有新 `CHAT_MODELS`，启动必须失败并提示迁移。

## 8. 文档和配置模板

需要同步：

- `.env.example`：删除旧模型选择变量，新增带注释的 `CHAT_MODELS`，并将 `GEMINI_URL` 改为原生 `v1` 基础地址。
- `README.md`：更新最小配置、完整参数表、模型回退说明、Gemini 原生接口、错误排查和迁移示例。
- 文档一致性测试：继续从运行时代码提取环境变量集合，不硬编码变量数量。

README 当前存在用户尚未提交的修改。实施时必须保留并合并这些修改，不得整体覆盖该文件。

真实 `.env` 不在本设计阶段修改。后续实施也只有在用户明确授权后才能迁移真实 `.env`。

## 9. 测试方案

### 9.1 解析单元测试

- 单模型。
- 多模型及顺序。
- 首尾空格。
- 提供商大小写归一。
- 模型名保持原样。
- 模型名内部包含冒号。
- 重复组合保序去重。

### 9.2 配置错误测试

- 缺失或空 `CHAT_MODELS`。
- 空项目。
- 缺少冒号。
- 空提供商。
- 空模型名。
- 未知提供商。
- Gemini Key 缺失。
- DeepSeek Key 缺失。
- Gemini 链使用旧 `/openai/` 地址。
- 错误文本不泄露密钥。

### 9.3 Gemini 原生协议测试

- 请求 URL 为 `{GEMINI_URL}/models/{model}:generateContent`。
- 模型名作为单个 URL 路径组件正确编码。
- 使用 `x-goog-api-key`，不使用 Bearer Token。
- system、user 和 assistant 文本正确转换。
- 图片 data URL 正确转换为 `inlineData`。
- 函数声明正确转换为 `functionDeclarations`。
- `tool_choice` 正确转换为原生函数调用配置。
- 原生 `functionCall` 正确转换为内部工具调用。
- 内部工具结果正确转换为 `functionResponse`。
- 多个文本 Part 和多个函数调用 Part 正确解析。
- 安全拦截、空 Candidate 和损坏响应触发回退，不产生原始 Key 或请求体日志。
- 请求体不再包含 OpenAI `messages`、`model` 或 `tools[].function` 结构。

### 9.4 模型链测试

- `LLMModelSpec` 顺序与 `CHAT_MODELS` 一致。
- 已知不支持工具的模型在工具请求中跳过。
- 文本失败后正确回退。
- 图片失败后正确回退。
- 重复模型不会被调用两次。

### 9.5 启动和接口测试

- 无效配置使启动返回非零退出码。
- 有效配置正常启动。
- `/health` 返回提供商和模型名，不返回 Key。

### 9.6 回归测试

- 旧模型变量从运行时代码、`.env.example` 和 README 参数表移除。
- Gemini OpenAI 兼容端点、Bearer 鉴权和 `choices[].message` 解析从运行时代码与文档移除。
- README 和 `.env.example` 环境变量集合继续与运行时一致。
- 聊天、搜索、图片、记忆、并发和 OneBot 功能测试继续通过。

## 10. 验收标准

- 用户只修改 `CHAT_MODELS` 即可改变主模型和回退顺序。
- 模型尝试顺序与 `.env` 从左到右完全一致。
- 配置格式或必要 Key 错误在启动阶段明确失败。
- 日志和错误信息不泄露秘密。
- 旧模型选择变量完全退出运行时和用户文档。
- Gemini 使用原生 `generateContent` 请求、原生 API Key 请求头和原生消息/工具格式。
- Gemini 与 DeepSeek 继续共享同一份本地权威历史，模型回退不依赖远端会话 ID。
- 现有聊天、网页搜索、图片理解、记忆和多会话并发行为不回退。
- `.env.example`、README 和运行时环境变量保持一致。

## 11. 非目标与限制

- 不增加 Gemini、DeepSeek 之外的提供商。
- 不在启动阶段联网验证模型是否存在或账号权限。
- 不让用户手工声明工具或图片能力。
- 不使用 Gemini Interactions API、`previous_interaction_id` 或服务端会话状态。
- 不新增 Google GenAI SDK 依赖。
- 不加入流式输出、后台执行或托管 Agent。
- 不增加运行时命令来动态切换模型。
- 不提供旧模型变量的兼容层。
- 不在未获授权时修改真实 `.env`。
