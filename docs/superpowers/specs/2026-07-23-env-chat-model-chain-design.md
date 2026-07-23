# `.env` 对话模型链配置设计

**日期：** 2026-07-23
**状态：** 已确认，等待实施计划
**目标：** 让用户只通过一个 `CHAT_MODELS` 环境变量选择对话主模型和回退模型，并在启动阶段严格发现配置错误。

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

## 2. 已确认的产品决策

1. 支持一个主模型和任意数量的回退模型。
2. 只支持项目现有的 `gemini` 和 `deepseek` 提供商。
3. 使用单行 `CHAT_MODELS` 配置模型链。
4. 删除旧模型选择变量，不提供兼容读取。
5. 配置错误时拒绝启动，不静默采用默认模型。
6. 模型链中出现的每个提供商都必须配置对应 API Key。
7. 模型的工具调用和图片输入能力由运行时自动处理，不要求用户声明。

## 3. 用户配置

完整示例：

```dotenv
GEMINI_API_KEY=你的_Gemini_Key
GEMINI_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions

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

## 5. 内部架构

### 5.1 `src/model_config.py`

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

### 5.2 `src/config.py`

`Config`：

- 保留 `GEMINI_API_KEY`、`GEMINI_URL`、`DEEPSEEK_API_KEY` 和 `DEEPSEEK_URL`。
- 新增解析后的 `chat_models` 字段。
- 删除 `gemini_model`、`deepseek_model` 及全部旧 `LLM_*` 模型链字段。
- 在全局配置初始化期间完成严格校验。

### 5.3 `src/services/llm_client.py`

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

### 5.4 `run_bot.py`

启动入口捕获 `ModelConfigurationError`：

- 向标准错误输出简短中文配置错误。
- 返回非零退出码。
- 不启动 Flask 服务。
- 保持正常导入和启动路径不变。

由于当前全局 `Config` 在导入 `src.main` 时初始化，`run_bot.py` 需要在导入主模块的边界捕获该专用异常。

### 5.5 健康检查

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

## 6. 删除与保留的环境变量

### 6.1 保留

- `GEMINI_API_KEY`
- `GEMINI_URL`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_URL`

### 6.2 新增

- `CHAT_MODELS`

### 6.3 删除

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

## 7. 文档和配置模板

需要同步：

- `.env.example`：删除旧模型选择变量，新增带注释的 `CHAT_MODELS`。
- `README.md`：更新最小配置、完整参数表、模型回退说明、错误排查和迁移示例。
- 文档一致性测试：继续从运行时代码提取环境变量集合，不硬编码变量数量。

README 当前存在用户尚未提交的修改。实施时必须保留并合并这些修改，不得整体覆盖该文件。

真实 `.env` 不在本设计阶段修改。后续实施也只有在用户明确授权后才能迁移真实 `.env`。

## 8. 测试方案

### 8.1 解析单元测试

- 单模型。
- 多模型及顺序。
- 首尾空格。
- 提供商大小写归一。
- 模型名保持原样。
- 模型名内部包含冒号。
- 重复组合保序去重。

### 8.2 配置错误测试

- 缺失或空 `CHAT_MODELS`。
- 空项目。
- 缺少冒号。
- 空提供商。
- 空模型名。
- 未知提供商。
- Gemini Key 缺失。
- DeepSeek Key 缺失。
- 错误文本不泄露密钥。

### 8.3 模型链测试

- `LLMModelSpec` 顺序与 `CHAT_MODELS` 一致。
- 已知不支持工具的模型在工具请求中跳过。
- 文本失败后正确回退。
- 图片失败后正确回退。
- 重复模型不会被调用两次。

### 8.4 启动和接口测试

- 无效配置使启动返回非零退出码。
- 有效配置正常启动。
- `/health` 返回提供商和模型名，不返回 Key。

### 8.5 回归测试

- 旧模型变量从运行时代码、`.env.example` 和 README 参数表移除。
- README 和 `.env.example` 环境变量集合继续与运行时一致。
- 聊天、搜索、图片、记忆、并发和 OneBot 功能测试继续通过。

## 9. 验收标准

- 用户只修改 `CHAT_MODELS` 即可改变主模型和回退顺序。
- 模型尝试顺序与 `.env` 从左到右完全一致。
- 配置格式或必要 Key 错误在启动阶段明确失败。
- 日志和错误信息不泄露秘密。
- 旧模型选择变量完全退出运行时和用户文档。
- 现有聊天、网页搜索、图片理解、记忆和多会话并发行为不回退。
- `.env.example`、README 和运行时环境变量保持一致。

## 10. 非目标与限制

- 不增加 Gemini、DeepSeek 之外的提供商。
- 不在启动阶段联网验证模型是否存在或账号权限。
- 不让用户手工声明工具或图片能力。
- 不增加运行时命令来动态切换模型。
- 不提供旧模型变量的兼容层。
- 不在未获授权时修改真实 `.env`。
