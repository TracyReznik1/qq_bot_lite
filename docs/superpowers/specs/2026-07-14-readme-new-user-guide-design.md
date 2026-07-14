# README 新手完整指南重写设计

## 背景

当前 README 能概括聊天、搜索、图片理解和少量启动配置，但不足以让第一次接触 Python、OneBot 或模型 API 的用户独立完成部署。源码实际读取 34 个环境变量，`.env.example` 和 README 仅覆盖其中一部分；模型链、OneBot 双向鉴权、数据和记忆语义、并发行为及常见故障也缺少系统说明。

本次工作将 README 重写为面向新手的完整中文指南，并同步补全 `.env.example`。文档只描述当前代码已经实现的能力，不恢复已删除功能，不修改运行行为，不读取、复制或输出真实 `.env` 内容。完成后提交当前所有本地提交并推送 `main` 到现有 GitHub 远端。

## 目标读者

- 第一次部署 QQ 机器人的 Windows 用户。
- 已安装或准备安装 NapCat、Lagrange 等 OneBot 11 兼容客户端的用户。
- 不熟悉 Python 虚拟环境、`.env`、模型回退或 HTTP 回调配置的用户。
- 需要查询全部配置默认值和影响范围的维护者。

## 文档组织方案

采用“教程优先 + 完整参考”结构：用户先用最少步骤启动，再通过完整参数手册理解或定制高级行为。避免把 34 个参数直接堆在开头，也不把必要信息拆散到多个文档。

README 章节顺序如下：

1. 项目简介、适用场景和能力边界。
2. 请求流程：QQ → OneBot → qqbot_lite → 模型/网页搜索 → QQ。
3. 功能清单和明确不支持的功能。
4. 前置条件。
5. Windows 新手快速开始。
6. OneBot/NapCat 接入配置。
7. 最小可用 `.env` 示例。
8. 完整 `.env` 参数参考。
9. 日常使用和命令示例。
10. 图片输入、多会话并发、数据、历史和记忆。
11. 健康检查、常见故障与处理方法。
12. 安全建议。
13. 开发测试和项目结构。

## 项目与能力说明

README 必须准确说明当前产品只保留：

- 普通聊天。
- 模型按需调用网页搜索。
- 显式 `/search` 关键词搜索。
- 用户发送图片或图片加文字后，由支持图片输入的模型识别并回答。
- `/remember` 个人长期记忆。
- `/globalremember` 管理员全局记忆。
- `/reset` 当前会话重置。
- `/help` 帮助。
- 不同会话并行、同一会话严格按入队顺序处理。

明确说明不支持图片生成、图片编辑、主动发图、视频理解、天气、B站能力、独立 URL 直读、文件处理或其他 Agent 工具。搜索服务可以在内部安全读取部分搜索结果页，但用户不能把 URL 当成独立读取命令。

## 新手安装与启动

Windows 快速开始使用可复制的 PowerShell 命令：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python run_bot.py
```

同时说明：

- Python 要求为 3.11 或更高版本。
- PowerShell 禁止激活脚本时，可以只在当前用户范围调整执行策略，或直接使用 `.venv\Scripts\python.exe` 执行安装和启动命令。
- `启动qqbot.bat` 假设项目根目录已经存在 `.venv`；若缺少依赖会尝试安装，然后启动机器人。
- 修改 `.env` 后需要重启进程。

## OneBot/NapCat 接入

文档用通用 OneBot 11 术语，并以 NapCat 的 HTTP 配置为例，不绑定某一客户端的界面截图：

- OneBot HTTP API 地址默认 `http://127.0.0.1:3000`，对应 `ONEBOT_API_URL`。
- OneBot 事件上报地址默认 `http://127.0.0.1:5000/`，由 `BOT_HOST` 和 `BOT_PORT` 决定。
- `ONEBOT_ACCESS_TOKEN` 用于 qqbot 调用 OneBot 的发送消息和图片解析 API。
- `CALLBACK_SECRET` 用于 OneBot 向 qqbot 上报事件时的入站鉴权；推荐请求头为 `Authorization: Bearer <secret>` 或 `X-QQBOT-Callback-Secret: <secret>`。
- 两端配置必须匹配；两个安全参数作用方向不同，不把它们描述成同一个值。
- 群聊默认必须 @ 机器人；`REQUIRE_GROUP_AT=false` 可关闭要求。

## 最小可用配置

分别给出 Gemini 和 DeepSeek 两种最小示例。示例只使用占位符：

- Gemini：配置 `GEMINI_API_KEY`，保留默认主 provider。
- DeepSeek：配置 `DEEPSEEK_API_KEY` 和 `LLM_PRIMARY_PROVIDER=deepseek`。
- 两种示例都展示 OneBot 地址，以及按实际 OneBot 配置决定是否填写 Token。
- 不嵌入真实 Key、QQ 号、Token、人设或代理地址。

## 完整环境变量参考

README 和 `.env.example` 必须覆盖源码当前读取的全部 34 个键，按以下类别组织。

### 身份

- `BOT_NAME`
- `BOT_PERSONA`

### Gemini

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `GEMINI_URL`

### DeepSeek

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL`
- `DEEPSEEK_URL`

### 模型链

- `LLM_PROVIDER`（旧版兼容项）
- `LLM_PRIMARY_PROVIDER`
- `LLM_PRIMARY_MODEL`
- `LLM_FALLBACK_1_PROVIDER`
- `LLM_FALLBACK_1_MODEL`
- `LLM_FALLBACK_2_PROVIDER`
- `LLM_FALLBACK_2_MODEL`
- `LLM_FALLBACK_3_PROVIDER`
- `LLM_FALLBACK_3_MODEL`

### OneBot 与服务监听

- `ONEBOT_API_URL`
- `ONEBOT_ACCESS_TOKEN`
- `CALLBACK_SECRET`
- `BOT_HOST`
- `BOT_PORT`
- `REQUIRE_GROUP_AT`

### 网络与搜索

- `PROXY_URL`
- `TAVILY_API_KEY`
- `SEARCH_MAX_RESULTS`
- `REQUEST_TIMEOUT`

### 管理、数据、历史、记忆、并发和回复

- `ADMIN_QQ_IDS`
- `DATA_DIR`
- `HISTORY_TURNS`
- `MEMORY_LIMIT`
- `PERSIST_HISTORY`
- `MESSAGE_WORKERS`
- `MAX_REPLY_CHARS`

每项参数表包含：是否必需、源码默认值、允许格式、用途、推荐配置和注意事项。条件必需项必须明确，例如至少要有模型链中某个实际使用 provider 的 API Key。

### 格式规则

- `BOT_PERSONA` 单行可直接填写；包含空格、`#` 或换行时使用成对的单引号或双引号。多行开始和结束引号必须同时存在。
- 布尔值按源码说明：`1`、`true`、`yes`、`y`、`on`（忽略大小写）为真；设置为其他文本时为假。未设置时使用各参数默认值。
- `ADMIN_QQ_IDS` 支持英文逗号或分号分隔多个 QQ 号。
- 整数和小数非法时使用源码默认值；`MESSAGE_WORKERS` 的 0 或负数归一为 1。
- URL 必须包含 `http://` 或 `https://`；代理不得写成重复的 `PROXY_URL=PROXY_URL=...`。
- `.env` 一行只定义一个键；普通注释单独使用以 `#` 开头的行。

### 模型链语义

- 主模型来自 `LLM_PRIMARY_PROVIDER` 和 `LLM_PRIMARY_MODEL`；主模型名留空时使用对应 provider 的 `GEMINI_MODEL` 或 `DEEPSEEK_MODEL`。
- 回退槽按 1、2、3 顺序尝试；把某个 `LLM_FALLBACK_N_PROVIDER` 留空可禁用该槽。
- `LLM_PROVIDER` 只作为旧配置兼容：实际代码中仅当 `LLM_PRIMARY_PROVIDER` 未设置时，它才成为主 provider 的后备值。新配置不推荐使用该键。
- 默认模型名按当前源码记录；模型是否存在、是否支持图片或工具调用，取决于用户账号、接口和服务商实际能力。

### 网络和搜索语义

- `PROXY_URL` 用于模型请求、Tavily/DDGS 搜索、搜索结果页读取，以及需要代理解析的部分 QQ 图片下载。
- 未配置 `TAVILY_API_KEY` 时仍可使用 DDGS；配置后优先使用 Tavily 并保留 DDGS 回退。
- `REQUEST_TIMEOUT` 是网络请求超时秒数，不是模型总处理时限保证。

## `.env.example` 设计

`.env.example` 同步补全为完整、可复制的模板：

- 使用类别注释和源码默认值。
- Key、Token、管理员 QQ 和代理保留空值或安全占位说明。
- 不复制真实 `.env` 的任何内容。
- `BOT_PERSONA` 使用安全的通用单行默认值，并通过注释指向 README 的多行写法。
- `LLM_PROVIDER` 标注为兼容项，默认留空。
- 所有键只出现一次。

## 使用、数据和记忆

README 给出私聊、群聊、图片和五个命令的实例，并准确说明：

- `/remember` 写入按 QQ 用户隔离的个人长期记忆，可跨该用户的会话读取。
- `/globalremember` 只允许 `ADMIN_QQ_IDS` 中的管理员写入全局记忆。
- `/reset` 清除当前 `session_key` 的聊天历史和当前会话记忆，不删除个人长期记忆或全局记忆。
- 历史和记忆默认存入 `qqbot_data/history` 与 `qqbot_data/memories`。
- 私聊会话键按用户隔离；群聊按群号和用户号隔离。
- 不同会话默认最多 8 个并行；同一会话的连续消息严格按接收顺序处理和回复。
- 每条消息最多 4 张图片，每张最多 5 MiB，支持 JPEG、PNG、WebP、GIF；识别能力取决于模型。

## 排错章节

至少覆盖以下症状、检查顺序和不泄密的解决方法：

- `python-dotenv could not parse statement`：检查重复键前缀、未闭合引号、多行值和错误注释。
- Gemini/DeepSeek `401 Unauthorized`：检查 Key、provider、URL、模型权限和重启。
- `所有模型暂时不可用`：按模型链检查 Key、模型名、接口地址、代理和服务状态。
- `API key is not configured`：配置当前链实际使用 provider 的 Key。
- OneBot 收不到回调：检查事件上报 URL、监听地址、端口、防火墙和 callback secret。
- OneBot 发不出消息：检查 API URL、端口、access token 和 OneBot 日志。
- 图片地址无效或图片读取失败：确认 OneBot `/get_image` 能返回图片、代理 URL 合法、文件大小和格式符合限制。
- 当前模型无法识别图片：使用支持图片输入的模型或调整模型链。
- 代理失败：确认 `PROXY_URL` 是单个合法 URL、代理进程运行、不要重复键名。
- PowerShell 拒绝激活虚拟环境：给出当前用户范围执行策略或直接调用 venv Python 的替代命令。

## 安全说明

- `.env` 已由 Git 忽略，禁止强制提交。
- 不在 README、日志、Issue、截图或测试中使用真实 Key、Token、QQ 号和私密人设。
- 默认 `BOT_HOST=127.0.0.1` 只监听本机；暴露到其他网络前必须理解访问控制并设置 `CALLBACK_SECRET`。
- `ONEBOT_ACCESS_TOKEN` 和 `CALLBACK_SECRET` 使用不同方向，均应使用高强度值。
- `DATA_DIR` 可能包含聊天历史和记忆，备份和分享前需视为敏感数据。

## 验证设计

新增或扩展文档一致性测试：

1. 从 `src/config.py` 提取所有环境变量名，并确认 README 和 `.env.example` 都覆盖全部 34 个键。
2. 确认 `.env.example` 中每个配置键只定义一次。
3. 确认 README 包含安装、OneBot、最小配置、命令、记忆、图片、并发、排错、安全、测试和项目结构章节。
4. 确认 README 不宣传已删除的生成、视频、天气、B站或独立 URL 直读能力。
5. 确认真实 `.env` 未进入 Git 差异或提交。
6. 运行全量单元测试、`compileall` 和 `git diff --check`。
7. 提交 README、`.env.example`、文档测试和设计/计划文档后，将 `main` 推送到现有 `origin`。

## 验收标准

1. 新用户只阅读 README 即可完成 Python 环境、`.env`、OneBot 和首次启动配置。
2. README 与 `.env.example` 覆盖源码实际读取的全部 34 个环境变量，默认值和语义与代码一致。
3. 多行 `BOT_PERSONA`、模型回退、双向 OneBot 鉴权和代理格式有明确且可复制的正确示例。
4. 使用、记忆、图片、多会话并发和能力边界描述不与源码冲突。
5. 常见日志错误能在 README 中找到针对性的排查步骤。
6. README 和 `.env.example` 不含真实秘密或私密配置。
7. 自动化测试能在以后新增环境变量但未更新文档时失败。
8. 所有测试与静态检查通过，最终提交成功推送到 GitHub `main`。

## 已知限制

- README 只能记录当前代码默认值；模型名称、API 可用性和第三方客户端界面可能随服务商更新。
- NapCat/Lagrange 的具体菜单名称可能因版本变化，文档使用协议字段和地址而非易过期截图。
- 文档不会替代 OneBot 客户端自身的安装说明，只说明本项目需要的 HTTP API 和事件上报配置。
- README 面向单机单进程部署；跨进程共享队列和分布式部署不在本项目范围。
