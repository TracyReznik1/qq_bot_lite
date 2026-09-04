# qqbot — qqbot_lite

**[简体中文](README.md)** | **[English](README_EN.md)**

> 基于 OneBot 协议的轻量级 QQ 聊天机器人，接入 Gemini 与 DeepSeek 大模型。支持上下文对话、网页 URL 自动前置直读、确定性证据化联网检索、多模态视觉理解与细粒度结构化记忆，并内置三层 Prompt 注入防御机制。

---

## 特性总览

- **💬 上下文对话**：基于 FIFO 队列保障会话顺序，支持角色设定（`config/persona.md`）与多轮历史管理。
- **🌐 网页 URL 直读**：自动识别聊天中的 HTTP/HTTPS 链接，5 秒快速直读正文并注入 XML 沙箱作答，自动短路冗余搜索；历史记录轻量化防 Token 膨胀。
- **🔍 确定性联网检索**：Tavily 是主搜索提供者，遇到异常平滑回退至 DDGS（DDGS 的阶段超时默认为 15 秒）。普通聊天始终固定为由轻量检索路由（SearchRouter，基于检索收益判定）在无需搜索与 `LIGHT 模式`（单查询快速检索，普通回复不展示任何来源编号、标题或 URL）之间动态裁决，绝不越级触发多查询；显式 `/search` 为 `STANDARD 模式`（多查询，附带来源），支持 `/skip` 跳过搜索。遇到网络不可用、证据不足、名称/前提不一致时清晰说明，固定边界降级，确保基于已有可用证据生成，不伪造在线证据；遇到服务拒绝有效范围参数时移除日期过滤后重试一次 Tavily；安全 Trace 全程记录但剔除敏感正文。
- **🛡️ 三层 Prompt 注入防御**：
  1. **XML 语义沙箱**：外部网页正文严格实体转义，封装于 `<external_webpage_content>` 标签；
  2. **权威边界约束**：System Prompt 声明最高安全级，严禁模型执行网页中的任何诱导指令；
  3. **出站凭据脱敏拦截**：所有回复在经 OneBot 发送前统一执行硬凭据扫描，自动脱敏保护 API Key、密码与私钥。
- **🖼️ 多模态图片理解**：支持接收单图或图文混合输入（每条消息最多 4 张，每张最多 5 MiB），由视觉模型识别分析；不支持图片生成、图片编辑与主动发图。
- **🧠 细粒度结构化记忆**：基于 SQLite 持久化，支持私聊、群聊及全局作用域，涵盖自动学习、纠正、撤回、争议与物理删除机制。支持独立配置记忆 API Key（`MEMORY_GEMINI_API_KEY`），实现前后台模型配额物理隔离，杜绝 15 RPM 限流争抢。
- **清晰边界**：保持实用窄边界，不支持视频理解、天气或 B 站等外部专用工具插件或复杂 Agent。`/search <关键词>` 专注于关键词证据化检索（不提供独立 URL 直读）；聊天中的网页直读由系统自动安全处理。

```text
QQ 用户
  ↓ 消息事件
OneBot 客户端
  ↓ HTTP POST 回调
Flask / 按会话 FIFO 队列
  ├─→ 命令路由 (/search, /skip, /remember, /reset, /help 等)
  └─→ LLM ←→ 网页 URL 直读沙箱 / 证据化网页搜索 / 历史 / 记忆 / 图片理解
  ↓ 出站敏感凭据脱敏拦截器 (Redactor)
OneBot HTTP API
  ↓ 回复文本
QQ 用户
```

---

## Windows 快速开始

在项目根目录下打开 PowerShell：

```powershell
# 1. 创建并激活虚拟环境
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. 复制配置模板并填写 API Key
Copy-Item .env.example .env
notepad .env

# 4. 启动机器人进程
python run_bot.py
```

> 也可直接双击运行 `启动qqbot.bat` 自动完成环境激活与启动。

---

## 配置 OneBot

机器人与 OneBot 客户端（如 NapCat、Lagrange）为双向 HTTP 连接：

1. **qqbot → OneBot：**让 OneBot 开启 HTTP API（默认示例为 `http://127.0.0.1:3000`），将其地址写入 `ONEBOT_API_URL`。如果 OneBot 的 HTTP API 启用了访问令牌，把同一个值写入 `ONEBOT_ACCESS_TOKEN`；qqbot 会在请求中发送 `Authorization: Bearer <令牌>`。
2. **OneBot → qqbot：**在 OneBot 中新增 HTTP 事件上报，目标为 `http://127.0.0.1:5000/`（端口随 `BOT_HOST` / `BOT_PORT` 调整），事件格式使用 OneBot 11。若设置了 `CALLBACK_SECRET`，OneBot 回调必须选择以下任一请求头：`Authorization: Bearer <密钥>` 或 `X-QQBOT-Callback-Secret: <密钥>`。

- 群聊默认 `REQUIRE_GROUP_AT=true`，只有消息中确实 `@` 机器人账号才触发；设为 `false` 后处理群内每条有效消息。

---

## 完整 `.env` 参数参考

配置统一在 `.env` 中维护（格式为 `KEY=value`，注释以 `#` 开头）。

### 模型链与 API Key 隔离配置
- 模型链格式为 `提供商:模型名`（多个用逗号隔开），首个为主模型，后续为故障回退。支持 `gemini` 和 `deepseek`。
- Gemini 使用原生 `generateContent` REST API（默认 `GEMINI_URL=https://generativelanguage.googleapis.com/v1`），DeepSeek 使用 OpenAI 兼容端点。
- **平滑回退链抗拥堵推荐**：针对 Google Gemini API 偶尔出现的 503 服务高负荷拥堵（High Demand），强烈建议在 `CHAT_MODELS` 中配置回退链（例如 `CHAT_MODELS=gemini:gemini-3.5-flash-lite,gemini:gemini-3.1-flash-lite` 或接入 DeepSeek）。当首选模型出现 503 或网络超时时，系统会毫秒级无缝切换到下一顺位模型，保障服务连续性。
- **免搜索回复与检索错误严格隔离**：普通对话（免搜索或检索路由裁决为无需搜索）若因模型过载等故障失败，将直接提示服务繁忙，绝不会误报为“在线搜索网络不可用”，清晰区隔故障源。
- **前后台配额物理隔离**：前台聊天对话使用 `GEMINI_API_KEY`；后台异步记忆提取可独立配置 `MEMORY_GEMINI_API_KEY`。两者配额彻底隔离，杜绝高频聊天时争抢 15 RPM 免费层并发触发 429 限流；留空时自动回退复用主 Key。示例：
  ```dotenv
  CHAT_MODELS=gemini:gemini-3.5-flash-lite,gemini:gemini-3.1-flash-lite,deepseek:deepseek-chat
  GEMINI_API_KEY=your_chat_gemini_key
  DEEPSEEK_API_KEY=your_deepseek_key

  # 后台记忆提取专属 Key（可选，留空则自动复用主 Key）：
  MEMORY_GEMINI_API_KEY=your_second_gemini_key
  ```

### 全部参数汇总表

| 参数 | 必需性 | 默认值 | 作用与配置说明 |
|---|---|---|---|
| `BOT_HOST` | 可选 | `127.0.0.1` | Flask 本地监听地址。 |
| `BOT_PORT` | 可选 | `5000` | Flask 监听端口，需与 OneBot 回调地址保持一致。 |
| `CALLBACK_SECRET` | 推荐 | 空 | OneBot → qqbot 回调鉴权密钥。 |
| `ONEBOT_API_URL` | 通常必需 | `http://127.0.0.1:3000` | qqbot 调用 OneBot HTTP API 的基准根地址。 |
| `ONEBOT_ACCESS_TOKEN` | 条件必需 | 空 | qqbot → OneBot 请求的 Bearer 访问令牌。 |
| `REQUIRE_GROUP_AT` | 可选 | `true` | 群聊中是否要求必须 `@` 机器人账号才响应。 |
| `ADMIN_QQ_IDS` | 条件必需 | 空 | 具备管理员权限的 QQ 号列表（逗号或分号分隔），用于 `/globalremember`。 |
| `CHAT_MODELS` | 必需 | 无 | 对话模型链，例如 `gemini:gemini-3.5-flash-lite,gemini:gemini-3.1-flash-lite`。每个列出的提供商都需配置对应 Key。 |
| `MEMORY_MODELS` | 可选 | 空 | 结构化记忆提取与合并的专用模型链；留空时自动复用 `CHAT_MODELS`。 |
| `MEMORY_GEMINI_API_KEY` | 可选 | 空 | 记忆提取专用 Gemini Key；留空时自动复用 `GEMINI_API_KEY`。配置可实现前后台配额物理隔离。 |
| `MEMORY_DEEPSEEK_API_KEY` | 可选 | 空 | 记忆提取专用 DeepSeek Key；留空时自动复用 `DEEPSEEK_API_KEY`。 |
| `GEMINI_API_KEY` | 条件必需 | 空 | Google Gemini API 密钥。 |
| `GEMINI_URL` | 可选 | `https://generativelanguage.googleapis.com/v1` | Gemini Developer API 基础地址（会自动追加 generateContent）。 |
| `DEEPSEEK_API_KEY` | 条件必需 | 空 | DeepSeek API 密钥。 |
| `DEEPSEEK_URL` | 可选 | `https://api.deepseek.com/chat/completions` | DeepSeek 对话端点地址。 |
| `TAVILY_API_KEY` | 可选 | 空 | 主搜索引擎 Tavily API Key；未配置或不可用时自动回退至 DDGS。 |
| `PROXY_URL` | 可选 | 空 | 全局 HTTP/HTTPS 代理地址，例如 `http://127.0.0.1:7890`。 |
| `SEARCH_ROUTER_MODEL` | 可选 | `gemini-3.1-flash-lite` | 检索路由模型名称，用于在无命令时基于检索收益动态判断是否需要搜索（跳过搜索或 LIGHT 模式）。 |
| `SEARCH_ROUTER_TIMEOUT` | 可选 | `5.0` | 检索路由决策阶段超时（秒）。 |
| `SEARCH_MAX_RESULTS` | 可选 | `4` | 每次搜索检索返回的文档数量上限。 |
| `SEARCH_PLANNER_TIMEOUT` | 可选 | `8.0` | 搜索查询规划阶段超时（秒）。 |
| `SEARCH_TAVILY_TIMEOUT` | 可选 | `8.0` | Tavily 查询阶段超时（秒）。 |
| `SEARCH_DDGS_TIMEOUT` | 可选 | `15.0` | DDGS 回退查询阶段超时（秒）。 |
| `SEARCH_READER_TIMEOUT` | 可选 | `5.0` | 搜索网页正文抓取阶段超时（秒）。 |
| `SEARCH_RANKER_TIMEOUT` | 可选 | `10.0` | 搜索结果相关性重排打分阶段超时（秒）。 |
| `SEARCH_ANSWER_TIMEOUT` | 可选 | `20.0` | 搜索聚合推理回答阶段超时（秒）。 |
| `REQUEST_TIMEOUT` | 可选 | `18.0` | 模型 API、OneBot 通信等常规 HTTP 请求超时（秒）。 |
| `DATA_DIR` | 可选 | `qqbot_data` | 本地数据存放目录（历史、记忆数据库）。 |
| `HISTORY_TURNS` | 可选 | `8` | 对话上下文保留的最近轮数（用户+助手各算一条）。 |
| `PERSIST_HISTORY` | 可选 | `true` | 是否持久化对话历史至磁盘；设为 false 时仅在内存保留。 |
| `MESSAGE_WORKERS` | 可选 | `8` | 活跃并发会话的处理工作线程数。 |
| `MAX_REPLY_CHARS` | 可选 | `1700` | 单段回复的最大字符数上限，超出时自动按标点/换行智能分段分发。 |

---

## 使用方法

非 `/` 开头的消息直接进入日常对话：
- **网页直读与短路**：若消息中包含 HTTP/HTTPS 链接，会自动前置直读网页全文并沙箱总结，短路跳过搜索；
- **智能检索路由（SearchRouter）**：普通对话由轻量路由根据“检索收益（Retrieval Benefit）”智能裁决——日常闲聊、逻辑创作或无需外部事实的问题跳过搜索直接由模型生成；当问题依赖外部事实或时效信息时，自动触发 `LIGHT 模式`（单查询快速检索，普通回复不展示任何来源编号、标题或 URL，基于证据生成）；绝不在普通聊天中静默发起高开销多查询；
- **确定性命令边界**：若需要多查询深度搜索并附带来源链接，显式使用 `/search`（STANDARD 模式）；若需要强制免搜索快速回答，使用 `/skip`。

| 指令 | 别名 | 功能说明 |
|---|---|---|
| `/search <关键词>` | `/s <关键词>` | 显式强制联网搜索（STANDARD 模式），生成多查询并展示来源链接。 |
| `/skip [问题]` | 无 | 跳过联网搜索（SKIP 模式），直接由模型回答，支持附带图片，零搜索耗时与 Trace。 |
| `/remember <内容>` | `/memo <内容>` | 保存你的个人偏好或当前群专属记忆。 |
| `/globalremember <内容>` | `/gremember <内容>` | 保存全员共享的全局记忆设定（仅管理员）。 |
| `/memories [查询词]` | 无 | 查看或检索当前作用域允许访问的记忆列表。 |
| `/forget <ID或内容>` | 无 | 按权限执行物理删除、撤回或标记争议。 |
| `/reset` | 无 | 清空当前会话的对话历史上下文；不会删除个人记忆或全局记忆。 |
| `/help` | `/h` | 显示使用说明、常用指令与能力边界。 |

---

## 图片输入

支持在聊天中随消息直接发送图片或“图片+文字”进行多模态理解与分析。
- **限制规则**：每条消息最多 4 张，每张最多 5 MiB，支持 JPEG、PNG、WebP、GIF。
- **工作机制**：读取 OneBot 上报的公网图片 URL 或通过 `get_image` API 解析。图片只在当前进程内短暂保留用于本次推理，不会持久化图片二进制数据。
- **能力边界**：当前能力为多模态图片理解，不支持图片生成、图片编辑或主动发图。

---

## 多会话并发

内置基于线程池的会话调度队列（默认 `MESSAGE_WORKERS=8`）：
- **并行与隔离**：不同会话可以并行，最多占用配置的工作线程数；私聊按 QQ 号隔离，群聊按“群号 + QQ 号”隔离。
- **FIFO 顺序性**：同一会话仍按顺序处理，每个会话维护 FIFO 队列，上一条处理完成后才取下一条，杜绝回复乱序。
- **生命周期**：聊天回复队列与消息去重状态维护在单个进程内；记忆学习任务与文本则由 SQLite 持久化并异步消费。

---

## 数据、历史与结构化记忆

数据默认保存在 `qqbot_data/` 目录下（`history/` 与 `memory.sqlite3`）：
- **对话历史**：记录近期 `HISTORY_TURNS` 轮交互，重启后按配置恢复。`/reset` 指令可随时清空当前会话历史。
- **图片与历史隔离**：图片只在当前进程内短暂保留，记忆学习作业只持久化文本与元数据。
- **结构化记忆提取**：在普通聊天回复流程中，该轮回复处理结束后才将事实文本交给后台异步 Worker 进行细粒度 Claim 提取，包含一次额外的后台模型调用。记忆具备最终一致特性，由 SQLite 持久化作业保障故障恢复与重试。
- **作用域与生命周期**：支持私聊记忆（用户独享）、群聊显式群记忆（群内共享）与全局记忆（管理员）。支持纠正、撤回、争议与物理删除等生命周期操作。
- **隐私防护**：私聊个性化称呼安全进入群聊，严密过滤密码、Token 等硬秘密敏感信息；旧版 JSON 记忆不会加载，统一以 SQLite 为唯一来源。

---

## 运行限制

本项目定位为轻量、可控的单进程桌面/服务器机器人，基于 Flask 内置服务与线程池运行。不包含分布式任务调度、多实例共享状态或复杂可观测性；多实例运行需依赖外部调度系统。

---

## 常见问题

### python-dotenv could not parse statement
通常是 `.env` 中的字符串引号未成对闭合或混入了非法换行。对照 `.env.example` 确保每行为 `KEY=value` 格式并重启。

### 401 Unauthorized
多为令牌配置不一致引起：检查 qqbot → OneBot 请求方向：`ONEBOT_ACCESS_TOKEN` 是否与 OneBot API 的 Access Token 一致。若是 OneBot 回调报错 403，请检查 `CALLBACK_SECRET` 请求头。

### 模型返回 404 或不可用
确认 `CHAT_MODELS` 中模型名填写正确且账号拥有调用权限。若使用 Gemini，确认基础地址为原生 `GEMINI_URL=https://generativelanguage.googleapis.com/v1` 且无多余后缀。

### 图片读取失败或提示格式不支持
确认 OneBot 上报了可达的公网图片 URL 或本地支持的图片缓存，检查代理设置是否影响图片拉取，并确认当前模型具备 Vision 多模态理解能力。

---

## 开发与回归测试

```powershell
# 运行全量单元测试（529 项测试）
python -B -m unittest discover -s tests -t . -v
# 或使用 pytest 快速并发测试
pytest -q

# 检查代码语法与编译
python -B -m compileall -q src tests run_bot.py

# 专项验证 README 与环境配置一致性
python -B -m unittest tests.test_readme_guide -v
```
