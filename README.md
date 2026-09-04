# qqbot — qqbot_lite

qqbot_lite 是一个通过 OneBot 接入 QQ 的轻量聊天机器人。它把收到的私聊或群聊消息交给 Gemini / DeepSeek。事实型问题默认由程序路由到证据化网页搜索，搜索结果整理成可追溯的 Evidence 后再让模型基于证据回答，最后通过 OneBot 回复。

## 功能与边界

项目刻意保持窄边界：支持聊天、自动或显式网页搜索、图片理解、对话历史，以及会话 / 个人 / 全局记忆；不支持图片生成、图片编辑、主动发图、视频、天气、B站专用能力、文件处理、独立 URL 直读或复杂 Agent。`/search <关键词>` 会把输入当作搜索词；搜索服务可能在内部安全读取部分结果页来整理证据，但不提供独立 URL 直读。

## 工作流程

```text
QQ 用户
  ↓ 消息事件
OneBot 客户端
  ↓ HTTP POST 回调
Flask / 按会话 FIFO 队列
  ├─→ 命令路由
  └─→ LLM ←→ 网页搜索 / 历史 / 记忆 / 图片数据
  ↓ 回复文本
OneBot HTTP API
  ↓
QQ 用户
```

## 前置条件

- Windows 10/11 与 PowerShell。
- Python 3.11 或更高版本；安装时勾选“Add Python to PATH”。
- 一个 OneBot 11 兼容客户端，例如 NapCat 或 Lagrange，并能提供 HTTP API 与 HTTP 事件上报。
- 至少一个 Gemini 或 DeepSeek API Key。
- 可访问相应模型和搜索服务的网络；必要时准备 HTTP(S) 代理。

## Windows 快速开始

在项目目录打开 PowerShell，依次执行：

```powershell
# 1. 创建并激活虚拟环境
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. 复制安全模板；不要把真实密钥写回 .env.example
Copy-Item .env.example .env

# 4. 编辑配置
notepad .env

# 5. 启动
python run_bot.py
```

保持这个窗口运行。修改 `.env` 后必须停止并重新启动进程，配置才会重新加载。

也可以在已经创建 `.venv` 后双击 `启动qqbot.bat`。脚本会激活虚拟环境；若其中还没有 Flask，会安装 `requirements.txt`，然后执行 `python run_bot.py`。首次安装依赖仍建议在 PowerShell 中完成，便于看到错误信息。

## 配置 OneBot

连接是双向的，不要混淆两个令牌：

1. **qqbot → OneBot：**让 OneBot 开启 HTTP API（默认示例为 `http://127.0.0.1:3000`），将其地址写入 `ONEBOT_API_URL`。如果 OneBot 的 HTTP API 启用了访问令牌，把同一个值写入 `ONEBOT_ACCESS_TOKEN`；qqbot 会在请求中发送 `Authorization: Bearer <令牌>`。
2. **OneBot → qqbot：**在 OneBot 中新增 HTTP 事件上报，目标为 `http://127.0.0.1:5000/`（端口随 `BOT_HOST` / `BOT_PORT` 调整），事件格式使用 OneBot 11。若设置了 `CALLBACK_SECRET`，OneBot 回调必须选择以下任一请求头：`Authorization: Bearer <密钥>` 或 `X-QQBOT-Callback-Secret: <密钥>`。留空则回调入口不鉴权，不建议暴露到不可信网络。

推荐让两个服务都只监听本机。如果它们位于不同机器，需改为可达地址、放行防火墙，并使用反向代理 / TLS；不要直接把未鉴权的开发服务器暴露到公网。

群聊默认 `REQUIRE_GROUP_AT=true`，只有 OneBot 事件中确实包含对机器人账号的 `@` 才处理；仅 `@` 而没有正文也不会回复。设为 `false` 后，机器人会处理群内每条有效消息。机器人显示身份来自 `config/persona.md` 中的名字字段，不决定 OneBot 的 `@` 检测。

## 最小 `.env` 配置

先复制 `.env.example`，再通过 `CHAT_MODELS` 配置模型链。模型链引用的每个提供商（`gemini` 或 `deepseek`）都必须填写对应 API Key；`ONEBOT_ACCESS_TOKEN` 仅在 OneBot HTTP API 开启鉴权时填写；`CALLBACK_SECRET` 推荐设置，启用后必须与 OneBot 回调端保持一致。机器人身份和完整角色设定保存在 `config/persona.md`，可参考 `config/persona.example.md` 修改。下面的占位符不要照抄。

Gemini 示例：

```dotenv
CHAT_MODELS=gemini:填写账号可用的模型名
GEMINI_API_KEY=填写你的_Gemini_Key
GEMINI_URL=https://generativelanguage.googleapis.com/v1
ONEBOT_API_URL=http://127.0.0.1:3000
ONEBOT_ACCESS_TOKEN=
CALLBACK_SECRET=
```

Gemini 带 DeepSeek 回退示例：

```dotenv
CHAT_MODELS=gemini:填写主模型名,deepseek:填写回退模型名
GEMINI_API_KEY=填写你的_Gemini_Key
DEEPSEEK_API_KEY=填写你的_DeepSeek_Key
ONEBOT_API_URL=http://127.0.0.1:3000
ONEBOT_ACCESS_TOKEN=
CALLBACK_SECRET=
```

DeepSeek 示例：

```dotenv
CHAT_MODELS=deepseek:填写账号可用的模型名
DEEPSEEK_API_KEY=填写你的_DeepSeek_Key
ONEBOT_API_URL=http://127.0.0.1:3000
ONEBOT_ACCESS_TOKEN=
CALLBACK_SECRET=
```

## `.env` 格式规则

- 一行一个 `KEY=value`；注释用单独一行的 `#` 开头。密钥通常不需要引号，但包含空格、`#` 或换行的文本应使用成对引号。
- 角色设定不写入 `.env`。编辑 `config/persona.md` 时必须保留非空的 `- 名字：...` 行；需要新建角色时可复制 `config/persona.example.md`。

- 布尔项不区分大小写；`1`、`true`、`yes`、`y`、`on` 表示真，其他已填写的值表示假。完全缺失时使用源码默认值。
- `ADMIN_QQ_IDS` 是列表，多个 QQ 号可用英文逗号或分号分隔，项目会去除首尾空白和重复项。
- 整数 / 浮点数填写错误时不会阻止启动，而会回退到源码默认值；这可能掩盖拼写错误，因此修改后应查看启动日志并实际验证。
- 不要在值末尾随意追加行内注释。编辑完保存，并重启 qqbot。

## 完整 `.env` 参数参考

“默认值”均来自 `src/config.py`。空表示源码默认空字符串；“条件必需”表示启用对应能力时必须配置。

### 服务

| 参数 | 必需性 | 源码默认值 | 作用与配置方法 |
|---|---|---|---|
| `BOT_HOST` | 可选 | `127.0.0.1` | Flask 监听地址。只有明确理解风险时才改为 `0.0.0.0`。 |
| `BOT_PORT` | 可选 | `5000` | Flask 监听端口，必须与 OneBot 回调地址一致。 |
| `CALLBACK_SECRET` | 推荐 | 空 | 验证 OneBot → qqbot 回调；设置后使用 Bearer 头或 `X-QQBOT-Callback-Secret`。 |

### OneBot 与权限

| 参数 | 必需性 | 源码默认值 | 作用与配置方法 |
|---|---|---|---|
| `ONEBOT_API_URL` | 通常必需 | `http://127.0.0.1:3000` | qqbot 调用 OneBot HTTP API 的根地址，不要带末尾 `/`。 |
| `ONEBOT_ACCESS_TOKEN` | 条件必需 | 空 | OneBot HTTP API 的访问令牌，用于 qqbot → OneBot；与 `CALLBACK_SECRET` 方向不同。 |
| `REQUIRE_GROUP_AT` | 可选 | `true` | 为真时群聊必须真实 `@` 机器人账号才响应。 |
| `ADMIN_QQ_IDS` | 条件必需 | 空集合 | 可执行 `/globalremember` 的 QQ 号；多个值用英文逗号或分号分隔。 |

### Gemini 与 DeepSeek

| 参数 | 必需性 | 源码默认值 | 作用与配置方法 |
|---|---|---|---|
| `GEMINI_API_KEY` | 条件必需 | 空 | Gemini Developer API 密钥；模型链包含 Gemini 时必须配置。 |
| `GEMINI_URL` | 可选 | `https://generativelanguage.googleapis.com/v1` | Gemini Developer API 基础地址；客户端会追加 `/models/{model}:generateContent`。不要填写旧 `/openai/chat/completions` 地址。 |
| `DEEPSEEK_API_KEY` | 条件必需 | 空 | DeepSeek API Key；模型链包含 DeepSeek 时必须配置。 |
| `DEEPSEEK_URL` | 可选 | `https://api.deepseek.com/chat/completions` | DeepSeek 的 OpenAI 兼容聊天端点。 |

### 模型链

| 参数 | 必需性 | 源码默认值 | 作用与配置方法 |
|---|---|---|---|
| `CHAT_MODELS` | 必需 | 无 | 对话模型链，格式为 `提供商:模型名`；第一个是主模型，后续依次回退。支持 `gemini` 和 `deepseek`，列出的每个提供商都必须配置对应 API Key。 |
| `MEMORY_MODELS` | 可选 | 空 | 结构化记忆提取与合并模型链；留空时自动复用 `CHAT_MODELS`。 |

模型按照 `CHAT_MODELS` 从左到右尝试。重复的“提供商 + 模型名”只保留第一次；格式错误、未知提供商或对应 Key 缺失时，机器人会在启动阶段停止并给出中文错误。模型名是否存在及是否支持工具或图片由实际 API 响应决定。

Gemini 使用原生无状态 `generateContent` REST API，DeepSeek 继续使用 OpenAI 兼容端点，两者共享同一份本地会话历史。

模型能力不会由配置自动补齐：事实型问题默认由程序执行检索路由，普通聊天不再向模型提供 `search_web` 工具。显式 `/search` 与普通聊天共用同一条证据化检索管线，先搜索、整理证据再让模型基于证据回答；图片请求会沿模型链尝试，但图片理解能力仍取决于模型和端点；全部不能处理时会提示当前模型无法识别图片。

### 搜索与网络

| 参数 | 必需性 | 源码默认值 | 作用与配置方法 |
|---|---|---|---|
| `TAVILY_API_KEY` | 可选 | 空 | Tavily 是主搜索提供者；未配置 Tavily Key，或 Tavily 不可用、出错、超时、无结果、未产生有效 URL 时，使用 DDGS 回退。网络不可用、搜索完成但证据不足、以及证据支持的名称/前提不一致会使用不同披露，不会伪装成已经获得在线证据。 |
| `PROXY_URL` | 可选 | 空 | HTTP(S) 代理地址，例如 `http://127.0.0.1:7890`；模型、搜索和图片下载会按实现使用它。 |
| `SEARCH_MAX_RESULTS` | 可选 | `4` | 每个搜索查询的 provider 结果上限（至少按 1 处理）。 |
| `SEARCH_PLANNER_TIMEOUT` | 可选 | `8.0` | 查询规划阶段的超时秒数，至少按 `0.1` 处理。 |
| `SEARCH_TAVILY_TIMEOUT` | 可选 | `8.0` | Tavily 批次的超时秒数，至少按 `0.1` 处理。 |
| `SEARCH_DDGS_TIMEOUT` | 可选 | `15.0` | DDGS 批次的超时秒数，至少按 `0.1` 处理。 |
| `SEARCH_READER_TIMEOUT` | 可选 | `5.0` | 正文读取阶段的超时秒数，至少按 `0.1` 处理。 |
| `SEARCH_RANKER_TIMEOUT` | 可选 | `10.0` | 搜索结果排序阶段的超时秒数，至少按 `0.1` 处理。 |
| `SEARCH_ANSWER_TIMEOUT` | 可选 | `20.0` | 搜索回答阶段的超时秒数，至少按 `0.1` 处理。 |
| `REQUEST_TIMEOUT` | 可选 | `18.0` | 模型、OneBot、搜索、图片等 HTTP 请求的超时秒数。 |

搜索固定分两档且由调用入口显式指定，不再由模型启发式判断：
- **普通聊天（LIGHT 模式）：** 无论纯文本、纯图片还是图文混排，普通聊天始终固定为 `LIGHT` 模式，且严格只生成 1 个搜索查询。普通回复不展示任何来源编号、标题或 URL。
- **`/search` 命令（STANDARD 模式）：** 文本、纯图片或图文均进入 `STANDARD` 模式，生成 1 到 3 个查询，并在回答后展示至多 3 个来源 URL。
- **`/skip` 命令（SKIP 模式）：** 用户明确跳过搜索的唯一入口，将文本和图片直接转发给多模态模型进行纯对话生成。完全不调用搜索流水线，不产生搜索依赖，也不生成任何搜索 Trace。若无输入则提示：`用法：/skip <内容>，也可以附带图片。`。

**提供商执行与降级：**
- Tavily 是主搜索提供者，每个查询优先并发请求 Tavily；
- 只有未完成（未配置、不可用、空、超时、错误或 URL 无效）的查询才会回退至 DDGS，DDGS 的阶段超时默认为 15 秒；
- 查询规划、正文抓取、重排和回答各阶段发生异常时按固定边界降级，保证回答依然基于已有可用证据生成；
- 安全 Trace（`SearchTrace`）仅记录请求 ID、入口来源、模式、计数、脱敏状态和耗时，严格剔除请求文本、证据正文及敏感信息。

每个检索阶段使用自己的独立超时（规划 `SEARCH_PLANNER_TIMEOUT=8`、Tavily `SEARCH_TAVILY_TIMEOUT=8`、DDGS `SEARCH_DDGS_TIMEOUT=15`、Reader `SEARCH_READER_TIMEOUT=5`、Ranker `SEARCH_RANKER_TIMEOUT=10`、Answer `SEARCH_ANSWER_TIMEOUT=20`），前一阶段耗时不扣减后一阶段的超时。这里的 planner 指查询规划，而非路由决策。服务拒绝有效范围参数时，会在同一阶段剩余预算内移除日期过滤后重试一次 Tavily。搜索失败时会返回明确提示，且不伪造在线证据。

`/health` 返回 `search_ready` 和 `search_providers`（每个提供者的 `configured` / `available`），不暴露任何密钥、查询、URL 或证据正文。

### 数据、历史、并发与回复

| 参数 | 必需性 | 源码默认值 | 作用与配置方法 |
|---|---|---|---|
| `DATA_DIR` | 可选 | `qqbot_data` | 数据目录；相对路径以项目根目录为基准。 |
| `HISTORY_TURNS` | 可选 | `8` | 每个会话保留的最近对话轮数；内部按用户 / 助手两条消息计算。 |
| `PERSIST_HISTORY` | 可选 | `true` | 是否把历史写入磁盘；为假时历史只在当前进程内存在。 |
| `MESSAGE_WORKERS` | 可选 | `8` | 同时处理的活跃会话工作线程数；小于 1 时按 1 处理。 |
| `MAX_REPLY_CHARS` | 可选 | `1700` | 单段回复字符上限；源码最低按 200 处理，长回复会优先按换行或标点拆分。 |

## 使用方法

非 `/` 开头的文本进入普通聊天。所有普通聊天（纯文本、纯图片、图文混排）均固定采用 LIGHT 搜索模式并基于检索到的证据回答，普通回复不展示任何来源编号、标题或 URL。如需跳过搜索可使用 `/skip`，如需更多查询与参考来源可使用 `/search`。

| 命令 | 别名 | 说明 |
|---|---|---|
| `/search <关键词>` | `/s <关键词>` | 显式网页搜索（STANDARD 模式），支持纯文本、纯图片或图文，生成 1~3 个查询并展示至多 3 个来源。 |
| `/skip <内容>` | 无 | 跳过联网搜索（SKIP 模式），直接使用多模态大模型回答，支持附带图片，不产生搜索依赖与 Trace。 |
| `/remember <内容>` | `/memo <内容>` | 写入对应作用域的个人或群聊结构化记忆。 |
| `/globalremember <内容>` | `/gremember <内容>` | 写入全局共享记忆（仅管理员）。 |
| `/memories [查询词]` | 无 | 查询当前作用域授权允许使用的记忆列表。 |
| `/forget <ID或内容>` | 无 | 删除指定的记忆证据项。 |
| `/reset` | 无 | 清除当前会话历史；不会删除个人记忆或全局记忆。 |
| `/help` | `/h` | 显示简短用法和能力边界。 |

未知命令不会交给模型执行，而会返回支持的命令列表。

## 图片输入

用户可以直接发图，也可以发送图片加文字来做图片理解。每条消息最多 4 张，每张最多 5 MiB，允许 JPEG、PNG、WebP、GIF。qqbot 会读取 OneBot 事件提供的公网 HTTP(S) 图片地址；地址缺失时会尝试通过 OneBot 的 `get_image` API 解析文件标识。

这不是图片生成能力。项目不能生成、编辑或主动发送图片。识别效果和是否支持多图取决于当前模型；格式不符、地址不可达、私网地址被拒绝或模型不支持图片时，机器人会给出错误提示。

## 多会话并发

模板默认是原样配置：`MESSAGE_WORKERS=8`。

- 不同会话可以并行，最多占用配置的工作线程数。
- 同一会话仍按顺序处理：每个会话维护 FIFO 队列，上一条处理完成后才取下一条，避免回复乱序。
- 单条消息处理抛出异常后会记录日志，队列仍继续处理该会话的后续消息。
- 私聊会话键按 QQ 号隔离；群聊按“群号 + 发送者 QQ”隔离，因此同一群的不同用户也可并行。

聊天回复队列和消息去重状态只存在于单个 Python 进程内；进程退出时，尚未处理的聊天回复任务不会恢复。记忆学习任务是另一条队列：消息文本与接收顺序会写入 SQLite 持久化作业，重启后可以恢复并继续处理。不要直接启动多个进程并期望它们共享聊天回复顺序、去重或线程上限；多实例仍需要额外的外部队列、跨进程协调与共享状态。

## 数据、历史与结构化记忆

默认数据位于 `qqbot_data/`：对话历史在 `qqbot_data/history/`，结构化记忆数据库位于 `qqbot_data/memory.sqlite3`。不要提交或公开这个目录。

- **历史：**最近 `HISTORY_TURNS` 轮用户 / 助手消息。私聊按 QQ 用户隔离，群聊按“群号 + QQ 用户”隔离。`PERSIST_HISTORY=true` 时重启后可恢复；为假时只保留内存状态。`/reset` 会清空当前会话的历史。
- **结构化记忆 (SQLite)：**持久化在 `DATA_DIR/memory.sqlite3` 中。普通聊天消息进入回复流程时会先写入记忆作业，该轮回复处理结束后（包括回复失败）才释放给后台异步 Worker，尝试提取成带有发布者、作用域和生效时间的细粒度 Claim；因此自动学习会增加一次额外的后台模型调用、延迟和 API 成本。
- **最终一致与恢复：**聊天回复不等待记忆提取完成，所以刚说过的内容不保证立刻能由 `/memories` 或下一条回复检索到。记忆学习任务及文本由 SQLite 持久化，失败会按作业状态重试，应用重启后也会按原接收顺序恢复；这是一致性延迟，不代表聊天回复失败。
- **作用域与隔离规则：**
  - **私聊记忆：**只有该 QQ 用户与机器人可见；
  - **群聊记忆：**属于该群聊共享，群内成员均可作为证据参考；
  - **全局记忆：**由管理员通过 `/globalremember` 写入，全会话可见。
- **归因、冲突与生命周期：**共享记忆会保留发言者归因。后续的纠正可以让旧说法被新说法取代；发言者对自己的群记忆执行 `/forget` 是撤回；被描述的本人对他人发布的群记忆执行 `/forget` 会标记争议并停止用于回答；私聊所有者或管理员的授权删除才会执行物理删除。冲突仍需连同双方说法、状态和归因一起显示，不能把单边当成普通事实。
- **私聊个性化进入群聊的限制：**群聊提示词最多引用当前发言者私聊中的短、无敏感信息的首选称呼或回复风格。地址、健康、身份、关系、链接、多行或任意长文本即使被错误标成昵称或风格，也不会从私聊进入群聊。
- **敏感信息：**硬秘密（例如密码、令牌、验证码、支付凭据和私钥）在任何模式都不保存。自动群聊学习还会拒绝地址、健康、联系方式等敏感信息，并按实际值判断而不是信任模型标签。用户通过当前群的 `/remember` 创建显式群记忆时，可以有意共享普通敏感个人信息，但硬秘密仍会被拒绝。
- **图片与物理删除：**图片只在当前进程内短暂保留，记忆作业只持久化文本和无正文元数据。授权物理删除会同时清除 Claim、检索索引、同源证据摘录和关联作业文本，并终止尚未完成的同源作业；SQLite 清理未完成时会明确返回可重试状态。
- **旧数据：**旧版 JSON 记忆不会导入，也不会参与检索；启动迁移只处理兼容的聊天历史。现行结构化记忆以 SQLite 为唯一来源。
- **记忆操作命令：**
  - `/remember <内容>`：手动写入记忆；
  - `/memories [查询词]`：查看授权可用的记忆列表及 ID；
  - `/forget <ID或内容>`：按所有权和作用域执行物理删除、撤回或争议，结果会明确说明实际动作。

## 健康检查

启动后在另一个 PowerShell 窗口执行：

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

成功会返回 `status=ok`，并显示机器人名称、Gemini / DeepSeek 是否配置、OneBot 地址和群聊 `@` 开关。这个端点只报告配置状态，不会实际请求模型或 OneBot，因此 `ok` 不代表所有外部服务都可用。

## 常见问题

### python-dotenv could not parse statement

通常是 `.env` 中引号没有成对，或混入了不合法的赋值。对照 `.env.example`，让每项保持 `KEY=value`，然后重启。角色设定应保存在 `config/persona.md`，不要作为 `.env` 多行值填写。

### `CHAT_MODELS` 配置错误

确认使用英文逗号分隔模型，并使用第一个英文冒号分隔提供商和模型名，例如 `gemini:模型名,deepseek:模型名`。不支持 Gemini、DeepSeek 之外的提供商。

### Gemini 返回 404 或模型不存在

确认 `GEMINI_URL=https://generativelanguage.googleapis.com/v1`，并确认 `CHAT_MODELS` 中的 Gemini 模型名确实对当前账号开放。不要把旧 OpenAI 兼容端点填入 `GEMINI_URL`。

### 401 Unauthorized

这通常发生在 qqbot → OneBot 请求方向：`ONEBOT_ACCESS_TOKEN` 与 OneBot HTTP API 的令牌不一致，或 OneBot 要求令牌但本地留空。统一两边的值并重启。若是 OneBot → qqbot 的 `CALLBACK_SECRET` 不匹配，本项目回调入口返回 403，应检查回调请求头。

### 提示"所有模型暂时不可用"或缺 Key

`CHAT_MODELS` 中列出的每个提供商都必须配置对应 API Key；不需要的模型应直接从链中删除，而不是留空 Key。确认提供商名称和模型名正确，账号有权访问该模型，URL 与代理可达。查看终端日志可判断是缺 Key、401/403、429、超时还是服务端错误。

### 能收到回调但不能发信，或完全收不到回调

- 收不到：确认 OneBot 事件上报地址是 qqbot 的 `http://主机:端口/`、网络与防火墙可达、请求是 POST JSON，并检查 `CALLBACK_SECRET` 请求头。只发 `@` 而没有正文不会回复。
- 能收到但发不出：检查 `ONEBOT_API_URL`、`ONEBOT_ACCESS_TOKEN`、OneBot 的 `send_private_msg` / `send_group_msg` HTTP API 和终端中的状态码。

### 图片提示地址无效、读取失败或无法识别

确认 OneBot 上报了图片 URL 或可供 `get_image` 解析的文件标识，地址是可访问的公网 HTTP(S)，格式和大小符合限制。代理错误也会导致下载失败。下载成功但识别失败时，检查当前模型是否支持图片理解；可调整模型链后重试。

### 搜索或模型网络失败，怎样使用代理

把 `PROXY_URL` 设为完整代理地址并重启，例如 `http://127.0.0.1:7890`。确认代理程序实际监听该端口；若配置代理后更差，先留空测试直连。Tavily Key 可选，未配置时搜索仍可尝试 DDGS。

### PowerShell 不能激活虚拟环境或找不到 Python

当前窗口可先执行 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`，再运行 `.\.venv\Scripts\Activate.ps1`；这不会永久修改系统策略。若 `python` 不存在，重新安装 Python 并加入 PATH，也可尝试 Windows 上的 `py -3.11`。确认命令提示符前出现 `(.venv)` 后再安装依赖。

## 安全建议

- `.env`、API Key、OneBot token、回调密钥、管理员 QQ 号和 `qqbot_data/` 都是敏感信息，不要截图、提交或粘贴到公开日志。
- 为 `ONEBOT_ACCESS_TOKEN` 与 `CALLBACK_SECRET` 使用不同的高强度随机值，并在泄露后立即轮换。
- 默认保持 `BOT_HOST=127.0.0.1`。跨机器部署时使用防火墙白名单、反向代理和 TLS。
- 定期备份数据目录，并限制只有运行账号可读写。记忆内容来自用户，不应被当作可信指令。
- 依赖和 OneBot 客户端应及时更新；升级后先运行测试并检查变更说明。

## 开发与测试

在已激活的虚拟环境中运行：

```powershell
# 全部 unittest
python -m unittest discover -s tests -t . -v

# 语法编译检查
python -m compileall -q src tests run_bot.py

# 只运行 README / 配置一致性测试
python -m unittest tests.test_readme_guide -v
```

专项测试会用 AST 从 `src/config.py` 提取运行时环境变量，并确保 README 和 `.env.example` 同步。增加或删除配置项时，应同时更新源码、模板、参数表和测试。

## 项目结构

```text
run_bot.py                     启动入口
启动qqbot.bat                  Windows 启动脚本
.env.example                   可复制的安全配置模板
src/config.py                  dotenv 读取、默认值与路径解析
src/main.py                    Flask 回调、鉴权、群聊 @ 与回复
src/messaging.py               消息去重、会话 FIFO 和线程池
src/router.py                  普通聊天与 / 命令路由
src/chat/                      LLM 对话、提示词、历史与记忆
src/commands/                  search/help/reset/remember 等命令
src/services/                  模型、OneBot、搜索、图片与页面读取
src/utils/                     JSON 存储、数据迁移等工具
tests/                         unittest 回归测试
qqbot_data/                    运行时历史和记忆（不应提交）
```

## 运行限制

这是轻量单进程应用，直接使用 Flask 内置服务器和进程内聊天线程池；它没有跨进程聊天队列、分布式锁、集群调度或完整可观测性。消息去重只保留有限的近期 ID 且重启后清空，处理中进程退出的聊天回复不会自动恢复；只有结构化记忆学习作业由 SQLite 持久化恢复。若需要公网高可用、多个副本或严格聊天投递保证，需要在项目之外增加成熟的 WSGI 部署、外部消息队列、共享存储、监控和入口安全层。
