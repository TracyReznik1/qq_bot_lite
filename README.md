# qqbot — qqbot_lite

qqbot_lite 是一个通过 OneBot 接入 QQ 的轻量聊天机器人。它把收到的私聊或群聊消息交给 Gemini / DeepSeek，并按需使用网页搜索，再通过 OneBot 回复。

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

群聊默认 `REQUIRE_GROUP_AT=true`，只有 OneBot 事件中确实包含对机器人账号的 `@` 才处理；仅 `@` 而没有正文也不会回复。设为 `false` 后，机器人会处理群内每条有效消息。`BOT_NAME` 只控制显示身份，不决定 OneBot 的 `@` 检测。

## 最小 `.env` 配置

先复制 `.env.example`，再选择一套模型。模板中的空密钥和令牌必须替换成你自己的值，下面的占位符不要照抄。

Gemini 示例：

```dotenv
BOT_NAME=qqbot
BOT_PERSONA="你是一个自然、友好、简洁、可靠的 QQ 聊天助手。"
GEMINI_API_KEY=替换为你的_Gemini_Key
LLM_PRIMARY_PROVIDER=gemini
ONEBOT_API_URL=http://127.0.0.1:3000
ONEBOT_ACCESS_TOKEN=
CALLBACK_SECRET=
```

DeepSeek 示例：

```dotenv
BOT_NAME=qqbot
BOT_PERSONA="你是一个自然、友好、简洁、可靠的 QQ 聊天助手。"
DEEPSEEK_API_KEY=替换为你的_DeepSeek_Key
LLM_PRIMARY_PROVIDER=deepseek
ONEBOT_API_URL=http://127.0.0.1:3000
ONEBOT_ACCESS_TOKEN=
CALLBACK_SECRET=
```

## `.env` 格式规则

- 一行一个 `KEY=value`；注释用单独一行的 `#` 开头。密钥通常不需要引号，但包含空格、`#` 或换行的文本应使用成对引号。
- 多行 `BOT_PERSONA` 必须有成对双引号。以下是可由 python-dotenv 解析的合法写法：

  ```dotenv
  BOT_PERSONA="你是一个简洁可靠的 QQ 助手。
  回答时先给结论，再补充必要说明。"
  ```

- 布尔项不区分大小写；`1`、`true`、`yes`、`y`、`on` 表示真，其他已填写的值表示假。完全缺失时使用源码默认值。
- `ADMIN_QQ_IDS` 是列表，多个 QQ 号可用英文逗号或分号分隔，项目会去除首尾空白和重复项。
- 整数 / 浮点数填写错误时不会阻止启动，而会回退到源码默认值；这可能掩盖拼写错误，因此修改后应查看启动日志并实际验证。
- 不要在值末尾随意追加行内注释。编辑完保存，并重启 qqbot。

## 完整 `.env` 参数参考

“默认值”均来自 `src/config.py`。空表示源码默认空字符串；“条件必需”表示启用对应能力时必须配置。

### 身份与服务

| 参数 | 必需性 | 源码默认值 | 作用与配置方法 |
|---|---|---|---|
| `BOT_NAME` | 可选 | `qqbot` | 机器人显示身份；空白也回退为默认值。 |
| `BOT_PERSONA` | 可选 | `你是一个自然、友好、简洁、可靠的 QQ 聊天助手。` | 只影响语气与角色风格，不能扩大能力边界；多行值使用成对双引号。 |
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
| `GEMINI_API_KEY` | 条件必需 | 空 | Gemini 模型使用的 Bearer Key；模型链包含 Gemini 时应配置。 |
| `GEMINI_MODEL` | 可选 | `gemini-3.1-flash-lite` | Gemini 默认模型；主模型未单独指定时使用。模型名必须是账号实际可用的名称。 |
| `GEMINI_URL` | 可选 | `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` | Gemini 的 OpenAI 兼容聊天端点。 |
| `DEEPSEEK_API_KEY` | 条件必需 | 空 | DeepSeek API Key；模型链包含 DeepSeek 时应配置。 |
| `DEEPSEEK_MODEL` | 可选 | `deepseek-v4-flash` | DeepSeek 默认模型；主模型未单独指定时使用。模型名必须与服务端一致。 |
| `DEEPSEEK_URL` | 可选 | `https://api.deepseek.com/chat/completions` | DeepSeek 的 OpenAI 兼容聊天端点。 |

至少配置 `GEMINI_API_KEY` 或 `DEEPSEEK_API_KEY` 之一；若模型链还包含另一提供商但没有它的 Key，该项会被跳过。

### 模型链

| 参数 | 必需性 | 源码默认值 | 作用与配置方法 |
|---|---|---|---|
| `LLM_PROVIDER` | 兼容项 | 空 | 仅当 `LLM_PRIMARY_PROVIDER` **缺失**时，为旧配置提供主提供商兜底。复制完整模板后应直接改后者。 |
| `LLM_PRIMARY_PROVIDER` | 可选 | `gemini` | 主提供商，支持 `gemini` 或 `deepseek`。 |
| `LLM_PRIMARY_MODEL` | 可选 | 空 | 主模型覆盖值；为空时按主提供商采用 `GEMINI_MODEL` 或 `DEEPSEEK_MODEL`。 |
| `LLM_FALLBACK_1_PROVIDER` | 可选 | `gemini` | 第 1 回退提供商；设为空可禁用该槽位。 |
| `LLM_FALLBACK_1_MODEL` | 可选 | `gemma-4-26b-a4b-it` | 第 1 回退模型。 |
| `LLM_FALLBACK_2_PROVIDER` | 可选 | `deepseek` | 第 2 回退提供商；设为空可禁用该槽位。 |
| `LLM_FALLBACK_2_MODEL` | 可选 | `deepseek-v4-flash` | 第 2 回退模型。 |
| `LLM_FALLBACK_3_PROVIDER` | 可选 | `deepseek` | 第 3 回退提供商；设为空可禁用该槽位。 |
| `LLM_FALLBACK_3_MODEL` | 可选 | `deepseek-v4-pro` | 第 3 回退模型。 |

实际顺序是主模型 → fallback 1 → fallback 2 → fallback 3，相同的“提供商 + 模型”组合会保序去重。缺 Key、网络错误、限流、服务端错误、无效或空响应等会继续尝试下一项；全部失败时返回“所有模型暂时不可用”。

模型能力不会由配置自动补齐：普通聊天会向模型提供唯一工具 `search_web`，不支持工具调用的模型会在该轮跳过；当前默认明确把 `gemma-4-26b-a4b-it` 标为不支持工具。显式 `/search` 先搜索再让模型整理，不要求模型调用工具。图片请求会沿模型链尝试，但图片理解能力仍取决于模型和端点；全部不能处理时会提示当前模型无法识别图片。

### 搜索与网络

| 参数 | 必需性 | 源码默认值 | 作用与配置方法 |
|---|---|---|---|
| `TAVILY_API_KEY` | 可选 | 空 | 配置后优先使用 Tavily；未配置、失败或结果弱时可使用 DDGS。 |
| `PROXY_URL` | 可选 | 空 | HTTP(S) 代理地址，例如 `http://127.0.0.1:7890`；模型、搜索和图片下载会按实现使用它。 |
| `SEARCH_MAX_RESULTS` | 可选 | `4` | 每个搜索提供商最多规范化的结果数，至少按 1 处理。 |
| `REQUEST_TIMEOUT` | 可选 | `18.0` | 模型、OneBot、搜索、图片等 HTTP 请求的超时秒数。 |

### 数据、历史、并发与回复

| 参数 | 必需性 | 源码默认值 | 作用与配置方法 |
|---|---|---|---|
| `DATA_DIR` | 可选 | `qqbot_data` | 数据目录；相对路径以项目根目录为基准。 |
| `HISTORY_TURNS` | 可选 | `8` | 每个会话保留的最近对话轮数；内部按用户 / 助手两条消息计算。 |
| `MEMORY_LIMIT` | 可选 | `30` | 每个记忆文件最多保留的事实数。 |
| `PERSIST_HISTORY` | 可选 | `true` | 是否把历史写入磁盘；为假时历史只在当前进程内存在。 |
| `MESSAGE_WORKERS` | 可选 | `8` | 同时处理的活跃会话工作线程数；小于 1 时按 1 处理。 |
| `MAX_REPLY_CHARS` | 可选 | `1700` | 单段回复字符上限；源码最低按 200 处理，长回复会优先按换行或标点拆分。 |

## 使用方法

非 `/` 开头的文本进入普通聊天。模型会在遇到实时、冷门或不确定事实时自行决定调用网页搜索；你也可以用显式命令强制按关键词搜索。

| 命令 | 别名 | 说明 |
|---|---|---|
| `/search <关键词>` | `/s <关键词>` | 显式网页搜索，再由模型基于结果回答；URL 也只作为关键词搜索。 |
| `/remember <内容>` | `/memo <内容>` | 写入该 QQ 账号的个人记忆，在私聊和该用户参与的群聊中可用。 |
| `/globalremember <内容>` | `/gremember <内容>` | 写入所有会话可见的全局记忆，仅 `ADMIN_QQ_IDS` 中的管理员可用。 |
| `/reset` | 无 | 清除当前会话历史与当前会话记忆；不会删除个人记忆或全局记忆。 |
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

该队列和消息去重状态只存在于单个 Python 进程内。不要直接启动多个进程并期望它们共享顺序、去重或线程上限；多实例需要额外的外部队列与共享状态，本项目没有实现。

## 数据、历史与记忆

默认数据位于 `qqbot_data/`：历史在 `qqbot_data/history/`，记忆在 `qqbot_data/memories/`。不要提交或公开这个目录。

- **历史：**最近 `HISTORY_TURNS` 轮用户 / 助手消息。私聊按用户隔离，群聊按群号与用户共同隔离。`PERSIST_HISTORY=true` 时重启后可恢复；为假时只保留内存状态。
- **会话记忆：**绑定当前私聊或“群 + 用户”会话，优先级最高；`/reset` 会清除它。当前公开命令不会新增会话记忆。
- **个人记忆：**`/remember` 写入，按 QQ 账号共享到该用户的不同会话；`/reset` 不删除。
- **全局记忆：**`/globalremember` 写入，所有用户和会话可见；只有管理员能写，`/reset` 不删除。

提示上下文中的冲突优先级为：当前会话记忆 > 个人记忆 > 全局记忆；搜索得到的外部信息与记忆冲突时，以外部信息为准。记忆是用户可写数据，只作为非可信上下文，不会改变系统规则或能力边界。

## 健康检查

启动后在另一个 PowerShell 窗口执行：

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

成功会返回 `status=ok`，并显示机器人名称、Gemini / DeepSeek 是否配置、OneBot 地址和群聊 `@` 开关。这个端点只报告配置状态，不会实际请求模型或 OneBot，因此 `ok` 不代表所有外部服务都可用。

## 常见问题

### 启动时出现 `python-dotenv could not parse statement`

通常是 `.env` 中引号没有成对、把多行角色设定写成了多个裸行，或混入了不合法的赋值。对照 `.env.example`，让每项保持 `KEY=value`；多行 `BOT_PERSONA` 使用上文的成对双引号，然后重启。

### OneBot 返回 `401 Unauthorized`

这通常发生在 qqbot → OneBot 请求方向：`ONEBOT_ACCESS_TOKEN` 与 OneBot HTTP API 的令牌不一致，或 OneBot 要求令牌但本地留空。统一两边的值并重启。若是 OneBot → qqbot 的 `CALLBACK_SECRET` 不匹配，本项目回调入口返回 403，应检查回调请求头。

### 提示“所有模型暂时不可用”或缺 Key

确认至少一个 `GEMINI_API_KEY` / `DEEPSEEK_API_KEY` 非空，提供商名称和模型名正确，账号有权访问该模型，URL 与代理可达。查看终端日志可判断是缺 Key、401/403、429、超时还是服务端错误。只配置一个提供商时，可把不需要的 fallback provider 设为空，减少无效尝试。

### 能收到回调但不能发信，或完全收不到回调

- 收不到：确认 OneBot 事件上报地址是 qqbot 的 `http://主机:端口/`、网络与防火墙可达、请求是 POST JSON，并检查 `CALLBACK_SECRET` 请求头。只发 `@` 而没有正文不会回复。
- 能收到但发不出：检查 `ONEBOT_API_URL`、`ONEBOT_ACCESS_TOKEN`、OneBot 的 `send_private_msg` / `send_group_msg` HTTP API 和终端中的状态码。

### 图片提示地址无效、读取失败或无法识别

确认 OneBot 上报了图片 URL 或可供 `get_image` 解析的文件标识，地址是可访问的公网 HTTP(S)，格式和大小符合限制。代理错误也会导致下载失败。下载成功但识别失败时，检查当前模型是否支持 OpenAI 兼容的图片输入；可调整模型链后重试。

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
$env:BOT_NAME='qqbot'
python -m unittest discover -s tests -v

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

这是轻量单进程应用，直接使用 Flask 内置服务器和进程内线程池；它不是带持久任务队列、跨进程锁、集群调度或完整可观测性的生产平台。消息去重只保留有限的近期 ID 且重启后清空；处理中进程退出的消息不会自动恢复。若需要公网高可用、多个副本或严格投递保证，需要在项目之外增加成熟的 WSGI 部署、持久队列、共享存储、监控和入口安全层。
