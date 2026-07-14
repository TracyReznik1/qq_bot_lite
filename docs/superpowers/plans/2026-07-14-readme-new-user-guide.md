# README New User Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the short README with a complete Chinese beginner guide, document every runtime environment variable, synchronize `.env.example`, verify documentation consistency automatically, and publish the accumulated `main` commits to GitHub.

**Architecture:** Treat `src/config.py` as the source of truth. A new AST-based test derives every environment variable read by the runtime and requires both README and `.env.example` to cover that set. The README remains tutorial-first, while `.env.example` is the copyable full reference template.

**Tech Stack:** Markdown, dotenv syntax, Python 3.11 `ast`/`re`/`unittest`, Git.

---

## File map

- Create `tests/test_readme_guide.py`: derive runtime environment keys from `src/config.py`, enforce complete/unique documentation, and verify required beginner-guide topics.
- Replace `README.md`: tutorial, complete configuration reference, usage, troubleshooting, safety, tests, and project structure.
- Replace `.env.example`: all 34 runtime keys exactly once with safe defaults/placeholders.
- Preserve `.env`: never read values, rewrite it, stage it, or include it in a commit.

### Task 1: Add documentation consistency tests (RED)

**Files:**
- Create: `tests/test_readme_guide.py`
- Read: `src/config.py`
- Read: `README.md`
- Read: `.env.example`

- [ ] **Step 1: Create the source-of-truth documentation test**

Create `tests/test_readme_guide.py` with exactly this implementation:

```python
import ast
import re
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src" / "config.py"
README_PATH = ROOT / "README.md"
ENV_EXAMPLE_PATH = ROOT / ".env.example"

ENV_HELPERS = {
    "env_text",
    "env_bool",
    "env_int",
    "env_float",
    "env_csv_set",
}


def runtime_environment_names() -> set[str]:
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        first_argument = node.args[0]
        if not isinstance(first_argument, ast.Constant) or not isinstance(
            first_argument.value, str
        ):
            continue
        is_getenv = isinstance(node.func, ast.Attribute) and node.func.attr == "getenv"
        is_helper = isinstance(node.func, ast.Name) and node.func.id in ENV_HELPERS
        if is_getenv or is_helper:
            names.add(first_argument.value)
    return names


def env_example_assignments() -> list[str]:
    text = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    return re.findall(r"^([A-Z][A-Z0-9_]*)=", text, flags=re.MULTILINE)


class ReadmeGuideTests(unittest.TestCase):
    def test_readme_documents_every_runtime_environment_variable(self):
        runtime_names = runtime_environment_names()
        readme = README_PATH.read_text(encoding="utf-8")
        documented_names = set(re.findall(r"`([A-Z][A-Z0-9_]*)`", readme))

        self.assertEqual(34, len(runtime_names))
        self.assertEqual(set(), runtime_names - documented_names)

    def test_env_example_defines_every_runtime_variable_once(self):
        runtime_names = runtime_environment_names()
        assignments = env_example_assignments()
        counts = Counter(assignments)

        self.assertEqual(runtime_names, set(assignments))
        self.assertEqual([], sorted(name for name, count in counts.items() if count != 1))

    def test_readme_contains_complete_beginner_guide_topics(self):
        readme = README_PATH.read_text(encoding="utf-8")
        required_fragments = (
            "## 工作流程",
            "## Windows 快速开始",
            "## 配置 OneBot",
            "## 最小 `.env` 配置",
            "## 完整 `.env` 参数参考",
            "## 使用方法",
            "## 图片输入",
            "## 多会话并发",
            "## 数据、历史与记忆",
            "## 健康检查",
            "## 常见问题",
            "## 安全建议",
            "## 开发与测试",
            "## 项目结构",
            "python-dotenv could not parse statement",
            "401 Unauthorized",
            "/remember",
            "/globalremember",
            "MESSAGE_WORKERS",
            "CALLBACK_SECRET",
            "ONEBOT_ACCESS_TOKEN",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, readme)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
python -m unittest tests.test_readme_guide -v
```

Expected: 3 failures. The README test reports missing runtime names/topics, and `.env.example` reports that its assignment set does not equal the 34 runtime keys. Fix any syntax/import error until failures are exclusively caused by incomplete documentation.

### Task 2: Replace `.env.example` with the complete safe template

**Files:**
- Modify: `.env.example`
- Test: `tests/test_readme_guide.py`

- [ ] **Step 1: Replace `.env.example`**

Use this exact parseable template; do not copy any value from the ignored `.env`:

```dotenv
# 机器人身份
BOT_NAME=qqbot
# 包含 #、首尾空格或换行时，请使用成对引号；多行示例见 README。
BOT_PERSONA="你是一个自然、友好、简洁、可靠的 QQ 聊天助手。"

# Gemini / Google AI Studio
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions

# DeepSeek（OpenAI 兼容接口）
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_URL=https://api.deepseek.com/chat/completions

# 模型链。LLM_PROVIDER 仅用于兼容旧配置，新配置请使用 LLM_PRIMARY_PROVIDER。
LLM_PROVIDER=
LLM_PRIMARY_PROVIDER=gemini
LLM_PRIMARY_MODEL=
LLM_FALLBACK_1_PROVIDER=gemini
LLM_FALLBACK_1_MODEL=gemma-4-26b-a4b-it
LLM_FALLBACK_2_PROVIDER=deepseek
LLM_FALLBACK_2_MODEL=deepseek-v4-flash
LLM_FALLBACK_3_PROVIDER=deepseek
LLM_FALLBACK_3_MODEL=deepseek-v4-pro

# OneBot HTTP API 与 qqbot 回调服务
ONEBOT_API_URL=http://127.0.0.1:3000
ONEBOT_ACCESS_TOKEN=
CALLBACK_SECRET=
BOT_HOST=127.0.0.1
BOT_PORT=5000
REQUIRE_GROUP_AT=true

# 网络和网页搜索
PROXY_URL=
TAVILY_API_KEY=
SEARCH_MAX_RESULTS=4
REQUEST_TIMEOUT=18

# 管理员、数据、历史、记忆、并发和回复
ADMIN_QQ_IDS=
DATA_DIR=qqbot_data
HISTORY_TURNS=8
MEMORY_LIMIT=30
PERSIST_HISTORY=true
MESSAGE_WORKERS=8
MAX_REPLY_CHARS=1700
```

- [ ] **Step 2: Verify the `.env.example` test turns GREEN while README tests remain RED**

Run:

```powershell
python -m unittest tests.test_readme_guide.ReadmeGuideTests.test_env_example_defines_every_runtime_variable_once -v
python -m unittest tests.test_readme_guide.ReadmeGuideTests.test_readme_documents_every_runtime_environment_variable tests.test_readme_guide.ReadmeGuideTests.test_readme_contains_complete_beginner_guide_topics -v
```

Expected: the first command passes; the second command still has 2 documentation failures.

### Task 3: Rewrite README as the complete beginner guide

**Files:**
- Replace: `README.md`
- Test: `tests/test_readme_guide.py`

- [ ] **Step 1: Replace README with the following guide**

Use this content as the implementation baseline. Preserve the headings and every configuration row; wording may only be tightened when it remains factually equivalent and still satisfies the tests.

````markdown
# qqbot — qqbot_lite

一个通过 OneBot 11 接入 QQ 的轻量聊天机器人。项目只保留聊天、网页搜索、图片理解、历史和记忆，默认支持多个会话并行处理；不包含复杂 Agent 或内容生成功能。

## 功能与边界

支持：

- 普通聊天；遇到实时、冷门或模型不确定的信息时，模型可以自动调用网页搜索。
- `/search <关键词>` 显式关键词搜索。
- 用户发送图片或图片加文字后，由支持图片输入的模型识别并回答。
- `/remember` 个人长期记忆、`/globalremember` 管理员全局记忆。
- 聊天历史持久化与 `/reset` 当前会话重置。
- 不同会话可以并行，同一会话仍按顺序处理和回复。
- 可配置机器人名称和完整人设；每次模型回复都会参考当前身份设定。

项目不提供独立 URL 直读，也不支持图片生成、图片编辑、主动发图、视频理解、天气、B站专用能力、文件处理或其他 Agent 工具。网页搜索内部可能安全读取部分结果页，但直接发送 URL 不等于读取该页面。

## 工作流程

```text
QQ 用户
  ↓ 消息事件
OneBot 11 客户端（NapCat / Lagrange 等）
  ↓ HTTP 事件上报
qqbot_lite（Flask + 会话队列）
  ├─ 聊天 / 图片理解 → Gemini 或 DeepSeek 模型链
  ├─ 需要实时信息 → Tavily / DDGS 网页搜索
  └─ 历史与记忆 → qqbot_data/
  ↓ 调用 OneBot HTTP API
QQ 私聊或群聊回复
```

OneBot 客户端负责登录 QQ、上报消息和发送回复；qqbot_lite 负责路由消息、调用模型、搜索、保存历史和记忆。

## 前置条件

- Windows 10/11（项目也可在其他能运行 Python 的系统上使用，但本文以 Windows PowerShell 为例）。
- Python 3.11 或更高版本。
- 一个 OneBot 11 兼容客户端，例如 NapCat 或 Lagrange。
- 至少一个实际使用的模型服务 API Key：Gemini 或 DeepSeek。
- 可选的 Tavily API Key；不配置时搜索会使用 DDGS。

## Windows 快速开始

在项目目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

先按“最小 `.env` 配置”和“OneBot 配置”填写必要参数，然后启动：

```powershell
python run_bot.py
```

也可以双击 `启动qqbot.bat`。这个脚本要求项目根目录已有 `.venv`；若检测不到 Flask，会先安装 `requirements.txt` 再启动。

如果 PowerShell 拒绝执行激活脚本，可以在当前用户范围允许本地脚本：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

不想修改执行策略时，直接调用虚拟环境中的 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_bot.py
```

`.env` 在进程启动时读取；修改后必须重启机器人。

## 配置 OneBot

以下以 NapCat 的 HTTP 服务和 HTTP 事件上报为例；不同客户端的菜单名称可能不同，但地址和方向相同。

| 方向 | 默认地址 | qqbot 配置 |
|---|---|---|
| qqbot 调用 OneBot HTTP API | `http://127.0.0.1:3000` | `ONEBOT_API_URL` |
| OneBot 上报事件到 qqbot | `http://127.0.0.1:5000/` | `BOT_HOST` + `BOT_PORT` |

配置步骤：

1. 在 OneBot 客户端启用 HTTP API 服务，监听 `127.0.0.1:3000`。
2. 将 HTTP 事件上报地址设为 `http://127.0.0.1:5000/`。
3. OneBot HTTP API 如果设置了 Access Token，把相同值填入 `ONEBOT_ACCESS_TOKEN`。
4. 如果为入站回调设置 `CALLBACK_SECRET`，OneBot 上报请求必须携带以下任一方式：
   - `Authorization: Bearer <CALLBACK_SECRET>`
   - `X-QQBOT-Callback-Secret: <CALLBACK_SECRET>`
5. 启动 OneBot 客户端，再启动 qqbot_lite。

`ONEBOT_ACCESS_TOKEN` 保护“qqbot → OneBot”的 API 请求；`CALLBACK_SECRET` 验证“OneBot → qqbot”的事件回调。两者方向不同，不要求取相同值。

群聊默认只有在 @ 机器人时响应；设置 `REQUIRE_GROUP_AT=false` 后可以关闭这个要求。

## 最小 `.env` 配置

### 使用 Gemini 作为主模型

```dotenv
GEMINI_API_KEY=在这里填写你的_Gemini_Key
ONEBOT_API_URL=http://127.0.0.1:3000
ONEBOT_ACCESS_TOKEN=
```

默认 `LLM_PRIMARY_PROVIDER=gemini`，因此不必重复填写。若 OneBot 没有设置 Token，`ONEBOT_ACCESS_TOKEN` 保持为空。

### 使用 DeepSeek 作为主模型

```dotenv
DEEPSEEK_API_KEY=在这里填写你的_DeepSeek_Key
LLM_PRIMARY_PROVIDER=deepseek
ONEBOT_API_URL=http://127.0.0.1:3000
ONEBOT_ACCESS_TOKEN=
```

默认回退链还包含 Gemini 和 DeepSeek 的其他槽位。未配置对应 Key 的 provider 会被跳过；也可以在完整配置中把不需要的回退 provider 留空。

## `.env` 格式规则

- 一行只定义一个 `键=值`，不要重复键名。
- 普通注释使用单独一行 `# 注释`。
- `BOT_PERSONA` 包含 `#`、需要保留的首尾空格或换行时，使用成对的单引号或双引号。
- 多行值的开始和结束引号都必须存在。例如：

```dotenv
BOT_PERSONA="你是小Q。
回答先给结论，再给必要解释。
语气自然，不假装拥有真实意识。"
```

- 不要写成 `PROXY_URL=PROXY_URL=http://127.0.0.1:7890`；正确写法是 `PROXY_URL=http://127.0.0.1:7890`。
- 布尔值中，`1`、`true`、`yes`、`y`、`on`（忽略大小写）表示真；其他已填写文本表示假。未设置时使用参数默认值。
- `ADMIN_QQ_IDS` 可使用英文逗号或分号分隔多个 QQ 号。
- 整数或小数格式错误时会使用源码默认值；`MESSAGE_WORKERS` 的 0 或负数会归一为 1。

## 完整 `.env` 参数参考

### 机器人身份

| 参数 | 必需 | 默认值 | 含义与配置 |
|---|---|---|---|
| `BOT_NAME` | 否 | `qqbot` | 机器人显示名称，也会写入模型身份提示。 |
| `BOT_PERSONA` | 否 | 通用 QQ 聊天助手设定 | 完整角色设定。含 `#` 或多行时必须使用成对引号。 |

### Gemini

| 参数 | 必需 | 默认值 | 含义与配置 |
|---|---|---|---|
| `GEMINI_API_KEY` | 条件必需 | 空 | 模型链使用 Gemini 时必须配置。 |
| `GEMINI_MODEL` | 否 | `gemini-3.1-flash-lite` | Gemini 默认模型；是否可用取决于账号和接口实际支持。 |
| `GEMINI_URL` | 否 | `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` | Gemini OpenAI 兼容聊天接口。 |

### DeepSeek

| 参数 | 必需 | 默认值 | 含义与配置 |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | 条件必需 | 空 | 模型链使用 DeepSeek 时必须配置。 |
| `DEEPSEEK_MODEL` | 否 | `deepseek-v4-flash` | DeepSeek 默认模型；是否可用取决于账号和接口实际支持。 |
| `DEEPSEEK_URL` | 否 | `https://api.deepseek.com/chat/completions` | DeepSeek OpenAI 兼容聊天接口。 |

### 主模型和回退链

| 参数 | 必需 | 默认值 | 含义与配置 |
|---|---|---|---|
| `LLM_PROVIDER` | 否 | 空 | 旧版兼容项。仅在未设置 `LLM_PRIMARY_PROVIDER` 时作为主 provider 后备值；新配置不要使用。 |
| `LLM_PRIMARY_PROVIDER` | 否 | `gemini` | 主 provider，支持 `gemini` 或 `deepseek`。 |
| `LLM_PRIMARY_MODEL` | 否 | 空 | 主模型名。留空时使用对应 provider 的 `GEMINI_MODEL` 或 `DEEPSEEK_MODEL`。 |
| `LLM_FALLBACK_1_PROVIDER` | 否 | `gemini` | 第一个回退 provider；留空禁用该槽。 |
| `LLM_FALLBACK_1_MODEL` | 否 | `gemma-4-26b-a4b-it` | 第一个回退模型。当前项目将该默认模型视为不支持工具调用。 |
| `LLM_FALLBACK_2_PROVIDER` | 否 | `deepseek` | 第二个回退 provider；留空禁用该槽。 |
| `LLM_FALLBACK_2_MODEL` | 否 | `deepseek-v4-flash` | 第二个回退模型。 |
| `LLM_FALLBACK_3_PROVIDER` | 否 | `deepseek` | 第三个回退 provider；留空禁用该槽。 |
| `LLM_FALLBACK_3_MODEL` | 否 | `deepseek-v4-pro` | 第三个回退模型。 |

模型按主模型、回退 1、回退 2、回退 3 的顺序尝试，重复的 provider + model 组合会去重。缺少 Key、网络错误、限流、服务端错误或无效响应时会继续下一个模型。带搜索工具的请求会跳过已知不支持工具调用的模型；图片识别能力也取决于实际模型。

### OneBot 和服务监听

| 参数 | 必需 | 默认值 | 含义与配置 |
|---|---|---|---|
| `ONEBOT_API_URL` | 否 | `http://127.0.0.1:3000` | OneBot HTTP API 根地址，末尾 `/` 会自动移除。 |
| `ONEBOT_ACCESS_TOKEN` | 视 OneBot 配置 | 空 | qqbot 调用 OneBot API 时使用的 Bearer Token。 |
| `CALLBACK_SECRET` | 否但推荐 | 空 | OneBot 上报事件到 qqbot 时的入站鉴权密钥；为空则不校验回调。 |
| `BOT_HOST` | 否 | `127.0.0.1` | Flask 监听地址。除非理解网络风险，否则不要改为 `0.0.0.0`。 |
| `BOT_PORT` | 否 | `5000` | Flask 监听端口，也是事件上报 URL 的端口。 |
| `REQUIRE_GROUP_AT` | 否 | `true` | 群聊是否必须 @ 机器人后才响应。 |

### 网络和网页搜索

| 参数 | 必需 | 默认值 | 含义与配置 |
|---|---|---|---|
| `PROXY_URL` | 否 | 空 | HTTP/HTTPS 代理，例如 `http://127.0.0.1:7890`。用于模型、Tavily/DDGS、结果页和部分 QQ 图片请求。 |
| `TAVILY_API_KEY` | 否 | 空 | Tavily 搜索 Key；未配置时仍可使用 DDGS。 |
| `SEARCH_MAX_RESULTS` | 否 | `4` | 每次搜索请求的最大结果数。 |
| `REQUEST_TIMEOUT` | 否 | `18` | 单次网络请求超时秒数，可填写整数或小数。 |

### 管理、数据、历史、记忆、并发和回复

| 参数 | 必需 | 默认值 | 含义与配置 |
|---|---|---|---|
| `ADMIN_QQ_IDS` | 使用全局记忆时必需 | 空 | 允许执行 `/globalremember` 的 QQ 号，多个值用英文逗号或分号分隔。 |
| `DATA_DIR` | 否 | `qqbot_data` | 历史和记忆目录；相对路径以项目根目录为基准。 |
| `HISTORY_TURNS` | 否 | `8` | 每个会话保留的历史轮数，一轮包含用户和助手两条消息。 |
| `MEMORY_LIMIT` | 否 | `30` | 每个记忆文件最多保留的事实数。 |
| `PERSIST_HISTORY` | 否 | `true` | 是否把聊天历史保存为 JSON 文件。关闭后只保留当前进程内历史。 |
| `MESSAGE_WORKERS` | 否 | `8` | 最多同时处理的活跃会话数；同一会话仍严格顺序处理。最小值为 1。 |
| `MAX_REPLY_CHARS` | 否 | `1700` | 单条 QQ 回复的目标最大字符数；长回复会按换行或中文标点拆分，代码内部最低按 200 处理。 |

## 使用方法

普通私聊直接发送文字即可。模型会先参考全局记忆、个人记忆和当前会话上下文；遇到实时、冷门或不确定事实时可以自动调用网页搜索。

群聊默认需要先 @ 机器人：

```text
@qqbot 帮我查一下这个项目最近的更新
```

支持的命令：

| 命令 | 别名 | 作用 |
|---|---|---|
| `/search <关键词>` | `/s` | 强制执行关键词网页搜索，再由模型整理结果。 |
| `/remember <内容>` | `/memo` | 保存当前 QQ 用户的个人长期记忆。 |
| `/globalremember <内容>` | `/gremember` | 保存全局记忆，仅管理员可用。 |
| `/reset` | 无 | 清除当前会话历史和当前会话记忆。 |
| `/help` | `/h` | 显示帮助。 |

`/reset` 不会删除 `/remember` 保存的个人长期记忆，也不会删除全局记忆。

## 图片输入

- 可以直接发送图片，也可以发送“图片 + 文字问题”。
- 每条消息最多 4 张图片。
- 每张图片最大 5 MiB。
- 支持 JPEG、PNG、WebP 和 GIF。
- qqbot 会通过 OneBot 获取收到的图片并传给模型；最终能否识别取决于模型是否支持图片输入。
- 项目只能理解用户输入图片，不能生成、编辑或主动发送图片。

## 多会话并发

默认 `MESSAGE_WORKERS=8`，最多同时处理 8 个不同会话：

- 不同私聊用户可以并行。
- 同一群中的不同用户可以并行。
- 同一用户在不同群或私聊中属于不同会话，可以并行。
- 同一会话连续发送多条消息时，只会逐条处理；前一条完成并回复后才处理下一条，因此回复顺序与接收顺序一致。
- 单条消息失败不会阻塞该会话后续消息。

这只是单进程内并发，实际速度仍受模型 API、网络、代理和服务商限流影响。

## 数据、历史与记忆

默认数据目录为 `qqbot_data/`：

```text
qqbot_data/
├─ history/      # 按会话保存的聊天历史
└─ memories/     # 会话、个人和全局记忆 JSON
```

记忆参考优先级为：当前会话记忆 > 个人长期记忆 > 全局记忆；外部搜索结果与记忆冲突时，以外部信息为准。

- 私聊会话按 QQ 用户隔离。
- 群聊会话按“群号 + QQ 用户”隔离。
- `/remember` 写入个人长期记忆，可在该用户的不同会话中参考。
- `/globalremember` 写入所有用户可参考的全局记忆。
- `/reset` 只清除当前会话历史和当前会话记忆。

`DATA_DIR` 可能包含私人聊天和偏好，备份、上传或分享前请按敏感数据处理。

## 健康检查

机器人启动后访问：

```text
http://127.0.0.1:5000/health
```

返回 JSON 会显示服务状态、机器人名称、Gemini/DeepSeek 是否配置、OneBot 地址和群聊 @ 设置；不会返回 API Key。

## 常见问题

### `python-dotenv could not parse statement`

逐行检查 `.env`：是否出现重复的 `KEY=KEY=value`、未闭合引号、多行 `BOT_PERSONA` 缺少结束引号、把多行说明直接写成未注释文本，或在值中使用未正确引用的 `#`。修正后重启。

### `401 Unauthorized`

确认当前模型链使用的 provider、API Key、接口 URL、模型权限和账户状态一致。Key 修改后必须重启。不要把 Gemini Key 填到 DeepSeek，反之亦然。

### `所有模型暂时不可用，请稍后再试`

按主模型和三个回退槽依次检查：对应 Key 是否存在、模型名是否被账号支持、URL 是否正确、代理是否可用、服务是否限流或故障。日志会指出被跳过的 provider 和 model，但不要公开包含秘密的完整配置。

### 提示 API Key 未配置

模型链会跳过没有 Key 的 provider。至少给实际使用的主模型或某个可用回退模型配置相应的 `GEMINI_API_KEY` 或 `DEEPSEEK_API_KEY`。

### OneBot 收不到消息回调

检查事件上报地址是否为 `http://127.0.0.1:5000/`、qqbot 是否已监听、端口是否被占用、防火墙是否拦截，以及上报请求是否带了正确的 `CALLBACK_SECRET`。

### qqbot 收到消息但发不出回复

检查 `ONEBOT_API_URL`、OneBot HTTP 服务端口和 `ONEBOT_ACCESS_TOKEN`。同时查看 OneBot 日志是否拒绝 `send_private_msg` 或 `send_group_msg`。

### 图片地址无效或图片读取失败

确认 OneBot 的 `/get_image` API 能根据收到的图片 file ID 返回 URL；检查代理 URL、图片格式和 5 MiB 限制。若使用 Fake-IP 代理模式，QQ 临时图片必须通过有效代理访问。

### `当前模型无法识别该图片`

当前模型链已经全部失败或没有可用的图片输入模型。换用支持图片输入的模型，并确认对应 Key、模型名和接口能力。

### 代理连接失败

`PROXY_URL` 必须是一个完整 URL，例如 `http://127.0.0.1:7890`。确认代理进程和端口在运行，不要重复写键名。普通公网请求可能在代理连接失败时尝试直连；受信 QQ 图片的 Fake-IP 例外路径不会在代理失败后直连。

### PowerShell 无法激活 `.venv`

使用前文的 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，或者直接执行 `.\.venv\Scripts\python.exe run_bot.py`。

## 安全建议

- `.env` 已在 `.gitignore` 中，绝不要使用 `git add -f .env`。
- 不要在日志、截图、Issue 或聊天中公开 API Key、Token、管理员 QQ、完整私密人设或 `qqbot_data/`。
- 默认 `BOT_HOST=127.0.0.1` 只监听本机。只有理解防火墙、反向代理和鉴权后才改成 `0.0.0.0`。
- 对外暴露回调时配置高强度 `CALLBACK_SECRET`。
- OneBot HTTP API 设置了 Token 时，同步配置 `ONEBOT_ACCESS_TOKEN`。

## 开发与测试

安装依赖后运行全量测试：

```powershell
$env:BOT_NAME='qqbot'
python -m unittest discover -s tests -v
python -m compileall -q src tests run_bot.py
git diff --check
```

测试会覆盖数据迁移、身份配置、图片输入、模型回退、产品范围、并发调度和 README/`.env.example` 配置一致性。

## 项目结构

```text
run_bot.py             启动入口
启动qqbot.bat          Windows 一键启动脚本
.env.example           完整配置模板
src/main.py            Flask 回调、消息处理和回复发送
src/config.py          `.env` 配置来源
src/messaging.py       消息去重、跨会话并发和会话内 FIFO
src/router.py          命令与普通聊天路由
src/chat/              模型对话、提示词、历史和记忆
src/commands/          search/help/reset/remember/globalremember
src/services/          模型、OneBot、搜索、图片和 URL 服务
src/utils/             JSON 存储和旧数据迁移
tests/                 单元与回归测试
qqbot_data/            本地历史与记忆（不提交）
```

## 运行限制

- 模型名、模型权限和 API 可用性由服务商决定，README 记录的是当前源码默认值。
- 多会话并发仅限单进程；大量突发会话会在进程内等待线程池调度。
- 第三方 OneBot 客户端界面可能随版本变化，请以 HTTP API 和事件上报字段为准。
- 项目没有数据库或分布式任务队列，适合个人和轻量部署。
````

- [ ] **Step 2: Run all README guide tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_readme_guide -v
```

Expected: 3 tests pass.

- [ ] **Step 3: Run existing user-facing documentation regression tests**

Run:

```powershell
python -m unittest tests.test_qqbot_branding tests.test_user_facing_scope tests.test_product_scope -v
```

Expected: all tests pass. If an existing assertion expects wording removed by the complete rewrite, update only that assertion to verify the same current product rule against the new wording; do not weaken capability-boundary checks.

- [ ] **Step 4: Commit the README implementation**

Stage only the documented files:

```powershell
git add README.md .env.example tests/test_readme_guide.py
git diff --cached --check
git diff --cached --name-only
git commit -m "docs: rewrite README as complete setup guide"
```

Expected staged files: `README.md`, `.env.example`, `tests/test_readme_guide.py`, plus an existing documentation test only if Step 3 required a wording-alignment update. `.env` must not appear.

### Task 4: Verify, review, integrate, and publish

**Files:**
- Verify: `README.md`
- Verify: `.env.example`
- Verify: `tests/test_readme_guide.py`
- Verify: all accumulated local commits since `origin/main`

- [ ] **Step 1: Run the complete verification suite**

```powershell
$env:BOT_NAME='qqbot'
python -m unittest discover -s tests -v
python -m compileall -q src tests run_bot.py
git diff --check origin/main..HEAD
git status --short --branch
git check-ignore -v .env
```

Expected: all tests pass; compilation and diff checks exit 0; `.env` is ignored and absent from tracked/staged files.

- [ ] **Step 2: Verify documentation coverage and dotenv parseability independently**

```powershell
python -m unittest tests.test_readme_guide -v
python -c "from dotenv import dotenv_values; values=dotenv_values('.env.example'); print(len(values), len(set(values)))"
```

Expected: guide tests pass; the dotenv command prints `34 34` without parse warnings.

- [ ] **Step 3: Request final code/documentation review**

Review the complete diff from the pre-design base through HEAD. The reviewer must verify:

```text
- all 34 runtime keys are documented exactly once in .env.example;
- README defaults and semantics match src/config.py and runtime code;
- OneBot token and callback-secret directions are not confused;
- memory, reset, image, concurrency, model fallback, proxy, and product boundaries are accurate;
- commands and PowerShell examples are executable;
- no real .env value, API key, token, QQ number, or private persona is present;
- no Critical or Important documentation issue remains.
```

- [ ] **Step 4: Merge the reviewed branch into local `main` if an isolated worktree was used**

Use a fast-forward merge when possible, then rerun the full test suite on `main`. Do not remove the worktree until the merged-result tests pass.

- [ ] **Step 5: Synchronize and push GitHub**

From the main repository root:

```powershell
git pull --ff-only
git push origin main
git status --short --branch
```

Expected: push succeeds; final status reports `main...origin/main` with no ahead/behind count and no tracked changes. Never stage or push `.env`.
