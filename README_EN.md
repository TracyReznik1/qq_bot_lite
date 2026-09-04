# qqbot — qqbot_lite

**[English](README_EN.md)** | **[简体中文](README.md)**

> A lightweight QQ chat bot built on the OneBot protocol, powered by Gemini and DeepSeek models. Features conversational context, automatic webpage URL direct-reading, deterministic grounded web search, multimodal vision understanding, fine-grained structured memory, and a triple-layer prompt injection defense mechanism.

---

## Features Overview

- **💬 Conversational Context**: Guarantees FIFO session ordering, supports custom persona configurations (`config/persona.md`), and provides multi-turn history management.
- **🌐 Automatic URL Direct-Reading**: Automatically detects HTTP/HTTPS URLs in user messages, fetches webpage content within a 5-second timeout, and injects it into an XML sandbox to summarize/answer, short-circuiting redundant web searches; prevents token bloating in persistent chat history.
- **🔍 Deterministic Grounded Search**: Tavily is the primary search provider, with seamless fallback to DDGS (DDGS stage timeout defaults to 15s). Normal chat is strictly locked to `LIGHT mode` (single query, replies do not expose source numbers, titles, or URLs), while explicit `/search` uses `STANDARD mode` (multi-query with source citations). Supports `/skip` to bypass web search entirely. Transparently handles network unavailability, insufficient evidence, and inconsistent premise/entity names with fixed boundary degradation, ensuring answers are grounded only on available evidence without hallucinating online sources. Automatically removes date filters and retries Tavily once if parameter range constraints are rejected. Search traces record audit metadata while stripping all sensitive content.
- **🛡️ Triple-Layer Prompt Injection Defense**:
  1. **XML Semantic Sandbox**: External webpage text is fully escaped with XML entities and encapsulated within `<external_webpage_content>` tags;
  2. **Authoritative Boundary Constraints**: System prompt enforces the highest security hierarchy, forbidding the model from executing any instructions or roleplay prompts contained in external web content;
  3. **Outbound Credential Redaction**: All replies pass through an automated secret scanner before dispatch via OneBot, redacting API keys, passwords, and private keys into `[redacted:credential]`.
- **🖼️ Multimodal Vision Input**: Supports sending single images or mixed image+text messages (up to 4 images per message, max 5 MiB each) for visual recognition and understanding; image generation, editing, or proactive image sending are not supported.
- **🧠 Fine-Grained Structured Memory**: SQLite-backed persistent memory supporting private, group, and global scopes, with full lifecycle management including automatic extraction, correction, retraction, disputing, and physical deletion.
- **Clean Scope**: Intentionally focused with clear boundaries; does not support video understanding, weather/Bilibili plugins, or complex multi-agent setups. `/search <keyword>` focuses strictly on keyword search (does not provide standalone URL direct-fetch); chat-based URL direct-reading is handled automatically by the system.

```text
QQ User
  ↓ Message event
OneBot Client
  ↓ HTTP POST callback
Flask / Per-session FIFO Queue
  ├─→ Command router (/search, /skip, /remember, /reset, /help, etc.)
  └─→ LLM ←→ URL Direct-Reading Sandbox / Grounded Search / History / Memory / Vision
  ↓ Outbound Secret Redactor
OneBot HTTP API
  ↓ Reply text
QQ User
```

---

## Windows Quick Start

Open PowerShell in the project root directory:

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Copy configuration template and configure API Keys
Copy-Item .env.example .env
notepad .env

# 4. Start the bot process
python run_bot.py
```

> You can also double-click `启动qqbot.bat` to automatically activate the environment and launch the bot.

---

## Configuring OneBot

Communication between qqbot and OneBot clients (e.g. NapCat, Lagrange) uses a two-way HTTP setup:

1. **qqbot → OneBot:** Enable OneBot's HTTP API (default: `http://127.0.0.1:3000`) and set its base URL in `ONEBOT_API_URL`. If the OneBot HTTP API requires access authentication, set the same token in `ONEBOT_ACCESS_TOKEN`; qqbot will include `Authorization: Bearer <token>` in requests.
2. **OneBot → qqbot:** Add an HTTP event reporting endpoint in OneBot pointing to `http://127.0.0.1:5000/` (port follows `BOT_HOST` / `BOT_PORT`), using OneBot 11 format. If `CALLBACK_SECRET` is set, OneBot callbacks must provide either `Authorization: Bearer <secret>` or `X-QQBOT-Callback-Secret: <secret>`.

- Group chat defaults to `REQUIRE_GROUP_AT=true`, meaning the bot only responds when explicitly `@` mentioned. Set to `false` to respond to every message in group chats.

---

## Full `.env` Parameter Reference

All configurations are maintained in `.env` (`KEY=value` format, lines starting with `#` are comments).

### Model Chain Configuration
- Model chains use `provider:model_name` syntax (comma-separated). The first entry is the primary model; subsequent entries serve as fallbacks. Supported providers: `gemini` and `deepseek`.
- Gemini uses native `generateContent` REST API (default `GEMINI_URL=https://generativelanguage.googleapis.com/v1`), while DeepSeek uses an OpenAI-compatible endpoint. Example:
  ```dotenv
  CHAT_MODELS=gemini:gemini-2.5-flash,deepseek:deepseek-chat
  GEMINI_API_KEY=your_gemini_key
  DEEPSEEK_API_KEY=your_deepseek_key
  ```

### Complete Parameter Summary Table

| Parameter | Required | Default | Description |
|---|---|---|---|
| `BOT_HOST` | Optional | `127.0.0.1` | Local Flask listener address. |
| `BOT_PORT` | Optional | `5000` | Local Flask listener port, must match OneBot callback destination. |
| `CALLBACK_SECRET` | Recommended | Empty | Secret token for OneBot → qqbot callback authentication. |
| `ONEBOT_API_URL` | Usually Required | `http://127.0.0.1:3000` | Root URL for calling the OneBot HTTP API. |
| `ONEBOT_ACCESS_TOKEN` | Conditional | Empty | Bearer token for qqbot → OneBot API requests. |
| `REQUIRE_GROUP_AT` | Optional | `true` | Whether an explicit `@` mention is required in group chats. |
| `ADMIN_QQ_IDS` | Conditional | Empty | Comma/semicolon-separated QQ IDs with admin rights for `/globalremember`. |
| `CHAT_MODELS` | Required | None | Conversation model chain, e.g. `gemini:gemini-2.5-flash`. Keys required for listed providers. |
| `MEMORY_MODELS` | Optional | Empty | Dedicated model chain for structured memory extraction; defaults to `CHAT_MODELS` if unset. |
| `GEMINI_API_KEY` | Conditional | Empty | API Key for Google Gemini. |
| `GEMINI_URL` | Optional | `https://generativelanguage.googleapis.com/v1` | Base API URL for Gemini (appends generateContent automatically). |
| `DEEPSEEK_API_KEY` | Conditional | Empty | API Key for DeepSeek. |
| `DEEPSEEK_URL` | Optional | `https://api.deepseek.com/chat/completions` | API endpoint for DeepSeek chat completions. |
| `TAVILY_API_KEY` | Optional | Empty | API Key for primary search provider Tavily; falls back to DDGS if unset or unavailable. |
| `PROXY_URL` | Optional | Empty | Global HTTP/HTTPS proxy address, e.g. `http://127.0.0.1:7890`. |
| `SEARCH_MAX_RESULTS` | Optional | `4` | Maximum number of search documents returned per query. |
| `SEARCH_PLANNER_TIMEOUT` | Optional | `8.0` | Timeout in seconds for search query planning. |
| `SEARCH_TAVILY_TIMEOUT` | Optional | `8.0` | Timeout in seconds for Tavily query stage. |
| `SEARCH_DDGS_TIMEOUT` | Optional | `15.0` | Timeout in seconds for DDGS fallback query stage. |
| `SEARCH_READER_TIMEOUT` | Optional | `5.0` | Timeout in seconds for webpage document reading stage. |
| `SEARCH_RANKER_TIMEOUT` | Optional | `10.0` | Timeout in seconds for search result ranking stage. |
| `SEARCH_ANSWER_TIMEOUT` | Optional | `20.0` | Timeout in seconds for search answer generation stage. |
| `REQUEST_TIMEOUT` | Optional | `18.0` | General HTTP request timeout in seconds for LLM, OneBot, etc. |
| `DATA_DIR` | Optional | `qqbot_data` | Directory for local data storage (history, memory database). |
| `HISTORY_TURNS` | Optional | `8` | Number of recent dialog turns kept in context (user+assistant count as 1 turn each). |
| `PERSIST_HISTORY` | Optional | `true` | Whether to persist chat history to disk; if false, history is memory-only. |
| `MESSAGE_WORKERS` | Optional | `8` | Number of worker threads for handling concurrent active sessions. |
| `MAX_REPLY_CHARS` | Optional | `1700` | Maximum character length for a single reply chunk before splitting. |

---

## Usage & Commands

Messages not starting with `/` enter normal conversation. If a message contains an HTTP/HTTPS URL, the bot will automatically direct-read the webpage content in a sandbox, short-circuiting search; general questions will trigger grounded web search in LIGHT mode as needed.

| Command | Alias | Description |
|---|---|---|
| `/search <keyword>` | `/s <keyword>` | Explicit web search (STANDARD mode), generating multiple queries with source links. |
| `/skip [prompt]` | None | Bypass web search entirely (SKIP mode), directly answered by the LLM with zero search latency/traces. |
| `/remember <text>` | `/memo <text>` | Save personal preferences or current group-specific memory. |
| `/globalremember <text>` | `/gremember <text>` | Save global shared memory configurations (Admin only). |
| `/memories [query]` | None | View or search authorized memory entries accessible in the current scope. |
| `/forget <ID or text>` | None | Delete, retract, or mark dispute on specified memory entries based on permissions. |
| `/reset` | None | Clear the conversation history context of the current session; does not delete memories. |
| `/help` | `/h` | Display usage instructions, available commands, and capability boundaries. |

---

## Image Input

Supports sending images directly or mixed image+text messages for multimodal vision understanding.
- **Limits**: Up to 4 images per message, max 5 MiB per image; supports JPEG, PNG, WebP, GIF.
- **Mechanism**: Reads public HTTP(S) image URLs from OneBot events or resolves file identifiers via `get_image` API. Images are retained ephemerally in memory during inference and are never persisted to disk.
- **Boundaries**: Image understanding only; image generation, editing, or proactive image sending are not supported.

---

## Multi-Session Concurrency

Uses a thread pool session dispatcher (`MESSAGE_WORKERS=8` by default):
- **Parallel & Isolated**: Independent sessions execute in parallel up to the worker thread limit. Private chats are isolated by QQ ID; group chats are isolated by `Group ID + QQ ID`.
- **FIFO Ordering**: Messages within the same session are strictly processed in FIFO order to prevent reply sequence scrambling.
- **Lifecycle**: In-flight chat replies and deduplication states reside in process memory; memory learning jobs and texts are persisted in SQLite for asynchronous consumption.

---

## Data, History, and Structured Memory

Data is stored in `qqbot_data/` (`history/` and `memory.sqlite3`):
- **Chat History**: Preserves recent `HISTORY_TURNS` interactions, restored upon restart according to configuration. `/reset` clears the current session history at any time.
- **Ephemeral Image Isolation**: Images are retained only briefly within the current process and are not persisted in memory tasks.
- **Structured Memory Extraction**: After the reply turn finishes, factual text is handed off to an asynchronous background worker for fine-grained claim extraction (an extra background model call). Features eventual consistency, resilient against crash recovery via SQLite jobs.
- **Scopes & Lifecycle**: Supports private memories (user-only), group memories (shared within group), and global memories (admin-only). Full support for corrections, retractions, disputes, and physical deletions.
- **Privacy & Security**: User nicknames are filtered safely before entering group prompts; hard credentials (passwords, tokens, keys) are strictly rejected. Legacy JSON memory is deprecated in favor of SQLite.

---

## Runtime Boundaries

Designed as a lightweight, single-process desktop/server bot running on Flask's built-in server with thread pooling. Does not include distributed queueing, cross-process clustering, or complex telemetry. Deploying multiple instances requires external load balancing and shared task queues.

---

## FAQ & Troubleshooting

### python-dotenv could not parse statement
Usually caused by unclosed quotation marks or unexpected line breaks in `.env`. Ensure each line follows `KEY=value` and restart.

### 401 Unauthorized
Check the qqbot → OneBot request direction: verify `ONEBOT_ACCESS_TOKEN` matches the OneBot API Access Token. If the OneBot callback returns 403, verify the `CALLBACK_SECRET` header.

### Model 404 or Unavailable
Verify model names in `CHAT_MODELS` are accurate and supported by your account. For Gemini, ensure `GEMINI_URL=https://generativelanguage.googleapis.com/v1` without extra paths.

### Image Fetch Failed or Format Unsupported
Ensure OneBot provides reachable image URLs, verify whether proxy settings affect image downloads, and confirm that the active model supports Vision capabilities.

---

## Development & Testing

```powershell
# Run full unit test suite (512 tests)
python -B -m unittest discover -s tests -t . -v

# Syntax and compilation check
python -B -m compileall -q src tests run_bot.py

# Validate README and environment variable consistency
python -B -m unittest tests.test_readme_guide -v
```
