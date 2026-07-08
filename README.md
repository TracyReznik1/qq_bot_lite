# ATRI — qqbot_lite

一个运行在 QQ 里的轻量聊天机器人（OneBot + Flask），**只保留联网搜索**，不搞复杂 Agent。

## 能做什么

**自然聊天** — 普通消息都能聊，遇到不确定、实时、冷门的内容会自动搜一下再回答。

**三个常用命令**：

| 命令 | 说明 |
|---|---|
| `/search <关键词>` | 网页搜索（Tavily + DDGS） |
| `/remember <内容>` | 让 bot 记住你的偏好 |
| `/globalremember <内容>` | 全局记忆（仅管理员） |

还有 `/help`、`/reset`。

**没有的功能** — 天气、B站、URL 直读、视频理解、图片生成，一概不加。普通聊天也只配有搜索这一个工具。

## 前置依赖

- Python 3.11+
- 一个 [OneBot](https://github.com/botuniverse/onebot-11) 兼容客户端（如 NapCat、Lagrange）
- Gemini 或 DeepSeek API Key
- （可选）Tavily API Key，不配也能用 DDGS 搜

## 快速开始

```powershell
# 1. 装依赖
pip install -r requirements.txt

# 2. 改配置
copy .env.example .env
# 编辑 .env，填入 API Key 和你的 QQ 号

# 3. 启动
python run_bot.py
# 或者双击 启动ATRI.bat
```

## 配置

`.env` 就这几项：

```env
GEMINI_API_KEY=     # 推荐（免费，https://aistudio.google.com/apikey）
DEEPSEEK_API_KEY=   # 备选
TAVILY_API_KEY=     # 可选，搜索用（https://tavily.com）
ADMIN_QQ_IDS=       # 你的 QQ 号
PROXY_URL=          # 代理地址，不用留空
PERSIST_HISTORY=true
```

加上 OneBot 的连接配置（默认就够用）：

```env
ONEBOT_API_URL=http://127.0.0.1:3000
ONEBOT_ACCESS_TOKEN=
```

## OneBot 设置

ATRI 监听 `http://127.0.0.1:5000/`，向 `http://127.0.0.1:3000` 发消息。

在 OneBot 客户端里把 HTTP 事件上报地址设为 `http://127.0.0.1:5000/`。

## 群聊行为

群聊默认需要 `@ATRI` 才响应，不想用的可以 `.env` 里设 `REQUIRE_GROUP_AT=false`。

## 模型 Fallback

默认链路：Gemini 3.1 Flash-Lite → Gemma 4 26B → DeepSeek V4 Flash → DeepSeek V4 Pro

一个挂了自动切下一个，不会因为哪个服务崩了就哑巴。想只用 DeepSeek 可以改 `LLM_PRIMARY_PROVIDER=deepseek`。

## 项目结构

```text
run_bot.py          启动入口
src/main.py         Flask 回调、消息分发、群聊 @ 检查
src/router.py       / 命令 vs 普通消息路由
src/messaging.py    消息去重、会话队列
src/config.py       .env 读取
src/chat/           聊天生成、system prompt、记忆
src/commands/       命令实现（search/help/reset/remember）
src/services/       LLM 客户端、OneBot、网页搜索
src/utils/          JSON 存储、文件名清洗
启动ATRI.bat        Windows 一键启动脚本
```

本地数据放在 `atri_data/`（不提交）。
