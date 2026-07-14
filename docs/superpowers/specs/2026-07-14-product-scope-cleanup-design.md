# qqbot_lite 产品范围精简设计

## 目标

将 qqbot_lite 收敛为只提供聊天和网页搜索的 QQ 机器人，同时保留聊天所需的上下文管理、历史记录和记忆能力。删除与图片、视频、OpenAI 媒体管线、ComfyUI 和独立 URL 直读有关的残留入口、配置与不可达代码。

## 保留范围

- 普通聊天。
- 普通聊天中的自动网页搜索工具调用。
- `/search <关键词>` 显式网页搜索。
- `/help` 和 `/reset`。
- `/remember`、`/globalremember` 和完整记忆模块。
- 聊天历史持久化。
- Gemini/DeepSeek 模型回退。
- OneBot 文本消息收发、群聊 `@` 检查、消息去重和会话队列。
- 网页搜索内部对搜索结果页面的正文抓取、安全校验和证据整理。

## 删除范围

- `/search <URL>` 直接读取指定网页的特殊分支。
- `OPENAI_API_KEY` 预留配置。
- `VIDEO_*` 配置。
- `IMAGE_*` 配置。
- `COMFYUI_*` 配置。
- `OneBotClient.send_image()` 及其仅为图片发送存在的依赖和注释。
- 与上述能力相关的过期说明或容易造成误解的文档表述。

## 不在本次范围内

- 不删除记忆功能或聊天历史。
- 不删除 `url_fetch_service.py`，因为网页搜索需要它读取搜索结果页面。
- 不重命名或重构搜索抓取模块。
- 不改变搜索来源排序、相关性判断、模型回退或 OneBot 文本发送逻辑。
- 不修改 `.env`、现有聊天历史或记忆数据。
- 不新增第三方运行依赖。

## 架构与数据流

普通聊天继续由 `src.chat.chat_service` 暴露唯一工具 `search_web`。显式 `/search` 命令只把输入作为搜索关键词交给 `search_service.search()`；不再检测 URL，也不直接调用 `fetch_url()`。

`search_service` 仍可在搜索完成后调用 `url_fetch_service.fetch_url()`，读取有限数量的结果页正文。这属于网页搜索内部证据增强，不是独立的用户功能。

OneBot 客户端只保留文本消息发送。图片、视频和媒体生成没有路由、命令、工具或配置入口。

## 文件变更

- 修改 `src/config.py`：删除 OpenAI、视频、图片和 ComfyUI 配置字段。
- 修改 `src/commands/search.py`：删除 URL 检测和直接读取分支。
- 修改 `src/services/onebot_client.py`：删除 `send_image()` 和 `Path` 导入。
- 修改 `README.md`：明确 `/search` 是关键词搜索，搜索引擎内部可读取结果页，但不提供独立 URL 直读或媒体能力。
- 修改 `src/commands/help.py`：使帮助文本与实际产品范围一致。
- 修改 `.gitignore`：停止忽略 `test_*.py`，确保范围测试可提交。
- 新增 `tests/` 下的标准库 `unittest` 测试。

## 错误处理

本次不改变现有错误处理策略：搜索失败仍由搜索服务返回结构化状态，模型不可用仍由回退客户端统一报错，OneBot 文本发送仍记录异常。删除 URL 直读分支后，包含 URL 的 `/search` 输入按普通搜索词处理，其失败行为与其他搜索词一致。

## 测试策略

使用 Python 标准库 `unittest`，不增加依赖。

测试覆盖：

1. `/search` 对包含 URL 的输入仍调用网页搜索，而不调用独立 URL 读取。
2. `Config` 不再暴露 OpenAI、视频、图片和 ComfyUI 配置字段。
3. `OneBotClient` 不再暴露 `send_image()`。
4. `/remember`、`/globalremember`、`/help`、`/reset` 和 `/search` 仍注册。
5. 普通聊天仍只暴露 `search_web` 工具。
6. `search_service` 仍使用 `fetch_url()` 增强搜索结果正文。

测试遵循红—绿流程：先在当前代码上运行并确认范围测试因残留能力存在而失败，再进行最小删除，最后运行全部测试、Python 语法检查和 Git 差异检查。

## 验收标准

- 产品可达能力只包含聊天、网页搜索、聊天历史和记忆管理。
- 不存在 OpenAI 媒体、视频、图片或 ComfyUI 配置字段。
- 不存在 OneBot 图片发送方法。
- `/search <URL>` 不再直接抓取该 URL。
- 搜索内部结果页抓取仍正常保留。
- 记忆相关命令与模块不被删除。
- 文档和帮助文本与代码一致。
- 新增测试全部通过，Python 文件可解析。
