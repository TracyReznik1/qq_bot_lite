# Chat URL Direct Fetch & Injection Defense Task Tracker

| Task ID | Task Description | Status | Evidence |
| :--- | :--- | :---: | :--- |
| Task 19 | 出站安全扫描与敏感凭据脱敏兜底（`send_reply` 中的出站拦截） | COMPLETED | `python -B -m unittest tests.test_url_chat_injection_defense -v` (2 passed) |
| Task 20 | XML 语义沙箱封套与间接注入防御 Prompt（`format_external_webpage_sandbox` 与 System Prompt 规则升级） | COMPLETED | `python -B -m unittest tests.test_url_chat_injection_defense -v` (5 passed) |
| Task 21 | 聊天 URL 前置自动直读与搜索短路流程（URL 自动探测、5s超时抓取、平滑降级与历史记录轻量化） | COMPLETED | `python -B -m unittest tests.test_url_chat_flow -v` (3 passed) |
| Task 22 | 针对性单元测试与全量回归验证 | COMPLETED | `python -B -m unittest discover -s tests -t . -v` (510 passed), `compileall` clean, `git diff --check` clean |
| Task 23 | 修复 URL 提取贪婪吞噬紧邻中文问题（限制 URI 安全字符集并覆盖边界测试） | COMPLETED | `python -B -m unittest tests.test_url_chat_flow -v` (5 passed), 全量 512 passed |
| Task 24 | 配置层扩展（`src/config.py` 支持 `MEMORY_GEMINI_API_KEY` 与 `MEMORY_DEEPSEEK_API_KEY` 及其校验与回退） | COMPLETED | `python -B -m unittest tests.test_memory_model_configuration -v` (6 passed) |
| Task 25 | 模型客户端层改造（`GeminiClient`、`DeepSeekClient`、`FallbackLLMClient`、`get_memory_llm_client` 支持独立 Key 注入） | COMPLETED | `python -B -m unittest tests.test_llm_image_fallback tests.test_llm_tool_affinity tests.test_memory_model_configuration -v` (14 passed) |
| Task 26 | 文档与示例同步（`.env.example`、`README.md`、`README_EN.md` 同步更新满足 AST 校验） | COMPLETED | `python -B -m unittest tests.test_readme_guide -v` (10 passed) |
| Task 27 | 单元测试与全量回归（`tests/test_memory_api_key_isolation.py` 及全量测试验证，推送 git） | COMPLETED | `python -B -m unittest discover -s tests -t . -v` (517 passed), `compileall` clean, `git diff --check` clean |
| Task 28 | 聊天层平滑回退（`src/chat/chat_service.py` 在 `SearchMode.LIGHT` 搜索无结果时无感平滑回退至 `_plain_reply`） | COMPLETED | `python -B -m unittest tests.test_simple_search_chat_flow -v` (12 passed) |
| Task 29 | 针对性单测与全量回归（`tests/test_simple_search_chat_flow.py` 覆盖 LIGHT 回退与 STANDARD 报错保留，全量回归并推送 git） | COMPLETED | `python -B -m unittest discover -s tests -t . -v` (519 passed), `compileall` clean, `git diff --check` clean |
