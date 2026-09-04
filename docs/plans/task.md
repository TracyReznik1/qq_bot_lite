# Chat URL Direct Fetch & Injection Defense Task Tracker

| Task ID | Task Description | Status | Evidence |
| :--- | :--- | :---: | :--- |
| Task 19 | 出站安全扫描与敏感凭据脱敏兜底（`send_reply` 中的出站拦截） | COMPLETED | `python -B -m unittest tests.test_url_chat_injection_defense -v` (2 passed) |
| Task 20 | XML 语义沙箱封套与间接注入防御 Prompt（`format_external_webpage_sandbox` 与 System Prompt 规则升级） | COMPLETED | `python -B -m unittest tests.test_url_chat_injection_defense -v` (5 passed) |
| Task 21 | 聊天 URL 前置自动直读与搜索短路流程（URL 自动探测、5s超时抓取、平滑降级与历史记录轻量化） | COMPLETED | `python -B -m unittest tests.test_url_chat_flow -v` (3 passed) |
| Task 22 | 针对性单元测试与全量回归验证 | COMPLETED | `python -B -m unittest discover -s tests -t . -v` (510 passed), `compileall` clean, `git diff --check` clean |
| Task 23 | 修复 URL 提取贪婪吞噬紧邻中文问题（限制 URI 安全字符集并覆盖边界测试） | COMPLETED | `python -B -m unittest tests.test_url_chat_flow -v` (5 passed), 全量 512 passed |




