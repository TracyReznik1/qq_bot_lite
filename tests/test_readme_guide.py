import ast
import io
import re
import unittest
from collections import Counter
from pathlib import Path

from dotenv import dotenv_values
from dotenv.parser import parse_stream


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src" / "config.py"
README_PATH = ROOT / "README.md"
ENV_EXAMPLE_PATH = ROOT / ".env.example"

ENV_HELPERS = {"env_text", "env_bool", "env_int", "env_float", "env_csv_set"}


def runtime_environment_variables() -> set[str]:
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"), filename=str(CONFIG_PATH))
    variables: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        first_argument = node.args[0]
        if not (
            isinstance(first_argument, ast.Constant)
            and isinstance(first_argument.value, str)
        ):
            continue

        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
        ):
            function_name = f"os.{node.func.attr}"
        else:
            continue

        if function_name == "os.getenv" or function_name in ENV_HELPERS:
            variables.add(first_argument.value)

    return variables


def readme_section(markdown: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}[ \t]*\n(.*?)(?=^## |\Z)",
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(1) if match else ""


class ReadmeGuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = README_PATH.read_text(encoding="utf-8")
        cls.runtime_variables = runtime_environment_variables()

    def test_readme_reference_matches_runtime_environment_variables(self):
        reference = readme_section(self.readme, "完整 `.env` 参数参考")
        self.assertTrue(reference, "README 缺少完整 .env 参数参考章节")

        table_variables = re.findall(
            r"^\|\s*`([A-Z][A-Z0-9_]*)`\s*\|",
            reference,
            flags=re.MULTILINE,
        )
        counts = Counter(table_variables)

        self.assertEqual(self.runtime_variables, set(table_variables))
        self.assertEqual(
            {},
            {name: count for name, count in counts.items() if count != 1},
            "完整参数参考中每个运行时变量必须恰好出现一次",
        )

    def test_env_example_matches_runtime_environment_variables(self):
        content = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
        bindings = list(parse_stream(io.StringIO(content)))
        errors = [binding for binding in bindings if binding.error]
        keys = [binding.key for binding in bindings if binding.key is not None]
        counts = Counter(keys)

        self.assertEqual([], errors, ".env.example 必须能被 python-dotenv 解析")
        self.assertEqual(
            {},
            {name: count for name, count in counts.items() if count != 1},
            ".env.example 中每个参数必须恰好赋值一次",
        )
        self.assertEqual(self.runtime_variables, set(keys))
        self.assertEqual(
            self.runtime_variables,
            set(dotenv_values(stream=io.StringIO(content))),
        )

    def test_readme_has_the_new_user_guide_entry_points(self):
        self.assertEqual("# qqbot — qqbot_lite", self.readme.splitlines()[0])

        headings = set(re.findall(r"^## (.+?)\s*$", self.readme, flags=re.MULTILINE))
        required = {
            "Windows 快速开始",
            "配置 OneBot",
            "完整 `.env` 参数参考",
            "使用方法",
            "常见问题",
        }
        self.assertEqual(set(), required - headings)

        windows_start = readme_section(self.readme, "Windows 快速开始")
        self.assertIn("python run_bot.py", windows_start)

    def test_readme_explains_onebot_token_directions(self):
        onebot = readme_section(self.readme, "配置 OneBot")
        self.assertRegex(
            onebot,
            r"(?m)^1\. \*\*qqbot → OneBot：\*\*.*`ONEBOT_ACCESS_TOKEN`.*"
            r"`Authorization: Bearer <令牌>`",
        )
        self.assertRegex(
            onebot,
            r"(?m)^2\. \*\*OneBot → qqbot：\*\*.*`CALLBACK_SECRET`.*"
            r"`X-QQBOT-Callback-Secret: <密钥>`",
        )

    def test_readme_records_core_usage_and_runtime_behavior(self):
        usage = readme_section(self.readme, "使用方法")
        for command in ("/search", "/remember", "/globalremember", "/reset"):
            self.assertIn(f"`{command}", usage)
        self.assertIn("不会删除个人记忆或全局记忆", usage)

        images = readme_section(self.readme, "图片输入")
        self.assertIn("每条消息最多 4 张，每张最多 5 MiB", images)

        concurrency = readme_section(self.readme, "多会话并发")
        for fact in ("MESSAGE_WORKERS=8", "不同会话可以并行", "同一会话仍按顺序"):
            self.assertIn(fact, concurrency)

        faq = readme_section(self.readme, "常见问题")
        self.assertIn("### python-dotenv could not parse statement", faq)
        self.assertIn("### 401 Unauthorized", faq)
        self.assertIn("qqbot → OneBot 请求方向：`ONEBOT_ACCESS_TOKEN`", faq)

    def test_readme_documents_structured_memory_runtime_boundaries(self):
        memory = readme_section(
            self.readme,
            "数据、历史与结构化记忆",
        )
        for fact in (
            "额外的后台模型调用",
            "最终一致",
            "SQLite 持久化",
            "纠正",
            "撤回",
            "争议",
            "物理删除",
            "私聊个性化",
            "敏感信息",
            "显式群记忆",
            "图片只在当前进程内短暂保留",
            "旧版 JSON 记忆",
        ):
            with self.subTest(fact=fact):
                self.assertIn(fact, memory)
        self.assertNotIn("普通聊天回复成功后", memory)
        self.assertIn("该轮回复处理结束后", memory)

        concurrency = readme_section(self.readme, "多会话并发")
        self.assertIn("聊天回复队列", concurrency)
        self.assertIn("进程内", concurrency)
        self.assertIn("记忆学习任务", concurrency)
        self.assertIn("SQLite", concurrency)

        limitations = readme_section(self.readme, "运行限制")
        self.assertNotIn("不是带持久任务队列", limitations)

    def test_readme_documents_chat_models_and_native_gemini(self):
        reference = readme_section(
            self.readme,
            "完整 `.env` 参数参考",
        )
        self.assertIn("`CHAT_MODELS`", reference)
        self.assertIn(
            "https://generativelanguage.googleapis.com/v1",
            self.readme,
        )
        self.assertIn("generateContent", self.readme)
        self.assertIn(
            "CHAT_MODELS=gemini:",
            self.readme,
        )

    def test_readme_documents_tavily_primary_ddgs_fallback(self):
        self.assertIn("Tavily 是主搜索提供者", self.readme)
        self.assertIn("DDGS", self.readme)
        self.assertIn("回退", self.readme)
        self.assertNotIn("DDGS 是主搜索提供者", self.readme)
        self.assertIn("DDGS 的阶段超时默认为 15 秒", self.readme)
        self.assertIn(
            "不会向 QQ 用户展示引用编号、来源标题或 URL",
            self.readme,
        )
        self.assertIn(
            "证据映射仅保留在后台校验与 Trace 中",
            self.readme,
        )
        self.assertIn(
            "`light` 约 58 秒、`standard` 约 112 秒",
            self.readme,
        )
        self.assertIn("在线搜索服务暂时不可用", self.readme)
        self.assertIn("暂未找到足以确认结论的信息", self.readme)
        self.assertIn("名称或前提不一致", self.readme)

    def test_readme_describes_resilient_search_recovery(self):
        self.assertIn("事件时间", self.readme)
        self.assertIn("网页发布时间", self.readme)
        self.assertIn("移除日期过滤后重试一次 Tavily", self.readme)
        self.assertIn("搜索摘要", self.readme)
        self.assertIn("低置信度", self.readme)
        self.assertIn("不会额外调用 LLM 复核", self.readme)

    def test_operator_docs_remove_old_model_variables(self):
        env_example = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
        combined = self.readme + "\n" + env_example
        for old_name in (
            "GEMINI_MODEL",
            "DEEPSEEK_MODEL",
            "LLM_PROVIDER",
            "LLM_PRIMARY_PROVIDER",
            "LLM_PRIMARY_MODEL",
            "LLM_FALLBACK_1_PROVIDER",
            "LLM_FALLBACK_1_MODEL",
            "LLM_FALLBACK_2_PROVIDER",
            "LLM_FALLBACK_2_MODEL",
            "LLM_FALLBACK_3_PROVIDER",
            "LLM_FALLBACK_3_MODEL",
        ):
            with self.subTest(old_name=old_name):
                self.assertNotIn(old_name, combined)


if __name__ == "__main__":
    unittest.main()
