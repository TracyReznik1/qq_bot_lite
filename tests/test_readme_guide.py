import ast
import io
import logging
import re
import unittest
import warnings
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path

from dotenv import dotenv_values
from dotenv.parser import parse_stream


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src" / "config.py"
README_PATH = ROOT / "README.md"
ENV_EXAMPLE_PATH = ROOT / ".env.example"

ENV_HELPERS = {"env_text", "env_bool", "env_int", "env_float", "env_csv_set"}
REQUIRED_LEVEL_TWO_HEADINGS = (
    "功能与边界",
    "工作流程",
    "前置条件",
    "Windows 快速开始",
    "配置 OneBot",
    "最小 `.env` 配置",
    "`.env` 格式规则",
    "完整 `.env` 参数参考",
    "使用方法",
    "图片输入",
    "多会话并发",
    "数据、历史与记忆",
    "健康检查",
    "常见问题",
    "安全建议",
    "开发与测试",
    "项目结构",
    "运行限制",
)
REQUIRED_README_FRAGMENTS = (
    "python-dotenv could not parse statement",
    "401 Unauthorized",
    "/remember",
    "/globalremember",
    "MESSAGE_WORKERS",
    "CALLBACK_SECRET",
    "ONEBOT_ACCESS_TOKEN",
    "图片理解",
    "/search <关键词>",
    "不提供独立 URL 直读",
    "qqbot_data/",
    "MESSAGE_WORKERS=8",
    "不同会话可以并行",
    "同一会话仍按顺序",
)


def runtime_environment_variables() -> set[str]:
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"), filename=str(CONFIG_PATH))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        first_argument = node.args[0]
        if not isinstance(first_argument, ast.Constant) or not isinstance(
            first_argument.value, str
        ):
            continue

        function_name = None
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
        ):
            function_name = f"os.{node.func.attr}"

        if function_name == "os.getenv" or function_name in ENV_HELPERS:
            names.add(first_argument.value)
    return names


def env_example_assignments(content: str | None = None) -> list[str]:
    if content is None:
        content = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    return [
        binding.key
        for binding in parse_stream(io.StringIO(content))
        if binding.key is not None
    ]


def parsed_dotenv_values_and_diagnostics() -> tuple[dict[str, str | None], list[str]]:
    content = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    stderr = io.StringIO()
    log_output = io.StringIO()
    log_handler = logging.StreamHandler(log_output)
    dotenv_logger = logging.getLogger("dotenv.main")
    dotenv_logger.addHandler(log_handler)
    try:
        with redirect_stderr(stderr), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values = dict(dotenv_values(stream=io.StringIO(content)))
    finally:
        dotenv_logger.removeHandler(log_handler)

    diagnostics = [
        message.strip()
        for message in (
            stderr.getvalue(),
            log_output.getvalue(),
            *(str(item.message) for item in caught),
        )
        if message.strip()
    ]
    return values, diagnostics


def level_two_headings(markdown: str) -> list[str]:
    return re.findall(r"^## ([^\r\n]+)$", markdown, flags=re.MULTILINE)


def level_two_section(markdown: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$",
        markdown,
        flags=re.MULTILINE,
    )
    if match is None:
        return ""
    next_heading = re.search(r"^## ", markdown[match.end() :], flags=re.MULTILINE)
    end = match.end() + next_heading.start() if next_heading else len(markdown)
    return markdown[match.end() : end]


class ReadmeGuideTests(unittest.TestCase):
    def test_env_assignment_parser_recognizes_export_spacing_and_duplicates(self):
        assignments = env_example_assignments(
            "  BOT_NAME = first\n"
            "export EXTRA=value\n"
            "  export BOT_NAME = duplicate\n"
        )

        self.assertEqual(["BOT_NAME", "EXTRA", "BOT_NAME"], assignments)

    def test_readme_documents_every_runtime_environment_variable(self):
        runtime_variables = runtime_environment_variables()
        self.assertEqual(34, len(runtime_variables))

        readme = README_PATH.read_text(encoding="utf-8")
        reference = level_two_section(readme, "完整 `.env` 参数参考")
        self.assertTrue(reference, "README 缺少完整 .env 参数参考章节")
        table_variables = re.findall(
            r"^\|\s*`([A-Z][A-Z0-9_]*)`\s*\|",
            reference,
            flags=re.MULTILINE,
        )
        counts = Counter(table_variables)
        self.assertEqual(runtime_variables, set(table_variables))
        self.assertEqual(
            {},
            {name: count for name, count in counts.items() if count != 1},
            "完整参数参考的表格中每个运行时变量必须恰好出现一次",
        )

    def test_env_example_matches_runtime_environment_variables_exactly_once(self):
        runtime_variables = runtime_environment_variables()
        assignments = env_example_assignments()
        counts = Counter(assignments)
        parsed_values, diagnostics = parsed_dotenv_values_and_diagnostics()

        self.assertEqual(runtime_variables, set(assignments))
        self.assertEqual(
            {},
            {name: count for name, count in counts.items() if count != 1},
            "每个 .env.example 参数必须恰好赋值一次",
        )
        self.assertEqual([], diagnostics, ".env.example 必须无解析警告")
        self.assertEqual(34, len(parsed_values))
        self.assertEqual(runtime_variables, set(parsed_values))

    def test_readme_contains_complete_new_user_guide_sections_and_terms(self):
        readme = README_PATH.read_text(encoding="utf-8")
        self.assertEqual("# qqbot — qqbot_lite", readme.splitlines()[0])

        headings = level_two_headings(readme)
        missing_headings = [
            heading for heading in REQUIRED_LEVEL_TWO_HEADINGS if heading not in headings
        ]
        self.assertEqual([], missing_headings, f"README 缺少二级章节：{missing_headings}")
        self.assertEqual(
            list(REQUIRED_LEVEL_TWO_HEADINGS),
            [heading for heading in headings if heading in REQUIRED_LEVEL_TWO_HEADINGS],
            "关键二级章节顺序错误",
        )

        missing = [fragment for fragment in REQUIRED_README_FRAGMENTS if fragment not in readme]
        self.assertEqual([], missing, f"README 缺少必要内容：{missing}")

        boundaries = level_two_section(readme, "功能与边界")
        for fragment in (
            "聊天",
            "自动或显式网页搜索",
            "图片理解",
            "对话历史",
            "会话 / 个人 / 全局记忆",
            "图片生成",
            "图片编辑",
            "主动发图",
            "视频",
            "天气",
            "B站专用能力",
            "文件处理",
            "独立 URL 直读",
            "复杂 Agent",
        ):
            self.assertIn(fragment, boundaries, f"功能与边界章节缺少：{fragment}")

        onebot = level_two_section(readme, "配置 OneBot")
        self.assertRegex(onebot, r"(?s)qqbot → OneBot.+`ONEBOT_ACCESS_TOKEN`")
        self.assertRegex(onebot, r"(?s)OneBot → qqbot.+`CALLBACK_SECRET`")
        self.assertIn("Authorization: Bearer <密钥>", onebot)
        self.assertIn("X-QQBOT-Callback-Secret: <密钥>", onebot)

        reference = level_two_section(readme, "完整 `.env` 参数参考")
        self.assertRegex(
            reference,
            r"(?m)^\| `LLM_PROVIDER` \|.*仅当 `LLM_PRIMARY_PROVIDER` \*\*缺失\*\*时",
        )

        usage = level_two_section(readme, "使用方法")
        self.assertRegex(
            usage,
            r"(?m)^\| `/reset` \|.*不会删除个人记忆或全局记忆。",
        )

        images = level_two_section(readme, "图片输入")
        for fragment in ("最多 4 张", "最多 5 MiB", "JPEG", "PNG", "WebP", "GIF"):
            self.assertIn(fragment, images, f"图片输入章节缺少：{fragment}")


if __name__ == "__main__":
    unittest.main()
