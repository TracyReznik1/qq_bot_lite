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


def visible_markdown(markdown: str) -> str:
    visible_lines: list[str] = []
    in_comment = False
    fence_character = ""
    fence_length = 0
    for line in markdown.splitlines(keepends=True):
        if fence_character:
            fence = re.match(
                r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*(?:\r?\n)?$",
                line,
            )
            if (
                fence
                and fence.group(1)[0] == fence_character
                and len(fence.group(1)) >= fence_length
            ):
                fence_character = ""
                fence_length = 0
            continue

        visible_parts: list[str] = []
        remainder = line
        while remainder:
            if in_comment:
                _, end, remainder = remainder.partition("-->")
                if not end:
                    remainder = ""
                else:
                    in_comment = False
                continue
            before, start, after = remainder.partition("<!--")
            visible_parts.append(before)
            if not start:
                remainder = ""
            else:
                in_comment = True
                remainder = after
        visible_line = "".join(visible_parts)

        fence = re.match(
            r"^[ \t]{0,3}(`{3,}|~{3,})([^\r\n]*)(?:\r?\n)?$",
            visible_line,
        )
        if fence and not (fence.group(1)[0] == "`" and "`" in fence.group(2)):
            fence_character = fence.group(1)[0]
            fence_length = len(fence.group(1))
            continue
        visible_lines.append(visible_line)
    return "".join(visible_lines)


def markdown_headings(markdown: str, level: int) -> list[str]:
    hashes = "#" * level
    return [
        match.group(1).strip()
        for match in re.finditer(
            rf"^{re.escape(hashes)}[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$",
            visible_markdown(markdown),
            flags=re.MULTILINE,
        )
    ]


def markdown_section(markdown: str, level: int, heading: str) -> str:
    visible = visible_markdown(markdown)
    matches = list(
        re.finditer(
            r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$",
            visible,
            flags=re.MULTILINE,
        )
    )
    for index, match in enumerate(matches):
        if len(match.group(1)) != level or match.group(2).strip() != heading:
            continue
        end = len(visible)
        for next_match in matches[index + 1 :]:
            if len(next_match.group(1)) <= level:
                end = next_match.start()
                break
        return visible[match.end() : end]
    return ""


def level_two_headings(markdown: str) -> list[str]:
    return markdown_headings(markdown, 2)


def level_three_headings(markdown: str) -> list[str]:
    return markdown_headings(markdown, 3)


def level_two_section(markdown: str, heading: str) -> str:
    return markdown_section(markdown, 2, heading)


def level_three_section(markdown: str, heading: str) -> str:
    return markdown_section(markdown, 3, heading)


def markdown_statements(markdown: str) -> list[str]:
    statements: list[str] = []
    for line in visible_markdown(markdown).splitlines():
        if re.match(r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+", line):
            statements.append(line.strip())
            continue
        statements.extend(
            statement.strip()
            for statement in re.split(r"[；。！？]", line)
            if statement.strip()
        )
    return statements


def capability_polarities(statements: list[str], pattern: str) -> set[str]:
    polarities: set[str] = set()
    for statement in statements:
        markers = list(re.finditer(r"不支持|不提供|支持", statement))
        for capability in re.finditer(pattern, statement):
            preceding = [marker for marker in markers if marker.start() < capability.start()]
            if not preceding:
                continue
            marker = preceding[-1].group(0)
            polarities.add("unsupported" if marker.startswith("不") else "supported")
    return polarities


def onebot_token_bindings(statements: list[str]) -> dict[str, set[str]]:
    directions = ("qqbot → OneBot", "OneBot → qqbot")
    tokens = ("`ONEBOT_ACCESS_TOKEN`", "`CALLBACK_SECRET`")
    bindings = {direction: set() for direction in directions}
    for statement in statements:
        direction_matches = sorted(
            (
                (match.start(), match.end(), direction)
                for direction in directions
                for match in re.finditer(re.escape(direction), statement)
            ),
            key=lambda item: item[0],
        )
        for token in tokens:
            for token_match in re.finditer(re.escape(token), statement):
                distances = [
                    (
                        min(
                            abs(token_match.start() - direction_end),
                            abs(direction_start - token_match.end()),
                        ),
                        direction,
                    )
                    for direction_start, direction_end, direction in direction_matches
                ]
                if not distances:
                    continue
                nearest_distance = min(distance for distance, _ in distances)
                for distance, direction in distances:
                    if distance == nearest_distance:
                        bindings[direction].add(token)
    return bindings


def readme_semantic_errors(markdown: str) -> list[str]:
    errors: list[str] = []

    boundaries = level_two_section(markdown, "功能与边界")
    boundary_statements = markdown_statements(boundaries)
    supported = (
        ("聊天", r"聊天"),
        ("网页搜索", r"网页搜索"),
        ("图片理解", r"图片理解"),
        ("历史", r"历史"),
        ("记忆", r"记忆"),
    )
    unsupported = (
        "独立 URL 直读",
        "图片生成",
        "图片编辑",
        "主动发图",
        "视频",
        "天气",
        "B站专用能力",
        "文件处理",
        "复杂 Agent",
    )
    for capability, pattern in supported:
        if capability_polarities(boundary_statements, pattern) != {"supported"}:
            errors.append(f"功能与边界缺少肯定支持：{capability}")
    for capability in unsupported:
        if capability_polarities(
            boundary_statements, re.escape(capability)
        ) != {"unsupported"}:
            errors.append(f"功能与边界缺少明确边界：{capability}")

    onebot = level_two_section(markdown, "配置 OneBot")
    onebot_statements = markdown_statements(onebot)
    token_bindings = onebot_token_bindings(onebot_statements)
    if token_bindings["qqbot → OneBot"] != {"`ONEBOT_ACCESS_TOKEN`"}:
        errors.append("配置 OneBot 未把 ONEBOT_ACCESS_TOKEN 绑定到 qqbot → OneBot")
    if token_bindings["OneBot → qqbot"] != {"`CALLBACK_SECRET`"}:
        errors.append("配置 OneBot 未把 CALLBACK_SECRET 绑定到 OneBot → qqbot")

    data = level_two_section(markdown, "数据、历史与记忆")
    if not re.search(r"私聊按\s*QQ\s*用户(?:账号)?隔离", data):
        errors.append("数据、历史与记忆缺少私聊按 QQ 用户隔离")
    if not re.search(
        r"群聊按[“\"]?群号\s*\+\s*QQ\s*用户[”\"]?隔离",
        data,
    ):
        errors.append("数据、历史与记忆缺少群聊按群号 + QQ 用户隔离")

    faq = level_two_section(markdown, "常见问题")
    faq_headings = level_three_headings(faq)
    for heading in (
        "python-dotenv could not parse statement",
        "401 Unauthorized",
    ):
        if heading not in faq_headings:
            errors.append(f"常见问题缺少真实三级标题：{heading}")

    return errors


class ReadmeGuideTests(unittest.TestCase):
    def test_markdown_section_helpers_ignore_comments_and_fenced_code(self):
        markdown = """
<!--
## 注释中的二级标题
### 注释中的三级标题
-->
```markdown
## 代码块中的二级标题
### 代码块中的三级标题
<!-- 代码块内的注释起始符不应影响外部 -->
```not-a-closing-fence
## 非关闭 fence 后的伪标题
```
## 真实章节
正文
### 真实问题
真实回答
## 下一章节
下一节正文
"""

        self.assertEqual(["真实章节", "下一章节"], level_two_headings(markdown))
        self.assertEqual(["真实问题"], level_three_headings(markdown))
        self.assertIn("### 真实问题", level_two_section(markdown, "真实章节"))
        self.assertNotIn("下一节正文", level_two_section(markdown, "真实章节"))
        self.assertEqual("真实回答", level_three_section(markdown, "真实问题").strip())

    def test_semantic_checks_accept_equivalent_list_punctuation(self):
        markdown = """
## 功能与边界
- 支持聊天、网页搜索、图片理解、历史和记忆
- 不提供独立 URL 直读
- 不支持图片生成、图片编辑和主动发图
- 不支持视频、天气、B站专用能力、文件处理和复杂 Agent

## 配置 OneBot
1. qqbot → OneBot 使用 `ONEBOT_ACCESS_TOKEN`。
2. OneBot → qqbot 使用 `CALLBACK_SECRET`。

## 数据、历史与记忆
私聊按 QQ 用户隔离，群聊按群号 + QQ 用户隔离。

## 常见问题
### python-dotenv could not parse statement
### 401 Unauthorized
"""

        self.assertEqual([], readme_semantic_errors(markdown))

    def test_onebot_bindings_accept_token_before_direction(self):
        statements = markdown_statements(
            "1. `ONEBOT_ACCESS_TOKEN` 用于 qqbot → OneBot。\n"
            "2. `CALLBACK_SECRET` 用于 OneBot → qqbot。\n"
        )

        self.assertEqual(
            {
                "qqbot → OneBot": {"`ONEBOT_ACCESS_TOKEN`"},
                "OneBot → qqbot": {"`CALLBACK_SECRET`"},
            },
            onebot_token_bindings(statements),
        )

    def test_semantic_checks_bind_polarity_and_onebot_direction_locally(self):
        readme = README_PATH.read_text(encoding="utf-8")
        boundary_mutant = readme.replace(
            "不支持图片生成、图片编辑、主动发图、视频、天气、B站专用能力、文件处理、"
            "独立 URL 直读或复杂 Agent。",
            "不支持图片生成，但支持图片编辑、主动发图、视频、天气、B站专用能力、"
            "文件处理、独立 URL 直读或复杂 Agent。",
        )
        self.assertIn(
            "功能与边界缺少明确边界：图片编辑",
            readme_semantic_errors(boundary_mutant),
        )

        onebot_section = level_two_section(readme, "配置 OneBot")
        onebot_mutation = re.sub(
            r"(?m)^1\..*\n2\..*$",
            "1. qqbot → OneBot 使用 `CALLBACK_SECRET`；"
            "OneBot → qqbot 使用 `ONEBOT_ACCESS_TOKEN`。",
            onebot_section,
            count=1,
        )
        onebot_mutant = readme.replace(onebot_section, onebot_mutation, 1)
        errors = readme_semantic_errors(onebot_mutant)
        self.assertIn(
            "配置 OneBot 未把 ONEBOT_ACCESS_TOKEN 绑定到 qqbot → OneBot",
            errors,
        )
        self.assertIn(
            "配置 OneBot 未把 CALLBACK_SECRET 绑定到 OneBot → qqbot",
            errors,
        )

    def test_readme_semantic_checks_reject_review_mutants(self):
        readme = README_PATH.read_text(encoding="utf-8")
        boundary_mutant = readme.replace(
            "支持聊天、自动或显式网页搜索、图片理解、对话历史，以及会话 / 个人 / 全局记忆；"
            "不支持图片生成、图片编辑、主动发图、视频、天气、B站专用能力、文件处理、"
            "独立 URL 直读或复杂 Agent。",
            "不支持聊天、自动或显式网页搜索、图片理解、对话历史，以及会话 / 个人 / 全局记忆；"
            "支持图片生成、图片编辑、主动发图、视频、天气、B站专用能力、文件处理、"
            "独立 URL 直读或复杂 Agent。",
        )

        onebot_section = level_two_section(readme, "配置 OneBot")
        onebot_mutation = onebot_section.replace(
            "`ONEBOT_ACCESS_TOKEN`", "`TOKEN_TO_SWAP`", 1
        ).replace("`CALLBACK_SECRET`", "`ONEBOT_ACCESS_TOKEN`", 1)
        onebot_mutation = onebot_mutation.replace(
            "`TOKEN_TO_SWAP`", "`CALLBACK_SECRET`", 1
        )
        onebot_mutant = readme.replace(onebot_section, onebot_mutation, 1)

        data_section = level_two_section(readme, "数据、历史与记忆")
        data_mutation = re.sub(
            r"私聊按(?:\s*QQ)?\s*用户(?:账号)?隔离，"
            r"群聊按(?:[“\"]?群号\s*\+\s*QQ\s*用户[”\"]?|群号与用户共同)隔离。?",
            "",
            data_section,
        )
        data_mutant = readme.replace(data_section, data_mutation, 1)

        faq_mutant = re.sub(
            r"^### .*python-dotenv could not parse statement.*$",
            "",
            readme,
            count=1,
            flags=re.MULTILINE,
        )
        faq_mutant = re.sub(
            r"^### .*401 Unauthorized.*$",
            "",
            faq_mutant,
            count=1,
            flags=re.MULTILINE,
        )
        faq_mutant += (
            "\n<!-- python-dotenv could not parse statement; 401 Unauthorized -->\n"
        )

        mutants = {
            "功能肯否定反转": (
                boundary_mutant,
                "功能与边界缺少肯定支持：聊天",
            ),
            "OneBot 方向反转": (
                onebot_mutant,
                "配置 OneBot 未把 ONEBOT_ACCESS_TOKEN 绑定到 qqbot → OneBot",
            ),
            "删除数据隔离": (
                data_mutant,
                "数据、历史与记忆缺少私聊按 QQ 用户隔离",
            ),
            "FAQ 仅留注释": (
                faq_mutant,
                "常见问题缺少真实三级标题：python-dotenv could not parse statement",
            ),
        }
        for name, (mutant, expected_error) in mutants.items():
            with self.subTest(name=name):
                self.assertNotEqual(readme, mutant, "变异必须实际改变 README")
                self.assertIn(expected_error, readme_semantic_errors(mutant))

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

        semantic_errors = readme_semantic_errors(readme)
        self.assertEqual([], semantic_errors, f"README 语义错误：{semantic_errors}")

        onebot = level_two_section(readme, "配置 OneBot")
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
