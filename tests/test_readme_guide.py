import ast
import re
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src" / "config.py"
README_PATH = ROOT / "README.md"
ENV_EXAMPLE_PATH = ROOT / ".env.example"

ENV_HELPERS = {"env_text", "env_bool", "env_int", "env_float", "env_csv_set"}
REQUIRED_README_FRAGMENTS = (
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


def env_example_assignments() -> list[str]:
    assignments = []
    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=.*", line)
        if match:
            assignments.append(match.group(1))
    return assignments


class ReadmeGuideTests(unittest.TestCase):
    def test_readme_documents_every_runtime_environment_variable(self):
        runtime_variables = runtime_environment_variables()
        self.assertEqual(34, len(runtime_variables))

        readme = README_PATH.read_text(encoding="utf-8")
        missing = sorted(name for name in runtime_variables if f"`{name}`" not in readme)
        self.assertEqual([], missing, f"README 缺少反引号参数名：{missing}")

    def test_env_example_matches_runtime_environment_variables_exactly_once(self):
        runtime_variables = runtime_environment_variables()
        assignments = env_example_assignments()
        counts = Counter(assignments)

        self.assertEqual(runtime_variables, set(assignments))
        self.assertEqual(
            {},
            {name: count for name, count in counts.items() if count != 1},
            "每个 .env.example 参数必须恰好赋值一次",
        )

    def test_readme_contains_complete_new_user_guide_sections_and_terms(self):
        readme = README_PATH.read_text(encoding="utf-8")
        missing = [fragment for fragment in REQUIRED_README_FRAGMENTS if fragment not in readme]
        self.assertEqual([], missing, f"README 缺少必要内容：{missing}")


if __name__ == "__main__":
    unittest.main()
