import tempfile
import unittest
from pathlib import Path

from src.persona import PersonaConfigurationError, load_persona


class PersonaFileTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_name_and_full_markdown(self):
        path = self.root / "persona.md"
        path.write_text("# 角色\n\n- 名字：ATRI\n\n我是高性能机器人。", encoding="utf-8")

        persona = load_persona(path)

        self.assertEqual("ATRI", persona.name)
        self.assertIn("我是高性能机器人。", persona.content)

    def test_rejects_missing_empty_and_nameless_files(self):
        with self.assertRaisesRegex(PersonaConfigurationError, "不存在"):
            load_persona(self.root / "missing.md")

        empty = self.root / "empty.md"
        empty.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(PersonaConfigurationError, "为空"):
            load_persona(empty)

        nameless = self.root / "nameless.md"
        nameless.write_text("# 角色\n\n只有描述", encoding="utf-8")
        with self.assertRaisesRegex(PersonaConfigurationError, "名字"):
            load_persona(nameless)


if __name__ == "__main__":
    unittest.main()
