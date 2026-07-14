import unittest
from pathlib import Path

from src.chat.prompt import build_system_prompt
from src.commands.help import help_text
from src.config import config


ROOT = Path(__file__).resolve().parents[1]


class UserFacingScopeTests(unittest.TestCase):
    def test_help_mentions_image_understanding_not_generation(self):
        text = help_text()
        self.assertIn("发送图片", text)
        self.assertIn("识别", text)
        self.assertIn("不支持图片生成", text)

    def test_system_prompt_allows_input_images_but_forbids_output_images(self):
        prompt = build_system_prompt("private:1")
        self.assertIn("理解用户随消息提供的图片", prompt)
        self.assertIn("不能生成、编辑或主动发送图片", prompt)

    def test_readme_describes_keyword_search_and_image_input(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("图片理解", readme)
        self.assertIn("/search <关键词>", readme)
        self.assertIn("不提供独立 URL 直读", readme)

    def test_readme_and_help_do_not_present_atri_as_identity(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        help_message = help_text()

        self.assertNotIn("ATRI", readme)
        self.assertNotIn("ATRI", help_message)
        self.assertIn(config.bot_name, help_message)


if __name__ == "__main__":
    unittest.main()
