import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import main
from src.services import url_fetch_service


ROOT = Path(__file__).resolve().parents[1]


class QqbotBrandingTests(unittest.TestCase):
    def _authorized(self, headers):
        fake_config = SimpleNamespace(callback_secret="secret")
        with (
            main.app.test_request_context("/", headers=headers),
            patch.object(main, "config", fake_config),
        ):
            return main.is_callback_authorized()

    def test_new_callback_secret_header_is_authorized(self):
        self.assertTrue(self._authorized({"X-QQBOT-Callback-Secret": "secret"}))

    def test_legacy_callback_secret_header_remains_compatible(self):
        with patch.object(main.logger, "warning") as warning:
            authorized = self._authorized({"X-ATRI-Callback-Secret": "secret"})

        self.assertTrue(authorized)
        warning.assert_called_once()
        self.assertNotIn("secret", " ".join(str(item) for item in warning.call_args.args))

    def test_wrong_callback_secret_is_rejected(self):
        self.assertFalse(self._authorized({"X-QQBOT-Callback-Secret": "wrong"}))

    def test_url_fetch_uses_qqbot_user_agent(self):
        self.assertEqual("qqbot-url-fetch/1.0", url_fetch_service.URL_FETCH_USER_AGENT)

    def test_qqbot_launcher_replaces_atri_launcher(self):
        self.assertTrue((ROOT / "启动qqbot.bat").is_file())
        self.assertFalse((ROOT / "启动ATRI.bat").exists())
        launcher = (ROOT / "启动qqbot.bat").read_text(encoding="utf-8")
        self.assertIn("Starting qqbot", launcher)
        self.assertNotIn("ATRI", launcher)

    def test_operator_files_describe_configurable_qqbot_identity(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertTrue(readme.startswith("# qqbot — qqbot_lite"))
        self.assertIn("BOT_NAME=qqbot", env_example)
        self.assertIn("BOT_PERSONA=你是一个自然、友好、简洁、可靠的 QQ 聊天助手。", env_example)
        self.assertIn("qqbot_data/", readme)
        self.assertNotIn("启动ATRI", readme)
        self.assertNotIn("@ATRI", readme)

    def test_runtime_atri_references_are_only_legacy_compatibility(self):
        matches = []
        for path in (ROOT / "src").rglob("*.py"):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "ATRI" in line:
                    matches.append((path.relative_to(ROOT).as_posix(), number, line.strip()))

        self.assertTrue(matches)
        for relative, _number, line in matches:
            self.assertIn(relative, {"src/main.py", "src/utils/data_migration.py"})
            self.assertTrue("X-ATRI-Callback-Secret" in line or "atri_data" in line)


if __name__ == "__main__":
    unittest.main()
