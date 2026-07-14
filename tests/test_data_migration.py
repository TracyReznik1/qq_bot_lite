import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class DataMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "atri_data"
        self.target = self.root / "qqbot_data"

    def tearDown(self):
        self.temp.cleanup()

    def test_merges_known_data_copies_missing_files_and_archives_source(self):
        write_json(self.source / "memories" / "global.json", {"facts": ["旧一", "重复", "旧二"]})
        write_json(self.target / "memories" / "global.json", {"facts": ["重复", "新一"]})
        write_json(
            self.source / "history" / "private_1.json",
            {"messages": [{"role": "user", "content": "旧问题"}, {"role": "assistant", "content": "旧回答"}]},
        )
        write_json(
            self.target / "history" / "private_1.json",
            {"messages": [{"role": "user", "content": "新问题"}, {"role": "assistant", "content": "新回答"}]},
        )
        (self.source / "legacy-note.txt").write_text("旧文件", encoding="utf-8")
        (self.source / "same.txt").write_text("旧版本", encoding="utf-8")
        self.target.mkdir(parents=True, exist_ok=True)
        (self.target / "same.txt").write_text("新版本", encoding="utf-8")

        from src.utils.data_migration import migrate_legacy_data

        backup = migrate_legacy_data(
            self.source,
            self.target,
            history_turns=2,
            memory_limit=3,
            timestamp="20260714-120000",
        )

        self.assertEqual(["重复", "旧二", "新一"], read_json(self.target / "memories" / "global.json")["facts"])
        self.assertEqual(
            ["旧问题", "旧回答", "新问题", "新回答"],
            [item["content"] for item in read_json(self.target / "history" / "private_1.json")["messages"]],
        )
        self.assertEqual("旧文件", (self.target / "legacy-note.txt").read_text(encoding="utf-8"))
        self.assertEqual("新版本", (self.target / "same.txt").read_text(encoding="utf-8"))
        self.assertEqual(self.root / "atri_data.backup-20260714-120000", backup)
        self.assertTrue(backup.is_dir())
        self.assertEqual("旧版本", (backup / "same.txt").read_text(encoding="utf-8"))
        self.assertFalse(self.source.exists())
        snapshot = (self.target / "history" / "private_1.json").read_text(encoding="utf-8")
        self.assertIsNone(migrate_legacy_data(self.source, self.target, 2, 3, timestamp="second"))
        self.assertEqual(snapshot, (self.target / "history" / "private_1.json").read_text(encoding="utf-8"))

    def test_history_keeps_old_then_new_and_applies_turn_limit(self):
        old_messages = [{"role": "user", "content": f"旧{i}"} for i in range(4)]
        new_messages = [{"role": "assistant", "content": f"新{i}"} for i in range(4)]
        write_json(self.source / "history" / "private_2.json", {"messages": old_messages})
        write_json(self.target / "history" / "private_2.json", {"messages": new_messages})

        from src.utils.data_migration import migrate_legacy_data

        migrate_legacy_data(self.source, self.target, history_turns=2, memory_limit=30, timestamp="fixed")

        contents = [item["content"] for item in read_json(self.target / "history" / "private_2.json")["messages"]]
        self.assertEqual(["新0", "新1", "新2", "新3"], contents)

    def test_existing_backup_name_gets_a_suffix(self):
        self.source.mkdir()
        (self.root / "atri_data.backup-fixed").mkdir()

        from src.utils.data_migration import migrate_legacy_data

        backup = migrate_legacy_data(self.source, self.target, 8, 30, timestamp="fixed")

        self.assertEqual(self.root / "atri_data.backup-fixed-1", backup)

    def test_no_source_returns_none_without_changing_target(self):
        self.target.mkdir()
        marker = self.target / "keep.txt"
        marker.write_text("保持", encoding="utf-8")

        from src.utils.data_migration import migrate_legacy_data

        self.assertIsNone(migrate_legacy_data(self.source, self.target, 8, 30, timestamp="fixed"))
        self.assertEqual("保持", marker.read_text(encoding="utf-8"))

    def test_source_archive_failure_restores_original_target(self):
        write_json(self.source / "memories" / "global.json", {"facts": ["旧"]})
        write_json(self.target / "memories" / "global.json", {"facts": ["新"]})

        from src.utils import data_migration

        original_archive = data_migration._archive_source
        with patch.object(data_migration, "_archive_source", side_effect=OSError("archive failed")):
            with self.assertRaises(data_migration.MigrationError):
                data_migration.migrate_legacy_data(self.source, self.target, 8, 30, timestamp="fixed")

        self.assertEqual(["新"], read_json(self.target / "memories" / "global.json")["facts"])
        self.assertTrue(self.source.exists())
        self.assertFalse((self.root / "atri_data.backup-fixed").exists())
        self.assertIsNotNone(original_archive)

    def test_invalid_known_json_keeps_source_and_original_target(self):
        invalid = self.source / "memories" / "broken.json"
        invalid.parent.mkdir(parents=True)
        invalid.write_text("not-json", encoding="utf-8")
        write_json(self.target / "memories" / "global.json", {"facts": ["保持"]})

        from src.utils.data_migration import MigrationError, migrate_legacy_data

        with self.assertRaises(MigrationError):
            migrate_legacy_data(self.source, self.target, 8, 30, timestamp="fixed")

        self.assertTrue(self.source.exists())
        self.assertEqual(["保持"], read_json(self.target / "memories" / "global.json")["facts"])


class StartupMigrationTests(unittest.TestCase):
    def test_startup_migrates_default_data_before_legacy_memory_layout(self):
        from src import main

        order = []
        previous_initialized = main._startup_initialized
        self.addCleanup(setattr, main, "_startup_initialized", previous_initialized)
        main._startup_initialized = False
        fake_config = SimpleNamespace(
            data_dir=main.BASE_DIR / "qqbot_data",
            history_turns=8,
            memory_limit=30,
        )
        with (
            patch.object(main, "config", fake_config),
            patch.object(main, "migrate_legacy_data", side_effect=lambda *args, **kwargs: order.append("directory")),
            patch.object(main, "migrate_legacy_memory_files", side_effect=lambda: order.append("memory-layout")),
        ):
            main.startup()
            main.startup()

        self.assertEqual(["directory", "memory-layout"], order)

    def test_startup_does_not_migrate_atri_data_into_custom_data_directory(self):
        from src import main

        previous_initialized = main._startup_initialized
        self.addCleanup(setattr, main, "_startup_initialized", previous_initialized)
        main._startup_initialized = False
        fake_config = SimpleNamespace(
            data_dir=main.BASE_DIR / "custom_data",
            history_turns=8,
            memory_limit=30,
        )
        with (
            patch.object(main, "config", fake_config),
            patch.object(main, "migrate_legacy_data") as migrate,
            patch.object(main, "migrate_legacy_memory_files"),
        ):
            main.startup()

        migrate.assert_not_called()

    def test_onebot_event_runs_startup_guard_before_accepting_message(self):
        from src import main

        fake_config = SimpleNamespace(callback_secret="")
        with (
            main.app.test_request_context("/", method="POST", json={"post_type": "meta_event"}),
            patch.object(main, "config", fake_config),
            patch.object(main, "startup") as startup,
        ):
            response = main.onebot_event()

        startup.assert_called_once_with()
        self.assertEqual({"status": "ok"}, response)


if __name__ == "__main__":
    unittest.main()
