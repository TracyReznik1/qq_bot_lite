import builtins
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def legacy_memory_io_guards(data_migration, *legacy_roots: Path) -> ExitStack:
    roots = {Path(root) for root in legacy_roots}

    def is_memory_path(value) -> bool:
        try:
            return "memories" in Path(value).parts
        except TypeError:
            return False

    def reject_open(path, *args, **kwargs):
        if is_memory_path(path):
            raise AssertionError(f"legacy memory path was opened: {path}")
        return original_open(path, *args, **kwargs)

    def reject_path_io(original, operation):
        def guarded(path, *args, **kwargs):
            if is_memory_path(path):
                raise AssertionError(f"legacy memory path was {operation}: {path}")
            return original(path, *args, **kwargs)

        return guarded

    def reject_copy(original, operation):
        def guarded(source, destination, *args, **kwargs):
            if is_memory_path(source) or is_memory_path(destination):
                raise AssertionError(
                    f"legacy memory path was {operation}: {source} -> {destination}"
                )
            return original(source, destination, *args, **kwargs)

        return guarded

    def reject_tree_copy(source, destination, *args, **kwargs):
        if Path(source) in roots or is_memory_path(source) or is_memory_path(destination):
            raise AssertionError(
                f"legacy data tree was copied: {source} -> {destination}"
            )
        return original_copytree(source, destination, *args, **kwargs)

    original_open = builtins.open
    original_copytree = data_migration.shutil.copytree
    stack = ExitStack()
    stack.enter_context(patch("builtins.open", reject_open))
    stack.enter_context(
        patch.object(
            Path,
            "open",
            reject_path_io(Path.open, "opened"),
        )
    )
    for method, operation in (
        ("read_text", "read"),
        ("read_bytes", "read"),
        ("write_text", "written"),
        ("write_bytes", "written"),
    ):
        original = getattr(Path, method)
        stack.enter_context(
            patch.object(Path, method, reject_path_io(original, operation))
        )
    for method in ("copy", "copy2", "copyfile"):
        original = getattr(data_migration.shutil, method)
        stack.enter_context(
            patch.object(
                data_migration.shutil,
                method,
                reject_copy(original, method),
            )
        )
    stack.enter_context(
        patch.object(data_migration.shutil, "copytree", reject_tree_copy)
    )
    return stack


class DataMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "atri_data"
        self.target = self.root / "qqbot_data"

    def tearDown(self):
        self.temp.cleanup()

    def patch_replace_failure(self, predicate, calls=None):
        original_replace = Path.replace

        def failing_replace(path, destination):
            destination = Path(destination)
            if calls is not None:
                calls.append((path, destination))
            if predicate(path, destination):
                raise OSError(f"injected replace failure: {path} -> {destination}")
            return original_replace(path, destination)

        return patch.object(Path, "replace", failing_replace)

    def write_history_pair(self, source: Path, target: Path, *, had_target: bool) -> None:
        write_json(
            source / "history" / "private_1.json",
            {"messages": [{"role": "user", "content": "旧消息"}]},
        )
        if had_target:
            write_json(
                target / "history" / "private_1.json",
                {"messages": [{"role": "assistant", "content": "新消息"}]},
            )

    @staticmethod
    def write_memory_sentinels(source: Path, target: Path) -> None:
        write_json(source / "memories" / "global.json", {"facts": ["源记忆"]})
        write_json(target / "memories" / "private.json", {"facts": ["目标记忆"]})

    def assert_memory_sentinels_unchanged(self, source: Path, target: Path) -> None:
        self.assertEqual(
            {"facts": ["源记忆"]},
            read_json(source / "memories" / "global.json"),
        )
        self.assertEqual(
            {"facts": ["目标记忆"]},
            read_json(target / "memories" / "private.json"),
        )

    def assert_clean_retry(self, source: Path, target: Path, *, had_target: bool) -> None:
        state = target.with_name(f".{target.name}.migration-state")
        self.assertFalse(state.exists())
        self.assertTrue(source.exists())
        if had_target:
            self.assertEqual(
                ["新消息"],
                [
                    item["content"]
                    for item in read_json(target / "history" / "private_1.json")["messages"]
                ],
            )
        else:
            self.assertFalse(target.exists())

        backup = self.migrate(source, target)

        self.assertIsNotNone(backup)
        expected = ["旧消息", "新消息"] if had_target else ["旧消息"]
        self.assertEqual(
            expected,
            [
                item["content"]
                for item in read_json(target / "history" / "private_1.json")["messages"]
            ],
        )

    @staticmethod
    def migrate(source: Path, target: Path):
        from src.utils.data_migration import migrate_legacy_data

        return migrate_legacy_data(source, target, 8, timestamp="fixed")

    def test_migrates_history_only_and_archives_source(self):
        write_json(
            self.source / "history" / "private_1.json",
            {"messages": [{"role": "user", "content": "旧问题"}, {"role": "assistant", "content": "旧回答"}]},
        )
        write_json(
            self.target / "history" / "private_1.json",
            {"messages": [{"role": "user", "content": "新问题"}, {"role": "assistant", "content": "新回答"}]},
        )
        (self.source / "legacy-note.txt").write_text("不迁移", encoding="utf-8")
        (self.source / "same.txt").write_text("旧版本", encoding="utf-8")
        self.target.mkdir(parents=True, exist_ok=True)
        (self.target / "same.txt").write_text("新版本", encoding="utf-8")

        from src.utils.data_migration import migrate_legacy_data

        backup = migrate_legacy_data(
            self.source,
            self.target,
            history_turns=2,
            timestamp="20260714-120000",
        )

        self.assertEqual(
            ["旧问题", "旧回答", "新问题", "新回答"],
            [item["content"] for item in read_json(self.target / "history" / "private_1.json")["messages"]],
        )
        self.assertEqual("新版本", (self.target / "same.txt").read_text(encoding="utf-8"))
        self.assertFalse((self.target / "legacy-note.txt").exists())
        self.assertEqual(self.root / "atri_data.backup-20260714-120000", backup)
        self.assertTrue(backup.is_dir())
        self.assertEqual("旧版本", (backup / "same.txt").read_text(encoding="utf-8"))
        self.assertFalse(self.source.exists())
        snapshot = (self.target / "history" / "private_1.json").read_text(encoding="utf-8")
        self.assertIsNone(migrate_legacy_data(self.source, self.target, 2, timestamp="second"))
        self.assertEqual(snapshot, (self.target / "history" / "private_1.json").read_text(encoding="utf-8"))

    def test_history_keeps_old_then_new_and_applies_turn_limit(self):
        old_messages = [{"role": "user", "content": f"旧{i}"} for i in range(4)]
        new_messages = [{"role": "assistant", "content": f"新{i}"} for i in range(4)]
        write_json(self.source / "history" / "private_2.json", {"messages": old_messages})
        write_json(self.target / "history" / "private_2.json", {"messages": new_messages})

        from src.utils.data_migration import migrate_legacy_data

        migrate_legacy_data(self.source, self.target, history_turns=2, timestamp="fixed")

        contents = [item["content"] for item in read_json(self.target / "history" / "private_2.json")["messages"]]
        self.assertEqual(["新0", "新1", "新2", "新3"], contents)

    def test_legacy_memory_sentinels_are_never_opened_or_copied_during_migration(self):
        source_sentinel = self.source / "memories" / "global.json"
        target_sentinel = self.target / "memories" / "private.json"
        write_json(source_sentinel, {"facts": ["源不得导入"]})
        write_json(target_sentinel, {"facts": ["目标不得导入"]})
        write_json(
            self.source / "history" / "private_1.json",
            {"messages": [{"role": "user", "content": "保留历史"}]},
        )
        write_json(
            self.target / "history" / "private_1.json",
            {"messages": [{"role": "assistant", "content": "保留目标历史"}]},
        )

        from src.utils import data_migration

        with legacy_memory_io_guards(
            data_migration,
            self.source,
            self.target,
        ):
            backup = data_migration.migrate_legacy_data(
                self.source,
                self.target,
                history_turns=8,
                timestamp="fixed",
            )

        self.assertIsNotNone(backup)
        self.assertEqual({"facts": ["目标不得导入"]}, read_json(target_sentinel))
        self.assertEqual(
            {"facts": ["源不得导入"]},
            read_json(backup / "memories" / "global.json"),
        )
        self.assertFalse((self.target / "memory.sqlite3").exists())
        self.assertEqual(
            ["保留历史", "保留目标历史"],
            [
                item["content"]
                for item in read_json(self.target / "history" / "private_1.json")["messages"]
            ],
        )

    def test_existing_backup_name_gets_a_suffix(self):
        self.source.mkdir()
        (self.root / "atri_data.backup-fixed").mkdir()

        from src.utils.data_migration import migrate_legacy_data

        backup = migrate_legacy_data(self.source, self.target, 8, timestamp="fixed")

        self.assertEqual(self.root / "atri_data.backup-fixed-1", backup)

    def test_no_source_returns_none_without_changing_target(self):
        self.target.mkdir()
        marker = self.target / "keep.txt"
        marker.write_text("保持", encoding="utf-8")

        from src.utils.data_migration import migrate_legacy_data

        self.assertIsNone(migrate_legacy_data(self.source, self.target, 8, timestamp="fixed"))
        self.assertEqual("保持", marker.read_text(encoding="utf-8"))

    def test_source_to_backup_failure_restores_original_target_and_allows_clean_retry(self):
        from src.utils.data_migration import MigrationError

        for had_target in (True, False):
            with self.subTest(had_target=had_target):
                root = self.root / f"source-to-backup-{had_target}"
                source = root / "atri_data"
                target = root / "qqbot_data"
                self.write_history_pair(source, target, had_target=had_target)

                with self.patch_replace_failure(
                    lambda path, destination: path == source
                    and destination.name.startswith("atri_data.backup-")
                ):
                    with self.assertRaises(MigrationError):
                        self.migrate(source, target)

                self.assert_clean_retry(source, target, had_target=had_target)

    def test_active_history_backup_failure_preserves_everything_and_allows_retry(self):
        self.write_history_pair(self.source, self.target, had_target=True)
        self.write_memory_sentinels(self.source, self.target)

        from src.utils.data_migration import MigrationError

        active = self.target / "history" / "private_1.json"
        with self.patch_replace_failure(
            lambda path, destination: path == active
            and destination.parent.parent.name.startswith(".qqbot_data.rollback-")
        ):
            with self.assertRaises(MigrationError):
                self.migrate(self.source, self.target)

        self.assert_memory_sentinels_unchanged(self.source, self.target)
        self.assertEqual(
            ["旧消息"],
            [item["content"] for item in read_json(self.source / "history" / "private_1.json")["messages"]],
        )
        self.assertEqual(
            ["新消息"],
            [item["content"] for item in read_json(active)["messages"]],
        )
        self.assertFalse(self.target.with_name(".qqbot_data.migration-state").exists())
        self.assertEqual([], list(self.root.glob("atri_data.backup-*")))
        self.assertEqual([], list(self.root.glob(".qqbot_data.*-*")))

        self.assertIsNotNone(self.migrate(self.source, self.target))
        self.assertEqual(
            ["旧消息", "新消息"],
            [item["content"] for item in read_json(active)["messages"]],
        )
        self.assertEqual(
            {"facts": ["目标记忆"]},
            read_json(self.target / "memories" / "private.json"),
        )

    def test_mid_install_failure_restores_all_history_files_and_leaves_memory_untouched(self):
        for name in ("private_1.json", "private_2.json"):
            write_json(
                self.source / "history" / name,
                {"messages": [{"role": "user", "content": f"源-{name}"}]},
            )
            write_json(
                self.target / "history" / name,
                {"messages": [{"role": "assistant", "content": f"目标-{name}"}]},
            )
        self.write_memory_sentinels(self.source, self.target)

        from src.utils.data_migration import MigrationError

        failed_name = "private_2.json"
        with self.patch_replace_failure(
            lambda path, destination: path.name == failed_name
            and path.parent.parent.name.startswith(".qqbot_data.migrating-")
            and destination == self.target / "history" / failed_name
        ):
            with self.assertRaises(MigrationError):
                self.migrate(self.source, self.target)

        self.assert_memory_sentinels_unchanged(self.source, self.target)
        for name in ("private_1.json", "private_2.json"):
            self.assertEqual(
                [f"源-{name}"],
                [item["content"] for item in read_json(self.source / "history" / name)["messages"]],
            )
            self.assertEqual(
                [f"目标-{name}"],
                [item["content"] for item in read_json(self.target / "history" / name)["messages"]],
            )
        self.assertFalse(self.target.with_name(".qqbot_data.migration-state").exists())
        self.assertEqual([], list(self.root.glob(".qqbot_data.*-*")))
        self.assertEqual([], list(self.root.glob("atri_data.backup-*")))

    def test_persistent_staging_cleanup_failure_preserves_recovery_material(self):
        self.write_history_pair(self.source, self.target, had_target=True)
        self.write_memory_sentinels(self.source, self.target)

        from src.utils import data_migration

        original_rmtree = data_migration.shutil.rmtree

        def fail_staging_cleanup(path, *args, **kwargs):
            if Path(path).name.startswith(".qqbot_data.migrating-"):
                raise OSError("persistent staging cleanup failure")
            return original_rmtree(path, *args, **kwargs)

        with patch.object(data_migration.shutil, "rmtree", side_effect=fail_staging_cleanup):
            with self.assertRaises(data_migration.MigrationError):
                self.migrate(self.source, self.target)

        self.assert_memory_sentinels_unchanged(self.source, self.target)
        self.assertEqual(
            ["旧消息"],
            [item["content"] for item in read_json(self.source / "history" / "private_1.json")["messages"]],
        )
        self.assertEqual(
            ["新消息"],
            [item["content"] for item in read_json(self.target / "history" / "private_1.json")["messages"]],
        )
        self.assertTrue(self.target.with_name(".qqbot_data.migration-state").exists())
        self.assertEqual(1, len(list(self.root.glob(".qqbot_data.migrating-*"))))
        self.assertEqual([], list(self.root.glob("atri_data.backup-*")))

        with self.assertRaises(data_migration.MigrationError):
            self.migrate(self.source, self.target)

    def test_failed_history_restore_keeps_original_history_for_manual_recovery(self):
        self.write_history_pair(self.source, self.target, had_target=True)
        self.write_memory_sentinels(self.source, self.target)

        from src.utils.data_migration import MigrationError

        def fails_archive_and_history_restore(path, destination):
            return (
                path == self.source
                and destination.name.startswith("atri_data.backup-")
            ) or (
                path.name == "private_1.json"
                and path.parent.parent.name.startswith(".qqbot_data.rollback-")
                and destination == self.target / "history" / "private_1.json"
            )

        with self.patch_replace_failure(fails_archive_and_history_restore):
            with self.assertRaises(MigrationError):
                self.migrate(self.source, self.target)

        rollback = list(self.root.glob(".qqbot_data.rollback-*"))
        self.assertEqual(1, len(rollback))
        self.assert_memory_sentinels_unchanged(self.source, self.target)
        self.assertTrue(self.source.exists())
        self.assertFalse((self.target / "history" / "private_1.json").exists())
        self.assertTrue(self.target.with_name(".qqbot_data.migration-state").exists())
        self.assertEqual(
            ["新消息"],
            [
                item["content"]
                for item in read_json(
                    rollback[0] / "history" / "private_1.json"
                )["messages"]
            ],
        )

    def test_rollback_cleanup_rename_failure_keeps_committed_history_and_marker(self):
        self.write_history_pair(self.source, self.target, had_target=True)
        self.write_memory_sentinels(self.source, self.target)

        from src.utils.data_migration import MigrationError

        with self.patch_replace_failure(
            lambda path, destination: path.name.startswith(".qqbot_data.rollback-")
            and destination.name.startswith(".qqbot_data.cleanup-")
        ):
            with self.assertRaises(MigrationError):
                self.migrate(self.source, self.target)

        backup = list(self.root.glob("atri_data.backup-*"))
        rollback = list(self.root.glob(".qqbot_data.rollback-*"))
        self.assertEqual(1, len(backup))
        self.assertEqual(1, len(rollback))
        self.assertFalse(self.source.exists())
        self.assertEqual(
            {"facts": ["源记忆"]},
            read_json(backup[0] / "memories" / "global.json"),
        )
        self.assertEqual(
            {"facts": ["目标记忆"]},
            read_json(self.target / "memories" / "private.json"),
        )
        self.assertEqual(
            ["旧消息", "新消息"],
            [
                item["content"]
                for item in read_json(
                    self.target / "history" / "private_1.json"
                )["messages"]
            ],
        )
        self.assertTrue(self.target.with_name(".qqbot_data.migration-state").exists())
        with self.assertRaises(MigrationError):
            self.migrate(self.source, self.target)

    def test_partial_commit_cleanup_keeps_committed_data_and_cleanup_artifact(self):
        self.write_history_pair(self.source, self.target, had_target=True)
        self.write_memory_sentinels(self.source, self.target)

        from src.utils import data_migration

        original_rmtree = data_migration.shutil.rmtree
        cleaned = []

        def partially_delete_cleanup(path, *args, **kwargs):
            path = Path(path)
            old_history = path / "history" / "private_1.json"
            if path.name.startswith(".qqbot_data.cleanup-") and old_history.exists():
                old_history.unlink()
                cleaned.append(path)
                raise OSError("partial post-commit cleanup failure")
            return original_rmtree(path, *args, **kwargs)

        with patch.object(
            data_migration.shutil,
            "rmtree",
            side_effect=partially_delete_cleanup,
        ):
            backup = self.migrate(self.source, self.target)

        self.assertTrue(backup.exists())
        self.assertFalse(self.source.exists())
        self.assertEqual(1, len(cleaned))
        self.assertEqual(
            {"facts": ["源记忆"]},
            read_json(backup / "memories" / "global.json"),
        )
        self.assertEqual(
            {"facts": ["目标记忆"]},
            read_json(self.target / "memories" / "private.json"),
        )
        self.assertEqual(
            ["旧消息", "新消息"],
            [
                item["content"]
                for item in read_json(
                    self.target / "history" / "private_1.json"
                )["messages"]
            ],
        )
        self.assertFalse(self.target.with_name(".qqbot_data.migration-state").exists())
        cleanup = list(self.root.glob(".qqbot_data.cleanup-*"))
        self.assertEqual(1, len(cleanup))
        self.assertFalse((cleanup[0] / "history" / "private_1.json").exists())

    def test_partial_failed_cleanup_after_recovery_does_not_keep_transaction_locked(self):
        self.write_history_pair(self.source, self.target, had_target=True)

        from src.utils import data_migration

        original_rmtree = data_migration.shutil.rmtree
        deleted_from = []

        def partial_delete_then_fail(path, *args, **kwargs):
            path = Path(path)
            redundant_file = path / "history" / "private_1.json"
            if not deleted_from and redundant_file.exists():
                redundant_file.unlink()
                deleted_from.append(path.name)
                raise OSError("injected partial failed cleanup failure")
            return original_rmtree(path, *args, **kwargs)

        with (
            self.patch_replace_failure(
                lambda path, destination: path == self.source
                and destination.name.startswith("atri_data.backup-")
            ),
            patch.object(data_migration.shutil, "rmtree", side_effect=partial_delete_then_fail),
        ):
            with self.assertRaises(data_migration.MigrationError):
                self.migrate(self.source, self.target)

        self.assertEqual(
            ["旧消息"],
            [item["content"] for item in read_json(self.source / "history" / "private_1.json")["messages"]],
        )
        self.assertEqual(
            ["新消息"],
            [
                item["content"]
                for item in read_json(self.target / "history" / "private_1.json")["messages"]
            ],
        )
        self.assertEqual(1, len(deleted_from))
        self.assertTrue(deleted_from[0].startswith(".qqbot_data.cleanup-"))
        self.assertFalse(self.target.with_name(".qqbot_data.migration-state").exists())
        cleanup = list(self.root.glob(".qqbot_data.cleanup-*"))
        self.assertEqual(1, len(cleanup))

        backup = self.migrate(self.source, self.target)

        self.assertIsNotNone(backup)
        self.assertEqual(
            ["旧消息", "新消息"],
            [
                item["content"]
                for item in read_json(self.target / "history" / "private_1.json")["messages"]
            ],
        )

    def test_failed_to_cleanup_rename_failure_keeps_recovery_data_and_blocks(self):
        self.write_history_pair(self.source, self.target, had_target=True)

        from src.utils.data_migration import MigrationError

        def fails_archive_and_failed_cleanup(path, destination):
            return (
                path == self.source and destination.name.startswith("atri_data.backup-")
            ) or (
                path.name.startswith(".qqbot_data.failed-")
                and destination.name.startswith(".qqbot_data.cleanup-")
            )

        with self.patch_replace_failure(fails_archive_and_failed_cleanup):
            with self.assertRaises(MigrationError):
                self.migrate(self.source, self.target)

        state = self.target.with_name(".qqbot_data.migration-state")
        failed = list(self.root.glob(".qqbot_data.failed-*"))
        self.assertTrue(state.exists())
        self.assertTrue(self.source.exists())
        self.assertEqual(1, len(failed))
        self.assertEqual(
            ["旧消息"],
            [item["content"] for item in read_json(self.source / "history" / "private_1.json")["messages"]],
        )
        self.assertEqual(
            ["旧消息", "新消息"],
            [item["content"] for item in read_json(failed[0] / "history" / "private_1.json")["messages"]],
        )
        self.assertEqual(
            ["新消息"],
            [
                item["content"]
                for item in read_json(self.target / "history" / "private_1.json")["messages"]
            ],
        )

        with self.assertRaises(MigrationError):
            self.migrate(self.source, self.target)

        self.assertTrue(state.exists())

    def test_existing_transaction_state_blocks_even_when_source_is_absent(self):
        state = self.target.with_name(".qqbot_data.migration-state")
        state.write_text("{}", encoding="utf-8")

        from src.utils.data_migration import MigrationError

        with self.assertRaises(MigrationError):
            self.migrate(self.source, self.target)

        self.assertTrue(state.exists())
        self.assertFalse(self.target.exists())

    def test_orphaned_recovery_material_blocks_even_without_state_or_source(self):
        from src.utils.data_migration import MigrationError

        for recovery_kind in ("migrating", "rollback", "failed"):
            with self.subTest(recovery_kind=recovery_kind):
                root = self.root / recovery_kind
                source = root / "atri_data"
                target = root / "qqbot_data"
                recovery = root / f".qqbot_data.{recovery_kind}-orphaned"
                recovery.mkdir(parents=True)

                with self.assertRaises(MigrationError):
                    self.migrate(source, target)

                self.assertTrue(recovery.exists())
                self.assertFalse(source.exists())
                self.assertFalse(target.exists())

    def test_atomic_transaction_state_blocks_a_second_caller(self):
        self.write_history_pair(self.source, self.target, had_target=True)

        from src.utils import data_migration

        entered = threading.Event()
        release = threading.Event()
        second_entered = threading.Event()
        call_lock = threading.Lock()
        call_count = 0
        original_merge = data_migration._merge_history

        def blocking_merge(source, target, staging, turns):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count
            if current_call == 1:
                entered.set()
                if not release.wait(timeout=2):
                    raise RuntimeError("test release timeout")
            else:
                second_entered.set()
            return original_merge(source, target, staging, turns)

        state = self.target.with_name(".qqbot_data.migration-state")
        with patch.object(data_migration, "_merge_history", side_effect=blocking_merge):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(self.migrate, self.source, self.target)
                self.assertTrue(entered.wait(timeout=2))
                second = executor.submit(self.migrate, self.source, self.target)
                try:
                    with self.assertRaises(data_migration.MigrationError):
                        second.result(timeout=2)
                    self.assertFalse(second_entered.is_set())
                    self.assertTrue(state.exists())
                finally:
                    release.set()
                self.assertIsNotNone(first.result(timeout=2))

        self.assertFalse(state.exists())

    def test_invalid_known_items_keep_source_and_original_target(self):
        cases = (
            ("missing-content", "history/private_1.json", {"messages": [{"role": "user"}]}),
            (
                "non-string-content",
                "history/private_1.json",
                {"messages": [{"role": "user", "content": 7}]},
            ),
            (
                "non-string-role",
                "history/private_1.json",
                {"messages": [{"role": 7, "content": "内容"}]},
            ),
        )

        from src.utils.data_migration import MigrationError, migrate_legacy_data

        for case_name, relative, invalid_data in cases:
            for invalid_side in ("source", "target"):
                with self.subTest(case=case_name, side=invalid_side):
                    root = self.root / f"{case_name}-{invalid_side}"
                    source = root / "atri_data"
                    target = root / "qqbot_data"
                    write_json(
                        source / "history" / "private_1.json",
                        {"messages": [{"role": "user", "content": "旧消息"}]},
                    )
                    write_json(
                        target / "history" / "private_1.json",
                        {"messages": [{"role": "assistant", "content": "新消息"}]},
                    )
                    (source / "source-marker.txt").write_text("源", encoding="utf-8")
                    (target / "target-marker.txt").write_text("目标", encoding="utf-8")
                    invalid_root = source if invalid_side == "source" else target
                    write_json(invalid_root / relative, invalid_data)

                    with self.assertRaises(MigrationError):
                        migrate_legacy_data(source, target, 8, timestamp="fixed")

                    self.assertEqual("源", (source / "source-marker.txt").read_text(encoding="utf-8"))
                    self.assertEqual("目标", (target / "target-marker.txt").read_text(encoding="utf-8"))
                    self.assertEqual(invalid_data, read_json(invalid_root / relative))


class StartupMigrationTests(unittest.TestCase):
    def test_startup_does_not_open_copy_or_import_legacy_memory_sentinels(self):
        from src import main
        from src.memory.service import MemoryService
        from src.memory.store import MemoryStore

        previous_initialized = main._startup_initialized
        self.addCleanup(setattr, main, "_startup_initialized", previous_initialized)
        main._startup_initialized = False

        with tempfile.TemporaryDirectory() as root:
            base_dir = Path(root)
            source_sentinel = base_dir / "atri_data" / "memories" / "global.json"
            target_sentinel = base_dir / "qqbot_data" / "memories" / "private.json"
            write_json(source_sentinel, {"facts": ["源不得导入"]})
            write_json(target_sentinel, {"facts": ["目标不得导入"]})
            write_json(
                base_dir / "atri_data" / "history" / "private_1.json",
                {"messages": [{"role": "user", "content": "保留历史"}]},
            )
            write_json(
                base_dir / "qqbot_data" / "history" / "private_1.json",
                {"messages": [{"role": "assistant", "content": "保留目标历史"}]},
            )
            fake_config = SimpleNamespace(
                data_dir=base_dir / "qqbot_data",
                history_turns=8,
            )
            store = MemoryStore(fake_config.data_dir / "memory.sqlite3")
            service = MemoryService(store=store)

            from src.utils import data_migration

            try:
                with (
                    patch.object(main, "BASE_DIR", base_dir),
                    patch.object(main, "config", fake_config),
                    patch.object(main, "get_persona"),
                    patch.object(main, "get_memory_service", return_value=service),
                    legacy_memory_io_guards(
                        data_migration,
                        base_dir / "atri_data",
                        fake_config.data_dir,
                    ),
                ):
                    main.startup()
            finally:
                service.stop()

            self.assertEqual({"facts": ["目标不得导入"]}, read_json(target_sentinel))
            backup = next(base_dir.glob("atri_data.backup-*"))
            self.assertEqual(
                {"facts": ["源不得导入"]},
                read_json(backup / "memories" / "global.json"),
            )
            connection = sqlite3.connect(store.path)
            try:
                self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM memory_claims").fetchone()[0])
                self.assertEqual("ok", connection.execute("PRAGMA integrity_check").fetchone()[0])
            finally:
                connection.close()
            self.assertEqual(
                ["保留历史", "保留目标历史"],
                [
                    item["content"]
                    for item in read_json(
                        fake_config.data_dir / "history" / "private_1.json"
                    )["messages"]
                ],
            )

    def test_startup_serializes_concurrent_initialization(self):
        from src import main

        previous_initialized = main._startup_initialized
        self.addCleanup(setattr, main, "_startup_initialized", previous_initialized)
        main._startup_initialized = False
        fake_config = SimpleNamespace(
            data_dir=main.BASE_DIR / "qqbot_data",
            history_turns=8,
        )
        start = threading.Barrier(3)
        call_lock = threading.Lock()
        calls = {"directory": 0}
        errors = []

        def directory_migration(*args, **kwargs):
            with call_lock:
                calls["directory"] += 1
            time.sleep(0.05)

        def worker():
            start.wait()
            try:
                main.startup()
            except Exception as error:
                errors.append(error)

        with (
            patch.object(main, "config", fake_config),
            patch.object(main, "migrate_legacy_data", side_effect=directory_migration),
        ):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            start.wait()
            for thread in threads:
                thread.join(timeout=2)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual([], errors)
        self.assertEqual({"directory": 1}, calls)

    def test_startup_migrates_default_data_once_before_starting_memory_service(self):
        from src import main

        order = []
        previous_initialized = main._startup_initialized
        self.addCleanup(setattr, main, "_startup_initialized", previous_initialized)
        main._startup_initialized = False
        fake_config = SimpleNamespace(
            data_dir=main.BASE_DIR / "qqbot_data",
            history_turns=8,
        )
        with (
            patch.object(main, "config", fake_config),
            patch.object(main, "migrate_legacy_data", side_effect=lambda *args, **kwargs: order.append("directory")),
        ):
            main.startup()
            main.startup()

        self.assertEqual(["directory"], order)

    def test_startup_does_not_migrate_atri_data_into_custom_data_directory(self):
        from src import main

        previous_initialized = main._startup_initialized
        self.addCleanup(setattr, main, "_startup_initialized", previous_initialized)
        main._startup_initialized = False
        fake_config = SimpleNamespace(
            data_dir=main.BASE_DIR / "custom_data",
            history_turns=8,
        )
        with (
            patch.object(main, "config", fake_config),
            patch.object(main, "migrate_legacy_data") as migrate,
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
