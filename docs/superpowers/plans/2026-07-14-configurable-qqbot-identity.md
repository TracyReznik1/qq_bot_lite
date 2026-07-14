# Configurable qqbot Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ATRI as the product identity with a configurable `qqbot` identity, inject that identity into every model call, and safely merge legacy `atri_data` into `qqbot_data` without losing existing history or memory.

**Architecture:** Keep `BOT_NAME` and `BOT_PERSONA` as the only identity inputs and build the character section centrally in `build_system_prompt()`. Add a focused `data_migration` module that stages and validates legacy data before replacing the live directory, then wire it through the idempotent startup guard. Finish by updating callback compatibility, runtime branding, documentation, the launcher, `.env.example`, and only the approved lines in the ignored `.env`.

**Tech Stack:** Python 3.11+, standard library (`dataclasses`, `json`, `pathlib`, `shutil`, `tempfile`, `datetime`), Flask, `unittest`, `unittest.mock`, PowerShell, Git.

## Global Constraints

- Default identity is `BOT_NAME=qqbot` and `BOT_PERSONA=你是一个自然、友好、简洁、可靠的 QQ 聊天助手。`.
- `BOT_NAME` and `BOT_PERSONA` are the only identity configuration sources; changes take effect after restart.
- Remove the fixed “温柔、日系、治愈、偶尔玩梗” traits; `BOT_PERSONA` is the complete persona.
- Every `llm.chat()` call used to generate a user reply must receive the configured name and persona in the first system message.
- Preserve chat, automatic web search, `/search`, image understanding, history, `/remember`, `/globalremember`, `/help`, `/reset`, and all memory modules.
- Default data directory is `qqbot_data`; merge legacy `atri_data` before accepting messages and keep a timestamped backup.
- Merge conflicts in known JSON use old data first and new data second, then apply the configured retention limit.
- Use `X-QQBOT-Callback-Secret` as the new callback header while continuing to accept `X-ATRI-Callback-Secret` as an undocumented compatibility header.
- Runtime ATRI references are allowed only for legacy data migration and legacy callback-header compatibility.
- Do not add dependencies.
- Do not expose or rewrite `.env` secrets. Only add or replace `BOT_NAME`, `BOT_PERSONA`, and an explicit legacy `DATA_DIR=atri_data`; preserve every other line.
- `.env` remains ignored and must never be staged or committed.
- Every production behavior change follows RED → GREEN TDD with the failing output recorded before implementation.

---

## File Responsibility Map

- `src/config.py`: normalize identity environment values and select `qqbot_data` as the default data directory.
- `src/chat/prompt.py`: build the character prompt exclusively from configured name and persona.
- `src/chat/chat_service.py`: continue passing the same system message to every model call in a reply flow; no structural rewrite is expected.
- `src/commands/search.py`: remove the hard-coded ATRI search-failure instruction.
- `src/utils/data_migration.py`: stage, merge, validate, swap, roll back, and archive legacy data.
- `.gitignore`: ignore the new data directory, legacy backups, and migration recovery directories.
- `src/main.py`: run migration through the startup guard and support new plus legacy callback headers.
- `src/services/url_fetch_service.py`: expose the qqbot User-Agent constant used by page fetching.
- `tests/test_identity_configuration.py`: identity defaults, custom values, prompt content, and every-call coverage.
- `tests/test_data_migration.py`: data merge, retention, backup, idempotence, and rollback behavior.
- `tests/test_qqbot_branding.py`: callback compatibility, runtime branding, launcher, docs, and ATRI allowlist.
- `.env.example`, `.env`, `README.md`, `启动qqbot.bat`: operator-facing identity configuration and branding.

---

### Task 1: Make identity configuration authoritative in every model call

**Files:**
- Create: `tests/test_identity_configuration.py`
- Modify: `src/config.py:1-53`
- Modify: `src/chat/prompt.py:58-85`
- Modify: `src/commands/search.py:1-24`

**Interfaces:**
- Produces: `DEFAULT_BOT_NAME: str`, `DEFAULT_BOT_PERSONA: str`, `env_text(name: str, default: str) -> str`.
- Produces: `Config.bot_name` and `Config.bot_persona` as `default_factory` fields that normalize whitespace and blank values.
- Preserves: `build_system_prompt(memory_key: str, tool_context: str = "") -> str` and `generate_reply(...) -> str` public signatures.

- [ ] **Step 1: Write failing identity configuration tests**

Create `tests/test_identity_configuration.py` with tests for blank defaults and custom values:

```python
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.chat import chat_service
from src.chat.prompt import build_system_prompt
import src.commands.search as search_command
from src.config import Config, DEFAULT_BOT_NAME, DEFAULT_BOT_PERSONA
from src.services.llm_types import ChatResponse


class CapturingLlm:
    def __init__(self):
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        if len(self.messages) == 1:
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": "search_1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"测试关键词"}',
                        },
                    }
                ],
            )
        return ChatResponse(content="按身份整理后的回答")


class ReplyingLlm:
    def __init__(self):
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        return ChatResponse(content="按身份回答")


class IdentityConfigurationTests(unittest.TestCase):
    def test_blank_identity_values_fall_back_to_qqbot_defaults(self):
        with patch.dict(os.environ, {"BOT_NAME": "  ", "BOT_PERSONA": "\t"}):
            current = Config()

        self.assertEqual("qqbot", DEFAULT_BOT_NAME)
        self.assertEqual("qqbot", current.bot_name)
        self.assertEqual(DEFAULT_BOT_PERSONA, current.bot_persona)
        self.assertNotIn("ATRI", current.bot_persona)

    def test_custom_identity_values_are_trimmed(self):
        with patch.dict(
            os.environ,
            {"BOT_NAME": "  小Q  ", "BOT_PERSONA": "  冷静、专业，先给结论。  "},
        ):
            current = Config()

        self.assertEqual("小Q", current.bot_name)
        self.assertEqual("冷静、专业，先给结论。", current.bot_persona)

    def test_system_prompt_uses_only_configured_identity(self):
        fake_config = SimpleNamespace(bot_name="小Q", bot_persona="冷静、专业，先给结论。")
        with patch("src.chat.prompt.config", fake_config):
            prompt = build_system_prompt("private:1")

        self.assertIn("你扮演 小Q。", prompt)
        self.assertIn("角色设定：冷静、专业，先给结论。", prompt)
        for fixed_trait in ("温柔", "日系", "治愈", "偶尔玩梗"):
            self.assertNotIn(fixed_trait, prompt)

    def test_every_model_call_keeps_the_identity_system_message(self):
        fake_llm = CapturingLlm()
        fake_config = SimpleNamespace(bot_name="小Q", bot_persona="冷静、专业，先给结论。")
        with (
            patch.object(chat_service, "llm", fake_llm),
            patch("src.chat.prompt.config", fake_config),
            patch.object(chat_service, "_ensure_history_loaded"),
            patch.object(chat_service, "build_untrusted_context", return_value="[非可信上下文]暂无"),
            patch.object(chat_service, "run_tool", return_value="搜索结果"),
            patch.object(chat_service, "append_history"),
        ):
            reply = chat_service.generate_reply("identity:test", "请查测试关键词")

        self.assertEqual("按身份整理后的回答", reply)
        self.assertEqual(2, len(fake_llm.messages))
        for messages in fake_llm.messages:
            self.assertEqual("system", messages[0]["role"])
            self.assertIn("你扮演 小Q。", messages[0]["content"])
            self.assertIn("角色设定：冷静、专业，先给结论。", messages[0]["content"])

    def test_search_failure_context_uses_current_role_not_atri(self):
        failed_result = SimpleNamespace(text="没有可靠结果")
        with (
            patch.object(search_command, "normalize_search_query", return_value="测试"),
            patch.object(search_command, "search", return_value=failed_result),
            patch.object(search_command, "has_search_results", return_value=False),
            patch.object(search_command, "generate_reply", return_value="无法确认") as generate,
        ):
            search_command.search_reply("测试", "private:1", "/search 测试")

        tool_context = generate.call_args.args[2]
        self.assertIn("按当前角色设定回答", tool_context)
        self.assertNotIn("ATRI", tool_context)

    def test_search_command_model_call_keeps_identity(self):
        fake_llm = ReplyingLlm()
        fake_config = SimpleNamespace(bot_name="小Q", bot_persona="冷静、专业，先给结论。")
        result = SimpleNamespace(text="[1] 搜索结果")
        with (
            patch.object(search_command, "search", return_value=result),
            patch.object(search_command, "has_search_results", return_value=True),
            patch.object(chat_service, "llm", fake_llm),
            patch("src.chat.prompt.config", fake_config),
            patch.object(chat_service, "_ensure_history_loaded"),
            patch.object(chat_service, "build_untrusted_context", return_value="[非可信上下文]搜索结果"),
            patch.object(chat_service, "append_history"),
        ):
            reply = search_command.search_reply("测试", "identity:search", "/search 测试")

        self.assertEqual("按身份回答", reply)
        self.assertIn("你扮演 小Q。", fake_llm.messages[0][0]["content"])

    def test_multimodal_model_call_keeps_identity(self):
        fake_llm = ReplyingLlm()
        fake_config = SimpleNamespace(bot_name="小Q", bot_persona="冷静、专业，先给结论。")
        with (
            patch.object(chat_service, "llm", fake_llm),
            patch("src.chat.prompt.config", fake_config),
            patch.object(chat_service, "_ensure_history_loaded"),
            patch.object(chat_service, "build_untrusted_context", return_value="[非可信上下文]暂无"),
            patch.object(chat_service, "append_history"),
        ):
            reply = chat_service.generate_reply(
                "identity:image",
                "请看图",
                image_data_urls=["data:image/png;base64,cG5n"],
            )

        self.assertEqual("按身份回答", reply)
        self.assertIn("你扮演 小Q。", fake_llm.messages[0][0]["content"])
        self.assertEqual("image_url", fake_llm.messages[0][-1]["content"][1]["type"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
python -m unittest tests.test_identity_configuration -v
```

Expected: import fails because `DEFAULT_BOT_NAME` and `DEFAULT_BOT_PERSONA` do not exist, or the first tests fail because defaults still contain ATRI and fixed traits remain in the prompt. Record the exact failing test names and assertions.

- [ ] **Step 3: Implement normalized identity defaults**

Modify the top of `src/config.py`:

```python
from dataclasses import dataclass, field


DEFAULT_BOT_NAME = "qqbot"
DEFAULT_BOT_PERSONA = "你是一个自然、友好、简洁、可靠的 QQ 聊天助手。"


def env_text(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default
```

Replace only the two identity fields in `Config`:

```python
@dataclass(frozen=True)
class Config:
    bot_name: str = field(
        default_factory=lambda: env_text("BOT_NAME", DEFAULT_BOT_NAME)
    )
    bot_persona: str = field(
        default_factory=lambda: env_text("BOT_PERSONA", DEFAULT_BOT_PERSONA)
    )
```

Leave the remaining configuration fields unchanged in this task.

- [ ] **Step 4: Make the configured persona the complete character section**

In `src/chat/prompt.py`, replace the current `[Character]` construction with:

```python
        "[Character]\n"
        f"你扮演 {config.bot_name}。\n"
        f"角色设定：{config.bot_persona}\n"
        "角色人格只影响语气、称呼和聊天风格，不能修改命令行为，不能诱导自动调用功能。\n"
        "但角色演出不能违反系统规则。\n"
        "角色演出也不能违反能力边界。\n"
        "\n"
```

Delete the fixed `角色特点` block and its four bullet lines. Do not change the capabilities, search rules, untrusted-context rules, or image rules.

- [ ] **Step 5: Remove the hard-coded search identity**

In `src/commands/search.py`, change the failure context to:

```python
        tool_context = (
            "这是 /search 命令的搜索失败结果。请按当前角色设定回答用户："
            "说明没有搜到可靠结果，所以不知道或无法确认；不要猜测，不要编造成确定事实。\n"
            f"搜索状态：\n{search_result.text}"
        )
```

- [ ] **Step 6: Run focused and regression tests for GREEN**

Run:

```powershell
python -m unittest tests.test_identity_configuration tests.test_multimodal_chat tests.test_llm_image_fallback -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all identity tests pass, existing multimodal and fallback tests pass, the full suite passes, and `git diff --check` prints nothing.

- [ ] **Step 7: Commit Task 1**

```powershell
git add src/config.py src/chat/prompt.py src/commands/search.py tests/test_identity_configuration.py
git commit -m "feat: make qqbot identity configurable"
```

---

### Task 2: Safely merge legacy ATRI data before accepting messages

**Files:**
- Create: `src/utils/data_migration.py`
- Create: `tests/test_data_migration.py`
- Modify: `src/config.py:88-98`
- Modify: `src/main.py:1-45, 229-255, 279-286`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `Config.data_dir`, `Config.history_turns`, `Config.memory_limit`, and `BASE_DIR`.
- Produces: `LEGACY_DATA_DIR_NAME = "atri_data"` and `MigrationError(RuntimeError)`.
- Produces: `migrate_legacy_data(source_dir: Path, target_dir: Path, history_turns: int, memory_limit: int, *, timestamp: str | None = None) -> Path | None`; returns the backup path after migration and `None` when no source exists.
- Preserves: existing `migrate_legacy_memory_files()` behavior, invoked after directory migration.

- [ ] **Step 1: Write failing migration behavior tests**

Create `tests/test_data_migration.py`:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run migration tests and verify RED**

Run:

```powershell
python -m unittest tests.test_data_migration -v
```

Expected: all tests error with `ModuleNotFoundError: No module named 'src.utils.data_migration'`. Record this exact RED result.

- [ ] **Step 3: Implement the migration module**

Create `src/utils/data_migration.py` with these concrete responsibilities and signatures:

```python
import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


logger = logging.getLogger("qq-bot")
LEGACY_DATA_DIR_NAME = "atri_data"


class MigrationError(RuntimeError):
    pass


def _read_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _write_object(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _facts(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    values = _read_object(path).get("facts", [])
    if not isinstance(values, list):
        raise ValueError(f"facts must be a list: {path}")
    return [str(item).strip() for item in values if str(item).strip()]


def _messages(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    values = _read_object(path).get("messages", [])
    if not isinstance(values, list):
        raise ValueError(f"messages must be a list: {path}")
    return [
        item
        for item in values
        if isinstance(item, dict) and "role" in item and "content" in item
    ]


def _copy_unknown_files(source: Path, staging: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if len(relative.parts) == 2 and relative.parts[0] in {"memories", "history"} and path.suffix.lower() == ".json":
            continue
        destination = staging / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _merge_memories(source: Path, target: Path, staging: Path, limit: int) -> None:
    names = {
        path.name for directory in (source / "memories", target / "memories")
        if directory.exists() for path in directory.glob("*.json")
    }
    for name in names:
        merged: list[str] = []
        for fact in _facts(source / "memories" / name) + _facts(target / "memories" / name):
            if fact not in merged:
                merged.append(fact)
        _write_object(staging / "memories" / name, {"facts": merged[-max(limit, 1):]})


def _merge_history(source: Path, target: Path, staging: Path, turns: int) -> None:
    names = {
        path.name for directory in (source / "history", target / "history")
        if directory.exists() for path in directory.glob("*.json")
    }
    limit = max(turns, 1) * 2
    for name in names:
        merged = _messages(source / "history" / name) + _messages(target / "history" / name)
        _write_object(staging / "history" / name, {"messages": merged[-limit:]})


def _validate_known_json(staging: Path) -> None:
    memory_paths = (staging / "memories").glob("*.json") if (staging / "memories").exists() else ()
    for path in memory_paths:
        _facts(path)
    history_paths = (staging / "history").glob("*.json") if (staging / "history").exists() else ()
    for path in history_paths:
        _messages(path)


def _backup_path(source: Path, timestamp: str) -> Path:
    base = source.with_name(f"{source.name}.backup-{timestamp}")
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = source.with_name(f"{base.name}-{suffix}")
        suffix += 1
    return candidate


def _archive_source(source: Path, timestamp: str) -> Path:
    backup = _backup_path(source, timestamp)
    source.replace(backup)
    return backup
```

Add `migrate_legacy_data()` using this exact transaction order:

```python
def migrate_legacy_data(
    source_dir: Path,
    target_dir: Path,
    history_turns: int,
    memory_limit: int,
    *,
    timestamp: str | None = None,
) -> Path | None:
    source = source_dir.resolve()
    target = target_dir.resolve()
    if not source.exists():
        return None
    if source == target:
        raise MigrationError("旧数据目录和新数据目录不能相同。")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.migrating-", dir=target.parent))
    rollback = target.with_name(f".{target.name}.rollback-{uuid4().hex}")
    failed = target.with_name(f".{target.name}.failed-{uuid4().hex}")
    had_target = target.exists()
    moved_target = False
    installed_staging = False
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        if had_target:
            shutil.copytree(target, staging, dirs_exist_ok=True)
        _copy_unknown_files(source, staging)
        _merge_memories(source, target, staging, memory_limit)
        _merge_history(source, target, staging, history_turns)
        _validate_known_json(staging)

        if had_target:
            target.replace(rollback)
            moved_target = True
        staging.replace(target)
        installed_staging = True
        backup = _archive_source(source, stamp)

        if rollback.exists():
            try:
                shutil.rmtree(rollback)
            except OSError:
                logger.warning("Migration rollback copy remains at %s", rollback)
        logger.info(
            "Legacy data migrated source=%s target=%s backup=%s",
            source,
            target,
            backup,
        )
        return backup
    except Exception as error:
        try:
            if installed_staging and target.exists():
                target.replace(failed)
            if moved_target and rollback.exists():
                rollback.replace(target)
        except Exception:
            logger.exception("Failed to restore data directory target=%s", target)
        raise MigrationError("旧数据迁移失败，服务未启动。") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
```

Do not use `src.utils.storage.read_json()` here because it swallows malformed JSON; migration must fail closed and preserve recoverability.

- [ ] **Step 4: Run migration tests for GREEN**

Run:

```powershell
python -m unittest tests.test_data_migration -v
```

Expected: six migration tests pass. Inspect the temporary directories only through assertions; no real `atri_data` or `qqbot_data` is touched.

- [ ] **Step 5: Write failing startup-wiring tests**

Append a `StartupMigrationTests` class to `tests/test_data_migration.py`:

```python
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
```

Run:

```powershell
python -m unittest tests.test_data_migration.StartupMigrationTests -v
```

Expected: RED because `src.main` does not import or call `migrate_legacy_data`, and the callback does not invoke `startup()`.

- [ ] **Step 6: Wire default data selection and startup migration**

In `src/config.py`, define and use the new default directory:

```python
DEFAULT_DATA_DIR_NAME = "qqbot_data"
```

Replace the `data_dir` field with:

```python
    data_dir: Path = field(
        default_factory=lambda: resolve_path(
            os.getenv("DATA_DIR", ""),
            DEFAULT_DATA_DIR_NAME,
        )
    )
```

In `src/main.py`, import `BASE_DIR` and the migration function:

```python
from src.config import BASE_DIR, config
from src.utils.data_migration import LEGACY_DATA_DIR_NAME, migrate_legacy_data
```

Replace `startup()` with:

```python
def startup() -> None:
    global _startup_initialized
    if _startup_initialized:
        return

    default_data_dir = (BASE_DIR / "qqbot_data").resolve()
    if config.data_dir.resolve() == default_data_dir:
        migrate_legacy_data(
            BASE_DIR / LEGACY_DATA_DIR_NAME,
            config.data_dir,
            config.history_turns,
            config.memory_limit,
        )
    migrate_legacy_memory_files()
    _startup_initialized = True
```

Call `startup()` inside `onebot_event()` after callback authorization and before reading the event JSON:

```python
    if not is_callback_authorized():
        logger.warning("Rejected unauthorized OneBot callback")
        return {"status": "forbidden"}, 403

    startup()
    data = request.get_json(silent=True) or {}
```

This preserves the existing `run()` call and gives WSGI callback operation the same idempotent guard.

- [ ] **Step 7: Ignore live, backup, and recovery data directories**

Keep the existing `atri_data/` rule and add these entries to `.gitignore`:

```gitignore
qqbot_data/
atri_data.backup-*/
.qqbot_data.migrating-*/
.qqbot_data.rollback-*/
.qqbot_data.failed-*/
```

Verify the patterns without creating tracked files:

```powershell
git check-ignore -v qqbot_data/.probe atri_data.backup-fixed/.probe .qqbot_data.migrating-test/.probe .qqbot_data.rollback-test/.probe .qqbot_data.failed-test/.probe
```

Expected: every probe is matched by the corresponding `.gitignore` rule.

- [ ] **Step 8: Run Task 2 and full regression tests**

Run:

```powershell
python -m unittest tests.test_data_migration -v
python -m unittest tests.test_main_image_flow tests.test_product_scope -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: migration and startup tests pass; main-flow, product-scope, and full suites remain green; diff check is empty. Confirm `atri_data` in the real workspace has not been renamed during tests.

- [ ] **Step 9: Commit Task 2**

```powershell
git add .gitignore src/config.py src/main.py src/utils/data_migration.py tests/test_data_migration.py
git commit -m "feat: migrate legacy ATRI data safely"
```

---

### Task 3: Finish qqbot branding, callback compatibility, docs, launcher, and local `.env`

**Files:**
- Create: `tests/test_qqbot_branding.py`
- Create by rename: `启动qqbot.bat`
- Delete by rename: `启动ATRI.bat`
- Modify: `src/main.py:233-251`
- Modify: `src/services/url_fetch_service.py:1-240`
- Modify: `.env.example`
- Modify: `.env` (ignored; never stage)
- Modify: `README.md`
- Modify: `tests/test_user_facing_scope.py`

**Interfaces:**
- Produces: `CALLBACK_SECRET_HEADER = "X-QQBOT-Callback-Secret"` and `LEGACY_CALLBACK_SECRET_HEADER = "X-ATRI-Callback-Secret"`.
- Produces: `URL_FETCH_USER_AGENT = "qqbot-url-fetch/1.0"`.
- Preserves: `is_callback_authorized() -> bool`, `fetch_url(text: str) -> UrlFetchResult`, and existing OneBot behavior.

- [ ] **Step 1: Write failing branding and callback tests**

Create `tests/test_qqbot_branding.py`:

```python
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
```

Extend `tests/test_user_facing_scope.py` so README and help assertions reject ATRI and accept configurable qqbot wording. Add this import and module constant:

```python
from src.config import config


ROOT = Path(__file__).resolve().parents[1]
```

Add this method inside `UserFacingScopeTests`:

```python
    def test_readme_and_help_do_not_present_atri_as_identity(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        help_message = help_text()

        self.assertNotIn("ATRI", readme)
        self.assertNotIn("ATRI", help_message)
        self.assertIn(config.bot_name, help_message)
```

- [ ] **Step 2: Run branding tests and verify RED**

Run:

```powershell
python -m unittest tests.test_qqbot_branding tests.test_user_facing_scope -v
```

Expected failures: new callback header is rejected, URL-fetch constant does not exist, qqbot launcher is absent, ATRI launcher and README strings remain, and `.env.example` lacks identity keys.

- [ ] **Step 3: Implement callback-header compatibility without leaking secrets**

In `src/main.py`, define:

```python
CALLBACK_SECRET_HEADER = "X-QQBOT-Callback-Secret"
LEGACY_CALLBACK_SECRET_HEADER = "X-ATRI-Callback-Secret"
```

Replace `is_callback_authorized()` with:

```python
def is_callback_authorized() -> bool:
    secret = config.callback_secret.strip()
    if not secret:
        return True

    authorization = request.headers.get("Authorization", "").strip()
    callback_secret = request.headers.get(CALLBACK_SECRET_HEADER, "").strip()
    legacy_secret = request.headers.get(LEGACY_CALLBACK_SECRET_HEADER, "").strip()
    if hmac.compare_digest(authorization, f"Bearer {secret}") or hmac.compare_digest(
        callback_secret,
        secret,
    ):
        return True
    if legacy_secret and hmac.compare_digest(legacy_secret, secret):
        logger.warning("Legacy callback secret header accepted; update OneBot configuration")
        return True
    return False
```

Do not log either header value.

- [ ] **Step 4: Replace the runtime User-Agent**

In `src/services/url_fetch_service.py`, define near the other constants:

```python
URL_FETCH_USER_AGENT = "qqbot-url-fetch/1.0"
```

Replace the request header value with:

```python
"User-Agent": URL_FETCH_USER_AGENT,
```

- [ ] **Step 5: Rename and update the Windows launcher**

Verify the source and destination are both inside the repository, then rename:

```powershell
Resolve-Path -LiteralPath .\启动ATRI.bat
Move-Item -LiteralPath .\启动ATRI.bat -Destination .\启动qqbot.bat
```

Change the one remaining status line in `启动qqbot.bat` to:

```bat
    echo Done. Starting qqbot...
```

- [ ] **Step 6: Update `.env.example` and README**

Add these lines at the start of `.env.example`:

```env
BOT_NAME=qqbot
BOT_PERSONA=你是一个自然、友好、简洁、可靠的 QQ 聊天助手。
```

Update `README.md` with these exact content rules:

- Title: `# qqbot — qqbot_lite`.
- Quick start references `启动qqbot.bat`.
- Configuration example includes `BOT_NAME` and `BOT_PERSONA` with the values above.
- OneBot section says `qqbot` listens on port 5000.
- Group section says the default name is `qqbot`, users normally `@qqbot`, and `BOT_NAME` customizes it.
- Project tree lists `启动qqbot.bat`.
- Local data path is `qqbot_data/`.
- No README line contains `ATRI` or presents the legacy callback header.

- [ ] **Step 7: Make the approved minimal `.env` change**

Before editing, compute a hash of every non-target line without displaying values:

```powershell
$lines = Get-Content .env | Where-Object { $_ -notmatch '^(BOT_NAME|BOT_PERSONA|DATA_DIR)=' }
$before = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($lines -join "`n"))))
$before
```

The current `.env` has none of the three target keys. Use `apply_patch` to insert only these two lines at the start; do not add `DATA_DIR` because the code default already selects `qqbot_data`:

```env
BOT_NAME=qqbot
BOT_PERSONA=你是一个自然、友好、简洁、可靠的 QQ 聊天助手。
```

Recompute the same non-target hash and assert it equals `$before`. Then run:

```powershell
git status --short --ignored .env
git check-ignore -v .env
```

Expected: `.env` is ignored and does not appear as a staged or tracked change. Never print the full `.env`.

- [ ] **Step 8: Run focused tests and close branding gaps**

Run:

```powershell
python -m unittest tests.test_qqbot_branding tests.test_user_facing_scope -v
```

Expected: all branding tests pass. If the runtime ATRI allowlist reports a line outside `src/main.py` or `src/utils/data_migration.py`, remove that hard-coded identity unless it is explicitly added to the approved spec first.

- [ ] **Step 9: Run full verification**

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests
@'
import ast
from pathlib import Path
files = sorted(Path("src").rglob("*.py")) + sorted(Path("tests").rglob("*.py"))
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"AST_OK={len(files)}")
'@ | python -
git diff --check
git status --short
```

Expected: full tests pass, compile and AST checks exit zero, diff check prints nothing, and status includes only planned tracked files. `.env` must not appear.

Run targeted residual scans:

```powershell
rg -n "ATRI" src README.md .env.example 启动qqbot.bat
rg -n "BOT_NAME|BOT_PERSONA|qqbot_data|X-QQBOT-Callback-Secret|qqbot-url-fetch" src tests README.md .env.example 启动qqbot.bat
```

Expected: the first scan returns only `src/main.py` legacy-header compatibility and `src/utils/data_migration.py` legacy-directory migration. The second scan confirms the new identity, path, header, and User-Agent are wired and tested.

- [ ] **Step 10: Commit Task 3 without staging `.env`**

```powershell
git add -A -- src/main.py src/services/url_fetch_service.py .env.example README.md tests/test_qqbot_branding.py tests/test_user_facing_scope.py 启动ATRI.bat 启动qqbot.bat
git status --short
git commit -m "refactor: separate qqbot identity from ATRI"
```

Expected: Git records the launcher rename and planned source/docs/tests. `.env` remains absent from the index.

---

## Final Review Checklist

- [ ] Re-read `docs/superpowers/specs/2026-07-14-configurable-qqbot-identity-design.md` and map every acceptance criterion to a passing test or explicit file check.
- [ ] Confirm all three task commits contain only their planned scope.
- [ ] Run the full test, compile, AST, diff, and residual commands from Task 3 Step 9 again from a clean checkout.
- [ ] Verify the real `atri_data/` has not been migrated merely by tests or imports; migration should occur only when the application startup guard runs.
- [ ] Verify `.env` contains the two approved identity keys, its non-target-line hash is unchanged, and it is not tracked.
- [ ] Perform a final code review from the pre-feature base to HEAD, fix all Critical and Important findings, and repeat verification.
