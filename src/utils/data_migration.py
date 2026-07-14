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
