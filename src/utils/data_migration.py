import json
import logging
import os
import shutil
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
    if any(not isinstance(item, str) for item in values):
        raise ValueError(f"facts must contain only strings: {path}")
    return [item.strip() for item in values if item.strip()]


def _messages(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    values = _read_object(path).get("messages", [])
    if not isinstance(values, list):
        raise ValueError(f"messages must be a list: {path}")
    for item in values:
        if not isinstance(item, dict):
            raise ValueError(f"messages must contain only objects: {path}")
        if not isinstance(item.get("role"), str) or not isinstance(item.get("content"), str):
            raise ValueError(f"message role and content must be strings: {path}")
    return values


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


def _transaction_state_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.migration-state")


def _has_recovery_material(target: Path) -> bool:
    return any(
        path
        for kind in ("migrating", "rollback", "failed")
        for path in target.parent.glob(f".{target.name}.{kind}-*")
    )


def _create_transaction_state(path: Path, data: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise MigrationError("检测到未完成的数据迁移，服务未启动。") from error
    except OSError as error:
        raise MigrationError("无法创建数据迁移事务标记，服务未启动。") from error


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _rollback_transaction(
    *,
    state: Path,
    source: Path,
    backup: Path,
    target: Path,
    staging: Path,
    rollback: Path,
    failed: Path,
    had_target: bool,
    installed_staging: bool,
) -> bool:
    recovery_errors: list[Exception] = []

    if backup.exists() and not source.exists():
        try:
            backup.replace(source)
        except Exception as error:
            recovery_errors.append(error)

    if installed_staging and target.exists():
        try:
            target.replace(failed)
        except Exception as error:
            recovery_errors.append(error)

    if had_target and rollback.exists():
        try:
            rollback.replace(target)
        except Exception as error:
            recovery_errors.append(error)

    try:
        _remove_tree(staging)
    except Exception as error:
        recovery_errors.append(error)

    original_target_restored = (
        target.exists() and not rollback.exists()
        if had_target
        else not target.exists() and not rollback.exists()
    )
    if source.exists() and original_target_restored:
        try:
            _remove_tree(failed)
        except Exception as error:
            recovery_errors.append(error)

    recovered = (
        source.exists()
        and not backup.exists()
        and original_target_restored
        and not staging.exists()
        and not rollback.exists()
        and not failed.exists()
    )
    if recovered:
        try:
            state.unlink()
        except Exception as error:
            recovery_errors.append(error)
            recovered = False

    for error in recovery_errors:
        logger.error("Data migration recovery step failed: %s", error)
    return recovered


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
    if source == target:
        raise MigrationError("旧数据目录和新数据目录不能相同。")

    target.parent.mkdir(parents=True, exist_ok=True)
    state = _transaction_state_path(target)
    if state.exists() or _has_recovery_material(target):
        raise MigrationError("检测到未完成的数据迁移，服务未启动。")
    if not source.exists():
        if state.exists() or _has_recovery_material(target):
            raise MigrationError("检测到未完成的数据迁移，服务未启动。")
        return None

    token = uuid4().hex
    staging = target.with_name(f".{target.name}.migrating-{token}")
    rollback = target.with_name(f".{target.name}.rollback-{token}")
    failed = target.with_name(f".{target.name}.failed-{token}")
    had_target = target.exists()
    installed_staging = False
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = _backup_path(source, stamp)

    _create_transaction_state(
        state,
        {
            "source": str(source),
            "target": str(target),
            "staging": str(staging),
            "rollback": str(rollback),
            "failed": str(failed),
            "backup": str(backup),
            "had_target": had_target,
        },
    )

    try:
        staging.mkdir()
        if had_target:
            shutil.copytree(target, staging, dirs_exist_ok=True)
        _validate_known_json(source)
        _validate_known_json(target)
        _copy_unknown_files(source, staging)
        _merge_memories(source, target, staging, memory_limit)
        _merge_history(source, target, staging, history_turns)
        _validate_known_json(staging)

        if had_target:
            target.replace(rollback)
        staging.replace(target)
        installed_staging = True
        source.replace(backup)

        _remove_tree(rollback)
    except Exception as error:
        recovered = _rollback_transaction(
            state=state,
            source=source,
            backup=backup,
            target=target,
            staging=staging,
            rollback=rollback,
            failed=failed,
            had_target=had_target,
            installed_staging=installed_staging,
        )
        if not recovered:
            logger.error(
                "Data migration recovery incomplete; transaction remains blocked state=%s",
                state,
            )
        raise MigrationError("旧数据迁移失败，服务未启动。") from error

    if (
        source.exists()
        or not backup.exists()
        or not target.exists()
        or staging.exists()
        or rollback.exists()
        or failed.exists()
    ):
        raise MigrationError("数据迁移提交状态异常，服务未启动。")
    try:
        state.unlink()
    except OSError as error:
        raise MigrationError("无法清除数据迁移事务标记，服务未启动。") from error

    logger.info(
        "Legacy data migrated source=%s target=%s backup=%s",
        source,
        target,
        backup,
    )
    return backup
