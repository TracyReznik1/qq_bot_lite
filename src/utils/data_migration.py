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


def _merge_history(source: Path, target: Path, staging: Path, turns: int) -> None:
    names = {
        path.name for directory in (source / "history", target / "history")
        if directory.exists() for path in directory.glob("*.json")
    }
    limit = max(turns, 1) * 2
    for name in names:
        merged = _messages(source / "history" / name) + _messages(target / "history" / name)
        _write_object(staging / "history" / name, {"messages": merged[-limit:]})


def _validate_history_json(staging: Path) -> None:
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


def _best_effort_cleanup(path: Path) -> None:
    try:
        _remove_tree(path)
    except OSError:
        logger.warning("Migration cleanup remains at %s", path)


def _restore_history(
    *,
    target: Path,
    rollback: Path,
    failed: Path,
    names: list[str],
    installed: list[str],
    created_target: bool,
    created_history: bool,
) -> list[Exception]:
    errors: list[Exception] = []
    history = target / "history"

    for name in reversed(installed):
        active = history / name
        if not active.exists():
            continue
        try:
            destination = failed / "history" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            active.replace(destination)
        except Exception as error:
            errors.append(error)

    for name in names:
        original = rollback / "history" / name
        if not original.exists():
            continue
        try:
            history.mkdir(parents=True, exist_ok=True)
            original.replace(history / name)
        except Exception as error:
            errors.append(error)

    if not errors:
        try:
            _remove_tree(rollback)
        except Exception as error:
            errors.append(error)

    if created_history and history.exists():
        try:
            history.rmdir()
        except OSError as error:
            errors.append(error)
    if created_target and target.exists():
        try:
            target.rmdir()
        except OSError as error:
            errors.append(error)
    return errors


def _rollback_transaction(
    *,
    state: Path,
    source: Path,
    backup: Path,
    target: Path,
    staging: Path,
    rollback: Path,
    failed: Path,
    cleanup: Path,
    names: list[str],
    installed: list[str],
    created_target: bool,
    created_history: bool,
) -> bool:
    recovery_errors: list[Exception] = []

    if backup.exists() and not source.exists():
        try:
            backup.replace(source)
        except Exception as error:
            recovery_errors.append(error)

    recovery_errors.extend(
        _restore_history(
            target=target,
            rollback=rollback,
            failed=failed,
            names=names,
            installed=installed,
            created_target=created_target,
            created_history=created_history,
        )
    )

    try:
        _remove_tree(staging)
    except Exception as error:
        recovery_errors.append(error)

    recovered = (
        not recovery_errors
        and source.exists()
        and not backup.exists()
        and not staging.exists()
        and not rollback.exists()
    )
    if recovered:
        if failed.exists():
            try:
                failed.replace(cleanup)
            except Exception as error:
                recovery_errors.append(error)
                recovered = False
        if not recovered:
            for error in recovery_errors:
                logger.error("Data migration recovery step failed: %s", error)
            return False
        try:
            state.unlink()
        except Exception as error:
            recovery_errors.append(error)
            recovered = False
        else:
            _best_effort_cleanup(cleanup)

    for error in recovery_errors:
        logger.error("Data migration recovery step failed: %s", error)
    return recovered


def migrate_legacy_data(
    source_dir: Path,
    target_dir: Path,
    history_turns: int,
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
    cleanup = target.with_name(f".{target.name}.cleanup-{token}")
    names = sorted(
        {
            path.name
            for directory in (source / "history", target / "history")
            if directory.exists()
            for path in directory.glob("*.json")
        }
    )
    created_target = not target.exists()
    created_history = not (target / "history").exists()
    installed: list[str] = []
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
            "cleanup": str(cleanup),
            "backup": str(backup),
            "history_names": names,
        },
    )

    try:
        staging.mkdir()
        _merge_history(source, target, staging, history_turns)
        _validate_history_json(staging)
        target_history = target / "history"
        if names:
            target_history.mkdir(parents=True, exist_ok=True)
        for name in names:
            active = target_history / name
            if active.exists():
                original = rollback / "history" / name
                original.parent.mkdir(parents=True, exist_ok=True)
                active.replace(original)
            (staging / "history" / name).replace(active)
            installed.append(name)
        _remove_tree(staging)
        source.replace(backup)
    except Exception as error:
        recovered = _rollback_transaction(
            state=state,
            source=source,
            backup=backup,
            target=target,
            staging=staging,
            rollback=rollback,
            failed=failed,
            cleanup=cleanup,
            names=names,
            installed=installed,
            created_target=created_target,
            created_history=created_history,
        )
        if not recovered:
            logger.error(
                "Data migration recovery incomplete; transaction remains blocked state=%s",
                state,
            )
        raise MigrationError("旧数据迁移失败，服务未启动。") from error

    if rollback.exists():
        try:
            rollback.replace(cleanup)
        except OSError as error:
            raise MigrationError("无法提交数据迁移清理状态，服务未启动。") from error

    if (
        source.exists()
        or not backup.exists()
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
    _best_effort_cleanup(cleanup)
    return backup
