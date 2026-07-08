import logging
import time
from pathlib import Path
from threading import Lock

from src.config import config
from src.utils.storage import read_json, safe_id, write_json


logger = logging.getLogger("qq-bot")

GLOBAL_MEMORY_KEY = "global"
PERSONAL_MEMORY_PREFIX = "user:"
MEMORY_DIR = config.data_dir / "memories"
for directory in [config.data_dir, MEMORY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

memory_lock = Lock()


def memory_path(memory_key: str) -> Path:
    return MEMORY_DIR / f"{safe_id(memory_key)}.json"


def _get_memory_unlocked(memory_key: str) -> dict[str, list[str]]:
    data = read_json(memory_path(memory_key), {"facts": []})
    facts = data.get("facts", []) if isinstance(data, dict) else []
    facts = [str(item).strip() for item in facts if str(item).strip()]
    return {"facts": facts[-config.memory_limit :]}


def get_memory(memory_key: str) -> dict[str, list[str]]:
    with memory_lock:
        return _get_memory_unlocked(memory_key)


def add_memory(memory_key: str, fact: str) -> None:
    fact = fact.strip()
    if not fact:
        return
    with memory_lock:
        memory = _get_memory_unlocked(memory_key)
        facts = [item for item in memory["facts"] if item != fact]
        facts.append(fact)
        write_json(memory_path(memory_key), {"facts": facts[-config.memory_limit :]})


def get_global_memory() -> dict[str, list[str]]:
    return get_memory(GLOBAL_MEMORY_KEY)


def add_global_memory(fact: str) -> None:
    add_memory(GLOBAL_MEMORY_KEY, fact)


def personal_memory_key(uid: str) -> str:
    return f"{PERSONAL_MEMORY_PREFIX}{safe_id(uid)}"


def session_uid(session_key: str) -> str:
    parts = str(session_key or "").split(":")
    if len(parts) >= 2 and parts[0] == "private":
        return parts[1]
    if len(parts) >= 3 and parts[0] == "group":
        return parts[-1]
    return str(session_key or "")


def get_personal_memory(uid: str) -> dict[str, list[str]]:
    return get_memory(personal_memory_key(uid))


def add_personal_memory(uid: str, fact: str) -> None:
    add_memory(personal_memory_key(uid), fact)


def reset_session_memory(session_key: str) -> None:
    path = memory_path(session_key)
    with memory_lock:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to reset session memory file: %s", path)


def migrate_legacy_memory_files() -> None:
    legacy_dir = config.data_dir / "legacy_memories"
    global_stem = safe_id(GLOBAL_MEMORY_KEY)
    for path in MEMORY_DIR.glob("*.json"):
        if path.stem == global_stem or path.stem.startswith(("private_", "group_", "user_")):
            continue

        legacy_memory = read_json(path, {"facts": []})
        legacy_facts = legacy_memory.get("facts", []) if isinstance(legacy_memory, dict) else []
        legacy_facts = [str(item).strip() for item in legacy_facts if str(item).strip()]
        target_path = memory_path(personal_memory_key(path.stem))

        if legacy_facts:
            current_memory = read_json(target_path, {"facts": []})
            current_facts = current_memory.get("facts", []) if isinstance(current_memory, dict) else []
            merged_facts = [str(item).strip() for item in current_facts if str(item).strip()]
            for fact in legacy_facts:
                if fact not in merged_facts:
                    merged_facts.append(fact)
            write_json(target_path, {"facts": merged_facts[-config.memory_limit :]})

        legacy_dir.mkdir(parents=True, exist_ok=True)
        archive_path = legacy_dir / path.name
        if archive_path.exists():
            archive_path = legacy_dir / f"{path.stem}_{int(time.time())}.json"
        path.replace(archive_path)
        logger.info("Archived legacy memory file %s to %s", path, archive_path)
