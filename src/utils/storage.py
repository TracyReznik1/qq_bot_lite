import json
import logging
import re
from pathlib import Path
from typing import Any


logger = logging.getLogger("qq-bot")


def safe_id(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z_-]", "_", str(value or "unknown"))


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read json: %s", path)
    return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
