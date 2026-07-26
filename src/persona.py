from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from src.config import config


NAME_PATTERN = re.compile(r"^\s*-\s*名字[：:]\s*(.+?)\s*$", re.MULTILINE)


class PersonaConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Persona:
    name: str
    content: str


def load_persona(path: Path) -> Persona:
    if not path.is_file():
        raise PersonaConfigurationError(f"角色文件不存在：{path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise PersonaConfigurationError(f"角色文件为空：{path}")
    match = NAME_PATTERN.search(content)
    if not match or not match.group(1).strip():
        raise PersonaConfigurationError("角色文件缺少“- 名字：...”")
    return Persona(name=match.group(1).strip(), content=content)


@lru_cache(maxsize=1)
def get_persona() -> Persona:
    return load_persona(config.persona_path)
