from dataclasses import dataclass
from typing import Callable

from src.chat.memory import add_global_memory, add_personal_memory
from src.config import config
from src.router import Route

from .help import help_text
from .reset import reset_context
from .search import search_reply


@dataclass(frozen=True)
class CommandContext:
    uid: str
    session_key: str
    raw_message: str


@dataclass(frozen=True)
class CommandResult:
    handled: bool
    reply: str = ""


CommandHandler = Callable[[str, CommandContext], CommandResult]


def _text_result(reply: str) -> CommandResult:
    return CommandResult(handled=True, reply=reply)


def _help_command(_query: str, _context: CommandContext) -> CommandResult:
    return _text_result(help_text())


def _reset_command(_query: str, context: CommandContext) -> CommandResult:
    return _text_result(reset_context(context.session_key))


def _remember_command(query: str, context: CommandContext) -> CommandResult:
    memory = query.strip()
    if not memory:
        return _text_result("想让我记住什么？比如：/remember 我喜欢简洁回答")

    add_personal_memory(context.uid, memory)
    return _text_result("记住了。")


def _global_remember_command(query: str, context: CommandContext) -> CommandResult:
    if str(context.uid) not in config.admin_qq_ids:
        return _text_result("全局记忆只能由管理员写入。")

    memory = query.strip()
    if not memory:
        return _text_result("想让我全局记住什么？比如：/globalremember 大家都喜欢热茶")
    add_global_memory(memory)
    return _text_result("全局记忆已保存。")


def _search_command(query: str, context: CommandContext) -> CommandResult:
    return _text_result(search_reply(query, context.session_key, context.raw_message))


COMMANDS: dict[str, CommandHandler] = {
    "help": _help_command,
    "h": _help_command,
    "search": _search_command,
    "s": _search_command,
    "remember": _remember_command,
    "memo": _remember_command,
    "globalremember": _global_remember_command,
    "gremember": _global_remember_command,
    "reset": _reset_command,
}


def handle_command(route: Route, context: CommandContext) -> CommandResult:
    command_handler = COMMANDS.get(route.command)
    if command_handler is None:
        command_text = f"/{route.command}" if route.command else "/"
        return _text_result(
            f"暂不支持这个命令：{command_text}。"
            "qqbot_lite 只提供 /search、/help、/reset、/remember、/globalremember。"
        )

    return command_handler(route.query, context)
