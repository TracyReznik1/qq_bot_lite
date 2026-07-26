from dataclasses import dataclass
from typing import Callable

from src.config import config
from src.memory.extractor import MemoryExtractor
from src.memory.models import CandidateClaim, MemoryClaim, MemoryContext, MemoryEvent
from src.memory.policy import MemoryPolicy
from src.memory.retriever import MemoryRetriever
from src.memory.service import get_memory_service
from src.memory.store import MemoryStore
from src.persona import get_persona
from src.router import Route

from .help import help_text
from .reset import reset_context


@dataclass(frozen=True)
class CommandContext:
    uid: str
    session_key: str
    raw_message: str
    memory_context: MemoryContext | None = None
    message_id: str = ""
    is_admin: bool = False

    def __post_init__(self) -> None:
        if self.memory_context is None:
            is_group = self.session_key.startswith("group:")
            group_id = None
            if is_group:
                parts = self.session_key.split(":")
                if len(parts) >= 2:
                    group_id = parts[1]
            object.__setattr__(
                self,
                "memory_context",
                MemoryContext(
                    user_id=self.uid,
                    session_key=self.session_key,
                    is_group=is_group,
                    group_id=group_id,
                ),
            )
        if not self.is_admin and config.admin_qq_ids and str(self.uid) in config.admin_qq_ids:
            object.__setattr__(self, "is_admin", True)


@dataclass(frozen=True)
class CommandOutcome:
    code: str
    facts: tuple[str, ...]
    fallback_reply: str
    already_rendered: bool = False


@dataclass(frozen=True)
class CommandResult:
    handled: bool
    reply: str = ""
    outcome: CommandOutcome | None = None


CommandHandler = Callable[[str, CommandContext, MemoryStore], CommandResult]


def render_command_outcome(outcome: CommandOutcome) -> str:
    if outcome.already_rendered:
        return outcome.fallback_reply
    return outcome.fallback_reply


def _help_command(_query: str, _context: CommandContext, _store: MemoryStore) -> CommandResult:
    reply = help_text()
    outcome = CommandOutcome(code="help", facts=(), fallback_reply=reply, already_rendered=True)
    return CommandResult(handled=True, reply=reply, outcome=outcome)


def _reset_command(_query: str, context: CommandContext, _store: MemoryStore) -> CommandResult:
    reply = reset_context(context.session_key)
    outcome = CommandOutcome(code="reset", facts=(), fallback_reply=reply, already_rendered=True)
    return CommandResult(handled=True, reply=reply, outcome=outcome)


def _remember_command(query: str, context: CommandContext, store: MemoryStore) -> CommandResult:
    text = query.strip()
    if not text:
        reply = "想让我记住什么？比如：/remember 我喜欢简洁回答"
        outcome = CommandOutcome(code="missing_query", facts=(), fallback_reply=reply, already_rendered=True)
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    extractor = MemoryExtractor()
    candidates = extractor.extract(text=text)
    if not candidates:
        candidates = [
            CandidateClaim(
                subject_ref="speaker",
                predicate="remembered_text",
                value=text,
                memory_type="preference",
                modality="asserted",
                confidence="high",
            )
        ]

    event = MemoryEvent(
        context=context.memory_context,
        text=text,
        message_id=context.message_id or "cmd",
        sequence=0,
    )
    policy = MemoryPolicy(store)
    policy.apply(event, candidates)

    reply = "记住了。"
    outcome = CommandOutcome(code="remembered", facts=(text,), fallback_reply=reply)
    return CommandResult(handled=True, reply=render_command_outcome(outcome), outcome=outcome)


def _global_remember_command(query: str, context: CommandContext, store: MemoryStore) -> CommandResult:
    if not context.is_admin:
        reply = "全局记忆只能由管理员写入。"
        outcome = CommandOutcome(code="forbidden", facts=(), fallback_reply=reply, already_rendered=True)
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    text = query.strip()
    if not text:
        reply = "想让我全局记住什么？比如：/globalremember 所有人都喜欢热茶"
        outcome = CommandOutcome(code="missing_query", facts=(), fallback_reply=reply, already_rendered=True)
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    extractor = MemoryExtractor()
    candidates = extractor.extract(text=text)
    if not candidates:
        candidates = [
            CandidateClaim(
                subject_ref="speaker",
                predicate="global_fact",
                value=text,
                memory_type="fact",
                modality="asserted",
                confidence="high",
            )
        ]

    global_ctx = MemoryContext(
        user_id=context.uid,
        session_key="global:global",
        is_group=False,
        group_id=None,
    )
    event = MemoryEvent(
        context=global_ctx,
        text=text,
        message_id=context.message_id or "cmd",
        sequence=0,
    )
    policy = MemoryPolicy(store)
    policy.apply(event, candidates)

    reply = "全局记忆已保存。"
    outcome = CommandOutcome(code="remembered_global", facts=(text,), fallback_reply=reply)
    return CommandResult(handled=True, reply=render_command_outcome(outcome), outcome=outcome)


def _memories_command(query: str, context: CommandContext, store: MemoryStore) -> CommandResult:
    retriever = MemoryRetriever(store)
    results = retriever.retrieve(context.memory_context, query=query.strip())
    if not results:
        reply = "没有找到相关记忆。"
        outcome = CommandOutcome(code="memories_empty", facts=(), fallback_reply=reply, already_rendered=True)
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    lines = ["【允许使用的记忆列表】"]
    facts = []
    for i, item in enumerate(results, 1):
        lines.append(f"{i}. [{item.claim.memory_type}] {item.claim.predicate}: {item.claim.value} (ID: {item.claim.id})")
        facts.append(str(item.claim.id))

    reply = "\n".join(lines)
    outcome = CommandOutcome(code="memories_list", facts=tuple(facts), fallback_reply=reply, already_rendered=True)
    return CommandResult(handled=True, reply=reply, outcome=outcome)


def _can_forget(actor: str, claim: MemoryClaim, is_admin: bool) -> bool:
    if is_admin:
        return True
    if claim.scope_type == "private":
        return claim.scope_id == actor
    return claim.speaker_qq == actor


def _forget_command(query: str, context: CommandContext, store: MemoryStore) -> CommandResult:
    target = query.strip()
    if not target:
        reply = "请指定要忘记的记忆 ID 或内容，例如：/forget 12 或 /forget 喜爱喝茶"
        outcome = CommandOutcome(code="missing_target", facts=(), fallback_reply=reply, already_rendered=True)
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    retriever = MemoryRetriever(store)
    all_items = retriever.retrieve(context.memory_context, query="")

    matched_claims: list[MemoryClaim] = []
    if target.isdigit():
        target_id = int(target)
        matched_claims = [item.claim for item in all_items if item.claim.id == target_id]
    else:
        low_target = target.lower()
        matched_claims = [
            item.claim for item in all_items
            if low_target in item.claim.value.lower() or low_target in item.claim.predicate.lower()
        ]

    if not matched_claims:
        reply = "未找到匹配的记忆。"
        outcome = CommandOutcome(code="not_found", facts=(target,), fallback_reply=reply, already_rendered=True)
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    if len(matched_claims) > 1:
        reply = "找到多个相关记忆，请提供更具体的描述或 ID。"
        outcome = CommandOutcome(
            code="ambiguous",
            facts=tuple(str(c.id) for c in matched_claims),
            fallback_reply=reply,
            already_rendered=True,
        )
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    target_claim = matched_claims[0]
    if not _can_forget(context.uid, target_claim, context.is_admin):
        reply = "权限不足：你只能删除你自己发表的记忆。"
        outcome = CommandOutcome(code="forbidden", facts=(str(target_claim.id),), fallback_reply=reply, already_rendered=True)
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    store.delete_claim_physically(target_claim.id, reason="user_requested_forget")
    reply = f"已删除记忆 (ID: {target_claim.id})。"
    outcome = CommandOutcome(code="forgot", facts=(str(target_claim.id), target_claim.value), fallback_reply=reply)
    return CommandResult(handled=True, reply=render_command_outcome(outcome), outcome=outcome)


def _search_command(query: str, context: CommandContext, _store: MemoryStore) -> CommandResult:
    from .search import search_reply
    reply = search_reply(query, context.session_key, context.raw_message)
    outcome = CommandOutcome(code="search", facts=(), fallback_reply=reply, already_rendered=True)
    return CommandResult(handled=True, reply=reply, outcome=outcome)


COMMANDS: dict[str, CommandHandler] = {
    "help": _help_command,
    "h": _help_command,
    "search": _search_command,
    "s": _search_command,
    "remember": _remember_command,
    "memo": _remember_command,
    "globalremember": _global_remember_command,
    "gremember": _global_remember_command,
    "memories": _memories_command,
    "forget": _forget_command,
    "reset": _reset_command,
}


def handle_command(route: Route, context: CommandContext, store: MemoryStore | None = None) -> CommandResult:
    if store is None:
        store = get_memory_service().store
    command_handler = COMMANDS.get(route.command)
    if command_handler is None:
        command_text = f"/{route.command}" if route.command else "/"
        reply = (
            f"暂不支持这个命令：{command_text}。"
            "qqbot_lite 只提供 /search、/help、/reset、/remember、/globalremember、/memories、/forget。"
        )
        return CommandResult(handled=True, reply=reply)

    return command_handler(route.query, context, store)
