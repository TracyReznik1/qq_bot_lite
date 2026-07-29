import sqlite3
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
from .renderer import PersonaCommandRenderer, TrustedCommandFacts
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
    status: str = "ok"
    scope: str = "none"
    cause: str = "completed"


@dataclass(frozen=True)
class CommandResult:
    handled: bool
    reply: str = ""
    outcome: CommandOutcome | None = None


CommandHandler = Callable[[str, CommandContext, MemoryStore], CommandResult]


def render_command_outcome(
    outcome: CommandOutcome,
    renderer: PersonaCommandRenderer | None = None,
) -> str:
    if outcome.already_rendered:
        return outcome.fallback_reply
    trusted_facts = TrustedCommandFacts(
        code=outcome.code,
        status=outcome.status,
        scope=outcome.scope,
        cause=outcome.cause,
        details=outcome.facts,
    )
    return (renderer or PersonaCommandRenderer()).render(
        trusted_facts,
        outcome.fallback_reply,
    )


def _help_command(_query: str, _context: CommandContext, _store: MemoryStore) -> CommandResult:
    reply = help_text()
    outcome = CommandOutcome(
        code="help",
        facts=(),
        fallback_reply=reply,
        status="shown",
        cause="user_requested",
    )
    return CommandResult(handled=True, reply=reply, outcome=outcome)


def _reset_command(_query: str, context: CommandContext, _store: MemoryStore) -> CommandResult:
    reply = reset_context(context.session_key)
    outcome = CommandOutcome(
        code="reset",
        facts=(),
        fallback_reply=reply,
        status="cleared",
        scope=context.session_key,
        cause="user_requested",
    )
    return CommandResult(handled=True, reply=reply, outcome=outcome)


def _remember_command(query: str, context: CommandContext, store: MemoryStore) -> CommandResult:
    text = query.strip()
    if not text:
        reply = "想让我记住什么？比如：/remember 我喜欢简洁回答"
        outcome = CommandOutcome(
            code="missing_query",
            facts=(),
            fallback_reply=reply,
            status="rejected",
            scope=_context_scope(context),
            cause="missing_query",
        )
        return CommandResult(handled=True, reply=reply, outcome=outcome)
    if not context.message_id:
        reply = (
            f"未写入记忆：scope={_context_scope(context)}；"
            "status=rejected；cause=missing_message_id。"
        )
        outcome = CommandOutcome(
            code="remember_rejected",
            facts=(text,),
            fallback_reply=reply,
            status="rejected",
            scope=_context_scope(context),
            cause="missing_message_id",
        )
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    event = MemoryEvent(
        context=context.memory_context,
        text=text,
        message_id=context.message_id,
        sequence=0,
    )
    extractor = MemoryExtractor()
    candidates = extractor.extract(event)
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

    policy = MemoryPolicy(store)
    decisions = policy.apply_command(event, tuple(candidates))
    scope = _context_scope(context)
    if not decisions:
        reply = (
            f"未写入记忆：scope={scope}；"
            "status=rejected；cause=policy_rejected。"
        )
        outcome = CommandOutcome(
            code="remember_rejected",
            facts=(text,),
            fallback_reply=reply,
            status="rejected",
            scope=scope,
            cause="policy_rejected",
        )
        return CommandResult(
            handled=True,
            reply=outcome.fallback_reply,
            outcome=outcome,
        )

    actions = ",".join(decision.action for decision in decisions)
    reply = (
        f"记住了：scope={scope}；status=applied；"
        f"cause=policy_{actions}。"
    )
    outcome = CommandOutcome(
        code="remembered",
        facts=(text, *(str(decision.claim_id) for decision in decisions)),
        fallback_reply=reply,
        status="applied",
        scope=scope,
        cause=f"policy_{actions}",
    )
    return CommandResult(handled=True, reply=outcome.fallback_reply, outcome=outcome)


def _global_remember_command(query: str, context: CommandContext, store: MemoryStore) -> CommandResult:
    if not context.is_admin:
        reply = "全局记忆只能由管理员写入。"
        outcome = CommandOutcome(
            code="forbidden",
            facts=(),
            fallback_reply=reply,
            status="forbidden",
            scope="global:global",
            cause="admin_required",
        )
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    text = query.strip()
    if not text:
        reply = "想让我全局记住什么？比如：/globalremember 所有人都喜欢热茶"
        outcome = CommandOutcome(
            code="missing_query",
            facts=(),
            fallback_reply=reply,
            status="rejected",
            scope="global:global",
            cause="missing_query",
        )
        return CommandResult(handled=True, reply=reply, outcome=outcome)
    if not context.message_id:
        reply = (
            "未写入全局记忆：scope=global:global；"
            "status=rejected；cause=missing_message_id。"
        )
        outcome = CommandOutcome(
            code="global_remember_rejected",
            facts=(text,),
            fallback_reply=reply,
            status="rejected",
            scope="global:global",
            cause="missing_message_id",
        )
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    event = MemoryEvent(
        context=context.memory_context,
        text=text,
        message_id=context.message_id,
        sequence=0,
    )
    extractor = MemoryExtractor()
    candidates = extractor.extract(event)
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

    policy = MemoryPolicy(store)
    decisions = policy.apply_global_command(
        event,
        tuple(candidates),
        authorized=context.is_admin,
    )
    if not decisions:
        reply = (
            "未写入全局记忆：scope=global:global；"
            "status=rejected；cause=policy_rejected。"
        )
        outcome = CommandOutcome(
            code="global_remember_rejected",
            facts=(text,),
            fallback_reply=reply,
            status="rejected",
            scope="global:global",
            cause="policy_rejected",
        )
        return CommandResult(
            handled=True,
            reply=outcome.fallback_reply,
            outcome=outcome,
        )

    actions = ",".join(decision.action for decision in decisions)
    reply = (
        "全局记忆已保存：scope=global:global；status=applied；"
        f"cause=policy_{actions}。"
    )
    outcome = CommandOutcome(
        code="remembered_global",
        facts=(text, *(str(decision.claim_id) for decision in decisions)),
        fallback_reply=reply,
        status="applied",
        scope="global:global",
        cause=f"policy_{actions}",
    )
    return CommandResult(handled=True, reply=outcome.fallback_reply, outcome=outcome)


def _context_scope(context: CommandContext) -> str:
    scope_type, scope_id = context.memory_context.primary_scope
    return f"{scope_type}:{scope_id}"


def _memories_command(query: str, context: CommandContext, store: MemoryStore) -> CommandResult:
    retriever = MemoryRetriever(store)
    results = retriever.retrieve(context.memory_context, query=query.strip())
    if context.memory_context.is_group:
        results = tuple(
            item
            for item in results
            if item.claim.scope_type != "private"
        )
    if not results:
        reply = "没有找到相关记忆。"
        outcome = CommandOutcome(
            code="memories_empty",
            facts=(),
            fallback_reply=reply,
            status="empty",
            scope=_context_scope(context),
            cause="no_permitted_records",
        )
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    lines = ["【允许使用的记忆列表】"]
    facts = []
    for i, item in enumerate(results, 1):
        lines.append(f"{i}. [{item.claim.memory_type}] {item.claim.predicate}: {item.claim.value} (ID: {item.claim.id})")
        facts.append(str(item.claim.id))

    reply = "\n".join(lines)
    outcome = CommandOutcome(
        code="memories_list",
        facts=tuple(facts),
        fallback_reply=reply,
        status="listed",
        scope=_context_scope(context),
        cause="permitted_records",
    )
    return CommandResult(handled=True, reply=reply, outcome=outcome)


def _claim_is_permitted(
    context: CommandContext,
    claim: MemoryClaim,
) -> bool:
    if claim.status not in {"active", "disputed"}:
        return False
    if context.memory_context.is_group:
        return (
            claim.scope_type == "group"
            and claim.scope_id == str(context.memory_context.group_id or "")
        ) or (
            claim.scope_type == "global"
            and claim.scope_id == "global"
        )
    return (
        claim.scope_type == "private"
        and claim.scope_id == context.uid
    ) or (
        claim.scope_type == "global"
        and claim.scope_id == "global"
    )


def _permitted_claims(
    context: CommandContext,
    store: MemoryStore,
) -> tuple[MemoryClaim, ...]:
    statuses = ("active", "disputed")
    if context.memory_context.is_group:
        claims = list(
            store.find_claims_exact(
                scope_type="group",
                scope_id=str(context.memory_context.group_id or ""),
                statuses=statuses,
            )
        )
    else:
        claims = list(
            store.find_claims_exact(
                scope_type="private",
                scope_id=context.uid,
                statuses=statuses,
            )
        )
    claims.extend(
        store.find_claims_exact(
            scope_type="global",
            scope_id="global",
            statuses=statuses,
        )
    )
    return tuple(claims)


def _store_unavailable_forget_result(
    claim_reference: int | str,
    claim_scope: str,
) -> CommandResult:
    is_claim_id = str(claim_reference).isdigit()
    reference_label = (
        f"ID: {claim_reference}"
        if is_claim_id
        else "内容描述"
    )
    facts = (
        (str(claim_reference), "retryable=true")
        if is_claim_id
        else ("retryable=true",)
    )
    reply = (
        f"记忆存储暂时不可用，本次未能确认或执行记忆变更 ({reference_label})："
        f"scope={claim_scope}；status=failed；cause=store_unavailable。"
    )
    outcome = CommandOutcome(
        code="forget_failed",
        facts=facts,
        fallback_reply=reply,
        status="failed",
        scope=claim_scope,
        cause="store_unavailable",
    )
    return CommandResult(handled=True, reply=reply, outcome=outcome)


def _forget_command(query: str, context: CommandContext, store: MemoryStore) -> CommandResult:
    target = query.strip()
    if not target:
        reply = "请指定要忘记的记忆 ID 或内容，例如：/forget 12 或 /forget 喜爱喝茶"
        outcome = CommandOutcome(
            code="missing_target",
            facts=(),
            fallback_reply=reply,
            status="rejected",
            scope=_context_scope(context),
            cause="missing_target",
        )
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    matched_claims: list[MemoryClaim] = []
    try:
        if target.isdigit():
            exact = store.get_claim(int(target))
            if exact is not None and _claim_is_permitted(context, exact):
                matched_claims = [exact]
            elif exact is None:
                retry = store.retry_pending_delete_cleanup(
                    int(target),
                    actor_qq=context.uid,
                    is_admin=context.is_admin,
                )
                if retry is not None:
                    deletion, claim_scope = retry
                    if deletion.status == "cleanup_completed":
                        prefix = "已完成记忆隐私清理"
                        cause = "privacy_cleanup_completed"
                    else:
                        prefix = "记忆正文已删除但隐私清理仍待重试"
                        cause = "privacy_cleanup_pending"
                    reply = (
                        f"{prefix} (ID: {target})：scope={claim_scope}；"
                        f"status={deletion.status}；cause={cause}。"
                    )
                    outcome = CommandOutcome(
                        code=(
                            "forget_cleanup_completed"
                            if deletion.cleanup_complete
                            else "forget_partial"
                        ),
                        facts=(
                            target,
                            f"retryable={str(deletion.retryable).lower()}",
                        ),
                        fallback_reply=reply,
                        status=deletion.status,
                        scope=claim_scope,
                        cause=cause,
                    )
                    return CommandResult(
                        handled=True,
                        reply=reply,
                        outcome=outcome,
                    )
        else:
            low_target = target.lower()
            matched_claims = [
                claim
                for claim in _permitted_claims(context, store)
                if (
                    low_target in claim.value.lower()
                    or low_target in claim.predicate.lower()
                )
            ]
    except sqlite3.Error:
        return _store_unavailable_forget_result(
            target,
            _context_scope(context),
        )

    if not matched_claims:
        reply = "未找到匹配的记忆。"
        outcome = CommandOutcome(
            code="not_found",
            facts=(target,),
            fallback_reply=reply,
            status="not_found",
            scope=_context_scope(context),
            cause="no_permitted_match",
        )
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    if len(matched_claims) > 1:
        reply = "找到多个相关记忆，请提供更具体的描述或 ID。"
        outcome = CommandOutcome(
            code="ambiguous",
            facts=tuple(str(c.id) for c in matched_claims),
            fallback_reply=reply,
            status="ambiguous",
            scope=_context_scope(context),
            cause="multiple_permitted_matches",
        )
        return CommandResult(handled=True, reply=reply, outcome=outcome)

    target_claim = matched_claims[0]
    claim_scope = f"{target_claim.scope_type}:{target_claim.scope_id}"
    if context.is_admin:
        try:
            deletion = store.delete_claim_physically_with_outcome(
                target_claim.id,
                reason="administrator_delete",
            )
        except sqlite3.Error:
            return _store_unavailable_forget_result(
                target_claim.id,
                claim_scope,
            )
        if deletion.status == "deleted":
            reply = (
                f"已删除记忆 (ID: {target_claim.id})：scope={claim_scope}；"
                "status=deleted；cause=administrator_delete。"
            )
            outcome = CommandOutcome(
                code="forgot",
                facts=(str(target_claim.id),),
                fallback_reply=reply,
                status="deleted",
                scope=claim_scope,
                cause="administrator_delete",
            )
        elif deletion.status == "partial":
            reply = (
                f"记忆正文已删除但隐私清理尚待重试 "
                f"(ID: {target_claim.id})：scope={claim_scope}；"
                "status=partial；cause=privacy_cleanup_pending。"
            )
            outcome = CommandOutcome(
                code="forget_partial",
                facts=(str(target_claim.id), "retryable=true"),
                fallback_reply=reply,
                status="partial",
                scope=claim_scope,
                cause="privacy_cleanup_pending",
            )
        else:
            reply = (
                f"未删除记忆 (ID: {target_claim.id})：scope={claim_scope}；"
                "status=no_op；cause=missing_at_mutation。"
            )
            outcome = CommandOutcome(
                code="forget_no_op",
                facts=(str(target_claim.id),),
                fallback_reply=reply,
                status="no_op",
                scope=claim_scope,
                cause="missing_at_mutation",
            )
        return CommandResult(
            handled=True,
            reply=outcome.fallback_reply,
            outcome=outcome,
        )

    if (
        target_claim.scope_type == "private"
        and target_claim.scope_id == context.uid
    ):
        try:
            deletion = store.delete_claim_physically_with_outcome(
                target_claim.id,
                reason="private_privacy_delete",
            )
        except sqlite3.Error:
            return _store_unavailable_forget_result(
                target_claim.id,
                claim_scope,
            )
        status = deletion.status
        if status == "deleted":
            cause = "private_privacy_delete"
            prefix = "已删除"
        elif status == "partial":
            cause = "privacy_cleanup_pending"
            prefix = "正文已删除但隐私清理尚待重试"
        else:
            status = "no_op"
            cause = "missing_at_mutation"
            prefix = "未删除"
        reply = (
            f"{prefix}记忆 (ID: {target_claim.id})：scope={claim_scope}；"
            f"status={status}；cause={cause}。"
        )
        outcome = CommandOutcome(
            code=(
                "forgot"
                if status == "deleted"
                else "forget_partial"
                if status == "partial"
                else "forget_no_op"
            ),
            facts=(
                str(target_claim.id),
                f"retryable={str(deletion.retryable).lower()}",
            ),
            fallback_reply=reply,
            status=status,
            scope=claim_scope,
            cause=cause,
        )
        return CommandResult(
            handled=True,
            reply=outcome.fallback_reply,
            outcome=outcome,
        )

    if (
        target_claim.scope_type == "group"
        and target_claim.speaker_qq == context.uid
    ):
        retracted = store.retract_group_claim(
            target_claim.id,
            actor_qq=context.uid,
            group_id=str(context.memory_context.group_id or ""),
        )
        if not retracted:
            reply = (
                f"未撤回记忆 (ID: {target_claim.id})：scope={claim_scope}；"
                "status=no_op；cause=missing_at_mutation。"
            )
            outcome = CommandOutcome(
                code="forget_no_op",
                facts=(str(target_claim.id),),
                fallback_reply=reply,
                status="no_op",
                scope=claim_scope,
                cause="missing_at_mutation",
            )
        else:
            reply = (
                f"已撤回本人记忆 (ID: {target_claim.id})：scope={claim_scope}；"
                "status=retracted；cause=author_withdrawal。"
            )
            outcome = CommandOutcome(
                code="retracted",
                facts=(str(target_claim.id),),
                fallback_reply=reply,
                status="retracted",
                scope=claim_scope,
                cause="author_withdrawal",
            )
        return CommandResult(
            handled=True,
            reply=outcome.fallback_reply,
            outcome=outcome,
        )

    if (
        target_claim.scope_type == "group"
        and target_claim.subject_type == "qq_user"
        and target_claim.subject_id == context.uid
    ):
        disputed = store.register_subject_dispute(
            target_claim.id,
            actor_qq=context.uid,
            group_id=str(context.memory_context.group_id or ""),
            source_message_id=context.message_id,
        )
        if disputed is None:
            reply = (
                f"未争议记忆 (ID: {target_claim.id})：scope={claim_scope}；"
                "status=no_op；cause=missing_at_mutation。"
            )
            outcome = CommandOutcome(
                code="forget_no_op",
                facts=(str(target_claim.id),),
                fallback_reply=reply,
                status="no_op",
                scope=claim_scope,
                cause="missing_at_mutation",
            )
        else:
            reply = (
                f"已将关于你的记忆标记为争议并禁止用于回答 "
                f"(ID: {target_claim.id})：scope={claim_scope}；"
                "status=disputed；cause=subject_dispute。"
            )
            outcome = CommandOutcome(
                code="disputed",
                facts=(str(target_claim.id),),
                fallback_reply=reply,
                status="disputed",
                scope=claim_scope,
                cause="subject_dispute",
            )
        return CommandResult(
            handled=True,
            reply=outcome.fallback_reply,
            outcome=outcome,
        )

    if target_claim.scope_type == "group":
        reply = "权限不足：你只能删除你自己发表的记忆。"
        cause = "foreign_author"
    else:
        reply = "权限不足：只有管理员可以删除这条记忆。"
        cause = "administrator_required"
    outcome = CommandOutcome(
        code="forbidden",
        facts=(str(target_claim.id),),
        fallback_reply=reply,
        status="forbidden",
        scope=claim_scope,
        cause=cause,
    )
    return CommandResult(
        handled=True,
        reply=outcome.fallback_reply,
        outcome=outcome,
    )


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


def handle_command(
    route: Route,
    context: CommandContext,
    store: MemoryStore | None = None,
    renderer: PersonaCommandRenderer | None = None,
) -> CommandResult:
    if store is None:
        store = get_memory_service().store
    command_handler = COMMANDS.get(route.command)
    if command_handler is None:
        command_text = f"/{route.command}" if route.command else "/"
        reply = (
            f"暂不支持这个命令：{command_text}。"
            "qqbot_lite 只提供 /search、/help、/reset、/remember、/globalremember、/memories、/forget。"
        )
        outcome = CommandOutcome(
            code="unknown",
            facts=(command_text,),
            fallback_reply=reply,
            status="unsupported",
            cause="unknown_command",
        )
        result = CommandResult(
            handled=True,
            reply=reply,
            outcome=outcome,
        )
    else:
        result = command_handler(route.query, context, store)

    if result.outcome is None or result.outcome.already_rendered:
        return result
    return CommandResult(
        handled=result.handled,
        reply=render_command_outcome(result.outcome, renderer=renderer),
        outcome=result.outcome,
    )
