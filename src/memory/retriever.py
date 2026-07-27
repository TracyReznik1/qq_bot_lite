from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from src.config import config
from src.memory.models import MemoryClaim, MemoryContext, RetrievedMemory
from src.memory.store import MemoryStore


_CURRENT_STATUSES = ("active", "disputed")
_NAME_PREDICATES = ("name", "real_name", "preferred_name")
_PREFERRED_NAME_PREDICATES = ("preferred_name",)
_RESPONSE_STYLE_PREDICATES = ("response_style",)
_QUERY_PREDICATE_MARKERS = {
    "identity": ("是谁", "叫什么", "名字", "姓名", "身份"),
    "preferred_name": ("怎么称呼", "称呼", "昵称", "叫我"),
    "likes": ("喜欢", "爱好", "偏好"),
    "response_style": ("回复风格", "说话风格", "语气", "怎么回复"),
}


def _get_default_store() -> MemoryStore:
    store = MemoryStore(config.memory_database_path)
    store.initialize()
    return store


class MemoryRetriever:
    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store or _get_default_store()

    def retrieve(
        self,
        context: MemoryContext,
        query: str = "",
        limit: int = 12,
    ) -> tuple[RetrievedMemory, ...]:
        if limit <= 0:
            return ()
        query_text = str(query or "").strip()
        user_id = str(context.user_id)
        now_utc = datetime.now(timezone.utc)
        predicate_hints = _predicate_hints(query_text)

        if context.is_group:
            group_id = str(context.group_id or "")
            group_claims = _current_claims(
                self.store.find_claims_exact(
                    scope_type="group",
                    scope_id=group_id,
                    statuses=_CURRENT_STATUSES,
                ),
                now_utc,
            )
            global_claims = _attributed_global_claims(
                self.store.find_claims_exact(
                    scope_type="global",
                    scope_id="global",
                    statuses=_CURRENT_STATUSES,
                ),
                now_utc,
            )
            private_claims = _current_claims(
                self.store.find_claims_exact(
                    scope_type="private",
                    scope_id=user_id,
                    statuses=_CURRENT_STATUSES,
                    subject_type="qq_user",
                    subject_id=user_id,
                ),
                now_utc,
            )
            private_personalization = tuple(
                claim
                for claim in private_claims
                if _is_group_safe_personalization(claim)
            )
            fallback_ids = self._group_personalization_fallback_ids(
                group_id=group_id,
                user_id=user_id,
                private_personalization=private_personalization,
                now_utc=now_utc,
            )
            candidates = [
                (
                    claim,
                    "personalization"
                    if claim.id in fallback_ids
                    else "evidence",
                )
                for claim in group_claims
            ]
            candidates.extend((claim, "evidence") for claim in global_claims)
            candidates.extend(
                (claim, "personalization")
                for claim in private_personalization
            )
        else:
            private_claims = _current_claims(
                self.store.find_claims_exact(
                    scope_type="private",
                    scope_id=user_id,
                    statuses=_CURRENT_STATUSES,
                ),
                now_utc,
            )
            global_claims = _attributed_global_claims(
                self.store.find_claims_exact(
                    scope_type="global",
                    scope_id="global",
                    statuses=_CURRENT_STATUSES,
                ),
                now_utc,
            )
            candidates = [
                (
                    claim,
                    "personalization"
                    if (
                        claim.subject_id == user_id
                        and _is_group_safe_personalization(claim)
                    )
                    else "evidence",
                )
                for claim in private_claims
            ]
            candidates.extend((claim, "evidence") for claim in global_claims)

        resolved_subject = self._resolve_query_subject(
            context,
            query_text,
            now_utc,
        )
        unique_candidates: list[tuple[MemoryClaim, str]] = []
        seen_ids: set[int] = set()
        for claim, usage in candidates:
            if claim.id in seen_ids:
                continue
            seen_ids.add(claim.id)
            unique_candidates.append((claim, usage))

        scored_results = [
            RetrievedMemory(
                claim=claim,
                score=_calculate_score(
                    claim,
                    query_text,
                    context,
                    now_utc,
                    resolved_subject=resolved_subject,
                    predicate_hints=predicate_hints,
                ),
                evidence_excerpts=(
                    (claim.source_excerpt,)
                    if claim.source_excerpt
                    else ()
                ),
                relation_types=(),
                usage=usage,
            )
            for claim, usage in unique_candidates
        ]
        scored_results.sort(
            key=lambda result: (result.score, result.claim.id),
            reverse=True,
        )
        return tuple(scored_results[:limit])

    def _group_personalization_fallback_ids(
        self,
        *,
        group_id: str,
        user_id: str,
        private_personalization: Sequence[MemoryClaim],
        now_utc: datetime,
    ) -> set[int]:
        missing_predicates: list[str] = []
        if not any(
            _is_preferred_name_claim(claim)
            for claim in private_personalization
        ):
            missing_predicates.extend(_PREFERRED_NAME_PREDICATES)
        if not any(
            _is_response_style_claim(claim)
            for claim in private_personalization
        ):
            missing_predicates.extend(_RESPONSE_STYLE_PREDICATES)
        if not missing_predicates:
            return set()

        fallback_claims = self.store.find_claims_exact(
            scope_type="group",
            scope_id=group_id,
            statuses=_CURRENT_STATUSES,
            subject_type="qq_user",
            subject_id=user_id,
            predicates=tuple(missing_predicates),
            speaker_qq=user_id,
        )
        return {
            claim.id
            for claim in _current_claims(fallback_claims, now_utc)
            if _is_group_safe_personalization(claim)
        }

    def _resolve_query_subject(
        self,
        context: MemoryContext,
        query: str,
        now_utc: datetime,
    ) -> str | None:
        query_folded = query.casefold()
        scope_type, scope_id = context.primary_scope
        search_scopes = [(scope_type, scope_id)]
        if (scope_type, scope_id) != ("global", "global"):
            search_scopes.append(("global", "global"))

        for alias_scope_type, alias_scope_id in search_scopes:
            alias_claims = _current_claims(
                self.store.find_claims_exact(
                    scope_type=alias_scope_type,
                    scope_id=alias_scope_id,
                    statuses=_CURRENT_STATUSES,
                    subject_type="qq_user",
                    predicates=_NAME_PREDICATES,
                ),
                now_utc,
            )
            matching_aliases = sorted(
                {
                    claim.value
                    for claim in alias_claims
                    if len(claim.value.strip()) >= 2
                    and claim.value.casefold() in query_folded
                },
                key=len,
                reverse=True,
            )
            for alias in matching_aliases:
                subject_ids = {
                    claim.subject_id
                    for claim in alias_claims
                    if claim.value == alias
                }
                if len(subject_ids) == 1:
                    return next(iter(subject_ids))

        compact_query = "".join(query_folded.split())
        if (
            "我是谁" in compact_query
            or "我叫什么" in compact_query
            or compact_query in {"怎么称呼", "怎么称呼我", "该怎么称呼我"}
        ):
            return str(context.user_id)
        return None


def _is_claim_active_and_valid(claim: MemoryClaim, now_utc: datetime) -> bool:
    if claim.status not in _CURRENT_STATUSES:
        return False
    if claim.valid_to:
        try:
            vto = _parse_utc(claim.valid_to)
            if vto <= now_utc:
                return False
        except (TypeError, ValueError):
            pass
    return True


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _current_claims(
    claims: Sequence[MemoryClaim],
    now_utc: datetime,
) -> tuple[MemoryClaim, ...]:
    return tuple(
        claim
        for claim in claims
        if _is_claim_active_and_valid(claim, now_utc)
    )


def _attributed_global_claims(
    claims: Sequence[MemoryClaim],
    now_utc: datetime,
) -> tuple[MemoryClaim, ...]:
    return tuple(
        claim
        for claim in _current_claims(claims, now_utc)
        if claim.speaker_qq.strip()
    )


def _is_preferred_name_claim(claim: MemoryClaim) -> bool:
    return (
        claim.predicate in _PREFERRED_NAME_PREDICATES
        or claim.memory_type == "preferred_name"
    )


def _is_response_style_claim(claim: MemoryClaim) -> bool:
    return claim.predicate in _RESPONSE_STYLE_PREDICATES


def _is_group_safe_personalization(claim: MemoryClaim) -> bool:
    return (
        _is_preferred_name_claim(claim)
        or _is_response_style_claim(claim)
    )


def _predicate_hints(query: str) -> frozenset[str]:
    compact_query = "".join(query.casefold().split())
    hints: set[str] = set()
    for hint, markers in _QUERY_PREDICATE_MARKERS.items():
        if any(marker in compact_query for marker in markers):
            hints.add(hint)
    return frozenset(hints)


def _claim_matches_predicate_hints(
    claim: MemoryClaim,
    predicate_hints: frozenset[str],
) -> bool:
    if not predicate_hints:
        return False
    if claim.predicate in predicate_hints:
        return True
    if "identity" in predicate_hints:
        return (
            claim.memory_type in {"identity", "preferred_name"}
            or claim.predicate in _NAME_PREDICATES
        )
    return False


def _calculate_score(
    claim: MemoryClaim,
    query: str,
    context: MemoryContext,
    now_utc: datetime,
    *,
    resolved_subject: str | None = None,
    predicate_hints: frozenset[str] = frozenset(),
) -> float:
    confidence_map = {"high": 1.0, "medium": 0.7, "low": 0.4}
    truth_score = 2.0 * confidence_map.get(claim.truth_confidence, 0.7)
    subject_score = (
        3.0
        if resolved_subject is not None
        and claim.subject_id == resolved_subject
        else 0.0
    )
    predicate_score = (
        2.0
        if _claim_matches_predicate_hints(claim, predicate_hints)
        else 0.0
    )
    scope_score = (
        1.0
        if (claim.scope_type, claim.scope_id) == context.primary_scope
        else 0.35
        if claim.scope_type == "global"
        else 0.6
    )
    relevance_score = 0.0
    if query:
        q_lower = query.casefold()
        searchable_fields = (
            claim.value.casefold(),
            claim.predicate.casefold(),
            claim.subject_id.casefold(),
        )
        if q_lower in searchable_fields:
            relevance_score = 1.5
        elif any(
            field and (field in q_lower or q_lower in field)
            for field in searchable_fields
        ):
            relevance_score = 0.75

    age_factor = 1.0
    timestamp_str = claim.last_confirmed_at or claim.created_at
    if timestamp_str and not _is_preferred_name_claim(claim):
        try:
            ts = _parse_utc(timestamp_str)
            age_days = max((now_utc - ts).total_seconds() / 86400.0, 0.0)
            if (
                claim.memory_type == "preference"
                or claim.predicate in ("likes", "response_style")
            ):
                age_factor = max(0.5, 1.0 / (1.0 + 0.01 * age_days))
            else:
                age_factor = max(0.2, 1.0 / (1.0 + 0.05 * age_days))
        except (TypeError, ValueError):
            pass

    total = (
        truth_score
        + subject_score
        + predicate_score
        + scope_score
        + relevance_score
    )
    return round(total * age_factor, 4)


def format_memory_context(results: Sequence[RetrievedMemory]) -> str:
    evidence_items = [r for r in results if getattr(r, "usage", "evidence") == "evidence"]
    personalization_items = [r for r in results if getattr(r, "usage", "evidence") == "personalization"]

    evidence_lines: list[str] = []
    if evidence_items:
        for r in evidence_items:
            c = r.claim
            evidence_lines.append(
                f"- 作用域={c.scope_type}:{c.scope_id}；发言者={c.speaker_qq}；主体={c.subject_id}；类型={c.memory_type}；内容={c.predicate}为{c.value}"
            )
    else:
        evidence_lines.append("暂无")

    personalization_lines: list[str] = []
    if personalization_items:
        for r in personalization_items:
            c = r.claim
            if c.predicate == "preferred_name":
                personalization_lines.append(f"- 主体=当前发言者；首选称呼={c.value}")
            else:
                personalization_lines.append(f"- 主体=当前发言者；{c.predicate}={c.value}")

    lines = [
        "[允许使用的记忆证据]",
        *evidence_lines,
        "[/允许使用的记忆证据]",
    ]

    if personalization_lines:
        lines.extend([
            "[仅用于称呼和表达的个性化信息]",
            *personalization_lines,
            "禁止把本区内容作为公开身份、经历或关系事实。",
            "[/仅用于称呼和表达的个性化信息]",
        ])

    return "\n".join(lines)
