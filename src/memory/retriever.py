from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Sequence

from src.memory.models import MemoryClaim, MemoryContext, RetrievedMemory
from src.memory.privacy import (
    claim_contains_hard_secret,
    safe_group_personalization,
    shared_claim_is_safe,
)
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


class MemoryRetriever:
    def __init__(self, store: MemoryStore | None = None) -> None:
        if store is None:
            from src.memory.service import get_memory_service

            store = get_memory_service().store
        self.store = store

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
        resolved_subject = self._resolve_query_subject(
            context,
            query_text,
            now_utc,
        )
        if (
            resolved_subject is None
            and predicate_hints
            and "我" in "".join(query_text.split())
        ):
            resolved_subject = user_id

        candidate_limit = max(limit * 4, 32)
        base_claims = _current_claims(
            self.store.search_authorized_claims(
                context,
                query_text,
                limit=candidate_limit,
            ),
            now_utc,
        )
        structured_claims = _current_claims(
            self.store.search_authorized_claims(
                context,
                "",
                limit=candidate_limit,
            ),
            now_utc,
        )
        reserved_predicates = _reserved_predicates_for_query(
            predicate_hints,
            include_defaults=not query_text,
            include_group_personalization=context.is_group,
        )
        reserved_claims = _current_claims(
            self.store.find_reserved_authorized_claims(
                context,
                predicates=reserved_predicates,
                subject_id=resolved_subject,
                limit=max(limit * 2, 24),
            ),
            now_utc,
        )
        raw_claims: list[MemoryClaim] = [
            *base_claims,
            *structured_claims,
            *reserved_claims,
        ]

        fallback_ids: set[int] = set()
        if context.is_group:
            private_personalization = tuple(
                claim
                for claim in raw_claims
                if claim.scope_type == "private"
                and _is_group_safe_personalization(claim)
            )
            fallback_ids = self._group_personalization_fallback_ids(
                group_id=str(context.group_id or ""),
                user_id=user_id,
                private_personalization=private_personalization,
                now_utc=now_utc,
            )
            for claim_id in fallback_ids:
                fallback_claim = self.store.get_claim(claim_id)
                if fallback_claim is not None:
                    raw_claims.append(fallback_claim)
        else:
            private_claims = tuple(
                claim
                for claim in raw_claims
                if claim.scope_type == "private"
            )
            raw_claims.extend(
                self._private_group_fallback(
                    context=context,
                    private_claims=private_claims,
                    now_utc=now_utc,
                )
            )

        candidates = [
            (
                claim,
                _claim_usage(
                    context,
                    claim,
                    fallback_ids=fallback_ids,
                ),
            )
            for claim in raw_claims
            if _claim_passes_runtime_privacy(context, claim)
            and _is_claim_active_and_valid(claim, now_utc)
        ]
        suppressed_ids = self.store.subject_dispute_suppressed_ids(
            tuple(claim.id for claim, _usage in candidates)
        )
        candidates = [
            (claim, usage)
            for claim, usage in candidates
            if claim.id not in suppressed_ids
        ]
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
        selected: list[RetrievedMemory] = []
        selected_ids: set[int] = set()
        rejected_conflict_ids: set[int] = set()
        for result in scored_results:
            claim = result.claim
            if (
                claim.id in selected_ids
                or claim.id in rejected_conflict_ids
            ):
                continue
            if claim.status != "disputed":
                selected.append(result)
                selected_ids.add(claim.id)
                if len(selected) >= limit:
                    break
                continue

            conflict_claims = self.store.authorized_conflict_group(
                context,
                claim.id,
            )
            if conflict_claims is None:
                rejected_conflict_ids.add(claim.id)
                continue
            conflict_ids = {item.id for item in conflict_claims}
            if not conflict_ids:
                rejected_conflict_ids.add(claim.id)
                continue
            conflict_suppressed = self.store.subject_dispute_suppressed_ids(
                tuple(conflict_ids)
            )
            if conflict_suppressed or any(
                not _claim_passes_runtime_privacy(context, item)
                or not _is_claim_active_and_valid(item, now_utc)
                for item in conflict_claims
            ):
                rejected_conflict_ids.update(conflict_ids)
                continue
            new_conflict_claims = tuple(
                item
                for item in conflict_claims
                if item.id not in selected_ids
            )
            if len(selected) + len(new_conflict_claims) > limit:
                rejected_conflict_ids.update(conflict_ids)
                continue
            conflict_results = [
                RetrievedMemory(
                    claim=item,
                    score=_calculate_score(
                        item,
                        query_text,
                        context,
                        now_utc,
                        resolved_subject=resolved_subject,
                        predicate_hints=predicate_hints,
                    ),
                    evidence_excerpts=(
                        (item.source_excerpt,)
                        if item.source_excerpt
                        else ()
                    ),
                    relation_types=(),
                    usage=_claim_usage(
                        context,
                        item,
                        fallback_ids=fallback_ids,
                    ),
                )
                for item in new_conflict_claims
            ]
            conflict_results.sort(
                key=lambda item: (item.score, item.claim.id),
                reverse=True,
            )
            selected.extend(conflict_results)
            selected_ids.update(
                item.claim.id for item in conflict_results
            )
            if len(selected) >= limit:
                break

        relation_types = self.store.relation_types_for_claims(
            tuple(item.claim.id for item in selected)
        )
        return tuple(
            replace(
                item,
                relation_types=relation_types.get(item.claim.id, ()),
            )
            for item in selected
        )

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

    def _private_group_fallback(
        self,
        *,
        context: MemoryContext,
        private_claims: Sequence[MemoryClaim],
        now_utc: datetime,
    ) -> tuple[MemoryClaim, ...]:
        group_id = str(context.group_id or "").strip()
        if not group_id:
            return ()
        user_id = str(context.user_id)
        private_predicates = {
            claim.predicate
            for claim in private_claims
            if claim.subject_id == user_id
            and _is_private_fallback_information(claim)
        }
        group_claims = self.store.find_claims_exact(
            scope_type="group",
            scope_id=group_id,
            statuses=_CURRENT_STATUSES,
            subject_type="qq_user",
            subject_id=user_id,
            speaker_qq=user_id,
        )
        return tuple(
            claim
            for claim in _current_claims(group_claims, now_utc)
            if _is_private_fallback_information(claim)
            and claim.predicate not in private_predicates
        )

    def _resolve_query_subject(
        self,
        context: MemoryContext,
        query: str,
        now_utc: datetime,
    ) -> str | None:
        query_folded = query.casefold()
        scope_type, scope_id = context.primary_scope
        search_scopes = [(scope_type, scope_id)]
        source_group_id = str(context.group_id or "").strip()
        if not context.is_group and source_group_id:
            search_scopes.append(("group", source_group_id))
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
            suppressed_alias_ids = self.store.subject_dispute_suppressed_ids(
                tuple(claim.id for claim in alias_claims)
            )
            alias_claims = tuple(
                claim
                for claim in alias_claims
                if claim.id not in suppressed_alias_ids
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
            if matching_aliases:
                longest_length = len(matching_aliases[0])
                longest_aliases = {
                    alias
                    for alias in matching_aliases
                    if len(alias) == longest_length
                }
                subject_ids = {
                    claim.subject_id
                    for claim in alias_claims
                    if claim.value in longest_aliases
                }
                if len(subject_ids) == 1:
                    return next(iter(subject_ids))
                return None

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


def _reserved_predicates_for_query(
    predicate_hints: frozenset[str],
    *,
    include_defaults: bool,
    include_group_personalization: bool,
) -> tuple[str, ...]:
    predicates: set[str] = set()
    if include_defaults:
        predicates.update(
            (
                *_NAME_PREDICATES,
                "likes",
                *_RESPONSE_STYLE_PREDICATES,
            )
        )
    if "identity" in predicate_hints:
        predicates.update(_NAME_PREDICATES)
    if "preferred_name" in predicate_hints:
        predicates.update(_PREFERRED_NAME_PREDICATES)
    if "likes" in predicate_hints:
        predicates.add("likes")
    if "response_style" in predicate_hints:
        predicates.update(_RESPONSE_STYLE_PREDICATES)
    if include_group_personalization:
        predicates.update(
            (
                *_PREFERRED_NAME_PREDICATES,
                *_RESPONSE_STYLE_PREDICATES,
            )
        )
    return tuple(sorted(predicates))


def _claim_passes_runtime_privacy(
    context: MemoryContext,
    claim: MemoryClaim,
) -> bool:
    if claim_contains_hard_secret(claim):
        return False
    if claim.scope_type in {"group", "global"}:
        return shared_claim_is_safe(claim)
    if context.is_group and claim.scope_type == "private":
        return _is_group_safe_personalization(claim)
    return True


def _claim_usage(
    context: MemoryContext,
    claim: MemoryClaim,
    *,
    fallback_ids: set[int],
) -> str:
    if context.is_group:
        if claim.scope_type == "private" or claim.id in fallback_ids:
            return "personalization"
        return "evidence"
    if (
        claim.subject_id == str(context.user_id)
        and _is_group_safe_personalization(claim)
    ):
        return "personalization"
    return "evidence"


def _is_group_safe_personalization(claim: MemoryClaim) -> bool:
    return safe_group_personalization(claim)


def _is_private_fallback_information(claim: MemoryClaim) -> bool:
    return (
        claim.memory_type in {"identity", "preferred_name", "preference"}
        or claim.predicate in (
            *_NAME_PREDICATES,
            *_RESPONSE_STYLE_PREDICATES,
        )
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

    recency_score = 1.0
    timestamp_str = claim.last_confirmed_at or claim.created_at
    if timestamp_str and not _is_preferred_name_claim(claim):
        try:
            ts = _parse_utc(timestamp_str)
            age_days = max((now_utc - ts).total_seconds() / 86400.0, 0.0)
            if (
                claim.memory_type == "preference"
                or claim.predicate in ("likes", "response_style")
            ):
                recency_score = max(
                    0.5,
                    1.0 / (1.0 + 0.01 * age_days),
                )
            else:
                recency_score = max(
                    0.2,
                    1.0 / (1.0 + 0.05 * age_days),
                )
        except (TypeError, ValueError):
            pass

    total = (
        truth_score
        + subject_score
        + predicate_score
        + scope_score
        + relevance_score
    )
    return round(total + recency_score, 4)


def format_memory_context(results: Sequence[RetrievedMemory]) -> str:
    evidence_items = [r for r in results if getattr(r, "usage", "evidence") == "evidence"]
    personalization_items = [r for r in results if getattr(r, "usage", "evidence") == "personalization"]

    evidence_lines: list[str] = []
    if evidence_items:
        for r in evidence_items:
            c = r.claim
            relation_text = (
                ",".join(r.relation_types)
                if r.relation_types
                else "none"
            )
            evidence_lines.append(
                f"- 作用域={c.scope_type}:{c.scope_id}；"
                f"发言者={c.speaker_qq}；主体={c.subject_id}；"
                f"类型={c.memory_type}；内容={c.predicate}为{c.value}；"
                f"status={c.status}；relations={relation_text}"
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
