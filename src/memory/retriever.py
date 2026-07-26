from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.config import config
from src.memory.models import MemoryClaim, MemoryContext, RetrievedMemory
from src.memory.store import MemoryStore


def _get_default_store() -> MemoryStore:
    db_path = config.data_dir / "memory.db"
    store = MemoryStore(db_path)
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
        query_text = str(query or "").strip()
        user_id = str(context.user_id)
        now_utc = datetime.now(timezone.utc)

        # 1. Hard scope SQL retrieval
        if context.is_group:
            group_id = str(context.group_id or "")
            # Factual evidence for group: current group + global
            group_claims = self.store.find_claims_exact(
                scope_type="group",
                scope_id=group_id,
                statuses=("active", "disputed"),
            )
            global_claims = self.store.find_claims_exact(
                scope_type="global",
                scope_id="global",
                statuses=("active", "disputed"),
            )
            # Private fallback / personalization for current user only
            user_private_claims = self.store.find_claims_exact(
                scope_type="private",
                scope_id=user_id,
                statuses=("active", "disputed"),
            )

            # Filter user private claims
            private_fallback_claims = [
                c for c in user_private_claims
                if c.speaker_qq == user_id == c.subject_id
                and (c.memory_type in ("identity", "preferred_name", "preference")
                     or c.predicate in ("name", "preferred_name", "response_style"))
            ]
        else:
            # Private chat
            private_claims = self.store.find_claims_exact(
                scope_type="private",
                scope_id=user_id,
                statuses=("active", "disputed"),
            )
            global_claims = self.store.find_claims_exact(
                scope_type="global",
                scope_id="global",
                statuses=("active", "disputed"),
            )
            group_claims = ()
            private_fallback_claims = private_claims

        # 2. Filter out expired claims
        candidate_claims: list[tuple[MemoryClaim, str]] = []  # (claim, usage)

        # Add group and global claims as evidence
        for claim in (*group_claims, *global_claims):
            if _is_claim_active_and_valid(claim, now_utc):
                candidate_claims.append((claim, "evidence"))

        # Add personalization / private fallback claims
        for claim in private_fallback_claims:
            if not _is_claim_active_and_valid(claim, now_utc):
                continue
            if claim.predicate in ("preferred_name", "response_style") or claim.memory_type == "preferred_name":
                candidate_claims.append((claim, "personalization"))
            elif not context.is_group:
                candidate_claims.append((claim, "evidence"))
            else:
                candidate_claims.append((claim, "evidence"))

        # Deduplicate candidates by claim.id
        seen_ids = set()
        unique_candidates: list[tuple[MemoryClaim, str]] = []
        for claim, usage in candidate_claims:
            if claim.id not in seen_ids:
                seen_ids.add(claim.id)
                unique_candidates.append((claim, usage))

        # 3. Score and rank candidates
        scored_results: list[RetrievedMemory] = []
        for claim, usage in unique_candidates:
            score = _calculate_score(claim, query_text, context, now_utc)
            scored_results.append(
                RetrievedMemory(
                    claim=claim,
                    score=score,
                    evidence_excerpts=(claim.source_excerpt,) if claim.source_excerpt else (),
                    relation_types=(),
                    usage=usage,
                )
            )

        # Sort by score descending, then by claim id
        scored_results.sort(key=lambda r: (r.score, r.claim.id), reverse=True)
        return tuple(scored_results[:limit])


def _is_claim_active_and_valid(claim: MemoryClaim, now_utc: datetime) -> bool:
    if claim.status not in ("active", "disputed"):
        return False
    if claim.valid_to:
        try:
            vto = datetime.fromisoformat(claim.valid_to.replace("Z", "+00:00"))
            if vto <= now_utc:
                return False
        except ValueError:
            pass
    return True


def _calculate_score(
    claim: MemoryClaim,
    query: str,
    context: MemoryContext,
    now_utc: datetime,
) -> float:
    # Base confidence score
    confidence_map = {"high": 1.0, "medium": 0.7, "low": 0.4}
    base = confidence_map.get(claim.extraction_confidence, 0.7)

    # Scope weight
    scope_weight = 1.2 if (claim.scope_type in ("group", "private") and claim.scope_id != "global") else 1.0

    # Match / relevance score
    relevance = 1.0
    if query:
        q_lower = query.lower()
        if q_lower in claim.value.lower() or q_lower in claim.predicate.lower() or q_lower in claim.subject_id:
            relevance += 0.5
        if claim.subject_id == context.user_id:
            relevance += 0.3

    # Age decay
    age_factor = 1.0
    timestamp_str = claim.last_confirmed_at or claim.created_at
    if timestamp_str and claim.memory_type != "preferred_name" and claim.predicate != "preferred_name":
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            age_days = max((now_utc - ts).total_seconds() / 86400.0, 0.0)
            if claim.memory_type == "preference" or claim.predicate in ("likes", "response_style"):
                # Preferences retain a score floor of 0.5
                age_factor = max(0.5, 1.0 / (1.0 + 0.01 * age_days))
            else:
                age_factor = max(0.2, 1.0 / (1.0 + 0.05 * age_days))
        except ValueError:
            pass

    return round(base * scope_weight * relevance * age_factor, 4)


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
