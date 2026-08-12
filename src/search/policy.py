"""Pure answer policy: map immutable search state into a bounded answer state.

This module performs no I/O, no LLM calls, and no search.  It is the single
place where ``RequestAnalysis`` (risk/freshness/retrieval), the final immutable
``EvidenceBundle``, and any ``SearchFailureCode`` are turned into an
``AnswerState`` consumed by the chat dispatch and, later, the validator and
renderer.
"""

from __future__ import annotations

from dataclasses import replace

from src.search.models import (
    AllowedClaimScope,
    AnswerCertainty,
    AnswerGenerationMode,
    AnswerState,
    DisclosureCode,
    EvidenceBundle,
    EvidenceState,
    FreshnessRequirement,
    RenderOutcome,
    RenderState,
    RequestAnalysis,
    SearchFailureCode,
    SkipReason,
    ValidationReport,
    ValidatorRequirement,
    ValidatorStatus,
    WarningCode,
)
from src.search.validation import sanitize_visible_block_text


_CLOSED_TASK_SKIP_REASONS = frozenset(
    {
        SkipReason.SOCIAL_OR_EMOTIONAL,
        SkipReason.CREATIVE_OR_ROLEPLAY,
        SkipReason.PROVIDED_TEXT_TRANSFORM,
        SkipReason.PROVIDED_CONTENT_SUMMARY,
        SkipReason.PURE_MATH,
        SkipReason.CLOSED_LOGIC,
        SkipReason.CLOSED_CONTEXT_ONLY,
    }
)


def decide_answer_state(
    analysis: RequestAnalysis,
    evidence: EvidenceBundle | None,
    failure_code: SearchFailureCode | None,
) -> AnswerState:
    """Map request analysis and immutable search state to a bounded answer state."""
    state = evidence.evidence_state if evidence is not None else None
    warnings = (
        (WarningCode.HIGH_CONSEQUENCE,)
        if analysis.risk.warning_required
        else ()
    )
    validator_requirement = (
        ValidatorRequirement.FAIL_CLOSED
        if analysis.risk.fail_closed
        or analysis.freshness.requirement is FreshnessRequirement.CURRENT
        else ValidatorRequirement.NORMAL
    )

    if analysis.retrieval.skip_reason in _CLOSED_TASK_SKIP_REASONS:
        return AnswerState(
            None,
            AnswerGenerationMode.PLAIN,
            AnswerCertainty.UNVERIFIED,
            AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS,
            (),
            warnings,
            validator_requirement,
        )
    if analysis.retrieval.skip_reason is SkipReason.USER_FORBID_WEB:
        return AnswerState(
            None,
            AnswerGenerationMode.FIXED,
            AnswerCertainty.UNVERIFIED,
            AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS,
            (DisclosureCode.USER_FORBID_WEB,),
            warnings,
            validator_requirement,
        )
    if state is EvidenceState.SUFFICIENT:
        return AnswerState(
            state,
            AnswerGenerationMode.GROUNDED,
            AnswerCertainty.VERIFIED,
            AllowedClaimScope.ALL_SUPPORTED,
            (),
            warnings,
            validator_requirement,
        )
    if state is EvidenceState.PARTIAL:
        return AnswerState(
            state,
            AnswerGenerationMode.GROUNDED,
            AnswerCertainty.LIMITED,
            AllowedClaimScope.SUPPORTED_SUBSET,
            (DisclosureCode.PARTIAL_EVIDENCE,),
            warnings,
            validator_requirement,
        )
    if state is EvidenceState.CONFLICTING:
        return AnswerState(
            state,
            AnswerGenerationMode.GROUNDED,
            AnswerCertainty.CONFLICTING,
            _conflict_scope(evidence),
            (DisclosureCode.SOURCE_CONFLICT,),
            warnings,
            validator_requirement,
        )
    return AnswerState(
        state,
        AnswerGenerationMode.FIXED,
        AnswerCertainty.UNVERIFIED,
        AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS,
        (_failure_disclosure(failure_code),),
        warnings,
        validator_requirement,
    )


def _conflict_scope(evidence: EvidenceBundle | None) -> AllowedClaimScope:
    """Return the widest conflict-aware scope for a conflicting bundle.

    When at least one material topic remains supported and not covered by a
    structured conflict, the answer may still describe the uncontested subset.
    Otherwise the answer may only describe the conflicts themselves.
    """
    if evidence is None:
        return AllowedClaimScope.CONFLICT_DESCRIPTION_ONLY
    material_ids = {
        topic.topic_id for topic in evidence.plan.required_topics if topic.material
    }
    if not material_ids:
        return AllowedClaimScope.CONFLICT_DESCRIPTION_ONLY
    conflict_ids = _conflict_topic_ids(evidence)
    supported = set(evidence.supported_topic_ids)
    if any(
        topic_id in supported and topic_id not in conflict_ids
        for topic_id in material_ids
    ):
        return AllowedClaimScope.SUPPORTED_SUBSET_WITH_CONFLICTS
    return AllowedClaimScope.CONFLICT_DESCRIPTION_ONLY


def _conflict_topic_ids(evidence: EvidenceBundle) -> frozenset[str]:
    """Map conflict members back to material topic IDs via their evidence items."""
    label_to_ids: dict[str, set[str]] = {}
    for topic in evidence.plan.required_topics:
        if topic.material:
            label_to_ids.setdefault(topic.label, set()).add(topic.topic_id)
    item_by_id = {item.evidence_id: item for item in evidence.evidence_items}
    topic_ids: set[str] = set()
    for conflict in evidence.conflicts:
        for member in conflict.members:
            item = item_by_id.get(member.evidence_id)
            if item is None:
                continue
            for label in item.supported_topics:
                topic_ids.update(label_to_ids.get(label, ()))
    return frozenset(topic_ids)


def _failure_disclosure(failure_code: SearchFailureCode | None) -> DisclosureCode:
    """Map a search failure to its deterministic disclosure code."""
    if failure_code is SearchFailureCode.USER_FORBID_WEB:
        return DisclosureCode.USER_FORBID_WEB
    if failure_code is SearchFailureCode.VALIDATION_FAILED:
        return DisclosureCode.VALIDATION_FAILED
    return DisclosureCode.ONLINE_VERIFICATION_FAILED


def build_render_state(
    answer_state: AnswerState,
    validation: ValidationReport,
    evidence: EvidenceBundle | None,
) -> RenderState:
    """Build a deterministic render view from policy, validation and evidence."""
    scope = validation.effective_claim_scope
    visible_blocks = tuple(
        replace(block, text=sanitize_visible_block_text(block.text))
        for block in validation.retained_blocks
    )
    visible_claims = validation.retained_claims

    cited_ids: list[str] = []
    for claim in visible_claims:
        for evidence_id in claim.evidence_ids:
            if evidence_id and evidence_id not in cited_ids:
                cited_ids.append(evidence_id)
    citation_map = {
        evidence_id: index for index, evidence_id in enumerate(cited_ids, 1)
    }

    items = evidence.evidence_items if evidence is not None else ()
    item_by_id = {item.evidence_id: item for item in items}
    used_sources = tuple(
        item_by_id[evidence_id]
        for evidence_id in cited_ids
        if evidence_id in item_by_id and item_by_id[evidence_id].citable
    )
    conflicts = evidence.conflicts if evidence is not None else ()
    conflict_groups = tuple(
        conflict
        for conflict in conflicts
        if any(member.evidence_id in citation_map for member in conflict.members)
    )

    return RenderState(
        outcome=_render_outcome(scope, validation.status),
        visible_blocks=visible_blocks,
        visible_claims=visible_claims,
        citation_map=citation_map,
        used_sources=used_sources,
        conflict_groups=conflict_groups,
        disclosure_codes=answer_state.disclosure_codes,
        warning_codes=answer_state.warning_codes,
    )


def _render_outcome(
    scope: AllowedClaimScope,
    status: ValidatorStatus,
) -> RenderOutcome:
    if status is ValidatorStatus.MALFORMED:
        return RenderOutcome.VALIDATION_FAILURE
    if scope is AllowedClaimScope.ALL_SUPPORTED:
        return RenderOutcome.ANSWER
    if scope is AllowedClaimScope.SUPPORTED_SUBSET:
        return RenderOutcome.PARTIAL
    if scope in {
        AllowedClaimScope.SUPPORTED_SUBSET_WITH_CONFLICTS,
        AllowedClaimScope.CONFLICT_DESCRIPTION_ONLY,
    }:
        return RenderOutcome.CONFLICT
    return RenderOutcome.FAILURE
