"""Atomic claim validation: strict parsing, deterministic checks, model support."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from src.search.models import (
    AnswerBlock,
    Claim,
    EvidenceBundle,
    EvidenceState,
    GroundedDraft,
    SearchTier,
    SupportLabel,
    ValidationReport,
)

_FENCE_PATTERN = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)

_VALID_BLOCK_KINDS = frozenset({"factual", "inference", "non_factual"})
_NUMERIC_CITATION = re.compile(r"\[\d+\]")
_HTTP_URL = re.compile(r"^https?://", re.IGNORECASE)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _raise_invalid_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant: {value}")


def parse_grounded_draft(text: str) -> GroundedDraft:
    raw = str(text or "").strip()
    fenced = _FENCE_PATTERN.fullmatch(raw)
    if fenced is not None:
        raw = fenced.group("body").strip()
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_raise_invalid_constant,
        )
    except (json.JSONDecodeError, ValueError):
        raise ValueError("draft is not strict JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("draft must be a JSON object")

    raw_blocks = payload.get("answer_blocks")
    raw_claims = payload.get("claims")
    if not isinstance(raw_blocks, list) or not isinstance(raw_claims, list):
        raise ValueError("draft requires answer_blocks and claims arrays")

    block_ids: list[str] = []
    blocks: list[AnswerBlock] = []
    for index, raw_block in enumerate(raw_blocks):
        block = _parse_block(raw_block, index)
        block_ids.append(block.block_id)
        blocks.append(block)
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("block ids must be unique")

    claim_ids: list[str] = []
    claims: list[Claim] = []
    for index, raw_claim in enumerate(raw_claims):
        claim = _parse_claim(raw_claim, index)
        claim_ids.append(claim.claim_id)
        claims.append(claim)
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("claim ids must be unique")

    claims_by_id = {claim.claim_id: claim for claim in claims}
    block_ids_set = set(block_ids)
    for claim in claims:
        if claim.block_id not in block_ids_set:
            raise ValueError(f"claim {claim.claim_id} references unknown block")
    for block in blocks:
        for claim_id in block.claim_ids:
            if claim_id not in claims_by_id:
                raise ValueError(f"block {block.block_id} references unknown claim")

    limitations = _string_list(payload.get("limitations"))
    conflict_summary = _string_list(payload.get("conflict_summary"))
    used_fallback = payload.get("used_knowledge_fallback") is True

    return GroundedDraft(
        answer_blocks=tuple(blocks),
        claims=tuple(claims),
        limitations=tuple(limitations),
        conflict_summary=tuple(conflict_summary),
        used_knowledge_fallback=used_fallback,
    )


def _parse_block(raw: Any, index: int) -> AnswerBlock:
    if not isinstance(raw, dict):
        raise ValueError(f"answer_blocks[{index}] must be an object")
    block_id = raw.get("block_id")
    kind = raw.get("kind")
    text = raw.get("text")
    claim_ids = raw.get("claim_ids")
    if not isinstance(block_id, str) or not block_id:
        raise ValueError(f"answer_blocks[{index}].block_id invalid")
    if kind not in _VALID_BLOCK_KINDS:
        raise ValueError(f"answer_blocks[{index}].kind invalid")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"answer_blocks[{index}].text invalid")
    if not isinstance(claim_ids, list) or any(not isinstance(c, str) for c in claim_ids):
        raise ValueError(f"answer_blocks[{index}].claim_ids invalid")
    return AnswerBlock(block_id, kind, text, tuple(claim_ids))


def _parse_claim(raw: Any, index: int) -> Claim:
    if not isinstance(raw, dict):
        raise ValueError(f"claims[{index}] must be an object")
    claim_id = raw.get("claim_id")
    block_id = raw.get("block_id")
    text = raw.get("text")
    material = raw.get("material")
    evidence_ids = raw.get("evidence_ids")
    if not isinstance(claim_id, str) or not claim_id:
        raise ValueError(f"claims[{index}].claim_id invalid")
    if not isinstance(block_id, str) or not block_id:
        raise ValueError(f"claims[{index}].block_id invalid")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"claims[{index}].text invalid")
    if not isinstance(material, bool):
        raise ValueError(f"claims[{index}].material invalid")
    if not isinstance(evidence_ids, list) or any(not isinstance(e, str) for e in evidence_ids):
        raise ValueError(f"claims[{index}].evidence_ids invalid")
    return Claim(claim_id, block_id, text, material, tuple(evidence_ids))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)][:20]


# ── structural validation ──────────────────────────────────────────────

def _has_http_final_url(evidence_id: str, bundle: EvidenceBundle) -> bool:
    for item in bundle.evidence_items:
        if item.evidence_id == evidence_id:
            return bool(item.citable) and bool(_HTTP_URL.match(item.url or ""))
    return False


def _evidence_exists(evidence_id: str, bundle: EvidenceBundle) -> bool:
    return any(item.evidence_id == evidence_id for item in bundle.evidence_items)


class _StructuralReport:
    def __init__(self) -> None:
        self.removed_block_ids: list[str] = []
        self.kept_blocks: list[AnswerBlock] = []
        self.kept_claims: list[Claim] = []
        self.limitations: list[str] = []
        self.labels: dict[str, SupportLabel] = {}


def _apply_structural_checks(
    draft: GroundedDraft,
    bundle: EvidenceBundle,
    report: _StructuralReport,
) -> None:
    claims_by_id = {claim.claim_id: claim for claim in draft.claims}
    blocks_by_id = {block.block_id: block for block in draft.answer_blocks}

    # failed/insufficient retrieval cannot have claims or citations
    if bundle.evidence_state is EvidenceState.INSUFFICIENT:
        for block in draft.answer_blocks:
            if block.kind != "non_factual":
                report.removed_block_ids.append(block.block_id)
        report.limitations.append("insufficient_evidence")
        return

    if bundle.evidence_state is EvidenceState.PARTIAL:
        missing_topics = set(bundle.missing_claim_topics)
    else:
        missing_topics = set()

    for block in draft.answer_blocks:
        if block.block_id in report.removed_block_ids:
            continue
        block_ok = True
        block_failures: list[str] = []

        # A factual block must have at least one mapped claim; otherwise the
        # block carries an unguarded factual assertion.
        if block.kind == "factual" and not block.claim_ids:
            report.removed_block_ids.append(block.block_id)
            report.limitations.append(f"removed:{block.block_id}:uncited_fact")
            continue

        for claim_id in block.claim_ids:
            claim = claims_by_id.get(claim_id)
            if claim is None:
                block_ok = False
                block_failures.append("unmapped_claim")
                continue
            # missing topics in a partial bundle cannot appear in a retained block
            if missing_topics and any(topic in (claim.text or "") for topic in missing_topics):
                block_ok = False
                block_failures.append("missing_topic")
                continue
            if block.kind in {"factual", "inference"}:
                # Material factual claims and inference premises require Evidence.
                if block.kind == "factual" and not claim.evidence_ids:
                    block_ok = False
                    block_failures.append("uncited_fact")
                    continue
                if block.kind == "inference" and not _has_inferential_wording(claim.text):
                    block_ok = False
                    block_failures.append("inference_mapping")
                    continue
                for evidence_id in claim.evidence_ids:
                    if not _evidence_exists(evidence_id, bundle):
                        block_ok = False
                        block_failures.append("missing_evidence")
                        break
                    if not _has_http_final_url(evidence_id, bundle):
                        block_ok = False
                        block_failures.append("invalid_url")
                        break

        # strip numeric citations deterministically
        if block_ok:
            cleaned_text = _strip_numeric_citations(block.text)
            cleaned_claims: list[Claim] = []
            for claim_id in block.claim_ids:
                claim = claims_by_id[claim_id]
                cleaned_claims.append(Claim(
                    claim.claim_id,
                    claim.block_id,
                    _strip_numeric_citations(claim.text),
                    claim.material,
                    tuple(_strip_numeric_citations(e) for e in claim.evidence_ids),
                ))
            report.kept_blocks.append(AnswerBlock(block.block_id, block.kind, cleaned_text, block.claim_ids))
            report.kept_claims.extend(cleaned_claims)
        else:
            report.removed_block_ids.append(block.block_id)
            report.limitations.extend(f"removed:{block.block_id}:{failure}" for failure in block_failures)


def _has_inferential_wording(text: str) -> bool:
    lowered = str(text or "").casefold()
    return any(marker in lowered for marker in ("根据", "推测", "推断", "由此", "可能", "indicates", "suggests", "推测"))


def _strip_numeric_citations(text: str) -> str:
    return _NUMERIC_CITATION.sub("", str(text or "")).strip()


# ── model-assisted semantic validation ─────────────────────────────────

def _semantic_verify(
    draft: GroundedDraft,
    bundle: EvidenceBundle,
    verifier: Any,
    report: _StructuralReport,
) -> None:
    claims = list(report.kept_claims)
    if not claims:
        return
    verdict = verifier.verify(
        {"draft": _draft_to_dict(draft), "evidence": _evidence_to_dict(bundle)},
    )

    if not isinstance(verdict, dict):
        verdict = {}

    labels: dict[str, SupportLabel] = {}
    for claim in claims:
        label = _parse_label(verdict.get(claim.claim_id))
        labels[claim.claim_id] = label

    removed: set[str] = set(report.removed_block_ids)
    kept_blocks: list[AnswerBlock] = []
    for block in report.kept_blocks:
        block_bad = False
        for claim_id in block.claim_ids:
            label = labels.get(claim_id)
            if label in {
                SupportLabel.PARTIAL,
                SupportLabel.CONFLICT,
                SupportLabel.UNSUPPORTED,
                SupportLabel.UNMAPPED,
            }:
                block_bad = True
                report.limitations.append(f"removed:{block.block_id}:{label.value}")
        if block_bad:
            removed.add(block.block_id)
        else:
            kept_blocks.append(block)

    report.labels = labels
    report.kept_blocks = kept_blocks
    report.removed_block_ids = sorted(removed)
    report.kept_claims = [
        claim for claim in report.kept_claims if claim.block_id not in removed
    ]


def _parse_label(value: Any) -> SupportLabel:
    if isinstance(value, SupportLabel):
        return value
    try:
        return SupportLabel(str(value))
    except ValueError:
        return SupportLabel.UNSUPPORTED


def _draft_to_dict(draft: GroundedDraft) -> dict[str, Any]:
    return {
        "answer_blocks": [
            {"block_id": b.block_id, "kind": b.kind, "text": b.text, "claim_ids": list(b.claim_ids)}
            for b in draft.answer_blocks
        ],
        "claims": [
            {"claim_id": c.claim_id, "block_id": c.block_id, "text": c.text, "material": c.material, "evidence_ids": list(c.evidence_ids)}
            for c in draft.claims
        ],
    }


def _evidence_to_dict(bundle: EvidenceBundle) -> dict[str, Any]:
    return {
        "evidence_items": [
            {
                "evidence_id": item.evidence_id,
                "title": item.title,
                "url": item.url,
                "excerpt": (item.excerpt or "")[:500],
                "source_relation": item.source_relation.value,
            }
            for item in bundle.evidence_items
        ]
    }


class SemanticVerificationUnavailable(RuntimeError):
    """Raised when the semantic verifier cannot run (e.g. provider failure)."""


def validate_and_filter(
    draft: GroundedDraft,
    bundle: EvidenceBundle,
    decision: Any,
    *,
    claim_discoverer: Any,
    semantic_verifier: Any,
) -> ValidationReport:
    report = _StructuralReport()
    _apply_structural_checks(draft, bundle, report)

    # Run claim discovery over the entire draft. Any material external-fact
    # span that was not already mapped to a Claim marks its block uncovered.
    discovered = _discover_factual_spans(claim_discoverer, draft, bundle)
    _apply_discovered_spans(draft, bundle, report, discovered)

    verifier_unavailable = False
    try:
        _semantic_verify(draft, bundle, semantic_verifier, report)
    except SemanticVerificationUnavailable:
        verifier_unavailable = True
        _apply_verifier_unavailable(draft, bundle, decision, report)
    except Exception:
        verifier_unavailable = True
        _apply_verifier_unavailable(draft, bundle, decision, report)

    # non_factual blocks may omit claims only when discovery finds no factual span
    for block in draft.answer_blocks:
        if block.kind == "non_factual" and block.block_id not in report.removed_block_ids:
            if not block.claim_ids and block.text.strip():
                report.kept_blocks.append(block)

    limitations = list(report.limitations)
    if verifier_unavailable:
        limitations.append("semantic_verification_unavailable")

    return ValidationReport(
        draft=draft,
        retained_blocks=tuple(_dedupe_blocks(report.kept_blocks)),
        retained_claims=tuple(_dedupe_claims(report.kept_claims)),
        removed_block_ids=tuple(_dedupe(report.removed_block_ids)),
        claim_labels=dict(report.labels),
        limitations=tuple(limitations),
    )


def _discover_factual_spans(claim_discoverer: Any, draft: GroundedDraft, bundle: EvidenceBundle) -> tuple[str, ...]:
    """Return the block_ids that contain material external-fact spans the draft
    failed to map to a Claim. The discoverer's output is advisory: it only flags
    uncovered spans; it never invents claims or Evidence IDs."""
    try:
        spans = claim_discoverer.discover(draft, bundle)
    except Exception:
        return ()
    if not isinstance(spans, (list, tuple)):
        return ()
    covered: set[str] = set()
    for claim in draft.claims:
        covered.add(claim.block_id)
    flagged: list[str] = []
    for span in spans:
        if not isinstance(span, str) or not span.strip():
            continue
        # A discovered factual span must live in a block that has no mapped claim.
        for block in draft.answer_blocks:
            if block.block_id in covered:
                continue
            if block.kind == "non_factual" and span in (block.text or ""):
                flagged.append(block.block_id)
    return tuple(dict.fromkeys(flagged))


def _apply_discovered_spans(
    draft: GroundedDraft,
    bundle: EvidenceBundle,
    report: _StructuralReport,
    discovered: Sequence[str],
) -> None:
    del bundle
    if not discovered:
        return
    for block in draft.answer_blocks:
        if block.block_id in discovered and block.block_id not in report.removed_block_ids:
            report.removed_block_ids.append(block.block_id)
            report.limitations.append(f"removed:{block.block_id}:hidden_fact")


def _apply_verifier_unavailable(
    draft: GroundedDraft,
    bundle: EvidenceBundle,
    decision: Any,
    report: _StructuralReport,
) -> None:
    del draft, bundle
    route = getattr(decision, "route", None)
    # Deep dynamic/high-consequence output becomes non-definitive: remove every
    # factual/inference block. Lower tiers keep structurally mapped blocks but the
    # report carries a fixed "semantic verification unavailable" disclosure.
    if route is SearchTier.DEEP:
        for block in list(report.kept_blocks):
            if block.kind in {"factual", "inference"}:
                report.removed_block_ids.append(block.block_id)
                report.kept_blocks.remove(block)
        report.kept_claims = [
            claim for claim in report.kept_claims if claim.block_id not in report.removed_block_ids
        ]
    else:
        for block in report.kept_blocks:
            for claim_id in block.claim_ids:
                report.labels[claim_id] = SupportLabel.UNMAPPED


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _dedupe_blocks(blocks: Sequence[AnswerBlock]) -> list[AnswerBlock]:
    seen: set[str] = set()
    result: list[AnswerBlock] = []
    for block in blocks:
        if block.block_id not in seen:
            seen.add(block.block_id)
            result.append(block)
    return result


def _dedupe_claims(claims: Sequence[Claim]) -> list[Claim]:
    seen: set[str] = set()
    result: list[Claim] = []
    for claim in claims:
        if claim.claim_id not in seen:
            seen.add(claim.claim_id)
            result.append(claim)
    return result
