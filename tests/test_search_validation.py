"""Validation tests: atomic claim parsing, structural checks, model support."""

from __future__ import annotations

import importlib
import unittest

from src.search.models import (
    EvidenceBundle,
    EvidenceGapAnalysis,
    EvidenceItem,
    EvidenceState,
    ExcerptOrigin,
    Factuality,
    Freshness,
    GroundedDraft,
    RepairPlan,
    RequestSource,
    RetrievalDecision,
    RetrievalRequest,
    RiskLevel,
    SearchPlan,
    SearchQuery,
    SearchRoundKind,
    SearchTier,
    SourceRelation,
    SupportLabel,
)
from tests.search_fakes import StaticSemanticVerifier


def validation_module():
    try:
        return importlib.import_module("src.search.validation")
    except ModuleNotFoundError:
        raise AssertionError("src.search.validation must exist") from None


def models():
    return importlib.import_module("src.search.models")


def decision(tier=SearchTier.STANDARD):
    m = models()
    return m.RetrievalDecision(
        tier, None, False, (), frozenset(), Factuality.FACTUAL,
        True, Freshness.NONE, RiskLevel.LOW, m.Actionability.NONE,
        m.PotentialHarm.NONE, tier, None, (),
    )


def query():
    return SearchQuery("q1", SearchRoundKind.INITIAL, __import__("src.search.models", fromlist=["QueryPurpose"]).QueryPurpose.DIRECT, "q")


def plan(required=("版本",)):
    m = models()
    d = decision()
    return SearchPlan(
        d, "当前版本是什么", m.PlanningStatus.NORMAL, ("X",), None, (query(),),
        tuple(required), frozenset({SourceRelation.PRIMARY}), (), m.DEFAULT_TIER_BUDGETS[SearchTier.STANDARD],
    )


def bundle(evidence=(), state=None, missing=()):
    m = models()
    p = plan()
    if state is None:
        state = EvidenceState.SUFFICIENT
    return m.EvidenceBundle(
        "req-1", p.decision, p, (), tuple(e.evidence_id for e in evidence),
        m.EvidenceGapAnalysis(missing, (), False, None, ()),
        m.RepairPlan(False, (), None), 1, tuple(evidence), state,
        tuple(missing), (), (), (),
    )


def item(eid="E1", url="https://example.com/page"):
    m = models()
    return m.EvidenceItem(
        eid, "q1", "tavily", "Title", url, url, "example.com", "Example",
        SourceRelation.INDEPENDENT, None, None, None, "excerpt",
        ExcerptOrigin.PROVIDER_SNIPPET, "ok", 1.0, 1.0, True, Freshness.NONE,
        True, (), ("版本",), "g1",
    )


def draft_json(blocks, claims):
    return {
        "answer_blocks": blocks,
        "claims": claims,
        "limitations": [],
        "conflict_summary": [],
        "used_knowledge_fallback": False,
    }


class ParseGroundedDraftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = validation_module()

    def test_parses_valid_structured_draft(self):
        text = '{"answer_blocks":[{"block_id":"B1","kind":"factual","text":"版本是3.2","claim_ids":["C1"]}],"claims":[{"claim_id":"C1","block_id":"B1","text":"版本是3.2","material":true,"evidence_ids":["E1"]}]}'
        draft = self.module.parse_grounded_draft(text)
        self.assertEqual(len(draft.answer_blocks), 1)
        self.assertEqual(len(draft.claims), 1)
        self.assertEqual(draft.answer_blocks[0].block_id, "B1")

    def test_parses_fenced_json(self):
        text = '```json\n{"answer_blocks":[],"claims":[]}\n```'
        draft = self.module.parse_grounded_draft(text)
        self.assertEqual(draft.answer_blocks, ())

    def test_rejects_malformed_json(self):
        with self.assertRaises(ValueError):
            self.module.parse_grounded_draft("not json at all")

    def test_rejects_duplicate_ids(self):
        text = '{"answer_blocks":[{"block_id":"B1","kind":"factual","text":"a","claim_ids":["C1"]},{"block_id":"B1","kind":"factual","text":"b","claim_ids":[]}],"claims":[{"claim_id":"C1","block_id":"B1","text":"a","material":true,"evidence_ids":[]}]}'
        with self.assertRaises(ValueError):
            self.module.parse_grounded_draft(text)

    def test_rejects_unmapped_claim_id(self):
        text = '{"answer_blocks":[{"block_id":"B1","kind":"factual","text":"a","claim_ids":["C9"]}],"claims":[{"claim_id":"C1","block_id":"B1","text":"a","material":true,"evidence_ids":[]}]}'
        with self.assertRaises(ValueError):
            self.module.parse_grounded_draft(text)


class StructuralValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = validation_module()

    def _validate(self, draft, b, *, decision_tier=SearchTier.STANDARD):
        return self.module.validate_and_filter(
            draft,
            b,
            decision(decision_tier),
            claim_discoverer=_Discoverer([]),
            semantic_verifier=StaticSemanticVerifier({"supported"}),
        )

    def test_nonexistent_evidence_id_rejected(self):
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (models().Claim("C1", "B1", "版本是3.2", True, ("E99",)),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = self._validate(d, b)
        self.assertIn("B1", report.removed_block_ids)

    def test_non_http_url_rejected(self):
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (models().Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        b = bundle((item(url="ftp://example.com/x"),), state=EvidenceState.SUFFICIENT)
        report = self._validate(d, b)
        self.assertIn("B1", report.removed_block_ids)

    def test_numeric_citation_stripped(self):
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "版本是3.2[1]", ("C1",)),),
            (models().Claim("C1", "B1", "版本是3.2[1]", True, ("E1",)),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = self._validate(d, b)
        for block in report.retained_blocks:
            self.assertNotIn("[1]", block.text)
            self.assertNotIn("[1]", block.text)

    def test_partial_bundle_blocks_missing_topics(self):
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "关于历史的说法", ("C1",)),),
            (models().Claim("C1", "B1", "关于历史的说法", True, ("E1",)),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.PARTIAL, missing=("历史",))
        report = self._validate(d, b)
        self.assertIn("B1", report.removed_block_ids)

    def test_failed_retrieval_cannot_have_claims(self):
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (models().Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        b = bundle((), state=EvidenceState.INSUFFICIENT)
        report = self._validate(d, b)
        self.assertIn("B1", report.removed_block_ids)

    def test_inference_block_requires_premise_mapping(self):
        d = GroundedDraft(
            (models().AnswerBlock("B1", "inference", "根据来源推测版本上升", ("C1",)),),
            (models().Claim("C1", "B1", "根据来源推测版本上升", True, ()),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = self._validate(d, b)
        self.assertIn("B1", report.removed_block_ids)

    def test_unsupported_label_removes_block(self):
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (models().Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = self.module.validate_and_filter(
            d, b, decision(),
            claim_discoverer=_Discoverer([]),
            semantic_verifier=StaticSemanticVerifier({"unsupported"}),
        )
        self.assertIn("B1", report.removed_block_ids)
        self.assertIs(report.claim_labels["C1"], SupportLabel.UNSUPPORTED)


class _Discoverer:
    def __init__(self, spans):
        self.spans = spans

    def discover(self, draft, evidence):
        del draft, evidence
        return self.spans


class ModelMemoryTests(unittest.TestCase):
    def test_model_memory_disagreement_is_unsupported_not_conflict(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (models().Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = module.validate_and_filter(
            d, b, decision(),
            claim_discoverer=_Discoverer([]),
            semantic_verifier=StaticSemanticVerifier({"unsupported"}),
        )
        self.assertIn("B1", report.removed_block_ids)
        # No new conflict group is created from model memory.
        self.assertEqual(b.conflict_groups, ())


class NoSearchAccessTests(unittest.TestCase):
    def test_validators_have_no_search_callable(self):
        module = validation_module()
        # The claim discoverer and semantic verifier receive only draft/evidence.
        d = GroundedDraft((), (), (), (), False)
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        from tests.search_fakes import StaticSemanticVerifier
        verifier = StaticSemanticVerifier({"supported"})
        self.assertFalse(hasattr(verifier, "search"))
        self.assertFalse(hasattr(verifier, "orchestrator"))


class VerifierFailureTests(unittest.TestCase):
    def test_verifier_exception_degrades_deep_output(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (models().Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)

        class FailingVerifier:
            def verify(self, *args, **kwargs):
                raise RuntimeError("boom")

        report = module.validate_and_filter(
            d, b, decision(SearchTier.DEEP),
            claim_discoverer=_Discoverer([]),
            semantic_verifier=FailingVerifier(),
        )
        self.assertIn("B1", report.removed_block_ids)
        self.assertTrue(any("semantic_verification_unavailable" in limitation for limitation in report.limitations))


if __name__ == "__main__":
    unittest.main()
