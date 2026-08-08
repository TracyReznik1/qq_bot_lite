"""Validation tests: atomic claim parsing, structural checks, model support."""

from __future__ import annotations

import importlib
import json
import time
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
from src.services.llm_types import ChatResponse


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

    def test_rejects_cross_block_claim_ownership(self):
        text = json.dumps({
            "answer_blocks": [
                {"block_id": "B1", "kind": "factual", "text": "危险剂量是99毫克", "claim_ids": ["C1"]},
                {"block_id": "B2", "kind": "non_factual", "text": "请咨询专业人士", "claim_ids": []},
            ],
            "claims": [
                {"claim_id": "C1", "block_id": "B2", "text": "安全提醒", "material": True, "evidence_ids": ["E1"]},
            ],
        }, ensure_ascii=False)
        with self.assertRaises(ValueError):
            self.module.parse_grounded_draft(text)

    def test_rejects_claim_referenced_more_than_once(self):
        text = json.dumps({
            "answer_blocks": [
                {"block_id": "B1", "kind": "factual", "text": "版本是3.2", "claim_ids": ["C1", "C1"]},
            ],
            "claims": [
                {"claim_id": "C1", "block_id": "B1", "text": "版本是3.2", "material": True, "evidence_ids": ["E1"]},
            ],
        }, ensure_ascii=False)
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


class ClaimDiscoveryTests(unittest.TestCase):
    def test_hidden_fact_in_non_factual_block_is_removed(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "non_factual", "今天天气很好，顺带说一下版本是9.9", ()),),
            (), (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = module.validate_and_filter(
            d, b, decision(),
            claim_discoverer=_Discoverer(["版本是9.9"]),
            semantic_verifier=StaticSemanticVerifier({"supported"}),
        )
        self.assertIn("B1", report.removed_block_ids)
        self.assertNotIn("B1", {block.block_id for block in report.retained_blocks})
        self.assertFalse(
            set(report.removed_block_ids)
            & {block.block_id for block in report.retained_blocks}
        )

    def test_mapped_material_fact_in_non_factual_block_still_requires_evidence(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "non_factual", "顺带说一下版本是9.9", ("C1",)),),
            (models().Claim("C1", "B1", "版本是9.9", True, ()),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = module.validate_and_filter(
            d, b, decision(),
            claim_discoverer=_Discoverer([]),
            semantic_verifier=StaticSemanticVerifier({"C1": "supported"}),
        )
        self.assertIn("B1", report.removed_block_ids)
        self.assertNotIn("C1", {claim.claim_id for claim in report.retained_claims})

    def test_hidden_fact_is_scanned_in_factual_block_with_another_mapped_claim(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "版本是3.2，发布日期是明天", ("C1",)),),
            (models().Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = module.validate_and_filter(
            d, b, decision(),
            claim_discoverer=_Discoverer(["发布日期是明天"]),
            semantic_verifier=StaticSemanticVerifier({"C1": "supported"}),
        )
        self.assertIn("B1", report.removed_block_ids)
        self.assertNotIn("B1", {block.block_id for block in report.retained_blocks})
        self.assertNotIn("C1", {claim.claim_id for claim in report.retained_claims})

    def test_null_claim_id_is_uncovered_even_when_a_vague_claim_overlaps(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "发布日期是明天", ("C1",)),),
            (models().Claim("C1", "B1", "发布日期", True, ("E1",)),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = module.validate_and_filter(
            d, b, decision(),
            claim_discoverer=_Discoverer((
                module.DiscoveredClaimSpan("B1", "发布日期是明天", True, True, None),
            )),
            semantic_verifier=StaticSemanticVerifier({"C1": "supported"}),
        )
        self.assertIn("B1", report.removed_block_ids)
        self.assertNotIn("B1", {block.block_id for block in report.retained_blocks})

    def test_inference_without_explicit_premise_claim_is_removed(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "inference", "根据这些资料推测需求会上升", ()),),
            (), (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = module.validate_and_filter(
            d, b, decision(),
            claim_discoverer=_Discoverer([]),
            semantic_verifier=StaticSemanticVerifier({"supported"}),
        )
        self.assertIn("B1", report.removed_block_ids)
        self.assertNotIn("B1", {block.block_id for block in report.retained_blocks})

    def test_inference_with_supported_evidence_premise_is_retained(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "inference", "根据来源中的增长数据，推测需求可能上升", ("C1",)),),
            (models().Claim("C1", "B1", "来源记录了增长数据", True, ("E1",)),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = module.validate_and_filter(
            d, b, decision(),
            claim_discoverer=_Discoverer([]),
            semantic_verifier=StaticSemanticVerifier({"C1": "supported"}),
        )
        self.assertEqual(("B1",), tuple(block.block_id for block in report.retained_blocks))
        self.assertEqual(("C1",), tuple(claim.claim_id for claim in report.retained_claims))

    def test_validation_report_rejects_retained_removed_overlap(self):
        d = GroundedDraft(
            (models().AnswerBlock("B1", "non_factual", "安全提示", ()),),
            (), (), (), False,
        )
        with self.assertRaises(ValueError):
            models().ValidationReport(d, d.answer_blocks, (), ("B1",), {}, ())

    def test_factual_block_without_claims_is_rejected(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "版本是9.9", ()),),
            (), (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = module.validate_and_filter(
            d, b, decision(),
            claim_discoverer=_Discoverer([]),
            semantic_verifier=StaticSemanticVerifier({"supported"}),
        )
        self.assertIn("B1", report.removed_block_ids)

    def test_material_claim_requires_evidence(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "版本是9.9", ("C1",)),),
            (models().Claim("C1", "B1", "版本是9.9", True, ()),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        report = module.validate_and_filter(
            d, b, decision(),
            claim_discoverer=_Discoverer([]),
            semantic_verifier=StaticSemanticVerifier({"supported"}),
        )
        self.assertIn("B1", report.removed_block_ids)



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

    def test_production_discoverer_uses_bounded_closed_schema_without_search_or_memory(self):
        module = validation_module()

        class RecordingLLM:
            def __init__(self):
                self.calls = []

            def chat(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                return ChatResponse(content=json.dumps({
                    "spans": [{
                        "block_id": "B1",
                        "text": "版本是9.9",
                        "material": True,
                        "external_fact": True,
                        "claim_id": None,
                    }]
                }, ensure_ascii=False))

        llm = RecordingLLM()
        discoverer = module.LLMClaimDiscoverer(llm, max_tokens=321)
        d = GroundedDraft(
            (
                models().AnswerBlock("B1", "factual", "版本是9.9", ()),
                models().AnswerBlock("B2", "inference", "根据资料推测会上升", ()),
                models().AnswerBlock("B3", "non_factual", "请谨慎处理", ()),
            ),
            (), (), (), False,
        )
        discovered = discoverer.discover(d, bundle((item(),)))

        self.assertEqual("B1", discovered[0].block_id)
        self.assertEqual("版本是9.9", discovered[0].text)
        messages, kwargs = llm.calls[0]
        self.assertEqual(0.0, kwargs["temperature"])
        self.assertEqual(321, kwargs["max_tokens"])
        self.assertIsNone(kwargs["tools"])
        self.assertEqual("none", kwargs["tool_choice"])
        system = messages[0]["content"]
        self.assertIn("factual, inference, and non_factual", system)
        self.assertIn("all fields", system.lower())
        self.assertIn("claim_id", system)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(["factual", "inference", "non_factual"], [row["kind"] for row in payload["answer_blocks"]])
        serialized = json.dumps(payload, ensure_ascii=False).lower()
        self.assertNotIn("chat_history", serialized)
        self.assertNotIn("memory", serialized)
        self.assertFalse(hasattr(discoverer, "search"))
        self.assertFalse(hasattr(discoverer, "orchestrator"))

    def test_production_discoverer_fails_closed_on_malformed_or_exception(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "non_factual", "顺带说版本是9.9", ()),),
            (), (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)

        class BrokenLLM:
            def __init__(self, outcome):
                self.outcome = outcome

            def chat(self, *_args, **_kwargs):
                if isinstance(self.outcome, Exception):
                    raise self.outcome
                return ChatResponse(content=self.outcome)

        for outcome in ("not-json", RuntimeError("boom")):
            with self.subTest(outcome=type(outcome).__name__):
                with self.assertRaises(RuntimeError):
                    module.validate_and_filter(
                        d, b, decision(SearchTier.DEEP),
                        claim_discoverer=module.LLMClaimDiscoverer(BrokenLLM(outcome)),
                        semantic_verifier=StaticSemanticVerifier({}),
                    )

    def test_discovery_exception_records_attempted_semantic_time_and_parsed_claim_count(self):
        module = validation_module()
        d = GroundedDraft(
            (models().AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
            (models().Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
            (), (), False,
        )
        trace = models().SearchTrace("req-1", RequestSource.CHAT, SearchTier.DEEP)

        class DelayedFailingDiscoverer:
            def discover(self, _draft, _evidence):
                time.sleep(0.02)
                raise RuntimeError("boom")

        with self.assertRaises(module.ClaimDiscoveryUnavailable):
            module.validate_and_filter(
                d, bundle((item(),)), decision(SearchTier.DEEP),
                claim_discoverer=DelayedFailingDiscoverer(),
                semantic_verifier=StaticSemanticVerifier({}),
                trace=trace,
            )

        self.assertGreater(trace.semantic_validation_latency_ms, 0)
        self.assertEqual(1, trace.claim_count)
        self.assertEqual(0, trace.supported_claim_count)

    def test_production_discoverer_rejects_unsent_block_and_text_overflow(self):
        module = validation_module()
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)

        class EmptyLLM:
            def __init__(self):
                self.calls = 0

            def chat(self, *_args, **_kwargs):
                self.calls += 1
                return ChatResponse(content='{"spans":[]}')

        drafts = (
            GroundedDraft(
                tuple(
                    models().AnswerBlock(
                        f"B{index}", "non_factual",
                        "顺带说版本是9.9" if index == 41 else "请谨慎处理",
                        (),
                    )
                    for index in range(1, 42)
                ),
                (), (), (), False,
            ),
            GroundedDraft(
                (models().AnswerBlock("B1", "non_factual", "甲" * 1000 + "版本是9.9", ()),),
                (), (), (), False,
            ),
        )
        for d in drafts:
            with self.subTest(blocks=len(d.answer_blocks), chars=len(d.answer_blocks[0].text)):
                llm = EmptyLLM()
                with self.assertRaises(RuntimeError):
                    module.validate_and_filter(
                        d, b, decision(SearchTier.DEEP),
                        claim_discoverer=module.LLMClaimDiscoverer(llm),
                        semantic_verifier=StaticSemanticVerifier({}),
                    )
                self.assertEqual(0, llm.calls)

    def test_production_discoverer_rejects_output_overflow_and_invalid_claim_coverage(self):
        module = validation_module()
        d = GroundedDraft(
            (
                models().AnswerBlock("B1", "factual", "发布日期是明天", ("C1",)),
                models().AnswerBlock("B2", "non_factual", "请谨慎处理", ()),
            ),
            (models().Claim("C1", "B1", "发布日期", True, ("E1",)),),
            (), (), False,
        )
        b = bundle((item(),), state=EvidenceState.SUFFICIENT)
        invalid_payloads = (
            {"spans": [
                {"block_id": "B1", "text": "发布日期是明天", "material": True, "external_fact": True, "claim_id": None}
                for _ in range(81)
            ]},
            {"spans": [{
                "block_id": "B1", "text": "发布日期是明天", "material": True,
                "external_fact": True, "claim_id": "C1",
            }]},
            {"spans": [{
                "block_id": "B2", "text": "请谨慎处理", "material": True,
                "external_fact": True, "claim_id": "C1",
            }]},
        )

        class PayloadLLM:
            def __init__(self, payload):
                self.payload = payload

            def chat(self, *_args, **_kwargs):
                return ChatResponse(content=json.dumps(self.payload, ensure_ascii=False))

        for payload in invalid_payloads:
            with self.subTest(rows=len(payload["spans"])):
                with self.assertRaises(RuntimeError):
                    module.LLMClaimDiscoverer(PayloadLLM(payload)).discover(d, b)


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
