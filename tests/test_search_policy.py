"""Answer policy maps immutable search state into a bounded answer state."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.search.models import (
    AllowedClaimScope,
    AnswerCertainty,
    AnswerGenerationMode,
    DisclosureCode,
    EvidenceState,
    FreshnessRequirement,
    SearchFailureCode,
    SkipReason,
    ValidatorRequirement,
    WarningCode,
)
from src.search.policy import decide_answer_state
from tests.test_chat_retrieval_flow import models


def _evidence(state):
    return SimpleNamespace(evidence_state=state)


def _conflict_evidence(supported=(), conflicts=()):
    plan = SimpleNamespace(
        required_topics=(
            SimpleNamespace(topic_id="topic-1", material=True, label="定义"),
        ),
    )
    return SimpleNamespace(
        evidence_state=EvidenceState.CONFLICTING,
        plan=plan,
        supported_topic_ids=supported,
        evidence_items=(),
        conflicts=conflicts,
    )


def make_analysis(
    *,
    skip_reason=None,
    high_consequence=False,
    fail_closed=False,
    freshness=FreshnessRequirement.NOT_REQUIRED,
):
    m = models()
    return m.RequestAnalysis(
        m.RetrievalContext(
            must_search=skip_reason is None,
            skip_reason=skip_reason,
            factuality=(
                m.Factuality.NON_FACTUAL
                if skip_reason is not None
                else m.Factuality.FACTUAL
            ),
            external_fact_required=skip_reason is None,
            complexity_codes=(),
            source_requirement=m.SourceRequirement.ANY_RELEVANT,
        ),
        m.FreshnessContext(freshness, None, None, None, None),
        m.RiskContext(high_consequence, high_consequence, fail_closed),
    )


class AnswerStateMatrixTests(unittest.TestCase):
    def test_evidence_state_maps_to_certainty_and_scope(self):
        cases = (
            (EvidenceState.SUFFICIENT, AnswerCertainty.VERIFIED, AllowedClaimScope.ALL_SUPPORTED),
            (EvidenceState.PARTIAL, AnswerCertainty.LIMITED, AllowedClaimScope.SUPPORTED_SUBSET),
            (
                EvidenceState.INSUFFICIENT,
                AnswerCertainty.UNVERIFIED,
                AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS,
            ),
        )
        for state, certainty, scope in cases:
            with self.subTest(state=state):
                answer = decide_answer_state(
                    make_analysis(),
                    _evidence(state),
                    None,
                )
                self.assertIs(answer.certainty, certainty)
                self.assertIs(answer.allowed_claim_scope, scope)

    def test_conflicting_covers_every_core_topic_is_description_only(self):
        answer = decide_answer_state(
            make_analysis(),
            _conflict_evidence(),
            None,
        )
        self.assertIs(answer.certainty, AnswerCertainty.CONFLICTING)
        self.assertIs(
            answer.allowed_claim_scope,
            AllowedClaimScope.CONFLICT_DESCRIPTION_ONLY,
        )
        self.assertIn(DisclosureCode.SOURCE_CONFLICT, answer.disclosure_codes)


class AnswerStateRiskAndFreshnessTests(unittest.TestCase):
    def test_high_consequence_adds_exactly_one_warning(self):
        answer = decide_answer_state(
            make_analysis(high_consequence=True),
            _evidence(EvidenceState.SUFFICIENT),
            None,
        )
        self.assertEqual((WarningCode.HIGH_CONSEQUENCE,), answer.warning_codes)

    def test_ordinary_request_has_no_warning(self):
        answer = decide_answer_state(
            make_analysis(),
            _evidence(EvidenceState.SUFFICIENT),
            None,
        )
        self.assertEqual((), answer.warning_codes)

    def test_fail_closed_or_current_freshness_forces_fail_closed_validator(self):
        for kwargs in (
            {"fail_closed": True},
            {"freshness": FreshnessRequirement.CURRENT},
        ):
            with self.subTest(kwargs=kwargs):
                answer = decide_answer_state(
                    make_analysis(**kwargs),
                    _evidence(EvidenceState.SUFFICIENT),
                    None,
                )
                self.assertIs(
                    answer.validator_requirement,
                    ValidatorRequirement.FAIL_CLOSED,
                )


class AnswerStateSkipAndFailureTests(unittest.TestCase):
    def test_closed_task_skip_is_plain(self):
        answer = decide_answer_state(
            make_analysis(skip_reason=SkipReason.SOCIAL_OR_EMOTIONAL),
            None,
            None,
        )
        self.assertIs(answer.generation_mode, AnswerGenerationMode.PLAIN)
        self.assertEqual((), answer.disclosure_codes)

    def test_user_forbid_web_is_fixed_with_disclosure(self):
        answer = decide_answer_state(
            make_analysis(skip_reason=SkipReason.USER_FORBID_WEB),
            None,
            None,
        )
        self.assertIs(answer.generation_mode, AnswerGenerationMode.FIXED)
        self.assertEqual((DisclosureCode.USER_FORBID_WEB,), answer.disclosure_codes)
        self.assertIs(
            answer.allowed_claim_scope,
            AllowedClaimScope.NO_EXTERNAL_FACTUAL_CLAIMS,
        )

    def test_insufficient_failure_uses_online_verification_disclosure(self):
        answer = decide_answer_state(
            make_analysis(),
            _evidence(EvidenceState.INSUFFICIENT),
            SearchFailureCode.NO_RESULTS,
        )
        self.assertIs(answer.generation_mode, AnswerGenerationMode.FIXED)
        self.assertIs(answer.certainty, AnswerCertainty.UNVERIFIED)
        self.assertEqual(
            (DisclosureCode.ONLINE_VERIFICATION_FAILED,),
            answer.disclosure_codes,
        )


if __name__ == "__main__":
    unittest.main()
