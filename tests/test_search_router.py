"""Router tests: factual-default retrieval with program-owned floors."""

from __future__ import annotations

import importlib
import importlib.util
import unittest
from dataclasses import dataclass, field
from typing import Any

from src.search.models import (
    RequestSource,
    RetrievalRequest,
    SearchTier,
    SkipReason,
    TriggerCode,
)
from src.services.llm_types import ChatResponse
from tests.search_fakes import StaticRouterAdvisor


def router_module():
    try:
        return importlib.import_module("src.search.router")
    except ModuleNotFoundError:
        raise AssertionError("src.search.router must exist") from None


NEUTRAL = {
    "skip_candidate": None,
    "benefit_dimensions": ["accuracy"],
    "factuality": "factual",
    "external_fact_required": True,
    "freshness": "none",
    "risk": "low",
    "actionability": "none",
    "potential_harm": "none",
    "recommended_tier": "light",
    "trigger_codes": ["factual_default"],
}


def chat_request(question: str, *, force_search: bool = False) -> RetrievalRequest:
    return RetrievalRequest(
        question,
        force_search=force_search,
        request_source=RequestSource.CHAT,
    )


def decide(question, advisor_payload=NEUTRAL, **kwargs):
    module = router_module()
    assert module is not None, "src.search.router must exist"
    router = module.RetrievalBenefitRouter(StaticRouterAdvisor(advisor_payload))
    return router.decide(chat_request(question, **kwargs))


@dataclass
class _RecordingAdvisor:
    """Advisor that records every call for privacy assertions."""

    calls: list[Any] = field(default_factory=list)
    payload: Any = field(default_factory=lambda: dict(NEUTRAL))

    def advise(self, request: Any) -> Any:
        self.calls.append(request)
        return self.payload


@dataclass
class _FakeRoutingLLM:
    content: str = ""
    calls: list[Any] = field(default_factory=list)
    raise_error: Exception | None = None

    def chat(self, messages: list[Any], **kwargs: Any) -> ChatResponse:
        self.calls.append((messages, kwargs))
        if self.raise_error is not None:
            raise self.raise_error
        return ChatResponse(content=self.content, tool_calls=[])


class RouterTableTests(unittest.TestCase):
    """The fixed routing fixture rows from the plan."""

    def setUp(self) -> None:
        self.module = router_module()

    def _expect_skip(self, question, reason):
        decision = decide(question)
        self.assertIs(decision.route, SearchTier.SKIP)
        self.assertIs(decision.skip_reason, reason)

    def test_social_emotional_is_skip(self):
        self._expect_skip("你好，今天心情有点差", SkipReason.SOCIAL_OR_EMOTIONAL)

    def test_text_transform_is_skip(self):
        self._expect_skip("把这段话润色一下：天气很好", SkipReason.PROVIDED_TEXT_TRANSFORM)

    def test_explain_pasted_text_is_closed_context(self):
        self._expect_skip("解释我刚才贴的这段文字", SkipReason.CLOSED_CONTEXT_ONLY)

    def test_pure_math_is_skip(self):
        self._expect_skip("证明若 n 为偶数则 n² 为偶数", SkipReason.PURE_MATH)

    def test_stable_fact_is_light(self):
        decision = decide("什么是光合作用")
        self.assertIs(decision.route, SearchTier.LIGHT)
        self.assertIsNone(decision.skip_reason)

    def test_rust_go_comparison_is_at_least_standard(self):
        decision = decide("Rust 和 Go 的并发模型有什么区别")
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.STANDARD))

    def test_explain_own_stiff_sentence_is_closed_context(self):
        self._expect_skip("为什么我刚才那句话显得生硬", SkipReason.CLOSED_CONTEXT_ONLY)

    def test_what_is_stock_is_standard(self):
        decision = decide("什么是股票")
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.STANDARD))

    def test_should_i_buy_this_stock_is_deep(self):
        decision = decide("我现在是否应该买这只股票")
        self.assertIs(decision.route, SearchTier.DEEP)

    def test_what_is_civil_law_is_standard(self):
        decision = decide("什么是民法")
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.STANDARD))

    def test_is_my_situation_illegal_is_deep(self):
        decision = decide("我的具体情况是否违法")
        self.assertIs(decision.route, SearchTier.DEEP)

    def test_beijing_news_today_is_deep(self):
        decision = decide("北京今天有什么新闻")
        self.assertIs(decision.route, SearchTier.DEEP)

    def test_explicit_search_with_source_is_forced_standard(self):
        decision = decide("请搜索并给出来源：CRAG 是什么")
        self.assertTrue(decision.forced_search)
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.STANDARD))

    def test_no_web_summary_skips_with_zero_eligibility(self):
        decision = decide("不要联网，只根据我贴的内容总结")
        self.assertIs(decision.route, SearchTier.SKIP)
        self.assertIs(decision.skip_reason, SkipReason.USER_FORBID_WEB)
        self.assertFalse(decision.forced_search)
        self.assertNotIn(TriggerCode.EXPLICIT_SEARCH, decision.trigger_codes)
        self.assertIsNone(decision.program_minimum_tier)

    def test_conflicting_web_no_web_is_clarification(self):
        decision = decide("请联网查，但不要联网")
        self.assertIs(decision.route, SearchTier.SKIP)
        self.assertIs(decision.skip_reason, SkipReason.USER_FORBID_WEB)
        self.assertTrue(decision.requires_clarification)
        self.assertIn(TriggerCode.EXPLICIT_NO_WEB, decision.trigger_codes)
        self.assertIn(TriggerCode.EXPLICIT_SEARCH, decision.trigger_codes)

    def test_polish_price_intro_with_verification_searches(self):
        decision = decide("润色这段 GPT 价格介绍，并核实价格")
        self.assertNotEqual(decision.route, SearchTier.SKIP)
        self.assertIsNone(decision.skip_reason)
        self.assertTrue(decision.forced_search)

    def test_unknown_entity_is_light(self):
        decision = decide("什么是 QXZ-9 型反应堆冷却剂")
        self.assertIsNot(decision.route, SearchTier.SKIP)
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.LIGHT))

    def test_force_search_request_is_forced(self):
        decision = decide("什么是光合作用", force_search=True)
        self.assertTrue(decision.forced_search)
        self.assertIs(decision.route, SearchTier.LIGHT)


class RouterConflictAndFloorTests(unittest.TestCase):
    """C2 regression: forced search + no-web conflict, mixed floors, advisor failure."""

    def setUp(self) -> None:
        self.module = router_module()

    def test_force_search_plus_no_web_is_clarification_not_error(self):
        decision = decide("不要联网", force_search=True)
        self.assertIs(decision.route, SearchTier.SKIP)
        self.assertIs(decision.skip_reason, SkipReason.USER_FORBID_WEB)
        self.assertTrue(decision.requires_clarification)

    def test_mixed_social_plus_news_cannot_skip(self):
        decision = decide("今天心情不错，但北京今天有什么新闻")
        self.assertNotEqual(decision.route, SearchTier.SKIP)
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.DEEP))

    def test_proof_plus_current_version_cannot_skip(self):
        decision = decide("证明一下，顺便看看最新版本是多少")
        self.assertNotEqual(decision.route, SearchTier.SKIP)

    def test_advisor_failure_with_high_consequence_domain_is_not_light(self):
        decision = decide("这种药每天吃多少剂量", NEUTRAL)
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.STANDARD))

    def test_advisor_failure_floors_actionable_medical_and_chemical_requests_at_deep(self):
        # Removing the conservative safety signal must make these fall back to
        # light when the advisor returns no usable classification.
        cases = (
            "我发烧39度，该吃多少布洛芬？",
            "我胸口疼得厉害怎么办？",
            "这个清洁剂溅进眼睛怎么办？",
        )
        for question in cases:
            with self.subTest(question=question):
                decision = decide(question, {})
                self.assertIs(decision.route, SearchTier.DEEP)

    def test_advisor_failure_keeps_stable_regulated_definitions_standard(self):
        # These are concepts, not personal or urgent action requests. A domain
        # word alone must not promote them to deep.
        for question in ("什么是疫苗？", "什么是民法？", "什么是基金？"):
            with self.subTest(question=question):
                decision = decide(question, {})
                self.assertIs(decision.route, SearchTier.STANDARD)

    def test_stable_version_definition_is_not_deep(self):
        decision = decide("什么是版本控制")
        self.assertNotEqual(decision.route, SearchTier.DEEP)

    def test_current_version_question_is_deep(self):
        decision = decide("最新版本是多少")
        self.assertIs(decision.route, SearchTier.DEEP)


def _tier_index(tier: SearchTier) -> int:
    return {SearchTier.SKIP: 0, SearchTier.LIGHT: 1, SearchTier.STANDARD: 2, SearchTier.DEEP: 3}[tier]


class RouterProgramFloorTests(unittest.TestCase):
    """Program floors and model-can-only-raise guarantees."""

    def setUp(self) -> None:
        self.module = router_module()

    def test_model_claiming_knowledge_cannot_skip(self):
        payload = {
            **NEUTRAL,
            "skip_candidate": {"reason": "social_or_emotional"},
            "trigger_codes": [],
        }
        decision = decide("什么是光合作用", payload)
        self.assertNotEqual(decision.route, SearchTier.SKIP)

    def test_high_confidence_cannot_lower_route(self):
        payload = {**NEUTRAL, "recommended_tier": "light"}
        decision = decide("Rust 和 Go 的并发模型有什么区别", payload)
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.STANDARD))

    def test_common_knowledge_claim_cannot_skip(self):
        payload = {**NEUTRAL, "skip_candidate": {"reason": "closed_context_only"}}
        decision = decide("北京今天有什么新闻", payload)
        self.assertNotEqual(decision.route, SearchTier.SKIP)

    def test_unknown_skip_reason_falls_back_to_light(self):
        payload = {**NEUTRAL, "skip_candidate": {"reason": "model_knows_it"}}
        decision = decide("什么是光合作用", payload)
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.LIGHT))

    def test_invalid_advisor_json_is_light(self):
        for bad in (None, "not json", {}, {"factuality": "made_up"}, {"recommended_tier": "skip"}):
            with self.subTest(bad=bad):
                decision = decide("什么是光合作用", bad)
                self.assertIsNot(decision.route, SearchTier.SKIP)
                self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.LIGHT))

    def test_memory_conflict_claim_cannot_create_skip(self):
        payload = {**NEUTRAL, "skip_candidate": {"reason": "closed_context_only"}, "trigger_codes": ["explicit_no_web"]}
        decision = decide("什么是光合作用", payload)
        self.assertNotEqual(decision.route, SearchTier.SKIP)

    def test_model_cannot_downgrade_deep(self):
        payload = {**NEUTRAL, "recommended_tier": "light"}
        decision = decide("我现在是否应该买这只股票", payload)
        self.assertIs(decision.route, SearchTier.DEEP)

    def test_model_can_upgrade_light(self):
        payload = {**NEUTRAL, "recommended_tier": "deep"}
        decision = decide("什么是光合作用", payload)
        self.assertIs(decision.route, SearchTier.DEEP)

    def test_forced_search_conflict_never_calls_provider_route(self):
        decision = decide("请联网查，但不要联网", force_search=True)
        self.assertIs(decision.route, SearchTier.SKIP)
        self.assertTrue(decision.requires_clarification)


class AdversarialAdvisorTests(unittest.TestCase):
    """Model outputs claiming knowledge/confidence/memory cannot skip or lower."""

    def setUp(self) -> None:
        self.module = router_module()

    def _adversarial(self, reason):
        return {**NEUTRAL, "skip_candidate": {"reason": reason}, "trigger_codes": []}

    def test_i_know_this_cannot_skip(self):
        decision = decide("什么是光合作用", self._adversarial("closed_context_only"))
        self.assertIsNot(decision.route, SearchTier.SKIP)

    def test_high_confidence_cannot_skip(self):
        decision = decide("什么是民法", self._adversarial("closed_context_only"))
        self.assertIsNot(decision.route, SearchTier.SKIP)
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.STANDARD))

    def test_common_knowledge_cannot_skip(self):
        decision = decide("什么是光合作用", self._adversarial("social_or_emotional"))
        self.assertIsNot(decision.route, SearchTier.SKIP)

    def test_unknown_skip_reason_is_light(self):
        decision = decide("什么是光合作用", self._adversarial("model_knows_it"))
        self.assertIsNot(decision.route, SearchTier.SKIP)
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.LIGHT))

    def test_lower_tier_recommendation_is_rejected(self):
        decision = decide("我现在是否应该买这只股票", {**NEUTRAL, "recommended_tier": "light"})
        self.assertIs(decision.route, SearchTier.DEEP)

    def test_memory_conflict_claim_cannot_create_skip_or_conflict(self):
        payload = {
            **NEUTRAL,
            "skip_candidate": {"reason": "closed_context_only"},
            "trigger_codes": ["controversy_or_conflict"],
        }
        decision = decide("什么是光合作用", payload)
        self.assertIsNot(decision.route, SearchTier.SKIP)
        self.assertGreaterEqual(_tier_index(decision.route), _tier_index(SearchTier.LIGHT))


class AdvisorPrivacyTests(unittest.TestCase):
    """The advisor capture never receives memory/history/private identifiers."""

    def setUp(self) -> None:
        self.module = router_module()

    def test_llm_advisor_receives_only_question_and_schemas(self):
        llm = _FakeRoutingLLM(content='{"skip_candidate": null}')
        advisor = self.module.LLMRoutingAdvisor(llm)
        request = chat_request("什么是光合作用")
        advisor.advise(request)
        self.assertEqual(len(llm.calls), 1)
        messages = llm.calls[0][0]
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        self.assertIn("什么是光合作用", joined)
        for forbidden in ("记忆", "memory", "chat_history", "历史", "QQ", "group_id", "data:image", "回调", "callback"):
            self.assertNotIn(forbidden, joined)

    def test_static_advisor_capture_has_no_private_identifiers(self):
        advisor = _RecordingAdvisor()
        router = self.module.RetrievalBenefitRouter(advisor)
        router.decide(chat_request("什么是光合作用"))
        self.assertEqual(len(advisor.calls), 1)
        self.assertEqual(advisor.calls[0].question, "什么是光合作用")
        self.assertFalse(advisor.calls[0].has_images)
        # The advisor only sees the request, never memory or history.
        self.assertFalse(hasattr(advisor.calls[0], "memory"))
        self.assertFalse(hasattr(advisor.calls[0], "history"))


class LLMRoutingAdvisorTests(unittest.TestCase):
    """Strict JSON parsing and closed-enum validation."""

    def setUp(self) -> None:
        self.module = router_module()

    def _advise(self, content):
        llm = _FakeRoutingLLM(content=content)
        advisor = self.module.LLMRoutingAdvisor(llm)
        return advisor.advise(chat_request("什么是光合作用"))

    def test_parses_valid_json(self):
        result = self._advise('{"skip_candidate": null, "benefit_dimensions": ["accuracy"], "factuality": "mixed", "external_fact_required": true, "freshness": "low", "risk": "low", "actionability": "none", "potential_harm": "none", "recommended_tier": "standard", "trigger_codes": ["external_fact_explanation_or_comparison"]}')
        self.assertEqual(result["recommended_tier"], SearchTier.STANDARD)
        self.assertEqual(result["factuality"], __import__("src.search.models", fromlist=["Factuality"]).Factuality.MIXED)

    def test_parses_fenced_json(self):
        result = self._advise('```json\n{"skip_candidate": null}\n```')
        self.assertIsNone(result["skip_candidate"])

    def test_invalid_json_is_empty(self):
        result = self._advise("I think the answer is 42")
        self.assertEqual(result, {})

    def test_unknown_enum_is_empty(self):
        result = self._advise('{"factuality": "certainly_true"}')
        self.assertEqual(result, {})

    def test_recommending_skip_is_empty(self):
        result = self._advise('{"recommended_tier": "skip"}')
        self.assertEqual(result, {})

    def test_advisor_exception_is_empty(self):
        llm = _FakeRoutingLLM(raise_error=RuntimeError("boom"))
        advisor = self.module.LLMRoutingAdvisor(llm)
        self.assertEqual(advisor.advise(chat_request("什么是光合作用")), {})


if __name__ == "__main__":
    unittest.main()
