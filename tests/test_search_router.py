"""Router tests: factual-default retrieval with program-owned floors."""

from __future__ import annotations

import importlib
import importlib.util
import unittest
from dataclasses import dataclass, field
from typing import Any

from src.search.models import (
    Actionability,
    PotentialHarm,
    RequestSource,
    RetrievalRequest,
    RiskLevel,
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

    def test_no_web_high_consequence_keeps_deterministic_safety_signal(self):
        decision = decide("不要联网，我发烧39度，该吃多少布洛芬？", {})
        self.assertIs(decision.route, SearchTier.SKIP)
        self.assertIs(decision.skip_reason, SkipReason.USER_FORBID_WEB)
        self.assertIn(TriggerCode.EXPLICIT_NO_WEB, decision.trigger_codes)
        self.assertIn(TriggerCode.HIGH_CONSEQUENCE_ACTION, decision.trigger_codes)
        self.assertIs(decision.risk, RiskLevel.HIGH)
        self.assertIs(decision.actionability, Actionability.PERSONALIZED)
        self.assertIs(decision.potential_harm, PotentialHarm.HIGH)

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


class WarningSignalBoundaryTests(unittest.TestCase):
    def test_one_leg_sudden_weakness_is_high_consequence(self):
        decision = decide("一条腿突然没力气，要去急诊吗？", {})
        self.assertIs(decision.route, SearchTier.DEEP)
        self.assertIn(TriggerCode.HIGH_CONSEQUENCE_ACTION, decision.trigger_codes)
        self.assertIs(decision.potential_harm, PotentialHarm.HIGH)

    def test_meta_title_suffix_is_not_high_consequence(self):
        for label, advisor in (("empty", {}), ("neutral", NEUTRAL)):
            with self.subTest(advisor=label):
                decision = decide("一条腿突然没力气，要去急诊吗？这只是测试用例标题。", advisor)
                self.assertIs(decision.route, SearchTier.LIGHT)
                self.assertNotIn(TriggerCode.HIGH_CONSEQUENCE_ACTION, decision.trigger_codes)
                self.assertIs(decision.potential_harm, PotentialHarm.NONE)

    def test_negated_meta_title_suffix_keeps_high_consequence(self):
        for label, advisor in (("empty", {}), ("neutral", NEUTRAL)):
            with self.subTest(advisor=label):
                decision = decide("一条腿突然没力气，要去急诊吗？这不是测试用例标题。", advisor)
                self.assertIs(decision.route, SearchTier.DEEP)
                self.assertIn(TriggerCode.HIGH_CONSEQUENCE_ACTION, decision.trigger_codes)
                self.assertIs(decision.potential_harm, PotentialHarm.HIGH)

    def test_meta_title_suffix_keeps_later_emergency_active(self):
        decision = decide(
            "一条腿突然没力气，要去急诊吗？这只是测试用例标题。"
            "后来我真的一条腿突然没力气，要去急诊吗？",
            {},
        )
        self.assertIs(decision.route, SearchTier.DEEP)
        self.assertIn(TriggerCode.HIGH_CONSEQUENCE_ACTION, decision.trigger_codes)
        self.assertIs(decision.potential_harm, PotentialHarm.HIGH)

    def test_stable_or_negated_leg_text_is_not_high_consequence(self):
        for question in (
            "一条腿没力气可能有哪些一般原因",
            "我并没有一条腿突然没力气，只是在引用这个句子",
        ):
            with self.subTest(question=question):
                decision = decide(question, NEUTRAL)
                self.assertNotIn(
                    TriggerCode.HIGH_CONSEQUENCE_ACTION,
                    decision.trigger_codes,
                )


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

    def test_empty_and_malformed_advisor_keep_high_consequence_and_uncertain_codes(self):
        malformed_advisor = self.module.LLMRoutingAdvisor(_FakeRoutingLLM(content="not-json"))
        routers = (
            ("empty", self.module.RetrievalBenefitRouter(StaticRouterAdvisor({}))),
            ("malformed", self.module.RetrievalBenefitRouter(malformed_advisor)),
        )
        for label, router in routers:
            with self.subTest(advisor=label):
                decision = router.decide(chat_request("我发烧39度，该吃多少布洛芬？"))
                self.assertIs(decision.route, SearchTier.DEEP)
                self.assertIn(TriggerCode.HIGH_CONSEQUENCE_ACTION, decision.trigger_codes)
                self.assertIn(TriggerCode.CLASSIFIER_UNCERTAIN, decision.trigger_codes)

    def test_advisor_failure_keeps_stable_regulated_definitions_standard(self):
        # These are concepts, not personal or urgent action requests. A domain
        # word alone must not promote them to deep.
        for question in ("什么是疫苗？", "什么是民法？", "什么是基金？"):
            with self.subTest(question=question):
                decision = decide(question, {})
                self.assertIs(decision.route, SearchTier.STANDARD)

    def test_advisor_failure_structures_actionable_safety_intent_without_promoting_concepts(self):
        positive_cases = (
            "泰诺每次吃几片？",
            "泰 诺 每 次 吃 几 片？",
            "我突然半边脸麻、说话含糊，怎么办？",
            "84消毒液溅到眼里怎么办？",
            "发烧 39 度怎么办？",
        )
        for question in positive_cases:
            with self.subTest(kind="actionable", question=question):
                self.assertIs(decide(question, {}).route, SearchTier.DEEP)

        stable_concepts = (
            "什么是药物剂量？",
            "清洁剂为什么会刺激眼睛？",
            "我不是在问泰诺每次吃几片，想了解药物剂量的定义。",
        )
        for question in stable_concepts:
            with self.subTest(kind="concept", question=question):
                self.assertIs(decide(question, {}).route, SearchTier.STANDARD)

        quoted_or_nonpersonal = (
            "“泰诺每次吃几片？”这句话是什么意思？",
            "有人引用“84消毒液溅到眼里怎么办”，这句话是什么意思？",
        )
        for question in quoted_or_nonpersonal:
            with self.subTest(kind="quoted", question=question):
                self.assertIsNot(decide(question, {}).route, SearchTier.DEEP)

    def test_advisor_failure_handles_structural_safety_scope_and_unseen_variants(self):
        actionable_cases = (
            "扑热息痛一次吃几毫克？",
            "这个药应该间隔多久再吃？",
            "我突然嘴角歪斜、单侧无力，怎么办？",
            "洁厕灵弄进眼里了，怎么办？",
            "胸痛时应该采取什么急救措施？",
            "漂白剂进眼睛为什么很疼，该怎么办？",
            "这瓶家用液体溅进眼里了，该怎么处理？",
        )
        for question in actionable_cases:
            with self.subTest(kind="actionable", question=question):
                self.assertIs(decide(question, {}).route, SearchTier.DEEP)

        stable_concepts = (
            "什么是药物剂量？",
            "清洁剂会刺激眼睛吗？",
            "漂白剂刺激眼睛的机制是什么？",
        )
        for question in stable_concepts:
            with self.subTest(kind="concept", question=question):
                self.assertIs(decide(question, {}).route, SearchTier.STANDARD)

        excluded_or_harmless = (
            "薯片每次吃几片比较合适？",
            "我没有胸痛，不需要问怎么办。",
            "“胸痛怎么办”这几个字用于测试。",
            "“胸痛怎么办（示例文本）",
            "请把“洁厕灵弄进眼里怎么办”翻译成英文。",
            "我没有眼部暴露，也不打算采取急救措施。",
        )
        for question in excluded_or_harmless:
            with self.subTest(kind="excluded", question=question):
                self.assertIsNot(decide(question, {}).route, SearchTier.DEEP)

    def test_advisor_failure_covers_round_two_actionable_safety_families(self):
        actionable_cases = (
            "对乙酰氨基酚一次吃500mg可以吗？",
            "美林每隔六小时能再吃吗？",
            "这个药多久吃一回？",
            "布洛芬能服用吗？",
            "这个药可以突然停吗？",
            "一侧手脚无力、口齿不清怎么办？",
            "突然胸痛该做什么？",
            "心口疼得厉害，要叫救护车吗？",
            "胸痛没缓解怎么办？",
            "我没法呼吸怎么办？",
            "清洁剂误吞怎么办？",
            "喝下漂白剂怎么办？",
            "漂白剂溅进眼睛，需要马上冲洗吗？",
            "洗衣液不小心进眼了，应该怎么办？",
            "吃完饼干呼吸困难怎么办？",
            "清洁剂误吞后为什么恶心，该怎么处理？",
        )
        for question in actionable_cases:
            with self.subTest(question=question):
                self.assertIs(decide(question, {}).route, SearchTier.DEEP)

    def test_advisor_failure_handles_family_level_units_paraphrases_and_scoped_negation(self):
        actionable_cases = (
            "头孢一天三回，每回两粒行不行？",
            "这种口服液一次用十毫升可以吗？",
            "阿莫西林隔8小时能服一次吗？",
            "突然右臂抬不起来，讲话不清楚，要打120吗？",
            "胸口有压迫感并冒冷汗，需要去急诊吗？",
            "气喘不上来，该怎么处理？",
            "那瓶除垢液喷到眼里，要立即用水冲吗？",
            "吸入除虫喷雾后头晕，要不要就医？",
            "不小心咽下除垢剂，需要马上处理吗？",
            "不是没有胸痛，是疼痛一直没缓解，该怎么办？",
            "现在没有胸痛，但右手突然无力、说话不清，要去急诊吗？",
            "“胸痛怎么办只是示例。现在我胸痛还没完全缓解，要打120吗？",
            "我现在喘不上气，该怎么办”",
            "I don't know，我胸痛还没缓解，该怎么办？",
            "吃完面包突然胸口压迫、喘不上气，该怎么办？",
        )
        for question in actionable_cases:
            with self.subTest(question=question):
                self.assertIs(decide(question, {}).route, SearchTier.DEEP)

    def test_advisor_failure_keeps_stable_explanations_and_benign_scopes_below_deep(self):
        stable_explanations = (
            "布洛芬有哪些常见副作用？",
            "误吞清洁剂会有什么后果？",
            "漂白剂刺激眼睛的机制是什么？",
            "目前没有胸痛，也没有呼吸困难，只想了解这些症状的常见原因。",
        )
        for question in stable_explanations:
            with self.subTest(kind="stable", question=question):
                self.assertIs(decide(question, {}).route, SearchTier.STANDARD)

        quoted_or_harmless = (
            "胸痛怎么办”这只是示例。",
            "胸痛怎么办\" this is a test.",
            "“‘胸口压迫感要打120吗’只是测试文本”请改写。",
            "'胸痛怎么办' 只是示例文本。",
            "洗澡水弄进眼睛怎么办？",
            "矿泉水进眼睛，需要马上冲洗吗？",
            "蛋糕每隔六小时能再吃吗？",
            "我没有误吞清洁剂，也没有眼部接触，只是在校对急救说明。",
        )
        for question in quoted_or_harmless:
            with self.subTest(kind="excluded", question=question):
                self.assertIsNot(decide(question, {}).route, SearchTier.DEEP)

    def test_advisor_failure_covers_round_three_medication_acute_and_exposure_actions(self):
        actionable_cases = (
            "布洛芬能停吗？",
            "布洛芬需要加量吗？",
            "一只手突然抬不起来，需要去急诊吗？",
            "误喝洁厕液要催吐吗？",
            "闻了杀虫剂后头晕，是否要就医？",
            "阿莫西林可以减量吗？",
            "一边胳膊突然使不上劲，要叫救护车吗？",
            "不慎饮下除垢液，需要马上就医吗？",
            "吸到杀虫喷雾后恶心，该去医院吗？",
        )
        for question in actionable_cases:
            with self.subTest(question=question):
                self.assertIs(decide(question, {}).route, SearchTier.DEEP)

    def test_advisor_failure_scopes_absence_and_meta_without_hiding_later_emergencies(self):
        excluded_cases = (
            "我并没有出现胸痛，只想了解这种情况要去急诊吗？",
            "从未发生清洁剂入眼，只想了解这种情况需要冲洗吗？",
            "胸痛怎么办是测试用例标题。",
            "【胸痛怎么办】只是测试标题。",
            "`胸痛怎么办`只是代码示例。",
            "从来没有发生漂白剂入眼，只是在校对急救说明。",
            "这个标题是胸痛怎么办，仅用于回归测试。",
        )
        for question in excluded_cases:
            with self.subTest(kind="excluded", question=question):
                self.assertIsNot(decide(question, {}).route, SearchTier.DEEP)

        active_after_excluded_scope = (
            "我没有出现胸痛，但呼吸困难怎么办？",
            "从未发生清洁剂入眼；但现在胸口压迫，需要急诊吗？",
            "胸痛怎么办属于示例标题；但我现在呼吸困难，该怎么办？",
            "“胸痛怎么办”是示例；但我一只手抬不起来，需要急诊吗？",
            "【误喝清洁剂要催吐吗】是测试；但我一边胳膊使不上劲，要打120吗？",
        )
        for question in active_after_excluded_scope:
            with self.subTest(kind="active_after_excluded", question=question):
                self.assertIs(decide(question, {}).route, SearchTier.DEEP)

    def test_advisor_failure_binds_dose_frequency_and_change_to_medication_subject(self):
        benign_overlaps = (
            "面粉一次用500g可以吗？",
            "药品说明书每次打印几页可以吗？",
            "水泥每次用两袋够吗？",
            "这个药盒每次印几张标签可以吗？",
            "咖啡粉每天用10g合适吗？",
        )
        for question in benign_overlaps:
            with self.subTest(question=question):
                self.assertIsNot(decide(question, {}).route, SearchTier.DEEP)

    def test_advisor_failure_floors_stable_effect_use_and_meaning_explanations_at_standard(self):
        stable_explanations = (
            "布洛芬的主要作用有哪些？",
            "对乙酰氨基酚是做什么用的？",
            "胸痛通常代表什么？",
            "阿莫西林有什么用途？",
            "呼吸困难一般意味着什么？",
            "洁厕剂的作用是什么？",
        )
        for question in stable_explanations:
            with self.subTest(question=question):
                self.assertIs(decide(question, {}).route, SearchTier.STANDARD)

    def test_stable_version_definition_is_not_deep(self):
        decision = decide("什么是版本控制")
        self.assertNotEqual(decision.route, SearchTier.DEEP)

    def test_current_version_question_is_deep(self):
        decision = decide("最新版本是多少")
        self.assertIs(decision.route, SearchTier.DEEP)


class FreshnessAndGreetingRegressionTests(unittest.TestCase):
    def test_model_high_freshness_cannot_remain_light(self):
        payload = {
            **NEUTRAL,
            "freshness": "high",
            "recommended_tier": "light",
            "trigger_codes": ["freshness_marker"],
        }
        decision = decide("昨天天曼契约EDGVSTEC谁赢了", payload)
        self.assertIs(decision.route, SearchTier.DEEP)
        self.assertIs(decision.program_minimum_tier, SearchTier.DEEP)
        self.assertIn(TriggerCode.FRESHNESS_MARKER, decision.trigger_codes)
        self.assertEqual(decision.trigger_codes.count(TriggerCode.FRESHNESS_MARKER), 1)
        self.assertEqual(decision.final_reason_codes.count(TriggerCode.FRESHNESS_MARKER), 1)

    def test_model_high_freshness_sets_deep_floor_without_question_signal(self):
        payload = {
            **NEUTRAL,
            "freshness": "high",
            "recommended_tier": "light",
            "trigger_codes": ["freshness_marker"],
        }
        decision = decide("什么是光合作用", payload)
        self.assertIs(decision.route, SearchTier.DEEP)
        self.assertIs(decision.program_minimum_tier, SearchTier.DEEP)

    def test_relative_time_and_result_intent_is_deep_without_model_help(self):
        for question in (
            "昨天EDG和TEC谁赢了",
            "前天比赛比分是多少",
            "本周排名结果如何",
            "刚刚发生了什么",
        ):
            with self.subTest(question=question):
                self.assertIs(decide(question, NEUTRAL).route, SearchTier.DEEP)

    def test_pure_greetings_skip_search(self):
        for question in ("你好", "您好！", "下午好", "晚上好，ATRI", "哈喽"):
            with self.subTest(question=question):
                decision = decide(question, {})
                self.assertIs(decision.route, SearchTier.SKIP)
                self.assertIs(decision.skip_reason, SkipReason.SOCIAL_OR_EMOTIONAL)

    def test_greeting_plus_current_fact_does_not_skip(self):
        decision = decide("下午好，昨天EDG赢了吗", NEUTRAL)
        self.assertIs(decision.route, SearchTier.DEEP)
        self.assertIsNone(decision.skip_reason)

    def test_result_variant_without_relative_time_stays_light(self):
        decision = decide("他赢了吗", NEUTRAL)
        self.assertIs(decision.route, SearchTier.LIGHT)


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
