"""Program-owned retrieval benefit router.

The router decides whether a request is a complete closed task that carries
no retrieval benefit. Search execution is program-owned: the model may
classify benefits and recommend a tier, but it can never lower the program
floor or pick an unrecognized ``skip_reason``. Any ambiguity routes to at
least ``light``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from src.search.models import (
    Actionability,
    BenefitDimension,
    Factuality,
    Freshness,
    PotentialHarm,
    RequestSource,
    RetrievalDecision,
    RetrievalRequest,
    RiskLevel,
    SearchTier,
    SkipReason,
    TriggerCode,
    max_tier,
)

_CLASSIFIER_UNCERTAIN = TriggerCode.CLASSIFIER_UNCERTAIN
_FACTUAL_DEFAULT = TriggerCode.FACTUAL_DEFAULT


def _question_text(request: RetrievalRequest) -> str:
    return str(request.question or "").strip()


# ── deterministic explicit-constraint detection ─────────────────────────

_NO_WEB_MARKERS = (
    "不要联网",
    "别联网",
    "不联网",
    "不要搜索",
    "别搜索",
    "不用搜索",
    "不需要搜索",
    "不要搜",
    "别搜",
    "不用查",
    "不要查",
    "不要上网",
    "别上网",
    "只根据",
    "只依据",
    "只按",
    "只凭",
    "根据我贴",
    "依据我",
    "不查网",
)


def _detect_explicit_no_web(question: str) -> bool:
    return any(marker in question for marker in _NO_WEB_MARKERS)


_SEARCH_MARKERS = (
    "搜索",
    "搜一下",
    "搜一搜",
    "搜搜",
    "查一下",
    "查一查",
    "查查",
    "帮我查",
    "帮我搜",
    "帮我搜索",
    "请搜索",
    "请查",
    "去搜",
    "上网查",
    "联网查",
    "联网搜",
    "联网搜索",
    "在网上查",
    "网上搜",
    "帮我查询",
    "请查询",
)


def _detect_explicit_search(question: str) -> bool:
    return any(marker in question for marker in _SEARCH_MARKERS)


_VERIFICATION_MARKERS = (
    "核实",
    "验证",
    "查证",
    "核对",
    "求证",
    "确认一下",
    "verify",
    "核实一下",
    "查实",
)


def _detect_explicit_verification(question: str) -> bool:
    return any(marker in question for marker in _VERIFICATION_MARKERS)


_SOURCE_MARKERS = (
    "来源",
    "出处",
    "引用来源",
    "给出来源",
    "附上来源",
    "写上来源",
    "信息来源",
)


def _detect_explicit_source_request(question: str) -> bool:
    return any(marker in question for marker in _SOURCE_MARKERS)


def _explicit_trigger_codes(
    explicit_search: bool,
    explicit_verification: bool,
    explicit_source: bool,
) -> tuple[TriggerCode, ...]:
    codes: list[TriggerCode] = []
    if explicit_search:
        codes.append(TriggerCode.EXPLICIT_SEARCH)
    if explicit_verification:
        codes.append(TriggerCode.EXPLICIT_VERIFICATION)
    if explicit_source:
        codes.append(TriggerCode.EXPLICIT_SOURCE_REQUEST)
    return tuple(codes)


# ── deterministic dynamic / high-consequence / regulated-domain floors ──

_CURRENT_STATE_TRIGGERS = (
    ("新闻", TriggerCode.CURRENT_RULE_OR_POLICY),
    ("最新", TriggerCode.FRESHNESS_MARKER),
    ("最近", TriggerCode.FRESHNESS_MARKER),
    ("目前", TriggerCode.CURRENT_RULE_OR_POLICY),
    ("现在", TriggerCode.CURRENT_RULE_OR_POLICY),
    ("当前", TriggerCode.CURRENT_RULE_OR_POLICY),
    ("现状", TriggerCode.DYNAMIC_ATTRIBUTE),
    ("实时", TriggerCode.FRESHNESS_MARKER),
    ("价格", TriggerCode.DYNAMIC_ATTRIBUTE),
    ("版本", TriggerCode.DYNAMIC_ATTRIBUTE),
    ("行情", TriggerCode.DYNAMIC_ATTRIBUTE),
    ("规则", TriggerCode.CURRENT_RULE_OR_POLICY),
    ("政策", TriggerCode.CURRENT_RULE_OR_POLICY),
    ("法规", TriggerCode.CURRENT_RULE_OR_POLICY),
    ("利率", TriggerCode.DYNAMIC_ATTRIBUTE),
    ("汇率", TriggerCode.DYNAMIC_ATTRIBUTE),
)

# "今天"/"最近" only count when they modify an external-fact context word.
_DATED_EXTERNAL_CONTEXT = (
    "新闻",
    "消息",
    "比赛",
    "事件",
    "价格",
    "版本",
    "天气",
    "发生了什么",
    "有什么新闻",
    "股票",
    "基金",
    "政策",
    "发布",
    "更新",
    "汇率",
)


def _detect_current_state(question: str) -> tuple[TriggerCode, ...]:
    lowered = question.casefold()
    codes: list[TriggerCode] = []
    for marker, code in _CURRENT_STATE_TRIGGERS:
        if marker in lowered:
            codes.append(code)
    for dated_marker in ("今天", "昨日", "今天", "今明"):
        if dated_marker in lowered:
            if any(context in lowered for context in _DATED_EXTERNAL_CONTEXT):
                codes.append(TriggerCode.FRESHNESS_MARKER)
                break
    return _dedupe_codes(codes)


def _dedupe_codes(codes: Sequence[TriggerCode]) -> tuple[TriggerCode, ...]:
    seen: set[TriggerCode] = set()
    result: list[TriggerCode] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            result.append(code)
    return tuple(result)


_PERSONAL_MARKERS = (
    "我应该",
    "我该不该",
    "我是否",
    "我是不是",
    "我的情况",
    "我要不要",
    "我要不要买",
    "我要不要卖",
    "我该",
    "帮我看看",
    "我应该不应该",
    "适合我",
    "对我",
    "我可以买",
)

_HIGH_CONSEQUENCE_DOMAINS = (
    "股票",
    "基金",
    "投资",
    "理财",
    "买房",
    "卖房",
    "贷款",
    "还款",
    "手术",
    "吃药",
    "用药",
    "剂量",
    "诊断",
    "治疗",
    "违法",
    "犯罪",
    "起诉",
    "诉讼",
    "工伤",
    "赔偿",
    "辞职",
    "跳槽",
    "毒品",
    "遗嘱",
    "遗产",
)


def _detect_personalized_high_consequence(question: str) -> TriggerCode | None:
    lowered = question.casefold()
    personal = any(marker in lowered for marker in _PERSONAL_MARKERS)
    domain = any(marker in lowered for marker in _HIGH_CONSEQUENCE_DOMAINS)
    if personal and domain:
        return TriggerCode.HIGH_CONSEQUENCE_ACTION
    # A first-person consequence question such as "我的具体情况是否违法"
    # is a personalized legal determination even without a domain noun.
    first_person_consequence = any(
        marker in lowered
        for marker in (
            "我的情况",
            "我是否",
            "我是不是",
            "我该不该",
            "我要不要",
            "我应该",
            "我的",
        )
    )
    consequence_intent = any(
        marker in lowered
        for marker in (
            "违法",
            "犯罪",
            "责任",
            "起诉",
            "处罚",
            "算不算",
            "是否合法",
            "合不合法",
        )
    )
    if first_person_consequence and consequence_intent:
        return TriggerCode.HIGH_CONSEQUENCE_ACTION
    if domain and any(marker in lowered for marker in ("是否应该", "该不该", "要不要", "能不能")):
        return TriggerCode.HIGH_CONSEQUENCE_ACTION
    return None


_REGULATED_DOMAIN_FOUNDATION_WORDS = (
    "股票",
    "基金",
    "理财",
    "民法",
    "刑法",
    "合同法",
    "劳动法",
    "税法",
    "保险",
    "药品",
    "抗生素",
    "疫苗",
    "抵押权",
    "继承权",
    "刑事责任",
)

_FOUNDATION_PATTERNS = (
    "什么是",
    "是什么",
    "啥是",
    "的定义",
    "有哪些",
    "介绍一下",
    "指的是什么",
)


def _detect_regulated_foundation(question: str) -> TriggerCode | None:
    if not any(pattern in question for pattern in _FOUNDATION_PATTERNS):
        return None
    if any(marker in question for marker in _REGULATED_DOMAIN_FOUNDATION_WORDS):
        return TriggerCode.REGULATED_DOMAIN_FOUNDATION
    return None


_COMPARISON_MARKERS = (
    "区别",
    "不同",
    "对比",
    "相比",
    " vs ",
    "哪个好",
    "有什么不同",
    "差异",
)

_EXPLANATION_MARKERS = (
    "为什么",
    "为何",
    "怎么回事",
    "原因",
    "原理",
    "机制",
    "解释",
)


def _detect_external_explanation_or_comparison(
    question: str,
    external_fact_required: bool,
) -> TriggerCode | None:
    lowered = question.casefold()
    is_comparison = any(marker in lowered for marker in _COMPARISON_MARKERS)
    is_explanation = any(marker in lowered for marker in _EXPLANATION_MARKERS)
    if not (is_comparison or is_explanation):
        return None
    if not external_fact_required:
        return None
    return TriggerCode.EXTERNAL_FACT_EXPLANATION_OR_COMPARISON


# ── deterministic closed-task skip classification ───────────────────────

_TRANSFORM_MARKERS = (
    "润色",
    "改写",
    "翻译",
    "换一种说法",
    "换个说法",
    "转述",
    "扩写",
    "缩写",
    "重新组织",
    "重写",
    "简化",
)

_SUMMARY_MARKERS = (
    "总结",
    "概括",
    "归纳",
    "提炼",
    "总结一下",
)

_PROVIDED_CONTENT_MARKERS = (
    "这段话",
    "这段文字",
    "这段内容",
    "这段",
    "我贴的",
    "我发的",
    "刚才贴",
    "以下内容",
    "上面这段",
    "我给的内容",
    "这段介绍",
    "这段价格",
)

_CLOSED_CONTEXT_MARKERS = (
    "我刚才",
    "刚才贴",
    "刚才说",
    "上一条",
    "我贴的",
    "我发的",
    "我上面",
    "这段文字",
    "这段话",
    "刚才那句话",
    "我刚才那句话",
    "我们刚才",
    "我消息",
    "我写的",
)

_SOCIAL_EMOTIONAL_MARKERS = (
    "心情",
    "情绪",
    "有点难过",
    "不开心",
    "开心",
    "感觉",
    "聊聊",
    "安慰",
    "陪伴",
    "郁闷",
    "烦",
    "累",
    "今天过得",
    "怎么样最近",
)

_CREATIVE_MARKERS = (
    "写个故事",
    "写一首",
    "写首诗",
    "编个故事",
    "角色扮演",
    "扮演",
    "想象一下",
    "虚构",
    "小说",
    "同人",
)

_MATH_MARKERS = (
    "证明",
    "推导",
    "求证",
    "定理",
    "求导",
    "²",
    "³",
    "积分",
)

_LOGIC_MARKERS = (
    "逻辑",
    "前提",
    "三段论",
    "推理",
)


def _classify_closed_task(question: str) -> SkipReason | None:
    lowered = question.casefold()
    has_provided_content = any(marker in question for marker in _PROVIDED_CONTENT_MARKERS)

    if any(marker in lowered for marker in _TRANSFORM_MARKERS) and (
        has_provided_content or "：" in question or ":" in question
    ):
        return SkipReason.PROVIDED_TEXT_TRANSFORM

    if any(marker in lowered for marker in _SUMMARY_MARKERS) and has_provided_content:
        return SkipReason.PROVIDED_CONTENT_SUMMARY

    if any(marker in lowered for marker in _MATH_MARKERS):
        return SkipReason.PURE_MATH

    if any(marker in lowered for marker in _LOGIC_MARKERS):
        return SkipReason.CLOSED_LOGIC

    if any(marker in question for marker in _CLOSED_CONTEXT_MARKERS):
        explanation_intent = any(
            marker in question for marker in ("解释", "为什么", "什么意思", "说明", "分析")
        )
        if explanation_intent or has_provided_content:
            return SkipReason.CLOSED_CONTEXT_ONLY

    if any(marker in question for marker in _SOCIAL_EMOTIONAL_MARKERS):
        return SkipReason.SOCIAL_OR_EMOTIONAL

    if any(marker in lowered for marker in _CREATIVE_MARKERS):
        return SkipReason.CREATIVE_OR_ROLEPLAY

    return None


# ── closed enum parsing helpers ─────────────────────────────────────────

def _parse_enum(value: Any, enum_type: type[Any]) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError:
        return None


def _parse_enum_list(values: Any, enum_type: type[Any]) -> tuple[Any, ...] | None:
    if not isinstance(values, (list, tuple, set, frozenset)):
        return None
    result: list[Any] = []
    for value in values:
        parsed = _parse_enum(value, enum_type)
        if parsed is None:
            return None
        result.append(parsed)
    return tuple(result)


# ── classification envelope ─────────────────────────────────────────────

@dataclass(frozen=True)
class _Classification:
    factuality: Factuality = Factuality.AMBIGUOUS
    external_fact_required: bool = True
    freshness: Freshness = Freshness.NONE
    risk: RiskLevel = RiskLevel.LOW
    actionability: Actionability = Actionability.NONE
    potential_harm: PotentialHarm = PotentialHarm.NONE
    recommended_tier: SearchTier | None = None
    benefit_dimensions: frozenset[BenefitDimension] = frozenset()
    trigger_codes: tuple[TriggerCode, ...] = ()


def _validated_classification(raw: Any) -> _Classification:
    if not isinstance(raw, dict):
        return _Classification()
    factuality = _parse_enum(raw.get("factuality"), Factuality) or Factuality.AMBIGUOUS
    external_fact_required = raw.get("external_fact_required", True)
    if not isinstance(external_fact_required, bool):
        external_fact_required = True
    freshness = _parse_enum(raw.get("freshness"), Freshness) or Freshness.NONE
    risk = _parse_enum(raw.get("risk"), RiskLevel) or RiskLevel.LOW
    actionability = _parse_enum(raw.get("actionability"), Actionability) or Actionability.NONE
    potential_harm = _parse_enum(raw.get("potential_harm"), PotentialHarm) or PotentialHarm.NONE
    recommended = _parse_enum(raw.get("recommended_tier"), SearchTier)
    if recommended is SearchTier.SKIP:
        recommended = None
    benefits = _parse_enum_list(raw.get("benefit_dimensions"), BenefitDimension) or ()
    triggers = _parse_enum_list(raw.get("trigger_codes"), TriggerCode) or ()
    return _Classification(
        factuality=factuality,
        external_fact_required=external_fact_required,
        freshness=freshness,
        risk=risk,
        actionability=actionability,
        potential_harm=potential_harm,
        recommended_tier=recommended,
        benefit_dimensions=frozenset(benefits),
        trigger_codes=tuple(triggers),
    )


# ── strict JSON parsing for the routing advisor ─────────────────────────

_FENCE_PATTERN = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)

_ADVISOR_ENUM_FIELDS: Mapping[str, type[Any]] = {
    "factuality": Factuality,
    "freshness": Freshness,
    "risk": RiskLevel,
    "actionability": Actionability,
    "potential_harm": PotentialHarm,
    "recommended_tier": SearchTier,
    "benefit_dimensions": BenefitDimension,
    "trigger_codes": TriggerCode,
}


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _raise_invalid_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant: {value}")


def _normalized_advisor_output(payload: dict[str, Any]) -> dict[str, Any]:
    for field_name, enum_type in _ADVISOR_ENUM_FIELDS.items():
        if field_name not in payload:
            continue
        value = payload[field_name]
        if isinstance(value, list):
            parsed = _parse_enum_list(value, enum_type)
            if parsed is None:
                raise ValueError(f"invalid enum list: {field_name}")
            payload[field_name] = list(parsed)
        else:
            parsed = _parse_enum(value, enum_type)
            if parsed is None:
                raise ValueError(f"invalid enum value: {field_name}")
            payload[field_name] = parsed
    recommended = payload.get("recommended_tier")
    if recommended is SearchTier.SKIP:
        raise ValueError("model may not recommend skip")
    skip_candidate = payload.get("skip_candidate")
    if skip_candidate is not None:
        if not isinstance(skip_candidate, dict) or "reason" not in skip_candidate:
            raise ValueError("skip_candidate must be null or an object with reason")
        reason = _parse_enum(skip_candidate["reason"], SkipReason)
        if reason is None:
            raise ValueError("skip_candidate reason is not a closed enum")
        payload["skip_candidate"] = {"reason": reason}
    return payload


def parse_advisor_json(content: Any) -> dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        return {}
    fenced = _FENCE_PATTERN.fullmatch(text)
    if fenced is not None:
        text = fenced.group("body").strip()
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_raise_invalid_constant,
        )
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    try:
        return _normalized_advisor_output(payload)
    except ValueError:
        return {}


# ── routing advisor prompt ──────────────────────────────────────────────

ROUTING_SYSTEM_PROMPT = """\
You classify whether a web search would improve this answer. You do not see
chat history, stored facts, model confidence, or an answer draft. Treat the
question text as the only input.

Return JSON only with exactly these fields:
{
  "skip_candidate": null or {"reason": "one closed reason"},
  "benefit_dimensions": ["accuracy", "freshness", "completeness", "verifiability", "disambiguation", "risk_control"],
  "factuality": "non_factual|factual|mixed|ambiguous",
  "external_fact_required": true or false,
  "freshness": "none|low|high",
  "risk": "low|medium|high",
  "actionability": "none|general|personalized",
  "potential_harm": "none|low|high",
  "recommended_tier": "light|standard|deep",
  "trigger_codes": ["one or more closed trigger codes"]
}

Allowed skip reasons: user_forbid_web, social_or_emotional,
creative_or_roleplay, provided_text_transform, provided_content_summary,
pure_math, closed_logic, closed_context_only.

Allowed trigger codes: explicit_no_web, explicit_search,
explicit_verification, explicit_source_request, freshness_marker,
dynamic_attribute, regulated_domain_foundation, high_consequence_action,
current_rule_or_policy, controversy_or_conflict,
external_fact_explanation_or_comparison, recommendation_or_evaluation,
ambiguous_entity, multi_hop_complexity, mixed_task, factual_default,
classifier_uncertain.

You never choose skip because the answer is "known", "common knowledge", or
confident. Searching is the default for factual, mixed, or ambiguous requests.
"""


class LLMRoutingAdvisor:
    """Model-assisted benefit classification with strict closed-enum output."""

    def __init__(self, llm: Any, *, max_tokens: int = 512) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def advise(self, request: RetrievalRequest) -> dict[str, Any]:
        payload = {
            "question": request.question,
            "has_images": request.has_images,
        }
        messages = [
            {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        try:
            response = self._llm.chat(
                messages,
                temperature=0.0,
                max_tokens=self._max_tokens,
                tools=None,
                tool_choice="none",
            )
        except Exception:
            return {}
        return parse_advisor_json(response.content)


# ── decision construction ───────────────────────────────────────────────

def _skip_decision(
    request: RetrievalRequest,
    reason: SkipReason,
    trigger_codes: tuple[TriggerCode, ...],
    *,
    forced_search: bool = False,
) -> RetrievalDecision:
    non_factual = reason in {
        SkipReason.SOCIAL_OR_EMOTIONAL,
        SkipReason.CREATIVE_OR_ROLEPLAY,
        SkipReason.PURE_MATH,
        SkipReason.CLOSED_LOGIC,
        SkipReason.CLOSED_CONTEXT_ONLY,
    }
    factuality = Factuality.NON_FACTUAL if non_factual else Factuality.MIXED
    if reason is SkipReason.USER_FORBID_WEB:
        factuality = Factuality.AMBIGUOUS
    return RetrievalDecision(
        route=SearchTier.SKIP,
        skip_reason=reason,
        forced_search=forced_search,
        trigger_codes=trigger_codes,
        benefit_dimensions=frozenset(),
        factuality=factuality,
        external_fact_required=False,
        freshness=Freshness.NONE,
        risk=RiskLevel.LOW,
        actionability=Actionability.NONE,
        potential_harm=PotentialHarm.NONE,
        program_minimum_tier=None,
        model_recommended_tier=None,
        final_reason_codes=trigger_codes,
    )


class RetrievalBenefitRouter:
    """Decide the closed route and program floor for one request."""

    def __init__(self, advisor: Any, *, clock: Any = None) -> None:
        self._advisor = advisor
        self._clock = clock

    def decide(self, request: RetrievalRequest) -> RetrievalDecision:
        question = _question_text(request)

        explicit_no_web = _detect_explicit_no_web(question)
        explicit_search = _detect_explicit_search(question)
        explicit_verification = _detect_explicit_verification(question)
        explicit_source = _detect_explicit_source_request(question)
        explicit_any = explicit_search or explicit_verification or explicit_source
        explicit_codes = _explicit_trigger_codes(explicit_search, explicit_verification, explicit_source)
        has_force_search = explicit_any or bool(request.force_search)

        # Explicit no-web is the user's hard constraint.
        if explicit_no_web:
            conflict = explicit_any
            trigger_codes = (TriggerCode.EXPLICIT_NO_WEB,)
            if conflict:
                trigger_codes = (TriggerCode.EXPLICIT_NO_WEB, TriggerCode.EXPLICIT_SEARCH)
            return _skip_decision(
                request,
                SkipReason.USER_FORBID_WEB,
                trigger_codes,
                forced_search=has_force_search,
            )

        raw = self._advisor.advise(request)
        classification = _validated_classification(raw)
        valid_advisor = bool(raw)

        # Closed-task skip is the only program authority for skipping.
        program_skip = _classify_closed_task(question)
        if program_skip is not None and not has_force_search:
            trigger_codes = _dedupe_codes((*explicit_codes, *classification.trigger_codes))
            return _skip_decision(
                request,
                program_skip,
                trigger_codes,
                forced_search=False,
            )

        # Program floor: default to light, raise on confirmed triggers.
        floor, floor_codes = _compute_floor(
            question,
            classification,
            explicit_verification=explicit_verification,
            explicit_source=explicit_source,
        )
        if not valid_advisor:
            floor_codes = (TriggerCode.CLASSIFIER_UNCERTAIN,)
        if not floor_codes:
            floor_codes = (TriggerCode.FACTUAL_DEFAULT,)

        recommended = classification.recommended_tier
        final_route = max_tier(floor, recommended) if recommended is not None else floor

        trigger_codes = _dedupe_codes((*explicit_codes, *classification.trigger_codes, *floor_codes))
        final_reason_codes = _dedupe_codes((*floor_codes, *classification.trigger_codes, *explicit_codes))
        benefit_dimensions = classification.benefit_dimensions
        if not benefit_dimensions:
            benefit_dimensions = frozenset({BenefitDimension.ACCURACY})

        return RetrievalDecision(
            route=final_route,
            skip_reason=None,
            forced_search=has_force_search,
            trigger_codes=trigger_codes,
            benefit_dimensions=benefit_dimensions,
            factuality=classification.factuality,
            external_fact_required=classification.external_fact_required,
            freshness=classification.freshness,
            risk=classification.risk,
            actionability=classification.actionability,
            potential_harm=classification.potential_harm,
            program_minimum_tier=floor,
            model_recommended_tier=recommended,
            final_reason_codes=final_reason_codes,
        )


def _compute_floor(
    question: str,
    classification: _Classification,
    *,
    explicit_verification: bool,
    explicit_source: bool,
) -> tuple[SearchTier, tuple[TriggerCode, ...]]:
    floor = SearchTier.LIGHT
    codes: list[TriggerCode] = []

    current_state_codes = _detect_current_state(question)
    if current_state_codes:
        floor = SearchTier.DEEP
        codes.extend(current_state_codes)

    high_consequence = _detect_personalized_high_consequence(question)
    if high_consequence is not None:
        floor = SearchTier.DEEP
        codes.append(high_consequence)

    regulated = _detect_regulated_foundation(question)
    if regulated is not None and _rank(floor) < _rank(SearchTier.STANDARD):
        floor = SearchTier.STANDARD
        codes.append(regulated)

    external_compare = _detect_external_explanation_or_comparison(
        question,
        classification.external_fact_required,
    )
    if external_compare is not None and _rank(floor) < _rank(SearchTier.STANDARD):
        floor = SearchTier.STANDARD
        codes.append(external_compare)

    if (explicit_verification or explicit_source) and _rank(floor) < _rank(SearchTier.STANDARD):
        floor = SearchTier.STANDARD
        if explicit_verification:
            codes.append(TriggerCode.EXPLICIT_VERIFICATION)
        if explicit_source:
            codes.append(TriggerCode.EXPLICIT_SOURCE_REQUEST)

    return floor, tuple(codes)


def _rank(tier: SearchTier) -> int:
    return {SearchTier.SKIP: 0, SearchTier.LIGHT: 1, SearchTier.STANDARD: 2, SearchTier.DEEP: 3}[tier]
