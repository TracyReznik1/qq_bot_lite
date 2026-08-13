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
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Mapping, Sequence

from src.search.models import (
    Actionability,
    BenefitDimension,
    Factuality,
    Freshness,
    FreshnessContext,
    FreshnessRequirement,
    PotentialHarm,
    RequestAnalysis,
    RetrievalComplexityCode,
    RetrievalContext,
    RetrievalDecision,
    RetrievalRequest,
    RiskContext,
    RiskLevel,
    SearchTier,
    SkipReason,
    SourceRequirement,
    TriggerCode,
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
    "/search",
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

_RELATIVE_TIME_MARKERS = (
    "今天", "今日", "昨天", "昨日", "前天", "刚刚", "最近", "近期",
    "目前", "现在", "当前", "实时", "本周", "上周", "本月", "上月",
    "今年", "去年",
)

_CURRENT_RESULT_INTENTS = (
    "谁赢了", "赢了", "赢", "胜负", "比分", "赛果", "比赛结果", "排名",
    "发生了什么", "结果如何", "最新进展",
)

_PURE_GREETING_PREFIXES = (
    "你好", "您好", "嗨", "哈喽", "在吗", "早上好", "中午好",
    "下午好", "晚上好", "晚安",
)

_GREETING_VOCATIVES = ("", "atri", "亚托莉", "机器人")


def _is_pure_greeting(question: str) -> bool:
    normalized = unicodedata.normalize("NFKC", str(question or "")).casefold()
    normalized = re.sub(r"[\s，,。.!！?？、~～]+", "", normalized)
    return any(
        normalized == f"{greeting}{vocative}"
        for greeting in _PURE_GREETING_PREFIXES
        for vocative in _GREETING_VOCATIVES
    )


# Dynamic-attribute words only force a STANDARD/current-state floor when the
# request asks about the current value/state, not for a stable definition.
_DYNAMIC_ATTRIBUTE_WORDS = ("版本", "价格", "行情", "利率", "汇率")
_FRESHNESS_MODIFIERS = ("最新", "当前", "现在", "目前", "最近", "今天", "实时", "今日", "当下")
_CURRENT_RULE_WORDS = ("规则", "政策", "法规")


def _detect_current_state(question: str) -> tuple[TriggerCode, ...]:
    lowered = question.casefold()
    codes: list[TriggerCode] = []
    for marker, code in _CURRENT_STATE_TRIGGERS:
        if marker in lowered:
            if marker in _DYNAMIC_ATTRIBUTE_WORDS:
                # Only current-version/price questions trigger STANDARD; a stable
                # definition such as "什么是版本控制" must not.
                if any(mod in lowered for mod in _FRESHNESS_MODIFIERS):
                    codes.append(code)
                continue
            if marker in _CURRENT_RULE_WORDS:
                # Rules/policies need a current-state or personalized context.
                if any(mod in lowered for mod in _FRESHNESS_MODIFIERS) or "我" in question:
                    codes.append(code)
                continue
            codes.append(code)
    for dated_marker in ("今天", "昨日", "今日", "今明"):
        if dated_marker in lowered:
            if any(context in lowered for context in _DATED_EXTERNAL_CONTEXT):
                codes.append(TriggerCode.FRESHNESS_MARKER)
                break
    if (
        any(marker in lowered for marker in _RELATIVE_TIME_MARKERS)
        and (
            any(context in lowered for context in _DATED_EXTERNAL_CONTEXT)
            or any(intent in lowered for intent in _CURRENT_RESULT_INTENTS)
        )
    ):
        codes.append(TriggerCode.FRESHNESS_MARKER)
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

# Dose/usage/action directives that imply personalized high-consequence action
# even without an explicit first-person pronoun.
_HIGH_CONSEQUENCE_ACTION_PHRASES = (
    "吃多少",
    "每天吃",
    "该吃多少",
    "用量",
    "服药",
    "剂量",
    "能不能吃",
    "该不该",
    "要不要",
    "买不买",
    "卖不卖",
    "涨到多少",
    "跌到多少",
    "现在买",
    "现在卖",
)

@dataclass(frozen=True)
class _SafetySpan:
    start: int
    end: int


@dataclass(frozen=True)
class _SafetyView:
    normalized_text: str
    context_text: str
    active_text: str
    quoted_spans: tuple[_SafetySpan, ...]
    negated_spans: tuple[_SafetySpan, ...]
    meta_spans: tuple[_SafetySpan, ...]
    absence_spans: tuple[_SafetySpan, ...]


_QUOTE_PAIRS = {
    "“": "”", "‘": "’", "「": "」", "『": "』", "《": "》", "【": "】",
}
_QUOTE_OPEN_FOR_CLOSE = {closer: opener for opener, closer in _QUOTE_PAIRS.items()}
_SENTENCE_BOUNDARIES = "。！？!?；;\n"
_QUOTE_META_MARKERS = (
    "示例", "测试", "引用", "原文", "文本", "句子", "字样", "标题",
    "test", "example", "sample", "quote",
)
_NEGATED_META_PATTERN = re.compile(
    r"(?:我)?(?:不是|并非)(?:在)?(?:问|咨询|询问|想知道).{0,80}?(?=[，,。；;！？!?]|$)"
)
_META_DECLARATION_PATTERN = re.compile(
    r"(?:(?:是|作为|属于|用于|叫做|命名为).{0,12}(?:测试|示例|用例|标题|代码|文本|例句)|"
    r"(?:测试|示例|用例|标题|代码|文本|例句).{0,6}(?:是|为|[:：])|"
    r"(?:这个|该)?(?:标题|示例|测试用例|代码)(?:是|为|[:：]))"
)
_META_DECLARATION_NEGATION_PATTERN = re.compile(
    r"(?:不(?:是|作为|属于|用于|叫做|命名为)|并非|"
    r"未(?:作为|属于|用于|叫做|命名为)|"
    r"没(?:有)?(?:作为|属于|用于|叫做|命名为))"
)
_BACKTICK_SPAN_PATTERN = re.compile(r"(?P<ticks>`{1,3}).*?(?P=ticks)", re.DOTALL)
_ABSENCE_RELATION_PATTERN = re.compile(
    r"(?:(?:并)?(?:没有|没(?!法|能|敢|缓解|好转|改善|消失))|"
    r"从来没有|从未|未曾|不曾)"
    r"(?:出现|发生|经历|遭遇|接触到?|进入|吸入|误吞|误喝|误饮|喝下|饮下)?"
)
_RELATION_BOUNDARY_PATTERN = re.compile(r"[，,。；;！？!?\n]|但(?:是)?|不过|然而|却")

_NUMBER_TOKEN = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百两半几]+)"
_ADMIN_PATTERN = re.compile(r"(?:口服|服用|服|吃|注射|滴服|用药|使用(?:这个|这种|该)?药)")
_MEDICATION_CHANGE_PATTERN = re.compile(
    r"(?:停(?!产|售|工)|停药|停服|加量|增量|减量|减药|换药)"
)
_GENERIC_MEDICATION_PATTERN = re.compile(
    r"(?:(?:这个|这种|该)?药(?!品?说明书|物?手册|品?标签|品?包装|盒|箱|品?广告|品?目录)"
    r"(?:物|品)?|处方药|非处方药|胶囊|片剂|口服液|针剂|滴眼液)"
)
_DOSE_PATTERN = re.compile(
    rf"{_NUMBER_TOKEN}(?:mg|mcg|ug|μg|g|ml|毫克|微克|克|毫升|片|粒|丸|袋|支|滴)"
)
_STRONG_DOSE_PATTERN = re.compile(
    rf"{_NUMBER_TOKEN}(?:mg|mcg|ug|μg|g|ml|毫克|微克|克|毫升)"
)
_PILL_DOSE_PATTERN = re.compile(
    rf"{_NUMBER_TOKEN}(?:片|粒|丸|滴)"
)
_INTERVAL_PATTERN = re.compile(
    rf"(?:每(?:隔)?{_NUMBER_TOKEN}(?:个)?(?:小时|钟头|天|日)|"
    rf"隔{_NUMBER_TOKEN}(?:个)?(?:小时|钟头|天|日)|"
    rf"(?:一天|每日|每天|一日|每次|每回){_NUMBER_TOKEN}?(?:次|回)?|"
    r"多久(?:再)?|间隔多久|一回|一次)"
)
_MEDICATION_QUESTION_PATTERN = re.compile(
    r"(?:多少|几(?:片|粒|次|回|毫克|微克|克|毫升)|多久|间隔|行不行|可不可以|可以吗|"
    r"能(?:不能|否)?.{0,8}吗|(?:可以|是否|需要|要不要|该不该).{0,8}吗|"
    r"该.{0,8}(?:吗|多少|多久)|怎么(?:吃|用|服)|用量|剂量)"
)
_LIKELY_MEDICATION_NAME_PATTERN = re.compile(
    r"[a-z\u4e00-\u9fff]{1,16}(?:霉素|西林|沙星|普利|洛尔|地平|唑|酮|芬|酚|胺|定|松)"
)
_BENIGN_CONSUMPTION_PATTERN = re.compile(
    r"(?:饭|菜|肉|鱼|蛋|奶|茶|咖啡|水果|蔬菜|零食|薯片|饼干|面包|蛋糕|"
    r"巧克力|冰淇淋|糖果|粥|矿泉水|饮用水)"
)

_RED_FLAG_PATTERNS = (
    re.compile(r"(?:胸(?:口|部)?|心口)(?:疼(?:痛)?|痛|闷|憋|有?压迫感|压迫|紧缩|不适)"),
    re.compile(r"(?:没法|无法|不能)呼吸"),
    re.compile(r"呼吸(?:困难|急促|费力|不上来|不过来|不了)"),
    re.compile(r"(?:气喘不上来|喘不上气|喘不过气|透不过气|窒息)"),
    re.compile(
        r"(?:一只|一条|一边|一侧|单侧|半边|左侧|右侧|左|右).{0,4}"
        r"(?:手脚|手|脚|肢|臂|胳膊|腿|脸|面|身体).{0,6}"
        r"(?:无力|没力气|没有力气|使不上劲|麻木|发麻|麻|抬不起来|抬不动|动不了|歪斜|下垂)"
    ),
    re.compile(r"(?:嘴角|口角).{0,3}(?:歪|歪斜|下垂)"),
    re.compile(r"(?:口齿|言语|讲话|说话|发音).{0,5}(?:不清|含糊|不清楚|困难|说不出)"),
    re.compile(r"(?:昏迷|失去意识|意识不清|抽搐|大出血|血流不止|严重过敏)"),
    re.compile(r"(?:发烧|高烧|高热|体温).{0,5}(?:3[89]|4\d)(?:\.\d+)?度?"),
)
_TRIAGE_ACTION_PATTERN = re.compile(
    r"(?:怎么办|怎么(?:做|处理|急救)|如何(?:处理|急救)|该(?:怎么|做什么|如何)|"
    r"应该(?:怎么|做什么|如何|采取什么)|采取(?:什么|哪些).{0,5}(?:措施)?|"
    r"要(?:不要)?(?:去医院|就医|急诊|打120|叫救护车)|"
    r"(?:需要|要|能否|可以|是否|该(?:不该)?).{0,10}(?:冲(?:洗)?|催吐|就医|急诊|医院|打120|处理).{0,3}(?:吗|么)?|"
    r"(?:冲(?:洗)?|催吐|就医|急诊|打120|叫救护车).{0,3}(?:吗|么|\?|？))"
)

_HAZARD_CONTEXT_PATTERN = re.compile(
    r"(?:清洁|消毒|漂白|除垢|杀虫|除虫|农药|洁厕|洗涤|化学|试剂|溶剂|"
    r"强酸|强碱|腐蚀|有毒|84|喷雾|[\u4e00-\u9fff]{1,6}(?:剂|液|粉末))"
)
_UNKNOWN_SUBSTANCE_PATTERN = re.compile(
    r"(?:(?:这|那)(?:一)?(?:瓶|罐|杯|袋|种).{0,8}(?:液体|东西|产品|药水)|"
    r"(?:不明|未知|家用).{0,5}(?:液体|气体|粉末|产品))"
)
_EXPOSURE_ROUTE_PATTERNS = (
    re.compile(
        r"(?:(?:溅|滴|喷|弄|流|进|入|沾|接触).{0,6}(?:眼(?:里|睛|部)?|入眼|进眼)|"
        r"(?:眼(?:里|睛|部)?).{0,6}(?:溅|滴|喷|弄|流入|进入|接触))"
    ),
    re.compile(r"(?:误(?:吞|食|服|喝|饮)|不慎(?:吞|喝|饮)|(?:吞|咽|喝|饮|吃)(?:下|进|入|了))"),
    re.compile(r"(?:吸(?:入|进|到)|闻(?:入|了|到|见)|呛(?:入|到))"),
)
_STABLE_SAFETY_INFO_PATTERN = re.compile(
    r"(?:什么是|定义|原理|机制|为什么|为何|副作用|不良反应|禁忌|适应症|"
    r"常见原因|会有什么后果|有何后果|后果|危害|风险|刺激|科普|了解|"
    r"(?:主要)?作用|用途|做什么用|用来做什么|代表什么|意味着什么|含义)"
)

_SAFETY_ACTIONABLE = "actionable"
_SAFETY_STABLE = "stable"


def _sentence_start(text: str, index: int) -> int:
    boundary = max((text.rfind(marker, 0, index) for marker in _SENTENCE_BOUNDARIES), default=-1)
    return boundary + 1


def _sentence_end(text: str, index: int) -> int:
    boundaries = [text.find(marker, index) for marker in _SENTENCE_BOUNDARIES]
    valid = [boundary for boundary in boundaries if boundary >= 0]
    return min(valid) + 1 if valid else len(text)


def _quoted_spans(text: str) -> tuple[_SafetySpan, ...]:
    """Return quote offsets, including recoverable unmatched quote spans."""
    spans: list[_SafetySpan] = []
    stack: list[tuple[str, int]] = []
    for index, character in enumerate(text):
        if character in _QUOTE_PAIRS:
            stack.append((character, index))
            continue
        opener = _QUOTE_OPEN_FOR_CLOSE.get(character)
        if opener is None:
            continue
        opening_index = next(
            (position for position in range(len(stack) - 1, -1, -1) if stack[position][0] == opener),
            None,
        )
        if opening_index is not None:
            _, start = stack[opening_index]
            del stack[opening_index:]
            spans.append(_SafetySpan(start, index + 1))
            continue
        tail = text[index + 1 : _sentence_end(text, index + 1)]
        if any(marker in tail for marker in _QUOTE_META_MARKERS):
            spans.append(_SafetySpan(_sentence_start(text, index), index + 1))

    for _, start in stack:
        spans.append(_SafetySpan(start, _sentence_end(text, start + 1)))

    for quote in ('"', "'"):
        positions = [
            index
            for index, character in enumerate(text)
            if character == quote
            and not (
                quote == "'"
                and index > 0
                and index + 1 < len(text)
                and text[index - 1].isalnum()
                and text[index + 1].isalnum()
            )
        ]
        for offset in range(0, len(positions) - 1, 2):
            spans.append(_SafetySpan(positions[offset], positions[offset + 1] + 1))
        if len(positions) % 2:
            start = positions[-1]
            tail = text[start + 1 : _sentence_end(text, start + 1)]
            if any(marker in tail for marker in _QUOTE_META_MARKERS):
                spans.append(_SafetySpan(_sentence_start(text, start), start + 1))
            else:
                spans.append(_SafetySpan(start, _sentence_end(text, start + 1)))
    spans.extend(
        _SafetySpan(match.start(), match.end())
        for match in _BACKTICK_SPAN_PATTERN.finditer(text)
    )
    return tuple(spans)


def _is_affirmative_meta_declaration(clause: str) -> bool:
    """Only affirmative complete tail clauses may mask a safety question."""
    compact = "".join(
        character
        for character in clause
        if not character.isspace() and unicodedata.category(character) != "Cf"
    )
    return (
        _META_DECLARATION_PATTERN.search(compact) is not None
        and _META_DECLARATION_NEGATION_PATTERN.search(compact) is None
    )


def _meta_declaration_spans(text: str) -> tuple[_SafetySpan, ...]:
    """Return explicit meta clauses and their immediately preceding question."""
    spans: list[_SafetySpan] = []
    clause_start = 0
    preceding_question_boundary: int | None = None
    for boundary in _RELATION_BOUNDARY_PATTERN.finditer(text):
        clause_end = boundary.start()
        clause = text[clause_start:clause_end]
        if _is_affirmative_meta_declaration(clause):
            span_start = (
                _sentence_start(text, preceding_question_boundary)
                if preceding_question_boundary is not None
                else clause_start
            )
            spans.append(_SafetySpan(span_start, clause_end))
        preceding_question_boundary = (
            boundary.start() if boundary.group() in {"?", "？"} else None
        )
        clause_start = boundary.end()
    if _is_affirmative_meta_declaration(text[clause_start:]):
        span_start = (
            _sentence_start(text, preceding_question_boundary)
            if preceding_question_boundary is not None
            else clause_start
        )
        spans.append(_SafetySpan(span_start, len(text)))
    return tuple(spans)


def _absence_relation_spans(text: str) -> tuple[_SafetySpan, ...]:
    """Bind an absence auxiliary to predicates in its local relation only."""
    spans: list[_SafetySpan] = []
    for match in _ABSENCE_RELATION_PATTERN.finditer(text):
        prefix = text[max(0, match.start() - 4):match.start()]
        if re.search(r"(?:不是|并非)$", prefix):
            continue
        boundary = _RELATION_BOUNDARY_PATTERN.search(text, match.end())
        spans.append(_SafetySpan(match.start(), boundary.start() if boundary else len(text)))
    return tuple(spans)


def _safety_view(question: str) -> _SafetyView:
    normalized = unicodedata.normalize("NFKC", str(question or "")).casefold()
    quoted = _quoted_spans(normalized)
    active = list(normalized)
    for span in quoted:
        active[span.start : span.end] = " " * (span.end - span.start)
    unquoted = "".join(active)
    negated_meta = tuple(
        _SafetySpan(match.start(), match.end())
        for match in _NEGATED_META_PATTERN.finditer(unquoted)
    )
    declared_meta = _meta_declaration_spans(unquoted)
    meta = (*negated_meta, *declared_meta)
    for span in meta:
        active[span.start : span.end] = " " * (span.end - span.start)
    context_text = re.sub(r"\s+", "", "".join(active))
    absence = _absence_relation_spans(context_text)
    return _SafetyView(
        normalized_text=normalized,
        context_text=context_text,
        active_text=context_text,
        quoted_spans=quoted,
        negated_spans=(*negated_meta, *absence),
        meta_spans=meta,
        absence_spans=absence,
    )


def _predicate_is_explicitly_absent(
    text: str,
    start: int,
    end: int,
    absence_spans: tuple[_SafetySpan, ...] = (),
) -> bool:
    if any(span.start <= start and end <= span.end for span in absence_spans):
        return True
    prefix = text[max(0, start - 8) : start]
    suffix = text[end : end + 8]
    if re.search(r"(?:不是|并非)(?:没有|没|无|并无)$", prefix):
        return False
    if re.search(
        r"(?:(?:并)?(?:没有|没|并无|无)|从来没有|从未|未曾|不曾|否认有?)"
        r"(?:出现|发生|有|感觉到)?$",
        prefix,
    ):
        return True
    return bool(
        re.match(
            r"(?:(?:并)?(?:没有|没|未)|从来没有|从未|未曾|不曾)"
            r"(?:再)?(?:出现|发生)",
            suffix,
        )
    )


def _has_triage_action(text: str) -> bool:
    return bool(_TRIAGE_ACTION_PATTERN.search(text))


def _has_medication_administration(text: str) -> bool:
    generic_medication = bool(_GENERIC_MEDICATION_PATTERN.search(text))
    named_medication = bool(_LIKELY_MEDICATION_NAME_PATTERN.search(text))
    dose_matches = tuple(_DOSE_PATTERN.finditer(text))
    interval_matches = tuple(_INTERVAL_PATTERN.finditer(text))
    administration = bool(_ADMIN_PATTERN.search(text) or _MEDICATION_CHANGE_PATTERN.search(text))
    if not _MEDICATION_QUESTION_PATTERN.search(text):
        return False
    if generic_medication or named_medication:
        return administration or bool(dose_matches) or bool(interval_matches)
    if not dose_matches and not interval_matches:
        return False

    semantic_start = min(match.start() for match in (*dose_matches, *interval_matches))
    clause_start = max(text.rfind(marker, 0, semantic_start) for marker in "，,。；;！？!?") + 1
    subject_prefix = text[clause_start:semantic_start]
    benign_matches = tuple(_BENIGN_CONSUMPTION_PATTERN.finditer(subject_prefix))
    if benign_matches:
        subject_prefix = subject_prefix[benign_matches[-1].end() :]
        subject_prefix = re.sub(r"^(?:后|以后|之后|再)+", "", subject_prefix)
    plausible_subject = bool(re.search(r"[a-z\u4e00-\u9fff]{2,}", subject_prefix))
    if not plausible_subject:
        return False
    if _BENIGN_CONSUMPTION_PATTERN.search(text[semantic_start : semantic_start + 14]):
        return False
    pill_dose = bool(_PILL_DOSE_PATTERN.search(text))
    return (
        (pill_dose and bool(interval_matches))
        or (administration and pill_dose)
        or (administration and bool(_STRONG_DOSE_PATTERN.search(text)))
        or (administration and bool(interval_matches))
    )


def _active_red_flag_matches(
    text: str,
    absence_spans: tuple[_SafetySpan, ...] = (),
) -> tuple[re.Match[str], ...]:
    matches = [match for pattern in _RED_FLAG_PATTERNS for match in pattern.finditer(text)]
    return tuple(
        match
        for match in matches
        if not _predicate_is_explicitly_absent(
            text, match.start(), match.end(), absence_spans
        )
    )


def _mentions_red_flag(text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in _RED_FLAG_PATTERNS)


def _has_hazard_context(text: str) -> bool:
    return bool(_HAZARD_CONTEXT_PATTERN.search(text) or _UNKNOWN_SUBSTANCE_PATTERN.search(text))


def _has_active_exposure(
    text: str,
    absence_spans: tuple[_SafetySpan, ...] = (),
) -> bool:
    if not _has_hazard_context(text):
        return False
    for pattern in _EXPOSURE_ROUTE_PATTERNS:
        for match in pattern.finditer(text):
            if not _predicate_is_explicitly_absent(
                text, match.start(), match.end(), absence_spans
            ):
                return True
    return False


def _classify_safety_intent(question: str) -> str | None:
    """Classify coherent safety structures after quote and negation scoping."""
    view = _safety_view(question)
    text = view.active_text
    if _has_medication_administration(text):
        return _SAFETY_ACTIONABLE
    if _has_triage_action(text) and _active_red_flag_matches(text, view.absence_spans):
        return _SAFETY_ACTIONABLE
    if _has_triage_action(text) and _has_active_exposure(text, view.absence_spans):
        return _SAFETY_ACTIONABLE

    context_text = view.context_text
    safety_topic = (
        bool(
            _GENERIC_MEDICATION_PATTERN.search(context_text)
            or _LIKELY_MEDICATION_NAME_PATTERN.search(context_text)
        )
        or "副作用" in context_text
        or "不良反应" in context_text
        or _has_hazard_context(context_text)
        or _mentions_red_flag(context_text)
    )
    if safety_topic and _STABLE_SAFETY_INFO_PATTERN.search(context_text):
        return _SAFETY_STABLE
    return None


def _is_stable_or_nonpersonal_safety_text(question: str) -> bool:
    return _classify_safety_intent(question) == _SAFETY_STABLE


def _has_actionable_high_consequence_signal(question: str) -> bool:
    return _classify_safety_intent(question) == _SAFETY_ACTIONABLE


def _detect_personalized_high_consequence(question: str) -> TriggerCode | None:
    lowered = question.casefold()
    if _is_stable_or_nonpersonal_safety_text(question):
        return None
    if _has_actionable_high_consequence_signal(question):
        return TriggerCode.HIGH_CONSEQUENCE_ACTION
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
    # Dose/usage action phrases are personalized high-consequence even without
    # an explicit pronoun.
    if domain and any(phrase in lowered for phrase in _HIGH_CONSEQUENCE_ACTION_PHRASES):
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
    "药物",
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
    if _is_stable_or_nonpersonal_safety_text(question):
        return TriggerCode.REGULATED_DOMAIN_FOUNDATION
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
    # Explaining/rewriting the user's own text or current-conversation content
    # does not depend on external facts.
    closed_context = any(
        marker in question
        for marker in (
            "我刚才",
            "刚才贴",
            "刚才说",
            "这段话",
            "这段文字",
            "这段内容",
            "我贴的",
            "我发的",
            "我上面",
            "刚才那句话",
            "刚才那句",
        )
    )
    has_transform = any(marker in lowered for marker in _TRANSFORM_MARKERS)
    is_comparison = any(marker in lowered for marker in _COMPARISON_MARKERS)
    is_explanation = any(marker in lowered for marker in _EXPLANATION_MARKERS)
    if not (is_comparison or is_explanation):
        return None
    if not external_fact_required:
        return None
    if closed_context or has_transform:
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
    if _is_pure_greeting(question):
        return SkipReason.SOCIAL_OR_EMOTIONAL

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
class _RequestClassification:
    factuality: Factuality = Factuality.AMBIGUOUS
    external_fact_required: bool = True
    complexity_codes: tuple[RetrievalComplexityCode, ...] = ()
    source_requirement: SourceRequirement = SourceRequirement.ANY_RELEVANT
    freshness_requirement: FreshnessRequirement = FreshnessRequirement.NOT_REQUIRED
    as_of: date | None = None
    date_from: date | None = None
    date_to: date | None = None
    version_constraint: str | None = None
    high_consequence: bool = False
    warning_required: bool = False
    fail_closed: bool = False


# ── strict JSON parsing for the routing advisor ─────────────────────────

_FENCE_PATTERN = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)

_REQUEST_ANALYSIS_ENUM_FIELDS: Mapping[str, type[Any]] = {
    "factuality": Factuality,
    "source_requirement": SourceRequirement,
    "freshness_requirement": FreshnessRequirement,
}

_REQUEST_ANALYSIS_FIELDS = frozenset(
    {
        "factuality",
        "external_fact_required",
        "complexity_codes",
        "source_requirement",
        "freshness_requirement",
        "as_of",
        "date_from",
        "date_to",
        "version_constraint",
        "high_consequence",
        "warning_required",
        "fail_closed",
    }
)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _raise_invalid_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant: {value}")


def _parse_optional_date(value: Any, field_name: str) -> date | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{field_name} must be an ISO date or null")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError(f"{field_name} must be an ISO date or null")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date or null") from exc


def _normalized_advisor_output(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != _REQUEST_ANALYSIS_FIELDS:
        raise ValueError("request analysis must contain exactly the closed fields")

    normalized = dict(payload)
    for field_name, enum_type in _REQUEST_ANALYSIS_ENUM_FIELDS.items():
        parsed = _parse_enum(normalized[field_name], enum_type)
        if parsed is None:
            raise ValueError(f"invalid enum value: {field_name}")
        normalized[field_name] = parsed

    codes = normalized["complexity_codes"]
    if type(codes) is not list:
        raise ValueError("complexity_codes must be a JSON array")
    parsed_codes = _parse_enum_list(codes, RetrievalComplexityCode)
    if parsed_codes is None or len(set(parsed_codes)) != len(parsed_codes):
        raise ValueError("complexity_codes must be a unique closed enum array")
    normalized["complexity_codes"] = parsed_codes

    for field_name in (
        "external_fact_required",
        "high_consequence",
        "warning_required",
        "fail_closed",
    ):
        if type(normalized[field_name]) is not bool:
            raise ValueError(f"{field_name} must be a boolean")
    for field_name in ("as_of", "date_from", "date_to"):
        normalized[field_name] = _parse_optional_date(
            normalized[field_name],
            field_name,
        )
    version_constraint = normalized["version_constraint"]
    if version_constraint is not None and type(version_constraint) is not str:
        raise ValueError("version_constraint must be a string or null")
    return normalized


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


# ── one-shot request analysis ───────────────────────────────────────────

ROUTING_SYSTEM_PROMPT = """\
Classify the request into retrieval complexity, freshness needs, and answer
risk. You do not choose a search tier or a skip route. Treat the question text
as the only input.

Return JSON only with exactly these fields:
{
  "factuality": "non_factual|factual|mixed|ambiguous",
  "external_fact_required": true or false,
  "complexity_codes": ["multi_fact|multi_entity|comparison|recommendation|multi_source_required|cross_verification_required|ambiguous_entity"],
  "source_requirement": "any_relevant|independent_corroboration",
  "freshness_requirement": "not_required|current|as_of|window|version",
  "as_of": null or "YYYY-MM-DD",
  "date_from": null or "YYYY-MM-DD",
  "date_to": null or "YYYY-MM-DD",
  "version_constraint": null or "explicit version token",
  "high_consequence": false,
  "warning_required": false,
  "fail_closed": false
}

Never invent a source requirement that the user did not explicitly request.
"""


class LLMRequestAnalyzer:
    """One bounded LLM call merged with deterministic request constraints."""

    def __init__(self, llm: Any, *, max_tokens: int = 512) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def analyze(self, request: RetrievalRequest) -> RequestAnalysis:
        return _build_request_analysis(request, _validated_request_classification(self._call(request)))

    def _call(self, request: RetrievalRequest) -> dict[str, Any]:
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

_COMPLEXITY_TRIGGER_CODES: Mapping[RetrievalComplexityCode, TriggerCode] = {
    RetrievalComplexityCode.MULTI_FACT: TriggerCode.MULTI_HOP_COMPLEXITY,
    RetrievalComplexityCode.MULTI_ENTITY: TriggerCode.MULTI_HOP_COMPLEXITY,
    RetrievalComplexityCode.COMPARISON: TriggerCode.EXTERNAL_FACT_EXPLANATION_OR_COMPARISON,
    RetrievalComplexityCode.RECOMMENDATION: TriggerCode.RECOMMENDATION_OR_EVALUATION,
    RetrievalComplexityCode.MULTI_SOURCE_REQUIRED: TriggerCode.EXPLICIT_SOURCE_REQUEST,
    RetrievalComplexityCode.CROSS_VERIFICATION_REQUIRED: TriggerCode.EXPLICIT_VERIFICATION,
    RetrievalComplexityCode.AMBIGUOUS_ENTITY: TriggerCode.AMBIGUOUS_ENTITY,
}


def _retrieval_reason_codes(context: RetrievalContext) -> tuple[TriggerCode, ...]:
    codes = [
        _COMPLEXITY_TRIGGER_CODES[code]
        for code in context.complexity_codes
    ]
    if (
        context.source_requirement is SourceRequirement.INDEPENDENT_CORROBORATION
        and TriggerCode.EXPLICIT_SOURCE_REQUEST not in codes
        and TriggerCode.EXPLICIT_VERIFICATION not in codes
    ):
        codes.append(TriggerCode.EXPLICIT_SOURCE_REQUEST)
    return _dedupe_codes(codes)


def _skip_from_context(context: RetrievalContext) -> RetrievalDecision:
    if context.skip_reason is None:
        raise ValueError("skip context requires a skip_reason")
    return RetrievalDecision(
        route=SearchTier.SKIP,
        skip_reason=context.skip_reason,
        must_search=context.must_search,
        reason_codes=(),
    )


def _search_from_context(
    context: RetrievalContext,
    route: SearchTier,
) -> RetrievalDecision:
    reason_codes = tuple(context.complexity_codes)
    if context.source_requirement is SourceRequirement.INDEPENDENT_CORROBORATION:
        reason_codes = (*reason_codes, RetrievalComplexityCode.CROSS_VERIFICATION_REQUIRED)
    return RetrievalDecision(
        route=route,
        skip_reason=None,
        must_search=context.must_search,
        reason_codes=reason_codes,
    )


class RetrievalBenefitRouter:
    """Pure retrieval-context to route mapping; no LLM and no safety inputs."""

    def decide(self, context: RetrievalContext) -> RetrievalDecision:
        if not isinstance(context, RetrievalContext):
            raise TypeError("context must be a RetrievalContext")
        if context.skip_reason is SkipReason.USER_FORBID_WEB:
            return _skip_from_context(context)
        if context.skip_reason is not None and not context.must_search:
            return _skip_from_context(context)
        standard_reasons = {
            RetrievalComplexityCode.MULTI_FACT,
            RetrievalComplexityCode.MULTI_ENTITY,
            RetrievalComplexityCode.COMPARISON,
            RetrievalComplexityCode.RECOMMENDATION,
            RetrievalComplexityCode.MULTI_SOURCE_REQUIRED,
            RetrievalComplexityCode.CROSS_VERIFICATION_REQUIRED,
            RetrievalComplexityCode.AMBIGUOUS_ENTITY,
        }
        route = (
            SearchTier.STANDARD
            if standard_reasons.intersection(context.complexity_codes)
            or context.source_requirement is SourceRequirement.INDEPENDENT_CORROBORATION
            else SearchTier.LIGHT
        )
        return _search_from_context(context, route)


def _validated_request_classification(raw: Any) -> _RequestClassification:
    if not isinstance(raw, dict) or set(raw) != _REQUEST_ANALYSIS_FIELDS:
        return _RequestClassification()
    try:
        freshness = FreshnessContext(
            raw["freshness_requirement"],
            raw["as_of"],
            raw["date_from"],
            raw["date_to"],
            raw["version_constraint"],
        )
        risk = RiskContext(
            raw["high_consequence"],
            raw["warning_required"],
            raw["fail_closed"],
        )
        return _RequestClassification(
            factuality=raw["factuality"],
            external_fact_required=raw["external_fact_required"],
            complexity_codes=raw["complexity_codes"],
            source_requirement=raw["source_requirement"],
            freshness_requirement=freshness.requirement,
            as_of=freshness.as_of,
            date_from=freshness.date_from,
            date_to=freshness.date_to,
            version_constraint=freshness.version_constraint,
            high_consequence=risk.high_consequence,
            warning_required=risk.warning_required,
            fail_closed=risk.fail_closed,
        )
    except (KeyError, TypeError, ValueError):
        return _RequestClassification()


_MULTI_SOURCE_MARKERS = (
    "多个来源",
    "多来源",
    "两个来源",
    "两 个来源",
    "独立来源",
    "independent sources",
    "multiple sources",
)
_CROSS_VERIFICATION_MARKERS = (
    "交叉核验",
    "交叉验证",
    "相互核验",
    "互相核验",
    "cross verification",
)
_RECOMMENDATION_MARKERS = (
    "建议",
    "推荐",
    "是否应该",
    "应不应该",
    "应该买",
    "选哪个",
    "哪个好",
    "哪个更适合",
    "should i",
)
_MULTI_FACT_MARKERS = ("分别", "逐一", "一一", "同时说明")
_VERSION_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.])(?P<prefix>[vV]\s*)?(?P<token>\d{1,4}\.\d{1,4}(?:\.\d{1,4})?)(?![A-Za-z0-9.])"
)
_ASCII_ENTITY_BEFORE_VERSION = re.compile(
    r"(?:[A-Z][A-Za-z0-9_-]{1,30}|[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+)\s*$"
)
_CJK_ENTITY_BEFORE_VERSION = re.compile(r"(?P<entity>[\u4e00-\u9fff]{1,16})\s*$")
_MEASUREMENT_OR_CURRENCY_AFTER_NUMBER = re.compile(
    r"^\s*(?:mg|kg|g|ml|l|mm|cm|km|m|℃|°c|度|元|%)(?![A-Za-z])",
    re.IGNORECASE,
)
_GENERIC_CJK_VERSION_PREFIXES = (
    "今天",
    "价格",
    "剂量",
    "体温",
    "利率",
    "汇率",
)


def _dedupe_complexity_codes(
    codes: Sequence[RetrievalComplexityCode],
) -> tuple[RetrievalComplexityCode, ...]:
    seen: set[RetrievalComplexityCode] = set()
    result: list[RetrievalComplexityCode] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            result.append(code)
    return tuple(result)


def _source_requirement_from_question(
    question: str,
) -> tuple[SourceRequirement, tuple[RetrievalComplexityCode, ...]]:
    lowered = question.casefold()
    if any(marker in lowered for marker in _CROSS_VERIFICATION_MARKERS):
        return (
            SourceRequirement.INDEPENDENT_CORROBORATION,
            (RetrievalComplexityCode.CROSS_VERIFICATION_REQUIRED,),
        )
    if any(marker in lowered for marker in _MULTI_SOURCE_MARKERS):
        return (
            SourceRequirement.INDEPENDENT_CORROBORATION,
            (RetrievalComplexityCode.MULTI_SOURCE_REQUIRED,),
        )
    return SourceRequirement.ANY_RELEVANT, ()


def _deterministic_complexity_codes(
    question: str,
    model_codes: Sequence[RetrievalComplexityCode],
) -> tuple[RetrievalComplexityCode, ...]:
    lowered = question.casefold()
    source_requirement, source_codes = _source_requirement_from_question(question)
    del source_requirement
    codes = [
        code
        for code in model_codes
        if code
        not in {
            RetrievalComplexityCode.MULTI_SOURCE_REQUIRED,
            RetrievalComplexityCode.CROSS_VERIFICATION_REQUIRED,
        }
    ]
    if any(marker in lowered for marker in (*_COMPARISON_MARKERS, "比较", "比一比")):
        codes.append(RetrievalComplexityCode.COMPARISON)
    if any(marker in lowered for marker in _RECOMMENDATION_MARKERS):
        codes.append(RetrievalComplexityCode.RECOMMENDATION)
    if any(marker in lowered for marker in _MULTI_FACT_MARKERS):
        codes.append(RetrievalComplexityCode.MULTI_FACT)
    codes.extend(source_codes)
    return _dedupe_complexity_codes(codes)


def _extract_explicit_version_constraint(question: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", str(question or ""))
    for match in _VERSION_TOKEN_PATTERN.finditer(normalized):
        token = match.group("token")
        prefix = match.group("prefix")
        before = normalized[max(0, match.start() - 16):match.start()]
        after = normalized[match.end():match.end() + 16]
        has_version_label = bool(
            re.search(r"(?:版本|version)\s*$", before, re.IGNORECASE)
            or re.match(r"\s*版本", after)
            or re.match(r"\s*version\b", after, re.IGNORECASE)
        )
        if prefix or has_version_label:
            return token
        if _MEASUREMENT_OR_CURRENCY_AFTER_NUMBER.match(after):
            continue
        cjk_entity = _CJK_ENTITY_BEFORE_VERSION.search(before)
        has_adjacent_entity = bool(
            _ASCII_ENTITY_BEFORE_VERSION.search(before)
            or (
                cjk_entity is not None
                and not any(
                    cjk_entity.group("entity").endswith(generic_prefix)
                    for generic_prefix in _GENERIC_CJK_VERSION_PREFIXES
                )
            )
        )
        if has_adjacent_entity:
            return token
    return None


def _requires_current_freshness(question: str) -> bool:
    lowered = question.casefold()
    return bool(_detect_current_state(question)) or any(
        marker in lowered
        for marker in ("今晚", "今夜", "明晚", "明天", "明日")
    )


def _freshness_context(
    question: str,
    classification: _RequestClassification,
) -> FreshnessContext:
    version_constraint = _extract_explicit_version_constraint(question)
    if version_constraint is not None:
        return FreshnessContext(
            FreshnessRequirement.VERSION,
            None,
            None,
            None,
            version_constraint,
        )
    if _requires_current_freshness(question):
        return FreshnessContext(
            FreshnessRequirement.CURRENT,
            None,
            None,
            None,
            None,
        )
    if classification.freshness_requirement is FreshnessRequirement.VERSION:
        return FreshnessContext(
            FreshnessRequirement.NOT_REQUIRED,
            None,
            None,
            None,
            None,
        )
    return FreshnessContext(
        classification.freshness_requirement,
        classification.as_of,
        classification.date_from,
        classification.date_to,
        classification.version_constraint,
    )


def _risk_context(
    question: str,
    classification: _RequestClassification,
) -> RiskContext:
    deterministic_action = (
        _detect_personalized_high_consequence(question)
        is TriggerCode.HIGH_CONSEQUENCE_ACTION
    )
    lowered = question.casefold()
    symptom_directed_medication = (
        any(marker in lowered for marker in ("我的症状", "我症状", "我的病情"))
        and any(marker in lowered for marker in ("服用", "吃药", "用药"))
    )
    high_consequence = (
        deterministic_action
        or symptom_directed_medication
        or classification.high_consequence
    )
    return RiskContext(
        high_consequence,
        deterministic_action
        or symptom_directed_medication
        or classification.warning_required,
        deterministic_action
        or symptom_directed_medication
        or classification.fail_closed,
    )


def _build_request_analysis(
    request: RetrievalRequest,
    classification: _RequestClassification,
) -> RequestAnalysis:
    question = _question_text(request)
    explicit_search = _detect_explicit_search(question)
    explicit_verification = _detect_explicit_verification(question)
    explicit_source = _detect_explicit_source_request(question)
    must_search = bool(request.force_search) or any(
        (explicit_search, explicit_verification, explicit_source)
    )
    current_freshness = _requires_current_freshness(question)
    if _detect_explicit_no_web(question):
        skip_reason: SkipReason | None = SkipReason.USER_FORBID_WEB
    else:
        closed_task = _classify_closed_task(question)
        skip_reason = (
            closed_task
            if closed_task is not None and not must_search and not current_freshness
            else None
        )

    source_requirement, source_codes = _source_requirement_from_question(question)
    codes = _deterministic_complexity_codes(question, classification.complexity_codes)
    codes = _dedupe_complexity_codes((*codes, *source_codes))
    if skip_reason is SkipReason.USER_FORBID_WEB:
        factuality = Factuality.AMBIGUOUS
        external_fact_required = False
    elif skip_reason is not None:
        factuality = Factuality.NON_FACTUAL
        external_fact_required = False
    else:
        factuality = classification.factuality
        external_fact_required = True
    return RequestAnalysis(
        retrieval=RetrievalContext(
            must_search=must_search,
            skip_reason=skip_reason,
            factuality=factuality,
            external_fact_required=external_fact_required,
            complexity_codes=codes,
            source_requirement=source_requirement,
        ),
        freshness=_freshness_context(question, classification),
        risk=_risk_context(question, classification),
    )
