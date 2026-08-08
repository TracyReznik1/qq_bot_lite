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

_RELATIVE_TIME_MARKERS = (
    "今天", "今日", "昨天", "昨日", "前天", "刚刚", "最近", "近期",
    "目前", "现在", "当前", "实时", "本周", "上周", "本月", "上月",
    "今年", "去年",
)

_CURRENT_RESULT_INTENTS = (
    "谁赢了", "赢了", "胜负", "比分", "赛果", "比赛结果", "排名",
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


# Dynamic-attribute words only force a DEEP/current-state floor when the
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
                # Only current-version/price questions trigger DEEP; a stable
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
    r"(?:(?<!不)(?:是|作为|属于|用于|叫做|命名为).{0,12}(?:测试|示例|用例|标题|代码|文本|例句)|"
    r"(?:测试|示例|用例|标题|代码|文本|例句).{0,6}(?:是|为|[:：])|"
    r"(?:这个|该)?(?:标题|示例|测试用例|代码)(?:是|为|[:：]))"
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
        r"(?:无力|没力气|使不上劲|麻木|发麻|麻|抬不起来|抬不动|动不了|歪斜|下垂)"
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


def _meta_declaration_spans(text: str) -> tuple[_SafetySpan, ...]:
    """Return explicit meta clauses and their immediately preceding question."""
    spans: list[_SafetySpan] = []
    clause_start = 0
    preceding_question_boundary: int | None = None
    for boundary in _RELATION_BOUNDARY_PATTERN.finditer(text):
        clause_end = boundary.start()
        clause = text[clause_start:clause_end]
        if _META_DECLARATION_PATTERN.search(clause):
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
    if _META_DECLARATION_PATTERN.search(text[clause_start:]):
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
    high_consequence: bool = False,
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
        risk=RiskLevel.HIGH if high_consequence else RiskLevel.LOW,
        actionability=(
            Actionability.PERSONALIZED if high_consequence else Actionability.NONE
        ),
        potential_harm=(
            PotentialHarm.HIGH if high_consequence else PotentialHarm.NONE
        ),
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

        # Explicit no-web is the user's hard constraint. Any explicit search
        # signal — including force_search from the /search command — turns this
        # into the deterministic clarification conflict, never a forced SKIP.
        if explicit_no_web:
            conflict = explicit_any or has_force_search
            high_consequence = (
                _detect_personalized_high_consequence(question)
                is TriggerCode.HIGH_CONSEQUENCE_ACTION
            )
            trigger_codes = [TriggerCode.EXPLICIT_NO_WEB]
            if conflict:
                trigger_codes.append(TriggerCode.EXPLICIT_SEARCH)
            if high_consequence:
                trigger_codes.append(TriggerCode.HIGH_CONSEQUENCE_ACTION)
            return _skip_decision(
                request,
                SkipReason.USER_FORBID_WEB,
                tuple(trigger_codes),
                forced_search=has_force_search,
                high_consequence=high_consequence,
            )

        raw = self._advisor.advise(request)
        classification = _validated_classification(raw)
        valid_advisor = bool(raw)

        # Compute forced / dynamic / high-consequence / mixed floors BEFORE any
        # closed-task skip, so a mixed request can never skip retrieval.
        floors, floor_codes = _compute_floors(
            question,
            classification,
            explicit_verification=explicit_verification,
            explicit_source=explicit_source,
        )
        if not valid_advisor:
            # Classifier uncertainty must not silently under-route a request
            # that carries high-consequence or current-state domain signals.
            conservative = _conservative_uncertain_floor(question)
            floors = _max_tier(floors, conservative) if conservative is not None else floors
            uncertain_codes = [*floor_codes, TriggerCode.CLASSIFIER_UNCERTAIN]
            if (
                conservative is SearchTier.DEEP
                and _classify_safety_intent(question) == _SAFETY_ACTIONABLE
            ):
                uncertain_codes.append(TriggerCode.HIGH_CONSEQUENCE_ACTION)
            floor_codes = _dedupe_codes(tuple(uncertain_codes))
        floor = floors if floors is not None else SearchTier.LIGHT
        if not floor_codes:
            floor_codes = (TriggerCode.FACTUAL_DEFAULT,)
        deterministic_high_consequence = (
            TriggerCode.HIGH_CONSEQUENCE_ACTION in floor_codes
        )

        # A closed-task skip is only accepted when the whole request carries no
        # search trigger and no forced floor.
        program_skip = _classify_closed_task(question)
        if program_skip is not None and not has_force_search and not floors:
            trigger_codes = _dedupe_codes((*explicit_codes, *classification.trigger_codes))
            return _skip_decision(
                request,
                program_skip,
                trigger_codes,
                forced_search=False,
            )

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
            potential_harm=(
                PotentialHarm.HIGH
                if deterministic_high_consequence
                else classification.potential_harm
            ),
            program_minimum_tier=floor,
            model_recommended_tier=recommended,
            final_reason_codes=final_reason_codes,
        )


def _compute_floors(
    question: str,
    classification: _Classification,
    *,
    explicit_verification: bool,
    explicit_source: bool,
) -> tuple[SearchTier | None, tuple[TriggerCode, ...]]:
    """Return (forced_floor, reason_codes). None means no forced floor was found."""
    floor: SearchTier | None = None
    codes: list[TriggerCode] = []

    safety_intent = _classify_safety_intent(question)
    if classification.freshness is Freshness.HIGH:
        floor = _max_tier(floor, SearchTier.DEEP)
        codes.append(TriggerCode.FRESHNESS_MARKER)

    current_state_codes = (
        () if safety_intent == _SAFETY_STABLE else _detect_current_state(question)
    )
    if current_state_codes:
        floor = _max_tier(floor, SearchTier.DEEP)
        codes.extend(current_state_codes)

    high_consequence = _detect_personalized_high_consequence(question)
    if high_consequence is not None:
        floor = _max_tier(floor, SearchTier.DEEP)
        codes.append(high_consequence)

    regulated = _detect_regulated_foundation(question)
    if regulated is not None:
        floor = _max_tier(floor, SearchTier.STANDARD)
        codes.append(regulated)

    external_compare = _detect_external_explanation_or_comparison(
        question,
        classification.external_fact_required,
    )
    if external_compare is not None:
        floor = _max_tier(floor, SearchTier.STANDARD)
        codes.append(external_compare)

    if explicit_verification or explicit_source:
        floor = _max_tier(floor, SearchTier.STANDARD)
        if explicit_verification:
            codes.append(TriggerCode.EXPLICIT_VERIFICATION)
        if explicit_source:
            codes.append(TriggerCode.EXPLICIT_SOURCE_REQUEST)

    if floor is None:
        return None, tuple(codes)
    return floor, tuple(codes)


def _max_tier(current: SearchTier | None, candidate: SearchTier) -> SearchTier:
    if current is None:
        return candidate
    return current if _rank(current) >= _rank(candidate) else candidate


def _conservative_uncertain_floor(question: str) -> SearchTier | None:
    """When the classifier fails, requests that touch high-consequence or
    current-state domains must not be silently under-routed to light."""
    lowered = question.casefold()
    safety_intent = _classify_safety_intent(question)
    if safety_intent == _SAFETY_ACTIONABLE:
        return SearchTier.DEEP
    if safety_intent == _SAFETY_STABLE:
        return SearchTier.STANDARD
    if any(marker in lowered for marker in _HIGH_CONSEQUENCE_DOMAINS):
        if any(phrase in lowered for phrase in _HIGH_CONSEQUENCE_ACTION_PHRASES):
            return SearchTier.DEEP
        return SearchTier.STANDARD
    if _detect_current_state(question):
        return SearchTier.DEEP
    if any(marker in question for marker in _REGULATED_DOMAIN_FOUNDATION_WORDS):
        return SearchTier.STANDARD
    return None


def _rank(tier: SearchTier) -> int:
    return {SearchTier.SKIP: 0, SearchTier.LIGHT: 1, SearchTier.STANDARD: 2, SearchTier.DEEP: 3}[tier]
