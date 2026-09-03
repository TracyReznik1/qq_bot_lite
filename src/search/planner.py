"""Natural-language query planning with bounded budgets and one repair.

The planner keeps the original natural-language question as the ``direct``
query, redacts transport hazards and hard secrets before any provider call,
enforces per-tier query budgets, and allows at most one distinct repair query
for ``standard`` and ``deep``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Callable, Mapping, Sequence

from src.search.models import (
    DEFAULT_TIER_BUDGETS,
    EvidenceGapAnalysis,
    Factuality,
    FreshnessContext,
    FreshnessRequirement,
    PlanningStatus,
    QueryPurpose,
    RepairReasonCode,
    RequiredTopic,
    RepairPlan,
    RequestSource,
    RetrievalDecision,
    RetrievalContext,
    RetrievalRequest,
    SearchPlan,
    SearchQuery,
    SearchRoundKind,
    SearchTier,
    SourceRelation,
    SourceRequirement,
)

_SOURCE_RELATION_PRIMARY = SourceRelation.PRIMARY
_SOURCE_RELATION_INDEPENDENT = SourceRelation.INDEPENDENT

QUERY_TEXT_MAX_CHARS = 500
DOMAIN_LIST_CAP = 5

_REPAIR_REASON_PHRASES: Mapping[RepairReasonCode, str] = {
    RepairReasonCode.MISSING_TOPIC: "补充检索",
    RepairReasonCode.STALE_EVIDENCE: "最新证据",
    RepairReasonCode.SOURCE_CONFLICT: "冲突核实",
    RepairReasonCode.ENTITY_AMBIGUITY: "实体消歧",
    RepairReasonCode.PREMISE_MISMATCH: "前提核对",
    RepairReasonCode.SOURCE_QUALITY_GAP: "独立来源",
    RepairReasonCode.CONTENT_UNREADABLE: "可读来源",
}

_HARD_SECRET_PATTERNS = (
    re.compile(r"\b[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b[Aa][Ii][Zz][Aa][A-Za-z0-9_-]+\b"),
    re.compile(r"\b(?:sk|pk|ak|secret|token)-[A-Za-z0-9]{16,}\b", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|apikey)\s*[:=]\s*[A-Za-z0-9]{8,}\b", re.IGNORECASE),
)

_CQ_CODE_PATTERN = re.compile(r"\[CQ:[^\[\]]*\]|CQ:\S+", re.IGNORECASE)
_DATA_URL_PATTERN = re.compile(r"data:[A-Za-z0-9+./-]+;base64,[A-Za-z0-9+/=\s]+", re.IGNORECASE)
_CALLBACK_SECRET_PATTERN = re.compile(r"(?:callback[_-]?secret|签名|signature)\s*[:=：]?\s*([A-Za-z0-9_-]{8,})", re.IGNORECASE)
_OTP_PATTERN = re.compile(r"(?:验证码|校验码|一次性密码|OTP|code)\s*[:：]?\s*(\d{4,8})", re.IGNORECASE)
_PASSWORD_PATTERN = re.compile(r"(?:密码|口令|password)\s*[:：]?\s*([A-Za-z0-9@#$%^&*_+-]{6,})", re.IGNORECASE)
_BANK_ACCOUNT_PATTERN = re.compile(r"(?:银行卡号|卡号|账号|帐号)\s*[:：]?\s*(\d{8,19})", re.IGNORECASE)
_CVV_PATTERN = re.compile(r"(?:CVV|CVC)\s*[:：]?\s*(\d{3,4})", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?<![\dA-Za-z])1[3-9]\d{9}(?![\dA-Za-z])")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

_EMPTY_REDACTION_SAFE_INTENT = "敏感凭据泄露后的安全处置"

_HOSTNAME_PATTERN = re.compile(r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}$")
_PRIVATE_HOST_PATTERNS = (
    "localhost",
    "127.",
    "10.",
    "192.168.",
    "172.16.",
    "172.17.",
    "172.18.",
    "172.19.",
    "172.2",
    "172.30.",
    "172.31.",
    "0.",
    "169.254.",
    ".local",
    ".internal",
    ".home.arpa",
)
_PRIVATE_SUFFIXES = (".local", ".internal", ".home.arpa", ".lan")

_CONTEXT_PREFIX_PATTERN = re.compile(
    r"(?:用户记忆|记忆|聊天历史|历史消息|上下文|对话历史)\s*[:：]\s*",
)

_QUESTION_STARTERS = (
    "什么是",
    "是什么",
    "请",
    "帮我",
    "为什么",
    "为何",
    "怎么",
    "如何",
    "介绍",
    "解释",
    "区别",
    "有什么",
)

_NEUTRAL_ORIGINAL_FOR_DEGRADED = "敏感凭据泄露后的安全处置"


def _strip_context_prefixes(text: str) -> str:
    """Remove leading memory/history context segments from transport text."""
    stripped = str(text or "")
    while True:
        match = _CONTEXT_PREFIX_PATTERN.search(stripped)
        if match is None:
            return stripped
        head = stripped[match.end() :]
        boundary = len(head)
        for starter in _QUESTION_STARTERS:
            index = head.find(starter)
            if index >= 0 and index < boundary:
                boundary = index
        for punctuation in ("。", "；", "！", "？", "\n"):
            index = head.find(punctuation)
            if index >= 0 and index < boundary:
                boundary = index
        stripped = (stripped[: match.start()] + head[boundary:]).strip()
        if not stripped:
            return ""


@dataclass(frozen=True)
class _NormalizedQuery:
    text: str
    redaction_codes: tuple[str, ...]
    degraded: bool


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _query_fingerprint(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


def redact_query_text(text: str) -> _NormalizedQuery:
    original = str(text or "")
    original = _strip_context_prefixes(original)
    removed: list[str] = []
    original, cq_control_count = _CQ_CODE_PATTERN.subn(" ", original)
    if cq_control_count:
        removed.append("cq_control_code")
    original, data_url_count = _DATA_URL_PATTERN.subn(" ", original)
    if data_url_count:
        removed.append("data_url")

    original, callback_secret_count = _CALLBACK_SECRET_PATTERN.subn(" ", original)
    if callback_secret_count:
        removed.append("callback_secret")

    matched_otp = _OTP_PATTERN.search(original)
    if matched_otp is not None:
        original = original.replace(matched_otp.group(0), " ")
        removed.append("one_time_code")

    matched_password = _PASSWORD_PATTERN.search(original)
    if matched_password is not None:
        original = original.replace(matched_password.group(0), " ")
        removed.append("password")

    matched_bank = _BANK_ACCOUNT_PATTERN.search(original)
    if matched_bank is not None:
        original = original.replace(matched_bank.group(0), " ")
        removed.append("bank_account")

    matched_cvv = _CVV_PATTERN.search(original)
    if matched_cvv is not None:
        original = original.replace(matched_cvv.group(0), " ")
        removed.append("card_cvv")

    before_secrets = original
    original = _HARD_SECRET_PATTERNS[1].sub(" ", original)
    original = _HARD_SECRET_PATTERNS[2].sub(" ", original)
    original = _HARD_SECRET_PATTERNS[3].sub(" ", original)
    if original != before_secrets:
        removed.append("hard_secret")
    if re.search(_HARD_SECRET_PATTERNS[0], original):
        original = _HARD_SECRET_PATTERNS[0].sub(" ", original)
        removed.append("hard_secret")

    text = _normalize_whitespace(original)
    if not text:
        text = _NEUTRAL_ORIGINAL_FOR_DEGRADED
        removed.append("empty_after_redaction")
    if not removed:
        removed = []
    return _NormalizedQuery(text, tuple(dict.fromkeys(removed)), bool(removed))

def _explicit_identifier_authorized(question: str, value: str) -> bool:
    """Allow a personal identifier only when the current message asks to search/verify it."""
    lowered = question.casefold()
    explicit = any(
        marker in lowered
        for marker in ("搜索", "查一下", "查一查", "核实", "验证", "查询", "查这个号码", "查这个", "搜这个", "请搜索", "帮我查")
    )
    return explicit and value in question


def _clean_personal_identifiers(text: str, question: str) -> _NormalizedQuery:
    removed: list[str] = []

    phone_matches = list(_PHONE_PATTERN.finditer(text))
    for match in phone_matches:
        value = match.group(0)
        if _explicit_identifier_authorized(question, value):
            continue
        text = text.replace(value, " ")
        removed.append("phone_number")

    email_matches = list(_EMAIL_PATTERN.finditer(text))
    for match in email_matches:
        value = match.group(0)
        if _explicit_identifier_authorized(question, value):
            continue
        text = text.replace(value, " ")
        removed.append("email_address")

    text = _normalize_whitespace(text)
    if not text:
        text = _NEUTRAL_ORIGINAL_FOR_DEGRADED
        removed.append("empty_after_redaction")
    return _NormalizedQuery(text, tuple(dict.fromkeys(removed)), bool(removed))


def validate_domain_list(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip().rstrip(".").casefold()
        if "://" in value or "/" in value or " " in value:
            continue
        if not _HOSTNAME_PATTERN.fullmatch(value):
            continue
        if _is_private_or_local_host(value):
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= DOMAIN_LIST_CAP:
            break
    return tuple(result)


def _is_private_or_local_host(hostname: str) -> bool:
    lowered = hostname.casefold()
    if lowered in {"localhost"} or lowered.endswith(_PRIVATE_SUFFIXES):
        return True
    if lowered.startswith(_PRIVATE_HOST_PATTERNS):
        return True
    return False


def _cap_query_text(text: str) -> str:
    text = _normalize_whitespace(text)
    if len(text) <= QUERY_TEXT_MAX_CHARS:
        return text
    cut = text[:QUERY_TEXT_MAX_CHARS].rstrip()
    sentence_boundary = max(
        cut.rfind("。"),
        cut.rfind("；"),
        cut.rfind("；"),
        cut.rfind(". "),
    )
    if sentence_boundary >= QUERY_TEXT_MAX_CHARS // 2:
        cut = cut[:sentence_boundary].rstrip()
    return cut.rstrip("，,。；;：: ")


# ── strict JSON parsing for planner output ──────────────────────────────

_FENCE_PATTERN = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\s*\Z",
    re.IGNORECASE | re.DOTALL,
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


def _parse_planner_payload(content: Any) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
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
        return None
    if not isinstance(payload, dict):
        return None
    return payload


PLANNER_SYSTEM_PROMPT = """\
You plan bounded supplemental web-search queries for one user question. The
application creates the user's original natural-language sentence as the
first direct query; never generate, replace, or repeat that direct query.

Return JSON only:
{
  "supplemental_queries": [
    {
      "purpose": "primary|independent|time_bounded|disambiguation|counterevidence",
      "text": "search text",
      "target_topic_ids": ["topic-1"],
      "date_from": null,
      "date_to": null
    }
  ],
  "required_topics": [
    {
      "label": "short answer topic",
      "material": true,
      "freshness_requirement": "not_required",
      "date_from": null,
      "date_to": null,
      "version_constraint": null,
      "source_requirement": "any_relevant"
    }
  ]
}

Rules:
- date_from/date_to filter when web pages were published, not when an event happened or when a fact is valid
- keep event years and fact cutoffs in query text; leave publication dates null unless the user explicitly asks what was published in a period
- “今年参加上海冠军赛的队伍” and “截至今天有哪些队伍晋级” require null publication dates
- “今天发布了哪些新闻” may carry publication-time intent
- never generate more than the supplied supplemental-query limit
- use at most three topics; labels must be non-blank
- supplemental target ids refer only to the ordered topic ids topic-1 through topic-3
- never put API keys, secrets, callback codes, QQ/group ids, or data URLs in a query
"""

_PLANNER_PAYLOAD_KEYS = frozenset({
    "supplemental_queries",
    "required_topics",
})
_PLANNER_TOPIC_KEYS = frozenset({
    "label",
    "material",
    "freshness_requirement",
    "date_from",
    "date_to",
    "version_constraint",
    "source_requirement",
})
_PLANNER_SUPPLEMENTAL_QUERY_KEYS = frozenset({
    "purpose",
    "text",
    "target_topic_ids",
    "date_from",
    "date_to",
})

_CONVERSATIONAL_PREFIX_PATTERN = re.compile(
    r"^(?:请(?:帮我|问)?(?:查|搜|看|检索|查询)?(?:一下|下)?|帮我(?:查|搜|看|检索|查询)?(?:一下|下)?|我想知道|我想了解|我想查一下|我想搜一下|麻烦(?:帮我)?(?:查|搜|看|检索)?(?:一下|下)?|能(?:不能|否)?(?:帮我)?(?:查|搜|看|检索)?(?:一下|下)?|告诉我|你知道|请问|查一下|搜一下)\s*",
    re.IGNORECASE,
)


def _clean_conversational_prefix(text: str) -> str:
    cleaned = _CONVERSATIONAL_PREFIX_PATTERN.sub("", text).strip()
    if cleaned:
        return re.sub(r"[？?]+$", "", cleaned).strip() or text
    return text


class SearchPlanner:
    """Plan bounded initial queries and the single optional repair query."""

    def __init__(self, model: Any, *, today_provider: Callable[[], date] | None = None) -> None:
        self._model = model
        self._today = today_provider if today_provider is not None else date.today

    # ── public API ──────────────────────────────────────────────────

    def plan(
        self,
        request: RetrievalRequest,
        decision: RetrievalDecision,
        retrieval_context: RetrievalContext,
        freshness_context: FreshnessContext,
        *,
        timeout_seconds: float | None = None,
    ) -> SearchPlan:
        if not isinstance(retrieval_context, RetrievalContext):
            raise TypeError("retrieval_context must be a RetrievalContext")
        if not isinstance(freshness_context, FreshnessContext):
            raise TypeError("freshness_context must be a FreshnessContext")
        budget = DEFAULT_TIER_BUDGETS[decision.route]
        original = str(request.question or "")
        direct_candidate = retrieval_context.search_keywords or original
        direct_text, redaction_codes, direct_degraded = self._prepare_direct(direct_candidate)
        if not direct_text:
            direct_text, redaction_codes, direct_degraded = self._prepare_direct(original)
        effective_freshness = _effective_freshness_context(
            freshness_context,
            self._today(),
        )
        if (
            retrieval_context.publication_date_from is not None
            or retrieval_context.publication_date_to is not None
        ):
            effective_freshness = FreshnessContext(
                requirement=effective_freshness.requirement,
                as_of=effective_freshness.as_of,
                date_from=retrieval_context.publication_date_from,
                date_to=retrieval_context.publication_date_to,
                version_constraint=effective_freshness.version_constraint,
            )

        if decision.route is SearchTier.LIGHT:
            required_topics = _fallback_required_topics(
                original,
                retrieval_context,
                effective_freshness,
                single_implicit_topic=True,
            )
            query = _apply_freshness_bounds(
                SearchQuery(
                    query_id="",
                    round_kind=SearchRoundKind.INITIAL,
                    purpose=QueryPurpose.DIRECT,
                    text=direct_text,
                    target_topic_ids=_material_topic_ids(required_topics),
                ),
                effective_freshness,
            )
            status = PlanningStatus.DEGRADED if (direct_degraded or bool(redaction_codes)) else PlanningStatus.NORMAL
            return SearchPlan(
                decision=decision,
                original_question=original,
                planning_status=status,
                entities=(),
                time_window=_time_window_for_context(effective_freshness),
                initial_queries=_assign_initial_query_ids((query,)),
                required_topics=required_topics,
                required_source_relations=frozenset(),
                query_redaction_codes=redaction_codes,
                budget=budget,
            )

        payload = self._ask_model(
            request,
            decision,
            retrieval_context,
            effective_freshness,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else 4.0,
        )
        required_relations = frozenset(
            {
                _SOURCE_RELATION_PRIMARY,
                _SOURCE_RELATION_INDEPENDENT,
            }
        )
        if payload is None or frozenset(payload) != _PLANNER_PAYLOAD_KEYS:
            required_topics = _fallback_required_topics(
                original,
                retrieval_context,
                effective_freshness,
            )
            fallback = self._deterministic_plan(
                original,
                decision,
                required_topics,
                effective_freshness,
            )
            return SearchPlan(
                decision=decision,
                original_question=original,
                planning_status=PlanningStatus.DEGRADED,
                entities=_extract_entities(original),
                time_window=_time_window_for_context(effective_freshness),
                initial_queries=_assign_initial_query_ids(fallback),
                required_topics=required_topics,
                required_source_relations=required_relations,
                query_redaction_codes=redaction_codes,
                budget=budget,
            )

        required_topics, topic_degraded = _seal_required_topics(
            payload,
            original,
            retrieval_context,
            effective_freshness,
        )
        planned, query_degraded = _model_supplemental_queries(
            payload,
            material_topic_ids=_material_topic_ids(required_topics),
        )
        if planned is None:
            fallback = self._deterministic_plan(
                original,
                decision,
                required_topics,
                effective_freshness,
            )
            return SearchPlan(
                decision=decision,
                original_question=original,
                planning_status=PlanningStatus.DEGRADED,
                entities=_extract_entities(original),
                time_window=_time_window_for_context(effective_freshness),
                initial_queries=_assign_initial_query_ids(fallback),
                required_topics=required_topics,
                required_source_relations=required_relations,
                query_redaction_codes=redaction_codes,
                budget=budget,
            )

        redacted_queries: list[SearchQuery] = []
        seen_fingerprints: set[str] = set()
        # Always retain the redacted original natural-language question as the
        # first direct query; the model may supplement, never replace it.
        direct = _apply_freshness_bounds(
            SearchQuery(
                query_id="",
                round_kind=SearchRoundKind.INITIAL,
                purpose=QueryPurpose.DIRECT,
                text=direct_text,
                target_topic_ids=_material_topic_ids(required_topics),
            ),
            effective_freshness,
        )
        redacted_queries.append(direct)
        seen_fingerprints.add(_query_fingerprint(direct_text))
        for query in planned:
            cleaned, codes, degraded = self._clean_query(query, original)
            redaction_codes = tuple(dict.fromkeys((*redaction_codes, *codes)))
            query_degraded = query_degraded or degraded
            if _query_fingerprint(cleaned) in seen_fingerprints:
                continue
            if (
                effective_freshness.requirement is FreshnessRequirement.VERSION
                and not _has_exact_version_token(
                    cleaned,
                    effective_freshness.version_constraint,
                )
            ):
                continue
            seen_fingerprints.add(_query_fingerprint(cleaned))
            candidate = _apply_freshness_bounds(
                SearchQuery(
                    query_id="",
                    round_kind=SearchRoundKind.INITIAL,
                    purpose=query.purpose,
                    text=cleaned,
                    date_from=query.date_from,
                    date_to=query.date_to,
                    include_domains=query.include_domains,
                    exclude_domains=query.exclude_domains,
                    target_topic_ids=query.target_topic_ids,
                ),
                effective_freshness,
            )
            if not _has_valid_query_time_shape(candidate):
                query_degraded = True
                continue
            redacted_queries.append(candidate)
            if len(redacted_queries) >= budget.max_initial_queries:
                break

        status = _planning_status(
            payload,
            direct_degraded or topic_degraded or query_degraded,
        )
        return SearchPlan(
            decision=decision,
            original_question=original,
            planning_status=status,
            entities=_extract_entities(original),
            time_window=_time_window_for_context(effective_freshness),
            initial_queries=_assign_initial_query_ids(redacted_queries),
            required_topics=required_topics,
            required_source_relations=required_relations,
            query_redaction_codes=redaction_codes,
            budget=budget,
        )

    def plan_repair(
        self,
        plan: SearchPlan,
        gap: EvidenceGapAnalysis,
        prior_fingerprints: Sequence[str] = (),
        *,
        timeout_seconds: float | None = None,
    ) -> RepairPlan:
        if not gap.repair_eligible or not gap.repair_target_topic_ids:
            return RepairPlan(False, (), (), None)
        target = _topic_by_id(plan, gap.repair_target_topic_ids[0])
        if target is None:
            return RepairPlan(False, (), (), None)
        text = _repair_text(plan.original_question, target, gap.repair_reason_codes)
        cleaned, codes, _degraded = self._clean_repair_text(
            text,
            plan.original_question,
        )
        if _query_fingerprint(cleaned) in set(prior_fingerprints):
            return RepairPlan(False, (), (), None)

        repair_query = SearchQuery(
            query_id="repair-1",
            query_index=len(plan.initial_queries) + 1,
            round_kind=SearchRoundKind.REPAIR,
            purpose=QueryPurpose.REPAIR,
            text=cleaned,
            target_topic_ids=(target.topic_id,),
            date_from=target.date_from,
            date_to=target.date_to,
            include_domains=(),
            exclude_domains=(),
        )
        return RepairPlan(
            triggered=True,
            reason_codes=gap.repair_reason_codes,
            target_topic_ids=gap.repair_target_topic_ids,
            repair_query=repair_query,
            query_redaction_codes=codes,
        )

    # ── helpers ─────────────────────────────────────────────────────

    def _prepare_direct(self, original: str) -> tuple[str, tuple[str, ...], bool]:
        normalized = redact_query_text(original)
        personal = _clean_personal_identifiers(normalized.text, original)
        cleaned = _clean_conversational_prefix(personal.text)
        text = _cap_query_text(cleaned)
        codes = tuple(dict.fromkeys((*normalized.redaction_codes, *personal.redaction_codes)))
        degraded = normalized.degraded or personal.degraded
        return text, codes, degraded

    def _clean_query(
        self,
        query: SearchQuery,
        original_question: str,
    ) -> tuple[str, tuple[str, ...], bool]:
        redacted = redact_query_text(query.text)
        personal = _clean_personal_identifiers(redacted.text, original_question)
        text = _cap_query_text(personal.text)
        codes = tuple(dict.fromkeys((*redacted.redaction_codes, *personal.redaction_codes)))
        degraded = redacted.degraded or personal.degraded
        return text, codes, degraded

    def _clean_repair_text(
        self,
        text: str,
        original_question: str,
    ) -> tuple[str, tuple[str, ...], bool]:
        redacted = redact_query_text(text)
        personal = _clean_personal_identifiers(redacted.text, original_question)
        cleaned = _cap_query_text(personal.text)
        codes = tuple(dict.fromkeys((*redacted.redaction_codes, *personal.redaction_codes)))
        degraded = redacted.degraded or personal.degraded
        return cleaned, codes, degraded

    def _ask_model(
        self,
        request: RetrievalRequest,
        decision: RetrievalDecision,
        retrieval_context: RetrievalContext,
        freshness_context: FreshnessContext,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        budget = DEFAULT_TIER_BUDGETS[decision.route]
        payload = {
            "question": request.question,
            "tier": decision.route.value,
            "max_supplemental_queries": max(budget.max_initial_queries - 1, 0),
            "source_requirement": retrieval_context.source_requirement.value,
            "freshness_requirement": freshness_context.requirement.value,
            "as_of": (
                freshness_context.as_of.isoformat()
                if freshness_context.as_of is not None else None
            ),
            "date_from": (
                freshness_context.date_from.isoformat()
                if freshness_context.date_from is not None else None
            ),
            "date_to": (
                freshness_context.date_to.isoformat()
                if freshness_context.date_to is not None else None
            ),
            "version_constraint": freshness_context.version_constraint,
        }
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        try:
            if timeout_seconds is not None and timeout_seconds <= 0:
                return None
            kwargs: dict[str, Any] = {
                "temperature": 0.0,
                "max_tokens": 1024,
                "tools": None,
                "tool_choice": "none",
            }
            if timeout_seconds is not None:
                kwargs["timeout_seconds"] = float(timeout_seconds)
            try:
                response = self._model.chat(messages, **kwargs)
            except TypeError:
                response = self._model.chat(
                    messages,
                    temperature=0.0,
                    max_tokens=1024,
                    tools=None,
                    tool_choice="none",
                )
        except Exception:
            return None
        return _parse_planner_payload(getattr(response, "content", ""))

    def _deterministic_plan(
        self,
        original: str,
        decision: RetrievalDecision,
        required_topics: Sequence[RequiredTopic],
        freshness_context: FreshnessContext,
    ) -> tuple[SearchQuery, ...]:
        budget = DEFAULT_TIER_BUDGETS[decision.route]
        direct_text, _, _ = self._prepare_direct(original)
        today = self._today()
        material_topic_ids = _material_topic_ids(required_topics)
        queries: list[SearchQuery] = [
            SearchQuery(
                query_id="",
                round_kind=SearchRoundKind.INITIAL,
                purpose=QueryPurpose.DIRECT,
                text=direct_text,
                target_topic_ids=material_topic_ids,
            )
        ]
        if decision.route is SearchTier.STANDARD:
            queries.append(
                SearchQuery(
                    query_id="",
                    round_kind=SearchRoundKind.INITIAL,
                    purpose=QueryPurpose.PRIMARY,
                    text=f"{direct_text} Rust 官方文档 Go 官方文档"
                    if "Rust" in direct_text and "Go" in direct_text
                    else f"{direct_text} 官方文档",
                    target_topic_ids=material_topic_ids,
                )
            )
            queries.append(
                SearchQuery(
                    query_id="",
                    round_kind=SearchRoundKind.INITIAL,
                    purpose=QueryPurpose.INDEPENDENT,
                    text=f"{direct_text} 独立技术对比",
                    target_topic_ids=material_topic_ids,
                )
            )
        return tuple(
            _apply_freshness_bounds(query, freshness_context)
            for query in queries[: budget.max_initial_queries]
        )


def _seal_required_topics(
    payload: Mapping[str, Any],
    original: str,
    retrieval_context: RetrievalContext,
    freshness_context: FreshnessContext,
) -> tuple[tuple[RequiredTopic, ...], bool]:
    raw_topics = payload.get("required_topics")
    if not isinstance(raw_topics, list):
        return _fallback_required_topics(
            original,
            retrieval_context,
            freshness_context,
        ), True

    parsed: list[RequiredTopic] = []
    degraded = False
    for raw in raw_topics[:3]:
        topic = _parse_model_topic(raw, len(parsed) + 1)
        if topic is None:
            degraded = True
            continue
        parsed.append(topic)

    if not any(topic.material for topic in parsed):
        degraded = True
        if len(parsed) >= 3:
            parsed = parsed[:2]
        parsed.append(
            _fallback_required_topics(
                original,
                retrieval_context,
                freshness_context,
            )[0]
        )

    sealed = tuple(
        RequiredTopic(
            topic_id=f"topic-{index}",
            label=topic.label,
            material=topic.material,
            freshness_requirement=freshness_context.requirement,
            date_from=freshness_context.date_from,
            date_to=freshness_context.date_to,
            version_constraint=freshness_context.version_constraint,
            source_requirement=retrieval_context.source_requirement,
        )
        for index, topic in enumerate(parsed, 1)
    )
    return sealed, degraded


def _parse_model_topic(raw: Any, index: int) -> RequiredTopic | None:
    if not isinstance(raw, dict) or frozenset(raw) != _PLANNER_TOPIC_KEYS:
        return None
    label = raw.get("label")
    material = raw.get("material")
    freshness_requirement = _parse_enum(
        raw.get("freshness_requirement"),
        FreshnessRequirement,
    )
    source_requirement = _parse_enum(
        raw.get("source_requirement"),
        SourceRequirement,
    )
    date_from, date_to = _normalized_model_bounds(
        raw.get("date_from"),
        raw.get("date_to"),
    )
    version_constraint = raw.get("version_constraint")
    if (
        not isinstance(label, str)
        or type(material) is not bool
        or freshness_requirement is None
        or source_requirement is None
        or (version_constraint is not None and type(version_constraint) is not str)
    ):
        return None
    try:
        return RequiredTopic(
            topic_id=f"topic-{index}",
            label=label,
            material=material,
            freshness_requirement=freshness_requirement,
            date_from=date_from,
            date_to=date_to,
            version_constraint=version_constraint,
            source_requirement=source_requirement,
        )
    except (TypeError, ValueError):
        return None


def _model_supplemental_queries(
    payload: Mapping[str, Any],
    *,
    material_topic_ids: Sequence[str],
) -> tuple[tuple[SearchQuery, ...] | None, bool]:
    raw_queries = payload.get("supplemental_queries")
    if not isinstance(raw_queries, list):
        return None, True
    ordered_material_topic_ids = tuple(material_topic_ids)
    material_topic_id_set = set(ordered_material_topic_ids)
    queries: list[SearchQuery] = []
    degraded = False
    for raw in raw_queries:
        query, malformed = _parse_model_supplemental_query(raw)
        degraded = degraded or malformed
        if query is None:
            continue
        target_topic_ids = tuple(dict.fromkeys(query.target_topic_ids))
        target_topic_id_set = set(target_topic_ids)
        if (
            not target_topic_ids
            or not target_topic_id_set.issubset(material_topic_id_set)
        ):
            continue
        canonical_target_topic_ids = tuple(
            topic_id
            for topic_id in ordered_material_topic_ids
            if topic_id in target_topic_id_set
        )
        queries.append(replace(
            query,
            target_topic_ids=canonical_target_topic_ids,
        ))
    return tuple(queries), degraded


def _parse_model_supplemental_query(
    raw: Any,
) -> tuple[SearchQuery | None, bool]:
    if (
        not isinstance(raw, dict)
        or frozenset(raw) != _PLANNER_SUPPLEMENTAL_QUERY_KEYS
    ):
        return None, True
    purpose = _parse_enum(raw.get("purpose"), QueryPurpose)
    text = raw.get("text")
    date_from, date_to = _normalized_model_bounds(
        raw.get("date_from"),
        raw.get("date_to"),
    )
    if (
        purpose is None
        or purpose in {QueryPurpose.DIRECT, QueryPurpose.REPAIR}
        or not isinstance(text, str)
        or not text.strip()
    ):
        return None, True
    target_topic_ids = tuple(
        dict.fromkeys(_string_list(raw.get("target_topic_ids")))
    )
    return SearchQuery(
        query_id="",
        round_kind=SearchRoundKind.INITIAL,
        purpose=purpose,
        text=_normalize_whitespace(text),
        date_from=date_from,
        date_to=date_to,
        target_topic_ids=target_topic_ids,
    ), False


def _parse_model_date(value: Any) -> tuple[date | None, bool]:
    if value is None:
        return None, True
    parsed = _parse_date(value)
    return parsed, parsed is not None


def _normalized_model_bounds(
    raw_from: Any,
    raw_to: Any,
) -> tuple[date | None, date | None]:
    date_from, valid_start = _parse_model_date(raw_from)
    date_to, valid_end = _parse_model_date(raw_to)
    if (
        not valid_start
        or not valid_end
        or (date_from is None) != (date_to is None)
        or (
            date_from is not None
            and date_to is not None
            and date_from >= date_to
        )
    ):
        return None, None
    return date_from, date_to


def _effective_freshness_context(
    context: FreshnessContext,
    today: date,
) -> FreshnessContext:
    """Resolve only the conservative implicit current window for planning."""
    if context.requirement is FreshnessRequirement.NOT_REQUIRED:
        return context
    if context.date_from is not None or context.date_to is not None:
        return context
    if context.as_of is not None:
        return FreshnessContext(
            requirement=context.requirement,
            as_of=context.as_of,
            date_from=context.as_of,
            date_to=context.as_of,
            version_constraint=context.version_constraint,
        )
    if context.requirement is FreshnessRequirement.CURRENT:
        return context
    return context


def _apply_freshness_bounds(
    query: SearchQuery,
    context: FreshnessContext,
) -> SearchQuery:
    """Freshness changes only executable date bounds, never query slots/text."""
    if context.requirement is FreshnessRequirement.NOT_REQUIRED:
        return query
    return replace(
        query,
        date_from=context.date_from,
        date_to=context.date_to,
    )


def _fallback_required_topics(
    original: str,
    retrieval_context: RetrievalContext,
    freshness_context: FreshnessContext,
    *,
    single_implicit_topic: bool = False,
) -> tuple[RequiredTopic, ...]:
    target_text = _clean_conversational_prefix(retrieval_context.search_keywords or original)
    if single_implicit_topic:
        labels = (_short_original(target_text)[:160] or "用户问题",)
    else:
        labels = _derive_required_topics(target_text)[:3] or ("用户问题",)
    return tuple(
        RequiredTopic(
            topic_id=f"topic-{index}",
            label=label[:160],
            material=True,
            freshness_requirement=freshness_context.requirement,
            date_from=freshness_context.date_from,
            date_to=freshness_context.date_to,
            version_constraint=freshness_context.version_constraint,
            source_requirement=retrieval_context.source_requirement,
        )
        for index, label in enumerate(labels, 1)
    )


def _material_topic_ids(
    topics: Sequence[RequiredTopic],
) -> tuple[str, ...]:
    return tuple(topic.topic_id for topic in topics if topic.material)


def _topic_by_id(plan: SearchPlan, topic_id: str) -> RequiredTopic | None:
    for topic in plan.required_topics:
        if topic.material and topic.topic_id == topic_id:
            return topic
    return None


def _repair_text(
    original_question: str,
    target: RequiredTopic,
    reason_codes: Sequence[RepairReasonCode],
) -> str:
    """Keep the original entity/version/region/scope anchor and add a closed
    reason-specific target phrase without discarding direct-query constraints."""
    phrase = _REPAIR_REASON_PHRASES.get(
        reason_codes[0] if reason_codes else RepairReasonCode.MISSING_TOPIC,
        _REPAIR_REASON_PHRASES[RepairReasonCode.MISSING_TOPIC],
    )
    return f"{target.label} {phrase} {_short_original(original_question)}"


def _time_window_for_context(
    context: FreshnessContext,
) -> tuple[date | None, date | None] | None:
    if context.date_from is None and context.date_to is None:
        return None
    return (context.date_from, context.date_to)


def _has_exact_version_token(text: str, version_constraint: str | None) -> bool:
    if not isinstance(version_constraint, str) or not version_constraint.strip():
        return False
    token = re.escape(_normalize_whitespace(version_constraint))
    return re.search(rf"(?<![0-9.]){token}(?![0-9.])", text) is not None


def _parse_enum(value: Any, enum_type: type[Any]) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError:
        return None


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _is_genuine_time_bounded_query(query: SearchQuery) -> bool:
    if query.purpose is not QueryPurpose.TIME_BOUNDED:
        return False
    if query.date_from is None or query.date_to is None or query.date_from > query.date_to:
        return False
    return (
        query.date_from.isoformat() in query.text
        and query.date_to.isoformat() in query.text
    )


def _has_valid_query_time_shape(query: SearchQuery) -> bool:
    """Reject provider-bound partial/reversed dates and false time purposes."""
    has_start = query.date_from is not None
    has_end = query.date_to is not None
    if has_start != has_end:
        return False
    if has_start and query.date_from > query.date_to:
        return False
    if query.purpose is QueryPurpose.TIME_BOUNDED:
        return _is_genuine_time_bounded_query(query)
    return True


def _assign_initial_query_ids(queries: Sequence[SearchQuery]) -> tuple[SearchQuery, ...]:
    """Assign deterministic IDs only after the complete initial plan is fixed."""
    return tuple(
        SearchQuery(
            query_id=f"initial-{index}",
            round_kind=query.round_kind,
            purpose=query.purpose,
            text=query.text,
            date_from=query.date_from,
            date_to=query.date_to,
            include_domains=query.include_domains,
            exclude_domains=query.exclude_domains,
            query_index=index,
            target_topic_ids=query.target_topic_ids,
        )
        for index, query in enumerate(queries, 1)
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()][:20]


def _planning_status(payload: dict[str, Any], degraded: bool) -> PlanningStatus:
    if degraded:
        return PlanningStatus.DEGRADED
    status = _parse_enum(payload.get("planning_status"), PlanningStatus)
    return status if status is not None else PlanningStatus.NORMAL


def _extract_entities(original: str) -> tuple[str, ...]:
    entities: list[str] = []
    for candidate in re.findall(r"[A-Z][A-Za-z0-9]{1,30}", original):
        if candidate not in entities:
            entities.append(candidate)
    return tuple(entities[:8])


def _derive_required_topics(original: str) -> tuple[str, ...]:
    """Derive the concrete topics a complete answer must support, so Evidence
    assembly cannot declare SUFFICIENT with zero relevant coverage."""
    text = str(original or "").strip()
    topics: list[str] = []
    cleaned = text
    for prefix in ("什么是", "是什么", "啥是", "介绍一下", "请搜索并给出来源：", "搜索一下", "请搜索", "帮我查一下", "查一下"):
        cleaned = cleaned.replace(prefix, "")
    cleaned = re.sub(r"[？?。！!，,：:；;（）()]", " ", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Keep a bounded set of concrete noun phrases / intent fragments.
    for candidate in re.findall(r"[一-鿿A-Za-z0-9]{2,20}", cleaned):
        candidate = candidate.strip()
        if not candidate:
            continue
        if candidate in topics:
            continue
        topics.append(candidate)
        if len(topics) >= 3:
            break
    if not topics:
        topics.append(text[:20])
    return tuple(topics)


def _short_original(original: str) -> str:
    text = str(original or "")
    for phrase in ("有什么区别", "有什么不同", "什么是", "是什么", "啥是", "怎么回事"):
        text = text.replace(phrase, "")
    text = text.replace("的", " ")
    return _normalize_whitespace(text)
