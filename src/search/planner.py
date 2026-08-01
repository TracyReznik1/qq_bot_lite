"""Natural-language query planning with bounded budgets and one repair.

The planner keeps the original natural-language question as the ``direct``
query, redacts transport hazards and hard secrets before any provider call,
enforces per-tier query budgets, and allows at most one distinct repair query
for ``standard`` and ``deep``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Mapping, Sequence

from src.search.models import (
    DEFAULT_TIER_BUDGETS,
    EvidenceGapAnalysis,
    Factuality,
    Freshness,
    PlanningStatus,
    QueryPurpose,
    RepairPlan,
    RequestSource,
    RetrievalDecision,
    RetrievalRequest,
    SearchPlan,
    SearchQuery,
    SearchRoundKind,
    SearchTier,
    SourceRelation,
)

_SOURCE_RELATION_PRIMARY = SourceRelation.PRIMARY
_SOURCE_RELATION_INDEPENDENT = SourceRelation.INDEPENDENT

QUERY_TEXT_MAX_CHARS = 500
DOMAIN_LIST_CAP = 5

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
You plan bounded web-search queries for one user question. Preserve the user's
original natural-language sentence for the direct query. Entity/time/intent
extraction only creates supplementary queries; it never replaces the original
with mechanical keywords.

Return JSON only:
{
  "planning_status": "normal",
  "entities": ["short entity names"],
  "time_window": [null or "YYYY-MM-DD", null or "YYYY-MM-DD"],
  "initial_queries": [
    {
      "purpose": "direct|primary|independent|time_bounded|disambiguation",
      "text": "search text",
      "include_domains": [],
      "exclude_domains": []
    }
  ]
}

Rules:
- never exceed the tier's initial-query budget (light 1, standard 3, deep 5)
- for high-freshness requests include a dated time_bounded query
- never put API keys, secrets, callback codes, QQ/group ids, or data URLs in a query
"""


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
        *,
        deadline: float | None = None,
    ) -> SearchPlan:
        budget = DEFAULT_TIER_BUDGETS[decision.route]
        original = str(request.question or "")
        direct_text, redaction_codes, direct_degraded = self._prepare_direct(original)
        required_topics = _derive_required_topics(original)

        if decision.route is SearchTier.LIGHT:
            query = SearchQuery(
                query_id="",
                round_kind=SearchRoundKind.INITIAL,
                purpose=QueryPurpose.DIRECT,
                text=direct_text,
            )
            status = PlanningStatus.DEGRADED if (direct_degraded or bool(redaction_codes)) else PlanningStatus.NORMAL
            return SearchPlan(
                decision=decision,
                original_question=original,
                planning_status=status,
                entities=(),
                time_window=None,
                initial_queries=_assign_initial_query_ids((query,)),
                required_topics=required_topics,
                required_source_relations=frozenset(),
                query_redaction_codes=redaction_codes,
                budget=budget,
            )

        payload = self._ask_model(request, decision, deadline=deadline)
        planned = _model_initial_queries(payload, budget)
        degraded_due_to_model = payload is None or planned is None
        required_relations = frozenset(
            {
                _SOURCE_RELATION_PRIMARY,
                _SOURCE_RELATION_INDEPENDENT,
            }
        )
        if planned is None:
            fallback = self._deterministic_plan(original, decision)
            return SearchPlan(
                decision=decision,
                original_question=original,
                planning_status=PlanningStatus.DEGRADED if degraded_due_to_model else PlanningStatus.NORMAL,
                entities=_extract_entities(original),
                time_window=_time_window_for_decision(original, decision, self._today()),
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
        redacted_queries.append(
            SearchQuery(
                query_id="",
                round_kind=SearchRoundKind.INITIAL,
                purpose=QueryPurpose.DIRECT,
                text=direct_text,
            )
        )
        seen_fingerprints.add(_query_fingerprint(direct_text))
        for query in planned:
            cleaned, codes, degraded = self._clean_query(query, original)
            redaction_codes = tuple(dict.fromkeys((*redaction_codes, *codes)))
            if _query_fingerprint(cleaned) in seen_fingerprints:
                continue
            seen_fingerprints.add(_query_fingerprint(cleaned))
            redacted_queries.append(
                SearchQuery(
                    query_id="",
                    round_kind=SearchRoundKind.INITIAL,
                    purpose=query.purpose,
                    text=cleaned,
                    date_from=query.date_from,
                    date_to=query.date_to,
                    include_domains=query.include_domains,
                    exclude_domains=query.exclude_domains,
                )
            )

        redacted_queries = redacted_queries[: budget.max_initial_queries]
        needs_time_bounded_query = (
            decision.route is SearchTier.DEEP
            and decision.freshness is Freshness.HIGH
            and not any(_is_genuine_time_bounded_query(query) for query in redacted_queries)
        )
        if needs_time_bounded_query:
            today = self._today()
            time_bounded = SearchQuery(
                query_id="",
                round_kind=SearchRoundKind.INITIAL,
                purpose=QueryPurpose.TIME_BOUNDED,
                text=f"{_deep_location_hint(direct_text)} {today.isoformat()} 新闻 重要事件",
                date_from=today,
                date_to=today,
            )
            if len(redacted_queries) >= budget.max_initial_queries:
                # Preserve the original direct question (especially CJK/no-space
                # text) and replace the final supplemental slot with a genuine
                # date-bounded query after final slot selection.
                replacement_index = next(
                    (
                        index
                        for index in range(len(redacted_queries) - 1, 0, -1)
                        if redacted_queries[index].purpose is not QueryPurpose.DIRECT
                    ),
                    len(redacted_queries) - 1,
                )
                time_bounded = SearchQuery(
                    query_id="",
                    round_kind=time_bounded.round_kind,
                    purpose=time_bounded.purpose,
                    text=time_bounded.text,
                    date_from=time_bounded.date_from,
                    date_to=time_bounded.date_to,
                )
                redacted_queries[replacement_index] = time_bounded
            else:
                redacted_queries.append(time_bounded)

        status = _planning_status(payload, direct_degraded or degraded_due_to_model)
        entities = _string_list(payload.get("entities"))
        time_window = _parse_time_window(payload.get("time_window"))
        return SearchPlan(
            decision=decision,
            original_question=original,
            planning_status=status,
            entities=tuple(entities),
            time_window=time_window,
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
        *,
        repair_already_planned: bool = False,
    ) -> RepairPlan:
        if repair_already_planned:
            return RepairPlan(triggered=False, gap_codes=gap.repair_reason_codes, repair_query=None)
        if plan.decision.route is SearchTier.LIGHT:
            return RepairPlan(triggered=False, gap_codes=gap.repair_reason_codes, repair_query=None)
        if not gap.repair_eligible:
            return RepairPlan(triggered=False, gap_codes=gap.repair_reason_codes, repair_query=None)
        if not gap.repair_reason_codes:
            return RepairPlan(triggered=False, gap_codes=(), repair_query=None)

        missing_topics = gap.missing_claim_topics
        topic = missing_topics[0] if missing_topics else "证据缺口"
        repair_text = f"{topic} {_short_original(plan.original_question)}"
        repaired, codes, degraded = self._clean_repair_text(repair_text, plan.original_question, plan.decision)
        fingerprints = {_query_fingerprint(query.text) for query in plan.initial_queries}
        if _query_fingerprint(repaired) in fingerprints:
            return RepairPlan(triggered=False, gap_codes=gap.repair_reason_codes, repair_query=None)

        repair_query = SearchQuery(
            query_id="repair-1",
            round_kind=SearchRoundKind.REPAIR,
            purpose=QueryPurpose.REPAIR,
            text=repaired,
            date_from=plan.time_window[0] if plan.time_window is not None else None,
            date_to=plan.time_window[1] if plan.time_window is not None else None,
        )
        return RepairPlan(
            triggered=True,
            gap_codes=gap.repair_reason_codes,
            repair_query=repair_query,
            query_redaction_codes=codes,
        )

    # ── helpers ─────────────────────────────────────────────────────

    def _prepare_direct(self, original: str) -> tuple[str, tuple[str, ...], bool]:
        normalized = redact_query_text(original)
        personal = _clean_personal_identifiers(normalized.text, original)
        text = _cap_query_text(personal.text)
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
        decision: RetrievalDecision,
    ) -> tuple[str, tuple[str, ...], bool]:
        del decision
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
        *,
        deadline: float | None = None,
    ) -> dict[str, Any] | None:
        budget = DEFAULT_TIER_BUDGETS[decision.route]
        payload = {
            "question": request.question,
            "tier": decision.route.value,
            "max_initial_queries": budget.max_initial_queries,
            "freshness": decision.freshness.value,
        }
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        try:
            response = self._model.chat(
                messages,
                temperature=0.0,
                max_tokens=1024,
                tools=None,
                tool_choice="none",
            )
        except Exception:
            return None
        return _parse_planner_payload(response.content)

    def _deterministic_plan(
        self,
        original: str,
        decision: RetrievalDecision,
    ) -> tuple[SearchQuery, ...]:
        budget = DEFAULT_TIER_BUDGETS[decision.route]
        direct_text, _, _ = self._prepare_direct(original)
        today = self._today()
        queries: list[SearchQuery] = [
            SearchQuery(
                query_id="",
                round_kind=SearchRoundKind.INITIAL,
                purpose=QueryPurpose.DIRECT,
                text=direct_text,
            )
        ]
        if decision.route is SearchTier.DEEP:
            location_hint = _deep_location_hint(direct_text)
            queries.append(
                SearchQuery(
                    query_id="",
                    round_kind=SearchRoundKind.INITIAL,
                    purpose=QueryPurpose.TIME_BOUNDED,
                    text=f"{location_hint} {today.isoformat()} 新闻 重要事件",
                    date_from=today,
                    date_to=today,
                )
            )
            queries.append(
                SearchQuery(
                    query_id="",
                    round_kind=SearchRoundKind.INITIAL,
                    purpose=QueryPurpose.PRIMARY,
                    text=f"{location_hint} {today.isoformat()} 官方 通报",
                )
            )
            queries.append(
                SearchQuery(
                    query_id="",
                    round_kind=SearchRoundKind.INITIAL,
                    purpose=QueryPurpose.INDEPENDENT,
                    text=f"{location_hint} {today.isoformat()} 新闻 重要事件 独立报道",
                )
            )
        elif decision.route is SearchTier.STANDARD:
            queries.append(
                SearchQuery(
                    query_id="",
                    round_kind=SearchRoundKind.INITIAL,
                    purpose=QueryPurpose.PRIMARY,
                    text=f"{direct_text} Rust 官方文档 Go 官方文档"
                    if "Rust" in direct_text and "Go" in direct_text
                    else f"{direct_text} 官方文档",
                )
            )
            queries.append(
                SearchQuery(
                    query_id="",
                    round_kind=SearchRoundKind.INITIAL,
                    purpose=QueryPurpose.INDEPENDENT,
                    text=f"{direct_text} 独立技术对比",
                )
            )
        return tuple(queries[: budget.max_initial_queries])


def _model_initial_queries(payload: Any, budget: Any) -> tuple[SearchQuery, ...] | None:
    if not isinstance(payload, dict):
        return None
    raw_queries = payload.get("initial_queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        return None
    queries: list[SearchQuery] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_queries):
        if index >= budget.max_initial_queries:
            break
        parsed = _parse_model_query(raw)
        if parsed is None:
            return None
        fingerprint = _query_fingerprint(parsed.text)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        queries.append(parsed)
    if not queries:
        return None
    return tuple(queries)


def _parse_model_query(raw: Any) -> SearchQuery | None:
    if not isinstance(raw, dict):
        return None
    purpose = _parse_enum(raw.get("purpose"), QueryPurpose)
    if purpose is None:
        return None
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    date_from = _parse_date(raw.get("date_from"))
    date_to = _parse_date(raw.get("date_to"))
    include = validate_domain_list(raw.get("include_domains") or ())
    exclude = validate_domain_list(raw.get("exclude_domains") or ())
    return SearchQuery(
        query_id="",
        round_kind=SearchRoundKind.INITIAL,
        purpose=purpose,
        text=_normalize_whitespace(text),
        date_from=date_from,
        date_to=date_to,
        include_domains=include,
        exclude_domains=exclude,
    )


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
        )
        for index, query in enumerate(queries, 1)
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()][:20]


def _parse_time_window(value: Any) -> tuple[date | None, date | None] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start = _parse_date(value[0])
    end = _parse_date(value[1])
    return (start, end)


def _planning_status(payload: dict[str, Any], degraded: bool) -> PlanningStatus:
    if degraded:
        return PlanningStatus.DEGRADED
    status = _parse_enum(payload.get("planning_status"), PlanningStatus)
    return status if status is not None else PlanningStatus.NORMAL


def _time_window_for_decision(
    original: str,
    decision: RetrievalDecision,
    today: date,
) -> tuple[date | None, date | None] | None:
    if decision.freshness is Freshness.HIGH:
        return (today, today)
    if "今天" in original or "最近" in original or "最新" in original:
        return (today, today)
    return None


def _extract_entities(original: str) -> tuple[str, ...]:
    entities: list[str] = []
    for candidate in re.findall(r"[A-Z][A-Za-z0-9]{1,30}", original):
        if candidate not in entities:
            entities.append(candidate)
    return tuple(entities[:8])


def _deep_location_hint(original: str) -> str:
    match = re.search(r"([一-鿿]{2,6}?)(?:今天|最近|目前|现在|最新)", original)
    if match is not None:
        return match.group(1)
    return str(original or "").strip()


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
    for phrase in ("有什么区别", "有什么不同", "是什么", "怎么回事"):
        text = text.replace(phrase, "")
    text = text.replace("的", " ")
    return _normalize_whitespace(text)
