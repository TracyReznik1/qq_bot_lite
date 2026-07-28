"""Deterministic admission, attribution, scope, and conflict policy."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from src.memory.models import CandidateClaim, MemoryClaim, MemoryEvent
from src.memory.store import MemoryStore


_QQ_REF_PATTERN = re.compile(r"qq:([0-9]{1,12})")
_QUESTION_PATTERN = re.compile(
    r"(?:[?？]\s*$|^\s*(?:我是谁|谁是|什么是|是不是|是否|为何|为什么|怎么))"
)
_JOKE_PATTERN = re.compile(
    r"(?:开玩笑|逗你的|骗你的|说着玩|玩笑而已|假的啦|just kidding)",
    re.IGNORECASE,
)
_QUOTE_DELIMITERS = {
    "“": "”",
    "「": "」",
    "『": "』",
    "‘": "’",
    "《": "》",
    "〈": "〉",
    "【": "】",
    '"': '"',
    "'": "'",
    "`": "`",
}
_BLOCK_QUOTE_START_PATTERN = re.compile(r"^\s*>")
_FIRST_PERSON_PATTERN = re.compile(r"(?:我们|我|本人|俺|咱们|咱)")
_CORRECTION_PATTERN = re.compile(
    r"(?:纠正|更正|改口|说错|其实|以前.*现在|"
    r"不是.*(?:而是|我叫|是))"
)
_WITHDRAWAL_PATTERN = re.compile(
    r"(?:收回|撤回|作废|不算数|取消.*(?:说法|观点))"
)
_CLAUSE_BREAK_PATTERN = re.compile(r"[。.!！？?；;\r\n]+")
_SUBCLAUSE_BREAK_PATTERN = re.compile(r"[，,]+")
_PARALLEL_CLAUSE_BREAK_PATTERN = re.compile(r"(?:同时|另外|此外|并且|而且|以及)")
_STANDALONE_CORRECTION_LEAD_PATTERN = re.compile(
    r"^\s*(?:我\s*)?(?:纠正|更正|改口|说错)(?:了)?"
    r"(?:一下|下)?"
    r"(?:(?:前面|刚才|之前)(?:的)?(?:说法|内容|话))?"
    r"\s*[:：]?\s*$"
)
_NEGATION_PATTERN = re.compile(
    r"(?:不再|不喜欢|不是|没有|从不|并非|没(?:有)?)"
)
_HEARSAY_PATTERN = re.compile(
    r"(?:听说|听.*说|据说|传闻|别人说|有人说)"
)
_UNCERTAINTY_PATTERN = re.compile(
    r"(?:可能|也许|大概|不确定|好像|似乎)"
)
_SECRET_PATTERN = re.compile(
    r"(?:"
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----"
    r"|data:image/[a-z0-9.+-]+;base64,"
    r"|\bBearer\s+[a-z0-9._~+/=-]+"
    r"|\bsk-[a-z0-9_-]{8,}"
    r"|\bgh[pousr]_[a-z0-9]{20,}"
    r"|\bAKIA[A-Z0-9]{16}\b"
    r"|(?:api[_ -]?key|secret|access[_ -]?token|"
    r"payment[_ -]?(?:token|credential)|password|passwd|"
    r"credential|cookie|otp|authorization|密钥|密码|口令|验证码|"
    r"支付密码|银行卡号|cvv|cvc)\s*(?:[:=：]|\bis\b|是|为)\s*\S+"
    r"|(?:iVBORw0KGgo|/9j/|R0lGOD(?:lh|dh)|UklGR)[a-z0-9+/=]{12,}"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_PAYMENT_NUMBER_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_SENSITIVE_PREDICATE_PARTS = frozenset(
    {
        "address",
        "home_address",
        "phone",
        "mobile",
        "email",
        "id_card",
        "passport",
        "health",
        "medical",
        "diagnosis",
        "salary",
        "bank",
        "payment",
        "credential",
        "password",
        "secret",
        "location_exact",
        "住址",
        "电话",
        "邮箱",
        "身份证",
        "病史",
        "工资",
        "银行卡",
    }
)
_EXCLUSIVE_PREDICATES = frozenset(
    {
        "name",
        "preferred_name",
        "age",
        "birthday",
        "birth_date",
        "occupation",
        "employer",
        "school",
        "marital_status",
        "home_city",
        "current_city",
        "favorite_sport",
    }
)
_PREDICATE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_MEMORY_TYPES = frozenset(
    {
        "identity",
        "preferred_name",
        "preference",
        "opinion",
        "event",
        "plan",
        "relationship",
        "fact",
    }
)
_MODALITIES = frozenset(
    {"asserted", "uncertain", "hearsay", "negated"}
)
_CONFIDENCES = frozenset({"low", "medium", "high"})
_OPERATIONS = frozenset({"add", "confirm", "supersede", "retract"})
MAX_CLAIMS_PER_MESSAGE = 16
_PREDICATE_TEXT_MARKERS = {
    "name": ("名字", "姓名", "叫"),
    "preferred_name": ("叫我", "称呼", "昵称"),
    "likes": ("喜欢", "爱好"),
    "owns": ("拥有", "我的", "属于"),
    "age": ("年龄", "岁"),
    "birthday": ("生日", "出生"),
    "birth_date": ("生日", "出生"),
    "occupation": ("职业", "工作"),
    "employer": ("公司", "雇主", "工作"),
    "school": ("学校", "就读"),
    "current_city": ("现居", "住在", "城市"),
    "home_city": ("家乡", "老家"),
}


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    claim: MemoryClaim
    attribution_source: str
    related_claim_ids: tuple[int, ...] = ()

    @property
    def claim_id(self) -> int:
        return self.claim.id


@dataclass(frozen=True)
class _ResolvedSubject:
    subject_type: str
    subject_id: str
    source: str
    confidence: str


class MemoryPolicy:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def apply(
        self,
        event: MemoryEvent,
        candidates: tuple[CandidateClaim, ...],
    ) -> tuple[PolicyDecision, ...]:
        return self._apply(event, candidates)

    def apply_command(
        self,
        event: MemoryEvent,
        candidates: tuple[CandidateClaim, ...],
    ) -> tuple[PolicyDecision, ...]:
        """Apply an explicit user memory command in its real caller scope."""
        return self._apply(
            event,
            candidates,
            allow_implicit_speaker=True,
        )

    def apply_global_command(
        self,
        event: MemoryEvent,
        candidates: tuple[CandidateClaim, ...],
        *,
        authorized: bool,
    ) -> tuple[PolicyDecision, ...]:
        """Apply an explicitly authorized command to the real global scope."""
        if not authorized:
            return ()
        return self._apply(
            event,
            candidates,
            scope_override=("global", "global"),
            allow_implicit_speaker=True,
        )

    def _apply(
        self,
        event: MemoryEvent,
        candidates: tuple[CandidateClaim, ...],
        *,
        scope_override: tuple[str, str] | None = None,
        allow_implicit_speaker: bool = False,
    ) -> tuple[PolicyDecision, ...]:
        if (
            len(candidates) > MAX_CLAIMS_PER_MESSAGE
            or self._reject_whole_message(event)
        ):
            return ()
        decisions: list[PolicyDecision] = []
        with self.store.reconciliation() as transaction:
            for candidate in candidates:
                if candidate.confidence == "low":
                    continue
                candidate = self._validated_candidate(candidate)
                if candidate is None:
                    continue
                candidate = self._authorized_candidate(event, candidate)
                if candidate is None:
                    continue
                if (
                    candidate.subject_ref == "speaker"
                    and not allow_implicit_speaker
                    and not _has_unquoted_first_person(event.text)
                ):
                    continue
                subject = self.resolve_subject(
                    event,
                    candidate.subject_ref,
                    ledger=transaction,
                )
                if subject is None:
                    continue
                if self.contains_hard_secret(candidate.value):
                    continue
                scope_type, scope_id = (
                    event.context.primary_scope
                    if scope_override is None
                    else scope_override
                )
                if (
                    event.context.is_group
                    and self.is_sensitive_personal(candidate)
                ):
                    continue
                decision = self.decide_against_existing(
                    event,
                    candidate,
                    subject,
                    scope_type,
                    scope_id,
                    ledger=transaction,
                )
                if decision is not None:
                    decisions.append(decision)
        return tuple(decisions)

    def resolve_subject(
        self,
        event: MemoryEvent,
        subject_ref: str,
        *,
        ledger=None,
    ) -> _ResolvedSubject | None:
        ledger = self.store if ledger is None else ledger
        if subject_ref == "speaker":
            speaker = str(event.context.user_id or "").strip()
            if not speaker:
                return None
            return _ResolvedSubject("qq_user", speaker, "speaker", "high")
        if subject_ref == "bot":
            return _ResolvedSubject("bot", "bot", "bot", "high")
        if subject_ref == "reply_target":
            reply_target = str(event.reply_to_user_id or "").strip()
            if not reply_target.isdigit():
                return None
            return _ResolvedSubject(
                "qq_user",
                reply_target,
                "reply_target",
                "high",
            )
        match = _QQ_REF_PATTERN.fullmatch(subject_ref)
        if match is None:
            return None
        qq_id = match.group(1)
        if qq_id in event.mentioned_qq_ids:
            return _ResolvedSubject("qq_user", qq_id, "mention", "high")
        if _has_explicit_qq_marker(event.text, qq_id):
            return _ResolvedSubject(
                "qq_user",
                qq_id,
                "explicit_qq",
                "high",
            )
        if qq_id == str(event.context.user_id):
            if _has_unquoted_first_person(event.text):
                return _ResolvedSubject(
                    "qq_user",
                    qq_id,
                    "speaker",
                    "high",
                )
            return None
        if self._has_unique_scoped_alias(event, qq_id, ledger=ledger):
            return _ResolvedSubject(
                "qq_user",
                qq_id,
                "alias",
                "high",
            )
        return None

    @staticmethod
    def contains_hard_secret(value: str) -> bool:
        text = str(value or "")
        if _SECRET_PATTERN.search(text):
            return True
        return any(
            _passes_luhn(match.group(0))
            for match in _PAYMENT_NUMBER_PATTERN.finditer(text)
        )

    @staticmethod
    def is_sensitive_personal(candidate: CandidateClaim) -> bool:
        predicate = candidate.predicate.casefold()
        return any(
            part in predicate for part in _SENSITIVE_PREDICATE_PARTS
        )

    def decide_against_existing(
        self,
        event: MemoryEvent,
        candidate: CandidateClaim,
        subject: _ResolvedSubject,
        scope_type: str,
        scope_id: str,
        *,
        ledger=None,
    ) -> PolicyDecision | None:
        ledger = self.store if ledger is None else ledger
        existing = self._active_claims(
            scope_type,
            scope_id,
            subject.subject_type,
            subject.subject_id,
            candidate.predicate,
            ledger=ledger,
        )
        same_speaker = tuple(
            claim
            for claim in existing
            if claim.speaker_qq == event.context.user_id
        )
        other_speakers = tuple(
            claim
            for claim in existing
            if claim.speaker_qq != event.context.user_id
        )
        exact_same_speaker = next(
            (
                claim
                for claim in same_speaker
                if claim.value == candidate.value
                and claim.modality == candidate.modality
            ),
            None,
        )

        if candidate.operation == "retract":
            target = next(
                (
                    claim
                    for claim in same_speaker
                    if claim.value == candidate.value
                ),
                None,
            )
            if target is None:
                return None
            retraction = self._create_claim(
                event,
                candidate,
                subject,
                scope_type,
                scope_id,
                status="retracted",
                ledger=ledger,
            )
            closed_at = candidate.valid_to or _utc_now()
            ledger.update_claim(
                target.id,
                status="retracted",
                valid_to=closed_at,
            )
            ledger.add_relation(
                retraction.id,
                target.id,
                "retracts",
            )
            return PolicyDecision(
                "retracted",
                retraction,
                subject.source,
                (target.id,),
            )

        if (
            candidate.valid_to is not None
            and candidate.memory_type != "preferred_name"
            and candidate.predicate != "preferred_name"
        ):
            archived_matches = ledger.find_claims_exact(
                scope_type=scope_type,
                scope_id=scope_id,
                statuses=("archived",),
                subject_type=subject.subject_type,
                subject_id=subject.subject_id,
                predicates=(candidate.predicate,),
                value=candidate.value,
                speaker_qq=str(event.context.user_id),
            )
            exact_archived = next(
                (
                    claim
                    for claim in archived_matches
                    if claim.modality == candidate.modality
                    and claim.valid_from == candidate.valid_from
                    and claim.valid_to == candidate.valid_to
                ),
                None,
            )
            if exact_archived is not None:
                ledger.add_evidence(
                    exact_archived.id,
                    source_kind=f"message:{subject.source}",
                    source_message_id=event.message_id,
                    source_excerpt=event.text,
                )
                archived = exact_archived
            elif (
                exact_same_speaker is None
                or not _claim_can_close_at(
                    exact_same_speaker,
                    candidate.valid_to,
                )
            ):
                archived = self._create_claim(
                    event,
                    candidate,
                    subject,
                    scope_type,
                    scope_id,
                    status="archived",
                    ledger=ledger,
                )
            else:
                ledger.add_evidence(
                    exact_same_speaker.id,
                    source_kind=f"message:{subject.source}",
                    source_message_id=event.message_id,
                    source_excerpt=event.text,
                )
                archived = ledger.update_claim(
                    exact_same_speaker.id,
                    status="archived",
                    valid_from=(
                        exact_same_speaker.valid_from
                        or candidate.valid_from
                    ),
                    valid_to=candidate.valid_to,
                )
            return PolicyDecision(
                "archived",
                archived,
                subject.source,
            )

        if exact_same_speaker is not None:
            ledger.add_evidence(
                exact_same_speaker.id,
                source_kind=f"message:{subject.source}",
                source_message_id=event.message_id,
                source_excerpt=event.text,
            )
            confirmed = ledger.update_claim(
                exact_same_speaker.id,
                last_confirmed_at=_utc_now(),
                truth_confidence=max(
                    exact_same_speaker.truth_confidence,
                    self._truth_confidence(event, candidate, subject),
                    key=_confidence_rank,
                ),
            )
            return PolicyDecision(
                "confirmed",
                confirmed,
                subject.source,
            )

        exact_other = tuple(
            claim
            for claim in other_speakers
            if claim.value == candidate.value
            and claim.modality == candidate.modality
        )

        superseded_targets: tuple[MemoryClaim, ...] = ()
        if candidate.operation == "supersede" and same_speaker:
            correction_clauses = _bound_operation_clauses(
                event,
                candidate,
                _CORRECTION_PATTERN,
                allow_explicit_lead=True,
            )
            named_targets = tuple(
                claim
                for claim in same_speaker
                if any(
                    claim.value in clause
                    for clause in correction_clauses
                )
            )
            if named_targets:
                superseded_targets = named_targets
            elif len(same_speaker) == 1:
                superseded_targets = same_speaker

        conflicting_other = tuple(
            claim
            for claim in other_speakers
            if self._claims_conflict(candidate, claim)
        )
        status = "disputed" if conflicting_other else "active"
        created = self._create_claim(
            event,
            candidate,
            subject,
            scope_type,
            scope_id,
            status=status,
            ledger=ledger,
        )

        for claim in exact_other:
            ledger.add_relation(
                created.id,
                claim.id,
                "supports",
            )
            if (
                created.modality == "asserted"
                and claim.modality == "asserted"
                and candidate.confidence == "high"
            ):
                ledger.update_claim(
                    claim.id,
                    truth_confidence="high",
                )

        for claim in superseded_targets:
            ledger.update_claim(
                claim.id,
                status="superseded",
                valid_to=candidate.valid_from or _utc_now(),
            )
            ledger.add_relation(
                created.id,
                claim.id,
                "supersedes",
            )

        for claim in conflicting_other:
            ledger.update_claim(claim.id, status="disputed")
            ledger.add_relation(
                created.id,
                claim.id,
                "contradicts",
            )

        if conflicting_other:
            action = "disputed"
        elif superseded_targets:
            action = "superseded"
        elif exact_other:
            action = "supported"
        else:
            action = "created"
        related = tuple(
            dict.fromkeys(
                claim.id
                for claims in (
                    exact_other,
                    superseded_targets,
                    conflicting_other,
                )
                for claim in claims
            )
        )
        return PolicyDecision(action, created, subject.source, related)

    def _reject_whole_message(self, event: MemoryEvent) -> bool:
        text = str(event.text or "").strip()
        if not text:
            return True
        return bool(
            _QUESTION_PATTERN.search(text)
            or _JOKE_PATTERN.search(text)
            or self.contains_hard_secret(text)
        )

    @staticmethod
    def _authorized_candidate(
        event: MemoryEvent,
        candidate: CandidateClaim,
    ) -> CandidateClaim | None:
        text = event.text
        if (
            candidate.operation == "retract"
            and not _has_bound_operation_clause(
                event,
                candidate,
                _WITHDRAWAL_PATTERN,
            )
        ):
            return None
        correction_is_bound = _has_bound_operation_clause(
            event,
            candidate,
            _CORRECTION_PATTERN,
            allow_explicit_lead=True,
        )
        if (
            candidate.modality == "asserted"
            and _NEGATION_PATTERN.search(text)
            and not correction_is_bound
        ):
            return None
        if (
            candidate.operation == "supersede"
            and not correction_is_bound
        ):
            return replace(candidate, operation="add")
        return candidate

    @staticmethod
    def _validated_candidate(
        candidate: CandidateClaim,
    ) -> CandidateClaim | None:
        if (
            not isinstance(candidate.subject_ref, str)
            or not isinstance(candidate.predicate, str)
            or not isinstance(candidate.value, str)
            or not isinstance(candidate.memory_type, str)
            or not isinstance(candidate.modality, str)
            or not isinstance(candidate.confidence, str)
            or not isinstance(candidate.operation, str)
        ):
            return None
        if (
            _PREDICATE_PATTERN.fullmatch(candidate.predicate) is None
            or not candidate.value.strip()
            or len(candidate.value) > 500
            or candidate.memory_type not in _MEMORY_TYPES
            or candidate.modality not in _MODALITIES
            or candidate.confidence not in _CONFIDENCES
            or candidate.operation not in _OPERATIONS
        ):
            return None
        try:
            valid_from = _normalize_temporal(candidate.valid_from)
            valid_to = _normalize_temporal(candidate.valid_to)
        except ValueError:
            return None
        if (
            valid_from is not None
            and valid_to is not None
            and datetime.fromisoformat(valid_from)
            > datetime.fromisoformat(valid_to)
        ):
            return None
        return replace(
            candidate,
            valid_from=valid_from,
            valid_to=valid_to,
        )

    def _active_claims(
        self,
        scope_type: str,
        scope_id: str,
        subject_type: str,
        subject_id: str,
        predicate: str,
        *,
        ledger=None,
    ) -> tuple[MemoryClaim, ...]:
        ledger = self.store if ledger is None else ledger
        return ledger.find_claims_exact(
            scope_type=scope_type,
            scope_id=scope_id,
            statuses=("active", "disputed"),
            subject_type=subject_type,
            subject_id=subject_id,
            predicates=(predicate,),
        )

    def _has_unique_scoped_alias(
        self,
        event: MemoryEvent,
        qq_id: str,
        *,
        ledger=None,
    ) -> bool:
        ledger = self.store if ledger is None else ledger
        scope_type, scope_id = event.context.primary_scope
        aliases: set[str] = set()
        for predicate in ("name", "preferred_name"):
            aliases.update(
                claim.value
                for claim in self._active_claims(
                    scope_type,
                    scope_id,
                    "qq_user",
                    qq_id,
                    predicate,
                    ledger=ledger,
                )
                if claim.value and claim.value in event.text
            )
        for alias in aliases:
            matches = ledger.find_claims_exact(
                scope_type=scope_type,
                scope_id=scope_id,
                statuses=("active", "disputed"),
                subject_type="qq_user",
                predicates=("name", "preferred_name"),
                value=alias,
            )
            subject_ids = {
                claim.subject_id
                for claim in matches
            }
            if subject_ids == {qq_id}:
                return True
        return False

    def _create_claim(
        self,
        event: MemoryEvent,
        candidate: CandidateClaim,
        subject: _ResolvedSubject,
        scope_type: str,
        scope_id: str,
        *,
        status: str = "active",
        ledger=None,
    ) -> MemoryClaim:
        ledger = self.store if ledger is None else ledger
        memory_type = (
            "opinion"
            if subject.subject_type == "bot"
            else candidate.memory_type
        )
        valid_to = (
            None
            if memory_type == "preferred_name"
            else candidate.valid_to
        )
        dedupe_body = "\0".join(
            (
                scope_type,
                scope_id,
                str(event.context.user_id),
                subject.subject_type,
                subject.subject_id,
                candidate.predicate,
                candidate.value,
                candidate.modality,
                candidate.operation,
                event.message_id,
            )
        )
        dedupe_key = "policy:" + hashlib.sha256(
            dedupe_body.encode("utf-8")
        ).hexdigest()
        claim, _created = ledger.create_claim(
            scope_type=scope_type,
            scope_id=scope_id,
            speaker_qq=str(event.context.user_id),
            subject_type=subject.subject_type,
            subject_id=subject.subject_id,
            predicate=candidate.predicate,
            value=candidate.value,
            memory_type=memory_type,
            modality=candidate.modality,
            source_kind=f"message:{subject.source}",
            source_message_id=event.message_id,
            source_excerpt=event.text,
            extraction_confidence=candidate.confidence,
            attribution_confidence=subject.confidence,
            truth_confidence=self._truth_confidence(
                event,
                candidate,
                subject,
            ),
            dedupe_key=dedupe_key,
            status=status,
            valid_from=candidate.valid_from,
            valid_to=valid_to,
        )
        return claim

    @staticmethod
    def _truth_confidence(
        event: MemoryEvent,
        candidate: CandidateClaim,
        subject: _ResolvedSubject,
    ) -> str:
        if candidate.confidence == "medium":
            return "medium"
        if candidate.modality in {"uncertain", "hearsay", "negated"}:
            return "medium"
        if (
            _HEARSAY_PATTERN.search(event.text)
            or _UNCERTAINTY_PATTERN.search(event.text)
        ):
            return "medium"
        if (
            subject.subject_type == "qq_user"
            and subject.subject_id != str(event.context.user_id)
        ):
            return "medium"
        return "high"

    @staticmethod
    def _claims_conflict(
        candidate: CandidateClaim,
        existing: MemoryClaim,
    ) -> bool:
        if candidate.value == existing.value:
            return candidate.modality != existing.modality
        return (
            candidate.operation == "supersede"
            or candidate.modality == "negated"
            or candidate.predicate in _EXCLUSIVE_PREDICATES
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _confidence_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(value, 0)


def _claim_can_close_at(claim: MemoryClaim, valid_to: str) -> bool:
    if claim.valid_from is None:
        return True
    try:
        normalized_start = _normalize_temporal(claim.valid_from)
    except ValueError:
        return False
    return datetime.fromisoformat(normalized_start) <= datetime.fromisoformat(
        valid_to
    )


def _passes_luhn(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _unquoted_text(value: str) -> str:
    lines: list[str] = []
    in_block_quote = False
    for line in value.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        newline = line[len(content):]
        if _BLOCK_QUOTE_START_PATTERN.search(content):
            in_block_quote = True
            lines.append(newline or " ")
        elif in_block_quote and content.strip():
            lines.append(newline or " ")
        else:
            if not content.strip():
                in_block_quote = False
            lines.append(line)
    unblocked = "".join(lines)

    unquoted: list[str] = []
    index = 0
    while index < len(unblocked):
        if unblocked.startswith("```", index):
            closing = unblocked.find("```", index + 3)
            if closing < 0:
                break
            unquoted.append(" ")
            index = closing + 3
            continue
        opener = unblocked[index]
        closer = _QUOTE_DELIMITERS.get(opener)
        if closer is None:
            unquoted.append(opener)
            index += 1
            continue
        closing = unblocked.find(closer, index + 1)
        if closing < 0:
            break
        unquoted.append(" ")
        index = closing + 1
    return "".join(unquoted)


def _has_unquoted_first_person(value: str) -> bool:
    return _FIRST_PERSON_PATTERN.search(_unquoted_text(value)) is not None


def _has_explicit_qq_marker(value: str, qq_id: str) -> bool:
    return (
        re.search(
            rf"(?i)(?<![a-z0-9])(?:qq(?:号|号码)?\s*(?:[:：#]\s*)?|@)"
            rf"{re.escape(qq_id)}(?!\d)",
            _unquoted_text(value),
        )
        is not None
    )


def _message_subclauses(value: str) -> tuple[str, ...]:
    return tuple(
        segment.strip()
        for clause in _CLAUSE_BREAK_PATTERN.split(value)
        for subclause in _SUBCLAUSE_BREAK_PATTERN.split(clause)
        for segment in _PARALLEL_CLAUSE_BREAK_PATTERN.split(subclause)
        if segment.strip()
    )


def _has_bound_operation_clause(
    event: MemoryEvent,
    candidate: CandidateClaim,
    operation_pattern: re.Pattern[str],
    *,
    allow_explicit_lead: bool = False,
) -> bool:
    return bool(
        _bound_operation_clauses(
            event,
            candidate,
            operation_pattern,
            allow_explicit_lead=allow_explicit_lead,
        )
    )


def _bound_operation_clauses(
    event: MemoryEvent,
    candidate: CandidateClaim,
    operation_pattern: re.Pattern[str],
    *,
    allow_explicit_lead: bool = False,
) -> tuple[str, ...]:
    clauses = _message_subclauses(event.text)
    matched: list[str] = []
    for index, clause in enumerate(clauses):
        if operation_pattern.search(clause) is None:
            continue
        if _clause_supports_candidate(event, candidate, clause):
            matched.append(clause)
        elif (
            allow_explicit_lead
            and _is_standalone_correction_lead(clause)
            and index + 1 < len(clauses)
            and _clause_supports_candidate(
                event,
                candidate,
                clauses[index + 1],
            )
        ):
            matched.append(clauses[index + 1])
    return tuple(dict.fromkeys(matched))


def _is_standalone_correction_lead(clause: str) -> bool:
    return _STANDALONE_CORRECTION_LEAD_PATTERN.fullmatch(
        _unquoted_text(clause)
    ) is not None


def _clause_supports_candidate(
    event: MemoryEvent,
    candidate: CandidateClaim,
    clause: str,
) -> bool:
    unquoted = _unquoted_text(clause)
    return (
        candidate.value in unquoted
        and _clause_supports_subject(event, candidate.subject_ref, unquoted)
        and _clause_supports_predicate(candidate.predicate, unquoted)
    )


def _clause_supports_subject(
    event: MemoryEvent,
    subject_ref: str,
    clause: str,
) -> bool:
    if subject_ref == "speaker":
        return _has_unquoted_first_person(clause)
    if subject_ref == "bot":
        return re.search(r"(?i)(?:你|bot|机器人|助手)", clause) is not None
    if subject_ref == "reply_target":
        return re.search(r"(?:他|她|对方|回复对象)", clause) is not None
    match = _QQ_REF_PATTERN.fullmatch(subject_ref)
    if match is None:
        return False
    qq_id = match.group(1)
    if qq_id == str(event.context.user_id):
        return _has_unquoted_first_person(clause)
    return _has_explicit_qq_marker(clause, qq_id)


def _clause_supports_predicate(predicate: str, clause: str) -> bool:
    markers = _PREDICATE_TEXT_MARKERS.get(predicate)
    if markers is not None:
        return any(marker in clause for marker in markers)
    return predicate.casefold() in clause.casefold()


def _normalize_temporal(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError("temporal value must be concise text")
    candidate = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        raise ValueError("temporal value must be ISO-8601") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("temporal value must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()
