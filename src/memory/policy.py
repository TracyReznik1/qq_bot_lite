"""Deterministic admission, attribution, scope, and conflict policy."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone

from src.memory.models import CandidateClaim, MemoryClaim, MemoryEvent
from src.memory.store import MemoryStore


_QQ_REF_PATTERN = re.compile(r"qq:([0-9]+)")
_QUESTION_PATTERN = re.compile(
    r"(?:[?？]\s*$|^\s*(?:我是谁|谁是|什么是|是不是|是否|为何|为什么|怎么))"
)
_JOKE_PATTERN = re.compile(
    r"(?:开玩笑|逗你的|骗你的|说着玩|玩笑而已|假的啦|just kidding)",
    re.IGNORECASE,
)
_QUOTED_FIRST_PERSON_PATTERN = re.compile(
    r"[“\"「『][^”\"」』]*(?:我|我们)[^”\"」』]*[”\"」』]"
)
_CORRECTION_PATTERN = re.compile(
    r"(?:纠正|更正|改口|说错|其实|以前.*现在|"
    r"不是.*(?:而是|我叫|是))"
)
_WITHDRAWAL_PATTERN = re.compile(
    r"(?:收回|撤回|作废|不算数|取消.*(?:说法|观点))"
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
        if self._reject_whole_message(event):
            return ()
        decisions: list[PolicyDecision] = []
        for candidate in candidates:
            if candidate.confidence == "low":
                continue
            if not self._candidate_is_well_formed(candidate):
                continue
            candidate = self._authorized_candidate(event, candidate)
            if candidate is None:
                continue
            if (
                candidate.subject_ref == "speaker"
                and _QUOTED_FIRST_PERSON_PATTERN.search(event.text)
            ):
                continue
            subject = self.resolve_subject(event, candidate.subject_ref)
            if subject is None:
                continue
            if self.contains_hard_secret(candidate.value):
                continue
            scope_type, scope_id = event.context.primary_scope
            if event.context.is_group and self.is_sensitive_personal(candidate):
                continue
            decision = self.decide_against_existing(
                event,
                candidate,
                subject,
                scope_type,
                scope_id,
            )
            if decision is not None:
                decisions.append(decision)
        return tuple(decisions)

    def resolve_subject(
        self,
        event: MemoryEvent,
        subject_ref: str,
    ) -> _ResolvedSubject | None:
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
        if qq_id == str(event.context.user_id):
            return _ResolvedSubject("qq_user", qq_id, "speaker", "high")
        if qq_id in event.mentioned_qq_ids:
            return _ResolvedSubject("qq_user", qq_id, "mention", "high")
        if re.search(rf"(?<!\d){re.escape(qq_id)}(?!\d)", event.text):
            return _ResolvedSubject(
                "qq_user",
                qq_id,
                "explicit_qq",
                "high",
            )
        if self._has_unique_scoped_alias(event, qq_id):
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
    ) -> PolicyDecision | None:
        existing = self._active_claims(
            scope_type,
            scope_id,
            subject.subject_type,
            subject.subject_id,
            candidate.predicate,
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
            )
            closed_at = candidate.valid_to or _utc_now()
            self.store.update_claim(
                target.id,
                status="retracted",
                valid_to=closed_at,
            )
            self.store.add_relation(
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
            archived = self._create_claim(
                event,
                candidate,
                subject,
                scope_type,
                scope_id,
                status="archived",
            )
            return PolicyDecision(
                "archived",
                archived,
                subject.source,
            )

        if exact_same_speaker is not None:
            self.store.add_evidence(
                exact_same_speaker.id,
                source_kind=f"message:{subject.source}",
                source_message_id=event.message_id,
                source_excerpt=event.text,
            )
            confirmed = self.store.update_claim(
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
        if exact_other:
            supporting = self._create_claim(
                event,
                candidate,
                subject,
                scope_type,
                scope_id,
            )
            for claim in exact_other:
                self.store.add_relation(
                    supporting.id,
                    claim.id,
                    "supports",
                )
                if (
                    supporting.modality == "asserted"
                    and claim.modality == "asserted"
                    and candidate.confidence == "high"
                ):
                    self.store.update_claim(
                        claim.id,
                        truth_confidence="high",
                    )
            return PolicyDecision(
                "supported",
                supporting,
                subject.source,
                tuple(claim.id for claim in exact_other),
            )

        superseded_targets: tuple[MemoryClaim, ...] = ()
        if candidate.operation == "supersede" and same_speaker:
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
        )

        for claim in superseded_targets:
            self.store.update_claim(
                claim.id,
                status="superseded",
                valid_to=candidate.valid_from or _utc_now(),
            )
            self.store.add_relation(
                created.id,
                claim.id,
                "supersedes",
            )

        for claim in conflicting_other:
            self.store.update_claim(claim.id, status="disputed")
            self.store.add_relation(
                created.id,
                claim.id,
                "contradicts",
            )

        if conflicting_other:
            action = "disputed"
            related = tuple(claim.id for claim in conflicting_other)
        elif superseded_targets:
            action = "superseded"
            related = tuple(claim.id for claim in superseded_targets)
        else:
            action = "created"
            related = ()
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
            and _WITHDRAWAL_PATTERN.search(text) is None
        ):
            return None
        if (
            candidate.modality == "asserted"
            and _NEGATION_PATTERN.search(text)
            and _CORRECTION_PATTERN.search(text) is None
        ):
            return None
        if (
            candidate.operation == "supersede"
            and _CORRECTION_PATTERN.search(text) is None
        ):
            return replace(candidate, operation="add")
        return candidate

    @staticmethod
    def _candidate_is_well_formed(candidate: CandidateClaim) -> bool:
        if (
            not isinstance(candidate.subject_ref, str)
            or not isinstance(candidate.predicate, str)
            or not isinstance(candidate.value, str)
            or not isinstance(candidate.memory_type, str)
            or not isinstance(candidate.modality, str)
            or not isinstance(candidate.confidence, str)
            or not isinstance(candidate.operation, str)
        ):
            return False
        if (
            _PREDICATE_PATTERN.fullmatch(candidate.predicate) is None
            or not candidate.value.strip()
            or len(candidate.value) > 500
            or candidate.memory_type not in _MEMORY_TYPES
            or candidate.modality not in _MODALITIES
            or candidate.confidence not in _CONFIDENCES
            or candidate.operation not in _OPERATIONS
        ):
            return False
        return all(
            value is None
            or (
                isinstance(value, str)
                and len(value) <= 64
                and _is_iso_temporal(value)
            )
            for value in (candidate.valid_from, candidate.valid_to)
        )

    def _active_claims(
        self,
        scope_type: str,
        scope_id: str,
        subject_type: str,
        subject_id: str,
        predicate: str,
    ) -> tuple[MemoryClaim, ...]:
        candidates = self.store.search_claims(
            predicate,
            scope_type=scope_type,
            scope_id=scope_id,
            limit=512,
        )
        return tuple(
            claim
            for claim in candidates
            if claim.status in {"active", "disputed"}
            and claim.subject_type == subject_type
            and claim.subject_id == subject_id
            and claim.predicate == predicate
        )

    def _has_unique_scoped_alias(
        self,
        event: MemoryEvent,
        qq_id: str,
    ) -> bool:
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
                )
                if claim.value and claim.value in event.text
            )
        for alias in aliases:
            matches = self.store.search_claims(
                alias,
                scope_type=scope_type,
                scope_id=scope_id,
                limit=512,
            )
            subject_ids = {
                claim.subject_id
                for claim in matches
                if claim.status in {"active", "disputed"}
                and claim.subject_type == "qq_user"
                and claim.predicate in {"name", "preferred_name"}
                and claim.value == alias
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
    ) -> MemoryClaim:
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
        claim, _created = self.store.create_claim(
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


def _is_iso_temporal(value: str) -> bool:
    candidate = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        datetime.fromisoformat(candidate)
        return True
    except ValueError:
        try:
            date.fromisoformat(candidate)
            return True
        except ValueError:
            return False
