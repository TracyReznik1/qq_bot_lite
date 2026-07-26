"""Untrusted structured-memory extraction through the dedicated LLM chain."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from src.memory.models import CandidateClaim, MemoryEvent
from src.services.llm_client import get_memory_llm_client


_SUBJECT_REF_PATTERN = re.compile(
    r"(?:speaker|bot|reply_target|unknown|qq:[0-9]{1,12})"
)
_PREDICATE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_FENCE_PATTERN = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
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
_REQUIRED_CLAIM_FIELDS = frozenset(
    {
        "subject_ref",
        "predicate",
        "value",
        "memory_type",
        "modality",
        "confidence",
        "operation",
    }
)
_OPTIONAL_CLAIM_FIELDS = frozenset({"valid_from", "valid_to"})
MAX_CLAIMS_PER_MESSAGE = 16

EXTRACTION_SYSTEM_PROMPT = """\
You extract untrusted candidate memories from one QQ message.
Return JSON only. Do not answer the user, use tools, search the web, adopt a
persona, or infer facts that the message does not support.

The only allowed output shape is:
{
  "claims": [
    {
      "subject_ref": "speaker|bot|qq:<number>|reply_target|unknown",
      "predicate": "short_snake_case",
      "value": "concise value",
      "memory_type": "identity|preferred_name|preference|opinion|event|plan|relationship|fact",
      "modality": "asserted|uncertain|hearsay|negated",
      "confidence": "low|medium|high",
      "operation": "add|confirm|supersede|retract",
      "valid_from": null,
      "valid_to": null
    }
  ]
}

Use the real QQ sender as subject_ref "speaker" for first-person statements.
Use "qq:<number>" only when the message metadata or text identifies that QQ.
Use "reply_target" only for an unambiguous reply target. Use "unknown" rather
than guessing an ambiguous pronoun, alias, quoted speaker, image owner, or
image subject. Images are current-message evidence only. Do not infer
ownership or identity from an image without accompanying text.
Questions, jokes, quotations, negation, uncertainty, and hearsay must retain
their modality and must never be upgraded to certain facts. A correction or
withdrawal may request supersede or retract, but the downstream policy makes
every storage and lifecycle decision. valid_from and valid_to must be null or
an ISO-8601 timestamp.
"""


class MemoryExtractionError(RuntimeError):
    """A classifiable failure that is safe for the background job runner."""

    def __init__(self, message: str, *, error_type: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class _OutputValidationError(ValueError):
    pass


class MemoryExtractor:
    def __init__(self, llm=None) -> None:
        self.llm = llm if llm is not None else get_memory_llm_client()

    def extract(
        self,
        event: MemoryEvent,
        image_data_urls: Iterable[str] = (),
    ) -> tuple[CandidateClaim, ...]:
        messages = self._messages(event, tuple(image_data_urls))
        response = self._chat(messages)
        try:
            return _parse_candidates(response.content)
        except _OutputValidationError as first_error:
            repair_messages = [
                *messages,
                {"role": "assistant", "content": str(response.content or "")},
                {
                    "role": "user",
                    "content": (
                        "validation_error: "
                        f"{first_error}. Return one corrected JSON object only."
                    ),
                },
            ]
            repaired = self._chat(repair_messages)
            try:
                return _parse_candidates(repaired.content)
            except _OutputValidationError as second_error:
                raise MemoryExtractionError(
                    "memory extractor returned invalid structured output",
                    error_type="invalid_output",
                ) from second_error

    def _chat(self, messages: list[dict[str, Any]]):
        return self.llm.chat(
            messages,
            temperature=0.0,
            tools=None,
            tool_choice="none",
        )

    @staticmethod
    def _messages(
        event: MemoryEvent,
        image_data_urls: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        metadata = {
            "sender_qq": event.context.user_id,
            "is_group": event.context.is_group,
            "group_id": event.context.group_id,
            "mentioned_qq_ids": list(event.mentioned_qq_ids),
            "has_reply_target": bool(event.reply_to_user_id),
            "reply_target_qq": event.reply_to_user_id,
            "message_text": event.text,
            "image_count": len(image_data_urls),
        }
        text = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        if image_data_urls:
            user_content: str | list[dict[str, Any]] = [
                {"type": "text", "text": text},
                *(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    }
                    for image_data_url in image_data_urls
                ),
            ]
        else:
            user_content = text
        return [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]


def _parse_candidates(content: str) -> tuple[CandidateClaim, ...]:
    text = str(content or "").strip()
    fenced = _FENCE_PATTERN.fullmatch(text)
    if fenced is not None:
        text = fenced.group("body").strip()
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda value: (_raise_invalid_constant(value)),
        )
    except (json.JSONDecodeError, _OutputValidationError) as error:
        raise _OutputValidationError("output is not strict JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"claims"}:
        raise _OutputValidationError(
            "top-level object must contain only claims"
        )
    raw_claims = payload["claims"]
    if not isinstance(raw_claims, list):
        raise _OutputValidationError("claims must be an array")
    if len(raw_claims) > MAX_CLAIMS_PER_MESSAGE:
        raise _OutputValidationError(
            f"claims must contain at most {MAX_CLAIMS_PER_MESSAGE} items"
        )

    claims: list[CandidateClaim] = []
    for index, raw_claim in enumerate(raw_claims):
        claims.append(_parse_claim(raw_claim, index))
    return tuple(claims)


def _parse_claim(raw_claim: Any, index: int) -> CandidateClaim:
    if not isinstance(raw_claim, dict):
        raise _OutputValidationError(f"claims[{index}] must be an object")
    fields = set(raw_claim)
    if not _REQUIRED_CLAIM_FIELDS.issubset(fields):
        missing = sorted(_REQUIRED_CLAIM_FIELDS - fields)
        raise _OutputValidationError(
            f"claims[{index}] missing fields: {','.join(missing)}"
        )
    allowed = _REQUIRED_CLAIM_FIELDS | _OPTIONAL_CLAIM_FIELDS
    if not fields.issubset(allowed):
        extra = sorted(fields - allowed)
        raise _OutputValidationError(
            f"claims[{index}] has extra fields: {','.join(extra)}"
        )

    subject_ref = _required_text(raw_claim, "subject_ref", index)
    predicate = _required_text(raw_claim, "predicate", index)
    value = _required_text(raw_claim, "value", index)
    memory_type = _required_text(raw_claim, "memory_type", index)
    modality = _required_text(raw_claim, "modality", index)
    confidence = _required_text(raw_claim, "confidence", index)
    operation = _required_text(raw_claim, "operation", index)
    valid_from = _optional_text(raw_claim, "valid_from", index)
    valid_to = _optional_text(raw_claim, "valid_to", index)

    if _SUBJECT_REF_PATTERN.fullmatch(subject_ref) is None:
        raise _OutputValidationError(
            f"claims[{index}].subject_ref is invalid"
        )
    if _PREDICATE_PATTERN.fullmatch(predicate) is None:
        raise _OutputValidationError(
            f"claims[{index}].predicate is invalid"
        )
    if len(value) > 500:
        raise _OutputValidationError(f"claims[{index}].value is too long")
    _require_enum(memory_type, _MEMORY_TYPES, "memory_type", index)
    _require_enum(modality, _MODALITIES, "modality", index)
    _require_enum(confidence, _CONFIDENCES, "confidence", index)
    _require_enum(operation, _OPERATIONS, "operation", index)
    valid_from = _normalize_temporal(valid_from, "valid_from", index)
    valid_to = _normalize_temporal(valid_to, "valid_to", index)
    if (
        valid_from is not None
        and valid_to is not None
        and datetime.fromisoformat(valid_from) > datetime.fromisoformat(valid_to)
    ):
        raise _OutputValidationError(
            f"claims[{index}] valid_from must not exceed valid_to"
        )
    return CandidateClaim(
        subject_ref=subject_ref,
        predicate=predicate,
        value=value,
        memory_type=memory_type,
        modality=modality,
        confidence=confidence,
        operation=operation,
        valid_from=valid_from,
        valid_to=valid_to,
    )


def _required_text(
    value: dict[str, Any],
    field: str,
    index: int,
) -> str:
    item = value[field]
    if not isinstance(item, str) or not item.strip():
        raise _OutputValidationError(
            f"claims[{index}].{field} must be non-empty text"
        )
    return item.strip()


def _optional_text(
    value: dict[str, Any],
    field: str,
    index: int,
) -> str | None:
    item = value.get(field)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip() or len(item) > 64:
        raise _OutputValidationError(
            f"claims[{index}].{field} must be null or concise text"
        )
    return item.strip()


def _require_enum(
    value: str,
    allowed: frozenset[str],
    field: str,
    index: int,
) -> None:
    if value not in allowed:
        raise _OutputValidationError(
            f"claims[{index}].{field} is invalid"
        )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _OutputValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _raise_invalid_constant(value: str):
    raise _OutputValidationError(f"invalid JSON constant: {value}")


def _normalize_temporal(
    value: str | None,
    field: str,
    index: int,
) -> str | None:
    if value is None:
        return None
    candidate = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        raise _OutputValidationError(
            f"claims[{index}].{field} must be a timezone-aware ISO timestamp"
        ) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _OutputValidationError(
            f"claims[{index}].{field} must be a timezone-aware ISO timestamp"
        )
    return parsed.astimezone(timezone.utc).isoformat()
