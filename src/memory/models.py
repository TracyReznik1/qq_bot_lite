from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryContext:
    user_id: str
    session_key: str
    is_group: bool
    group_id: str | None = None

    @property
    def primary_scope(self) -> tuple[str, str]:
        if self.is_group:
            return "group", str(self.group_id or "")
        return "private", self.user_id


@dataclass(frozen=True)
class MemoryEvent:
    context: MemoryContext
    message_id: str
    sequence: int
    text: str
    image_count: int = 0
    mentioned_qq_ids: tuple[str, ...] = ()
    reply_to_message_id: str | None = None
    reply_to_user_id: str | None = None
    prior_dialogue_context: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CandidateClaim:
    subject_ref: str
    predicate: str
    value: str
    memory_type: str
    modality: str
    confidence: str
    operation: str = "add"
    valid_from: str | None = None
    valid_to: str | None = None


@dataclass(frozen=True)
class MemoryClaim:
    id: int
    scope_type: str
    scope_id: str
    speaker_qq: str
    subject_type: str
    subject_id: str
    predicate: str
    value: str
    memory_type: str
    modality: str
    source_kind: str
    source_message_id: str
    source_excerpt: str
    extraction_confidence: str
    attribution_confidence: str
    truth_confidence: str
    status: str
    created_at: str
    valid_from: str | None
    valid_to: str | None
    last_confirmed_at: str | None
    dedupe_key: str


@dataclass(frozen=True)
class RetrievedMemory:
    claim: MemoryClaim
    score: float
    evidence_excerpts: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    usage: str = "evidence"


@dataclass(frozen=True)
class MemoryJob:
    id: int
    dedupe_key: str
    scope_key: str
    sequence: int
    payload_json: str
    context: MemoryContext
    message_id: str
    text: str
    image_count: int
    mentioned_qq_ids: tuple[str, ...]
    reply_to_message_id: str | None
    reply_to_user_id: str | None
    state: str
    attempts: int
    retry_at: str | None
    error_type: str | None
    created_at: str
    updated_at: str
    prior_dialogue_context: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class MemoryEvidence:
    id: int
    claim_id: int
    source_kind: str
    source_message_id: str
    source_excerpt: str
    created_at: str


@dataclass(frozen=True)
class MemoryRelation:
    source_claim_id: int
    target_claim_id: int
    relation_type: str
    created_at: str


@dataclass(frozen=True)
class PhysicalDeleteOutcome:
    status: str
    row_deleted: bool
    cleanup_complete: bool
    retryable: bool
