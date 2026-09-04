from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
import math
from typing import Mapping


class SearchMode(StrEnum):
    SKIP = "skip"
    LIGHT = "light"
    STANDARD = "standard"


class RequestSource(StrEnum):
    CHAT = "chat"
    COMMAND = "command"
    COMPATIBILITY = "compatibility"


class SearchFailure(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    NO_RESULTS = "no_results"


class OutputKind(StrEnum):
    PLAIN = "plain"
    MODEL_ANSWER = "model_answer"
    SUMMARY_FALLBACK = "summary_fallback"
    SEARCH_FAILURE = "search_failure"


@dataclass(frozen=True)
class SearchRequest:
    mode: SearchMode
    text: str
    images: tuple[str, ...] = ()
    source: RequestSource = RequestSource.CHAT

    def __post_init__(self) -> None:
        mode = SearchMode(self.mode)
        text = " ".join(str(self.text or "").split())
        images = tuple(
            normalized
            for item in self.images
            if (normalized := str(item or "").strip())
        )
        if not text and not images:
            raise ValueError("text or images must be provided")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "text", text[:500])
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "source", RequestSource(self.source))


@dataclass(frozen=True)
class SearchQuery:
    query_id: str
    text: str
    date_from: date | None = None
    date_to: date | None = None
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    news: bool = False

    def __post_init__(self) -> None:
        query_id = str(self.query_id or "").strip()
        text = " ".join(str(self.text or "").split())
        if not query_id or not text:
            raise ValueError("query id and text must be non-empty")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from cannot exceed date_to")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "text", text[:500])
        object.__setattr__(self, "include_domains", tuple(self.include_domains))
        object.__setattr__(self, "exclude_domains", tuple(self.exclude_domains))


@dataclass(frozen=True)
class SearchPlan:
    mode: SearchMode
    queries: tuple[SearchQuery, ...]
    planner_degraded: bool = False

    def __post_init__(self) -> None:
        mode = SearchMode(self.mode)
        queries = tuple(self.queries)
        if mode is SearchMode.SKIP and queries:
            raise ValueError("skip cannot carry queries")
        if mode is SearchMode.LIGHT and len(queries) != 1:
            raise ValueError("light requires exactly one query")
        if mode is SearchMode.STANDARD and not 1 <= len(queries) <= 3:
            raise ValueError("standard requires one to three queries")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "queries", queries)


@dataclass(frozen=True)
class SearchResult:
    result_id: str
    title: str
    url: str
    excerpt: str
    provider: str
    score: float = 0.5

    def __post_init__(self) -> None:
        score = float(self.score)
        score = min(max(score, 0.0), 1.0) if math.isfinite(score) else 0.5
        object.__setattr__(self, "score", score)


@dataclass
class SearchTrace:
    request_id: str
    source: RequestSource
    mode: SearchMode
    query_count: int = 0
    provider_statuses: dict[str, str] = field(default_factory=dict)
    candidate_count: int = 0
    reader_count: int = 0
    planner_degraded: bool = False
    ranker_degraded: bool = False
    answer_degraded: bool = False
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    output_kind: OutputKind = OutputKind.PLAIN

    def __post_init__(self) -> None:
        self.source = RequestSource(self.source)
        self.mode = SearchMode(self.mode)

    def to_safe_dict(self) -> Mapping[str, object]:
        return {
            "request_id": self.request_id,
            "source": self.source.value,
            "mode": self.mode.value,
            "query_count": self.query_count,
            "provider_statuses": dict(self.provider_statuses),
            "candidate_count": self.candidate_count,
            "reader_count": self.reader_count,
            "planner_degraded": self.planner_degraded,
            "ranker_degraded": self.ranker_degraded,
            "answer_degraded": self.answer_degraded,
            "stage_latency_ms": dict(self.stage_latency_ms),
            "output_kind": self.output_kind.value,
        }


@dataclass(frozen=True)
class SearchOutcome:
    plan: SearchPlan
    results: tuple[SearchResult, ...]
    trace: SearchTrace
    warning: str | None = None
    failure: SearchFailure | None = None


@dataclass(frozen=True)
class SearchResponse:
    text: str
    sources: tuple[SearchResult, ...]
    trace: SearchTrace
