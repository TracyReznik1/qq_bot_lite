from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from src.search.simple.models import SearchMode, SearchQuery


class ProviderStatus(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"


class ProviderErrorCode(StrEnum):
    INVALID_PARAMETERS = "invalid_parameters"
    CONNECTION = "connection"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderReadiness:
    provider: str
    configured: bool
    available: bool


@dataclass(frozen=True)
class ProviderHit:
    provider: str
    query_id: str
    title: str
    url: str
    snippet: str | None = None
    score: float | None = None
    raw_content: str | None = None


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    status: ProviderStatus
    hits: tuple[ProviderHit, ...] = ()
    latency_ms: float = 0.0
    error_code: ProviderErrorCode | None = None
    date_filter_normalized: bool = False
    parameter_retry_attempted: bool = False


@runtime_checkable
class SearchProvider(Protocol):
    name: str

    def readiness(self) -> ProviderReadiness: ...

    def search(
        self,
        query: SearchQuery,
        *,
        mode: SearchMode,
        max_results: int,
        timeout_seconds: float,
    ) -> ProviderResult: ...
