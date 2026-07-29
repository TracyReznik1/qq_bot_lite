"""Offline fakes shared by evidence-search unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.search.models import ProviderReadiness, ProviderStatus


@dataclass
class StaticRouterAdvisor:
    decision: Any

    def advise(self, _request: Any) -> Any:
        return self.decision


@dataclass
class RecordingProvider:
    provider: str = "recording"
    hits: tuple[Any, ...] = ()
    status: ProviderStatus = ProviderStatus.SUCCESS
    readiness: ProviderReadiness | None = None
    calls: list[Any] = field(default_factory=list)

    def search(self, query: Any, **kwargs: Any) -> tuple[Any, ...]:
        self.calls.append((query, kwargs))
        return self.hits

    def get_readiness(self) -> ProviderReadiness:
        if self.readiness is not None:
            return self.readiness
        return ProviderReadiness(self.provider, configured=True, available=True, reason_code=None)


@dataclass
class StaticEvidenceJudge:
    result: Any

    def judge(self, *_args: Any, **_kwargs: Any) -> Any:
        return self.result


@dataclass
class StaticSemanticVerifier:
    result: Any

    def verify(self, *_args: Any, **_kwargs: Any) -> Any:
        return self.result


@dataclass
class FakeClock:
    now: float = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
