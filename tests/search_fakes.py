"""Offline fakes shared by evidence-search unit tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.search.models import (
    Factuality,
    FreshnessContext,
    FreshnessRequirement,
    ProviderReadiness,
    ProviderStatus,
    RequestAnalysis,
    RetrievalContext,
    RiskContext,
    SourceRequirement,
)
from src.services.llm_types import ChatResponse


def make_analysis(
    *,
    skip_reason=None,
    high_consequence=False,
    fail_closed=False,
    freshness=FreshnessRequirement.NOT_REQUIRED,
):
    """Build a minimal RequestAnalysis for tests that mock the orchestrator."""
    return RequestAnalysis(
        RetrievalContext(
            must_search=skip_reason is None,
            skip_reason=skip_reason,
            factuality=(
                Factuality.NON_FACTUAL
                if skip_reason is not None
                else Factuality.FACTUAL
            ),
            external_fact_required=skip_reason is None,
            complexity_codes=(),
            source_requirement=SourceRequirement.ANY_RELEVANT,
        ),
        FreshnessContext(freshness, None, None, None, None),
        RiskContext(high_consequence, high_consequence, fail_closed),
    )


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


@dataclass
class StaticPlannerModel:
    """LLM stand-in that returns a fixed planning JSON payload."""

    payload: Any = None
    calls: list[Any] = field(default_factory=list)
    raise_error: Exception | None = None

    def chat(self, messages: list[Any], **kwargs: Any) -> ChatResponse:
        self.calls.append((messages, kwargs))
        if self.raise_error is not None:
            raise self.raise_error
        if self.payload is None:
            return ChatResponse(content="{}", tool_calls=[])
        if isinstance(self.payload, dict):
            return ChatResponse(
                content=json.dumps(self.payload, ensure_ascii=False),
                tool_calls=[],
            )
        return ChatResponse(content=self.payload, tool_calls=[])
