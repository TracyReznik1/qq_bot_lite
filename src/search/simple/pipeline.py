"""Fixed-mode search pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable
from uuid import uuid4

from src.search.simple.models import (
    SearchFailure,
    SearchMode,
    SearchOutcome,
    SearchPlan,
    SearchQuery,
    SearchRequest,
    SearchTrace,
)
from src.search.simple.planning import IMAGE_ONLY_FALLBACK_QUERY

logger = logging.getLogger("qq-bot")


def _new_request_id() -> str:
    return f"req-{uuid4().hex}"


@dataclass(frozen=True)
class SearchTimeouts:
    planner: float
    tavily: float
    ddgs: float
    reader: float
    ranker: float
    answer: float


class SimpleSearchPipeline:
    def __init__(
        self,
        planner: Any,
        retriever: Any,
        reader: Any,
        ranker: Any,
        *,
        timeouts: SearchTimeouts,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._planner = planner
        self._retriever = retriever
        self._reader = reader
        self._ranker = ranker
        self._timeouts = timeouts
        self._clock = clock

    def run(self, request: SearchRequest) -> SearchOutcome:
        trace = SearchTrace(
            request_id=_new_request_id(),
            source=request.source,
            mode=request.mode,
        )

        if request.mode is SearchMode.SKIP:
            return SearchOutcome(
                plan=SearchPlan(SearchMode.SKIP, ()),
                results=(),
                trace=trace,
            )

        try:
            planner_started = self._clock()
            if request.mode is SearchMode.LIGHT and request.topics:
                plan = SearchPlan(
                    mode=SearchMode.LIGHT,
                    queries=(SearchQuery("q1", request.topics[0]),),
                    planner_degraded=False,
                )
            else:
                plan = self._planner.plan(
                    mode=request.mode,
                    text=request.text,
                    images=request.images,
                    timeout_seconds=self._timeouts.planner,
                )
            trace.stage_latency_ms["planner"] = max(
                (self._clock() - planner_started) * 1000.0, 0.0
            )
            trace.query_count = len(plan.queries)
            trace.planner_degraded = plan.planner_degraded

            if plan.mode is not request.mode:
                raise ValueError(
                    f"Planner returned mode {plan.mode} differing from request mode {request.mode}"
                )

            retrieval_started = self._clock()
            candidates = self._retriever.run(plan, trace)
            trace.stage_latency_ms["retrieval"] = max(
                (self._clock() - retrieval_started) * 1000.0, 0.0
            )

            reader_started = self._clock()
            read_limit = 1 if request.mode is SearchMode.LIGHT else 2
            read_results = self._reader.enrich(
                candidates,
                limit=read_limit,
                timeout_seconds=self._timeouts.reader,
                trace=trace,
            )
            trace.stage_latency_ms["reader"] = max(
                (self._clock() - reader_started) * 1000.0, 0.0
            )

            valid_results = tuple(
                r
                for r in read_results
                if r.url.strip() and (r.title.strip() or r.excerpt.strip())
            )

            ranker_started = self._clock()
            ranking = self._ranker.rank(
                request.text,
                valid_results,
                timeout_seconds=self._timeouts.ranker,
            )
            trace.stage_latency_ms["ranker"] = max(
                (self._clock() - ranker_started) * 1000.0, 0.0
            )
            trace.ranker_degraded = ranking.degraded
            warning = None

            if not ranking.results:
                return SearchOutcome(
                    plan=plan,
                    results=(),
                    trace=trace,
                    warning=None,
                    failure=SearchFailure.NO_RESULTS,
                )

            return SearchOutcome(
                plan=plan,
                results=ranking.results,
                trace=trace,
                warning=None,
                failure=None,
            )
        except Exception as error:
            logger.debug(
                "search pipeline run failed error_type=%s",
                type(error).__name__,
            )
            fallback_text = request.text or IMAGE_ONLY_FALLBACK_QUERY
            fallback_plan = SearchPlan(
                request.mode,
                (SearchQuery("q1", fallback_text),),
                planner_degraded=True,
            )
            return SearchOutcome(
                plan=fallback_plan,
                results=(),
                trace=trace,
                failure=SearchFailure.PROVIDER_UNAVAILABLE,
            )
