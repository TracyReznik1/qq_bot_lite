"""Production factory and lifecycle hooks for the simple search runtime."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

from src.config import config
from src.search.simple.ddgs import DDGSSearchProvider
from src.search.simple.pipeline import SearchTimeouts, SimpleSearchPipeline
from src.search.simple.planning import QueryPlanner
from src.search.simple.providers import ProviderReadiness
from src.search.simple.ranking import EvidenceRanker
from src.search.simple.reader import OnDemandReader
from src.search.simple.retrieval import ProviderRunner
from src.search.simple.tavily import TavilySearchProvider
from src.services.llm_client import get_llm_client

_pipeline_lock = Lock()
_pipeline: SimpleSearchPipeline | None = None
_providers: tuple[TavilySearchProvider, DDGSSearchProvider] | None = None


def _get_providers() -> tuple[TavilySearchProvider, DDGSSearchProvider]:
    global _providers
    if _providers is None:
        tavily = TavilySearchProvider(
            api_key=config.tavily_api_key,
            proxy_url=config.proxy_url,
        )
        ddgs = DDGSSearchProvider(
            proxy_url=config.proxy_url,
            timeout_seconds=config.search_ddgs_timeout,
        )
        _providers = (tavily, ddgs)
    return _providers


def get_search_readiness() -> tuple[ProviderReadiness, ...]:
    providers = _get_providers()
    return tuple(p.readiness() for p in providers)


def get_simple_search_pipeline() -> SimpleSearchPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        llm = get_llm_client()
        providers = _get_providers()
        timeouts = SearchTimeouts(
            planner=config.search_planner_timeout,
            tavily=config.search_tavily_timeout,
            ddgs=config.search_ddgs_timeout,
            reader=config.search_reader_timeout,
            ranker=config.search_ranker_timeout,
            answer=config.search_answer_timeout,
        )
        planner = QueryPlanner(llm)
        retriever = ProviderRunner(
            providers,
            tavily_timeout=config.search_tavily_timeout,
            ddgs_timeout=config.search_ddgs_timeout,
            max_results_per_query=config.search_max_results,
        )
        reader = OnDemandReader()
        ranker = EvidenceRanker(llm)
        _pipeline = SimpleSearchPipeline(
            planner,
            retriever,
            reader,
            ranker,
            timeouts=timeouts,
        )
        return _pipeline


def reset_simple_search_pipeline() -> None:
    global _pipeline, _providers
    with _pipeline_lock:
        _pipeline = None
        _providers = None
