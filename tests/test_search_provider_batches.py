"""Unit tests for independent DDGS-first and Tavily fallback query batches."""

import unittest
from datetime import date
from types import SimpleNamespace

from src.search.models import (
    EvidenceCandidate,
    EvidenceState,
    ExcerptOrigin,
    ProviderHit,
    ProviderReadiness,
    ProviderResult,
    ProviderStatus,
    RequestSource,
    RetrievalRequest,
    SearchTier,
)
from src.search.orchestrator import SearchOrchestrator
from tests.search_fakes import FakeClock


def _request(question="Rust 和 Go 的并发模型有什么区别"):
    return RetrievalRequest(
        question,
        request_source=RequestSource.CHAT,
    )


def _hit(url="https://example.com/page", snippet="正文内容", raw_content=None, query_id="q1", provider="fake"):
    return ProviderHit(
        provider=provider,
        query_id=query_id,
        title="Title",
        url=url,
        snippet=snippet,
        score=1.0,
        published_at=None,
        raw_content=raw_content,
        quality_flags=(),
    )


class _FakeExtractor:
    def extract(self, hit, query, *, allow_network_read=True, timeout_seconds=None):
        del allow_network_read, timeout_seconds
        return EvidenceCandidate(
            hit=hit,
            document=None,
            excerpt=hit.snippet or "正文",
            excerpt_origin=ExcerptOrigin.PROVIDER_SNIPPET,
            extraction_status="search_result_snippet",
            safety_flags=(),
            content_reads_consumed=0,
        )


class _FakeJudge:
    def __init__(self, verdicts=None, supported_topic_ids=None):
        self.verdicts = verdicts or {}
        self.supported_topic_ids = (
            None
            if supported_topic_ids is None
            else tuple(supported_topic_ids)
        )

    def judge(self, question, candidates, *, required_topics=None):
        del question
        available_topic_ids = tuple(
            row["topic_id"]
            for row in (required_topics or ())
            if isinstance(row, dict) and isinstance(row.get("topic_id"), str)
        )
        supported_topic_ids = (
            available_topic_ids
            if self.supported_topic_ids is None
            else tuple(
                topic_id
                for topic_id in self.supported_topic_ids
                if topic_id in available_topic_ids
            )
        )
        result = {}
        for index, candidate in enumerate(candidates, 1):
            if f"C{index}" in self.verdicts:
                result[f"C{index}"] = self.verdicts[f"C{index}"]
            else:
                result[f"C{index}"] = {
                    "candidate_id": f"C{index}",
                    "source_relation": "independent",
                    "publisher_entity_match": False,
                    "ownership_basis": None,
                    "publisher": None,
                    "supported_topic_ids": list(supported_topic_ids),
                    "freshness_by_topic": {
                        topic_id: "satisfied"
                        for topic_id in supported_topic_ids
                    },
                    "conflict_key": None,
                    "conflict_value": None,
                    "conflict_relation": None,
                }
        return result


def _make_request_analyzer():
    from src.search.router import LLMRequestAnalyzer
    class LLM:
        def chat(self, *args, **kwargs):
            payload = (
                '{"actionability":"none","potential_harm":"none","factuality":"factual",'
                '"freshness_requirement":"not_required","benefit_dimensions":["accuracy"],'
                '"trigger_codes":["factual_default"],"complexity_codes":["multi_entity"]}'
            )
            return SimpleNamespace(content=payload)
    return LLMRequestAnalyzer(LLM())


def _make_router():
    from src.search.router import RetrievalBenefitRouter
    return RetrievalBenefitRouter()


def _make_planner():
    from src.search.planner import SearchPlanner
    from tests.search_fakes import StaticPlannerModel
    return SearchPlanner(StaticPlannerModel(), today_provider=lambda: date(2026, 7, 29))


class ProviderBatchOrchestrationTests(unittest.TestCase):
    def _make_orchestrator(self, providers):
        return SearchOrchestrator(
            request_analyzer=_make_request_analyzer(),
            router=_make_router(),
            planner=_make_planner(),
            judge=_FakeJudge(),
            providers=providers,
            extractor=_FakeExtractor(),
        )

    def test_ddgs_resolves_all_queries_tavily_not_invoked(self):
        ddgs_calls = []
        tavily_calls = []

        class MockDDGS:
            name = "ddgs"
            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)
            def search(self, query, **kwargs):
                ddgs_calls.append(query.query_id)
                hit = _hit(url=f"https://example.com/{query.query_id}", query_id=query.query_id, provider="ddgs")
                return ProviderResult("ddgs", ProviderStatus.SUCCESS, (hit,), 1)

        class MockTavily:
            name = "tavily"
            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)
            def search(self, query, **kwargs):
                tavily_calls.append(query.query_id)
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (), 1)

        orchestrator = self._make_orchestrator((MockDDGS(), MockTavily()))
        result = orchestrator.run(_request())
        self.assertTrue(len(ddgs_calls) > 0)
        self.assertEqual(len(tavily_calls), 0)
        self.assertIs(result.evidence.evidence_state, EvidenceState.SUFFICIENT)

    def test_ddgs_fails_one_query_only_unresolved_query_falls_back_to_tavily(self):
        ddgs_calls = []
        tavily_calls = []

        class MockDDGS:
            name = "ddgs"
            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)
            def search(self, query, **kwargs):
                ddgs_calls.append(query.query_id)
                if query.query_id == "initial-1":
                    hit = _hit(url=f"https://example.com/{query.query_id}", query_id=query.query_id, provider="ddgs")
                    return ProviderResult("ddgs", ProviderStatus.SUCCESS, (hit,), 1)
                return ProviderResult("ddgs", ProviderStatus.EMPTY, (), 1)

        class MockTavily:
            name = "tavily"
            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)
            def search(self, query, **kwargs):
                tavily_calls.append(query.query_id)
                hit = _hit(url=f"https://tavily.example.com/{query.query_id}", query_id=query.query_id, provider="tavily")
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (hit,), 1)

        orchestrator = self._make_orchestrator((MockDDGS(), MockTavily()))
        result = orchestrator.run(_request())
        self.assertIn("initial-1", ddgs_calls)
        self.assertNotIn("initial-1", tavily_calls)
        self.assertIs(result.evidence.evidence_state, EvidenceState.SUFFICIENT)

    def test_sibling_query_failure_isolated(self):
        class MockDDGS:
            name = "ddgs"
            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)
            def search(self, query, **kwargs):
                if query.query_id == "initial-1":
                    hit = _hit(url="https://example.com/q1", query_id=query.query_id, provider="ddgs")
                    return ProviderResult("ddgs", ProviderStatus.SUCCESS, (hit,), 1)
                return ProviderResult("ddgs", ProviderStatus.TIMEOUT, (), 1)

        class MockTavily:
            name = "tavily"
            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)
            def search(self, query, **kwargs):
                return ProviderResult("tavily", ProviderStatus.TIMEOUT, (), 1)

        orchestrator = self._make_orchestrator((MockDDGS(), MockTavily()))
        result = orchestrator.run(_request())
        # Sibling failure of query 2/3 does not discard query 1 hits
        candidate_urls = [item.canonical_url for item in result.evidence.evidence_items]
        self.assertIn("https://example.com/q1", candidate_urls)


if __name__ == "__main__":
    unittest.main()
