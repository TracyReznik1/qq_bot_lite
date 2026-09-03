"""Unit tests for independent Tavily-first and DDGS fallback query batches."""

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
    QueryPurpose,
    RequestSource,
    RetrievalRequest,
    SearchFailureCode,
    SearchQuery,
    SearchRoundKind,
    SearchTier,
    SearchTrace,
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

    def test_tavily_resolves_all_queries_ddgs_not_invoked(self):
        tavily_calls = []
        ddgs_calls = []

        class MockTavily:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, query, **kwargs):
                tavily_calls.append(query.query_id)
                hit = _hit(
                    url=f"https://tavily.example.com/{query.query_id}",
                    query_id=query.query_id,
                    provider="tavily",
                )
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (hit,), 1)

        class MockDDGS:
            name = "ddgs"

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, query, **kwargs):
                ddgs_calls.append(query.query_id)
                return ProviderResult("ddgs", ProviderStatus.SUCCESS, (), 1)

        orchestrator = self._make_orchestrator((MockDDGS(), MockTavily()))
        result = orchestrator.run(_request())

        self.assertTrue(tavily_calls)
        self.assertEqual([], ddgs_calls)
        self.assertIs(result.evidence.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual("tavily", result.trace.provider_attempts[0].provider)

    def test_tavily_fails_one_query_only_unresolved_query_falls_back_to_ddgs(self):
        tavily_calls = []
        ddgs_calls = []

        class MockTavily:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, query, **kwargs):
                tavily_calls.append(query.query_id)
                if query.query_id == "initial-1":
                    hit = _hit(
                        url=f"https://tavily.example.com/{query.query_id}",
                        query_id=query.query_id,
                        provider="tavily",
                    )
                    return ProviderResult("tavily", ProviderStatus.SUCCESS, (hit,), 1)
                return ProviderResult("tavily", ProviderStatus.EMPTY, (), 1)

        class MockDDGS:
            name = "ddgs"

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, query, **kwargs):
                ddgs_calls.append(query.query_id)
                hit = _hit(
                    url=f"https://example.com/{query.query_id}",
                    query_id=query.query_id,
                    provider="ddgs",
                )
                return ProviderResult("ddgs", ProviderStatus.SUCCESS, (hit,), 1)

        orchestrator = self._make_orchestrator((MockDDGS(), MockTavily()))
        result = orchestrator.run(_request())

        self.assertIn("initial-1", tavily_calls)
        self.assertNotIn("initial-1", ddgs_calls)
        self.assertEqual(set(tavily_calls) - {"initial-1"}, set(ddgs_calls))
        self.assertIs(result.evidence.evidence_state, EvidenceState.SUFFICIENT)

    def test_each_unresolved_tavily_status_falls_back_to_ddgs(self):
        statuses = (
            ProviderStatus.EMPTY,
            ProviderStatus.TIMEOUT,
            ProviderStatus.ERROR,
            ProviderStatus.UNAVAILABLE,
            ProviderStatus.NOT_CONFIGURED,
        )
        for tavily_status in statuses:
            with self.subTest(status=tavily_status):
                ddgs_calls = []

                class MockTavily:
                    name = "tavily"

                    def readiness(self):
                        configured = tavily_status is not ProviderStatus.NOT_CONFIGURED
                        available = tavily_status not in {
                            ProviderStatus.NOT_CONFIGURED,
                            ProviderStatus.UNAVAILABLE,
                        }
                        reason = (
                            None
                            if available
                            else SearchFailureCode.PROVIDER_UNAVAILABLE
                            if configured
                            else SearchFailureCode.PROVIDER_NOT_CONFIGURED
                        )
                        return ProviderReadiness(
                            "tavily", configured, available, reason
                        )

                    def search(self, query, **kwargs):
                        return ProviderResult("tavily", tavily_status, (), 1)

                class MockDDGS:
                    name = "ddgs"

                    def readiness(self):
                        return ProviderReadiness("ddgs", True, True, None)

                    def search(self, query, **kwargs):
                        ddgs_calls.append(query.query_id)
                        hit = _hit(
                            url=f"https://example.com/{query.query_id}",
                            query_id=query.query_id,
                            provider="ddgs",
                        )
                        return ProviderResult("ddgs", ProviderStatus.SUCCESS, (hit,), 1)

                self._make_orchestrator((MockTavily(), MockDDGS())).run(_request())
                self.assertTrue(ddgs_calls)

    def test_tavily_invalid_urls_fall_back_to_ddgs(self):
        ddgs_calls = []

        class MockTavily:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, query, **kwargs):
                invalid = _hit(
                    url="ftp://example.com/private",
                    query_id=query.query_id,
                    provider="tavily",
                )
                return ProviderResult("tavily", ProviderStatus.SUCCESS, (invalid,), 1)

        class MockDDGS:
            name = "ddgs"

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, query, **kwargs):
                ddgs_calls.append(query.query_id)
                hit = _hit(
                    url=f"https://example.com/{query.query_id}",
                    query_id=query.query_id,
                    provider="ddgs",
                )
                return ProviderResult("ddgs", ProviderStatus.SUCCESS, (hit,), 1)

        orchestrator = self._make_orchestrator((MockTavily(), MockDDGS()))
        orchestrator.run(_request())
        self.assertTrue(ddgs_calls)

    def test_repair_round_uses_tavily_before_ddgs(self):
        for tavily_status, expected_providers in (
            (ProviderStatus.SUCCESS, ["tavily"]),
            (ProviderStatus.EMPTY, ["tavily", "ddgs"]),
        ):
            with self.subTest(status=tavily_status):
                calls = []

                class MockTavily:
                    name = "tavily"

                    def readiness(self):
                        return ProviderReadiness("tavily", True, True, None)

                    def search(self, query, **kwargs):
                        calls.append(("tavily", query.query_id))
                        hits = (
                            (_hit(
                                url="https://tavily.example.com/repair",
                                query_id=query.query_id,
                                provider="tavily",
                            ),)
                            if tavily_status is ProviderStatus.SUCCESS
                            else ()
                        )
                        return ProviderResult("tavily", tavily_status, hits, 1)

                class MockDDGS:
                    name = "ddgs"

                    def readiness(self):
                        return ProviderReadiness("ddgs", True, True, None)

                    def search(self, query, **kwargs):
                        calls.append(("ddgs", query.query_id))
                        return ProviderResult("ddgs", ProviderStatus.EMPTY, (), 1)

                orchestrator = self._make_orchestrator((MockDDGS(), MockTavily()))
                query = SearchQuery(
                    query_id="repair-1",
                    query_index=1,
                    round_kind=SearchRoundKind.REPAIR,
                    purpose=QueryPurpose.REPAIR,
                    text="补充",
                    target_topic_ids=("topic-1",),
                )
                trace = SearchTrace(
                    "req-repair", RequestSource.CHAT, SearchTier.STANDARD
                )
                orchestrator._run_provider_round(
                    (query,),
                    SearchTier.STANDARD,
                    SearchRoundKind.REPAIR,
                    trace,
                )

                self.assertEqual(
                    expected_providers,
                    [provider for provider, _query_id in calls],
                )

    def test_sibling_query_failure_isolated(self):
        class MockTavily:
            name = "tavily"

            def readiness(self):
                return ProviderReadiness("tavily", True, True, None)

            def search(self, query, **kwargs):
                if query.query_id == "initial-1":
                    hit = _hit(
                        url="https://example.com/q1",
                        query_id=query.query_id,
                        provider="tavily",
                    )
                    return ProviderResult("tavily", ProviderStatus.SUCCESS, (hit,), 1)
                return ProviderResult("tavily", ProviderStatus.TIMEOUT, (), 1)

        class MockDDGS:
            name = "ddgs"

            def readiness(self):
                return ProviderReadiness("ddgs", True, True, None)

            def search(self, query, **kwargs):
                return ProviderResult("ddgs", ProviderStatus.TIMEOUT, (), 1)

        orchestrator = self._make_orchestrator((MockDDGS(), MockTavily()))
        result = orchestrator.run(_request())
        candidate_urls = [
            item.canonical_url for item in result.evidence.evidence_items
        ]
        self.assertIn("https://example.com/q1", candidate_urls)


if __name__ == "__main__":
    unittest.main()
