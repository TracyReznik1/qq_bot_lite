"""Unit tests for pure outcome aggregation logic."""

import unittest
from src.search.models import (
    ProviderAttempt,
    ProviderHit,
    ProviderStatus,
    QueryBatchResult,
    QueryOutcome,
    QueryOutcomeStatus,
    QueryPurpose,
    RetrievalBatchState,
    SearchQuery,
    SearchRoundKind,
)
from src.search.outcomes import aggregate_query_outcomes


def _make_query(query_id: str, index: int) -> SearchQuery:
    return SearchQuery(
        query_id=query_id,
        round_kind=SearchRoundKind.INITIAL,
        purpose=QueryPurpose.DIRECT,
        text=f"test query {index}",
        query_index=index,
    )


def _make_hit(query_id: str, url: str) -> ProviderHit:
    return ProviderHit(
        provider="ddgs",
        query_id=query_id,
        title="Test Hit",
        url=url,
        snippet="snippet",
        raw_content=None,
        published_date=None,
        score=1.0,
        quality_flags=(),
    )


class OutcomeAggregationTests(unittest.TestCase):
    def test_empty_outcomes_returns_all_failed(self):
        batch = aggregate_query_outcomes(())
        self.assertEqual(RetrievalBatchState.ALL_FAILED, batch.state)
        self.assertEqual(0, len(batch.outcomes))
        self.assertEqual(0, batch.resolved_query_count)

    def test_all_resolved_returns_success(self):
        q1 = _make_query("q1", 1)
        q2 = _make_query("q2", 2)
        o1 = QueryOutcome(q1, QueryOutcomeStatus.RESOLVED, (_make_hit("q1", "https://a.com"),), ())
        o2 = QueryOutcome(q2, QueryOutcomeStatus.RESOLVED, (_make_hit("q2", "https://b.com"),), ())

        batch = aggregate_query_outcomes((o1, o2))
        self.assertEqual(RetrievalBatchState.SUCCESS, batch.state)
        self.assertEqual(2, batch.resolved_query_count)
        self.assertEqual(2, batch.total_query_count)

    def test_some_resolved_returns_partial_success(self):
        q1 = _make_query("q1", 1)
        q2 = _make_query("q2", 2)
        q3 = _make_query("q3", 3)
        o1 = QueryOutcome(q1, QueryOutcomeStatus.RESOLVED, (_make_hit("q1", "https://a.com"),), ())
        o2 = QueryOutcome(q2, QueryOutcomeStatus.TIMEOUT, (), ())
        o3 = QueryOutcome(q3, QueryOutcomeStatus.EMPTY, (), ())

        batch = aggregate_query_outcomes((o1, o2, o3))
        self.assertEqual(RetrievalBatchState.PARTIAL_SUCCESS, batch.state)
        self.assertEqual(1, batch.resolved_query_count)
        self.assertEqual(3, batch.total_query_count)

    def test_none_resolved_returns_all_failed(self):
        q1 = _make_query("q1", 1)
        q2 = _make_query("q2", 2)
        o1 = QueryOutcome(q1, QueryOutcomeStatus.TIMEOUT, (), ())
        o2 = QueryOutcome(q2, QueryOutcomeStatus.ERROR, (), ())

        batch = aggregate_query_outcomes((o1, o2))
        self.assertEqual(RetrievalBatchState.ALL_FAILED, batch.state)
        self.assertEqual(0, batch.resolved_query_count)
        self.assertEqual(2, batch.total_query_count)


if __name__ == "__main__":
    unittest.main()
