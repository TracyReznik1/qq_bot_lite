"""Unit tests for pure outcome aggregation and candidate selection logic."""

import unittest
from src.search.models import (
    EvidenceCandidate,
    EvidenceState,
    ExcerptOrigin,
    JudgeBatchStatus,
    ProviderHit,
    QueryBatchResult,
    QueryOutcome,
    QueryOutcomeStatus,
    QueryPurpose,
    ReadOutcome,
    ReadOutcomeStatus,
    RetrievalBatchState,
    SearchFailureCode,
    SearchQuery,
    SearchRoundKind,
)
from src.search.outcomes import aggregate_query_outcomes, final_search_failure, select_candidate_hits


def _make_query(query_id: str, index: int) -> SearchQuery:
    return SearchQuery(
        query_id=query_id,
        round_kind=SearchRoundKind.INITIAL,
        purpose=QueryPurpose.DIRECT,
        text=f"test query {index}",
        query_index=index,
    )


def _make_hit(query_id: str, url: str, title: str = "Test Hit") -> ProviderHit:
    return ProviderHit(
        provider="ddgs",
        query_id=query_id,
        title=title,
        url=url,
        snippet="snippet",
        score=1.0,
        published_at=None,
        raw_content=None,
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


class CandidateSelectionTests(unittest.TestCase):
    def test_round_robin_selection_and_safety_dedup(self):
        q1 = _make_query("q1", 1)
        q2 = _make_query("q2", 2)

        # q1 has:
        # 1. q1-a (valid https://example.com/page1)
        # 2. duplicate q1-dup (https://example.com/page1#frag)
        # 3. private/unsafe q1-private (http://127.0.0.1/admin)
        # 4. q1-b (valid https://example.com/page2:443/)
        # q2 has:
        # 1. q2-a (valid https://example.org/page1)
        hits_q1 = (
            _make_hit("q1", "https://example.com/page1", title="q1-a"),
            _make_hit("q1", "https://example.com/page1#frag", title="q1-dup"),
            _make_hit("q1", "http://127.0.0.1/admin", title="q1-private"),
            _make_hit("q1", "https://example.com/page2:443/", title="q1-b"),
        )
        hits_q2 = (
            _make_hit("q2", "https://example.org/page1", title="q2-a"),
        )

        o1 = QueryOutcome(q1, QueryOutcomeStatus.RESOLVED, hits_q1, ())
        o2 = QueryOutcome(q2, QueryOutcomeStatus.RESOLVED, hits_q2, ())
        batch = aggregate_query_outcomes((o1, o2))

        def fake_public(url: str) -> bool:
            return "127.0.0.1" not in url

        selected = select_candidate_hits(batch, max_urls=3, validator=fake_public)
        self.assertEqual(
            ("q1-a", "q2-a", "q1-b"),
            tuple(hit.title for hit in selected),
        )


if __name__ == "__main__":
    unittest.main()


class FinalSearchFailureTests(unittest.TestCase):
    def test_frozen_failure_matrix(self):
        q = _make_query("q1", 1)
        cases = (
            (
                "all_empty",
                aggregate_query_outcomes((QueryOutcome(q, QueryOutcomeStatus.EMPTY, (), ()),)),
                None,
                None,
                None,
                SearchFailureCode.NO_RESULTS,
            ),
            (
                "all_not_configured",
                aggregate_query_outcomes((QueryOutcome(q, QueryOutcomeStatus.ERROR, (), (), readiness_failure=SearchFailureCode.PROVIDER_NOT_CONFIGURED),)),
                None,
                None,
                None,
                SearchFailureCode.PROVIDER_NOT_CONFIGURED,
            ),
            (
                "all_timeout",
                aggregate_query_outcomes((QueryOutcome(q, QueryOutcomeStatus.TIMEOUT, (), ()),)),
                None,
                None,
                None,
                SearchFailureCode.PROVIDER_TIMEOUT,
            ),
            (
                "all_error_or_unavailable",
                aggregate_query_outcomes((QueryOutcome(q, QueryOutcomeStatus.ERROR, (), ()),)),
                None,
                None,
                None,
                SearchFailureCode.PROVIDER_UNAVAILABLE,
            ),
            (
                "partial_success_all_unreadable",
                aggregate_query_outcomes((
                    QueryOutcome(q, QueryOutcomeStatus.RESOLVED, (_make_hit("q1", "https://a.com"),), ()),
                    QueryOutcome(_make_query("q2", 2), QueryOutcomeStatus.EMPTY, (), ()),
                )),
                (ReadOutcome(_make_hit("q1", "https://a.com"), ReadOutcomeStatus.UNREADABLE, None, True),),
                None,
                None,
                SearchFailureCode.CONTENT_UNREADABLE,
            ),
            (
                "success_readable_judge_unavailable",
                aggregate_query_outcomes((QueryOutcome(q, QueryOutcomeStatus.RESOLVED, (_make_hit("q1", "https://a.com"),), ()),)),
                (
                    ReadOutcome(
                        _make_hit("q1", "https://a.com"),
                        ReadOutcomeStatus.READABLE,
                        EvidenceCandidate(
                            _make_hit("q1", "https://a.com"),
                            None,
                            "text",
                            ExcerptOrigin.PAGE_EXTRACT,
                            "extraction_ok",
                            (),
                            1,
                        ),
                        True,
                    ),
                ),
                JudgeBatchStatus.UNAVAILABLE,
                None,
                SearchFailureCode.JUDGE_UNAVAILABLE,
            ),
            (
                "success_readable_insufficient_evidence",
                aggregate_query_outcomes((QueryOutcome(q, QueryOutcomeStatus.RESOLVED, (_make_hit("q1", "https://a.com"),), ()),)),
                (
                    ReadOutcome(
                        _make_hit("q1", "https://a.com"),
                        ReadOutcomeStatus.READABLE,
                        EvidenceCandidate(
                            _make_hit("q1", "https://a.com"),
                            None,
                            "text",
                            ExcerptOrigin.PAGE_EXTRACT,
                            "extraction_ok",
                            (),
                            1,
                        ),
                        True,
                    ),
                ),
                JudgeBatchStatus.COMPLETED,
                EvidenceState.INSUFFICIENT,
                SearchFailureCode.INSUFFICIENT_EVIDENCE,
            ),
            (
                "success_readable_partial_evidence",
                aggregate_query_outcomes((QueryOutcome(q, QueryOutcomeStatus.RESOLVED, (_make_hit("q1", "https://a.com"),), ()),)),
                (
                    ReadOutcome(
                        _make_hit("q1", "https://a.com"),
                        ReadOutcomeStatus.READABLE,
                        EvidenceCandidate(
                            _make_hit("q1", "https://a.com"),
                            None,
                            "text",
                            ExcerptOrigin.PAGE_EXTRACT,
                            "extraction_ok",
                            (),
                            1,
                        ),
                        True,
                    ),
                ),
                JudgeBatchStatus.COMPLETED,
                EvidenceState.PARTIAL,
                SearchFailureCode.PARTIAL_EVIDENCE,
            ),
            (
                "success_readable_conflicting_evidence",
                aggregate_query_outcomes((QueryOutcome(q, QueryOutcomeStatus.RESOLVED, (_make_hit("q1", "https://a.com"),), ()),)),
                (
                    ReadOutcome(
                        _make_hit("q1", "https://a.com"),
                        ReadOutcomeStatus.READABLE,
                        EvidenceCandidate(
                            _make_hit("q1", "https://a.com"),
                            None,
                            "text",
                            ExcerptOrigin.PAGE_EXTRACT,
                            "extraction_ok",
                            (),
                            1,
                        ),
                        True,
                    ),
                ),
                JudgeBatchStatus.COMPLETED,
                EvidenceState.CONFLICTING,
                SearchFailureCode.SOURCE_CONFLICT,
            ),
            (
                "success_readable_sufficient_evidence",
                aggregate_query_outcomes((QueryOutcome(q, QueryOutcomeStatus.RESOLVED, (_make_hit("q1", "https://a.com"),), ()),)),
                (
                    ReadOutcome(
                        _make_hit("q1", "https://a.com"),
                        ReadOutcomeStatus.READABLE,
                        EvidenceCandidate(
                            _make_hit("q1", "https://a.com"),
                            None,
                            "text",
                            ExcerptOrigin.PAGE_EXTRACT,
                            "extraction_ok",
                            (),
                            1,
                        ),
                        True,
                    ),
                ),
                JudgeBatchStatus.COMPLETED,
                EvidenceState.SUFFICIENT,
                None,
            ),
        )
        for name, batch, reads, judge_status, evidence_state, expected in cases:
            with self.subTest(name=name):
                result = final_search_failure(
                    batch,
                    read_outcomes=reads,
                    judge_status=judge_status,
                    evidence_state=evidence_state,
                )
                self.assertIs(expected, result)
