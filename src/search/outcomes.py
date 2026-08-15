"""Pure aggregation functions for search query and stage outcomes."""

from __future__ import annotations

from typing import Sequence

from src.search.models import (
    QueryBatchResult,
    QueryOutcome,
    QueryOutcomeStatus,
    RetrievalBatchState,
)


def aggregate_query_outcomes(outcomes: Sequence[QueryOutcome]) -> QueryBatchResult:
    """Aggregate a sequence of QueryOutcome records into a deterministic QueryBatchResult."""
    ordered = tuple(sorted(outcomes, key=lambda item: item.query_index))
    resolved = sum(item.status is QueryOutcomeStatus.RESOLVED for item in ordered)
    state = (
        RetrievalBatchState.SUCCESS
        if resolved == len(ordered)
        else RetrievalBatchState.PARTIAL_SUCCESS
        if resolved
        else RetrievalBatchState.ALL_FAILED
    )
    return QueryBatchResult(ordered, state)
