"""Pure aggregation and candidate selection functions for search query and stage outcomes."""

from __future__ import annotations

from collections import deque
from typing import Callable, Sequence

from src.search.models import (
    ProviderHit,
    QueryBatchResult,
    QueryOutcome,
    QueryOutcomeStatus,
    RetrievalBatchState,
)
from src.search.url_policy import canonicalize_public_http_url


def aggregate_query_outcomes(outcomes: Sequence[QueryOutcome]) -> QueryBatchResult:
    """Aggregate a sequence of QueryOutcome records into a deterministic QueryBatchResult."""
    ordered = tuple(sorted(outcomes, key=lambda item: item.query_index))
    resolved = sum(item.status is QueryOutcomeStatus.RESOLVED for item in ordered)
    state = (
        RetrievalBatchState.SUCCESS
        if resolved == len(ordered) and len(ordered) > 0
        else RetrievalBatchState.PARTIAL_SUCCESS
        if resolved
        else RetrievalBatchState.ALL_FAILED
    )
    return QueryBatchResult(ordered, state)


def select_candidate_hits(
    batch: QueryBatchResult,
    *,
    max_urls: int,
    validator: Callable[[str], bool] | None = None,
) -> tuple[ProviderHit, ...]:
    """Select up to max_urls candidate hits by alternating across resolved queries in round-robin order,
    deduplicating by canonical URL and skipping invalid addresses."""
    if max_urls <= 0:
        return ()
    per_query = [deque(outcome.hits) for outcome in batch.outcomes if outcome.resolved]
    selected: list[ProviderHit] = []
    seen_canonical_urls: set[str] = set()

    while per_query and len(selected) < max_urls:
        next_round: list[deque[ProviderHit]] = []
        for queue in per_query:
            while queue and len(selected) < max_urls:
                hit = queue.popleft()
                if not hit.url:
                    continue
                canonical = canonicalize_public_http_url(hit.url)
                if not canonical or canonical in seen_canonical_urls:
                    continue
                if validator is not None and not validator(canonical):
                    continue
                seen_canonical_urls.add(canonical)
                selected.append(hit)
                break
            if queue:
                next_round.append(queue)
        per_query = next_round
    return tuple(selected)


def round_robin_hits(
    batch: QueryBatchResult,
    *,
    max_urls: int,
    validator: Callable[[str], bool] | None = None,
) -> tuple[ProviderHit, ...]:
    """Alias for select_candidate_hits."""
    return select_candidate_hits(batch, max_urls=max_urls, validator=validator)
