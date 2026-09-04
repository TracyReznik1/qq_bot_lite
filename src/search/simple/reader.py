"""Bounded on-demand page reader for the simple search runtime."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
import re
from typing import Any, Callable, Sequence

from src.search.simple.models import SearchResult, SearchTrace

_MIN_SNIPPET_CHARS = 80
_MAX_EXCERPT_CHARS = 1500
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class OnDemandReader:
    def __init__(self, fetch_document: Callable[..., Any] | None = None) -> None:
        if fetch_document is None:
            from src.services.url_fetch_service import fetch_document as _fetch
            self._fetch_document = _fetch
        else:
            self._fetch_document = fetch_document

    def enrich(
        self,
        results: tuple[SearchResult, ...],
        *,
        limit: int,
        timeout_seconds: float,
        trace: SearchTrace,
    ) -> tuple[SearchResult, ...]:
        if not results or limit <= 0:
            return results

        selected: list[SearchResult] = []
        for res in results:
            compacted = " ".join((res.excerpt or "").split())
            if len(compacted) < _MIN_SNIPPET_CHARS:
                selected.append(res)
                if len(selected) >= limit:
                    break

        if not selected:
            return results

        trace.reader_count += len(selected)
        futures = {}
        executor = ThreadPoolExecutor(
            max_workers=len(selected),
            thread_name_prefix="simple-search-reader",
        )
        try:
            for item in selected:
                future = executor.submit(
                    self._fetch_document,
                    item.url,
                    timeout=timeout_seconds,
                )
                futures[item.result_id] = future

            wait(tuple(futures.values()), timeout=timeout_seconds)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        enriched: list[SearchResult] = []
        for res in results:
            future = futures.get(res.result_id)
            if future is None or not future.done():
                enriched.append(res)
                continue

            try:
                doc = future.result()
            except Exception:
                enriched.append(res)
                continue

            if not getattr(doc, "ok", False):
                enriched.append(res)
                continue

            raw_text = getattr(doc, "text", "") or ""
            cleaned = _clean_page_text(raw_text)
            if not cleaned:
                enriched.append(res)
                continue

            enriched.append(replace(res, excerpt=cleaned[:_MAX_EXCERPT_CHARS]))

        return tuple(enriched)


def _clean_page_text(text: str) -> str:
    without_control = _CONTROL_CHARS_RE.sub(" ", text)
    return " ".join(without_control.split())
