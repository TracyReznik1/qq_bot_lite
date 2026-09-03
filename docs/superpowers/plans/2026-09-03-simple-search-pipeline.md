# Simplified Search Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current evidence state machine with a resilient `skip / light / standard` QQBot search pipeline that uses one planning call, one relevance-ranking call, natural-language answers, and deterministic fallbacks.

**Architecture:** Build the replacement under `src/search/simple/`, route chat and `/search` through it, then remove the old risk/freshness/topic/repair/claim pipeline. Reuse Tavily, DDGS, URL safety, and URL fetching during migration; every external call receives a real HTTP timeout and every failure becomes a bounded local degradation.

**Tech Stack:** Python 3.11+, dataclasses, `StrEnum`, `concurrent.futures`, existing Gemini/DeepSeek client, Tavily, DDGS, `unittest`/`unittest.mock`.

---

## File map

### New production files

- `src/search/simple/models.py` — minimal request, plan, result, trace, failure, and response contracts.
- `src/search/simple/planning.py` — combined LLM route/query planning plus deterministic fallback.
- `src/search/simple/retrieval.py` — bounded Tavily-first/DDGS-fallback query execution and URL deduplication.
- `src/search/simple/reader.py` — fetch only empty/short snippets and produce bounded excerpts.
- `src/search/simple/ranking.py` — tolerant parsing and stable relevance ordering.
- `src/search/simple/answering.py` — bounded natural-language model call and deterministic summary fallback.
- `src/search/simple/rendering.py` — QQ-length output and command-only source display.
- `src/search/simple/pipeline.py` — retrieval orchestration and top-level exception boundary.
- `src/search/simple/factory.py` — production dependency construction and reset hook.
- `src/search/simple/__init__.py` — narrow public exports.

### Modified production files

- `src/config.py` — centralized timeout settings.
- `src/chat/prompt.py` — natural-language search prompt without GroundedDraft instructions.
- `src/chat/chat_service.py` — use the new pipeline and responder; retain history and multimodal behavior.
- `src/services/search_service.py` — compatibility facade over the new pipeline.
- `src/search/providers/base.py` — final minimal provider protocol.
- `src/search/providers/tavily.py` — consume the final simple query/mode contracts.
- `src/search/providers/ddgs.py` — consume the final simple query/mode contracts.
- `src/search/__init__.py` — export only the new public API and URL policy.
- `tools/evaluate_search.py` — replace old state-machine checks with simple behavior metrics.
- `README.md` — document the two search modes, fallbacks, sources, and timeouts.

### New focused tests

- `tests/test_simple_search_models.py`
- `tests/test_simple_search_planning.py`
- `tests/test_simple_search_retrieval.py`
- `tests/test_simple_search_reader.py`
- `tests/test_simple_search_ranking.py`
- `tests/test_simple_search_pipeline.py`
- `tests/test_simple_search_answering.py`
- `tests/test_simple_search_chat_flow.py`

### Removed legacy files after cutover

- `src/search/budget.py`
- `src/search/evidence.py`
- `src/search/extraction.py`
- `src/search/models.py`
- `src/search/orchestrator.py`
- `src/search/outcomes.py`
- `src/search/planner.py`
- `src/search/policy.py`
- `src/search/renderer.py`
- `src/search/router.py`
- `src/search/stage_runner.py`
- `src/search/validation.py`
- `tests/search_fakes.py`
- `tests/test_chat_retrieval_flow.py`
- `tests/test_search_blind_acceptance_runner.py`
- `tests/test_search_budget.py`
- `tests/test_search_evaluation.py`
- `tests/test_search_evidence.py`
- `tests/test_search_extraction.py`
- `tests/test_search_models.py`
- `tests/test_search_orchestrator.py`
- `tests/test_search_outcomes.py`
- `tests/test_search_planner.py`
- `tests/test_search_policy.py`
- `tests/test_search_provider_batches.py`
- `tests/test_search_renderer.py`
- `tests/test_search_router.py`
- `tests/test_search_simplification_baseline.py`
- `tests/test_search_stage_runner.py`
- `tests/test_search_validation.py`

`tests/test_search_providers.py` and `tests/test_search_url_policy.py` remain, but provider tests move to the final simple contracts during cleanup.

---

### Task 1: Add minimal search contracts and centralized limits

**Files:**
- Create: `src/search/simple/__init__.py`
- Create: `src/search/simple/models.py`
- Modify: `src/config.py`
- Test: `tests/test_simple_search_models.py`

- [ ] **Step 1: Write failing contract tests**

```python
# tests/test_simple_search_models.py
from pathlib import Path
import unittest

from src.search.simple.models import (
    SearchMode, SearchPlan, SearchQuery, SearchRequest, SearchResult,
)


class SimpleSearchModelTests(unittest.TestCase):
    def test_mode_has_only_skip_light_and_standard(self):
        self.assertEqual(["skip", "light", "standard"], [item.value for item in SearchMode])

    def test_light_plan_rejects_multiple_queries(self):
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"), SearchQuery("q2", "b")))

    def test_standard_plan_rejects_more_than_three_queries(self):
        queries = tuple(SearchQuery(f"q{i}", str(i)) for i in range(1, 5))
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.STANDARD, queries)

    def test_search_result_clamps_score(self):
        result = SearchResult("R1", "title", "https://example.com", "body", "tavily", 1.8)
        self.assertEqual(1.0, result.score)

    def test_force_search_request_is_nonempty(self):
        with self.assertRaises(ValueError):
            SearchRequest("", force_search=True)
```

- [ ] **Step 2: Run the tests and confirm the module is absent**

Run: `python -B -m unittest tests.test_simple_search_models -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.search.simple'`.

- [ ] **Step 3: Implement the minimal contracts**

Create `src/search/simple/models.py` with these exact public shapes:

```python
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
import math
from typing import Mapping


class SearchMode(StrEnum):
    SKIP = "skip"
    LIGHT = "light"
    STANDARD = "standard"


class RequestSource(StrEnum):
    CHAT = "chat"
    COMMAND = "command"
    COMPATIBILITY = "compatibility"


class SearchFailure(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    NO_RESULTS = "no_results"


class OutputKind(StrEnum):
    PLAIN = "plain"
    MODEL_ANSWER = "model_answer"
    SUMMARY_FALLBACK = "summary_fallback"
    SEARCH_FAILURE = "search_failure"


@dataclass(frozen=True)
class SearchRequest:
    question: str
    force_search: bool = False
    has_images: bool = False
    request_source: RequestSource = RequestSource.CHAT

    def __post_init__(self):
        question = str(self.question or "").strip()
        if not question:
            raise ValueError("question must be non-empty")
        object.__setattr__(self, "question", question)


@dataclass(frozen=True)
class SearchQuery:
    query_id: str
    text: str
    date_from: date | None = None
    date_to: date | None = None
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    news: bool = False

    def __post_init__(self):
        query_id = str(self.query_id or "").strip()
        text = " ".join(str(self.text or "").split())
        if not query_id or not text:
            raise ValueError("query id and text must be non-empty")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from cannot exceed date_to")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "text", text[:500])
        object.__setattr__(self, "include_domains", tuple(self.include_domains))
        object.__setattr__(self, "exclude_domains", tuple(self.exclude_domains))


@dataclass(frozen=True)
class SearchPlan:
    mode: SearchMode
    queries: tuple[SearchQuery, ...]
    planner_degraded: bool = False

    def __post_init__(self):
        queries = tuple(self.queries)
        if self.mode is SearchMode.SKIP and queries:
            raise ValueError("skip cannot carry queries")
        if self.mode is SearchMode.LIGHT and len(queries) != 1:
            raise ValueError("light requires exactly one query")
        if self.mode is SearchMode.STANDARD and not 1 <= len(queries) <= 3:
            raise ValueError("standard requires one to three queries")
        object.__setattr__(self, "queries", queries)


@dataclass(frozen=True)
class SearchResult:
    result_id: str
    title: str
    url: str
    excerpt: str
    provider: str
    score: float = 0.5

    def __post_init__(self):
        score = float(self.score)
        score = min(max(score, 0.0), 1.0) if math.isfinite(score) else 0.5
        object.__setattr__(self, "score", score)


@dataclass
class SearchTrace:
    request_id: str
    mode: SearchMode = SearchMode.SKIP
    query_count: int = 0
    provider_statuses: dict[str, str] = field(default_factory=dict)
    candidate_count: int = 0
    reader_count: int = 0
    planner_degraded: bool = False
    judge_degraded: bool = False
    answer_degraded: bool = False
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    output_kind: OutputKind = OutputKind.PLAIN

    def to_safe_dict(self) -> Mapping[str, object]:
        return {
            "request_id": self.request_id,
            "mode": self.mode.value,
            "query_count": self.query_count,
            "provider_statuses": dict(self.provider_statuses),
            "candidate_count": self.candidate_count,
            "reader_count": self.reader_count,
            "planner_degraded": self.planner_degraded,
            "judge_degraded": self.judge_degraded,
            "answer_degraded": self.answer_degraded,
            "stage_latency_ms": dict(self.stage_latency_ms),
            "output_kind": self.output_kind.value,
        }


@dataclass(frozen=True)
class SearchOutcome:
    plan: SearchPlan
    results: tuple[SearchResult, ...]
    trace: SearchTrace
    warning: str | None = None
    failure: SearchFailure | None = None


@dataclass(frozen=True)
class SearchResponse:
    text: str
    sources: tuple[SearchResult, ...]
    trace: SearchTrace
```

Create `src/search/simple/__init__.py` exporting only these types initially. Add six `Config` fields in `src/config.py` using the existing `env_float` helper:

```python
search_planner_timeout: float = env_float("SEARCH_PLANNER_TIMEOUT", 8.0)
search_tavily_timeout: float = env_float("SEARCH_TAVILY_TIMEOUT", 8.0)
search_ddgs_timeout: float = env_float("SEARCH_DDGS_TIMEOUT", 15.0)
search_reader_timeout: float = env_float("SEARCH_READER_TIMEOUT", 5.0)
search_ranker_timeout: float = env_float("SEARCH_RANKER_TIMEOUT", 10.0)
search_answer_timeout: float = env_float("SEARCH_ANSWER_TIMEOUT", 20.0)
```

Normalize each value to at least `0.1` in `Config.__post_init__`.

- [ ] **Step 4: Run focused tests**

Run: `python -B -m unittest tests.test_simple_search_models -v`

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/search/simple/__init__.py src/search/simple/models.py src/config.py tests/test_simple_search_models.py
git commit -m "feat: add simple search contracts"
```

---

### Task 2: Implement one-call route and query planning

**Files:**
- Create: `src/search/simple/planning.py`
- Modify: `src/search/simple/__init__.py`
- Test: `tests/test_simple_search_planning.py`

- [ ] **Step 1: Write failing planner tests**

```python
# tests/test_simple_search_planning.py
import unittest
from types import SimpleNamespace

from src.search.simple.models import SearchMode, SearchRequest
from src.search.simple.planning import RoutePlanner


class FakeLLM:
    def __init__(self, content="", error=None):
        self.content, self.error, self.calls = content, error, []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


class RoutePlannerTests(unittest.TestCase):
    def test_force_search_overrides_model_skip_and_caps_three_queries(self):
        llm = FakeLLM('prefix ```json\n{"mode":"skip","queries":["a","b","c","d"]}\n``` suffix')
        plan = RoutePlanner(llm).plan(SearchRequest("question", force_search=True), timeout_seconds=8)
        self.assertEqual(SearchMode.STANDARD, plan.mode)
        self.assertEqual(("a", "b", "c"), tuple(q.text for q in plan.queries))

    def test_light_deduplicates_and_caps_one_query(self):
        llm = FakeLLM('{"mode":"light","queries":["  EDG  ","EDG"]}')
        plan = RoutePlanner(llm).plan(SearchRequest("EDG能去吗"), timeout_seconds=8)
        self.assertEqual(("EDG",), tuple(q.text for q in plan.queries))

    def test_invalid_output_falls_back_to_original_light_query(self):
        plan = RoutePlanner(FakeLLM("not json")).plan(SearchRequest("当前版本是什么"), timeout_seconds=8)
        self.assertEqual(SearchMode.LIGHT, plan.mode)
        self.assertTrue(plan.planner_degraded)
        self.assertEqual("当前版本是什么", plan.queries[0].text)

    def test_invalid_output_skips_obvious_social_chat(self):
        plan = RoutePlanner(FakeLLM(error=TimeoutError())).plan(SearchRequest("你好呀"), timeout_seconds=8)
        self.assertEqual(SearchMode.SKIP, plan.mode)

    def test_timeout_is_forwarded_to_llm(self):
        llm = FakeLLM('{"mode":"light","queries":["q"]}')
        RoutePlanner(llm).plan(SearchRequest("q"), timeout_seconds=7.5)
        self.assertEqual(7.5, llm.calls[0][1]["timeout_seconds"])
```

- [ ] **Step 2: Run and verify RED**

Run: `python -B -m unittest tests.test_simple_search_planning -v`

Expected: FAIL because `src.search.simple.planning` does not exist.

- [ ] **Step 3: Implement tolerant planning**

Implement in `src/search/simple/planning.py`:

```python
class RoutePlanner:
    def __init__(self, llm):
        self._llm = llm

    def plan(self, request: SearchRequest, *, timeout_seconds: float) -> SearchPlan:
        try:
            response = self._llm.chat(
                _planner_messages(request.question, request.has_images),
                temperature=0.0,
                max_tokens=256,
                tools=None,
                tool_choice="none",
                timeout_seconds=timeout_seconds,
            )
            parsed = _parse_plan(getattr(response, "content", ""), request)
            if parsed is not None:
                return parsed
        except Exception:
            pass
        return _fallback_plan(request)
```

Use a system prompt containing only `mode` and `queries`. `_parse_plan` must locate the first balanced JSON object, ignore unknown fields, normalize and deduplicate query strings, force command requests to standard, cap light at one and standard at three, and use the original question if a search mode has no valid query. `_fallback_plan` must use a small deterministic regex set for greetings, creative requests, text transforms, and pure arithmetic; force-search always returns standard, and every other request returns light.

- [ ] **Step 4: Run focused tests**

Run: `python -B -m unittest tests.test_simple_search_planning -v`

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/search/simple/planning.py src/search/simple/__init__.py tests/test_simple_search_planning.py
git commit -m "feat: add resilient search route planner"
```

---

### Task 3: Add bounded provider retrieval

**Files:**
- Create: `src/search/simple/retrieval.py`
- Test: `tests/test_simple_search_retrieval.py`

- [ ] **Step 1: Write failing provider-flow tests**

Create fakes whose `search()` records `timeout_seconds`, query text, and provider name. Add these tests:

```python
class ProviderRunnerTests(unittest.TestCase):
    def test_tavily_success_does_not_call_ddgs(self):
        plan = SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "EDG"),))
        tavily = FakeProvider("tavily", success("https://example.com/a", "A", "body"))
        ddgs = FakeProvider("ddgs", success("https://example.com/b", "B", "body"))
        results = ProviderRunner((tavily, ddgs), 8, 15, 4).run(plan, SearchTrace("r1"))
        self.assertEqual(1, len(results))
        self.assertEqual([], ddgs.calls)

    def test_ddgs_receives_only_unresolved_queries(self):
        plan = SearchPlan(SearchMode.STANDARD, (SearchQuery("q1", "a"), SearchQuery("q2", "b")))
        tavily = QueryFakeProvider("tavily", {"q1": success_hit("q1"), "q2": empty_result()})
        ddgs = QueryFakeProvider("ddgs", {"q2": success_hit("q2")})
        ProviderRunner((tavily, ddgs), 8, 15, 4).run(plan, SearchTrace("r1"))
        self.assertEqual(["q2"], [call.query_id for call in ddgs.calls])

    def test_results_drop_unsafe_urls_and_deduplicate_canonical_urls(self):
        # Fake Tavily returns localhost plus two tracking variants of one public URL.
        self.assertEqual(["https://example.com/a"], [item.url for item in results])

    def test_provider_timeouts_are_forwarded(self):
        self.assertEqual(8, tavily.timeout_values[0])
        self.assertEqual(15, ddgs.timeout_values[0])
```

Use the existing legacy Provider result classes in the fakes during migration so the production providers remain unchanged in this task.

- [ ] **Step 2: Run and verify RED**

Run: `python -B -m unittest tests.test_simple_search_retrieval -v`

Expected: FAIL because `ProviderRunner` is missing.

- [ ] **Step 3: Implement ProviderRunner**

`ProviderRunner.run(plan, trace)` must:

1. submit all Tavily queries to a request-local `ThreadPoolExecutor(max_workers=len(queries))`;
2. pass the configured Tavily timeout into every provider call;
3. collect successful non-empty queries;
4. submit only unresolved queries to DDGS with its configured timeout;
5. convert legacy hits into simple `SearchResult` records;
6. validate with `canonicalize_public_http_url` and deduplicate by canonical URL;
7. cap the combined list at 5 for light and 8 for standard;
8. write only body-free provider statuses and counts to `SearchTrace`.

During coexistence, isolate old types in two private helpers:

```python
def _legacy_query(query: SearchQuery, mode: SearchMode):
    from src.search.models import QueryPurpose, SearchQuery as OldQuery, SearchRoundKind
    return OldQuery(
        query_id=query.query_id,
        query_index=int(query.query_id.removeprefix("q")),
        round_kind=SearchRoundKind.INITIAL,
        purpose=QueryPurpose.DIRECT,
        text=query.text,
    )


def _legacy_tier(mode: SearchMode):
    from src.search.models import SearchTier
    return SearchTier.LIGHT if mode is SearchMode.LIGHT else SearchTier.STANDARD
```

Do not use `ProviderRegistry` or the global stage executor. Provider calls already receive their cooperative timeout; the executor is request-local and shuts down with `wait=False, cancel_futures=True` after collection.

- [ ] **Step 4: Run focused provider tests**

Run: `python -B -m unittest tests.test_simple_search_retrieval -v`

Expected: all new retrieval tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/search/simple/retrieval.py tests/test_simple_search_retrieval.py
git commit -m "feat: add simple provider retrieval"
```

---

### Task 4: Add on-demand page reading

**Files:**
- Create: `src/search/simple/reader.py`
- Test: `tests/test_simple_search_reader.py`

- [ ] **Step 1: Write failing Reader tests**

```python
class OnDemandReaderTests(unittest.TestCase):
    def test_does_not_fetch_eighty_character_excerpt(self):
        result = search_result(excerpt="字" * 80)
        enriched = OnDemandReader(fake_fetch).enrich((result,), limit=1, timeout_seconds=5, trace=trace)
        self.assertEqual((), fake_fetch.calls)
        self.assertEqual(result, enriched[0])

    def test_light_reads_only_first_short_result(self):
        results = (search_result("R1", excerpt="短"), search_result("R2", excerpt="短"))
        enriched = OnDemandReader(fake_fetch_success).enrich(results, limit=1, timeout_seconds=5, trace=trace)
        self.assertEqual(1, len(fake_fetch_success.calls))
        self.assertEqual(1500, len(enriched[0].excerpt))
        self.assertEqual("短", enriched[1].excerpt)

    def test_fetch_failure_preserves_provider_snippet(self):
        enriched = OnDemandReader(fake_fetch_failure).enrich((search_result(excerpt="短摘要"),), limit=1, timeout_seconds=5, trace=trace)
        self.assertEqual("短摘要", enriched[0].excerpt)

    def test_fetch_receives_timeout(self):
        self.assertEqual(5, fake_fetch_success.calls[0].timeout_seconds)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -B -m unittest tests.test_simple_search_reader -v`

Expected: FAIL because `OnDemandReader` is missing.

- [ ] **Step 3: Implement the Reader**

Implement `OnDemandReader(fetch_document=url_fetch.fetch_document)` with:

```python
_MIN_SNIPPET_CHARS = 80
_MAX_EXCERPT_CHARS = 1500


def _compact(text: str) -> str:
    clean = CONTROL_CHARS.sub("", str(text or ""))
    return " ".join(clean.split())


class OnDemandReader:
    def enrich(self, results, *, limit, timeout_seconds, trace):
        output = list(results)
        indexes = [i for i, item in enumerate(output) if len(_compact(item.excerpt)) < 80][:limit]
        # Fetch selected indexes concurrently; replace excerpts only for successful non-empty documents.
        # Preserve title, URL, provider, score, and original ordering.
        return tuple(output)
```

Use `src.services.url_fetch_service.fetch_document`, which already enforces redirects, DNS/IP safety, content type, and response-size bounds. Pass `timeout_seconds` to each fetch. Clean control characters, collapse whitespace, cap successful text to 1500 characters, and increment `trace.reader_count` only for actual fetch attempts.

- [ ] **Step 4: Run Reader and URL-policy tests**

Run: `python -B -m unittest tests.test_simple_search_reader tests.test_search_url_policy -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/search/simple/reader.py tests/test_simple_search_reader.py
git commit -m "feat: read short search snippets on demand"
```

---

### Task 5: Add tolerant relevance scoring

**Files:**
- Create: `src/search/simple/ranking.py`
- Test: `tests/test_simple_search_ranking.py`

- [ ] **Step 1: Write failing ranking tests**

```python
class SearchRankerTests(unittest.TestCase):
    def test_scores_sort_stably_and_zero_is_removed(self):
        llm = FakeLLM('text ```json\n{"scores":{"R1":0.2,"R2":0.9,"R3":0}}\n```')
        ranked = EvidenceRanker(llm).rank("q", three_results(), timeout_seconds=10)
        self.assertEqual(("R2", "R1"), tuple(item.result_id for item in ranked.results))
        self.assertFalse(ranked.degraded)

    def test_missing_and_invalid_scores_default_to_half(self):
        llm = FakeLLM('{"scores":{"R1":"bad","unknown":1,"R3":2}}')
        ranked = EvidenceRanker(llm).rank("q", three_results(), timeout_seconds=10)
        self.assertEqual(1.0, result_by_id(ranked, "R3").score)
        self.assertEqual(0.5, result_by_id(ranked, "R1").score)
        self.assertEqual(0.5, result_by_id(ranked, "R2").score)

    def test_no_valid_scores_preserves_provider_order_and_degrades(self):
        ranked = EvidenceRanker(FakeLLM("natural language")).rank("q", three_results(), timeout_seconds=10)
        self.assertEqual(("R1", "R2", "R3"), tuple(item.result_id for item in ranked.results))
        self.assertTrue(ranked.degraded)

    def test_ranker_forwards_timeout_and_excludes_urls_from_prompt(self):
        self.assertEqual(10, llm.calls[0].kwargs["timeout_seconds"])
        self.assertNotIn("https://", str(llm.calls[0].messages))
```

- [ ] **Step 2: Run and verify RED**

Run: `python -B -m unittest tests.test_simple_search_ranking -v`

Expected: FAIL because `EvidenceRanker` is missing.

- [ ] **Step 3: Implement ranking and permissive parsing**

Define:

```python
@dataclass(frozen=True)
class RankingResult:
    results: tuple[SearchResult, ...]
    degraded: bool


class EvidenceRanker:
    def rank(self, question, results, *, timeout_seconds):
        # One LLM call, title/excerpt/result_id only.
        # Parse the first balanced object or fenced JSON.
        # Default absent/invalid scores to 0.5, clamp finite numbers,
        # remove explicit zero, and use Python's stable sort.
```

Pass `temperature=0.0`, `max_tokens=512`, `tools=None`, `tool_choice="none"`, and `timeout_seconds` directly to `llm.chat`. Any exception or payload with no valid known-ID numeric score returns original ordering with `degraded=True`.

- [ ] **Step 4: Run focused tests**

Run: `python -B -m unittest tests.test_simple_search_ranking -v`

Expected: all ranking tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/search/simple/ranking.py tests/test_simple_search_ranking.py
git commit -m "feat: rank search results with tolerant scores"
```

---

### Task 6: Compose the retrieval pipeline and production factory

**Files:**
- Create: `src/search/simple/pipeline.py`
- Create: `src/search/simple/factory.py`
- Modify: `src/search/simple/__init__.py`
- Test: `tests/test_simple_search_pipeline.py`

- [ ] **Step 1: Write failing orchestration tests**

```python
class SimpleSearchPipelineTests(unittest.TestCase):
    def test_skip_never_calls_retriever_reader_or_ranker(self):
        outcome = pipeline(planner=skip_planner()).run(SearchRequest("你好"))
        self.assertEqual(SearchMode.SKIP, outcome.plan.mode)
        self.assertEqual(0, retriever.calls)

    def test_light_uses_one_reader_and_standard_uses_two(self):
        light = pipeline(planner=light_planner(), retriever=short_results(3)).run(SearchRequest("q"))
        standard = pipeline(planner=standard_planner(), retriever=short_results(3)).run(SearchRequest("q", force_search=True))
        self.assertEqual(1, light.trace.reader_count)
        self.assertEqual(2, standard.trace.reader_count)

    def test_ranker_failure_sets_warning_but_keeps_results(self):
        outcome = pipeline(ranker=degraded_ranker()).run(SearchRequest("q"))
        self.assertEqual("信息可能不完整。", outcome.warning)
        self.assertTrue(outcome.results)

    def test_no_usable_results_is_the_only_content_failure(self):
        outcome = pipeline(retriever=empty_retriever()).run(SearchRequest("q"))
        self.assertEqual(SearchFailure.NO_RESULTS, outcome.failure)

    def test_unexpected_exception_is_contained(self):
        outcome = pipeline(planner=raising_planner()).run(SearchRequest("q"))
        self.assertEqual(SearchFailure.PROVIDER_UNAVAILABLE, outcome.failure)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -B -m unittest tests.test_simple_search_pipeline -v`

Expected: FAIL because `SimpleSearchPipeline` and factory functions are missing.

- [ ] **Step 3: Implement orchestration and timing**

Implement this public interface:

```python
class SimpleSearchPipeline:
    def __init__(self, planner, retriever, reader, ranker, *, timeouts, clock=time.monotonic):
        self._planner = planner
        self._retriever = retriever
        self._reader = reader
        self._ranker = ranker
        self._timeouts = timeouts
        self._clock = clock

    def _timed(self, name, call, trace):
        started = self._clock()
        try:
            return call()
        finally:
            trace.stage_latency_ms[name] = max((self._clock() - started) * 1000.0, 0.0)

    def run(self, request: SearchRequest) -> SearchOutcome:
        trace = SearchTrace(request_id=uuid.uuid4().hex)
        try:
            plan = self._timed("planner", lambda: self._planner.plan(request, timeout_seconds=self._timeouts.planner), trace)
            trace.mode = plan.mode
            trace.query_count = len(plan.queries)
            trace.planner_degraded = plan.planner_degraded
            if plan.mode is SearchMode.SKIP:
                return SearchOutcome(plan, (), trace)
            results = self._timed("providers", lambda: self._retriever.run(plan, trace), trace)
            limit = 1 if plan.mode is SearchMode.LIGHT else 2
            results = self._timed("reader", lambda: self._reader.enrich(results, limit=limit, timeout_seconds=self._timeouts.reader, trace=trace), trace)
            usable = tuple(item for item in results if item.url and (item.title.strip() or item.excerpt.strip()))
            if not usable:
                return SearchOutcome(plan, (), trace, failure=SearchFailure.NO_RESULTS)
            ranking = self._timed("ranker", lambda: self._ranker.rank(request.question, usable, timeout_seconds=self._timeouts.ranker), trace)
            trace.judge_degraded = ranking.degraded
            trace.candidate_count = len(ranking.results)
            if not ranking.results:
                return SearchOutcome(plan, (), trace, failure=SearchFailure.NO_RESULTS)
            warning = "信息可能不完整。" if ranking.degraded else None
            return SearchOutcome(plan, ranking.results, trace, warning=warning)
        except Exception:
            logger.exception("simple search pipeline failed")
            fallback = SearchPlan(SearchMode.STANDARD if request.force_search else SearchMode.LIGHT, (SearchQuery("q1", request.question),), True)
            trace.mode = fallback.mode
            trace.planner_degraded = True
            return SearchOutcome(fallback, (), trace, failure=SearchFailure.PROVIDER_UNAVAILABLE)
```

Add a frozen `SearchTimeouts` record populated from `config`. `factory.py` constructs `RoutePlanner(get_llm_client())`, `TavilySearchProvider`, `DDGSSearchProvider`, `ProviderRunner`, `OnDemandReader`, and `EvidenceRanker`, and exposes cached `get_simple_search_pipeline()` plus `reset_simple_search_pipeline()`.

- [ ] **Step 4: Run pipeline tests**

Run: `python -B -m unittest tests.test_simple_search_pipeline -v`

Expected: all pipeline tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/search/simple/pipeline.py src/search/simple/factory.py src/search/simple/__init__.py tests/test_simple_search_pipeline.py
git commit -m "feat: compose simple search pipeline"
```

---

### Task 7: Add natural-language answering and deterministic rendering

**Files:**
- Create: `src/search/simple/answering.py`
- Create: `src/search/simple/rendering.py`
- Modify: `src/chat/prompt.py`
- Test: `tests/test_simple_search_answering.py`

- [ ] **Step 1: Write failing answer and renderer tests**

```python
class SimpleSearchAnswerTests(unittest.TestCase):
    def test_model_receives_evidence_without_urls(self):
        answer = SearchAnswerer(FakeLLM("EDG目前仍有晋级可能。" )).answer(
            "EDG能去吗", results(), base_messages=[], timeout_seconds=20
        )
        self.assertEqual("EDG目前仍有晋级可能。", answer.text)
        self.assertNotIn("https://", serialized_model_messages())

    def test_answer_timeout_returns_ranked_summary(self):
        answer = SearchAnswerer(FakeLLM(error=TimeoutError())).answer(
            "q", results(), base_messages=[], timeout_seconds=20
        )
        self.assertTrue(answer.degraded)
        self.assertIn("根据搜索结果：", answer.text)
        self.assertIn("标题：摘要", answer.text)

    def test_normal_chat_hides_sources(self):
        rendered = render_search_answer("回答", results(), warning=None, show_sources=False, qq_limit=1700, trace=trace)
        self.assertNotIn("https://", rendered.text)

    def test_command_shows_at_most_three_sources(self):
        rendered = render_search_answer("回答", four_results(), warning=None, show_sources=True, qq_limit=1700, trace=trace)
        self.assertEqual(3, rendered.text.count("https://"))

    def test_warning_appears_once(self):
        rendered = render_search_answer("回答", results(), warning="信息可能不完整。", show_sources=False, qq_limit=1700, trace=trace)
        self.assertEqual(1, rendered.text.count("信息可能不完整。"))
```

- [ ] **Step 2: Run and verify RED**

Run: `python -B -m unittest tests.test_simple_search_answering -v`

Expected: FAIL because answering and rendering modules are missing.

- [ ] **Step 3: Implement natural-language answer handling**

Add `build_search_system_prompt(context)` to `src/chat/prompt.py`. It must retain capability, privacy, prompt-injection, and persona boundaries but replace the GroundedDraft schema with:

```text
Use only the supplied search titles and excerpts for externally verifiable facts.
Answer naturally in Simplified Chinese. If the excerpts do not settle a detail,
say that it is uncertain. Do not output or invent URLs, source IDs, JSON, or an
internal verification status.
```

Implement:

```python
@dataclass(frozen=True)
class AnswerResult:
    text: str
    degraded: bool


class SearchAnswerer:
    def __init__(self, llm):
        self._llm = llm

    def answer(self, question, results, *, base_messages, timeout_seconds):
        evidence = [{"title": r.title, "excerpt": r.excerpt[:1500]} for r in results]
        messages = [*base_messages, {"role": "user", "content": json.dumps({"question": question, "search_results": evidence}, ensure_ascii=False)}]
        try:
            response = self._llm.chat(messages, temperature=0.2, timeout_seconds=timeout_seconds)
            text = _clean_text(getattr(response, "content", ""))
            if text:
                return AnswerResult(text, False)
        except Exception:
            logger.debug("search answer model failed", exc_info=True)
        return AnswerResult(_summary_fallback(results), True)
```

`render_search_answer(text, results, *, warning, show_sources, qq_limit, trace)` appends the warning once, optionally appends at most three `title + URL` source lines, and truncates components before joining so the result never exceeds `qq_limit`. It returns `SearchResponse(final_text, shown_sources, trace)` and never extracts URLs from model text.

- [ ] **Step 4: Run focused tests**

Run: `python -B -m unittest tests.test_simple_search_answering -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/search/simple/answering.py src/search/simple/rendering.py src/chat/prompt.py tests/test_simple_search_answering.py
git commit -m "feat: answer search results in natural language"
```

---

### Task 8: Switch chat and `/search` to the new pipeline

**Files:**
- Modify: `src/chat/chat_service.py`
- Modify: `src/services/search_service.py`
- Modify: `src/commands/search.py`
- Test: `tests/test_simple_search_chat_flow.py`
- Modify: `tests/test_identity_configuration.py`
- Modify: `tests/test_multimodal_chat.py`
- Modify: `tests/test_main_image_flow.py`
- Modify: `tests/test_product_scope.py`

- [ ] **Step 1: Write new black-box chat tests**

```python
class SimpleSearchChatFlowTests(unittest.TestCase):
    def test_force_search_sends_command_source_and_shows_sources(self):
        engine = FakeEngine(standard_outcome(results=results()))
        with patch("src.chat.chat_service.get_simple_search_pipeline", return_value=engine):
            reply = generate_reply("private:1", "EDG能去吗", force_search=True)
        self.assertEqual(RequestSource.COMMAND, engine.requests[0].request_source)
        self.assertIn("https://example.com", reply)

    def test_normal_search_hides_sources(self):
        engine = FakeEngine(light_outcome(results=results()))
        with patch("src.chat.chat_service.get_simple_search_pipeline", return_value=engine):
            reply = generate_reply("private:1", "EDG能去吗")
        self.assertNotIn("https://", reply)

    def test_skip_uses_plain_chat_without_search_answerer(self):
        engine = FakeEngine(skip_outcome())
        reply = run_with(engine, plain_llm_text="你好呀！")
        self.assertEqual("你好呀！", reply)
        self.assertEqual(0, answerer.calls)

    def test_search_answer_timeout_returns_summary_and_saves_history(self):
        reply = run_with(light_outcome(results=results()), answer_error=TimeoutError())
        self.assertIn("根据搜索结果：", reply)
        self.assertEqual(reply, chat_history["private:1"][-1]["content"])

    def test_unexpected_engine_error_never_escapes(self):
        engine = FakeEngine(error=RuntimeError("boom"))
        reply = run_with(engine)
        self.assertEqual("在线搜索暂时不可用，请稍后再试。", reply)
```

- [ ] **Step 2: Run and verify tests fail against old dispatch**

Run: `python -B -m unittest tests.test_simple_search_chat_flow -v`

Expected: FAIL because chat still calls `get_search_orchestrator()` and old answer policy.

- [ ] **Step 3: Replace chat search dispatch**

In `src/chat/chat_service.py`:

1. preserve history loading/saving, multimodal user content, tool-protocol helpers, and plain chat;
2. replace old search imports with `SearchRequest`, `RequestSource`, `SearchMode`, `get_simple_search_pipeline`, `SearchAnswerer`, and `render_search_answer`;
3. replace `get_search_orchestrator_for_chat()` with a cached `get_simple_search_pipeline_for_chat()` and matching reset;
4. call the engine exactly once per message;
5. use ordinary chat generation for skip;
6. use SearchAnswerer and command-only sources for successful search;
7. render a fixed provider/no-results message when no results exist;
8. pass `config.search_answer_timeout` directly to `llm.chat` for both plain and search answers;
9. wrap the complete dispatch in `try/except Exception` and always return a non-empty fixed failure message;
10. append history only after a reply exists.

Update `src/services/search_service.py` to call `get_simple_search_pipeline().run(SearchRequest(normalized, force_search=True, request_source=RequestSource.COMPATIBILITY))` and flatten the returned ranked results. `src/commands/search.py` continues calling `generate_reply` with `force_search=True`, the normalized query, session key, and original history text.

Update identity, multimodal, main-image, and product-scope tests to construct simple `SearchOutcome` fakes instead of old `SearchPipelineResult` graphs.

- [ ] **Step 4: Run chat and adjacent tests**

Run:

```bash
python -B -m unittest \
  tests.test_simple_search_chat_flow \
  tests.test_identity_configuration \
  tests.test_multimodal_chat \
  tests.test_main_image_flow \
  tests.test_product_scope -v
```

Expected: all selected tests pass; no old orchestrator patch is needed by these files.

- [ ] **Step 5: Commit the cutover**

```bash
git add src/chat/chat_service.py src/services/search_service.py src/commands/search.py tests/test_simple_search_chat_flow.py tests/test_identity_configuration.py tests/test_multimodal_chat.py tests/test_main_image_flow.py tests/test_product_scope.py
git commit -m "feat: route chat through simple search"
```

---

### Task 9: Migrate Providers and remove the old state machine atomically

**Files:**
- Create: `src/search/simple/providers.py`
- Modify: `src/search/providers/base.py`
- Modify: `src/search/providers/tavily.py`
- Modify: `src/search/providers/ddgs.py`
- Modify: `src/search/simple/retrieval.py`
- Modify: `src/search/simple/factory.py`
- Modify: `src/search/__init__.py`
- Modify: `tests/test_search_providers.py`
- Modify: `tests/test_simple_search_retrieval.py`
- Modify: `tests/test_simple_search_models.py`
- Delete: legacy production and test files listed in the file map

Provider migration and old-runtime deletion belong in one commit: changing Provider result identities before deleting the old orchestrator would leave the repository's full suite broken between tasks.

- [ ] **Step 1: Add final Provider-interface and legacy-removal tests**

Update Provider tests to call:

```python
result = provider.search(
    SearchQuery("q1", "EDG 上海冠军赛"),
    mode=SearchMode.LIGHT,
    max_results=4,
    timeout_seconds=8,
)
```

Assert that light Tavily uses `search_depth="basic"` without raw content, standard uses `search_depth="advanced"` with raw content, DDGS receives its bounded timeout, and optional query dates are forwarded only when present.

Add to `tests/test_simple_search_models.py`:

```python
def test_legacy_search_runtime_is_removed(self):
    root = Path(__file__).resolve().parents[1]
    removed = (
        "budget.py", "evidence.py", "extraction.py", "models.py", "orchestrator.py",
        "outcomes.py", "planner.py", "policy.py", "renderer.py", "router.py",
        "stage_runner.py", "validation.py",
    )
    self.assertEqual([], [name for name in removed if (root / "src" / "search" / name).exists()])
    production = "\n".join(path.read_text(encoding="utf-8") for path in (root / "src").rglob("*.py"))
    for forbidden in ("GroundedDraft", "ClaimDiscovery", "SemanticVerifier", "RepairPlan", "fail_closed"):
        self.assertNotIn(forbidden, production)
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
python -B -m unittest \
  tests.test_search_providers \
  tests.test_simple_search_retrieval \
  tests.test_simple_search_models.SimpleSearchModelTests.test_legacy_search_runtime_is_removed -v
```

Expected: FAIL because Providers still require the old `tier` contract and legacy runtime files still exist.

- [ ] **Step 3: Introduce final Provider records, switch callers, and delete legacy code**

Create `src/search/simple/providers.py` with these final records:

```python
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ProviderStatus(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"


class ProviderErrorCode(StrEnum):
    INVALID_PARAMETERS = "invalid_parameters"
    CONNECTION = "connection"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderReadiness:
    provider: str
    configured: bool
    available: bool


@dataclass(frozen=True)
class ProviderHit:
    provider: str
    query_id: str
    title: str
    url: str
    snippet: str | None = None
    score: float | None = None
    raw_content: str | None = None


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    status: ProviderStatus
    hits: tuple[ProviderHit, ...] = ()
    latency_ms: float = 0.0
    error_code: ProviderErrorCode | None = None
    date_filter_normalized: bool = False
    parameter_retry_attempted: bool = False
```

Change `SearchProvider.search` to:

```python
def search(
    self,
    query: SearchQuery,
    *,
    mode: SearchMode,
    max_results: int,
    timeout_seconds: float,
) -> ProviderResult:
    raise NotImplementedError
```

Update Tavily and DDGS to import these records plus `SearchMode/SearchQuery`. Tavily chooses basic/advanced from `mode`, forwards timeout, and retains one budget-contained retry when supplied optional date bounds are rejected. Update `ProviderRunner` to use `mode=plan.mode`; delete `_legacy_query`, `_legacy_tier`, `ProviderRegistry`, and every simple-pipeline import from `src.search.models`.

Delete exactly the legacy production and test files listed in the file map. Rewrite `src/search/__init__.py` to export:

```python
from .simple import (
    RequestSource, SearchFailure, SearchMode, SearchOutcome, SearchPlan,
    SearchQuery, SearchRequest, SearchResponse, SearchResult, SearchTrace,
    get_simple_search_pipeline, reset_simple_search_pipeline,
)
from .url_policy import UrlDecision, canonicalize_public_http_url, evaluate_public_http_url
```

Run the static scan and update every remaining live import to the simple API:

```bash
rg -n "src\.search\.(models|orchestrator|planner|router|evidence|validation|policy|renderer|outcomes|budget|stage_runner|extraction)" src tests tools
```

Do not add compatibility aliases for deleted risk, topic, freshness, claim, validation, or repair types.

- [ ] **Step 4: Run the complete maintained suite**

Run: `python -B -m unittest discover -s tests -t . -q`

Expected: exit 0. Also expect the static scan in Step 3 to return no matches.

- [ ] **Step 5: Commit Provider migration and deletion together**

```bash
git add -u src/search tests
git add src/search/simple/providers.py src/search/providers/base.py src/search/providers/tavily.py src/search/providers/ddgs.py src/search/simple/retrieval.py src/search/simple/factory.py src/search/__init__.py tests/test_search_providers.py tests/test_simple_search_retrieval.py tests/test_simple_search_models.py
git commit -m "refactor: remove legacy evidence search pipeline"
```

---

### Task 10: Replace evaluation and documentation

**Files:**
- Rewrite: `tools/evaluate_search.py`
- Rewrite: `tests/test_search_evaluation.py`
- Modify: `tests/test_readme_guide.py`
- Modify: `README.md`
- Modify: `.env.example` if present
- Modify: `eval/search/README.md`

- [ ] **Step 1: Write evaluator behavior tests**

The new evaluator accepts JSONL rows containing `mode`, `query_count`, `provider_statuses`, `candidate_count`, `planner_degraded`, `judge_degraded`, `answer_degraded`, and `output_kind`. Test:

```python
class SimpleSearchEvaluationTests(unittest.TestCase):
    def test_explicit_search_must_be_standard(self):
        report = evaluate_rows([row(request_source="command", mode="light")])
        self.assertIn("command_not_standard", report["violations"])

    def test_light_and_standard_query_caps(self):
        report = evaluate_rows([row(mode="light", query_count=2), row(mode="standard", query_count=4)])
        self.assertEqual(2, report["violations"]["query_cap_exceeded"])

    def test_trace_rejects_body_fields(self):
        report = evaluate_rows([dict(row(), url="https://example.com")])
        self.assertEqual(1, report["violations"]["unsafe_trace_field"])
```

- [ ] **Step 2: Run and verify RED**

Run: `python -B -m unittest tests.test_search_evaluation -v`

Expected: FAIL because the old evaluator expects evidence-state and claim metrics.

- [ ] **Step 3: Replace old evaluation contracts and update docs**

Rewrite `tools/evaluate_search.py` as a small CLI with:

```text
python tools/evaluate_search.py traces path/to/traces.jsonl
python tools/evaluate_search.py smoke
```

`traces` reports route counts, query-cap violations, provider success, Planner/Judge/Answer degradation rates, output kinds, and unsafe Trace fields. `smoke` runs an explicitly authorized short list through the production simple pipeline and prints body-free traces; it does not claim quality certification.

Update README and `eval/search/README.md` to state:

- light is one query; standard is up to three parallel queries;
- `/search` always uses standard and shows up to three sources;
- ordinary chat hides sources;
- Planner/Judge/Answer failures degrade instead of failing the request;
- no medical/legal/high-risk policy exists;
- no Repair, Claim Discovery, or Semantic Verifier remains;
- list all six timeout environment variables and defaults.

- [ ] **Step 4: Run evaluator/docs tests**

Run:

```bash
python -B -m unittest tests.test_search_evaluation tests.test_readme_guide -v
python -B tools/evaluate_search.py --help
```

Expected: tests pass and CLI help exits 0 with `traces` and `smoke` subcommands.

- [ ] **Step 5: Commit**

```bash
git add tools/evaluate_search.py tests/test_search_evaluation.py tests/test_readme_guide.py README.md eval/search/README.md .env.example
git commit -m "docs: document simplified search behavior"
```

If `.env.example` does not exist, omit it from `git add`; do not create a second environment template.

---

### Task 11: Verify behavior, live smoke test, and final cleanup

**Files:**
- Modify only files required by failures found in this task

- [ ] **Step 1: Run the complete hermetic suite**

Run: `python -B -m unittest discover -s tests -t . -q`

Expected: exit 0 with every maintained test passing.

- [ ] **Step 2: Run static and syntax checks**

Run:

```bash
python -B -m compileall -q src tests tools
git diff --check
rg -n "GroundedDraft|ClaimDiscovery|SemanticVerifier|RepairPlan|fail_closed|SearchTier\.DEEP" src tests tools
```

Expected: compile and diff checks exit 0; `rg` returns no matches.

- [ ] **Step 3: Run focused regression probes**

Run a local script with mocked providers/LLM to verify these exact outcomes:

```text
你好 → skip → plain reply
EDG能去上海冠军赛吗 → light → natural answer without URL
/search EDG能去上海冠军赛吗 → standard → answer plus at most three URLs
Judge timeout → answer plus “信息可能不完整。”
Answer timeout → deterministic title/summary output
```

Expected: all five probes produce non-empty replies and no exception.

- [ ] **Step 4: Run authorized live smoke queries**

With configured credentials, run one normal current-fact query and one `/search` query. Confirm from safe Trace output that normal chat uses no more than one query when routed light, `/search` uses standard, Provider fallback works when induced, and no raw query/URL/body appears in Trace. Do not print API keys or full response bodies.

- [ ] **Step 5: Review size and commit any verification fixes**

Run:

```bash
python - <<'PY'
from pathlib import Path
files = list(Path('src/search').rglob('*.py'))
print(sum(len(path.read_text(encoding='utf-8').splitlines()) for path in files))
PY
git status --short
git diff --stat
```

The final search package should be materially smaller than the current 10,479-line baseline and contain no unused compatibility layer. If verification required fixes, commit only those tracked files:

```bash
git add -u
git commit -m "fix: complete simple search migration"
```

Do not add `.tmp.driveupload/`, `docs/plans/`, or `websearch-simplification-report.md`.
