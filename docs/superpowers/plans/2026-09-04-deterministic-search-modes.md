# Deterministic Search Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace model-selected routing with caller-owned `LIGHT`, `STANDARD`, and `SKIP` modes while delivering a bounded multimodal Tavily-first search pipeline and a search-free `/skip` path.

**Architecture:** Perform a true clean rewrite entirely under `src/search/simple/` around an explicit `SearchRequest(mode, text, images)` contract: callers select the mode, final provider records/protocol and new Tavily/DDGS implementations are built there before retrieval/factory, `QueryPlanner` returns queries only through tolerant balanced JSON parsing, and the pipeline performs bounded retrieval, reading, and ranking without changing mode. Existing partial `simple` files may be overwritten; legacy state-machine, provider, and model source may be read only as behavior reference and must never be imported, adapted, or modified during implementation. Text, image-only, and text-plus-image inputs use the same explicit contract; `/skip` branches before pipeline construction and therefore emits no search trace; after cutover, all legacy search runtime is deleted atomically.

**Tech Stack:** Python 3.11+, dataclasses, `StrEnum`, `concurrent.futures`, existing Gemini/DeepSeek multimodal chat client, Tavily, DDGS, `requests`, `unittest`/`unittest.mock`.

---

> **Supersedes:** This plan supersedes `docs/superpowers/plans/2026-09-03-simple-search-pipeline.md`. Do not execute tasks from the superseded plan after starting this one. In particular, do not preserve its `force_search`, `has_images`, `RoutePlanner`, model-selected mode, or heuristic `SKIP` behavior.

## File map

### Clean-rewrite production files

- `src/search/simple/models.py` — replace current partial contracts with caller-owned mode, normalized text/image inputs, fixed query plans, result/outcome/trace records, and safe trace serialization.
- `src/search/simple/planning.py` — replace current `RoutePlanner` and routing heuristics with multimodal `QueryPlanner`, balanced-JSON parsing, query normalization, and deterministic fallbacks.
- `src/search/simple/providers.py` — final provider statuses, readiness/hit/result records, and protocol, defined before any provider or factory uses them.
- `src/search/simple/tavily.py` — new Tavily implementation using only simple contracts.
- `src/search/simple/ddgs.py` — new DDGS implementation using only simple contracts.
- `src/search/simple/retrieval.py` — replace the current partial adapter with simple-provider-only bounded Tavily-first execution, invalid-URL-aware DDGS fallback, canonical deduplication, and mode caps.
- `src/search/simple/reader.py` — request-local bounded page reads for missing/short snippets.
- `src/search/simple/ranking.py` — one tolerant relevance call with stable provider-order degradation.
- `src/search/simple/pipeline.py` — fixed-mode planning/retrieval/read/rank orchestration and failure boundary.
- `src/search/simple/answering.py` — evidence-only Simplified Chinese answer generation plus deterministic summary fallback.
- `src/search/simple/rendering.py` — replacement QQ message splitter, reply length bounds, warning placement, and `STANDARD` command-only source display.
- `src/search/simple/factory.py` — production dependency construction, timeout bundle, readiness, cache, and reset hook.
- `src/search/simple/__init__.py` — narrow public API for the rewritten runtime.

### Modified integration/configuration files

- `src/config.py` — retain the six active search timeout settings and normalize non-finite or sub-`0.1` values safely; no route-policy setting is introduced.
- `src/chat/prompt.py` — add natural search-answer instructions without legacy grounded-draft schemas.
- `src/chat/chat_service.py` — require explicit mode and caller-provided images; branch `SKIP` before factory access; search-answer and history integration.
- `src/commands/search.py` — invoke chat with `SearchMode.STANDARD`, including downloaded images and original command history text.
- `src/commands/skip.py` — new `/skip` handler invoking plain multimodal chat with `SearchMode.SKIP` and an empty-usage response.
- `src/commands/__init__.py` — add image inputs to `CommandContext`, register `skip`, and list it in unknown-command output.
- `src/commands/help.py` — document deterministic normal/search/skip behavior and image support.
- `src/main.py` — download images before command dispatch, pass them through command context, use explicit `LIGHT` for ordinary chat, preserve image cleanup, and expose new provider readiness.
- `src/services/search_service.py` — compatibility facade that always constructs a `STANDARD` request.
- `src/search/__init__.py` — export only simple runtime records/factory plus URL policy.
- `tools/evaluate_search.py` — deterministic-mode trace evaluator and explicitly authorized smoke command.
- `README.md` — user-facing modes, `/skip`, image behavior, source visibility, fallback/degradation, and timeout configuration.
- `.env.example` — list all six active search timeout variables and remove obsolete route-policy variables.
- `eval/search/README.md` — describe safe traces and deterministic evaluator metrics.

### New/reworked maintained tests

- `tests/test_simple_search_models.py` — rewrite current partial model tests for explicit mode/images and safe traces.
- `tests/test_simple_search_planning.py` — replace current `RoutePlanner` tests with fixed-mode multimodal `QueryPlanner` tests.
- `tests/test_simple_search_retrieval.py` — rewrite current partial retrieval tests, especially invalid Tavily URL fallback.
- `tests/test_simple_search_reader.py` — page-read selection, timeout forwarding, bounded concurrency, and snippet preservation.
- `tests/test_simple_search_ranking.py` — tolerant scoring, stable fallback, prompt privacy, and timeout forwarding.
- `tests/test_simple_search_pipeline.py` — fixed-mode orchestration, `SKIP` bypass, caps, traces, and failures.
- `tests/test_simple_search_answering.py` — answer prompt, image preservation, deterministic fallback, and rendering.
- `tests/test_simple_search_rendering.py` — exact replacement QQ splitting semantics used by `src/main.py`.
- `tests/test_simple_search_chat_flow.py` — explicit mode contract and black-box normal/search/skip behavior.
- `tests/test_command_renderer.py` — registry/help/unknown command assertions including `/skip`.
- `tests/test_main_image_flow.py` — command image download and propagation for `/search` and `/skip`.
- `tests/test_multimodal_chat.py` — planner and answer multimodal content plus image-safe history.
- `tests/test_identity_configuration.py` — simple-runtime fakes instead of legacy state graphs.
- `tests/test_product_scope.py` and `tests/test_user_facing_scope.py` — deterministic mode and `/skip` product wording.
- `tests/test_simple_search_providers.py` — new simple provider protocol/signature and Tavily/DDGS timeout/date behavior.
- `tests/test_search_service.py` — `STANDARD` compatibility facade and reset behavior.
- `tests/test_search_evaluation.py` — deterministic trace evaluator for chat/LIGHT and command-or-compatibility/STANDARD traces only.
- `tests/test_readme_guide.py` — README commands, modes, images, sources, and timeout variables.
- `tests/test_search_url_policy.py` — retained unchanged as the URL safety regression suite.

### Atomic legacy deletion after cutover

Delete these production files in the final cleanup commit; until then they are read-only behavior references and no new code may import them:

- `src/search/providers/__init__.py`
- `src/search/providers/base.py`
- `src/search/providers/tavily.py`
- `src/search/providers/ddgs.py`
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

Delete these obsolete tests/tools in the same commit:

- `tests/search_fakes.py`
- `tests/test_chat_retrieval_flow.py`
- `tests/test_search_blind_acceptance_runner.py`
- `tests/test_search_budget.py`
- `tests/test_search_evidence.py`
- `tests/test_search_extraction.py`
- `tests/test_search_models.py`
- `tests/test_search_orchestrator.py`
- `tests/test_search_outcomes.py`
- `tests/test_search_planner.py`
- `tests/test_search_policy.py`
- `tests/test_search_provider_batches.py`
- `tests/test_search_providers.py`
- `tests/test_search_renderer.py`
- `tests/test_search_router.py`
- `tests/test_search_simplification_baseline.py`
- `tests/test_search_stage_runner.py`
- `tests/test_search_validation.py`
- `tools/run_search_blind_acceptance.py`

Keep `src/search/url_policy.py` and `tests/test_search_url_policy.py`.

---

### Task 1: Rewrite contracts around caller-owned mode and images

**Files:**
- Rewrite: `src/search/simple/models.py`
- Create: `src/search/simple/providers.py`
- Modify: `src/search/simple/__init__.py`
- Modify: `src/config.py`
- Rewrite: `tests/test_simple_search_models.py`
- Create: `tests/test_simple_search_providers.py`

- [ ] **Step 1: Replace the partial model tests with failing explicit-contract tests**

Use these tests as the core of `tests/test_simple_search_models.py`:

```python
import math
import unittest

from src.config import Config
from src.search.simple.models import (
    OutputKind,
    RequestSource,
    SearchMode,
    SearchPlan,
    SearchQuery,
    SearchRequest,
    SearchTrace,
)


class SimpleSearchModelTests(unittest.TestCase):
    def test_request_normalizes_text_images_and_owns_mode(self):
        request = SearchRequest(
            mode=SearchMode.STANDARD,
            text="  看看   这个  ",
            images=[" data:image/png;base64,AAA ", ""],
            source=RequestSource.COMMAND,
        )
        self.assertIs(SearchMode.STANDARD, request.mode)
        self.assertEqual("看看 这个", request.text)
        self.assertEqual(("data:image/png;base64,AAA",), request.images)
        self.assertFalse(hasattr(request, "force_" + "search"))
        self.assertFalse(hasattr(request, "has_" + "images"))

    def test_text_or_image_is_required(self):
        with self.assertRaisesRegex(ValueError, "text or images"):
            SearchRequest(mode=SearchMode.LIGHT, text="", images=())

    def test_plan_enforces_fixed_query_counts(self):
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.LIGHT, ())
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"), SearchQuery("q2", "b")))
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.STANDARD, tuple(SearchQuery(f"q{i}", str(i)) for i in range(4)))
        self.assertEqual((), SearchPlan(SearchMode.SKIP, ()).queries)

    def test_safe_trace_has_closed_metadata_and_no_request_content(self):
        trace = SearchTrace(
            "r1", source=RequestSource.CHAT,
            mode=SearchMode.LIGHT, query_count=1,
        )
        trace.provider_statuses["tavily"] = "error"
        trace.output_kind = OutputKind.SEARCH_FAILURE
        safe = trace.to_safe_dict()
        self.assertEqual("chat", safe["source"])
        self.assertEqual("light", safe["mode"])
        self.assertEqual("error", safe["provider_statuses"]["tavily"])
        self.assertNotIn("text", safe)
        self.assertNotIn("images", safe)
        self.assertNotIn("url", repr(safe).lower())

    def test_all_search_timeouts_are_finite_and_at_least_point_one(self):
        fields = (
            "search_planner_timeout", "search_tavily_timeout",
            "search_ddgs_timeout", "search_reader_timeout",
            "search_ranker_timeout", "search_answer_timeout",
        )
        for field in fields:
            for value in (math.nan, math.inf, -math.inf, 0.0, -1.0):
                with self.subTest(field=field, value=value):
                    self.assertEqual(0.1, getattr(Config(**{field: value}), field))
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run: `python -B -m unittest tests.test_simple_search_models tests.test_simple_search_providers -v`

Expected: FAIL because `SearchRequest` still accepts `question/force_search/has_images` instead of `mode/text/images`, and the final simple provider contract does not exist.

- [ ] **Step 3: Rewrite the production records with these exact public signatures**

`src/search/simple/models.py` must define:

```python
@dataclass(frozen=True)
class SearchRequest:
    mode: SearchMode
    text: str
    images: tuple[str, ...] = ()
    source: RequestSource = RequestSource.CHAT

    def __post_init__(self) -> None:
        mode = SearchMode(self.mode)
        text = " ".join(str(self.text or "").split())
        images = tuple(
            normalized
            for item in self.images
            if (normalized := str(item or "").strip())
        )
        if not text and not images:
            raise ValueError("text or images must be provided")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "text", text[:500])
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "source", RequestSource(self.source))


@dataclass(frozen=True)
class SearchQuery:
    query_id: str
    text: str
    date_from: date | None = None
    date_to: date | None = None
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    news: bool = False


@dataclass(frozen=True)
class SearchPlan:
    mode: SearchMode
    queries: tuple[SearchQuery, ...]
    planner_degraded: bool = False
```

Keep `SearchResult`, `SearchFailure`, `OutputKind`, `SearchTrace`, `SearchOutcome`, and `SearchResponse`, but rename `judge_degraded` to `ranker_degraded`. Give `SearchTrace` the required constructor prefix `SearchTrace(request_id: str, source: RequestSource, mode: SearchMode, query_count: int = 0)`; coerce `source` and `mode` in `__post_init__`. `SearchTrace.to_safe_dict()` must return only request ID, source, mode, counts, provider status strings, the three degradation booleans, latency numbers, and output kind. Normalize query whitespace; reject reversed dates; enforce `SKIP == 0`, `LIGHT == 1`, and `STANDARD == 1..3` queries. Do not add compatibility properties.

Create `src/search/simple/providers.py` now—before retrieval or factory—with these final types and no imports from `src.search.models` or `src.search.providers`:

```python
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


class SearchProvider(Protocol):
    name: str

    def readiness(self) -> ProviderReadiness: ...

    def search(
        self,
        query: SearchQuery,
        *,
        mode: SearchMode,
        max_results: int,
        timeout_seconds: float,
    ) -> ProviderResult: ...
```

Add contract tests that instantiate every record, define a conforming fake provider, invoke `search(SearchQuery(...), mode=SearchMode.LIGHT, max_results=4, timeout_seconds=8)`, and assert `isinstance(fake, SearchProvider)` after decorating the protocol with `@runtime_checkable`. Also assert the source of `src/search/simple/providers.py` contains neither `"src.search." + "models"` nor `"src.search." + "providers"`; concatenate these strings in the test so the final guard scan remains clean.

Retain these exact timeout fields/defaults in `Config` and normalize non-finite or `<0.1` values to `0.1`:

```python
search_planner_timeout: float = env_float("SEARCH_PLANNER_TIMEOUT", 8.0)
search_tavily_timeout: float = env_float("SEARCH_TAVILY_TIMEOUT", 8.0)
search_ddgs_timeout: float = env_float("SEARCH_DDGS_TIMEOUT", 15.0)
search_reader_timeout: float = env_float("SEARCH_READER_TIMEOUT", 5.0)
search_ranker_timeout: float = env_float("SEARCH_RANKER_TIMEOUT", 10.0)
search_answer_timeout: float = env_float("SEARCH_ANSWER_TIMEOUT", 20.0)
```

- [ ] **Step 4: Run GREEN tests**

Run: `python -B -m unittest tests.test_simple_search_models tests.test_simple_search_providers -v`

Expected: all model/config/provider-contract tests pass.

- [ ] **Step 5: Commit the contract rewrite**

```bash
git add src/search/simple/models.py src/search/simple/providers.py src/search/simple/__init__.py src/config.py tests/test_simple_search_models.py tests/test_simple_search_providers.py
git commit -m "refactor: define clean simple search contracts"
```

---

### Task 2: Replace `RoutePlanner` with a fixed-mode multimodal `QueryPlanner`

**Files:**
- Rewrite: `src/search/simple/planning.py`
- Modify: `src/search/simple/__init__.py`
- Rewrite: `tests/test_simple_search_planning.py`

- [ ] **Step 1: Write failing fixed-mode planner tests**

Replace model-routing tests with:

```python
import unittest
from types import SimpleNamespace

from src.search.simple.models import SearchMode
from src.search.simple.planning import IMAGE_ONLY_FALLBACK_QUERY, QueryPlanner


class FakeLLM:
    def __init__(self, content="", error=None):
        self.content, self.error, self.calls = content, error, []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


class QueryPlannerTests(unittest.TestCase):
    def test_light_ignores_returned_mode_and_keeps_exactly_one_query(self):
        llm = FakeLLM('{"mode":"standard","queries":["  EDG   上海 ","second"]}')
        plan = QueryPlanner(llm).plan(
            mode=SearchMode.LIGHT, text="EDG能去吗", images=(), timeout_seconds=7.5
        )
        self.assertIs(SearchMode.LIGHT, plan.mode)
        self.assertEqual(("EDG 上海",), tuple(item.text for item in plan.queries))
        self.assertEqual(7.5, llm.calls[0][1]["timeout_seconds"])

    def test_standard_finds_first_valid_balanced_object_deduplicates_and_caps_three(self):
        llm = FakeLLM('{bad} prefix {"queries":["a"," a ","b","c","d"],"extra":1}')
        plan = QueryPlanner(llm).plan(
            mode=SearchMode.STANDARD, text="question", images=(), timeout_seconds=8
        )
        self.assertIs(SearchMode.STANDARD, plan.mode)
        self.assertEqual(("a", "b", "c"), tuple(item.text for item in plan.queries))

    def test_image_only_planning_is_multimodal(self):
        llm = FakeLLM('{"queries":["图中的相机型号"]}')
        plan = QueryPlanner(llm).plan(
            mode=SearchMode.LIGHT,
            text="",
            images=("data:image/png;base64,AAA",),
            timeout_seconds=8,
        )
        user_content = llm.calls[0][0][-1]["content"]
        self.assertEqual("image_url", user_content[-1]["type"])
        self.assertEqual("图中的相机型号", plan.queries[0].text)

    def test_malformed_text_falls_back_without_changing_mode(self):
        plan = QueryPlanner(FakeLLM("not json")).plan(
            mode=SearchMode.STANDARD, text="  原始   问题 ", images=(), timeout_seconds=8
        )
        self.assertIs(SearchMode.STANDARD, plan.mode)
        self.assertTrue(plan.planner_degraded)
        self.assertEqual("原始 问题", plan.queries[0].text)

    def test_image_only_exception_uses_fixed_fallback_without_changing_mode(self):
        plan = QueryPlanner(FakeLLM(error=TimeoutError())).plan(
            mode=SearchMode.LIGHT,
            text="",
            images=("data:image/png;base64,AAA",),
            timeout_seconds=8,
        )
        self.assertIs(SearchMode.LIGHT, plan.mode)
        self.assertTrue(plan.planner_degraded)
        self.assertEqual(IMAGE_ONLY_FALLBACK_QUERY, plan.queries[0].text)
```

Also assert that `planning.py` contains neither `"Route" + "Planner"` nor greeting/arithmetic/creative routing regexes; concatenate the forbidden identifier in the test so the final repository-wide scan does not match the guard itself.

- [ ] **Step 2: Run the planner tests and verify RED**

Run: `python -B -m unittest tests.test_simple_search_planning -v`

Expected: FAIL with `ImportError: cannot import name 'QueryPlanner'`; current partial code still asks the model to choose `mode`.

- [ ] **Step 3: Implement query-only parsing and multimodal messages**

Expose this exact API:

```python
IMAGE_ONLY_FALLBACK_QUERY = "识别并查找图片中的主体、事件或内容"


class QueryPlanner:
    def __init__(self, llm):
        self._llm = llm

    def plan(
        self,
        *,
        mode: SearchMode,
        text: str,
        images: tuple[str, ...],
        timeout_seconds: float,
    ) -> SearchPlan:
        if mode is SearchMode.SKIP:
            raise ValueError("skip mode must not invoke QueryPlanner")
        fallback = " ".join(text.split()) or IMAGE_ONLY_FALLBACK_QUERY
        try:
            response = self._llm.chat(
                _planner_messages(text, images),
                temperature=0.0,
                max_tokens=256,
                tools=None,
                tool_choice="none",
                timeout_seconds=timeout_seconds,
            )
            queries = _parse_queries(getattr(response, "content", ""))
        except Exception as error:
            logger.debug("query planner failed error_type=%s", type(error).__name__)
            queries = ()
        limit = 1 if mode is SearchMode.LIGHT else 3
        selected = queries[:limit] or (fallback,)
        return SearchPlan(
            mode=mode,
            queries=tuple(SearchQuery(f"q{index}", query) for index, query in enumerate(selected, 1)),
            planner_degraded=not bool(queries),
        )
```

The system prompt must request exactly `{"queries":["concise query"]}` and state one concise query for light or up to three diverse queries for standard without including a mode field. `_planner_messages` must make the final user content a string for text-only input and use the existing OpenAI-compatible list form for any images:

```python
content = [{"type": "text", "text": text or "请根据图片生成联网搜索词。"}]
content.extend({"type": "image_url", "image_url": {"url": image}} for image in images)
```

Reuse a quote-aware balanced-object scanner, continue after malformed objects, ignore unknown keys (including returned `mode`), accept only a list of strings under `queries`, collapse whitespace, deduplicate in first-seen order, drop empties, and cap each query through `SearchQuery` at 500 characters. Delete all model-routing prompts and fallback route heuristics.

- [ ] **Step 4: Run GREEN planner tests**

Run: `python -B -m unittest tests.test_simple_search_planning tests.test_llm_image_fallback -v`

Expected: all selected tests pass; the fake planner call contains image data only in the model request, not logs.

- [ ] **Step 5: Commit the fixed-mode planner**

```bash
git add src/search/simple/planning.py src/search/simple/__init__.py tests/test_simple_search_planning.py
git commit -m "feat: plan fixed mode multimodal queries"
```

---

### Task 3: Build new simple providers, then rewrite retrieval

**Files:**
- Create: `src/search/simple/tavily.py`
- Create: `src/search/simple/ddgs.py`
- Rewrite: `src/search/simple/retrieval.py`
- Modify: `src/search/simple/__init__.py`
- Extend: `tests/test_simple_search_providers.py`
- Rewrite: `tests/test_simple_search_retrieval.py`

The files under `src/search/providers/` and `src/search/models.py` are read-only behavior references. Do not import, adapt, or modify them. The new classes use only Task 1 contracts.

- [ ] **Step 1: Write failing tests for the new Tavily and DDGS classes**

Add these core tests to `tests/test_simple_search_providers.py`:

```python
class SimpleProviderImplementationTests(unittest.TestCase):
    def test_tavily_light_forwards_basic_mode_and_timeout(self):
        provider, client = tavily_provider({"results": []})
        result = provider.search(
            SearchQuery("q1", "EDG 上海冠军赛"),
            mode=SearchMode.LIGHT, max_results=4, timeout_seconds=8,
        )
        self.assertIs(ProviderStatus.EMPTY, result.status)
        client.search.assert_called_once_with(
            "EDG 上海冠军赛", search_depth="basic", max_results=4,
            timeout=8, include_raw_content=False,
        )

    def test_tavily_standard_news_and_filters_use_final_query_fields(self):
        provider, client = tavily_provider({"results": []})
        provider.search(
            SearchQuery(
                "q1", "news", date_from=date(2026, 9, 1),
                date_to=date(2026, 9, 4), include_domains=("example.com",),
                exclude_domains=("bad.example",), news=True,
            ),
            mode=SearchMode.STANDARD, max_results=8, timeout_seconds=7,
        )
        kwargs = client.search.call_args.kwargs
        self.assertEqual("advanced", kwargs["search_depth"])
        self.assertTrue(kwargs["include_raw_content"])
        self.assertEqual("news", kwargs["topic"])
        self.assertEqual("2026-09-01", kwargs["start_date"])
        self.assertEqual("2026-09-04", kwargs["end_date"])

    @patch("src.search.simple.ddgs.DDGS")
    def test_ddgs_uses_smaller_configured_or_call_timeout(self, ddgs):
        ddgs.return_value.__enter__.return_value.text.return_value = []
        provider = DDGSSearchProvider(proxy_url="", timeout_seconds=15)
        provider.search(
            SearchQuery("q1", "q"), mode=SearchMode.LIGHT,
            max_results=4, timeout_seconds=6,
        )
        ddgs.assert_called_once_with(proxy=None, timeout=6)

    def test_new_provider_modules_never_depend_on_legacy_search(self):
        for path in (Path("src/search/simple/tavily.py"), Path("src/search/simple/ddgs.py")):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("src.search." + "models", source)
            self.assertNotIn("src.search." + "providers", source)
```

Also test not-configured/unavailable readiness, timeout/connection/unknown mapping, empty results, hit conversion, equal date-bound normalization, and one bad-date retry that removes both date fields and uses only `timeout_seconds - elapsed`. Every expected value must use the final records from `src.search.simple.providers`.

- [ ] **Step 2: Run provider tests and verify RED**

Run: `python -B -m unittest tests.test_simple_search_providers -v`

Expected: FAIL with `ModuleNotFoundError` for the new simple provider modules.

- [ ] **Step 3: Implement new providers against only simple contracts**

Implement `TavilySearchProvider.__init__(self, *, api_key: str, proxy_url: str) -> None`, `TavilySearchProvider.readiness(self) -> ProviderReadiness`, and `TavilySearchProvider.search(self, query: SearchQuery, *, mode: SearchMode, max_results: int, timeout_seconds: float) -> ProviderResult`; set class attribute `name = "tavily"`. Implement the parallel DDGS signatures with constructor `DDGSSearchProvider.__init__(self, *, proxy_url: str, timeout_seconds: float) -> None` and class attribute `name = "ddgs"`.

The method behavior is exact: Tavily lazily creates `TavilyClient(api_key=..., proxies=...)`; LIGHT sends `basic/False`, STANDARD sends `advanced/True`; `query.news` adds `topic="news"`; optional date/domain keys are sent only when present; equal date bounds are omitted and set `date_filter_normalized=True`; one `BadRequestError` with date bounds retries once without both bounds using `max(0, timeout_seconds - elapsed)`; and exceptions map to closed status/error codes without exception text. DDGS ignores mode, creates `DDGS(proxy=self._proxy_url or None, timeout=min(self._timeout_seconds, max(float(timeout_seconds), 0.001)))`, chooses `cn-zh` for CJK and `us-en` otherwise, and converts only mapping results. Both return Task 1 records and use `time.monotonic()` for non-negative latency.

Run: `python -B -m unittest tests.test_simple_search_providers -v`

Expected: all provider tests pass before retrieval imports these classes.

- [ ] **Step 4: Add failing simple-only retrieval tests**

Define `FakeProvider.search(query, *, mode, max_results, timeout_seconds)` locally with Task 1 records, then add:

```python
class ProviderRunnerTests(unittest.TestCase):
    def trace(self, mode=SearchMode.LIGHT):
        return SearchTrace("r1", RequestSource.CHAT, mode)

    def test_tavily_usable_hit_resolves_query_without_ddgs(self):
        plan = SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"),))
        tavily = FakeProvider("tavily", {"q1": result("success", hit("q1", "https://example.com/a"))})
        ddgs = FakeProvider("ddgs", {})
        output = ProviderRunner((tavily, ddgs), 8, 15, 4).run(plan, self.trace())
        self.assertEqual(("https://example.com/a",), tuple(item.url for item in output))
        self.assertEqual([], ddgs.calls)

    def test_tavily_invalid_urls_fall_back_to_ddgs(self):
        plan = SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"),))
        tavily = FakeProvider("tavily", {"q1": result("success", hit("q1", "http://127.0.0.1/private"))})
        ddgs = FakeProvider("ddgs", {"q1": result("success", hit("q1", "https://example.com/ddgs", provider="ddgs"))})
        output = ProviderRunner((tavily, ddgs), 8, 15, 4).run(plan, self.trace())
        self.assertEqual(["q1"], [call.query_id for call in ddgs.calls])
        self.assertEqual("https://example.com/ddgs", output[0].url)

    def test_noncooperative_calls_are_bounded(self):
        started = time.monotonic()
        output = ProviderRunner(
            (BlockingProvider("tavily"), EmptyProvider("ddgs")), 0.05, 0.05, 4,
        ).run(SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"),)), self.trace())
        self.assertEqual((), output)
        self.assertLess(time.monotonic() - started, 0.3)
```

Also assert DDGS receives only failed/empty/unusable Tavily query IDs, canonical deduplication removes tracking variants, LIGHT caps at 5, STANDARD caps at 8, provider mode/max-results/timeouts are exact, completion order cannot change query order, and safe traces contain statuses/counts but no request, URL, exception, body, or credential data.

- [ ] **Step 5: Run retrieval tests and verify RED**

Run: `python -B -m unittest tests.test_simple_search_retrieval -v`

Expected: FAIL because partial retrieval still depends on legacy provider shapes and accepts all-invalid Tavily hits.

- [ ] **Step 6: Rewrite retrieval with no coexistence adapters**

Keep exactly `ProviderRunner.__init__(self, providers: tuple[SearchProvider, ...], tavily_timeout: float, ddgs_timeout: float, max_results_per_query: int) -> None` and `ProviderRunner.run(self, plan: SearchPlan, trace: SearchTrace) -> tuple[SearchResult, ...]`.

Require provider names `tavily` then `ddgs`. For each batch create `ThreadPoolExecutor(max_workers=len(queries))`, submit `provider.search(query, mode=plan.mode, max_results=self._max_results_per_query, timeout_seconds=provider_timeout)`, and call `wait(futures, timeout=provider_timeout)`. Convert unfinished futures to timeout, cancel them, and always call `executor.shutdown(wait=False, cancel_futures=True)` in `finally`; never use a context manager. Preserve query order.

A Tavily query is resolved only when `_safe_canonical_url` accepts at least one hit. Send failed, timeout, empty, unavailable, not-configured, and all-invalid queries—and only those—to DDGS. Canonicalization calls `canonicalize_public_http_url`, rejects localhost/`.local` and non-global IP literals, strips fragments and `utm_*`, `fbclid`, `gclid`, `dclid`, `msclkid`, `mc_cid`, `mc_eid`, `igshid`, `ref_src`, then canonicalizes again. Convert directly to `SearchResult`, deduplicate first-seen URLs, and cap to 5 LIGHT/8 STANDARD. There are no `_legacy_query`, `_legacy_tier`, provider-shape adapters, or imports outside `src.search.simple` except `src.search.url_policy`.

- [ ] **Step 7: Run GREEN provider/retrieval/URL tests and commit**

```bash
python -B -m unittest \
  tests.test_simple_search_providers \
  tests.test_simple_search_retrieval \
  tests.test_search_url_policy -v
```

Expected: all selected tests pass and architectural guards find no legacy provider/model imports.

```bash
git add src/search/simple/providers.py src/search/simple/tavily.py src/search/simple/ddgs.py src/search/simple/retrieval.py src/search/simple/__init__.py tests/test_simple_search_providers.py tests/test_simple_search_retrieval.py
git commit -m "feat: add clean simple search providers and retrieval"
```

---

### Task 4: Add bounded page reader and tolerant ranker

**Files:**
- Create: `src/search/simple/reader.py`
- Create: `src/search/simple/ranking.py`
- Create: `tests/test_simple_search_reader.py`
- Create: `tests/test_simple_search_ranking.py`

- [ ] **Step 1: Write failing reader tests**

```python
class OnDemandReaderTests(unittest.TestCase):
    def test_reads_only_missing_or_short_snippets_up_to_limit(self):
        results = (make_result("R1", "短"), make_result("R2", "字" * 80), make_result("R3", ""))
        enriched = OnDemandReader(fake_fetch).enrich(
            results, limit=1, timeout_seconds=5,
            trace=SearchTrace("r1", RequestSource.CHAT, SearchMode.LIGHT)
        )
        self.assertEqual([("https://example.com/1", 5)], fake_fetch.calls)
        self.assertEqual("字" * 80, enriched[1].excerpt)

    def test_failed_read_preserves_provider_snippet(self):
        result = make_result("R1", "短摘要")
        output = OnDemandReader(failing_fetch).enrich(
            (result,), limit=1, timeout_seconds=5,
            trace=SearchTrace("r1", RequestSource.CHAT, SearchMode.LIGHT)
        )
        self.assertEqual("短摘要", output[0].excerpt)

    def test_reader_cleans_and_caps_successful_page_text(self):
        output = OnDemandReader(long_fetch).enrich(
            (make_result("R1", ""),), limit=1, timeout_seconds=5,
            trace=SearchTrace("r1", RequestSource.CHAT, SearchMode.LIGHT)
        )
        self.assertEqual(1500, len(output[0].excerpt))
        self.assertNotIn("\x00", output[0].excerpt)

    def test_noncooperative_reader_is_request_bounded(self):
        started = time.monotonic()
        OnDemandReader(blocking_fetch).enrich(
            (make_result("R1", "短"),), limit=1, timeout_seconds=0.05,
            trace=SearchTrace("r1", RequestSource.CHAT, SearchMode.LIGHT)
        )
        self.assertLess(time.monotonic() - started, 0.3)
```

- [ ] **Step 2: Run reader tests and verify RED**

Run: `python -B -m unittest tests.test_simple_search_reader -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.search.simple.reader'`.

- [ ] **Step 3: Implement the request-local reader**

Use `_MIN_SNIPPET_CHARS = 80` and `_MAX_EXCERPT_CHARS = 1500`. Define `OnDemandReader.__init__(self, fetch_document=url_fetch_service.fetch_document)` to store the callable, and define `OnDemandReader.enrich(self, results: tuple[SearchResult, ...], *, limit: int, timeout_seconds: float, trace: SearchTrace) -> tuple[SearchResult, ...]`.

Select only the first `limit` results whose whitespace-compacted excerpt is shorter than 80 characters. Submit selected URLs concurrently; pass `timeout_seconds` to `fetch_document`; wait at most that timeout; cancel unfinished futures; and shut down with `wait=False, cancel_futures=True`. Increment `reader_count` for submitted reads. Replace only when `UrlDocumentResult.ok` is true and cleaned text is non-empty, using `dataclasses.replace`; strip control characters, collapse whitespace, cap at 1500, and preserve original order/snippet on every failure.

- [ ] **Step 4: Write failing ranker tests**

```python
class EvidenceRankerTests(unittest.TestCase):
    def test_known_numeric_scores_sort_stably_and_remove_explicit_zero(self):
        ranked = EvidenceRanker(FakeLLM('{"scores":{"R1":0.2,"R2":0.9,"R3":0}}')).rank(
            "q", three_results(), timeout_seconds=10
        )
        self.assertEqual(("R2", "R1"), tuple(item.result_id for item in ranked.results))
        self.assertFalse(ranked.degraded)

    def test_invalid_or_missing_scores_default_to_half(self):
        ranked = EvidenceRanker(FakeLLM('{"scores":{"R1":"bad","R3":2,"unknown":1}}')).rank(
            "q", three_results(), timeout_seconds=10
        )
        self.assertEqual(1.0, by_id(ranked.results, "R3").score)
        self.assertEqual(0.5, by_id(ranked.results, "R1").score)
        self.assertEqual(0.5, by_id(ranked.results, "R2").score)

    def test_exception_preserves_provider_order_and_degrades(self):
        ranked = EvidenceRanker(FakeLLM(error=TimeoutError())).rank("q", three_results(), timeout_seconds=10)
        self.assertEqual(("R1", "R2", "R3"), tuple(item.result_id for item in ranked.results))
        self.assertTrue(ranked.degraded)

    def test_prompt_excludes_urls_and_timeout_is_forwarded(self):
        llm = FakeLLM('{"scores":{"R1":1}}')
        EvidenceRanker(llm).rank("q", three_results(), timeout_seconds=9.5)
        self.assertNotIn("https://", repr(llm.calls[0][0]))
        self.assertEqual(9.5, llm.calls[0][1]["timeout_seconds"])
```

- [ ] **Step 5: Run ranker tests and verify RED**

Run: `python -B -m unittest tests.test_simple_search_ranking -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.search.simple.ranking'`.

- [ ] **Step 6: Implement one-call stable ranking**

Expose:

```python
@dataclass(frozen=True)
class RankingResult:
    results: tuple[SearchResult, ...]
    degraded: bool


class EvidenceRanker:
    def __init__(self, llm):
        self._llm = llm
```

Define `EvidenceRanker.rank(self, question: str, results: tuple[SearchResult, ...], *, timeout_seconds: float) -> RankingResult`.

Call `llm.chat` once with result ID/title/excerpt only, `temperature=0.0`, `max_tokens=512`, `tools=None`, `tool_choice="none"`, and the supplied timeout. Parse the first valid balanced object whose `scores` is a mapping. Clamp finite known-ID numeric scores to `[0,1]`; unknown IDs are ignored; invalid/missing known scores become `0.5`; explicit zero removes that result. If no known ID has a valid numeric score or the call raises, return the original tuple unchanged with `degraded=True`. Otherwise use Python's stable descending sort and `degraded=False`.

- [ ] **Step 7: Run GREEN reader/ranker tests and commit**

Run: `python -B -m unittest tests.test_simple_search_reader tests.test_simple_search_ranking tests.test_search_url_policy -v`

Expected: all selected tests pass.

```bash
git add src/search/simple/reader.py src/search/simple/ranking.py tests/test_simple_search_reader.py tests/test_simple_search_ranking.py
git commit -m "feat: read and rank simple search evidence"
```

---

### Task 5: Compose the fixed-mode pipeline and production factory

**Files:**
- Create: `src/search/simple/pipeline.py`
- Create: `src/search/simple/factory.py`
- Modify: `src/search/simple/__init__.py`
- Create: `tests/test_simple_search_pipeline.py`

- [ ] **Step 1: Write failing orchestration tests**

```python
class SimpleSearchPipelineTests(unittest.TestCase):
    def test_skip_returns_before_planner_and_all_search_dependencies(self):
        pipeline, calls = make_pipeline()
        outcome = pipeline.run(SearchRequest(SearchMode.SKIP, "你好"))
        self.assertIs(SearchMode.SKIP, outcome.plan.mode)
        self.assertEqual([], calls)

    def test_light_mode_is_passed_unchanged_and_reads_one(self):
        pipeline, calls = make_pipeline(results=short_results(3))
        outcome = pipeline.run(SearchRequest(SearchMode.LIGHT, "q", ("data:image/png;base64,AAA",)))
        self.assertEqual(("planner:light:1", "retriever:light", "reader:1", "ranker"), tuple(calls))
        self.assertIs(SearchMode.LIGHT, outcome.plan.mode)

    def test_standard_reads_two_and_never_accepts_planner_mode_change(self):
        pipeline, calls = make_pipeline(planner_result=SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "q"),)))
        outcome = pipeline.run(SearchRequest(SearchMode.STANDARD, "q"))
        self.assertIs(SearchMode.STANDARD, outcome.plan.mode)
        self.assertEqual(SearchFailure.PROVIDER_UNAVAILABLE, outcome.failure)

    def test_ranker_degradation_preserves_results_and_adds_warning(self):
        pipeline, _ = make_pipeline(ranker_degraded=True)
        outcome = pipeline.run(SearchRequest(SearchMode.LIGHT, "q"))
        self.assertTrue(outcome.results)
        self.assertTrue(outcome.trace.ranker_degraded)
        self.assertEqual("信息可能不完整。", outcome.warning)

    def test_no_usable_result_and_unexpected_exception_are_nonthrowing(self):
        empty, _ = make_pipeline(results=())
        broken, _ = make_pipeline(planner_error=RuntimeError("private"))
        self.assertEqual(SearchFailure.NO_RESULTS, empty.run(SearchRequest(SearchMode.LIGHT, "q")).failure)
        self.assertEqual(SearchFailure.PROVIDER_UNAVAILABLE, broken.run(SearchRequest(SearchMode.LIGHT, "q")).failure)
```

Assert planner receives exact `request.mode`, `request.text`, and `request.images`, timeout fields are selected from `SearchTimeouts`, trace latency values are non-negative, and no safe trace includes request content.

- [ ] **Step 2: Run pipeline tests and verify RED**

Run: `python -B -m unittest tests.test_simple_search_pipeline -v`

Expected: FAIL because pipeline/factory modules do not exist.

- [ ] **Step 3: Implement pipeline mode guards and failure records**

Use these public signatures:

```python
@dataclass(frozen=True)
class SearchTimeouts:
    planner: float
    tavily: float
    ddgs: float
    reader: float
    ranker: float
    answer: float


```

Define `SimpleSearchPipeline.__init__(self, planner, retriever, reader, ranker, *, timeouts: SearchTimeouts, clock=time.monotonic)` and `SimpleSearchPipeline.run(self, request: SearchRequest) -> SearchOutcome`.

`run()` must create a safe trace with `request.source` and `request.mode`; immediately return `SearchPlan(SKIP, ())` for SKIP only when called directly in unit tests (production `/skip` never obtains or invokes the pipeline); call `QueryPlanner.plan(mode=request.mode, text=request.text, images=request.images, timeout_seconds=timeouts.planner)` otherwise; reject any returned plan whose mode differs from the request; retrieve; read 1 result for LIGHT or 2 for STANDARD; discard results only if URL is empty or both title/excerpt are empty; rank once; and return warning `信息可能不完整。` only for ranker degradation. Empty usable/ranked results return `NO_RESULTS`. Any uncaught exception logs only `error_type`, preserves request mode in a degraded fallback plan using normalized text or `IMAGE_ONLY_FALLBACK_QUERY`, and returns `PROVIDER_UNAVAILABLE`.

`factory.py` must provide `get_simple_search_pipeline() -> SimpleSearchPipeline`, `reset_simple_search_pipeline() -> None`, and `get_search_readiness() -> tuple[ProviderReadiness, ...]`.

Import `ProviderReadiness` from `src.search.simple.providers`, `TavilySearchProvider` from `src.search.simple.tavily`, and `DDGSSearchProvider` from `src.search.simple.ddgs`. Construct `QueryPlanner(get_llm_client())`, the new simple Tavily then DDGS providers, `ProviderRunner`, `OnDemandReader`, and `EvidenceRanker`; populate every `SearchTimeouts` member from `config`; cache one pipeline; and clear only this cache in reset. Factory readiness must read provider readiness without constructing the legacy orchestrator.

- [ ] **Step 4: Run GREEN pipeline tests**

Run: `python -B -m unittest tests.test_simple_search_pipeline tests.test_simple_search_models -v`

Expected: all selected tests pass.

- [ ] **Step 5: Commit pipeline/factory**

```bash
git add src/search/simple/pipeline.py src/search/simple/factory.py src/search/simple/__init__.py tests/test_simple_search_pipeline.py
git commit -m "feat: compose fixed mode search pipeline"
```

---

### Task 6: Add multimodal search answering and deterministic rendering

**Files:**
- Create: `src/search/simple/answering.py`
- Create: `src/search/simple/rendering.py`
- Modify: `src/chat/prompt.py`
- Modify: `src/search/simple/__init__.py`
- Create: `tests/test_simple_search_answering.py`
- Create: `tests/test_simple_search_rendering.py`

- [ ] **Step 1: Write failing answer/render tests**

```python
class SearchAnsweringTests(unittest.TestCase):
    def test_answer_keeps_base_history_and_image_content_but_excludes_urls(self):
        llm = FakeLLM("这是某品牌相机，目前约有三款相关型号。")
        base = [{"role": "user", "content": [
            {"type": "text", "text": "这是什么"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]}]
        answer = SearchAnswerer(llm).answer(
            base_messages=base, question="这是什么", results=results(), timeout_seconds=20
        )
        self.assertFalse(answer.degraded)
        self.assertIn("image_url", repr(llm.calls[0][0]))
        self.assertNotIn("https://example.com", repr(llm.calls[0][0]))

    def test_answer_exception_returns_deterministic_ranked_summary(self):
        answer = SearchAnswerer(FakeLLM(error=TimeoutError())).answer(
            base_messages=[], question="q", results=results(), timeout_seconds=20
        )
        self.assertTrue(answer.degraded)
        self.assertEqual("根据搜索结果：\n1. 标题：摘要", answer.text)

    def test_light_render_hides_sources_and_standard_command_shows_three(self):
        light = render_search_answer("回答", four_results(), warning=None, show_sources=False, qq_limit=1700, trace=trace())
        standard = render_search_answer("回答", four_results(), warning=None, show_sources=True, qq_limit=1700, trace=trace())
        self.assertNotIn("https://", light.text)
        self.assertEqual(3, standard.text.count("https://"))
        self.assertEqual(3, len(standard.sources))

    def test_warning_once_and_output_never_exceeds_qq_limit(self):
        response = render_search_answer(
            "回答" * 1000, results(), warning="信息可能不完整。",
            show_sources=True, qq_limit=200, trace=trace(),
        )
        self.assertLessEqual(len(response.text), 200)
        self.assertEqual(1, response.text.count("信息可能不完整。"))
```

Also test empty model content falls back, answer timeout is forwarded, URLs in model text are stripped rather than displayed, failure rendering produces exact non-empty messages, and trace output kind/degradation are updated.

Create `tests/test_simple_search_rendering.py` with the replacement QQ splitter contract before `src/main.py` changes imports:

```python
class QQReplySplittingTests(unittest.TestCase):
    def test_empty_input_returns_no_parts(self):
        self.assertEqual([], split_qq_reply("", 200))

    def test_prefers_newline_and_preserves_every_character(self):
        text = "第一段\n第二段很长\n第三段"
        parts = split_qq_reply(text, 8)
        self.assertTrue(all(1 <= len(part) <= 8 for part in parts))
        self.assertEqual(text, "".join(parts))

    def test_hard_splits_a_single_long_line(self):
        self.assertEqual(["abcd", "efgh", "ij"], split_qq_reply("abcdefghij", 4))

    def test_rejects_nonpositive_limit(self):
        with self.assertRaisesRegex(ValueError, "max_chars"):
            split_qq_reply("text", 0)
```

Import exactly `split_qq_reply(text: str, max_chars: int) -> list[str]` from `src.search.simple.rendering`. The function preserves the input exactly when parts are joined, prefers splitting immediately after the last newline within the bound, hard-splits when no newline exists, emits no empty part, and raises `ValueError("max_chars must be positive")` for `max_chars <= 0`.

- [ ] **Step 2: Run answer tests and verify RED**

Run: `python -B -m unittest tests.test_simple_search_answering -v`

Expected: FAIL because answering/rendering modules do not exist.

- [ ] **Step 3: Add the natural search-answer prompt and answerer**

Add `build_search_system_prompt(mem_ctx)` to `src/chat/prompt.py`, retaining persona/privacy/prompt-injection boundaries and including this exact instruction:

```text
Use only the supplied search titles and excerpts for externally verifiable facts.
Answer naturally in Simplified Chinese. If the excerpts do not settle a detail,
say that it is uncertain. Do not output or invent URLs, source IDs, JSON, or an
internal verification status.
```

Define frozen `AnswerResult(text: str, degraded: bool)`. Define `SearchAnswerer.__init__(self, llm)` to store the LLM and `SearchAnswerer.answer(self, *, base_messages: list[dict[str, object]], question: str, results: tuple[SearchResult, ...], timeout_seconds: float) -> AnswerResult`.

Copy base messages, replace/add the search system prompt at index zero, and append one user message containing JSON with `question` and records shaped as `{"title": result.title, "excerpt": result.excerpt[:1500]}`; never include result IDs or URLs. Call the LLM at `temperature=0.2` with the supplied timeout. Collapse control/whitespace in returned text and remove any `http://` or `https://` token before accepting it. On exception or empty cleaned text, return exactly `根据搜索结果：` followed by up to five numbered `title：excerpt` lines in ranked order, each compacted and bounded.

- [ ] **Step 4: Implement rendering and fixed failure text**

Define `split_qq_reply(text: str, max_chars: int) -> list[str]`, `render_search_answer(text: str, results: tuple[SearchResult, ...], *, warning: str | None, show_sources: bool, qq_limit: int, trace: SearchTrace) -> SearchResponse`, and `render_search_failure(failure: SearchFailure, *, qq_limit: int, trace: SearchTrace) -> SearchResponse`. Implement the splitter exactly as tested in Step 1; it is the non-search QQ transport utility that replaces the legacy renderer import.

Use `在线搜索暂时不可用，请稍后再试。` for `PROVIDER_UNAVAILABLE` and `没有找到可用的在线搜索结果。` for `NO_RESULTS`. Append a warning once. If `show_sources`, append `来源：` and at most three unique canonical `title + URL` lines; otherwise expose no source URLs and return `sources=()`. Reserve length for warning/sources before truncating the answer so the joined response is non-empty and `<= qq_limit`. Set output kind to model answer, summary fallback, or search failure; callers set `answer_degraded` from `AnswerResult`.

- [ ] **Step 5: Run GREEN answer/render tests**

Run: `python -B -m unittest tests.test_simple_search_answering tests.test_simple_search_rendering tests.test_multimodal_chat -v`

Expected: answer/render tests pass; existing multimodal helper tests remain green.

- [ ] **Step 6: Commit answer/render**

```bash
git add src/search/simple/answering.py src/search/simple/rendering.py src/chat/prompt.py src/search/simple/__init__.py tests/test_simple_search_answering.py tests/test_simple_search_rendering.py
git commit -m "feat: answer and render simple search results"
```

---

### Task 7: Cut chat over to explicit modes and make `SKIP` search-free

**Files:**
- Rewrite search-related portions: `src/chat/chat_service.py`
- Modify: `tests/test_multimodal_chat.py`
- Modify: `tests/test_identity_configuration.py`
- Create: `tests/test_simple_search_chat_flow.py`

- [ ] **Step 1: Write failing chat contract and black-box tests**

```python
class SimpleSearchChatFlowTests(unittest.TestCase):
    def test_generate_reply_requires_explicit_mode(self):
        with self.assertRaises(TypeError):
            generate_reply("private:1", "你好")

    def test_light_constructs_light_request_and_hides_sources(self):
        engine = FakeEngine(success_outcome(SearchMode.LIGHT))
        reply = run_reply(engine, mode=SearchMode.LIGHT, text="最新消息")
        self.assertIs(SearchMode.LIGHT, engine.requests[0].mode)
        self.assertNotIn("https://", reply)

    def test_standard_constructs_standard_request_and_shows_sources(self):
        engine = FakeEngine(success_outcome(SearchMode.STANDARD))
        reply = run_reply(engine, mode=SearchMode.STANDARD, text="最新消息")
        self.assertIs(SearchMode.STANDARD, engine.requests[0].mode)
        self.assertIn("https://example.com", reply)

    def test_skip_never_gets_pipeline_or_search_answerer(self):
        with patch("src.chat.chat_service.get_simple_search_pipeline") as factory, patch(
            "src.chat.chat_service.SearchAnswerer"
        ) as answerer:
            reply = generate_reply(
                "private:1", "看图回答", ["data:image/png;base64,AAA"], mode=SearchMode.SKIP
            )
        factory.assert_not_called()
        answerer.assert_not_called()
        self.assertEqual("plain multimodal reply", reply)

    def test_image_reaches_light_planner_and_search_answer(self):
        engine = FakeEngine(success_outcome(SearchMode.LIGHT))
        run_reply(engine, mode=SearchMode.LIGHT, text="", images=["data:image/png;base64,AAA"])
        self.assertEqual(("data:image/png;base64,AAA",), engine.requests[0].images)
        self.assertIn("image_url", repr(fake_llm.calls[-1][0]))

    def test_unexpected_dispatch_error_returns_fixed_reply_and_saves_history(self):
        reply = run_reply(RaisingEngine(), mode=SearchMode.LIGHT, text="q")
        self.assertEqual("在线搜索暂时不可用，请稍后再试。", reply)
        self.assertEqual(reply, chat_history["private:1"][-1]["content"])
```

Add assertions that `force_search` is rejected, plain/search calls forward `config.search_answer_timeout`, all paths append exactly one user/assistant pair, and history stores image placeholders rather than image bytes.

- [ ] **Step 2: Run chat tests and verify RED**

Run: `python -B -m unittest tests.test_simple_search_chat_flow -v`

Expected: FAIL because current `generate_reply` defaults to model-routed behavior and still constructs the legacy orchestrator for every message.

- [ ] **Step 3: Replace the chat entry contract and dispatch**

Use the exact signature `generate_reply(context: MemoryContext | str, text: str, image_data_urls: list[str] | None = None, *, mode: SearchMode, history_text: str | None = None) -> str`.

Normalize context/text/images first. For `SKIP`, call `_plain_reply(mem_ctx, text, images, timeout_seconds=config.search_answer_timeout)` before any factory or search-answerer access. For LIGHT/STANDARD, construct `SearchRequest(mode=mode, text=text, images=tuple(images), source=CHAT if LIGHT else COMMAND)`, run the cached simple pipeline once, render fixed failure if needed, otherwise build history-aware multimodal base messages, invoke `SearchAnswerer`, set `trace.answer_degraded`, combine ranker/answer warnings, and call `render_search_answer(show_sources=mode is STANDARD)`.

Retain `normalize_chat_response`, tool-protocol helpers used by provider-client tests, history loading/saving, `build_user_content`, and `history_user_text`. Delete legacy evidence payload, decision/claim/validation/render state functions and imports. Catch the complete dispatch boundary, log only `error_type`, return a fixed non-empty reply, and append history in `finally` only after reply assignment. Use `history_text` verbatim when supplied; otherwise use `history_user_text(text, len(images))`.

Expose/reset a chat-local cache only through `get_simple_search_pipeline_for_chat()` and `reset_chat_search_pipeline() -> None`; the getter lazily stores `get_simple_search_pipeline()`, and reset clears the chat-local value and calls `reset_simple_search_pipeline()`.

- [ ] **Step 4: Update adjacent multimodal/identity tests and run GREEN**

Change their calls to pass `mode=SearchMode.SKIP` when testing plain chat and use simple `SearchOutcome` fakes when testing search. Run:

```bash
python -B -m unittest \
  tests.test_simple_search_chat_flow \
  tests.test_multimodal_chat \
  tests.test_identity_configuration \
  tests.test_deepseek_tool_context \
  tests.test_llm_tool_affinity -v
```

Expected: all selected tests pass; no selected test patches `get_search_orchestrator`.

- [ ] **Step 5: Keep the caller cutover uncommitted and proceed directly to Task 8**

Do not commit the required `mode` signature while `src/main.py` and command handlers still use the old call shape. Task 8 updates every production caller and affected test, reruns the full suite, and commits both tasks atomically. This avoids a knowingly broken intermediate commit.

---

### Task 8: Add `/skip`, command images, normal `LIGHT`, and command history

**Files:**
- Create: `src/commands/skip.py`
- Modify: `src/commands/search.py`
- Modify: `src/commands/__init__.py`
- Modify: `src/commands/help.py`
- Modify: `src/main.py`
- Modify: `tests/test_command_renderer.py`
- Modify: `tests/test_main_image_flow.py`
- Modify: `tests/test_product_scope.py`
- Modify: `tests/test_user_facing_scope.py`

- [ ] **Step 1: Add failing command-mode tests**

Add:

```python
class DeterministicCommandTests(unittest.TestCase):
    def test_search_passes_standard_mode_images_and_original_history(self):
        context = command_context("/search 看看", images=(IMAGE_DATA_URL,))
        with patch("src.commands.search.generate_reply", return_value="answer") as generate:
            result = handle_command(route_message("/search 看看"), context, renderer=identity_renderer())
        generate.assert_called_once_with(
            context.memory_context, "看看", [IMAGE_DATA_URL],
            mode=SearchMode.STANDARD, history_text="/search 看看",
        )
        self.assertEqual("answer", result.reply)

    def test_skip_empty_without_images_returns_usage_and_calls_no_chat(self):
        context = command_context("/skip")
        with patch("src.commands.skip.generate_reply") as generate:
            result = handle_command(route_message("/skip"), context, renderer=identity_renderer())
        self.assertEqual("用法：/skip <内容>，也可以附带图片。", result.reply)
        generate.assert_not_called()

    def test_skip_image_only_calls_plain_multimodal_mode(self):
        context = command_context("/skip", images=(IMAGE_DATA_URL,))
        with patch("src.commands.skip.generate_reply", return_value="看到了") as generate:
            result = handle_command(route_message("/skip"), context, renderer=identity_renderer())
        generate.assert_called_once_with(
            context.memory_context, "", [IMAGE_DATA_URL],
            mode=SearchMode.SKIP, history_text="/skip",
        )
        self.assertEqual("看到了", result.reply)

    def test_help_and_unknown_output_list_skip(self):
        self.assertIn("/skip", help_text())
        self.assertIn("/skip", handle_command(route_message("/unknown"), command_context("/unknown")).reply)
```

Extend `CommandContext` fixture construction with `image_data_urls: tuple[str, ...] = ()`.

- [ ] **Step 2: Run command tests and verify RED**

Run: `python -B -m unittest tests.test_command_renderer -v`

Expected: FAIL because `/skip` is not registered and command context carries no downloaded images.

- [ ] **Step 3: Implement command handlers with explicit modes**

`src/commands/search.py`:

```python
def search_reply(query: str, context: CommandContext) -> str:
    normalized = normalize_search_query(query)
    if not normalized and not context.image_data_urls:
        return "想搜什么？比如：/search DeepSeek 最新消息，也可以附带图片。"
    return generate_reply(
        context.memory_context,
        normalized,
        list(context.image_data_urls),
        mode=SearchMode.STANDARD,
        history_text=context.raw_message,
    )
```

`src/commands/skip.py`:

```python
def skip_reply(query: str, context: CommandContext) -> str:
    text = " ".join(str(query or "").split())
    if not text and not context.image_data_urls:
        return "用法：/skip <内容>，也可以附带图片。"
    return generate_reply(
        context.memory_context,
        text,
        list(context.image_data_urls),
        mode=SearchMode.SKIP,
        history_text=context.raw_message,
    )
```

Add `image_data_urls: tuple[str, ...] = ()` to frozen `CommandContext`; make `_search_command` pass the full context; add `_skip_command`; register only canonical `"skip"`; and list `/skip` in help and unknown-command text. The generic `route_message` remains syntax-only and unchanged.

- [ ] **Step 4: Add failing main-dispatch image tests**

Replace tests that expect command images to be discarded with:

```python
class MainDeterministicModeTests(unittest.TestCase):
    def test_ordinary_text_image_and_mixed_messages_are_light(self):
        for text, images in (("hello", []), ("", [IMAGE_DATA_URL]), ("hello", [IMAGE_DATA_URL])):
            with self.subTest(text=text, images=images):
                process_with_loaded_images(text, images)
                generate_reply.assert_called_once_with(
                    "private:1", text, image_data_urls=images, mode=SearchMode.LIGHT
                )

    def test_search_command_downloads_and_passes_images(self):
        process_command_event("/search 看看", image_url=IMAGE_URL)
        load_chat_images.assert_called_once()
        self.assertEqual((IMAGE_DATA_URL,), handle_command.call_args.args[1].image_data_urls)

    def test_skip_command_downloads_file_only_image(self):
        process_command_event("/skip", image_file_id="opaque-file-id")
        onebot.get_image_url.assert_called_once_with("opaque-file-id")
        self.assertEqual((IMAGE_DATA_URL,), handle_command.call_args.args[1].image_data_urls)
```

Also assert command image load failures return the existing user-facing `ImageInputError`, normal and command logs omit temporary URL/data bytes, and command images are cleared from the local list after handler return.

- [ ] **Step 5: Run main image tests and verify RED**

Run: `python -B -m unittest tests.test_main_image_flow -v`

Expected: FAIL because `src/main.py` currently handles commands before `load_chat_images` and calls ordinary chat without an explicit mode.

- [ ] **Step 6: Move image download ahead of both dispatch branches**

After mention/reply stripping and `parse_image_message`, call `load_chat_images(...)` once before the command/chat branch. Build `CommandContext(..., image_data_urls=tuple(image_data_urls))` for commands. For ordinary chat call:

```python
reply = generate_reply(
    mem_ctx,
    parsed_message.text,
    image_data_urls=image_data_urls,
    mode=SearchMode.LIGHT,
)
```

Stage/release memory only for ordinary chat as today; command dispatch must still clear the local image-data list in a `finally` block. Keep routing/logging based on image count and visible text, never URL/data content. Update `_search_readiness()` to call `get_search_readiness()` from the simple factory. Update `split_reply()` to use `from src.search.simple.rendering import split_qq_reply`; Task 6 has already defined and tested the exact replacement before this import changes.

- [ ] **Step 7: Run command/main/product GREEN tests**

```bash
python -B -m unittest \
  tests.test_command_renderer \
  tests.test_simple_search_rendering \
  tests.test_main_image_flow \
  tests.test_product_scope \
  tests.test_user_facing_scope \
  tests.test_image_input_service -v
```

Expected: all selected tests pass; ordinary text/image/mixed events forward their images with LIGHT, `/search` forwards its images with STANDARD, and `/skip` forwards its images to plain SKIP generation with zero search dependencies.

Do not run the full discovery suite at this point: obsolete legacy tests still exercise the old chat/orchestrator contract and are intentionally removed atomically in Task 9. The maintained caller-focused suite above must be green; Task 9 is the first valid full-suite boundary after legacy deletion.

- [ ] **Step 8: Commit the chat and entry-point cutover atomically**

```bash
git add src/chat/chat_service.py src/commands/skip.py src/commands/search.py src/commands/__init__.py src/commands/help.py src/main.py tests/test_simple_search_chat_flow.py tests/test_multimodal_chat.py tests/test_identity_configuration.py tests/test_command_renderer.py tests/test_main_image_flow.py tests/test_product_scope.py tests/test_user_facing_scope.py
git commit -m "feat: cut over deterministic search and skip entry points"
```

---

### Task 9: Cut over compatibility exports and delete all legacy runtime atomically

**Files:**
- Modify: `src/services/search_service.py`
- Rewrite: `src/search/__init__.py`
- Modify: `tests/test_simple_search_models.py`
- Create: `tests/test_search_service.py`
- Delete: all legacy production/tests/tools listed in the file map, including the complete `src/search/providers/` package

The new providers have already run in production composition since Task 5. This task does not migrate or modify legacy providers; it removes those read-only references together with every remaining legacy import/test so the full suite never observes a half-deleted tree.

- [ ] **Step 1: Write failing compatibility and removal tests**

Create `tests/test_search_service.py` with:

```python
class SearchServiceCompatibilityTests(unittest.TestCase):
    @patch("src.services.search_service.get_simple_search_pipeline")
    def test_compatibility_search_is_always_standard(self, factory):
        factory.return_value.run.return_value = standard_outcome()
        result = search("  current   news ")
        request = factory.return_value.run.call_args.args[0]
        self.assertIs(SearchMode.STANDARD, request.mode)
        self.assertIs(RequestSource.COMPATIBILITY, request.source)
        self.assertEqual("current news", request.text)
        self.assertEqual((), request.images)
        self.assertTrue(result.ok)

    @patch("src.services.search_service.reset_simple_search_pipeline")
    def test_reset_delegates_only_to_simple_factory(self, reset):
        reset_search_service()
        reset.assert_called_once_with()
```

Add exact failure mapping assertions (`ok=False`, `status=failure.value`, `text="在线检索未完成。"`) and successful ranked title/excerpt/URL flattening. Add removal assertions to `tests/test_simple_search_models.py`:

```python
class LegacyRemovalTests(unittest.TestCase):
    def test_legacy_runtime_paths_are_absent(self):
        for relative in LEGACY_RUNTIME_PATHS:
            self.assertFalse(Path(relative).exists(), relative)

    def test_live_tree_has_no_legacy_imports_or_symbols(self):
        roots = tuple(Path(root) for root in ("src", "tests", "tools"))
        this_test = Path(__file__).resolve()
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for root in roots for path in root.rglob("*.py")
            if path.resolve() != this_test
        )
        for forbidden in LEGACY_IMPORTS + LEGACY_SYMBOLS:
            self.assertNotIn(forbidden, source)
```

Set `LEGACY_RUNTIME_PATHS` to every production/test/tool path in the file-map deletion lists. Build `LEGACY_IMPORTS` and `LEGACY_SYMBOLS` from concatenated fragments (for example, `"src.search." + "providers"` and `"Route" + "Planner"`) so the guard file itself does not match the later `rg` scan. Cover all deleted `src.search.*` prefixes plus `RoutePlanner`, `force_search`, `has_images`, `SearchTier`, `GroundedDraft`, `ClaimDiscovery`, `SemanticVerifier`, `RepairPlan`, and `fail_closed`.

- [ ] **Step 2: Run cutover tests and verify RED**

Run: `python -B -m unittest tests.test_search_service tests.test_simple_search_models -v`

Expected: FAIL because the compatibility facade still uses the legacy orchestrator and the listed legacy files still exist.

- [ ] **Step 3: Replace the compatibility facade and package exports**

`src/services/search_service.py` must retain its public response type and normalization, but call only:

```python
request = SearchRequest(
    mode=SearchMode.STANDARD,
    text=normalized,
    images=(),
    source=RequestSource.COMPATIBILITY,
)
outcome = get_simple_search_pipeline().run(request)
```

Map `outcome.failure` to `ok=False/status=failure.value/text="在线检索未完成。"`; otherwise flatten ranked title/excerpt/URL records. Its reset function delegates to `reset_simple_search_pipeline()`.

Rewrite `src/search/__init__.py` to export only simple request/plan/result/outcome/trace/response/provider records and factory functions plus `UrlDecision`, `canonicalize_public_http_url`, and `evaluate_public_http_url`. Do not export compatibility aliases for legacy tiers, routes, evidence, claims, validation, repair, budget, stage runners, registries, or providers.

- [ ] **Step 4: Delete legacy runtime and update all remaining tests/imports in the same working-tree change**

Delete every path under “Atomic legacy deletion after cutover,” including `src/search/providers/__init__.py`, `base.py`, `tavily.py`, and `ddgs.py`. Before running any suite, update any remaining maintained import to the final `src.search.simple` API and delete obsolete legacy tests/tools in that same change. Do not add shims, aliases, adapters, or temporary modules.

Run:

```bash
rg -n "src\.search\.(providers|models|orchestrator|planner|router|evidence|validation|policy|renderer|outcomes|budget|stage_runner|extraction)" src tests tools
rg -n "RoutePlanner|force_search|has_images|SearchTier|GroundedDraft|ClaimDiscovery|SemanticVerifier|RepairPlan|fail_closed|freshness|risk_policy" src tests tools
```

Expected: both commands exit 1 with no matches. Fix every live import/test before proceeding.

- [ ] **Step 5: Run focused cutover tests, then the complete maintained suite**

```bash
python -B -m unittest \
  tests.test_search_service \
  tests.test_simple_search_models \
  tests.test_simple_search_providers \
  tests.test_simple_search_retrieval \
  tests.test_simple_search_pipeline \
  tests.test_simple_search_chat_flow -v
python -B -m unittest discover -s tests -t . -q
```

Expected: both commands exit 0. Provider behavior remains covered by `tests.test_simple_search_providers`; no test imports the deleted package.

- [ ] **Step 6: Commit compatibility cutover and deletion atomically**

```bash
git add -u src tests tools
git add src/services/search_service.py src/search/__init__.py tests/test_search_service.py tests/test_simple_search_models.py
git commit -m "refactor: delete legacy search runtime"
```

---

### Task 10: Replace evaluator and document deterministic behavior

**Files:**
- Rewrite: `tools/evaluate_search.py`
- Rewrite: `tests/test_search_evaluation.py`
- Modify: `tests/test_readme_guide.py`
- Modify: `README.md`
- Modify: `.env.example`
- Rewrite: `eval/search/README.md`

- [ ] **Step 1: Write failing evaluator tests**

```python
class DeterministicSearchEvaluationTests(unittest.TestCase):
    def test_entry_point_mode_invariants(self):
        report = evaluate_rows([
            row(source="chat", mode="standard", query_count=1),
            row(source="command", mode="light", query_count=1),
            row(source="compatibility", mode="light", query_count=1),
        ])
        self.assertEqual(1, report["violations"]["chat_not_light"])
        self.assertEqual(2, report["violations"]["standard_source_not_standard"])

    def test_mode_query_caps(self):
        report = evaluate_rows([
            row(source="chat", mode="light", query_count=2),
            row(source="command", mode="standard", query_count=4),
        ])
        self.assertEqual(2, report["violations"]["query_cap_exceeded"])

    def test_trace_rejects_sensitive_or_open_ended_fields(self):
        report = evaluate_rows([dict(row(), query="secret", url="https://example.com", exception="body")])
        self.assertEqual(1, report["violations"]["unsafe_trace_record"])
```

Test provider success rate, planner/ranker/answer degradation rates, output-kind counts, malformed JSONL handling, and CLI `traces` output. Do not construct a `/skip` trace fixture: command/chat tests in Tasks 7–8 prove that `/skip` obtains no pipeline and therefore creates no `SearchTrace`.

- [ ] **Step 2: Run evaluator tests and verify RED**

Run: `python -B -m unittest tests.test_search_evaluation -v`

Expected: FAIL because the current evaluator reports legacy evidence/claim metrics.

- [ ] **Step 3: Implement deterministic trace evaluation**

Expose `evaluate_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, object]`. Accept only `request_id`, `source`, `mode`, `query_count`, `provider_statuses`, `candidate_count`, `reader_count`, `planner_degraded`, `ranker_degraded`, `answer_degraded`, `output_kind`, and `stage_latency_ms`, exactly matching `SearchTrace.to_safe_dict()`. Validate only trace-producing entry points: `source="chat"` requires LIGHT, while `source="command"` and `source="compatibility"` require STANDARD. Treat any other source or any SKIP trace as malformed rather than defining a `/skip` metric. Count query-cap violations for LIGHT (`==1`) and STANDARD (`1..3`), malformed closed statuses, unsafe fields, provider success, degradation rates, and output kinds.

The CLI must expose exactly:

```text
python tools/evaluate_search.py traces path/to/traces.jsonl
python tools/evaluate_search.py smoke
```

`traces` prints aggregate JSON only. `smoke` requires `QQBOT_ALLOW_LIVE_SEARCH_SMOKE=1`, runs one `SearchRequest(..., source=RequestSource.CHAT, mode=LIGHT)` and one `SearchRequest(..., source=RequestSource.COMMAND, mode=STANDARD)` through production, and prints only each safe trace; without authorization it exits nonzero with `Set QQBOT_ALLOW_LIVE_SEARCH_SMOKE=1 to authorize live provider calls.` It must not print query text, URLs, bodies, image bytes, credentials, or claim quality certification.

- [ ] **Step 4: Update user/operator documentation and environment template**

README/help documentation must state:

- ordinary text, image-only, and mixed chat always select LIGHT and exactly one query;
- `/search` text, image-only, and mixed input always selects STANDARD, with one to three queries and up to three displayed sources;
- `/skip` is the only user route to SKIP, forwards text and images to plain multimodal generation, invokes no search dependency, and creates no search trace;
- `/skip` without text/images returns `用法：/skip <内容>，也可以附带图片。`;
- ordinary replies contain no source URLs;
- Tavily is attempted first per query and DDGS receives only unresolved/empty/invalid-URL queries;
- planner/read/rank/answer failures degrade to the exact bounded behavior in the design;
- safe traces omit request/evidence data;
- `SEARCH_PLANNER_TIMEOUT=8`, `SEARCH_TAVILY_TIMEOUT=8`, `SEARCH_DDGS_TIMEOUT=15`, `SEARCH_READER_TIMEOUT=5`, `SEARCH_RANKER_TIMEOUT=10`, and `SEARCH_ANSWER_TIMEOUT=20` are active; planner means query planning, not routing.

Remove adaptive/model-selected routing, `force_search`, high-risk/freshness policy, claim/repair/semantic validation, and route-timeout wording from `README.md`, `.env.example`, and `eval/search/README.md`. Keep credential/live-smoke warnings explicit.

- [ ] **Step 5: Run evaluator/docs GREEN tests**

```bash
python -B -m unittest tests.test_search_evaluation tests.test_readme_guide tests.test_product_scope tests.test_user_facing_scope -v
python -B tools/evaluate_search.py --help
```

Expected: all tests pass; help exits 0 and lists `traces` and `smoke`.

- [ ] **Step 6: Commit evaluator/docs**

```bash
git add tools/evaluate_search.py tests/test_search_evaluation.py tests/test_readme_guide.py README.md .env.example eval/search/README.md
git commit -m "docs: describe deterministic search modes"
```

---

### Task 11: Run hermetic verification and mocked text/image probes

**Files:**
- Modify only a specific production/test/doc file if a command below exposes a failure; add that file explicitly to the verification-fix commit.

- [ ] **Step 1: Run the full hermetic suite from a clean Python process**

Run: `python -B -m unittest discover -s tests -t . -q`

Expected: exit 0; no network credential is required and every maintained test passes.

- [ ] **Step 2: Run syntax, whitespace, import, and forbidden-symbol checks**

```bash
python -B -m compileall -q src tests tools
git diff --check
rg -n "src\.search\.(providers|models|orchestrator|planner|router|evidence|validation|policy|renderer|outcomes|budget|stage_runner|extraction)" src tests tools
rg -n "RoutePlanner|force_search|has_images|SearchTier|GroundedDraft|ClaimDiscovery|SemanticVerifier|RepairPlan|fail_closed|route[_ -]?policy|routing timeout" src tests tools README.md .env.example eval/search/README.md
```

Expected: `compileall` and `git diff --check` exit 0; both `rg` commands exit 1 with no matches.

- [ ] **Step 3: Run exact mocked routing probes**

Create an inline test process—without writing a probe file—that patches image download, simple factory, LLM, and OneBot send. Exercise these cases:

```text
ordinary text               -> LIGHT    -> exactly 1 query -> no URL
ordinary image-only         -> LIGHT    -> exactly 1 multimodal query -> image reaches answer -> no URL
ordinary text plus image    -> LIGHT    -> exactly 1 multimodal query -> image reaches answer -> no URL
/search text                -> STANDARD -> 1..3 queries -> at most 3 URLs
/search image-only          -> STANDARD -> 1..3 multimodal queries -> at most 3 URLs
/search text plus image     -> STANDARD -> 1..3 multimodal queries -> at most 3 URLs
/skip text                  -> SKIP     -> no factory/planner/provider/reader/ranker/search-answerer
/skip image-only            -> SKIP     -> image reaches plain answer and no search dependency
/skip text plus image       -> SKIP     -> image reaches plain answer and no search dependency
/skip without text/images   -> usage message and no model/search call
```

Run:

```bash
python -B -m unittest \
  tests.test_simple_search_planning \
  tests.test_simple_search_providers \
  tests.test_simple_search_retrieval \
  tests.test_simple_search_pipeline \
  tests.test_simple_search_answering \
  tests.test_simple_search_chat_flow \
  tests.test_command_renderer \
  tests.test_simple_search_rendering \
  tests.test_main_image_flow -v
```

Expected: all focused probes pass and every reply is non-empty.

- [ ] **Step 4: Verify history, privacy, and bounded-failure probes**

Run:

```bash
python -B -m unittest \
  tests.test_multimodal_chat \
  tests.test_main_image_flow \
  tests.test_simple_search_reader \
  tests.test_simple_search_ranking \
  tests.test_search_url_policy -v
```

Expected: all tests pass; original `/search` and `/skip` command text is stored, replies are stored, persisted history has placeholders but no image bytes/temporary URLs, invalid Tavily URLs trigger DDGS, and blocking provider/reader fakes return within their asserted bound.

- [ ] **Step 5: Inspect the final diff and commit only verification fixes if needed**

```bash
git status --short
git diff --stat
git diff -- docs/superpowers/specs/2026-09-04-deterministic-search-modes-design.md
```

Expected: the approved spec has no diff; no `.tmp.driveupload/`, `docs/plans/task.md`, or `websearch-simplification-report.md` is staged. If verification required tracked fixes, stage each named file rather than `git add -A`, then run:

```bash
git commit -m "fix: complete deterministic search rewrite"
```

If no fixes were required, do not create an empty commit.

---

### Task 12: Run authorized live text and image smoke tests

**Files:**
- No repository files are modified.

- [ ] **Step 1: Confirm authorization and credentials without printing secrets**

Run:

```bash
python -B - <<'PY'
import os
required = {
    "live_authorized": os.getenv("QQBOT_ALLOW_LIVE_SEARCH_SMOKE") == "1",
    "chat_model_configured": bool(os.getenv("GEMINI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")),
    "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
}
print(required)
raise SystemExit(0 if required["live_authorized"] and required["chat_model_configured"] else 2)
PY
```

Expected when authorized: printed booleans only and exit 0. If exit 2, record `SKIPPED: live smoke not authorized/configured`; do not weaken hermetic verification and do not print key values.

- [ ] **Step 2: Run live LIGHT and STANDARD text probes**

Run: `QQBOT_ALLOW_LIVE_SEARCH_SMOKE=1 python -B tools/evaluate_search.py smoke`

Expected: two non-empty results; safe traces show chat/LIGHT with exactly one query and command/STANDARD with one to three queries. Trace output contains no query, URL, body, image data, credential, or exception message.

- [ ] **Step 3: Run an authorized image probe through each mode**

Set `QQBOT_SMOKE_IMAGE_DATA_URL` in the shell to a benign `data:image/...;base64,...` value without writing it to disk, then run:

```bash
QQBOT_ALLOW_LIVE_SEARCH_SMOKE=1 python -B - <<'PY'
import os
import re
from unittest.mock import patch

from src.chat.chat_service import generate_reply
from src.search.simple import SearchMode, get_simple_search_pipeline

image = os.environ.get("QQBOT_SMOKE_IMAGE_DATA_URL", "")
if os.environ.get("QQBOT_ALLOW_LIVE_SEARCH_SMOKE") != "1" or not image.startswith("data:image/"):
    raise SystemExit("Set authorization and QQBOT_SMOKE_IMAGE_DATA_URL before the image smoke test.")


class RecordingPipeline:
    def __init__(self, delegate):
        self.delegate = delegate
        self.outcomes = []

    def run(self, request):
        outcome = self.delegate.run(request)
        self.outcomes.append(outcome)
        return outcome


pipeline = RecordingPipeline(get_simple_search_pipeline())
for mode, history in ((SearchMode.LIGHT, None), (SearchMode.STANDARD, "/search")):
    with patch("src.chat.chat_service.get_simple_search_pipeline_for_chat", return_value=pipeline), patch(
        "src.chat.chat_service.append_history"
    ):
        reply = generate_reply(
            "smoke:image", "识别并查询图片中的主体", [image],
            mode=mode, history_text=history,
        )
    outcome = pipeline.outcomes[-1]
    url_count = len(re.findall(r"https?://", reply))
    print({"mode": mode.value, "reply_chars": len(reply), "url_count": url_count, "trace": outcome.trace.to_safe_dict()})
    assert reply
    assert outcome.plan.mode is mode
    assert len(outcome.plan.queries) == (1 if mode is SearchMode.LIGHT else len(outcome.plan.queries))
    assert 1 <= len(outcome.plan.queries) <= (1 if mode is SearchMode.LIGHT else 3)
    assert url_count == 0 if mode is SearchMode.LIGHT else url_count <= 3

with patch("src.chat.chat_service.get_simple_search_pipeline_for_chat") as forbidden_factory, patch(
    "src.chat.chat_service.append_history"
):
    skip_reply = generate_reply(
        "smoke:image", "识别图片", [image], mode=SearchMode.SKIP, history_text="/skip"
    )
forbidden_factory.assert_not_called()
print({"mode": "skip", "reply_chars": len(skip_reply), "url_count": len(re.findall(r"https?://", skip_reply)), "search_factory_calls": 0})
assert skip_reply
PY
```

Expected: LIGHT prints exactly one query in its safe trace and zero displayed URLs; STANDARD prints one to three queries and at most three URLs; SKIP prints a nonzero reply length with `search_factory_calls: 0`. No printed record contains the image data URL.

- [ ] **Step 4: Induce Tavily invalid-URL degradation without exposing content**

Run this process with DDGS installed/network-authorized:

```bash
QQBOT_ALLOW_LIVE_SEARCH_SMOKE=1 python -B - <<'PY'
import os

from src.config import config
from src.search.simple.ddgs import DDGSSearchProvider
from src.search.simple.models import RequestSource, SearchMode, SearchPlan, SearchQuery, SearchTrace
from src.search.simple.providers import ProviderHit, ProviderReadiness, ProviderResult, ProviderStatus
from src.search.simple.retrieval import ProviderRunner

if os.environ.get("QQBOT_ALLOW_LIVE_SEARCH_SMOKE") != "1":
    raise SystemExit("Live fallback smoke is not authorized.")


class InvalidUrlTavily:
    name = "tavily"

    def readiness(self):
        return ProviderReadiness("tavily", True, True)

    def search(self, query, *, mode, max_results, timeout_seconds):
        return ProviderResult(
            "tavily", ProviderStatus.SUCCESS,
            (ProviderHit("tavily", query.query_id, "invalid", "http://127.0.0.1/private", "blocked"),),
        )


trace = SearchTrace("live-fallback", RequestSource.CHAT, SearchMode.LIGHT)
runner = ProviderRunner(
    (InvalidUrlTavily(), DDGSSearchProvider(proxy_url=config.proxy_url, timeout_seconds=config.search_ddgs_timeout)),
    config.search_tavily_timeout,
    config.search_ddgs_timeout,
    config.search_max_results,
)
results = runner.run(
    SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "Python official documentation"),)),
    trace,
)
print({"provider_statuses": trace.provider_statuses, "result_count": len(results)})
assert "ddgs" in trace.provider_statuses
assert all("127.0.0.1" not in result.url for result in results)
PY
```

Expected: output contains only provider statuses and result count; DDGS has a status because the invalid Tavily hit did not resolve the query, and no private URL is admitted. The process-local fake disappears on exit and no environment or repository file is changed.

- [ ] **Step 5: Finish without a live-smoke commit**

Run: `git status --short`

Expected: live probes created no tracked repository changes. Do not commit credentials, image files/data URLs, trace bodies, or live output.
