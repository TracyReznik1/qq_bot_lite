# DDGS-First Search Routing and Disclosures Implementation Plan

> **Historical baseline:** This 2026-08-08 plan is superseded by the approved
> `2026-08-09-websearch-simplification.md` spec/plan. The runtime now keeps only
> `skip / light / standard` (operational `deep` is removed), DDGS-first with
> Tavily fallback only, and risk affects the answer policy rather than search
> depth. Preserved here for historical reasoning only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DDGS the primary search provider, use Tavily only as the bounded fallback, route high-freshness questions conservatively, skip pure greetings, and show search uncertainty warnings only for high-consequence conversations.

**Architecture:** Keep the existing retrieval-benefit router, provider-neutral registry, Evidence pipeline, validator, and QQ renderer. Add program-owned freshness and greeting rules in the router, invert provider selection with a tier-specific Tavily reserve inside the existing absolute deadline, neutralize DDGS hit metadata, and separate search-failure disclosures from high-consequence warnings. Both normal chat and `/search` continue through `generate_reply()` and the same renderer.

**Tech Stack:** Python 3.13, standard-library `dataclasses`, `enum`, `re`, `threading`, `concurrent.futures`, `unittest`, Flask/OneBot, existing DDGS and Tavily adapters.

## Global Constraints

- DDGS is the deterministic primary provider; Tavily is called only when DDGS is unavailable, errors, times out, or returns no usable hits.
- A successful DDGS call suppresses Tavily for the same `SearchQuery`.
- Provider fallback for one `query_id` remains one semantic query and one retrieval round, with separate immutable `ProviderAttempt` rows.
- Preserve the single absolute monotonic tier deadline and the fixed eight-worker adapter executor; no late provider invocation or Trace mutation is allowed after return.
- Tavily fallback reserves are exactly 3.5 seconds for `light`, 5.0 seconds for `standard`, and 8.0 seconds for `deep` when enough time remains at query start.
- Do not increase the existing 8/20/40-second tier hard timeouts.
- `classification.freshness == high` creates a program-owned `deep` floor; the model cannot lower it.
- Pure greetings are `skip/social_or_emotional`; a greeting mixed with an external-fact request must still search.
- Search success in chat and `/search` contains only the grounded answer and used sources—never “检索完成”, “搜索成功”, or `搜索状态：success`.
- The fixed “搜索结果可能不完整或不准确” warning appears only when `potential_harm == high` or `high_consequence_action` is present, exactly once.
- Search failure disclosures remain independent: low-risk failures may say “在线检索未完成” without a high-consequence warning.
- Explicit no-web high-consequence requests use “本次未联网核验” wording and must not claim that search results exist.
- Do not edit `eval/search/*.jsonl`, fabricate human review metadata, perform controlled online evaluation, merge, push, or restart the running bot during implementation.
- Implement every behavior with failing tests first; use fake adapters and deterministic/local timing only, with no external HTTP in the hermetic suite.
- If subagents are used, prefer GPT-5.6 Luna at MAX reasoning when that model is exposed by the runtime; if Luna is unavailable or cannot resolve the task, use GPT-5.6 Sol. Do not claim a model was used unless the spawn interface actually accepted it.

---

## File Map

- `src/search/router.py`: owns pure-greeting skip rules, relative-time/result intent detection, high-freshness floor, and the known unilateral-weakness high-consequence signal.
- `src/search/providers/base.py`: owns DDGS-first selection, conditional Tavily reserve, attempts, deadlines, and fallback outcomes.
- `src/search/providers/ddgs.py`: maps DDGS SDK results to provider-neutral hits without order-dependent quality flags.
- `src/search/orchestrator.py`: builds the production provider list in DDGS-then-Tavily order.
- `src/search/renderer.py`: owns failure disclosures, high-consequence warnings, deduplication, citations, and QQ chunks.
- `src/services/search_service.py`: compatibility facade; must not emit a user-visible success-status banner.
- `src/commands/search.py`: stays a thin `/search` wrapper over `generate_reply(force_search=True)`.
- `README.md`, `.env.example`: document the new provider order and output behavior.
- `tests/test_search_router.py`: routing and high-consequence regressions.
- `tests/test_search_providers.py`: adapter order, reserve, attempts, deadlines, and DDGS metadata.
- `tests/test_search_orchestrator.py`, `tests/test_health.py`: production graph and readiness order.
- `tests/test_search_renderer.py`: warning/failure/success output matrix.
- `tests/test_chat_retrieval_flow.py`, `tests/test_product_scope.py`: normal chat, `/search`, and compatibility facade behavior.
- `tests/test_readme_guide.py`, `tests/test_user_facing_scope.py`: documentation and static user-facing copy invariants.

---

### Task 1: Enforce High-Freshness Floors and Pure-Greeting Skips

**Files:**
- Modify: `src/search/router.py:145-217`
- Modify: `src/search/router.py:881-958`
- Modify: `src/search/router.py:1334-1370`
- Test: `tests/test_search_router.py`

**Interfaces:**
- Consumes: `_Classification.freshness: Freshness`, `RetrievalDecision`, `SearchTier`, `TriggerCode.FRESHNESS_MARKER`, `SkipReason.SOCIAL_OR_EMOTIONAL`.
- Produces: `_detect_current_state(question) -> tuple[TriggerCode, ...]`, `_is_pure_greeting(question) -> bool`, and a `SearchTier.DEEP` program floor whenever `classification.freshness is Freshness.HIGH`.

- [ ] **Step 1: Add failing router tests for model freshness, relative time, and greetings**

Add a focused class to `tests/test_search_router.py`:

```python
class FreshnessAndGreetingRegressionTests(unittest.TestCase):
    def test_model_high_freshness_cannot_remain_light(self):
        payload = {
            **NEUTRAL,
            "freshness": "high",
            "recommended_tier": "light",
            "trigger_codes": ["freshness_marker"],
        }
        decision = decide("昨天天曼契约EDGVSTEC谁赢了", payload)
        self.assertIs(decision.route, SearchTier.DEEP)
        self.assertIs(decision.program_minimum_tier, SearchTier.DEEP)
        self.assertIn(TriggerCode.FRESHNESS_MARKER, decision.trigger_codes)

    def test_relative_time_and_result_intent_is_deep_without_model_help(self):
        for question in (
            "昨天EDG和TEC谁赢了",
            "前天比赛比分是多少",
            "本周排名结果如何",
            "刚刚发生了什么",
        ):
            with self.subTest(question=question):
                self.assertIs(decide(question, NEUTRAL).route, SearchTier.DEEP)

    def test_pure_greetings_skip_search(self):
        for question in ("你好", "您好！", "下午好", "晚上好，ATRI", "哈喽"):
            with self.subTest(question=question):
                decision = decide(question, {})
                self.assertIs(decision.route, SearchTier.SKIP)
                self.assertIs(decision.skip_reason, SkipReason.SOCIAL_OR_EMOTIONAL)

    def test_greeting_plus_current_fact_does_not_skip(self):
        decision = decide("下午好，昨天EDG赢了吗", NEUTRAL)
        self.assertIs(decision.route, SearchTier.DEEP)
        self.assertIsNone(decision.skip_reason)
```

- [ ] **Step 2: Run the new tests and verify the expected RED state**

Run:

```powershell
python -B -m unittest tests.test_search_router.FreshnessAndGreetingRegressionTests -v
```

Expected: failures show high freshness still routes `light`, “昨天…谁赢了” remains below `deep`, and pure greetings do not `skip`.

- [ ] **Step 3: Add closed deterministic time/result and greeting helpers**

In `src/search/router.py`, define exact closed sets near `_DATED_EXTERNAL_CONTEXT`:

```python
_RELATIVE_TIME_MARKERS = (
    "今天", "今日", "昨天", "昨日", "前天", "刚刚", "最近", "近期",
    "目前", "现在", "当前", "实时", "本周", "上周", "本月", "上月",
    "今年", "去年",
)

_CURRENT_RESULT_INTENTS = (
    "谁赢了", "胜负", "比分", "赛果", "比赛结果", "排名",
    "发生了什么", "结果如何", "最新进展",
)

_PURE_GREETING_PREFIXES = (
    "你好", "您好", "嗨", "哈喽", "在吗", "早上好", "中午好",
    "下午好", "晚上好", "晚安",
)

_GREETING_VOCATIVES = ("", "atri", "亚托莉", "机器人")


def _is_pure_greeting(question: str) -> bool:
    normalized = unicodedata.normalize("NFKC", str(question or "")).casefold()
    normalized = re.sub(r"[\s，,。.!！?？、~～]+", "", normalized)
    return any(
        normalized == f"{greeting}{vocative}"
        for greeting in _PURE_GREETING_PREFIXES
        for vocative in _GREETING_VOCATIVES
    )
```

Extend `_detect_current_state()` with a structural combination, not a standalone keyword hit:

```python
    if (
        any(marker in lowered for marker in _RELATIVE_TIME_MARKERS)
        and (
            any(context in lowered for context in _DATED_EXTERNAL_CONTEXT)
            or any(intent in lowered for intent in _CURRENT_RESULT_INTENTS)
        )
    ):
        codes.append(TriggerCode.FRESHNESS_MARKER)
```

Check `_is_pure_greeting(question)` at the beginning of `_classify_closed_task()` and return `SkipReason.SOCIAL_OR_EMOTIONAL` before substring-based social markers.

- [ ] **Step 4: Turn model high freshness into a program-owned floor**

At the start of `_compute_floors()` after safety classification, add:

```python
    if classification.freshness is Freshness.HIGH:
        floor = _max_tier(floor, SearchTier.DEEP)
        codes.append(TriggerCode.FRESHNESS_MARKER)
```

Keep `_dedupe_codes()` as the only final deduplicator so a deterministic and model freshness marker cannot appear twice.

- [ ] **Step 5: Run router tests**

Run:

```powershell
python -B -m unittest tests.test_search_router -v
```

Expected: all router tests pass; stable definitions remain `light`/`standard`, pure greetings skip, and mixed greeting-plus-fact requests search.

- [ ] **Step 6: Commit the routing change**

```powershell
git add src/search/router.py tests/test_search_router.py
git commit -m "fix: enforce freshness search floors"
```

---

### Task 2: Preserve High-Consequence Detection After Removing Universal Warnings

**Files:**
- Modify: `src/search/router.py:360-390`
- Test: `tests/test_search_router.py`

**Interfaces:**
- Consumes: `_RED_FLAG_PATTERNS`, `_TRIAGE_ACTION_PATTERN`, `_classify_safety_intent()`.
- Produces: `TriggerCode.HIGH_CONSEQUENCE_ACTION`, `PotentialHarm.HIGH`, and `SearchTier.DEEP` for the known actionable unilateral-weakness phrasing; stable or negated descriptions remain non-high-consequence.

- [ ] **Step 1: Add failing positive and negative safety regressions**

Add to `tests/test_search_router.py`:

```python
class WarningSignalBoundaryTests(unittest.TestCase):
    def test_one_leg_sudden_weakness_is_high_consequence(self):
        decision = decide("一条腿突然没力气，要去急诊吗？", {})
        self.assertIs(decision.route, SearchTier.DEEP)
        self.assertIn(TriggerCode.HIGH_CONSEQUENCE_ACTION, decision.trigger_codes)
        self.assertIs(decision.potential_harm, PotentialHarm.HIGH)

    def test_stable_or_negated_leg_text_is_not_high_consequence(self):
        for question in (
            "一条腿没力气可能有哪些一般原因",
            "我并没有一条腿突然没力气，只是在引用这个句子",
        ):
            with self.subTest(question=question):
                decision = decide(question, NEUTRAL)
                self.assertNotIn(
                    TriggerCode.HIGH_CONSEQUENCE_ACTION,
                    decision.trigger_codes,
                )
```

- [ ] **Step 2: Run the boundary tests and verify RED**

Run:

```powershell
python -B -m unittest tests.test_search_router.WarningSignalBoundaryTests -v
```

Expected: the actionable “一条腿” case is not classified high consequence; negative controls continue to pass.

- [ ] **Step 3: Extend only the existing unilateral subject grammar**

In the unilateral `_RED_FLAG_PATTERNS` expression, add `一条` to the existing subject alternatives:

```python
re.compile(
    r"(?:一只|一条|一边|一侧|单侧|半边|左侧|右侧|左|右).{0,4}"
    r"(?:手脚|手|脚|肢|臂|胳膊|腿|脸|面|身体).{0,6}"
    r"(?:无力|没力气|使不上劲|麻木|发麻|麻|抬不起来|抬不动|动不了|歪斜|下垂)"
),
```

Do not add a global substring shortcut. Continue requiring an active red-flag match plus `_TRIAGE_ACTION_PATTERN`; quote, meta, absence, and stable-explanation scoping must remain active.

- [ ] **Step 4: Run the safety and full router suites**

Run:

```powershell
python -B -m unittest tests.test_search_router.WarningSignalBoundaryTests tests.test_search_router.RouterConflictAndFloorTests -v
python -B -m unittest tests.test_search_router -v
```

Expected: all pass, including prior medication, acute symptom, chemical exposure, negation, and stable-definition boundaries.

- [ ] **Step 5: Commit the safety signal change**

```powershell
git add src/search/router.py tests/test_search_router.py
git commit -m "fix: retain high consequence warning signals"
```

---

### Task 3: Make DDGS Primary and Reserve Time for Tavily Fallback

**Files:**
- Modify: `src/search/providers/base.py:74-352`
- Test: `tests/test_search_providers.py`

**Interfaces:**
- Consumes: `ProviderRegistry.search_with_attempts()`, `SearchTier`, `ProviderReadiness`, `_call_until_deadline()` and the existing global adapter executor.
- Produces: `_primary_provider() -> DDGS`, `_fallback_provider() -> Tavily`, `_primary_deadline(deadline, tier, fallback) -> float`, and immutable attempt order `("ddgs", "tavily")` when fallback occurs.

- [ ] **Step 1: Rewrite old provider-order expectations as failing DDGS-first tests**

Rename and update the existing registry tests instead of keeping contradictory Tavily-first names:

```python
def _ready_provider(self, name, *, status=ProviderStatus.SUCCESS):
    provider = mock.Mock()
    provider.name = name
    provider.readiness.return_value = ProviderReadiness(
        name, True, True, None,
    )
    provider.search.return_value = _provider_result(
        name, status=status, latency_ms=3,
    )
    return provider

def test_registry_exposes_ddgs_before_tavily(self):
    tavily = self._ready_provider("tavily")
    ddgs = self._ready_provider("ddgs")
    registry = self._registry(tavily, ddgs)
    self.assertEqual(registry.readiness()[0].provider, "ddgs")

def test_primary_search_uses_ddgs(self):
    tavily = self._ready_provider("tavily")
    ddgs = self._ready_provider("ddgs")
    ddgs.search.return_value = _provider_result("ddgs", latency_ms=3)
    result = self._registry(tavily, ddgs).search(
        query(), tier=SearchTier.STANDARD, max_results=8, timeout_seconds=20.0,
    )
    self.assertEqual(result.provider, "ddgs")
    ddgs.search.assert_called_once()
    tavily.search.assert_not_called()

def test_no_usable_ddgs_falls_back_to_tavily(self):
    tavily = self._ready_provider("tavily")
    ddgs = self._ready_provider("ddgs")
    ddgs.search.return_value = _provider_result(
        "ddgs", status=ProviderStatus.ERROR, latency_ms=5,
    )
    tavily.search.return_value = _provider_result("tavily", latency_ms=3)
    outcome = self._registry(tavily, ddgs).search_with_attempts(
        query(), tier=SearchTier.LIGHT, max_results=5, timeout_seconds=8.0,
    )
    self.assertEqual(outcome.status, ProviderStatus.SUCCESS)
    self.assertEqual(
        [(attempt.provider, attempt.query_id) for attempt in outcome.attempts],
        [("ddgs", "q1"), ("tavily", "q1")],
    )
```

Change the test helper to construct the registry in intended readiness order:

```python
def _registry(self, tavily=None, ddgs=None):
    return self.base.ProviderRegistry(
        [item for item in (ddgs, tavily) if item is not None]
    )
```

Apply these exact test renames and expectation changes:

| Existing test | Required DDGS-first expectation |
|---|---|
| `test_fallback_timeout_preserves_completed_primary_and_real_fallback_attempt` | DDGS completed attempt precedes a real Tavily timeout attempt. |
| `test_primary_error_then_queued_fallback_timeout_dominates_without_synthetic_attempt` | DDGS is the recorded error; queued Tavily never starts; final status is timeout. |
| `test_public_search_reports_queued_fallback_timeout_after_primary_error` | Public `ProviderResult` is timeout after DDGS error and unstarted Tavily. |
| `test_no_usable_tavily_falls_back_to_ddgs` | Rename to `test_no_usable_ddgs_falls_back_to_tavily`; attempts are `ddgs, tavily`. |
| `test_unavailable_primary_is_skipped_without_invocation` | Unavailable DDGS is not invoked; Tavily is the sole real attempt. |
| `test_fallback_same_query_id_is_one_semantic_query` | Both DDGS and Tavily attempts retain `q1`. |
| `test_attempts_keep_each_adapter_latency_and_reduced_fallback_time` | Rename local callbacks and assert attempt order `ddgs, tavily`; primary receives the reserve-capped timeout. |
| `test_no_fallback_after_usable_tavily_hits` | Rename to `test_no_fallback_after_usable_ddgs_hits`; Tavily call count is zero. |

Do not weaken the existing fixed-pool saturation, invocation-start readiness snapshot, timeout-winner, start-winner, concurrent request isolation, or late-callback assertions; only substitute provider roles where the test models primary versus fallback.

- [ ] **Step 2: Add a failing scaled reserve-budget test**

Add this deterministic-duration test using a patched small reserve so the suite stays fast:

```python
def test_ddgs_primary_timeout_preserves_tavily_fallback_window(self):
    observed = []
    tavily = self._ready_provider("tavily")
    ddgs = self._ready_provider("ddgs")

    def slow_ddgs(_query, **kwargs):
        observed.append(("ddgs", kwargs["timeout_seconds"]))
        time.sleep(0.2)
        return ProviderResult("ddgs", ProviderStatus.ERROR, (), 0)

    def fast_tavily(search_query, **kwargs):
        observed.append(("tavily", kwargs["timeout_seconds"]))
        return _provider_result("tavily", query_id=search_query.query_id)

    ddgs.search.side_effect = slow_ddgs
    tavily.search.side_effect = fast_tavily
    reserve = {
        SearchTier.LIGHT: 0.08,
        SearchTier.STANDARD: 0.08,
        SearchTier.DEEP: 0.08,
    }
    with mock.patch.object(
        self.base, "_TAVILY_FALLBACK_RESERVE_SECONDS", reserve,
    ):
        outcome = self._registry(tavily, ddgs).search_with_attempts(
            query(), tier=SearchTier.LIGHT, max_results=1, timeout_seconds=0.2,
        )

    self.assertEqual(outcome.status, ProviderStatus.SUCCESS)
    self.assertEqual([name for name, _ in observed], ["ddgs", "tavily"])
    self.assertLessEqual(observed[0][1], 0.125)
    self.assertGreater(observed[1][1], 0.04)
```

Retain the existing start-winner/timeout-winner race tests. They must still prove that an unstarted DDGS or Tavily call cannot begin after timeout sealing.

- [ ] **Step 3: Run provider tests and verify RED**

Run:

```powershell
python -B -m unittest tests.test_search_providers.RegistryTests -v
```

Expected: provider order and reserve tests fail against the Tavily-first registry; deadline-race tests should remain internally valid.

- [ ] **Step 4: Implement DDGS-first selection and tier reserves**

In `src/search/providers/base.py`, add the closed reserve map:

```python
_TAVILY_FALLBACK_RESERVE_SECONDS = {
    SearchTier.LIGHT: 3.5,
    SearchTier.STANDARD: 5.0,
    SearchTier.DEEP: 8.0,
}
```

Change the class description and selectors:

```python
class ProviderRegistry:
    """Select DDGS first and use Tavily as the bounded fallback."""

    def _primary_provider(self) -> SearchProvider | None:
        return self._provider_named("ddgs")

    def _fallback_provider(self) -> SearchProvider | None:
        return self._provider_named("tavily")

    def _provider_named(self, name: str) -> SearchProvider | None:
        return next(
            (provider for provider in self._providers if provider.name == name),
            None,
        )
```

Add a helper that derives a sub-deadline without changing the overall deadline:

```python
def _primary_deadline(
    self,
    deadline: float,
    tier: SearchTier,
    fallback: SearchProvider | None,
) -> float:
    reserve = _TAVILY_FALLBACK_RESERVE_SECONDS[tier]
    if (
        fallback is not None
        and fallback.readiness().available
        and self._remaining(deadline) > reserve
    ):
        return deadline - reserve
    return deadline
```

Resolve `primary` and `fallback` before the primary call. Pass `_primary_deadline(...)` only to the DDGS `_call_until_deadline()` invocation; keep the original absolute `deadline` for Tavily. If DDGS returns early, Tavily receives all actual remaining time. If the remaining time was already at or below the reserve, DDGS receives the true remaining deadline and Tavily is not guaranteed to run.

- [ ] **Step 5: Run the full provider suite, including race stress**

Run:

```powershell
python -B -m unittest tests.test_search_providers -v
```

Expected: all adapter, readiness, timeout, fallback, fixed-pool, request-local attempt, and late-mutation tests pass with DDGS-first expectations.

- [ ] **Step 6: Commit the provider registry change**

```powershell
git add src/search/providers/base.py tests/test_search_providers.py
git commit -m "feat: prefer DDGS with Tavily fallback"
```

---

### Task 4: Align the Production Provider Graph and Neutralize DDGS Metadata

**Files:**
- Modify: `src/search/orchestrator.py:930-958`
- Modify: `src/search/providers/ddgs.py:75-87`
- Modify: `tests/test_search_providers.py`
- Modify: `tests/test_search_orchestrator.py`
- Modify: `tests/test_health.py`

**Interfaces:**
- Consumes: `DDGSSearchProvider`, `TavilySearchProvider`, `ProviderHit.quality_flags`, `get_search_orchestrator()`.
- Produces: production `_providers == (ddgs, tavily?)`; DDGS hits carry no order-dependent `availability_fallback` flag.

- [ ] **Step 1: Add failing production-order and neutral-metadata tests**

Update `DDGSAdapterTests.test_normalizes_fields_without_score_or_raw`:

```python
self.assertNotIn("availability_fallback", hit.quality_flags)
self.assertEqual(hit.quality_flags, ())
```

Add to `tests/test_search_orchestrator.py` with a closed fake configuration and inert provider constructors:

```python
def test_production_provider_graph_is_ddgs_then_tavily(self):
    module = orchestrator_module()
    fake_config = SimpleNamespace(
        tavily_api_key="test-key",
        proxy_url="",
        request_timeout=18.0,
    )
    ddgs = SimpleNamespace(name="ddgs")
    tavily = SimpleNamespace(name="tavily")
    with (
        mock.patch.object(module, "config", fake_config),
        mock.patch(
            "src.services.llm_client.get_llm_client",
            return_value=SimpleNamespace(chat=mock.Mock()),
        ),
        mock.patch.object(module, "DDGSSearchProvider", return_value=ddgs),
        mock.patch.object(module, "TavilySearchProvider", return_value=tavily),
    ):
        orchestrator = module._build_production_orchestrator()
    self.assertEqual(
        [provider.name for provider in orchestrator._providers],
        ["ddgs", "tavily"],
    )
```

Update `tests/test_health.py` readiness fixture and assertion:

```python
readiness = (
    ProviderReadiness("ddgs", True, True, None),
    ProviderReadiness("tavily", True, True, None),
)
self.assertEqual("ddgs", response["search_providers"][0]["provider"])
self.assertEqual("tavily", response["search_providers"][1]["provider"])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -B -m unittest tests.test_search_providers.DDGSAdapterTests tests.test_search_orchestrator tests.test_health -v
```

Expected: DDGS metadata still contains `availability_fallback`, production graph/health expose Tavily first, and new order assertions fail.

- [ ] **Step 3: Build providers in production order**

Change `_build_production_orchestrator()` to append DDGS first and optional Tavily second:

```python
providers: list[Any] = [
    DDGSSearchProvider(
        proxy_url=config.proxy_url,
        timeout_seconds=config.request_timeout,
    )
]
if config.tavily_api_key:
    providers.append(
        TavilySearchProvider(
            api_key=config.tavily_api_key,
            proxy_url=config.proxy_url,
        )
    )
```

Do not change the lazy singleton or readiness API.

- [ ] **Step 4: Remove the order-dependent DDGS quality flag**

In `_ddgs_hit()`, set:

```python
quality_flags=(),
```

Do not replace it with `primary`, `trusted`, or another provider-order flag. Relevance, readable page extraction, publisher/source relation, and semantic support remain Evidence-layer decisions.

- [ ] **Step 5: Run production graph, adapter, Evidence, and health suites**

Run:

```powershell
python -B -m unittest tests.test_search_providers tests.test_search_orchestrator tests.test_search_evidence tests.test_health -v
```

Expected: all pass; no Evidence test depends on `availability_fallback` as a proxy for relevance or authority.

- [ ] **Step 6: Commit graph and metadata changes**

```powershell
git add src/search/orchestrator.py src/search/providers/ddgs.py tests/test_search_providers.py tests/test_search_orchestrator.py tests/test_health.py
git commit -m "refactor: align production search provider order"
```

---

### Task 5: Separate Failure Disclosures from High-Consequence Warnings

**Files:**
- Modify: `src/search/renderer.py:23-58`
- Modify: `src/search/renderer.py:190-330`
- Modify: `src/search/renderer.py:561-570`
- Modify: `src/services/search_service.py:52-94`
- Test: `tests/test_search_renderer.py`
- Test: `tests/test_chat_retrieval_flow.py`
- Test: `tests/test_product_scope.py`

**Interfaces:**
- Consumes: `RetrievalDecision.potential_harm`, `TriggerCode.HIGH_CONSEQUENCE_ACTION`, `SkipReason.USER_FORBID_WEB`, `SearchFailureCode`, `ValidationReport`.
- Produces: `_is_high_consequence(result) -> bool`, `_search_warning_disclosures(result) -> list[str]`, ordinary success/failure without the warning, and compatibility success text without status banners.

- [ ] **Step 1: Replace universal-warning tests with the approved output matrix**

In `tests/test_search_renderer.py`, first add one exact supported-report helper:

```python
def supported_report():
    draft = GroundedDraft(
        (AnswerBlock("B1", "factual", "版本是3.2", ("C1",)),),
        (Claim("C1", "B1", "版本是3.2", True, ("E1",)),),
        (),
        (),
        False,
    )
    return ValidationReport(
        draft,
        draft.answer_blocks,
        draft.claims,
        (),
        {},
        (),
    )
```

Then replace tests that require one warning for every searched answer with:

```python
def test_ordinary_success_has_answer_and_source_without_warning_or_status(self):
    rendered = renderer_module().render_search_reply(
        result(bundle((item(),))), supported_report(), qq_limit=1700,
    )
    self.assertIn("版本是3.2", rendered.text)
    self.assertIn("https://example.com/page", rendered.text)
    self.assertNotIn("搜索结果可能不完整或不准确", rendered.text)
    self.assertNotIn("检索完成", rendered.text)
    self.assertNotIn("搜索成功", rendered.text)
    self.assertNotIn("搜索状态：success", rendered.text)

def test_ordinary_failure_keeps_failure_disclosure_without_risk_warning(self):
    rendered = renderer_module().render_search_reply(
        result(None, SearchFailureCode.NO_RESULTS, route=SearchTier.LIGHT),
        None,
        knowledge_fallback_text="有限知识",
        qq_limit=1700,
    )
    self.assertIn("在线检索未完成", rendered.text)
    self.assertNotIn("不能替代适当的专业判断", rendered.text)

def test_ordinary_deep_news_success_has_no_high_consequence_warning(self):
    rendered = renderer_module().render_search_reply(
        result(bundle((item(),)), route=SearchTier.DEEP),
        supported_report(),
        qq_limit=1700,
    )
    self.assertNotIn("不能替代适当的专业判断", rendered.text)
```

In the existing ordinary `PartialRenderingTests` and `ConflictRenderingTests`, add this assertion to the successful low-risk render cases:

```python
self.assertNotIn("不能替代适当的专业判断", rendered.text)
```

Do not remove their partial-prefix, conflict-member, citation, limitation, or source-order assertions.

Keep and strengthen the high-consequence success/failure/partial/conflict tests so each asserts `warning count == 1`.

Add the no-web wording boundary:

```python
def test_no_web_high_consequence_uses_no_web_warning_once(self):
    from src.search.router import RetrievalBenefitRouter
    from tests.search_fakes import StaticRouterAdvisor

    module = renderer_module()
    decision = RetrievalBenefitRouter(StaticRouterAdvisor({})).decide(
        models().RetrievalRequest("不要联网，我发烧39度，该吃多少布洛芬？")
    )
    search_result = models().SearchPipelineResult(
        decision,
        None,
        None,
        trace(SearchTier.SKIP),
        SearchFailureCode.USER_FORBID_WEB,
    )
    rendered = module.render_search_reply(search_result, None, qq_limit=1700)
    self.assertEqual(1, rendered.text.count("本次未联网核验"))
    self.assertNotIn("本次没有联网核验", rendered.text)
    self.assertNotIn("搜索结果可能不完整或不准确", rendered.text)
```

- [ ] **Step 2: Add chat and `/search` success regressions**

In `tests/test_chat_retrieval_flow.py`, extend `SearchFlowTests`:

```python
def test_normal_and_force_search_success_emit_no_status_banner(self):
    for force_search in (False, True):
        with self.subTest(force_search=force_search):
            reply, _llm, _orch = self._run(
                search_result(SearchTier.LIGHT, bundle((item(),))),
                force_search=force_search,
            )
            self.assertIn("版本是3.2", reply)
            self.assertIn("来源：", reply)
            for forbidden in ("检索完成", "搜索成功", "搜索状态：success"):
                self.assertNotIn(forbidden, reply)
            self.assertNotIn("不能替代适当的专业判断", reply)
```

Extend `FailureFlowTests` with the dynamic low-risk failure boundary:

```python
def test_dynamic_low_risk_failure_does_not_use_memory_or_risk_warning(self):
    result = search_result(
        SearchTier.DEEP,
        failure=SearchFailureCode.PROVIDER_TIMEOUT,
    )
    reply, llm_chat = self._run(
        result,
        text="昨天天曼契约EDGVSTEC谁赢了",
    )
    self.assertEqual(0, llm_chat.call_count)
    self.assertIn("无法完成在线核验", reply)
    self.assertNotIn("不能替代适当的专业判断", reply)
    self.assertNotIn("EDG赢了", reply)
    self.assertNotIn("TEC赢了", reply)
```

In `tests/test_product_scope.py`, add this complete compatibility-facade success test:

```python
def test_compatibility_search_success_has_no_status_banner(self):
    trace = SearchTrace("req-1", RequestSource.COMPATIBILITY, SearchTier.LIGHT)
    evidence = SimpleNamespace(
        evidence_items=(
            SimpleNamespace(
                title="Example",
                url="https://example.com/page",
                excerpt="版本是3.2",
            ),
        ),
    )
    pipeline_result = SimpleNamespace(
        decision=SimpleNamespace(route=SearchTier.LIGHT),
        evidence=evidence,
        failure_code=None,
        trace=trace,
    )
    orchestrator = mock.Mock(run=mock.Mock(return_value=pipeline_result))
    with (
        mock.patch.object(
            search_service, "get_search_orchestrator", return_value=orchestrator,
        ),
        mock.patch("src.search.orchestrator.finalize_search_trace"),
    ):
        result = search_service.search("当前版本是什么")

    self.assertTrue(result.ok)
    self.assertNotIn("搜索状态：success", result.text)
    self.assertNotIn("搜索成功", result.text)
    self.assertIn("版本是3.2", result.text)
    self.assertIn("https://example.com/page", result.text)
```

- [ ] **Step 3: Run renderer/chat/product tests and verify RED**

Run:

```powershell
python -B -m unittest tests.test_search_renderer tests.test_chat_retrieval_flow tests.test_product_scope -v
```

Expected: ordinary searched paths still receive the universal warning, no-web high-consequence uses search-result wording, and the compatibility facade still emits `搜索状态：success`.

- [ ] **Step 4: Implement the closed high-consequence predicate and no-web copy**

In `src/search/renderer.py`, add:

```python
_NO_WEB_HIGH_CONSEQUENCE_WARNING = (
    "重要提示：本次未联网核验，以下内容不能替代适当的专业判断。"
)


def _is_high_consequence(result: SearchPipelineResult) -> bool:
    decision = result.decision
    return (
        decision.potential_harm is PotentialHarm.HIGH
        or TriggerCode.HIGH_CONSEQUENCE_ACTION in decision.trigger_codes
    )


def _search_warning_disclosures(result: SearchPipelineResult) -> list[str]:
    if not _is_high_consequence(result):
        return []
    if (
        result.decision.route is SearchTier.SKIP
        and result.decision.skip_reason is not None
        and result.decision.skip_reason.value == "user_forbid_web"
    ):
        return [_NO_WEB_HIGH_CONSEQUENCE_WARNING]
    return [_HIGH_CONSEQUENCE_WARNING]
```

Do not add `route is not SKIP` or `route is DEEP` as a warning condition. Keep `_dedupe_strings()` and `_strip_markers()` so model-supplied duplicate fixed warnings are removed.

In the `SearchTier.SKIP` branch, make the no-web disclosure mutually exclusive:

```python
if decision.skip_reason and decision.skip_reason.value == "user_forbid_web":
    if not _is_high_consequence(result):
        disclosures.append(_NO_WEB_DYNAMIC_LIMIT)
disclosures.extend(_search_warning_disclosures(result))
```

This prevents a high-consequence no-web reply from containing both “本次没有联网核验” and “本次未联网核验”.

- [ ] **Step 5: Remove success status from the compatibility facade**

In `src/services/search_service.py`, replace:

```python
lines = ["搜索状态：success", f"搜索词：{normalized}", ""]
```

with:

```python
lines: list[str] = []
```

Keep each admitted Evidence entry as title/excerpt/URL. Do not add another success phrase. The actual `/search` command remains unchanged because `src/commands/search.py` already delegates to `generate_reply(force_search=True)`.

- [ ] **Step 6: Run output and end-to-end flow suites**

Run:

```powershell
python -B -m unittest tests.test_search_renderer tests.test_chat_retrieval_flow tests.test_product_scope tests.test_identity_configuration -v
```

Expected: low-risk success/failure has no professional-risk warning; high-consequence paths warn once; `/search` success contains answer and sources without a status banner; failure still discloses incomplete retrieval.

- [ ] **Step 7: Commit rendering and command-output behavior**

```powershell
git add src/search/renderer.py src/services/search_service.py tests/test_search_renderer.py tests/test_chat_retrieval_flow.py tests/test_product_scope.py
git commit -m "fix: scope search warnings to high risk"
```

---

### Task 6: Update Documentation and Run Final Acceptance

**Files:**
- Modify: `README.md:160-172`
- Modify: `.env.example:26-31`
- Modify: `tests/test_readme_guide.py`
- Modify: `tests/test_user_facing_scope.py`
- Verify: all search and integration tests

**Interfaces:**
- Consumes: the final DDGS-first provider contract and user-output matrix from Tasks 1-5.
- Produces: user documentation that matches production behavior and static tests that prevent old provider order or success banners from returning.

- [ ] **Step 1: Add failing documentation and static-copy tests**

In `tests/test_readme_guide.py`, assert the exact provider relationship:

```python
def test_readme_documents_ddgs_primary_and_tavily_fallback(self):
    self.assertIn("DDGS 是主搜索提供者", self.readme)
    self.assertIn("Tavily", self.readme)
    self.assertIn("回退", self.readme)
    self.assertNotIn("Tavily 作为主搜索提供者", self.readme)
```

In `tests/test_user_facing_scope.py`, scan user-facing production files:

```python
for relative in (
    "src/search/renderer.py",
    "src/services/search_service.py",
    "src/commands/search.py",
):
    source = (ROOT / relative).read_text(encoding="utf-8")
    self.assertNotIn("搜索状态：success", source)
```

The renderer constant `搜索结果可能不完整或不准确` is allowed, so do not ban the string globally. Its behavior is covered by renderer tests.

- [ ] **Step 2: Run documentation tests and verify RED**

Run:

```powershell
python -B -m unittest tests.test_readme_guide tests.test_user_facing_scope -v
```

Expected after Tasks 1-5: the README provider-order assertion fails because documentation still describes Tavily primary/DDGS fallback; the static user-facing copy assertion already passes because Task 5 removed the success banner.

- [ ] **Step 3: Update exact documentation copy**

Replace the `TAVILY_API_KEY` README description with:

```markdown
| `TAVILY_API_KEY` | 可选 | 空 | DDGS 是主搜索提供者；DDGS 不可用、出错、超时或无结果时，若配置了 Tavily Key，则使用 Tavily 回退。两者都不可用时，事实型请求明确返回检索失败，不会伪装成已经获得在线证据。 |
```

Update `.env.example` comments to:

```dotenv
# DDGS 是主搜索提供者。TAVILY_API_KEY 可选；配置后 Tavily 仅在 DDGS
# 不可用、出错、超时或无结果时作为回退。两者都失败时不得伪造在线证据。
TAVILY_API_KEY=
```

Add a README behavior paragraph:

```markdown
搜索成功时，普通聊天和 `/search` 都直接输出答案与实际使用的来源，不显示“搜索成功”或“检索完成”。“搜索结果可能不完整或不准确”的专业风险提示只用于医疗、法律、金融、安全等可能影响现实行动的高危请求；普通新闻、比赛、产品和技术搜索不显示该提示。搜索失败时仍会明确说明“在线检索未完成”。
```

- [ ] **Step 4: Run focused search and integration suites**

Run:

```powershell
python -B -m unittest tests.test_search_router tests.test_search_planner tests.test_search_providers tests.test_search_extraction tests.test_search_evidence tests.test_search_orchestrator tests.test_search_validation tests.test_search_renderer tests.test_chat_retrieval_flow -v
python -B -m unittest tests.test_health tests.test_product_scope tests.test_identity_configuration tests.test_main_image_flow tests.test_multimodal_chat tests.test_readme_guide tests.test_user_facing_scope -v
```

Expected: every test passes; no external HTTP is performed.

- [ ] **Step 5: Run static invariants**

Run:

```powershell
rg -n "搜索状态：success|Tavily 作为主搜索提供者" src README.md .env.example
rg -n "quality_flags=.*availability_fallback" src
git diff --check
```

Expected:

- first `rg` exits 1 with no matches;
- second `rg` exits 1 because no production hit is assigned an order-dependent quality flag;
- `git diff --check` exits 0.

- [ ] **Step 6: Run the full hermetic suite and compile check**

Run:

```powershell
python -B -m unittest discover -s tests -v
python -B -m compileall -q src tests
```

Expected: all tests pass, no real external provider is contacted, and compileall exits 0.

- [ ] **Step 7: Perform a final behavior review without online calls**

Using fake router advisors/providers, confirm this matrix in the final review report:

```text
下午好                                      -> SKIP, zero provider attempts
昨天天曼契约EDGVSTEC谁赢了                 -> DEEP
DDGS success                               -> Tavily not called
DDGS error + Tavily success                -> attempts [ddgs, tavily]
ordinary success                           -> answer + sources, no warning/status
ordinary failure                           -> online retrieval incomplete, no risk warning
high-consequence success/failure            -> exactly one risk warning
no-web high-consequence                     -> exactly one no-web warning
```

Do not run real Tavily/DDGS probes unless the user separately authorizes controlled online evaluation.

- [ ] **Step 8: Commit documentation and acceptance tests**

```powershell
git add README.md .env.example tests/test_readme_guide.py tests/test_user_facing_scope.py
git commit -m "docs: describe DDGS-first search behavior"
```

Record the exact commit table, test counts, static-check outputs, and any remaining external evaluation gates in the handoff. Leave the branch clean and unpushed unless the user separately requests integration or push.
