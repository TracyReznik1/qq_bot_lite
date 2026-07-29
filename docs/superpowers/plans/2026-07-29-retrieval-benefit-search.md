# Retrieval Benefit Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace model-decided first-search behavior with a program-owned Retrieval Benefit Router and a provider-neutral, evidence-grounded search pipeline in which factual requests default to search, search failures degrade explicitly, and every displayed citation is deterministically tied to evidence obtained in the current request.

**Architecture:** Introduce a new `src/search/` vertical slice that owns routing, query planning, provider adapters, safe content extraction, Evidence assembly, the initial-query plus optional single-repair state machine, claim validation, deterministic citations, and QQ rendering. Ordinary chat and `/search` enter the same pipeline before answer generation. `src/services/search_service.py` remains only as a compatibility facade; it no longer controls chat search behavior.

**Tech Stack:** Python 3.13, standard-library dataclasses/enums/concurrency/JSON/logging, `unittest`, Flask/OneBot, existing provider-neutral LLM client, Tavily, DDGS, `requests`, and `pypdf` for text PDFs.

**Confirmed specification:** `docs/superpowers/specs/2026-07-29-retrieval-benefit-search-design.md` at commit `e2c5848`.

## Global Constraints

- Execute implementation in an isolated worktree on branch `codex/retrieval-benefit-search`; do not mix unrelated user changes into the branch.
- Treat the confirmed specification as authoritative. If an implementation detail conflicts with it, stop and amend this plan/spec through review instead of silently changing product behavior.
- Search execution is program-owned. The model may classify benefits, plan queries, identify gaps, or recommend an upgrade, but it may never lower the program floor or choose an unrecognized `skip_reason`.
- A request that is ambiguous, invalidly classified, or missing a legal skip reason routes to at least `light`.
- User memory, chat history, model confidence, “common knowledge,” and training knowledge are never inputs that can justify `skip`.
- Only the current user question and privacy-safe query metadata may reach search providers. Do not send retrieved memories, prior chat turns, image data URLs, QQ IDs, group IDs, callback secrets, or provider keys.
- Relevant Evidence outranks model memory. Model memory cannot create a hidden conflict; only conflicts between recorded Evidence items enter conflict handling.
- Query relevance is a hard Evidence admission gate. First-party/source relationship is evaluated only after relevance and cannot rescue an irrelevant page.
- One parallel initial query batch is one retrieval round. `standard` and `deep` may run at most one adaptive repair query after initial Evidence gap analysis. `light` has no repair. Answer generation or validation may never initiate another search.
- Deterministic structure and failure behavior are tested as 100% invariants. Claim discovery, semantic support, and relevance judging are measured against human labels with precision/recall/F1; never treat a validator model’s self-report as ground truth.
- Unit tests must remain hermetic under `tests/runtime.py`; mock provider/HTTP/LLM boundaries and never make live network calls during `python -m unittest`.
- Keep existing image handling, persona, memory lifecycle, OneBot ordering, command rendering, and generic provider tool-call compatibility intact unless a task below explicitly changes the integration surface.
- Do not add a feature flag that restores “default no search” inside the new pipeline. Rollback is release/commit rollback only.
- Every production edit follows red-green-refactor: add the focused failing test, run it and confirm the expected failure, implement the smallest complete behavior, rerun focused tests, then commit.

---

## Final Module and File Map

New production files:

```text
src/search/
  __init__.py
  models.py
  router.py
  planner.py
  extraction.py
  evidence.py
  orchestrator.py
  validation.py
  renderer.py
  providers/
    __init__.py
    base.py
    tavily.py
    ddgs.py
```

New test/evaluation files:

```text
tests/search_fakes.py
tests/test_search_models.py
tests/test_search_router.py
tests/test_search_planner.py
tests/test_search_providers.py
tests/test_search_extraction.py
tests/test_search_evidence.py
tests/test_search_orchestrator.py
tests/test_search_validation.py
tests/test_search_renderer.py
tests/test_chat_retrieval_flow.py
tests/test_search_evaluation.py
eval/search/README.md
eval/search/cases.jsonl
eval/search/provider_recordings.jsonl
eval/search/model_predictions.jsonl
tools/evaluate_search.py
```

Modified compatibility/integration files:

```text
requirements.txt
.env.example
README.md
src/config.py
src/services/url_fetch_service.py
src/services/search_service.py
src/chat/prompt.py
src/chat/chat_service.py
src/commands/search.py
src/commands/__init__.py
src/main.py
tests/runtime.py
tests/test_chat_tool_finalization.py        # removed after replacement
tests/test_command_renderer.py
tests/test_health.py
tests/test_identity_configuration.py
tests/test_main_image_flow.py
tests/test_multimodal_chat.py
tests/test_product_scope.py
tests/test_readme_guide.py
```

## Canonical Contracts

All tasks use these names and ownership boundaries. Do not create parallel ad-hoc dictionaries for the same state.

| Contract | Owner | Required fields/behavior |
|---|---|---|
| `RetrievalRequest` | `models.py` | `question`, `force_search`, `has_images`, closed `request_source`; it deliberately has no memory/history field |
| `RetrievalDecision` | `models.py` | the confirmed route contract, closed enums, program floor, model recommendation, auditable reason codes |
| `TierBudget` | `models.py` | initial-query, candidate-URL, content-read, repair-query, total-query, round, and hard-timeout ceilings |
| `SearchQuery` / `SearchPlan` / `RepairPlan` | `models.py` | original natural-language question, purpose, round kind, entity/time metadata, planning status, and the single optional repair decision |
| `ProviderReadiness` / `ProviderHit` / `ProviderResult` / `ProviderAttempt` | `models.py` | configuration/availability and provider metadata preserved without assigning Evidence authority |
| `FetchedDocument` / `EvidenceCandidate` | `models.py` | requested/final URL, content type, title, bounded excerpt, origin, fetch status, untrusted-content flags |
| `EvidenceItem` / `EvidenceBundle` / `EvidenceGapAnalysis` | `models.py` | relevance, source relation, dates, supported/weak/missing topics, limitations, conflict groups, and sufficiency state |
| `GroundedDraft` / `Claim` / `AnswerBlock` / `ValidationReport` | `models.py` | atomic answer blocks, claim-to-Evidence mappings, semantic labels, known-failure removal |
| `SearchTrace` / `SearchPipelineResult` | `models.py` | route/start/attempt/sufficiency denominators, exact latency boundaries, failure code, round/query budgets |
| `RenderedReply` | `models.py` | final text, precomputed QQ chunks, used Evidence IDs, shown source URLs, degradation disclosures |

The only route values are `skip`, `light`, `standard`, and `deep`. The only legal skip reasons are:

```text
user_forbid_web
social_or_emotional
creative_or_roleplay
provided_text_transform
provided_content_summary
pure_math
closed_logic
closed_context_only
```

Default tier budgets are immutable:

| Tier | Initial queries | Candidate URLs | Content reads | Repair queries | Total semantic queries | Retrieval rounds | Hard timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| `light` | 1 | 5 | 2 | 0 | 1 | 1 | 8 s |
| `standard` | 3 | 8 | 5 | 1 | 4 | 2 | 20 s |
| `deep` | 5 | 15 | 8 | 1 | 6 | 2 | 40 s |

---

## Task 0: Establish the Implementation Baseline

**Files:** None.

- [ ] Invoke `superpowers:using-git-worktrees` and create an isolated worktree on `codex/retrieval-benefit-search` from the commit containing this plan. Confirm that commit’s specification parent/ancestor includes `e2c5848`; do not branch from `e2c5848` alone because that would omit the plan.
- [ ] Invoke `superpowers:test-driven-development` before Task 1. If any failure is not the expected red-test failure, invoke `superpowers:systematic-debugging` before changing implementation.
- [ ] In the worktree, run `git status --short` and confirm it is empty. If it is not empty, stop and resolve ownership of the changes before continuing.
- [ ] Run the unchanged baseline:

```powershell
python -m unittest discover -s tests -t . -v
```

Expected result at planning time: `Ran 397 tests` and `OK`.

- [ ] Record the exact Python and dependency snapshot:

```powershell
python --version
python -m pip show tavily-python ddgs requests
```

- [ ] Do not create a baseline commit. The first commit belongs to Task 1.

---

## Task 1: Add Closed Data Contracts and Budget Invariants

**Files:**

- Create: `src/search/__init__.py`
- Create: `src/search/models.py`
- Create: `tests/search_fakes.py`
- Create: `tests/test_search_models.py`

**Required enum families:**

```python
from enum import StrEnum


class SearchTier(StrEnum):
    SKIP = "skip"
    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"


class SkipReason(StrEnum):
    USER_FORBID_WEB = "user_forbid_web"
    SOCIAL_OR_EMOTIONAL = "social_or_emotional"
    CREATIVE_OR_ROLEPLAY = "creative_or_roleplay"
    PROVIDED_TEXT_TRANSFORM = "provided_text_transform"
    PROVIDED_CONTENT_SUMMARY = "provided_content_summary"
    PURE_MATH = "pure_math"
    CLOSED_LOGIC = "closed_logic"
    CLOSED_CONTEXT_ONLY = "closed_context_only"


class SearchRoundKind(StrEnum):
    INITIAL = "initial"
    REPAIR = "repair"


class EvidenceState(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"
```

Also define the exact confirmed values for `RequestSource`, `TriggerCode`, `BenefitDimension`, `Factuality`, `Freshness`, `RiskLevel`, `Actionability`, `PotentialHarm`, `QueryPurpose`, `PlanningStatus`, `ProviderStatus`, `ExcerptOrigin`, `CandidateRelevance`, `SourceRelation`, `SupportLabel`, and `SearchFailureCode`. In particular:

```text
RequestSource: chat | command | compatibility
BenefitDimension: accuracy | freshness | completeness | verifiability | disambiguation | risk_control
Factuality: non_factual | factual | mixed | ambiguous
Freshness: none | low | high
RiskLevel: low | medium | high
Actionability: none | general | personalized
PotentialHarm: none | low | high
QueryPurpose: direct | primary | independent | time_bounded | disambiguation | repair
PlanningStatus: normal | degraded
ExcerptOrigin: provider_snippet | page_extract | document_extract
SourceRelation: primary | independent | secondary | community | unknown
SupportLabel: supported | partial | conflict | unsupported | unmapped
SearchFailureCode:
  provider_not_configured | provider_unavailable | provider_timeout |
  no_results | content_unreadable | insufficient_evidence |
  partial_evidence | source_conflict | validation_failed | user_forbid_web
```

Unknown strings must fail parsing rather than becoming enum members. Provider runtime statuses may additionally distinguish `success`, `empty`, `timeout`, `error`, `not_configured`, and `unavailable`; they map to the closed user-visible failure codes above.

`TriggerCode` must contain exactly:

```text
explicit_no_web | explicit_search | explicit_verification |
explicit_source_request | freshness_marker | dynamic_attribute |
regulated_domain_foundation | high_consequence_action |
current_rule_or_policy | controversy_or_conflict |
external_fact_explanation_or_comparison | recommendation_or_evaluation |
ambiguous_entity | multi_hop_complexity | mixed_task |
factual_default | classifier_uncertain
```

**Required field sets:**

```text
RetrievalRequest:
  question, force_search, has_images, request_source

RetrievalDecision:
  route, skip_reason, forced_search, trigger_codes, benefit_dimensions,
  factuality, external_fact_required, freshness, risk, actionability,
  potential_harm, program_minimum_tier, model_recommended_tier,
  final_reason_codes

TierBudget:
  max_initial_queries, max_candidate_urls, max_content_reads,
  max_repair_queries, max_total_queries, max_retrieval_rounds,
  hard_timeout_seconds

SearchQuery:
  query_id, round_kind, purpose, text, date_from, date_to,
  include_domains, exclude_domains

SearchPlan:
  decision, original_question, planning_status, entities, time_window,
  initial_queries, required_topics, required_source_relations,
  query_redaction_codes, budget

RepairPlan:
  triggered, gap_codes, repair_query

ProviderHit:
  provider, query_id, title, url, snippet, score, published_at,
  raw_content, quality_flags

EvidenceItem:
  evidence_id, query_id, provider, title, url, canonical_url, domain,
  publisher, source_relation, source_relation_basis, published_at,
  retrieved_at, excerpt,
  excerpt_origin, extraction_status, provider_score, relevance_score,
  relevance_gate_passed, freshness_state, citable, safety_flags,
  supported_topics, independence_group

EvidenceBundle:
  request_id, decision, plan, attempts, initial_evidence_ids,
  gap_analysis, repair_plan, retrieval_round_count, evidence_items,
  evidence_state, missing_claim_topics, weak_source_topics,
  conflict_groups, limitations

SearchPipelineResult:
  decision, plan (optional for skip), evidence (optional for skip/provider
  configuration failure), trace, failure_code (optional on success)
```

**Core invariant implementation:**

```python
_TIER_RANK = {
    SearchTier.SKIP: 0,
    SearchTier.LIGHT: 1,
    SearchTier.STANDARD: 2,
    SearchTier.DEEP: 3,
}


def max_tier(left: SearchTier, right: SearchTier) -> SearchTier:
    return left if _TIER_RANK[left] >= _TIER_RANK[right] else right
```

`RetrievalDecision.__post_init__` must reject:

- `skip` without a legal `skip_reason`;
- a search tier with any `skip_reason`;
- a search tier without `program_minimum_tier`;
- a final route below `program_minimum_tier`;
- `program_minimum_tier=skip`;
- free-text reason or trigger values;
- `forced_search=True` with `skip`, except the explicit no-web/search conflict represented by both closed trigger codes.

Expose a computed `requires_clarification` property for that conflict; do not add a fifth route.

**Trace requirements:**

`SearchTrace.to_log_dict()` must preserve the specification’s exact field names and include body-free metadata only:

```text
request_id
request_source
route
skip_reason
trigger_codes
factuality
external_fact_required
program_minimum_tier
final_tier
orchestrator_started
initial_query_count
initial_round_started
adaptive_repair_round_started
adaptive_repair_query
retrieval_round_count
executed_queries
provider_configured
provider_attempts
provider_invocation_started
provider_failures
candidate_url_count
citable_evidence_count
evidence_state
repair_used
claim_count
supported_claim_count
citation_count
knowledge_fallback_used
degradation_reason
route_latency_ms
query_planning_latency_ms
initial_provider_search_latency_ms
provider_search_total_latency_ms
initial_content_read_latency_ms
content_read_total_latency_ms
initial_evidence_assembly_latency_ms
evidence_assembly_total_latency_ms
gap_analysis_latency_ms
adaptive_repair_latency_ms
answer_generation_latency_ms
structural_validation_latency_ms
semantic_validation_latency_ms
qq_render_latency_ms
retrieval_pipeline_latency_ms
total_response_latency_ms
```

`executed_queries` and `adaptive_repair_query` store query IDs and purposes, not raw text. `provider_attempts` stores provider/status/count/latency metadata, not exception text. The Trace must not include question text, memory text, answer text, Evidence excerpts, URLs, API keys, QQ IDs, group IDs, or image data.

Expose these body-free derived values in the serialized log without replacing the specification fields:

```text
semantic_query_count = count of distinct query IDs in executed_queries
repair_query_count = 1 when adaptive_repair_round_started else 0
content_read_count = count of provider-native/page/document reads consumed
provider_attempted = provider_invocation_started
sufficient_evidence = evidence_state == sufficient
```

- [ ] Write `tests/test_search_models.py` first. Cover valid decisions for all four tiers, each illegal combination above, immutable budgets, `max_tier`, trace redaction, and JSON-safe log serialization.
- [ ] Run:

```powershell
python -m unittest tests.test_search_models -v
```

Expected failure: `ModuleNotFoundError: No module named 'src.search'`.

- [ ] Implement every enum and dataclass in `models.py`, including a body-free `ProviderReadiness(provider, configured, available, reason_code)` record. Use frozen dataclasses for immutable request/decision/evidence values; keep `SearchTrace` mutable because stages fill timings and counters.
- [ ] Put the exact immutable `DEFAULT_TIER_BUDGETS` mapping in `models.py`. Derived totals must be validated in `TierBudget.__post_init__`.
- [ ] Add reusable fakes in `tests/search_fakes.py`: `StaticRouterAdvisor`, `RecordingProvider`, `StaticEvidenceJudge`, `StaticSemanticVerifier`, and `FakeClock`. They must never import a real provider SDK or make HTTP calls.
- [ ] Export only the public contracts and later `get_search_orchestrator` accessor from `src/search/__init__.py`; do not construct a singleton yet.
- [ ] Rerun the focused test and then:

```powershell
python -m unittest tests.test_search_models tests.test_model_config -v
```

Expected result: all selected tests pass.

- [ ] Commit:

```powershell
git add src/search/__init__.py src/search/models.py tests/search_fakes.py tests/test_search_models.py
git commit -m "feat: define evidence search contracts"
```

---

## Task 2: Implement the Retrieval Benefit Router

**Files:**

- Create: `src/search/router.py`
- Create: `tests/test_search_router.py`
- Modify: `tests/search_fakes.py`

**Public interface:**

```text
RetrievalBenefitRouter(advisor, *, clock=None)
RetrievalBenefitRouter.decide(request: RetrievalRequest) -> RetrievalDecision
```

The advisor receives only `request.question`, `request.has_images`, and closed enum schemas. It does not receive chat history, retrieved memory, an answer draft, or model confidence. Its JSON contract is:

```json
{
  "skip_candidate": null,
  "benefit_dimensions": ["accuracy"],
  "factuality": "mixed",
  "external_fact_required": true,
  "freshness": "low",
  "risk": "low",
  "actionability": "none",
  "potential_harm": "none",
  "recommended_tier": "standard",
  "trigger_codes": ["external_fact_explanation_or_comparison"]
}
```

Missing fields, malformed JSON, unknown enums, contradictions, timeout, or advisor exceptions are classifier uncertainty and must return at least `light`.

**Program decision order:**

1. Detect explicit no-web.
2. Detect explicit search/verification/source/current requirements.
3. Establish dynamic/high-consequence/current-rule/controversy/complexity floors.
4. Validate a complete closed-task match for a skip candidate.
5. Default every remaining factual, mixed, or ambiguous request to `light`.
6. Raise external-fact explanations/comparisons and stable regulated-domain concepts to `standard`.
7. Raise dynamic or personalized high-consequence requests to `deep`.
8. Apply `max_tier(program_floor, model_recommendation)`.

The closed-task validator must be conservative:

- a transform/summary skip requires user-provided source text or an explicit current-context reference;
- pure math/logic requires all premises in the request;
- creative/social/image-only closed-context skips are rejected when the request also asks for real facts, verification, sources, current state, or named-entity background;
- question words such as “为什么”, “解释”, and “区别” are not sufficient for `standard`;
- medical/legal/financial/security words are not sufficient for `deep`;
- temporal words such as “今天” or “最近” trigger freshness only when they modify an external fact, event, price, version, rule, service, or public status; “今天心情不好” remains social/emotional;
- mixed tasks cannot skip.

**Required routing fixture rows in `tests/test_search_router.py`:**

| Request | Expected |
|---|---|
| `你好，今天心情有点差` | `skip/social_or_emotional` |
| `把这段话润色一下：天气很好` | `skip/provided_text_transform` |
| `解释我刚才贴的这段文字` | `skip/closed_context_only` |
| `证明若 n 为偶数则 n² 为偶数` | `skip/pure_math` |
| `什么是光合作用` | `light` |
| `Rust 和 Go 的并发模型有什么区别` | at least `standard` |
| `为什么我刚才那句话显得生硬` | closed-context skip |
| `什么是股票` | `standard` |
| `我现在是否应该买这只股票` | `deep` |
| `什么是民法` | `standard` |
| `我的具体情况是否违法` | `deep` |
| `北京今天有什么新闻` | `deep` |
| `请搜索并给出来源：CRAG 是什么` | at least `standard`, forced |
| `不要联网，只根据我贴的内容总结` | `skip/user_forbid_web`, zero provider eligibility |
| `请联网查，但不要联网` | no provider; clarification conflict |
| `润色这段 GPT 价格介绍，并核实价格` | search, not transform skip |
| unknown entity or invalid advisor JSON | at least `light` |

Add adversarial advisor outputs containing “I know this,” “high confidence,” “common knowledge,” an unknown skip reason, a lower tier, and an alleged memory conflict. None may lower or skip the program route.

- [ ] Add the table-driven and adversarial tests first.
- [ ] Run:

```powershell
python -m unittest tests.test_search_router -v
```

Expected failure: import failure for `src.search.router`.

- [ ] Implement deterministic signal detection and closed skip validation as pure functions with named reason codes. Do not log free-text model reasoning.
- [ ] Implement `LLMRoutingAdvisor` with temperature `0`, bounded output, strict JSON extraction, and closed-enum validation.
- [ ] Ensure `request.force_search=True` or an explicit search/verification/source marker sets `decision.forced_search=True`. A plain explicit search establishes at least `light`; explicit verification/source establishes at least `standard`; current/dynamic/high-consequence triggers remain `deep`.
- [ ] Assert in tests that the advisor capture contains no memory/history/private identifiers.
- [ ] Run:

```powershell
python -m unittest tests.test_search_router tests.test_search_models -v
```

Expected result: all selected tests pass.

- [ ] Commit:

```powershell
git add src/search/router.py tests/search_fakes.py tests/test_search_router.py
git commit -m "feat: route by retrieval benefit"
```

---

## Task 3: Build Natural-Language Query Planning and One Repair Contract

**Files:**

- Create: `src/search/planner.py`
- Create: `tests/test_search_planner.py`
- Modify: `tests/search_fakes.py`

**Public interface:**

```text
SearchPlanner(model, *, today_provider=None)
SearchPlanner.plan(
    request: RetrievalRequest,
    decision: RetrievalDecision,
) -> SearchPlan
SearchPlanner.plan_repair(
    plan: SearchPlan,
    gap: EvidenceGapAnalysis,
) -> RepairPlan
```

Planner rules:

- `light` always uses exactly one `direct` query equal to the user’s normalized natural-language question after mandatory privacy redaction. No model call is required.
- `standard` emits at most three initial queries selected from `direct`, `primary`, `independent`, and `disambiguation`.
- `deep` emits at most five initial queries and must include a dated/time-bounded purpose when freshness is high.
- Chinese/Japanese queries keep the original natural-language question as the direct query. Entity/time/intent extraction creates supplementary queries; it never replaces the original with mechanical tokens.
- Before any provider call, remove CQ control codes, data URLs, credentials, one-time codes, API keys, callback secrets, and hidden/system identifiers. Record only closed `query_redaction_codes`.
- Email addresses, telephone numbers, and exact addresses may enter a query only when the user explicitly asks in the current message to search/verify that exact value; hard secrets are never allowed even with such wording.
- If mandatory redaction removes the only value, replace it with a safe intent query such as `敏感凭据泄露后的安全处置`, record `planning_status=degraded`, and never send the removed value.
- Validate include/exclude domains as bare public hostnames, deduplicate them, reject URLs/private/local names, and cap each list at five entries.
- Cap each provider query at 500 Unicode characters after privacy redaction while preserving the untruncated `original_question` internally; prefer a sentence boundary when truncation is required.
- A failed/invalid planning model produces `planning_status=degraded` and deterministic queries without lowering the route.
- Repair is permitted only for `standard` or `deep`, only when the initial bundle reports a concrete missing topic or Evidence conflict, and returns a `RepairPlan` with `triggered=True`, `gap_codes=gap.repair_reason_codes`, and exactly one distinct `repair_query`.
- `plan_repair` returns a `RepairPlan` with `triggered=False`, the current `gap.repair_reason_codes`, and `repair_query=None` if a repair was already planned, the route is `light`, the gap is empty, or the proposed query duplicates any initial query after Unicode/case/space normalization.

The deterministic fallback for `北京今天有什么新闻` on 2026-07-29 must preserve:

```text
direct: 北京今天有什么新闻
time_bounded: 北京 2026-07-29 新闻 重要事件
primary: 北京 2026-07-29 官方 通报
independent: 北京 2026-07-29 新闻 重要事件 独立报道
```

For a standard technical comparison fallback, preserve the original and add:

```text
primary: Rust 和 Go 的并发模型有什么区别 Rust 官方文档 Go 官方文档
independent: Rust 和 Go 的并发模型有什么区别 独立技术对比
```

- [ ] Write tests for all tier counts, original CJK query preservation, entity/time/intent fields, secret/CQ/data-URL redaction, explicit versus implicit personal-identifier handling, public-domain-list validation/caps, deterministic degradation, duplicate rejection, one repair, light no-repair, and total semantic-query ceilings.
- [ ] Run:

```powershell
python -m unittest tests.test_search_planner -v
```

Expected failure: import failure for `src.search.planner`.

- [ ] Implement strict planner JSON parsing. A model may choose fewer queries but never exceed the budget or invent a `QueryPurpose`.
- [ ] Normalize only transport hazards and whitespace. Do not delete “什么”, “为什么”, “区别”, temporal intent, or natural sentence structure.
- [ ] Store `original_question` separately from provider query strings and never mutate it.
- [ ] Add a test proving memory text placed in a fake surrounding context never appears in a `SearchPlan`.
- [ ] Run:

```powershell
python -m unittest tests.test_search_planner tests.test_search_router -v
```

Expected result: all selected tests pass.

- [ ] Commit:

```powershell
git add src/search/planner.py tests/search_fakes.py tests/test_search_planner.py
git commit -m "feat: plan bounded search queries"
```

---

## Task 4: Introduce Provider-Neutral Tavily and DDGS Adapters

**Files:**

- Create: `src/search/providers/__init__.py`
- Create: `src/search/providers/base.py`
- Create: `src/search/providers/tavily.py`
- Create: `src/search/providers/ddgs.py`
- Create: `tests/test_search_providers.py`
- Modify: `tests/search_fakes.py`

**Provider interface:**

```text
SearchProvider.name: str
SearchProvider.readiness() -> ProviderReadiness
SearchProvider.search(
    query: SearchQuery,
    *,
    tier: SearchTier,
    max_results: int,
    timeout_seconds: float,
) -> ProviderResult
```

`ProviderRegistry` owns the ordered adapters. It reports readiness without secrets, selects Tavily first when configured, and uses DDGS only as the availability fallback when Tavily is not configured, unavailable, errors, times out, or returns no results. A fallback call for the same `SearchQuery.query_id` is still one semantic query and one retrieval round but creates a separate `ProviderAttempt`.

Adapter rules:

- `TavilySearchProvider` uses `basic` for `light`, `advanced` for `standard/deep`, and requests raw content for `standard/deep`.
- `timeout_seconds` is the smaller of `config.request_timeout` and the tier deadline remaining. If an SDK cannot apply it directly, the registry still enforces the future deadline and records timeout without waiting for answer generation.
- Provider result count is `min(max_results, max(config.search_max_results, 1))`; the orchestrator later enforces aggregate tier URL budgets.
- Preserve title, URL, snippet/content, provider score, published date, raw content, provider name, and query ID.
- Map `SearchQuery.date_from`, `date_to`, `include_domains`, and `exclude_domains` to Tavily’s corresponding supported parameters; a news/time-bounded purpose also selects the provider’s news topic where supported.
- `DDGSSearchProvider` preserves available title/body/href/date fields and leaves unsupported score/raw-content fields as `None`.
- Every DDGS hit records the body-free quality flag `availability_fallback`; it remains weak until relevance and content support are established and is never treated as primary merely because it was the fallback.
- Neither adapter assigns `relevance`, `source_relation`, “official,” Evidence IDs, or citation numbers.
- Distinguish `not_configured`, `unavailable`, `timeout`, `error`, `empty`, and `success`.
- Exception messages and API keys never enter `ProviderResult`, Trace, or logs. Log only provider, status, exception type, and elapsed time.

- [ ] Write fake-SDK tests first. Assert exact Tavily depth/raw-content/date/domain/topic behavior, DDGS field normalization and weak fallback flag, metadata preservation, proxy/timeout use, fallback order, separate attempts, no fallback after usable Tavily hits, and readiness with no installed/configured provider.
- [ ] Add an assertion that a URL containing `/docs` remains relation-unknown at this layer.
- [ ] Run:

```powershell
python -m unittest tests.test_search_providers -v
```

Expected failure: provider package modules do not exist.

- [ ] Implement `ProviderRegistry` and adapters. Catch SDK exceptions only at the adapter boundary and translate them to closed statuses.
- [ ] Keep SDK imports optional so a missing optional package becomes `unavailable`, not an import-time crash.
- [ ] Use one monotonic timing measurement per actual adapter call and populate `ProviderAttempt`.
- [ ] Run:

```powershell
python -m unittest tests.test_search_providers tests.test_product_scope -v
```

At this stage, legacy `test_product_scope` must still pass because `search_service.py` has not been changed.

- [ ] Commit:

```powershell
git add src/search/providers tests/search_fakes.py tests/test_search_providers.py
git commit -m "feat: add neutral search providers"
```

---

## Task 5: Refactor Safe Page Reading into Structured Extraction

**Files:**

- Modify: `requirements.txt`
- Modify: `src/services/url_fetch_service.py`
- Create: `src/search/extraction.py`
- Create: `tests/test_search_extraction.py`
- Modify: `tests/test_product_scope.py`

**Service compatibility contract:**

`src/services/url_fetch_service.py` gains:

```python
@dataclass(frozen=True)
class UrlDocumentResult:
    ok: bool
    status: str
    requested_url: str
    final_url: str
    title: str
    content_type: str
    text: str
```

and the structured entry point:

```text
fetch_document(url: str, *, timeout_seconds: float | None = None) -> UrlDocumentResult
```

The existing `fetch_url(text: str) -> UrlFetchResult` remains as a compatibility wrapper that extracts the first URL, calls `fetch_document`, and formats the legacy text result. New search code must never parse that formatted text.

**Safety and extraction behavior:**

- Preserve HTTP(S)-only, DNS/public-IP validation, redirect revalidation, redirect count, byte limit, text limit, proxy, timeout, and user-agent protections.
- Close every response in `finally`, including redirect, oversize, unsupported content type, parse failure, and success paths.
- Treat `Content-Length` as an early rejection only; enforce the byte cap while streaming even when the header is missing or false.
- Add `application/pdf` and `application/x-pdf`; use `pypdf.PdfReader` on bounded bytes, extract text page by page, and apply the existing text bound.
- Keep HTML/text/JSON extraction. Return final URL and page title structurally.
- Do not execute JavaScript, follow meta refresh, load embedded resources, or accept non-public redirect targets.

**Search extraction interface:**

```text
SearchExtractor.extract(
    hit: ProviderHit,
    query: SearchQuery,
    *,
    allow_network_read: bool,
    timeout_seconds: float,
) -> EvidenceCandidate
```

Rules:

- A bounded provider `raw_content` or result snippet uses `excerpt_origin=provider_snippet`; `extraction_status` distinguishes `provider_raw_content` from `search_result_snippet`.
- Extracted HTML/plain/JSON text uses `excerpt_origin=page_extract`; extracted PDF text uses `excerpt_origin=document_extract`.
- A usable provider-native raw-content read counts toward `content_read_count`; fetching the same hit after using raw content would be a second read and is forbidden unless the raw content is empty/unusable.
- Rank fetch candidates by direct query/entity/time overlap and provider score, not by “official-looking” domain/path.
- Select query-aware excerpts by sentence/paragraph relevance using planner entities, time range, and intent. Do not reduce CJK text to whitespace tokens only.
- Flag likely prompt-injection instructions such as attempts to change system rules, expose secrets, or call tools. Keep the factual excerpt as untrusted data; never promote those instructions into a system message.
- Bound every excerpt and remove control characters before it reaches Evidence or an LLM.

- [ ] Write tests first for: structured success, requested/final URL, title, redirect revalidation, private-IP rejection, unsupported protocol/content type, stream byte cap, guaranteed `close()`, HTML script removal, text PDF extraction through a mocked `PdfReader`, provider raw-content preference, query-aware CJK excerpt selection, and prompt-injection flagging.
- [ ] Update the legacy URL tests in `tests/test_product_scope.py` to assert `fetch_url` still formats a compatible user-readable result.
- [ ] Run:

```powershell
python -m unittest tests.test_search_extraction tests.test_product_scope -v
```

Expected failure: missing `UrlDocumentResult`, `fetch_document`, and `src.search.extraction`.

- [ ] Add `pypdf>=5.0.0` to `requirements.txt`, then install the updated requirements in the implementation worktree’s active environment:

```powershell
python -m pip install -r requirements.txt
```

If package download requires network authorization, request it rather than bypassing the dependency or vendoring an unreviewed parser.

- [ ] Refactor URL fetching without weakening any existing SSRF check. Keep DNS validation on every redirect.
- [ ] Implement the extractor and make content-read accounting explicit in its result.
- [ ] Rerun focused tests, then run:

```powershell
python -m unittest tests.test_test_runtime tests.test_image_input_service tests.test_search_extraction tests.test_product_scope -v
```

Expected result: all selected tests pass and no real HTTP request is attempted.

- [ ] Commit:

```powershell
git add requirements.txt src/services/url_fetch_service.py src/search/extraction.py tests/test_search_extraction.py tests/test_product_scope.py
git commit -m "feat: extract structured web content safely"
```

---

## Task 6: Assemble Relevance-Gated Evidence and Conflicts

**Files:**

- Create: `src/search/evidence.py`
- Create: `tests/test_search_evidence.py`
- Modify: `tests/search_fakes.py`

**Public interface:**

```text
EvidenceAssembler.assemble(
    plan: SearchPlan,
    candidates: Sequence[EvidenceCandidate],
    *,
    previous: EvidenceBundle | None = None,
) -> EvidenceBundle

EvidenceAssembler.analyze_gap(
    plan: SearchPlan,
    bundle: EvidenceBundle,
) -> EvidenceGapAnalysis
```

`EvidenceJudge` receives only the question/plan and bounded candidate metadata/excerpts. It returns one closed record per candidate:

```json
{
  "candidate_id": "C1",
  "relevance": "direct",
  "source_relation": "primary",
  "publisher_entity_match": true,
  "ownership_basis": "The page publisher is the entity named in the query",
  "supported_topics": ["release date"],
  "conflict_key": null
}
```

**Admission and ranking rules:**

1. Normalize/canonicalize the final URL; deduplicate redirects, fragments, default ports, and equivalent trailing slashes.
2. Group near-identical syndicated excerpts or explicit canonical-source copies into one `independence_group`; multiple domains in that group do not count as independent corroboration.
3. Evaluate direct query relevance.
4. Exclude `unrelated` candidates before assigning Evidence IDs.
5. Among `direct` candidates, use source relationship and recency for ordering.
6. A candidate may be `primary` only when the judge records both `publisher_entity_match=true` and a non-empty query-specific ownership basis. `docs.*`, `developer.*`, `.gov`, `.edu`, `/docs`, and “official” in a title are never sufficient by themselves.
7. When judging fails, use a conservative deterministic relevance fallback and assign `source_relation=unknown`; never upgrade to `primary`.
8. Assign stable `E1`, `E2`, and subsequent IDs only after admission, deduplication, and ordering.

**Sufficiency and conflict rules:**

- `required_topics` come from the plan, not from model memory.
- `sufficient` means all required topics have directly relevant support and there is no unresolved material conflict.
- `partial` lists exact `missing_claim_topics`; final generation may not answer those topics.
- `conflicting` contains two or more actual Evidence IDs with incompatible current/date/version/claim values.
- Conflict members retain publication dates and an explicit `contradicts` or `claims_supersession` relation. A newer date alone does not silently erase the older source; the renderer states any claimed update relationship.
- `insufficient` means no directly relevant support for a material topic.
- Internal model knowledge is not passed to conflict detection and cannot create a `conflict_group`.
- Published time, provider-reported time, and retrieval time remain separate fields.
- A material dynamic/high-consequence conclusion normally requires a directly relevant primary source and an independent domain. When only one directly relevant authoritative source exists, it may be sufficient only with `single_source_authority` recorded in `EvidenceBundle.limitations`; the renderer must expose that limitation.
- Provider title text alone is never citable. A directly relevant provider snippet may support a low-risk stable claim, but dynamic/high-consequence sufficiency requires readable provider raw content or extracted page/document text.
- DDGS fallback snippets cannot alone satisfy a dynamic/high-consequence topic unless a readable underlying page passes the relevance/support checks.

- [ ] Write tests first for URL canonicalization/deduplication, syndicated-content independence grouping, relevance admission, source ordering only after relevance, unknown relation fallback, date separation, topic coverage, partial state, conflicting state, explicit supersession relationship, deep cross-check requirements, single-authority limitation, weak DDGS handling, and injection flags retained as untrusted metadata.
- [ ] Include these required adversarial cases:
  - an irrelevant product homepage marked primary versus a directly relevant independent release note: only the latter supports the release claim;
  - `https://unrelated.example/docs` cannot become first-party;
  - a model-memory statement disagrees with one valid Evidence item: no conflict group is created;
  - two recorded Evidence items disagree on a version/date: a conflict group is created.
- [ ] Run:

```powershell
python -m unittest tests.test_search_evidence -v
```

Expected failure: import failure for `src.search.evidence`.

- [ ] Implement deterministic canonicalization before any model-assisted judgement.
- [ ] Implement one batch `LLMEvidenceJudge` call per Evidence assembly, strict JSON parsing, and deterministic fallback for invalid/missing candidate rows.
- [ ] Ensure the prompt explicitly marks excerpts as untrusted data and excludes memory/history.
- [ ] Implement `EvidenceGapAnalysis` with `missing_claim_topics`, `conflict_group_ids`, `repair_eligible`, and one `repair_purpose`.
- [ ] Run:

```powershell
python -m unittest tests.test_search_evidence tests.test_search_extraction tests.test_search_planner -v
```

Expected result: all selected tests pass.

- [ ] Commit:

```powershell
git add src/search/evidence.py tests/search_fakes.py tests/test_search_evidence.py
git commit -m "feat: assemble relevance-gated evidence"
```

---

## Task 7: Implement the Bounded Two-Stage Search Orchestrator

**Files:**

- Create: `src/search/orchestrator.py`
- Create: `tests/test_search_orchestrator.py`
- Modify: `src/search/__init__.py`
- Modify: `tests/search_fakes.py`

**Public interface:**

```text
SearchOrchestrator.run(request: RetrievalRequest) -> SearchPipelineResult
get_search_orchestrator() -> SearchOrchestrator
reset_search_orchestrator() -> None
```

`get_search_orchestrator` lazily constructs one production graph from the current config and LLM client. `reset_search_orchestrator` exists for isolated tests/config reloads; it is not a route bypass.

**Exact state machine:**

```text
route
  ├─ skip -> return decision/trace; provider count stays zero
  └─ search
       -> mark orchestrator_started
       -> plan initial queries
       -> start retrieval round 1
       -> run initial query batch in parallel
       -> provider fallback per query when required
       -> deduplicate/cap candidate URLs
       -> extract/cap content reads
       -> assemble initial Evidence
       -> analyze one gap
       ├─ no repair -> seal Evidence
       └─ standard/deep + eligible gap + provider available
            -> plan exactly one repair query
            -> start retrieval round 2
            -> retrieve/extract/merge
            -> reassemble Evidence once
            -> seal Evidence
```

No loop is permitted around gap analysis. The orchestrator seals the result before answer generation and exposes no “search again after validation” callback.

**Budget/accounting rules:**

- Initial queries may run concurrently with a bounded `ThreadPoolExecutor`; one query’s failure must not cancel successful siblings.
- After aggregate URL selection, content reads may also run concurrently within the same remaining tier deadline; their timing is wall-clock, not the sum of child durations.
- Aggregate candidate URL and content-read caps apply across all providers and both rounds.
- Same-query provider fallback increments provider attempts but not semantic-query count.
- Derived `semantic_query_count` counts distinct query IDs dispatched to at least one provider.
- `retrieval_round_count` becomes 1 when the initial batch stage starts, even if no provider is configured; it becomes 2 only when the single repair stage starts.
- `provider_attempted=True` only when an adapter actually attempts its external call. Missing/unconfigured adapters are readiness failures, not attempts.
- `provider_not_configured`, `provider_unavailable`, timeout, empty result, unreadable content, and insufficient Evidence remain distinct failure codes.
- Provider failure never changes `decision.route` to `skip`.
- Use a monotonic hard deadline per tier. Do not calculate totals by summing overlapping timing fields.
- Planner and Evidence-judge model calls are also bounded by the same retrieval deadline. If they miss it, use the already-defined deterministic degraded plan/judgement path; never extend the tier hard timeout.

**Trace timing boundaries:**

Populate every field named in specification section 22.3:

```text
route_latency_ms
query_planning_latency_ms
initial_provider_search_latency_ms
provider_search_total_latency_ms
initial_content_read_latency_ms
content_read_total_latency_ms
initial_evidence_assembly_latency_ms
evidence_assembly_total_latency_ms
gap_analysis_latency_ms
adaptive_repair_latency_ms
answer_generation_latency_ms
structural_validation_latency_ms
semantic_validation_latency_ms
qq_render_latency_ms
retrieval_pipeline_latency_ms
total_response_latency_ms
```

The orchestrator owns through `retrieval_pipeline_latency_ms`; later layers fill answer/validation/render/total fields on the same Trace.

- [ ] Write orchestrator tests first for every skip, light, standard, and deep branch; parallel initial queries; provider fallback; partial sibling failure; aggregate caps; no provider; timeout; unreadable pages; one repair; duplicate repair rejection; light no repair; and sealed-result behavior.
- [ ] Assert three standard initial queries produce `retrieval_round_count=1`, while one later repair produces `2`.
- [ ] Assert standard/deep never exceed one repair, two rounds, or their total-query caps, including when the second Evidence bundle remains insufficient.
- [ ] Assert the Trace distinguishes route, orchestrator start, provider attempt, and sufficient Evidence.
- [ ] Run:

```powershell
python -m unittest tests.test_search_orchestrator -v
```

Expected failure: import failure for `src.search.orchestrator`.

- [ ] Implement the state machine as straight-line code with one conditional repair block; do not use `while` for repair.
- [ ] Pass every planner/judge/provider/extractor call the remaining deadline and each HTTP provider the smaller of normal request timeout and remaining tier time. On deadline, cancel pending futures and shut down the per-request executor with `wait=False, cancel_futures=True`; do not let context-manager shutdown wait past the user-visible hard timeout.
- [ ] Return the mutable Trace without logging it yet. Implement `finalize_search_trace(trace, *, response_finished_at)` as an idempotent helper that calculates `total_response_latency_ms`, serializes body-free fields, and logs once; the final caller invokes it only after validation/rendering. This avoids logging a retrieval-only record before answer timings exist.
- [ ] Implement lazy singleton construction and reset.
- [ ] Run:

```powershell
python -m unittest tests.test_search_orchestrator tests.test_search_providers tests.test_search_evidence -v
```

Expected result: all selected tests pass.

- [ ] Commit:

```powershell
git add src/search/__init__.py src/search/orchestrator.py tests/search_fakes.py tests/test_search_orchestrator.py
git commit -m "feat: orchestrate bounded evidence search"
```

---

## Task 8: Validate Atomic Claims Without Treating Models as Perfect

**Files:**

- Create: `src/search/validation.py`
- Create: `tests/test_search_validation.py`
- Modify: `tests/search_fakes.py`

**Structured answer contract:**

Answer generation will produce JSON in this form:

```json
{
  "answer_blocks": [
    {
      "block_id": "B1",
      "kind": "factual",
      "text": "The current release is version 3.2.",
      "claim_ids": ["C1"]
    }
  ],
  "claims": [
    {
      "claim_id": "C1",
      "block_id": "B1",
      "text": "The current release is version 3.2.",
      "material": true,
      "evidence_ids": ["E1"]
    }
  ]
}
```

Each factual block must be atomic enough that removing the whole block removes the unsupported factual assertion. The generation model may refer to Evidence IDs but may not emit numeric citations; numeric citation rendering is deterministic in Task 9.

**Public interface:**

```text
parse_grounded_draft(text: str) -> GroundedDraft
validate_and_filter(
    draft: GroundedDraft,
    bundle: EvidenceBundle,
    decision: RetrievalDecision,
    *,
    claim_discoverer,
    semantic_verifier,
) -> ValidationReport
```

**Deterministic structural checks:**

- every block/claim ID is unique and mapped;
- every referenced Evidence ID exists in the current bundle and has a real HTTP(S) final URL;
- a claim cannot cite an excluded/unrelated candidate;
- no numeric citation supplied by the model survives parsing;
- failed/insufficient retrieval cannot have claims or citations;
- missing topics in a partial bundle cannot appear in a retained factual block;
- an `inference` block must map its premises to Evidence and use explicit inferential wording; it is not a channel for uncited facts;
- a `non_factual` block may omit claims only when claim discovery finds no external factual span;
- only retained blocks contribute sources;
- any claim already labeled `partial`, `conflict`, `unsupported`, or `unmapped` cannot remain a fully certain fact.

**Model-assisted checks:**

- `ClaimDiscoverer` finds material external-fact spans omitted by the draft’s claim list.
- `SemanticVerifier` labels each claim/Evidence mapping as `supported`, `partial`, `conflict`, or `unsupported`.
- Both receive only the draft, current Evidence, and closed schemas. They do not receive memory or an instruction to decide whether searching was needed.
- Known unsupported/hidden facts remove their entire atomic block.
- A verifier exception is not reported as successful validation:
  - `deep` dynamic/high-consequence output becomes non-definitive fixed degradation;
  - `light/standard` may retain only structurally mapped blocks with a fixed “semantic verification unavailable” disclosure.
- Semantic validation never calls the planner or orchestrator.

- [ ] Write tests first for strict JSON parsing, duplicate IDs, nonexistent Evidence, non-HTTP URL, numeric citation stripping, unused source elimination, partial-topic blocking, inference-premise mapping, hidden facts inside non-factual blocks, conflict labels, unsupported block removal, hidden claim discovery, and verifier failure policy.
- [ ] Add a test where a draft follows model memory instead of valid Evidence. The mismatch must be `unsupported` and removed; it must not become a new Evidence conflict.
- [ ] Add a spy asserting neither validator has access to a search callable.
- [ ] Run:

```powershell
python -m unittest tests.test_search_validation -v
```

Expected failure: import failure for `src.search.validation`.

- [ ] Implement fenced-JSON cleanup and strict parsing. Do not silently invent missing claims or Evidence IDs.
- [ ] Implement deterministic validation before model-assisted validation so malformed drafts never reach a validator.
- [ ] Preserve validator predictions and ground-truth fields separately; production code stores predictions only, while Task 13 owns human labels and metrics.
- [ ] Fill structural and semantic validation timing on the existing Trace.
- [ ] Run:

```powershell
python -m unittest tests.test_search_validation tests.test_search_evidence tests.test_search_models -v
```

Expected result: all selected tests pass.

- [ ] Commit:

```powershell
git add src/search/validation.py tests/search_fakes.py tests/test_search_validation.py
git commit -m "feat: validate grounded search claims"
```

---

## Task 9: Render Deterministic Citations, Failures, Conflicts, and QQ Chunks

**Files:**

- Create: `src/search/renderer.py`
- Create: `tests/test_search_renderer.py`
- Modify: `src/main.py`
- Modify: `tests/test_main_image_flow.py`

**Public interface:**

```text
render_search_reply(
    result: SearchPipelineResult,
    validation: ValidationReport | None,
    *,
    knowledge_fallback_text: str = "",
    qq_limit: int,
) -> RenderedReply

render_plain_reply(text: str, *, trace: SearchTrace, qq_limit: int) -> RenderedReply

split_qq_reply(text: str, limit: int) -> list[str]
```

**Citation rendering:**

- Number sources by first use in retained answer blocks.
- Append `[n]` to the rendered factual block based on its validated Evidence IDs; never trust model-supplied numbers.
- Build `来源：` using only used Evidence. Each entry contains the same number, a bounded title, and the real final URL.
- Deduplicate equivalent final URLs while allowing one source to support multiple blocks.
- Never display an unused provider hit, excluded candidate, failed fetch, or Evidence item removed during validation.
- Reject/strip any model-written `来源：` section before rendering.

**Fixed degradation strings:**

```text
provider_not_configured:
当前搜索服务未配置，无法完成在线核验。

dynamic_or_high_consequence_without_evidence:
我暂时无法完成在线核验，因此不能确认当前结论。

stable_knowledge_fallback_prefix:
在线检索未完成。以下仅按已有知识作有限说明，可能不完整或已经过时：

partial:
以下只回答已获得证据支持的部分；其余部分暂无法确认。

conflict:
来源之间存在未解决差异，下面分别列出，不合并为单一结论。

explicit_search_failed:
你要求了在线搜索，但本次检索未成功完成。

semantic_validation_unavailable:
已获得网页材料，但本次未能完成语义支撑核验；以下表述应谨慎看待。

user_forbid_web_dynamic_limit:
根据你的要求，本次没有联网核验；涉及当前状态的结论无法确认。
```

The explicit-search sentence is additive to the applicable failure, not a replacement.

**Conflict rendering:**

- Show each conflicting Evidence position separately with its own source number.
- Never select a winner merely because one source looks official.
- A conflict section is mandatory whenever the bundle retains a material conflict group, even if the draft omitted it.

**QQ splitting:**

- `src/main.py::split_reply` delegates to `split_qq_reply(text, max(config.max_reply_chars, 200))`.
- Split the answer body at newline/sentence boundaries where possible.
- Treat each source entry’s title+URL as one atomic block; never split a URL.
- If sources do not fit with the final body chunk, start source chunks after the body. All source chunks remain at the end and retain numbering.
- Every returned chunk obeys the limit unless one atomic URL itself exceeds the limit; in that case emit that URL alone without truncation and log body-free `oversize_source_url=true`.
- `RenderedReply.chunks` is computed during rendering with the same `split_qq_reply` function used by `main.split_reply`. `generate_reply` may continue returning `.text` for compatibility, but Trace timing ends only after these chunks have been computed.

- [ ] Write renderer tests first for numbering, multi-claim source reuse, unused-source suppression, nonexistent Evidence rejection, failure with zero citations, stable disclosure, dynamic refusal, partial scope, mandatory conflict display, model-written source stripping, and no dangling source number.
- [ ] Write split tests for body+sources near limits, multiple source chunks, intact URLs, CJK punctuation, and an oversize URL.
- [ ] Run:

```powershell
python -m unittest tests.test_search_renderer -v
```

Expected failure: import failure for `src.search.renderer`.

- [ ] Implement deterministic search/failure rendering without an LLM call. `render_plain_reply` only normalizes final text, computes chunks, and fills render timing; it never adds citations.
- [ ] Replace the body of `main.split_reply` with a call to `split_qq_reply`; preserve the public function for existing callers.
- [ ] Update only split-related assertions in `tests/test_main_image_flow.py`; message/image orchestration must remain unchanged.
- [ ] Run:

```powershell
python -m unittest tests.test_search_renderer tests.test_main_image_flow tests.test_messaging -v
```

Expected result: all selected tests pass.

- [ ] Commit:

```powershell
git add src/search/renderer.py src/main.py tests/test_search_renderer.py tests/test_main_image_flow.py
git commit -m "feat: render verified search replies"
```

---

## Task 10: Switch Ordinary Chat to Program-First Retrieval

**Files:**

- Modify: `src/chat/prompt.py`
- Modify: `src/chat/chat_service.py`
- Delete: `tests/test_chat_tool_finalization.py`
- Create: `tests/test_chat_retrieval_flow.py`
- Modify: `tests/test_identity_configuration.py`
- Modify: `tests/test_multimodal_chat.py`
- Modify: `tests/test_main_image_flow.py`

**New `generate_reply` contract:**

```text
generate_reply(
    context: MemoryContext | str,
    text: str,
    image_data_urls: list[str] | None = None,
    *,
    force_search: bool = False,
    history_text: str | None = None,
) -> str
```

Remove the unstructured `tool_context` parameter. The only internal search input is `SearchPipelineResult` with structured Evidence.

`generate_reply` constructs `RetrievalRequest(request_source=RequestSource.COMMAND)` when `force_search=True`; otherwise it uses `RequestSource.CHAT`. The legacy facade constructs `RequestSource.COMPATIBILITY` itself.

**Ordinary chat flow:**

```text
latest user text only
  -> SearchOrchestrator.run
  -> skip:
       for ordinary closed tasks, build memory/persona context and make one normal answer call with no search tool
       for user_forbid_web + stable knowledge, permit a limited answer with a fixed no-web disclosure
       for user_forbid_web + dynamic/high-consequence facts, return the fixed limitation without a factual model answer
  -> sufficient/partial/conflicting Evidence:
       build bounded Evidence JSON
       generate atomic GroundedDraft at temperature 0.2
       validate/filter
       deterministic render
  -> stable fact retrieval failure:
       generate a limited knowledge answer without retrieved-memory facts
       prepend fixed online-verification disclosure
       no citations
  -> dynamic/high-consequence retrieval failure:
       deterministic degradation/refusal
       no memory-completion call and no citations
  -> explicit no-web/search conflict:
       deterministic clarification
       no provider call
  -> finalize and log exactly one body-free SearchTrace
```

**Prompt changes:**

- Delete `SEARCH_WEB_TOOL`, `SUPPORTED_TOOL_NAMES`, `MAX_TOOL_CALL_ROUNDS`, tool query normalization, tool execution, and the final tool-synthesis loop from `chat_service.py`.
- Do not pass `tools` or `tool_choice` to ordinary answer generation.
- Remove all prompt language saying memory hits skip search, the model decides when to search, full questions should not be queries, or search failure can be guessed from memory.
- Keep persona and safety rules in the user-visible answer-generation system message.
- Build Evidence context as bounded JSON with Evidence IDs, titles, URLs, excerpts, dates, supported topics, relations, and conflict groups.
- Keep Evidence and retrieved memory in separately labeled untrusted sections. The prompt states that factual answer blocks require Evidence and that memory cannot override Evidence or become hidden counterevidence.
- For stable search-failure knowledge fallback, call `build_untrusted_context(context, query=text, evidence_payload="", include_memories=False)` so retrieved memory cannot masquerade as model general knowledge.
- Router/planner/provider calls happen before memory retrieval and receive no memory/history.

**Image behavior:**

- Preserve image data only in the final multimodal answer call and never in provider queries, SearchTrace, persistent history, or Evidence.
- An image-only identification/description request may use `closed_context_only`.
- If the text explicitly requests current/external facts but does not identify a searchable entity, route normally; when Evidence remains insufficient, ask for the entity/name instead of silently using vision/model memory as external evidence.

**Grounded generation prompt:**

Require the exact Task 8 JSON schema, one atomic factual statement per factual block, Evidence IDs from the supplied set only, explicit conflict blocks, and omission of missing topics. Instruct the model that its memory cannot override Evidence. Strip internal markers such as `[SRCH:*]`, `[MEM:*]`, and `[CHAT:*]` only after deterministic rendering.

- [ ] Replace the old tool-loop test with `tests/test_chat_retrieval_flow.py`. First add tests for:
  - social chat skip with zero provider attempts;
  - stable factual question starts the orchestrator even when the answer model “knows” it;
  - sufficient Evidence produces structurally validated citations;
  - memory disagrees with Evidence and cannot change the answer;
  - stable failure uses fixed disclosure and no retrieved memory;
  - dynamic/high-consequence failure emits no definite fact;
  - explicit no-web dynamic question calls no provider and shows the fixed current-state limitation;
  - partial Evidence answers only supported topics;
  - conflict is shown;
  - answer validation failure cannot trigger a new search;
  - no ordinary LLM call receives `tools`;
  - history is appended exactly once with the final rendered reply.
- [ ] Run:

```powershell
python -m unittest tests.test_chat_retrieval_flow -v
```

Expected failure: the current model-controlled tool loop runs and the new orchestration hooks do not exist.

- [ ] Refactor `prompt.py` to accept structured Evidence and an `include_memories` switch.
- [ ] Refactor `generate_reply` to the program-first flow and keep image/history/persona behavior.
- [ ] Call `finalize_search_trace` exactly once after citation/failure rendering and precomputed QQ segmentation finish, and before the text is appended to history. Skip decisions receive the same complete route/total Trace with provider counts at zero.
- [ ] Delete old search-tool helpers and `tests/test_chat_tool_finalization.py` with `apply_patch` only after the new tests cover their replacement behavior.
- [ ] Update identity tests so every user-visible answer-generation prompt contains the full persona. Internal routing/planning/validation prompts remain role-neutral and are not required to roleplay.
- [ ] Update multimodal tests to remove `tool_context` and assert image data never reaches the fake provider or saved history.
- [ ] Run:

```powershell
python -m unittest tests.test_chat_retrieval_flow tests.test_identity_configuration tests.test_multimodal_chat tests.test_main_image_flow tests.test_llm_tool_affinity tests.test_deepseek_tool_context tests.test_gemini_native_client -v
```

Expected result: all selected tests pass. Generic LLM provider tool protocol tests remain because provider clients still support tools for other callers; ordinary chat simply no longer supplies a search tool.

- [ ] Commit:

```powershell
git add src/chat/prompt.py src/chat/chat_service.py tests/test_chat_retrieval_flow.py tests/test_identity_configuration.py tests/test_multimodal_chat.py tests/test_main_image_flow.py
git add tests/test_chat_tool_finalization.py
git commit -m "feat: make chat search program-first"
```

---

## Task 11: Unify `/search` and Retire the Legacy Search Controller

**Files:**

- Modify: `src/commands/search.py`
- Modify: `src/commands/__init__.py`
- Replace internals: `src/services/search_service.py`
- Modify: `tests/test_command_renderer.py`
- Modify: `tests/test_identity_configuration.py`
- Modify: `tests/test_product_scope.py`

**Command flow:**

```python
def search_reply(query: str, session_key: str, raw_message: str) -> str:
    normalized = normalize_search_query(query)
    if not normalized:
        return "想搜什么？比如：/search DeepSeek 最新消息"
    return generate_reply(
        session_key,
        normalized,
        force_search=True,
        history_text=raw_message,
    )
```

`normalize_search_query` now performs only Unicode normalization, whitespace cleanup, optional `/search` or `/s` prefix removal, and a 500-character transport cap. It must preserve the original question form, intent words, dates, and CJK sentence.

**Compatibility facade:**

Keep these existing names in `src/services/search_service.py`:

```text
SearchResult(ok: bool, status: str, text: str)
search(query: str) -> SearchResult
web_search(query: str) -> str
has_search_results(result: SearchResult) -> bool
normalize_search_query(query: str) -> str
```

The facade invokes `get_search_orchestrator().run(RetrievalRequest(force_search=True, request_source=RequestSource.COMPATIBILITY))` and flattens only admitted Evidence into a bounded compatibility text format. It does not generate a chat answer, number final citations, classify source relationship heuristically, or expose excluded candidates. It fills unused answer/validation/render timings with zero and finalizes its Trace once before returning. New chat/command code must not call this facade.

Remove the old monolithic provider, relevance, `docs`-domain, flattening, fetch, and freshness-controller implementation from `search_service.py`; adapters and Evidence now own those behaviors.

**Command guarantees:**

- `/search` always supplies `force_search=True`.
- `/search` and ordinary chat share route, planner, providers, extraction, Evidence, validation, failure, conflict, and renderer code.
- A `/search` request that also clearly forbids networking becomes the explicit conflict clarification and calls no provider.
- `CommandOutcome(already_rendered=True)` remains, so `PersonaCommandRenderer` cannot rewrite citations or fixed failure text.
- The command stores the raw command in history once but sends only the normalized query to the planner/provider.

- [ ] Update command and product-scope tests first. Cover preserved natural-language query, forced search, empty command, same pipeline identity, no double rendering, failure with no citation, explicit no-web conflict, and compatibility facade output.
- [ ] Run:

```powershell
python -m unittest tests.test_product_scope tests.test_command_renderer tests.test_identity_configuration -v
```

Expected failure: `/search` still builds flat `tool_context` and the legacy controller remains active.

- [ ] Implement the command flow and compatibility facade.
- [ ] Delete obsolete private helpers from `search_service.py`; retain no second provider/fetch/routing implementation.
- [ ] Use `rg` to prove ordinary chat/command no longer imports legacy search execution:

```powershell
rg -n "search_service import (search|web_search)|SEARCH_WEB_TOOL|run_tool|tool_context" src/chat src/commands
```

Expected result: no matches, except `normalize_search_query` if `commands/search.py` deliberately imports the compatibility normalizer.

- [ ] Run:

```powershell
python -m unittest tests.test_product_scope tests.test_command_renderer tests.test_identity_configuration tests.test_chat_retrieval_flow -v
```

Expected result: all selected tests pass.

- [ ] Commit:

```powershell
git add src/commands/search.py src/commands/__init__.py src/services/search_service.py tests/test_command_renderer.py tests/test_identity_configuration.py tests/test_product_scope.py
git commit -m "refactor: unify explicit and chat search"
```

---

## Task 12: Add Readiness, Trace Coverage, and Operator Documentation

**Files:**

- Modify: `src/config.py`
- Modify: `.env.example`
- Modify: `src/main.py`
- Modify: `tests/runtime.py`
- Modify: `tests/test_health.py`
- Modify: `README.md`
- Modify: `tests/test_readme_guide.py`

**Configuration:**

- Keep `TAVILY_API_KEY`, `PROXY_URL`, `SEARCH_MAX_RESULTS`, and `REQUEST_TIMEOUT`.
- Normalize `SEARCH_MAX_RESULTS` to at least 1. It is a per-query provider result ceiling only; immutable tier budgets still enforce aggregate candidate URL and content-read maxima.
- Do not add an environment switch for route default, model-confidence bypass, semantic-validation bypass, or unlimited repair.

**Readiness:**

At startup, obtain provider readiness after persona/config validation:

- if at least one adapter is available, log body-free `search_ready=true` and provider names;
- if none is available, log a warning and continue serving closed-context tasks;
- factual requests still route to search and return `provider_not_configured`/`provider_unavailable`, never ordinary chat.

Add to `/health`:

```json
{
  "search_ready": true,
  "search_providers": [
    {"provider": "tavily", "configured": true, "available": true},
    {"provider": "ddgs", "configured": true, "available": true}
  ]
}
```

Never expose API keys, proxy credentials, exception messages, queries, URLs, Evidence text, or user identifiers.

**Test singleton reset:**

`tests/runtime.py::reset_runtime_singletons()` calls `reset_search_orchestrator()` when `src.search` is imported, so every isolated runtime uses its patched config/provider state.

**README requirements:**

- Explain factual-default search and the closed skip exceptions.
- Explain `light`/`standard`/`deep` budgets and the initial+single-repair distinction.
- Explain Tavily primary/DDGS availability fallback.
- Explain Evidence, relevance-first admission, deterministic citations, and failure disclosures.
- State that `SEARCH_MAX_RESULTS` is per provider query and cannot raise tier ceilings.
- Document `/health` readiness and body-free Trace fields.
- Document that unit tests are offline and how to run controlled online evaluation separately.

- [ ] Add health/readiness/redaction/singleton-reset/README assertions first.
- [ ] Run:

```powershell
python -m unittest tests.test_health tests.test_readme_guide tests.test_test_runtime -v
```

Expected failure: health/readiness fields and documentation are missing.

- [ ] Implement config normalization, startup readiness, health metadata, and singleton reset.
- [ ] Update `.env.example` comments and README without adding new secrets or bypasses.
- [ ] Test startup with no provider: startup succeeds, factual orchestration still starts, provider attempt remains false, and fixed failure rendering is used.
- [ ] Run:

```powershell
python -m unittest tests.test_health tests.test_readme_guide tests.test_test_runtime tests.test_search_orchestrator -v
```

Expected result: all selected tests pass.

- [ ] Commit:

```powershell
git add src/config.py .env.example src/main.py tests/runtime.py tests/test_health.py README.md tests/test_readme_guide.py
git commit -m "docs: expose evidence search readiness"
```

---

## Task 13: Build the 140-Case Human-Label Evaluation Harness

**Files:**

- Create: `eval/search/README.md`
- Create: `eval/search/cases.jsonl`
- Create: `eval/search/provider_recordings.jsonl`
- Create: `eval/search/model_predictions.jsonl`
- Create: `tools/evaluate_search.py`
- Create: `tests/test_search_evaluation.py`

**Dataset quotas:**

`cases.jsonl` contains exactly 140 unique, synthetic, privacy-safe cases:

| Category | Count |
|---|---:|
| clear no-retrieval-benefit | 20 |
| stable facts / light | 20 |
| explanation, comparison, technical, product / standard | 25 |
| news, current state, dynamic facts / deep | 20 |
| regulated-domain or controversy, split foundation vs high-consequence | 15 |
| explicit search, verification, source request | 10 |
| ambiguous entity or mixed task | 10 |
| failure, partial Evidence, conflict, prompt injection | 20 |
| **Total** | **140** |

Every row has:

```json
{
  "case_id": "route-light-001",
  "category": "stable_fact",
  "question": "什么是光合作用",
  "allow_skip": false,
  "skip_reason": null,
  "minimum_tier": "light",
  "external_fact_required": true,
  "actionability": "none",
  "potential_harm": "none",
  "expected_query_purposes": ["direct"],
  "expected_initial_query_min": 1,
  "expected_initial_query_max": 1,
  "expected_max_rounds": 1,
  "material_claim_spans": ["光合作用的定义"],
  "acceptable_source_relations": ["primary", "independent", "unknown"],
  "semantic_labels": [],
  "expected_outcome": "grounded_answer",
  "fixture_id": "stable-fact-001",
  "label_status": "reviewed",
  "reviewed_by": "project_owner",
  "reviewed_at": "2026-07-29"
}
```

`reviewed_by` must contain the actual reviewer identifier in the completed dataset; empty values and the reserved value `unreviewed` are rejected by the integrity test.

**Recorded data separation:**

- `provider_recordings.jsonl` contains only synthetic or intentionally recorded public test pages, provider metadata, bounded excerpts, and expected fetch statuses. It contains no secrets or private user data.
- `model_predictions.jsonl` stores component (`router`, `planner`, `relevance`, `claim_discovery`, or `semantic_support`), model/version, prompt schema version, case ID, closed prediction fields, and run timestamp.
- Human labels remain only in `cases.jsonl`. Predictions can never overwrite labels.
- Online provider runs are opt-in and never part of `unittest`.

**Evaluation CLI:**

```powershell
python tools/evaluate_search.py integrity
python tools/evaluate_search.py offline
python tools/evaluate_search.py traces --traces path\to\search-traces.jsonl --labels path\to\human-audit.jsonl
python tools/evaluate_search.py online --limit 10
```

`integrity` validates count, quotas, unique IDs, enums, human-review fields, no hard secrets, and recording references. `offline` prints JSON and exits nonzero when an acceptance threshold fails.

**Required metrics:**

- mandatory-search route rate;
- explicit-search route and orchestrator-start rates;
- tier macro precision/recall/F1;
- meaningless-search rate on legal non-factual skips;
- query/URL/read/repair budget violations;
- initial batch versus retrieval-round accounting;
- deterministic citation/failure invariant violations;
- claim discovery precision/recall/F1;
- semantic support macro precision/recall/F1;
- relevance-gate precision/recall/F1;
- dynamic/high-consequence subgroup metrics;
- production `route_coverage`, `orchestrator_start_rate`, `provider_attempt_rate`, and `sufficient_evidence_rate` over a human-labelled `D_factual`.
- P50/P95/P99 for every latency field, plus per-tier `retrieval_pipeline_latency_ms` P95 and hard-timeout violation counts; answer/validation/render time must not be folded into retrieval P95.

For production trace aggregation:

- exclude explicit no-web and legal closed-context tasks from `D_factual`;
- report provider-not-configured/unavailable separately;
- never reclassify execution failure as route skip;
- report numerator and denominator beside every rate.

- [ ] Write metric unit tests first using small hand-calculable confusion matrices and trace samples. Assert 100% structural invariants separately from model metrics.
- [ ] Run:

```powershell
python -m unittest tests.test_search_evaluation -v
```

Expected failure: evaluation files and loader do not exist.

- [ ] Implement the dependency-free CLI with `argparse`, `json`, and standard-library metric calculations.
- [ ] Author and schema-check the 20 clear no-benefit rows.
- [ ] Author and schema-check the 20 stable-fact/light rows.
- [ ] Author and schema-check the 25 external-fact explanation/comparison/technical/product rows.
- [ ] Author and schema-check the 20 news/current/dynamic rows.
- [ ] Author and schema-check the 15 regulated-domain/controversy rows, with separate foundation and personalized/high-consequence labels.
- [ ] Author and schema-check the 10 explicit-search and 10 ambiguity/mixed-task rows.
- [ ] Author and schema-check the 20 failure/partial/conflicting/injection rows.
- [ ] Run the quota/duplicate checker across all 140 rows. Include the routing/adversarial examples from the confirmed specification and reject rows that differ only by punctuation.
- [ ] Author synthetic provider recordings for every Evidence/failure case and link them by `fixture_id`.
- [ ] Run an offline verifier prediction pass against the fixed fixtures, store predictions separately, and obtain a human label review. Do not check off this step until every row has the actual `reviewed_by` and `reviewed_at`.
- [ ] Run:

```powershell
python tools/evaluate_search.py integrity
python tools/evaluate_search.py offline
```

Required acceptance:

```text
mandatory search route rate = 1.00
explicit search route rate = 1.00
explicit search orchestrator start rate = 1.00
explicit search provider-attempt rate = 1.00 when a provider is configured
fixed-set factual route coverage = 1.00
tier macro F1 >= 0.90
legal non-factual meaningless-search rate <= 0.10
all deterministic structure/failure violation counts = 0
claim discovery precision, recall, F1 >= 0.95
semantic support macro precision, recall, F1 >= 0.95
relevance gate precision, recall, F1 >= 0.95
dynamic/high-consequence claim and support precision, recall, F1 >= 0.97
```

For `traces` mode, a production human-audited sample additionally requires `D_factual route_coverage >= 0.98`; provider-unavailable cases remain in that denominator and are reported as execution failures, not route misses.

For a controlled online/production latency sample, separately require retrieval P95 of `light <= 6 s`, `standard <= 15 s`, and `deep <= 30 s`, with zero hard-timeout violations beyond `8/20/40 s`. Do not infer these latency results from synthetic offline recordings.

If a model-quality threshold fails, retain the human labels, diagnose prompts/classifier behavior, add a regression, and rerun predictions. Never edit labels to fit predictions.

- [ ] Run:

```powershell
python -m unittest tests.test_search_evaluation tests.test_search_router tests.test_search_validation tests.test_search_evidence -v
```

Expected result: all selected tests pass.

- [ ] Commit:

```powershell
git add eval/search tools/evaluate_search.py tests/test_search_evaluation.py
git commit -m "test: add evidence search evaluation set"
```

---

## Task 14: Run End-to-End Acceptance and Prepare the Rollback Boundary

**Files:**

- Modify only files needed for regressions discovered by this task.
- Do not add a runtime “disable default retrieval” switch.

- [ ] Run the focused search suite:

```powershell
python -m unittest tests.test_search_models tests.test_search_router tests.test_search_planner tests.test_search_providers tests.test_search_extraction tests.test_search_evidence tests.test_search_orchestrator tests.test_search_validation tests.test_search_renderer tests.test_chat_retrieval_flow tests.test_search_evaluation -v
```

Expected result: all tests pass.

- [ ] Run all integration surfaces affected by the change:

```powershell
python -m unittest tests.test_product_scope tests.test_command_renderer tests.test_identity_configuration tests.test_multimodal_chat tests.test_main_image_flow tests.test_health tests.test_readme_guide tests.test_llm_tool_affinity tests.test_deepseek_tool_context tests.test_gemini_native_client tests.test_messaging -v
```

Expected result: all tests pass.

- [ ] Run the complete hermetic suite:

```powershell
python -m unittest discover -s tests -t . -v
```

Expected result: all tests pass, no external HTTP guard fires, and no repository runtime database is opened.

- [ ] Run static invariant searches:

```powershell
rg -n "模型.*(知道|有信心)|记忆.*不要.*搜索|由模型自行判断是否调用|SEARCH_WEB_TOOL|MAX_TOOL_CALL_ROUNDS" src
rg -n "_source_type|HIGH_VALUE_SOURCE_TYPES|source_priority|startswith.*docs|/docs.*primary|developer.*primary" src/search
rg -n "while .*repair|while .*search|for .*MAX_TOOL_CALL_ROUNDS" src/search src/chat
```

Expected results:

- no prompt or routing code allows knowledge/confidence/memory to skip;
- no source-relation code promotes a source from URL shape alone;
- no repair/search loop exists.

- [ ] Run evaluation:

```powershell
python tools/evaluate_search.py integrity
python tools/evaluate_search.py offline
```

Expected result: both exit 0 and print every required numerator, denominator, and quality metric.

- [ ] If provider credentials are available and the user explicitly authorizes a controlled online run, execute:

```powershell
python tools/evaluate_search.py online --limit 10
```

Record provider/date/status/latency only in the evaluation artifact. If authorization or credentials are absent, mark online verification as not run; do not claim it passed.

- [ ] Inspect `git diff --check` and `git status --short`. Resolve whitespace errors and ensure only planned files changed.
- [ ] If Task 14 required a regression fix, commit it with a scoped message after its failing test passes. Otherwise do not create an empty commit.
- [ ] Invoke `superpowers:requesting-code-review` for a spec-to-code review that checks all 15 program invariants, all failure states, two-round accounting, metric boundaries, and privacy.
- [ ] Address only verified findings, rerun the focused and full suites, then invoke `superpowers:verification-before-completion`.
- [ ] Use `superpowers:finishing-a-development-branch` to present integration choices. The rollback boundary is the pre-merge commit/release; rollback must not be implemented as a silent default-no-search runtime mode.

---

## Implementation Completion Checklist

- [ ] Every factual/mixed/ambiguous request without a legal closed skip reason routes to at least `light`.
- [ ] Explicit search/verification/source requests route and start orchestration 100% of the time, except the explicit contradictory no-web clarification.
- [ ] Regulated-domain foundations are `standard`; personalized/current/high-consequence requests are `deep`.
- [ ] CJK original natural-language queries are retained and supplemented, not replaced by mechanical keywords.
- [ ] Relevance gates Evidence before source relationship.
- [ ] The initial batch counts as one round; at most one adaptive repair query occurs; answer validation cannot search.
- [ ] Provider unavailability is an execution failure, never a skip or route miss.
- [ ] Dynamic/high-consequence facts without valid Evidence are not stated definitively.
- [ ] Stable knowledge fallback always discloses failed online verification and uses no citations or retrieved-memory evidence.
- [ ] Model memory cannot skip, fill dynamic failure, override valid Evidence, or create hidden conflicts.
- [ ] Every displayed source is used, has a real final URL, and is numbered deterministically.
- [ ] Every known unsupported/partial/conflicting claim is removed or rendered non-definitively.
- [ ] Trace fields separately support route, orchestrator, provider-attempt, and sufficiency metrics with exact latency boundaries.
- [ ] The 140-case human-labelled evaluation passes deterministic requirements and reports model quality honestly.
- [ ] Full hermetic tests, offline evaluation, and review complete before merge.
