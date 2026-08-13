# WebSearch reliability simplification

## 1. Goal

Make the existing QQ Bot WebSearch reliably use completed retrieval work instead
of treating one slow or malformed sibling as a request-wide failure. The change
must simplify responsibilities, preserve useful search capabilities, and avoid
turning the bot into a Deep Research system.

This design keeps the existing linear architecture:

```text
Request Analysis
→ Retrieval Router
→ Planner
→ DDGS / Tavily
→ Reader
→ Candidate Judge
→ Evidence
→ Answer Policy
→ Validator
→ Renderer
```

It does not add an Agent, an LLM call, a search tier, a retrieval loop, a third
round, or an Answer-to-Search feedback path.

## 2. Frozen boundaries

### 2.1 Independent concepts

- Search tier describes expected retrieval complexity only.
- Freshness describes how current each material topic's Evidence must be.
- Risk describes how cautious the final answer must be.
- Provider, Query, Reader, Judge, Evidence, Validation, and Rendering failures
  remain separate facts and may not overwrite one another.

Risk and Freshness never raise a tier. Explicit search forbids `skip` but does
not by itself select `standard`.

### 2.2 Preserved capabilities

The implementation must preserve:

- `skip / light / standard` and no other production tier;
- DDGS first and Tavily fallback;
- light Direct Query only and no Repair;
- standard initial Direct Query plus at most two supplemental Queries;
- standard at most one Repair Query and no Repair-of-Repair;
- request-level Query caps of 1 for light and 4 for standard including Repair;
- request-level candidate URL caps of 5 for light and 8 for standard;
- request-level Read-attempt caps of 2 for light and 5 for standard;
- URL safety, canonicalization, and deduplication;
- page/document extraction and the existing conservative snippet fallback;
- per-material-topic Freshness and version checks;
- Source Requirement and independence checks;
- material-topic Conflict handling;
- meaningful `PARTIAL` answers and uncontested subsets;
- Citation closure;
- high-consequence Answer Policy;
- Validator monotonicity and deterministic Renderer output.

## 3. Budget policy

### 3.1 Independent stage budgets

Every stage receives its own full timeout. Time consumed by an earlier stage
does not reduce a later stage's timeout. Unused time is neither borrowed nor
transferred. A stage may stop early whenever its own work is complete.

The initial policy is:

| Route | Stage | Hard cap |
|---|---|---:|
| light | analysis and route | 3 s |
| light | DDGS batch | 6 s |
| light | Tavily fallback batch | 6 s |
| light | Reader | 4 s |
| light | Candidate Judge | 4 s |
| light | Answer draft | 4 s |
| light | Validator | 4 s |
| light | Renderer | 1 s |
| standard | analysis and route | 3 s |
| standard | Planner | 4 s |
| standard | initial DDGS batch | 8 s |
| standard | initial Tavily fallback batch | 8 s |
| standard | initial Reader | 6 s |
| standard | initial Candidate Judge | 5 s |
| standard | Gap Analysis | 1 s |
| standard | Repair Planner | 2 s |
| standard | Repair DDGS | 5 s |
| standard | Repair Tavily fallback | 5 s |
| standard | Repair Reader | 3 s |
| standard | post-Repair Candidate Judge | 4 s |
| standard | Answer draft | 4 s |
| standard | Validator | 4 s |
| standard | Renderer | 1 s |

The scheduling margin is a named policy value. It is not a business-stage
budget and may not be spent to extend Provider, Reader, Judge, or generation.

### 3.2 Derived request watchdog

There is no separately hand-written light or standard watchdog constant.

```python
watchdog_seconds = budget_policy.maximum_request_seconds(route)
```

`maximum_request_seconds(route)` is a pure sum of every stage that can legally
execute for that route plus the scheduling margin. The watchdog deadline is
anchored at the request start time so Analysis is counted exactly once.

The watchdog prevents non-cooperative dependencies, repeated stages, or illegal
extra rounds from running forever. It must not serve as an ordinary stage
timeout, erase sealed results, or turn partial success into `provider_timeout`.

### 3.3 Data caps stay request-level

Independent timeouts do not grant new data allowances. DDGS and Tavily share
the route's candidate URL cap. Initial and Repair Reader stages share the
route's Read-attempt cap. Repair does not reset any Query, URL, or Read count.

## 4. Router contract

The request-analysis LLM may suggest complexity codes, but deterministic
structure checks decide whether those codes are applicable.

Light is the default for one clear entity and one material fact without a
comparison, recommendation, complete list, or explicit independent-source
requirement. Version-like name tokens such as `V4 Pro` and organization labels
such as `CN赛区` do not by themselves make a request multi-entity.

Standard is selected only for actual multi-fact, multi-entity, comparison,
recommendation, complete-list/schedule, multi-source, cross-verification, or
material ambiguity requirements.

Examples are illustrative, not production lexicon entries:

- one product release date → light;
- one match result → light;
- one current version or price → light;
- a complete playoff schedule → standard;
- a comparison or recommendation → standard.

## 5. Provider and Query outcomes

### 5.1 Layered state

`ProviderAttempt` records one Provider's attempt for one Query. It is diagnostic
truth only and cannot decide the request result.

`QueryOutcome` records the final result of a semantic Query after DDGS and any
necessary Tavily fallback:

```text
resolved | empty | timeout | error | unavailable
```

A Query is `resolved` when at least one Provider returns at least one safe,
usable candidate URL.

The deterministic batch aggregate is:

```text
all Queries resolved                         → success
some resolved and some unresolved            → partial_success
no Query has a safe, usable candidate URL     → all_failed
```

Only `all_failed` may terminate at the Provider layer.

### 5.2 DDGS and Tavily stages

All initial Queries run concurrently in one bounded DDGS batch. At its deadline
the system harvests completed results, marks only unfinished Queries as timed
out, cancels work that has not begun, and does not wait indefinitely for a
non-cooperative dependency.

Only DDGS-empty, timed-out, errored, unavailable, or URL-invalid Queries enter
the Tavily batch. Tavily receives its complete independent stage timeout. A
Query already resolved by DDGS is not repeated on Tavily.

The same contract applies to the single Repair Query.

### 5.3 Partial success

One Query's timeout or error never deletes sibling hits. A batch with any
resolved Query continues to Candidate aggregation and Reader. An expired stage
stops only that stage; it does not synthesize a request-wide timeout.

### 5.4 Candidate aggregation

Candidates are selected round-robin by Query, then by stable Provider result
order, so one Query cannot consume the entire URL cap before another Query is
represented. HTTP(S), public-host, canonicalization, and duplicate checks run
before a hit is counted toward the candidate cap.

## 6. Reader contract

Each candidate URL produces one closed Read outcome:

```text
readable | unreadable | timeout | unsafe_url | unsupported_type
```

At the Reader deadline, all completed readable documents are retained. An
unfinished or failed URL affects itself only. If no page/document is readable,
an available Provider snippet may remain a low-authority Candidate under the
existing policy, but a snippet used after a fetch failure cannot become citable
Evidence merely because it exists.

## 7. Candidate Judge contract

### 7.1 Sole responsibility

The Candidate Judge answers only:

> Which known material topics does this Candidate's actual content directly
> support?

It may also provide the Freshness, Source, Publisher, and Conflict metadata
needed to evaluate those support edges.

It does not decide:

- Candidate relevance level;
- Evidence admission;
- Evidence sufficiency;
- `PARTIAL`, `SUFFICIENT`, `CONFLICTING`, or `INSUFFICIENT`;
- Repair or stop behavior;
- final-answer certainty.

### 7.2 Closed row

The Judge row contains:

```json
{
  "candidate_id": "C1",
  "supported_topic_ids": ["topic-1"],
  "freshness_by_topic": {"topic-1": "satisfied"},
  "source_relation": "primary",
  "publisher_entity_match": true,
  "ownership_basis": "publisher matches the queried organization",
  "publisher": "organization",
  "conflict_key": null,
  "conflict_value": null,
  "conflict_relation": null
}
```

`relevance` is removed. `supported_topic_ids` is the only expression of direct
Candidate-to-Topic support. An empty support list is a valid negative judgement,
not a malformed row and not a Judge failure.

The Judge remains one batch LLM call. There is no per-Candidate call, format
repair model, Judge retry Agent, Relevance Agent, or self-reflection stage.

### 7.3 Failure isolation

- a valid expected row is retained;
- a missing expected row fails only that Candidate;
- a malformed row fails only that Candidate;
- an expected duplicate ID fails only that Candidate;
- an unknown ID is discarded and recorded as a body-free anomaly;
- a damaged root object or failed/empty LLM response is a batch-level Judge
  failure.

All valid negative rows and zero malformed rows mean the Judge completed and
found no Candidate support. No valid rows combined with an actual call or root
schema failure means `judge_unavailable`. These cases must not share one state.

## 8. Evidence, Repair, and answering

### 8.1 Evidence remains program-derived

A topic support edge is admitted only when:

```text
the Judge explicitly names the known material topic
AND the Candidate content is usable and citable
AND topic Freshness is eligible
AND Source Requirement is satisfied
AND no unresolved material Conflict invalidates that edge
```

Publisher or primary-source metadata cannot rescue an empty support list.

The program computes Evidence State with one priority rule:

```text
unresolved material conflict                  → CONFLICTING
all material topics supported                 → SUFFICIENT
meaningful supported material subset          → PARTIAL
otherwise                                     → INSUFFICIENT
```

Freshness is assessed per material topic. Conflict removes only the affected
topic support and preserves uncontested supported topics. Candidate Judge
metadata never writes Evidence State directly.

### 8.2 One Repair only

Only standard requests may Repair. A deterministic Evidence Gap plus remaining
request-level Query/URL/Read capacity may trigger one targeted Repair. Repair
has independent time budgets but does not reset data caps. Post-Repair Judge
completion ends Retrieval unconditionally.

Judge failure, Risk, Answer Policy, Validator, and Renderer cannot trigger
search or Repair.

### 8.3 Failure mapping

User-facing failure is derived after the relevant stage state is known:

| State | Meaning |
|---|---|
| every Query failed across DDGS/Tavily | online retrieval did not complete |
| candidates exist but no content is reliably readable | pages were found but content could not be verified |
| Judge batch unavailable | retrieved content could not be reliably judged |
| Judge completed, Evidence insufficient | current results cannot confirm the requested fact |
| Evidence partial | answer only the meaningful supported subset |
| Evidence conflicting | answer uncontested facts and display the conflict |
| Evidence sufficient | answer directly with actual sources |

Successful ordinary searches display no success/status banner. Accuracy and
professional-judgement warnings appear only when Answer Policy has already
emitted the appropriate high-consequence warning code. Renderer remains a pure
view layer.

## 9. Trace and privacy

Trace records only body-free closed metadata. It may record:

- stage name, start/completion/timeout, and latency;
- Query index, purpose, Provider attempt status, final Query outcome, and hit
  count;
- Retrieval batch state and resolved/unresolved counts;
- Read outcome counts;
- Judge anomaly codes/count and Judge batch status;
- opaque topic IDs and Evidence State;
- final failure/disclosure states.

It may not record raw Query text, URL, title, page content, Candidate ID,
Candidate judgement text, or answer content.

Provider, Reader, Judge, Evidence, Validation, and Rendering failures remain
separate fields. A local failure is never rewritten as another stage's failure.

## 10. Migration order

Implementation proceeds in the following order, with RED-to-GREEN and review
after each step:

1. Capture current stage-level regression baselines and production-shaped
   reproductions without blessing current failures as expected behavior.
2. Introduce the single Budget Policy and derived watchdog; then remove shared
   rolling-deadline use stage by stage.
3. Add deterministic Router validation so unsupported LLM complexity labels do
   not raise the tier.
4. Separate DDGS and Tavily batches and introduce deterministic QueryOutcome
   aggregation.
5. Make Candidate aggregation round-robin and make Reader harvest partial
   completion under its independent budget.
6. Migrate Judge fixtures and Prompt/Parser to the support-edge schema, verify
   Evidence parity, and immediately delete runtime relevance compatibility.
7. Feed the new batch/read/Judge states into existing Evidence and the single
   Repair path.
8. Centralize final failure mapping and remove obsolete timeout/relevance
   branches, schema fields, Trace fields, fixtures, and evaluator compatibility.
9. Run hermetic, adversarial, fresh blind online, and independent review gates.

There is no long-lived runtime dual Judge. Old fixtures may be mechanically
converted offline to prove Evidence/Citation parity before the production
Prompt switches.

## 11. Simplification acceptance

The final implementation must satisfy:

- exactly three production tiers: skip/light/standard;
- no additional Agent or LLM call;
- one batch Judge call per retrieval round;
- light has one Direct Query and no Repair;
- standard has at most three initial Queries and one Repair Query;
- no third retrieval round;
- no Answer/Validator/Renderer-to-Search feedback;
- one Budget Policy and one pure maximum-request calculation;
- one Provider/Query aggregation rule;
- one Candidate-to-Topic support representation: `supported_topic_ids`;
- one Evidence State computation;
- one final failure-to-disclosure mapping;
- fewer duplicated branches, state aliases, and compatibility paths when the
  migration is complete.

Production LOC is not an absolute gate. Any net increase must represent an
explicit responsibility boundary or reliability invariant rather than a new
framework, repeated state, or hidden fallback.

## 12. Verification and quality gates

### 12.1 Hermetic and adversarial gates

Tests must include:

- one factual entity is light despite version/name tokens;
- complete schedules, comparisons, recommendations, and explicit independent
  sourcing are standard;
- DDGS success does not invoke Tavily;
- DDGS timeout/empty/error gives only that Query a full Tavily fallback budget;
- sibling success plus timeout/error is `partial_success` and enters Reader;
- Provider stage timeout does not reduce Reader or Judge time;
- Reader partial completion preserves readable documents;
- Query round-robin candidate selection and cross-Provider URL caps;
- Judge support rows without relevance;
- empty support is a valid negative row;
- malformed/unknown/duplicate/missing rows are Candidate-isolated;
- Judge root failure differs from all-valid-negative output;
- primary Source cannot bypass empty topic support;
- stale support cannot enter the admitted set;
- `PARTIAL` is derived only from meaningful material-topic coverage;
- Judge failure cannot trigger search;
- one Judge call per round;
- exact Query/URL/Read/Round/Repair caps;
- Privacy-safe Trace and evaluator closure;
- existing Freshness, Source, Conflict, Citation, Answer Policy, Validation,
  Rendering, and high-consequence invariants.

No existing test method or material case may be deleted merely to obtain a
green suite.

### 12.2 Known-issue regression set

Questions already used in chat, design, debugging, prompts, or fixtures may be
used only to prove that known root causes are closed. They cannot certify live
search quality.

### 12.3 Fresh blind online set

Real-search acceptance must use questions that were not present in:

- the user's chat history;
- this design or its implementation plan;
- prompts, fixtures, tests, evaluation JSONL, commit messages, or development
  reports;
- any diagnostic run performed during implementation.

The blind set is generated and sealed only after production implementation is
complete. The implementer must not see the exact questions while writing code.
An independent reviewer reveals and runs them. It must cover, using new entities
and facts:

- a current single factual answer;
- a current release/version fact;
- a current event result;
- a complete multi-fact schedule/list;
- an official announcement;
- a comparison requiring multiple topics;
- DDGS failure with successful Tavily fallback;
- sibling Query partial success;
- partial Reader completion;
- partial Judge-row failure.

Each question is manually reviewed for whether the cited Evidence actually
supports the visible claim. Passing requires:

- any usable completed hits prevent a request-wide Provider timeout;
- any readable completed Candidate prevents a Reader-wide no-content result;
- one valid Judge row survives sibling row failure;
- sufficient Evidence yields an answer with only actual sources;
- partial Evidence yields a meaningful supported subset;
- insufficient Evidence stays conservative;
- normal successful searches show no status banner;
- stage and total latency remain within the derived policy maximum.

The blind questions and outcomes are recorded only after execution. Hermetic
fixtures, known chat examples, and unreviewed online output cannot be described
as real-search quality certification.

### 12.4 External quality disclaimer

Passing architecture and hermetic gates proves contract conformance. It does
not prove general internet quality. Real DDGS/Tavily quality remains a separate,
manually reviewed external gate and must be reported honestly, including any
failed blind case.
