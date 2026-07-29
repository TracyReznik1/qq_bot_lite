# Structured Memory Final Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:systematic-debugging. This document is one final-review fix wave and must be implemented in the listed order, verified as a whole, and committed once.

**Goal:** Make the structured memory implementation conform to the already-approved design under real production object combinations, close privacy-deletion and lifecycle failures, and make the test suite hermetic.

**Architecture:** Treat `untrusted extraction → deterministic validation → transactional claim/job update` as one boundary. Durable jobs own receive ordering and temporary source text; policy owns deterministic authorization and minimal evidence; physical deletion atomically removes or tombstones every durable source that could recreate or retain the deleted content. Retrieval uses one initialized store, permission-filtered SQL/FTS candidates, reserved identity/preference matches, and conflict-group closure.

**Tech Stack:** Python 3.13, standard-library `sqlite3`, `threading`, `unittest`, Flask/OneBot, Gemini native `generateContent`.

## Global Constraints

- Work only in `D:\Desktop\qqbot_lite\.worktrees\structured-adaptive-memory` on `codex/structured-adaptive-memory`.
- Starting HEAD is `9d408eb39925da24b1afe4ed6bd347db3e1f8035`.
- The approved behavior remains defined by `docs/superpowers/specs/2026-07-26-structured-adaptive-memory-design.md`.
- Rewriting affected memory modules is authorized; unrelated chat, search, image, OneBot, provider, and history behavior must remain unchanged.
- Do not read or modify the real `.env`, existing runtime databases, or `tests/test_model_config.py`.
- Hard secrets are never stored. Permission checks occur before ranking, formatting, model input, listing, deletion, or retry.
- Use real object combinations in regression tests. Mock only the external LLM/HTTP boundary, not the interface under test.
- Every production change needs a regression test that first fails for the expected reason.
- Runtime logs and command-renderer inputs may contain only body-free metadata.
- Use standard-library components only.

---

## Task 1: Make Background Learning a Real Atomic Contract

**Files:**

- Modify: `src/memory/service.py`
- Modify: `src/memory/extractor.py` only if an adapter is genuinely needed
- Modify: `src/memory/policy.py`
- Modify: `src/memory/store.py`
- Modify: `src/messaging.py`
- Modify: `src/main.py`
- Test: `tests/test_memory_service.py`
- Test: `tests/test_memory_end_to_end.py`
- Test: `tests/test_messaging.py`

**Interfaces:**

- Production extraction call:

```python
candidates = extractor.extract(event, image_data_urls=images)
```

- Policy application may accept an optional durable job ID, but claim reconciliation and final `done` state must commit in the same SQLite transaction:

```python
decision = policy.apply(event, candidates, complete_job_id=job.id)
```

- Store lifecycle APIs:

```python
store.recover_staged_jobs() -> int
store.max_job_sequence() -> int
message_queue.ensure_sequence_at_least(sequence: int) -> None
```

- [ ] Add a real `MemoryService + MemoryExtractor(fake LLM)` test. Verify the current call raises `TypeError`, then correct the production call and verify a Claim is created and the job becomes `done`.
- [ ] Add a test where policy writes succeed but job completion raises. Verify the transaction rolls back both the Claim and job completion; a retry produces exactly one active Claim.
- [ ] Move job completion into the policy/store transaction. Remove the separate post-policy `complete_job()` success boundary.
- [ ] Add restart tests for:
  - first job `staged`, second job `ready`: startup promotes or otherwise safely recovers the stranded text-only job and the scope makes progress in receive order;
  - first `start()` initialization failure: `_running` is false, workers are empty, and a second `start()` retries initialization;
  - existing durable sequence 100 plus a new callback after restart: the new receive sequence is greater than 100 and cannot overtake the recovered job.
- [ ] Initialize the receive-sequence high-water mark before an accepted callback can be assigned a sequence. Do not seed after enqueue.
- [ ] Keep same-session reply FIFO and cross-user reply concurrency unchanged.
- [ ] Run focused service, end-to-end, messaging, and main tests.

## Task 2: Make Physical Deletion Complete and Final

**Files:**

- Modify: `src/memory/store.py`
- Modify: `src/memory/service.py`
- Modify: `src/memory/policy.py`
- Modify: `src/commands/__init__.py`
- Test: `tests/test_memory_store.py`
- Test: `tests/test_memory_commands.py`
- Test: `tests/test_memory_end_to_end.py`

**Interfaces:**

- Every durable job must have an indexed body-free `source_message_id` association. Use a schema migration and backfill existing rows from valid payload JSON.
- Physical deletion outcome remains structured:

```python
PhysicalDeleteOutcome(
    status: str,
    row_deleted: bool,
    cleanup_complete: bool,
    retryable: bool,
)
```

- [ ] Add an end-to-end test: real job → real policy Claim → `done` → owner `/forget`. Assert Claim/FTS are gone, associated job text is empty, same-source evidence excerpts cannot contain the marker, and the marker is absent from the checkpointed SQLite bytes.
- [ ] Add retry resurrection test: policy commit is forced to roll back with job completion, then `/forget`/authorized deletion terminates or tombstones the source. A subsequent worker retry or duplicate callback cannot recreate the Claim.
- [ ] Add a same-message multi-Claim test. Deleting one private Claim must remove the source message body and all same-source excerpts that could retain that Claim's text without deleting unrelated sibling Claim values.
- [ ] In the physical-delete transaction:
  - authorize by owner/admin independently of lifecycle;
  - clear associated durable job payload text;
  - terminally cancel unfinished associated jobs or enforce a durable deletion tombstone;
  - clear same-source evidence excerpts and synchronize FTS;
  - preserve only body-free audit/pending-cleanup metadata.
- [ ] Allow private owner/admin exact-ID physical deletion of `active`, `disputed`, `retracted`, `superseded`, and `archived` Claims. Listing and answer retrieval remain current-state only.
- [ ] Keep post-commit optimize/checkpoint retry semantics truthful.
- [ ] Run store, command, and end-to-end deletion tests.

## Task 3: Replace Label-Trust with Deterministic Privacy Validation

**Files:**

- Modify or create a focused validator under `src/memory/`
- Modify: `src/memory/policy.py`
- Modify: `src/memory/retriever.py`
- Modify: `src/memory/models.py` only if an explicit learning-mode field is needed
- Test: `tests/test_memory_policy.py`
- Test: `tests/test_memory_retrieval.py`
- Test: `tests/test_memory_commands.py`

**Interfaces:**

```python
class LearningMode(str, Enum):
    AUTOMATIC = "automatic"
    EXPLICIT_PRIVATE = "explicit_private"
    EXPLICIT_GROUP = "explicit_group"
    EXPLICIT_GLOBAL = "explicit_global"

def classify_sensitive_text(text: str) -> Sensitivity: ...
def safe_group_personalization(claim: MemoryClaim) -> bool: ...
def minimal_claim_excerpt(event: MemoryEvent, candidate: MemoryCandidate) -> str: ...
```

- [ ] Add tests proving the same exact address or health marker is rejected from automatic group learning when predicates are `home_address`, `lives_at`, `condition`, `fact`, or `contact_point`.
- [ ] Add tests proving hard secrets are rejected in every mode.
- [ ] Add tests proving explicit `/remember` in the current group may store deliberately shared sensitive personal information while remaining group-scoped and attributed; this exception never applies to credentials, verification codes, payment data, or private keys.
- [ ] Add a mixed-message test: harmless `likes=苹果` plus an exact address. If the harmless Claim is retained, its evidence excerpt contains only the minimal harmless value and never the address or full source sentence.
- [ ] Implement value/source validation independent of model-supplied predicate and type. Keep explicit and automatic learning modes distinct.
- [ ] Store only a bounded, sanitized per-Claim excerpt; never use the complete event text as evidence.
- [ ] Add private→group tests where sensitive identity, address, health, relationship, or arbitrary long text is mislabeled `preferred_name`/`response_style`. None may enter the group prompt.
- [ ] Accept group personalization only when:
  - the subject is the current QQ;
  - preferred name is a short safe name; or
  - response style is a short safe expression preference;
  - no sensitive/hard-secret marker, URL, newline, identity fact, health fact, address, or relationship assertion is present.
- [ ] Apply the same validator to old stored Claims during retrieval, not only new writes.

## Task 4: Rebuild Retrieval Around Permission SQL, FTS, and Reserved Facts

**Files:**

- Modify: `src/memory/retriever.py`
- Modify: `src/memory/store.py`
- Modify: `src/memory/service.py` or a focused store-provider module
- Modify: `src/commands/__init__.py`
- Test: `tests/test_memory_retrieval.py`
- Test: `tests/test_memory_commands.py`
- Test: `tests/test_memory_store.py`

**Interfaces:**

```python
store.search_authorized_claims(
    context: MemoryContext,
    query: str,
    *,
    limit: int,
) -> tuple[MemoryClaim, ...]

store.list_authorized_claims(
    context: MemoryContext,
    *,
    include_private_personalization: bool,
) -> tuple[MemoryClaim, ...]
```

- [ ] Use one already-initialized runtime store. A normal chat retrieval must not call `initialize()` or create a new default store.
- [ ] Apply permission SQL first, then bounded FTS/structured candidate selection. Add a test that runtime retrieval calls the FTS-backed API for a normal lexical query.
- [ ] Add a high-cardinality test with more than 12 newer unrelated same-subject Claims. An old exact `likes` preference and preferred name must still be included for “我喜欢什么/我叫什么”.
- [ ] Rank additively: age affects only recency, not the whole relevance score. Preferred names have no age penalty; preferences retain the configured floor.
- [ ] Add conflict-closure tests with more results than the limit. A selected disputed Claim must include all required counterpart Claims and relation/status attribution, or the entire conflict group is omitted. Never include one side as an ordinary fact.
- [ ] Populate `relation_types` and render `status=disputed` plus speaker attribution.
- [ ] Replace `/memories` use of the answer retriever with `list_authorized_claims()`. It must not become falsely empty after filtering private personalization from a top-12 answer set.
- [ ] Preserve group/private/global hard boundaries and the approved conservative private fallback.

## Task 5: Make Explicit Commands Accurate, Body-Free, and Fully Persona-Aware

**Files:**

- Modify: `src/commands/__init__.py`
- Modify: `src/commands/renderer.py`
- Modify: `src/main.py`
- Test: `tests/test_memory_commands.py`
- Test: `tests/test_command_renderer.py`
- Test: `tests/test_user_facing_scope.py`

**Interfaces:**

```python
CommandOutcome(
    status="failed",
    cause="extractor_unavailable" | "store_unavailable" | "provider_unavailable",
    facts=("retryable=true",),
)
```

- [ ] Add real command-path tests where extractor, provider, and store fail. `/remember` and `/globalremember` must return deterministic body-free failure outcomes and must not be reported as configuration errors.
- [ ] Remove raw command text and memory values from renderer facts. Facts contain only status, scope, cause, IDs where permitted, and retryability.
- [ ] Inject the complete `persona.content`, not only `persona.name`, into the tone-rendering prompt. The model still cannot rewrite deterministic facts.
- [ ] Add tests that persona-content markers reach the renderer, while secret/user-memory markers do not.
- [ ] Preserve already-rendered search results and exact fallback behavior.

## Task 6: Make Tests Hermetic Before Trusting the Suite

**Files:**

- Modify: `tests/__init__.py`
- Create or modify: focused helpers under `tests/`
- Modify: `tests/test_data_migration.py`
- Modify: `tests/test_main_image_flow.py`
- Modify: affected chat/service tests

- [ ] Add a test-runtime helper that binds `DATA_DIR` to a `TemporaryDirectory`, resets config/store/service/LLM/persona singletons, and stops workers during cleanup.
- [ ] Apply it before any test can call `startup()`, `process_message()`, `generate_reply()`, or a default `MemoryRetriever`.
- [ ] Direct message-flow tests inject a fake service unless the real service is the behavior under test.
- [ ] Add a guard that fails if any test opens or writes the repository/default `qqbot_data/memory.sqlite3`.
- [ ] Verify the full test suite makes no provider network request. Mock only the external transport boundary.
- [ ] Run the full suite twice in isolated temporary data roots to expose leaked singleton/worker state.

## Task 7: Align Operations, Documentation, and Minor Semantics

**Files:**

- Modify: `src/memory/service.py`
- Modify: `src/memory/store.py`
- Modify: `README.md`
- Modify: `tests/test_readme_guide.py`
- Modify: relevant maintenance tests

- [ ] Make `wait_for_scope()` include retry as unfinished, or return an explicit state that cannot be mistaken for completion.
- [ ] Prevent cleanup from rewriting already-empty old job payloads. Use an actual-change predicate.
- [ ] Avoid repeated FTS optimize for already-maintained archived rows by recording body-free maintenance completion while preserving retry after failure.
- [ ] Update README with:
  - automatic learning adds a background model call and cost;
  - eventual consistency and durable memory jobs;
  - attributed conflicts and correction/retraction/dispute/delete differences;
  - private→group personalization limits;
  - sensitive filtering and explicit group-memory behavior;
  - ephemeral images and physical deletion semantics;
  - old JSON memories are ignored;
  - chat reply queue is process-memory while memory-learning jobs are SQLite-durable.
- [ ] Remove or correct the statement that the project has no persistent task queue.

## Final Verification and Commit

- [ ] Run each focused RED before its production change and record the expected failure.
- [ ] Run focused GREEN after each workstream.
- [ ] Run:

```powershell
$env:CHAT_MODELS='gemini:test-model'
$env:MEMORY_MODELS=''
$env:GEMINI_API_KEY='test-key'
python -m unittest discover -s tests -t . -v
python -m unittest discover -s tests -t . -v
python -m compileall -q src tests run_bot.py
git diff --check
```

- [ ] Run independent probes:
  - real service/extractor contract;
  - restart staged/high-water ordering;
  - automatic sensitive-label bypass;
  - private→group personalization leak;
  - high-cardinality old preference;
  - auto-job physical delete and disk marker;
  - retry→delete→retry no resurrection;
  - real startup with temporary data and SQLite integrity.
- [ ] Static-search logs, renderer facts, README, `.env.example`, and runtime paths for message text, secrets, `memory.db`, `MEMORY_LIMIT`, and legacy memory imports.
- [ ] Inspect staged paths. Exclude the real `.env`, runtime data, `tests/test_model_config.py`, and unrelated files.
- [ ] Commit the complete final hardening wave with:

```powershell
git commit -m "fix: harden structured memory production boundaries"
```
