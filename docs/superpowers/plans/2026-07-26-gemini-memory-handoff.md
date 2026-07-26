# Gemini Structured Memory Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven development and execute this plan task-by-task. Every task must be independently reviewed before starting the next task.

**Goal:** Complete the remaining retrieval, background-learning, command, legacy-removal, and documentation work for the structured adaptive memory system without changing unrelated product behavior.

**Architecture:** The project is a lightweight OneBot QQ bot. Chat replies, web search, and image understanding remain the only product capabilities outside memory management. A standalone Markdown persona is injected into normal model calls. Structured memories are stored as attributed, scoped, conflict-aware SQLite claims; model extraction produces untrusted candidates, while deterministic policy, permission filtering, and transactional storage decide what can be saved or shown.

**Tech Stack:** Python 3.13, Flask, OneBot HTTP callbacks, standard-library `sqlite3`, `threading`, `concurrent.futures`, Gemini native `generateContent`, DeepSeek clients, `unittest`, Markdown and dotenv configuration.

## Global Constraints

- Work only in `D:\Desktop\qqbot_lite\.worktrees\structured-adaptive-memory` on branch `codex/structured-adaptive-memory`.
- The required starting commit is `a3a70e7d6b53f25cc574237afbd75174757fb8c0`. Stop and report the mismatch if HEAD is different and the extra commits are not documentation-only handoff commits.
- Do not edit the main checkout at `D:\Desktop\qqbot_lite`.
- Do not read, copy, print, rewrite, stage, or commit the real ignored `.env`. It contains user secrets. Only `.env.example` and README documentation are in Gemini's scope.
- Do not stage `tests/test_model_config.py` unless a remaining task demonstrably requires a content change. At handoff it appears modified because of Windows index/line-ending metadata, but `git diff`, `git diff --raw`, and `git diff --numstat` show no content change.
- Preserve only these product capabilities: chat, web search, image understanding, `/remember`, `/globalremember`, `/memories`, `/forget`, `/reset`, `/help`, and search aliases already approved by the product scope tests.
- Do not reintroduce ATRI as a hard-coded runtime identity. Runtime identity comes only from `config/persona.md`.
- Every normal user-facing reply must follow the persona. Accurate and transparent operation or failure status takes priority over roleplay.
- Do not suggest memory commands during ordinary conversation. Explain them only after `/help`, direct feature questions, or explicit command use.
- Private, each group, and global memories are distinct. Permission filtering must happen before relevance ranking, formatting, or model input.
- Group A data must never reach group B. Another user's private data must never become group factual evidence.
- The current sender's private preferred name or response-style preference may personalize a group reply, but it must never be exposed as a public fact.
- `/globalremember` is the only promotion path to global scope and is restricted to `ADMIN_QQ_IDS`.
- Automatic learning evaluates ordinary private messages and accepted group @Bot messages. It must not store every message indiscriminately.
- Automatic learning is reply-first and eventually consistent. Explicit memory commands wait for their confirmed write result.
- Raw images, image data URLs, secrets, credentials, verification codes, private keys, payment data, and private excerpts must not enter long-term memory or logs.
- Preferred names do not decay. Preferences keep a non-trivial retrieval floor until corrected or retracted.
- Runtime must not read, copy, merge, or migrate `qqbot_data/memories/*.json`.
- Use only the standard library for storage, FTS, queues, migrations, and concurrency. Add no dependency.
- Use TDD for every behavioral change: add a failing regression test, verify the expected failure, implement the smallest correction, then run focused and full tests.
- Do not refactor unrelated code, rename unrelated interfaces, reformat whole files, normalize unrelated line endings, or update documentation for features that do not exist.
- Commit each task separately with only its listed files. Before every commit, inspect staged paths and run `git diff --cached --check`.
- After Task 8, stop. Do not merge, rebase, push, delete branches, or modify the real `.env`. Codex will perform the final independent acceptance and integration after the user returns.

---

## Project Background and Decisions

The original flat JSON memory design could confuse one person's statement with another person's identity. The replacement design treats memory as attributed claims:

- Every claim records scope, speaker QQ, subject, predicate, value, type, modality, truth confidence, attribution confidence, lifecycle status, validity interval, and evidence.
- Different speakers may hold conflicting claims at the same time. Conflicts are preserved with directed relations instead of forcing one global truth.
- The same speaker can confirm, explicitly correct, or explicitly retract their own earlier claim.
- First-person statements are resolved to the real sender, not to whoever later asks a question.
- Group memory is visible only inside that group. Private memory remains private. Global memory remains attributed to its speaker and subject.
- When one side lacks reliable identity/preference information, narrowly defined fallback may consult the same user's other scope. Fallback does not copy or promote the claim.
- The model may ask naturally for missing information according to the persona. It must not expose internal memory mechanics during normal conversation.

Authoritative design and full original plan:

- `docs/superpowers/specs/2026-07-26-structured-adaptive-memory-design.md`
- `docs/superpowers/plans/2026-07-26-structured-adaptive-memory.md`

This handoff document is authoritative where it narrows or corrects the older plan.

## Current Progress

### Completed and independently reviewed

1. Standalone persona loading
   - Commits: `f92b146`, `f30fbd8`
   - Runtime identity comes from `config/persona.md`.
   - Missing, unreadable, empty, or nameless persona files fail with a clear configuration error.

2. Optional memory model chain
   - Commit: `4e3949e`
   - `MEMORY_MODELS` is optional and reuses the validated `CHAT_MODELS` tuple when blank.
   - Chat and memory clients use separate cached fallback clients.

3. SQLite claim ledger and durable jobs
   - Commits: `137c287` through `2cf4b69`
   - Includes immutable domain models, schema versioning, claims, evidence, relations, FTS, durable jobs, retry state, exact physical deletion, recovery API, UTC retry times, payload redaction, and transaction-safe FTS synchronization.

4. Extraction, entity resolution, and conflict policy
   - Commits: `c1c19cc`, `974c1df`, `a3a70e7`
   - Strict extractor JSON with one repair attempt.
   - Deterministic attribution, scope, operation authorization, lifecycle, sensitive filtering, exact scoped lookup, and single-transaction reconciliation.
   - Same-speaker correction/retraction and different-speaker support/conflict relations are covered.
   - Latest focused evidence: policy `47/47`; memory + image/OneBot `122/122`.
   - Latest full evidence: `252/253`; the only expected failure is README not yet documenting `MEMORY_MODELS`, which Task 8 must close.

### Not started

- Task 5: permission-first retrieval and prompt integration.
- Task 6: durable background learning and message integration.
- Task 7: scoped memory commands and persona rendering.
- Task 8: legacy removal, documentation, and complete acceptance.

## Starting Verification

- [ ] Confirm branch and starting history:

```powershell
git branch --show-current
git log --oneline -12
git status --short
```

Expected branch: `codex/structured-adaptive-memory`.

- [ ] Confirm no content diff exists for the pre-existing status noise:

```powershell
git diff --raw -- tests/test_model_config.py
git diff --numstat -- tests/test_model_config.py
git diff -- tests/test_model_config.py
```

Expected: no content diff. Do not stage or rewrite the file.

- [ ] Establish the inherited test baseline with process-only dummy configuration:

```powershell
$env:CHAT_MODELS='gemini:test-model'
$env:MEMORY_MODELS=''
$env:GEMINI_API_KEY='test-key'
python -m unittest discover -s tests -t . -v
```

Expected before Task 8: all tests pass except `tests.test_readme_guide.ReadmeGuideTests.test_readme_reference_matches_runtime_environment_variables`, because README does not yet list `MEMORY_MODELS`.

---

### Task 5: Permission-First Retrieval and Prompt Integration

**Files:**

- Create: `src/memory/retriever.py`
- Modify: `src/chat/prompt.py`
- Modify: `src/chat/chat_service.py`
- Create: `tests/test_memory_retrieval.py`
- Modify only as required: `tests/test_multimodal_chat.py`, `tests/test_chat_tool_finalization.py`, `tests/test_user_facing_scope.py`

**Interfaces:**

- Produce `MemoryRetriever.retrieve(context, query, limit=12) -> tuple[RetrievedMemory, ...]`.
- Produce `format_memory_context(results) -> str`.
- Change `generate_reply(context: MemoryContext, text: str, tool_context: str = "", image_data_urls: list[str] | None = None) -> str`.
- Preserve `context.session_key` as the conversation-history key.

- [ ] Write failing privacy tests for private A/private B, group 1/group 2, global attribution, old preferred names, old preferences, disputed claims, and excluded lifecycle states.

- [ ] Prove the tests fail before adding `MemoryRetriever`.

- [ ] Implement SQL hard-scope filtering before FTS, scoring, alias resolution, formatting, or model input:

  - Group factual evidence: current `group:<group_id>` plus attributed global claims.
  - Private factual evidence: current user's private scope plus attributed global claims.
  - Private fallback from group: only claims where `speaker_qq == subject_id == current user`, and only identity, preferred-name, or preference types.
  - Group personalization from private: only current user's preferred name and response-style preferences, marked `usage="personalization"`.
  - Never retrieve another user's private record for group evidence.
  - Never retrieve another group's claim.

- [ ] Resolve first-person identity queries to the current QQ. Resolve explicit aliases in the current group before global scope. Never use a false-unique bounded query; use the exact unbounded scoped APIs added in Task 4 when identity resolution affects permissions or semantics.

- [ ] Rank only the already-authorized set using subject, predicate, direct scope, confidence, confirmation recency, and FTS relevance. Preferred names have no age penalty. Preferences retain the documented score floor.

- [ ] Exclude retracted, superseded, archived, hard-deleted, expired-current-state, and otherwise non-current claims from normal current-fact retrieval. Preserve disputed claims with attribution instead of silently selecting one speaker.

- [ ] Format two separate prompt blocks:

```text
[允许使用的记忆证据]
- 作用域=group:1；发言者=A；主体=A；类型=opinion；内容=……
[/允许使用的记忆证据]
[仅用于称呼和表达的个性化信息]
- 主体=当前发言者；首选称呼=安安
禁止把本区内容作为公开身份、经历或关系事实。
[/仅用于称呼和表达的个性化信息]
```

- [ ] Set prompt priority to capability/safety, privacy/permissions, persona, then untrusted evidence. Remove all flat-memory reads and old labels such as `个人基础信息` and `当前会话记忆`.

- [ ] Preserve web-search tool loops, provider affinity, history behavior, and multimodal content exactly.

- [ ] Run:

```powershell
python -m unittest tests.test_memory_retrieval tests.test_multimodal_chat tests.test_chat_tool_finalization tests.test_user_facing_scope -v
```

- [ ] Run full discovery. Before Task 8, only the known README/MEMORY_MODELS failure is allowed.

- [ ] Commit only Task 5 files:

```powershell
git add src/memory/retriever.py src/chat/prompt.py src/chat/chat_service.py tests/test_memory_retrieval.py tests/test_multimodal_chat.py tests/test_chat_tool_finalization.py tests/test_user_facing_scope.py
git diff --cached --check
git commit -m "feat: retrieve memories with hard scope filters"
```

**Task 5 acceptance gate:**

- Another user's private identity never appears in a group prompt.
- Group 1 data never appears in group 2.
- Global first-person memory remains attributed to its real speaker and subject.
- Private personalization in a group cannot be formatted as factual evidence.
- Preferred names and preferences remain retrievable after long time intervals.
- Search tools and image messages still work.

---

### Task 6: Durable Background Learning and Message Integration

**Files:**

- Create: `src/memory/service.py`
- Modify: `src/main.py`
- Modify: `src/messaging.py`
- Modify: `src/services/onebot_client.py`
- Create: `tests/test_memory_service.py`
- Create: `tests/test_memory_end_to_end.py`
- Modify only as required: `tests/test_messaging.py`, `tests/test_main_image_flow.py`

**Interfaces:**

- Produce `MemoryService.start()`, `stop()`, `stage_event(event) -> int`, `release_job(job_id, image_data_urls=())`, and `wait_for_scope(scope_key, timeout)`.
- Produce `get_memory_service() -> MemoryService`.
- Expose `MemoryService.store: MemoryStore`.
- Add `MemoryStore.integrity_check() -> str` using `PRAGMA integrity_check`; test that a healthy initialized database returns `ok`.
- Add an internal monotonic `_qqbot_sequence: int` to accepted callback dictionaries.

- [ ] Write failing tests for staged/ready jobs, reply-first release, restart recovery, transient retry, permanent failure, dedupe, images, logs, and ordering.

- [ ] Implement `stage_event()` so it durably stores only text and minimal metadata in `staged`. Implement `release_job()` so raw image data exists only in an in-memory map, the job changes to `ready`, and the method returns without waiting for extraction.

- [ ] Correct the current cross-task ordering mismatch:

  - Chat reply FIFO remains keyed by the existing conversation session key, including `group:<group_id>:<user_id>`, so different group users may receive replies concurrently.
  - Shared memory commit FIFO must be keyed by the memory scope: `group:<group_id>` for every user in that group and `private:<user_id>` for private chat.
  - Current `MemoryStore.create_job()` uses `context.session_key` as `scope_key`; update it and its tests so group jobs use the shared group memory key while preserving the original conversation `session_key` inside the payload/context.
  - One worker may write a given memory scope at a time. Different groups and private users may learn concurrently.
  - This correction is mandatory because group claims are shared and must follow the group's receive sequence.

- [ ] In `MessageQueue.enqueue`, assign a lock-protected process-wide monotonic receive sequence before queueing. Do not change existing callback dedupe or reply FIFO.

- [ ] On an accepted ordinary private message or accepted group @Bot message, build `MemoryContext`, resolve reply author through `get_message_author()` when applicable, and stage before chat generation. Release in `finally` after the reply attempt with the already-loaded ephemeral image data.

- [ ] Never auto-enqueue slash commands. Task 7 commands use synchronous service methods.

- [ ] Retry transient failures after 2, 10, and 30 seconds; fail after four total attempts. Store only the error class name. Recover abandoned `running` jobs with the existing atomic recovery API.

- [ ] Always remove ephemeral images in `finally`, on success and failure. A restart may recover text and metadata but never raw images.

- [ ] Initialize persona, SQLite, and MemoryService exactly once under `_startup_lock`. Health output may expose worker status and failed-job counts, never content.

- [ ] Implement the 90-day cleanup for completed job payloads and archived source excerpts without removing claim bodies that are still current.

- [ ] Verify logs contain job IDs, scope keys, attempt counts, and error class names only. They must not contain message text, claim values, image/base64 data, API keys, tokens, or private excerpts.

- [ ] Run:

```powershell
python -m unittest tests.test_memory_service tests.test_memory_end_to_end tests.test_messaging tests.test_main_image_flow -v
```

- [ ] Required concurrency assertions:

  - Same user's replies remain FIFO.
  - Different private users reply concurrently.
  - Different users in one group may reply concurrently but their shared group-memory commits follow receive sequence.
  - Duplicate callbacks create one job and one claim graph.
  - A blocked extractor does not delay a reply.

- [ ] Run full discovery, then commit only Task 6 files:

```powershell
git add src/memory/service.py src/main.py src/messaging.py src/services/onebot_client.py src/memory/store.py tests/test_memory_service.py tests/test_memory_end_to_end.py tests/test_messaging.py tests/test_main_image_flow.py tests/test_memory_store.py
git diff --cached --check
git commit -m "feat: learn memories in durable background jobs"
```

**Task 6 acceptance gate:**

- Automatic learning is durable, asynchronous, ordered by memory scope, recoverable, deduplicated, and redacted.
- Chat success does not depend on extractor success.
- Reply-author attribution is used by the production background path.
- No raw image survives outside the in-memory release window.

---

### Task 7: Scoped Memory Commands and Persona Rendering

**Files:**

- Modify: `src/commands/__init__.py`
- Modify: `src/commands/help.py`
- Modify: `src/commands/reset.py`
- Modify: `src/chat/chat_service.py`
- Modify: `src/main.py`
- Modify: `src/memory/service.py`
- Create: `tests/test_memory_commands.py`
- Modify only as required: `tests/test_product_scope.py`, `tests/test_user_facing_scope.py`

- [ ] Extend `CommandContext` with `memory_context`, `message_id`, and `is_admin`.

- [ ] Add immutable `CommandOutcome(code, facts, fallback_reply, already_rendered=False)`.

- [ ] Write failing scope and permission tests:

  - Private `/remember` writes private scope for the sender.
  - Group `/remember` writes only the current group scope and attributes first person to the sender.
  - `/globalremember` writes global scope, remains attributed to the admin speaker/subject, and rejects non-admins.
  - `/memories` returns only records the caller may see.
  - `/forget` never deletes multiple ambiguous matches.
  - A user may retract their own authored group claim.
  - A different speaker cannot retract someone else's claim.
  - The subject of another speaker's group claim may dispute and suppress it for answers without falsifying the original speaker's retraction state.
  - An administrator may perform authorized physical deletion.
  - Hard secrets are rejected.

- [ ] Implement synchronous `remember`, `global_remember`, `list_memories`, and `forget` using the same extractor and deterministic policy as automatic learning. Commands wait for the real write outcome.

- [ ] Enforce deletion:

```python
def can_forget(actor, claim, is_admin):
    if is_admin:
        return True
    if claim.scope_type == "private":
        return claim.scope_id == actor
    return claim.speaker_qq == actor
```

- [ ] Resolve `/forget` by short ID first. A natural-language description may delete only when exactly one permitted match exists. Ambiguity must ask for clarification and make no mutation.

- [ ] Perform command mutation before persona rendering. Give the renderer a trusted exact status. If rendering fails, return a factual fallback containing the exact scope, status, and cause.

- [ ] Keep search results already rendered. Route help, reset, unknown commands, and memory outcomes through persona-aware rendering without allowing roleplay to change facts.

- [ ] Final supported command registry:

```python
{
    "help", "h", "search", "s", "reset",
    "remember", "memo",
    "globalremember", "gremember",
    "memories", "forget",
}
```

- [ ] `/reset` clears conversation history only. It does not delete structured memory.

- [ ] Run:

```powershell
python -m unittest tests.test_memory_commands tests.test_product_scope tests.test_user_facing_scope -v
```

- [ ] Run full discovery, then commit only Task 7 files:

```powershell
git add src/commands/__init__.py src/commands/help.py src/commands/reset.py src/chat/chat_service.py src/main.py src/memory/service.py tests/test_memory_commands.py tests/test_product_scope.py tests/test_user_facing_scope.py
git diff --cached --check
git commit -m "feat: manage scoped memories with commands"
```

**Task 7 acceptance gate:**

- Command scope and attribution are exact.
- Non-admin global writes are impossible.
- Users cannot delete another speaker's group statement.
- Privacy deletion removes body and FTS content while preserving only body-free audit metadata.
- Every normal command response follows persona without falsifying status.

---

### Task 8: Remove Legacy Memory Paths, Document, and Run Full Acceptance

**Files:**

- Delete: `src/chat/memory.py`
- Modify: `src/utils/data_migration.py`
- Modify: `src/main.py`
- Modify: `src/config.py`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/test_data_migration.py`
- Modify: `tests/test_readme_guide.py`
- Modify: `tests/test_product_scope.py`
- Modify only tests that still import removed flat-memory or persona configuration interfaces

- [ ] Write failing tests that prove runtime code has no `src.chat.memory` import, no `migrate_legacy_memory_files`, and README documents private/group command scope, persona files, and `MEMORY_MODELS`.

- [ ] Delete the flat JSON memory module and remove all callers.

- [ ] Remove legacy-memory copying/merging from data migration while preserving approved history migration. Remove `memory_limit`, `MEMORY_LIMIT`, and related arguments.

- [ ] Add a sentinel test proving startup neither opens nor imports a file under `qqbot_data/memories/`.

- [ ] Keep `/reset` as history-only reset.

- [ ] Update README with:

  - `config/persona.example.md` → `config/persona.md` setup.
  - Identity/persona are no longer stored in `.env`.
  - Required `CHAT_MODELS`; optional `MEMORY_MODELS` falling back to chat chain.
  - An additional background model call and possible cost for learned messages.
  - Private, current-group, global, fallback, and group-personalization privacy rules.
  - `/remember`, `/globalremember`, `/memories`, `/forget`, and `/reset`.
  - Eventual consistency, conflicts, attribution, sensitive filtering, ephemeral image handling, deletion, backups, and redacted logging.
  - SQLite location under `DATA_DIR`.
  - Old JSON memories are unsupported and ignored.

- [ ] Update `.env.example` only. Do not touch the real `.env`.

- [ ] Static searches:

```powershell
rg -n "BOT_NAME|BOT_PERSONA|MEMORY_LIMIT|src\\.chat\\.memory|migrate_legacy_memory_files" src tests README.md .env.example
rg -n "ATRI" src .env.example README.md
```

Expected: only intentional negative assertions in tests for the first search; no hard-coded runtime identity for the second. `config/persona.md` is intentionally excluded from the ATRI search.

- [ ] Run the complete suite:

```powershell
$env:CHAT_MODELS='gemini:test-model'
$env:MEMORY_MODELS=''
$env:GEMINI_API_KEY='test-key'
python -m unittest discover -s tests -t . -v
```

Expected: zero failures and zero errors. The former README/MEMORY_MODELS failure must now be closed.

- [ ] Run smoke checks:

```powershell
python -c "from src.persona import get_persona; from src.memory.service import get_memory_service; print(get_persona().name); print(get_memory_service().store.integrity_check())"
python -c "from src.main import startup; startup(); print('startup-ok')"
```

Expected: configured persona name, `ok`, and `startup-ok`; no legacy memory migration log.

- [ ] Inspect for secrets and unrelated changes:

```powershell
git diff --check
git status --short
git diff --stat
git diff --cached --name-only
```

The real `.env`, `qqbot_data/`, SQLite files, WAL/SHM files, image data, API keys, QQ IDs, backups, and the pre-existing line-ending-only `tests/test_model_config.py` state must not be staged.

- [ ] Commit cleanup and documentation:

```powershell
git add src/chat/memory.py src/utils/data_migration.py src/main.py src/config.py .env.example README.md tests
git diff --cached --check
git commit -m "docs: finalize adaptive memory migration"
```

**Task 8 acceptance gate:**

- No legacy JSON memory is read or migrated.
- Documentation exactly matches runtime configuration and privacy behavior.
- Full discovery has zero failures.
- Startup and SQLite integrity smoke checks pass.
- No secret, runtime data, or unrelated file is staged.

---

## Mandatory Review Between Tasks

After each task:

- [ ] Record the task base SHA and head SHA.
- [ ] Review the complete task diff, not only the last commit.
- [ ] Compare actual behavior with this handoff and the original design.
- [ ] Treat untrusted model output, privacy scope, transaction atomicity, and logs as security boundaries.
- [ ] Fix every Critical or Important issue before starting the next task.
- [ ] Re-run focused and full tests after fixes.
- [ ] Keep review notes outside tracked source files unless they belong in README or tests.

## Final Acceptance Checklist

- [ ] Private identity isolation passes.
- [ ] Per-group isolation passes.
- [ ] Global first-person attribution passes.
- [ ] Same-speaker confirmation, supersede, and retract pass.
- [ ] Different-speaker support and conflict preservation pass.
- [ ] Preferred-name no-decay and preference retrieval floor pass.
- [ ] Question, negation, hearsay, joke, quotation, and ambiguous-pronoun rejection pass.
- [ ] Secrets, payment data, credentials, images, and ownership boundaries pass.
- [ ] Same-user reply FIFO and different-user reply concurrency pass.
- [ ] Shared group-memory commits follow group receive order across users.
- [ ] Duplicate callback dedupe passes.
- [ ] Background retry, restart recovery, and ephemeral image cleanup pass.
- [ ] `/remember`, `/globalremember`, `/memories`, and `/forget` scope/permission tests pass.
- [ ] Persona is present in normal chat and command-rendering model calls.
- [ ] Fault fallbacks state exact status and cause.
- [ ] Old JSON memories are not opened, read, copied, merged, or migrated.
- [ ] README and `.env.example` match final behavior.
- [ ] Full unittest discovery passes with zero failures and zero errors.
- [ ] `git diff --check` and Python compilation pass.
- [ ] No real `.env`, secrets, runtime database, image data, QQ-private excerpt, or unrelated change is committed.

## Handoff Back to Codex

When Gemini completes Task 8, provide Codex with:

1. Final branch name and HEAD SHA.
2. One commit SHA per task and any fix commits.
3. Focused and full test commands with exact pass/fail counts.
4. Static-search and smoke-test outputs.
5. `git status --short`, `git diff --stat <handoff-base>..HEAD`, and staged-file status.
6. Every intentional deviation from this handoff, with technical evidence.
7. Any unverified real-provider or real-OneBot behavior.

Codex will then independently inspect the complete branch diff, run acceptance tests, review security/privacy boundaries, and report defects. Gemini must not merge, push, delete branches, or edit the real `.env`.

After the branch passes acceptance, Codex—not Gemini—will make the separately authorized real `.env` integration edit: remove only `BOT_NAME` and the complete quoted multiline `BOT_PERSONA` assignment, optionally add the user's chosen `MEMORY_MODELS`, and preserve every other line and secret exactly.
