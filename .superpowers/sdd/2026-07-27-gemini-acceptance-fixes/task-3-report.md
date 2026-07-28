# Task 3 Report — Memory Command Authorization and Exact Outcomes

## Status

Implemented Task 3 on `codex/structured-adaptive-memory` from base
`73d03f63794e8a837f78837a4962cd298e199e95`.

## Root causes

1. `src.main._process_message` created `CommandContext` with only the user,
   session, and text. The real callback `message_id` and already-derived
   `MemoryContext` never crossed the command boundary.
2. `/globalremember` constructed a `MemoryContext` whose session string looked
   global but whose `is_group=False` made `MemoryContext.primary_scope` return
   the administrator's private scope. The handler ignored `PolicyDecision`
   results and unconditionally reported success.
3. `/remember` likewise discarded policy results and reused `"cmd"` whenever
   the command context lacked an ID, so rejection/no-op and dedupe behavior
   could not be represented truthfully.
4. `/forget` treated the first 12 `MemoryRetriever` results as the authorization
   and identity set. Exact IDs outside that window were unreachable, while the
   single `_can_forget` boolean collapsed private privacy deletion, group
   retraction, subject dispute, foreign denial, and administrator deletion into
   one physical-delete path.
5. Group `/memories` reused answer retrieval, including private
   personalization intended only to shape replies, and then printed those
   private values publicly.
6. `render_command_outcome` was an identity function. Help/reset/memory
   outcomes were marked already rendered, unknown commands had no outcome, and
   there was no boundary preventing model prose from replacing deterministic
   status/scope/cause.

## TDD evidence

### Entry, scopes, IDs, and policy outcomes

RED:

```text
python -m unittest
  tests.test_main_image_flow.MainImageFlowTests.test_command_receives_real_message_id_and_memory_context
  tests.test_memory_commands.MemoryCommandsTests.test_remember_uses_private_or_current_group_scope
  tests.test_memory_commands.MemoryCommandsTests.test_admin_globalremember_creates_global_claim_without_private_claim
  tests.test_memory_commands.MemoryCommandsTests.test_real_command_message_ids_preserve_distinct_confirmation_evidence
  tests.test_memory_commands.MemoryCommandsTests.test_policy_rejection_is_not_reported_as_success -v

FAILED: global claim count 0; distinct command evidence incomplete;
CommandOutcome lacked status; the corrected main reproduction expected
message_id 98765 but received "".
```

GREEN:

```text
Ran 5 tests in 0.104s
OK
```

### Exact deletion, authorization, dispute, and listing privacy

RED:

```text
python -m unittest
  tests.test_memory_commands.MemoryCommandsTests.test_exact_id_outside_top_twelve_retracts_own_group_claim
  tests.test_memory_commands.MemoryCommandsTests.test_different_group_speaker_cannot_retract_foreign_claim
  tests.test_memory_commands.MemoryCommandsTests.test_group_claim_subject_creates_answer_suppression_not_retraction
  tests.test_memory_commands.MemoryCommandsTests.test_private_owner_forget_physically_deletes_body_and_keeps_audit_only
  tests.test_memory_commands.MemoryCommandsTests.test_admin_forget_physically_deletes_group_claim
  tests.test_memory_commands.MemoryCommandsTests.test_natural_description_requires_one_permitted_match
  tests.test_memory_commands.MemoryCommandsTests.test_group_memories_never_lists_private_personalization_value -v

Ran 7 tests; FAILED (failures=7).
```

GREEN:

```text
Ran 7 tests in 0.224s
OK
```

### Trusted persona rendering and search pass-through

RED:

```text
python -m unittest tests.test_command_renderer -v
ModuleNotFoundError: No module named 'src.commands.renderer'
```

GREEN:

```text
Ran 5 tests in 0.062s
OK
```

An additional coverage extension proved missing-target `/forget` bypassed the
renderer:

```text
test_help_reset_unknown_and_memory_outcomes_use_renderer ... FAIL
AssertionError: False is not true : forget
```

After the minimal fix:

```text
Ran 1 test in 0.029s
OK
```

### Independent self-review fix round

The read-only reviewer found two Critical and two Important issues. Each was
reproduced before changing production code:

```text
test_remember_calls_real_extractor_contract_with_memory_event
ERROR TypeError: MemoryExtractor.extract() got an unexpected keyword
argument 'text'

test_admin_delete_after_subject_dispute_removes_every_body_copy
FAIL: archived dispute body remained searchable after admin delete

test_group_author_retraction_is_one_conditional_store_mutation
ERROR AttributeError: MemoryStore has no retract_group_claim

test_physical_delete_cleanup_failure_reports_partial_and_is_retryable
ERROR RuntimeError: active reader
```

Minimal fixes:

- build the real `MemoryEvent` before extraction and call
  `MemoryExtractor.extract(event)`; the regression uses a fake LLM under the
  real extractor rather than mocking the method contract;
- replace the claim-shaped dispute marker with body-free
  `memory_subject_disputes` metadata and an `ON DELETE CASCADE` reference;
- add a single-transaction conditional group retraction and derive the outcome
  from rowcount;
- add `PhysicalDeleteOutcome`, return deterministic `partial` plus
  `retryable=true` after post-commit maintenance failure, and allow the store
  cleanup call to be retried until `cleanup_completed`.

GREEN:

```text
Ran 4 tests in 0.106s
OK

Ran 3 tests in 0.107s
OK
```

## Implementation summary

- Main now passes the callback ID and real `MemoryContext` into
  `CommandContext`.
- `MemoryPolicy.apply_command` keeps explicit commands in the caller's real
  scope. `apply_global_command(..., authorized=True)` is the only command path
  that overrides the write scope to `global:global`; the original event
  speaker/subject attribution remains intact.
- Command outcomes now carry deterministic `status`, `scope`, and `cause`.
  Replies are derived from actual `PolicyDecision` and store mutation returns.
- Missing command IDs are rejected instead of silently sharing a constant.
- `/forget <numeric ID>` resolves the exact permitted record through
  `get_claim` before any description matching. Natural descriptions enumerate
  the complete permitted current set and mutate only one unique match.
- Private owners and administrators physically delete through the existing
  secure deletion path and leave only ID/reason/time audit metadata. Cleanup
  failures return a truthful, retryable `partial` outcome instead of success;
  a later maintenance retry reports `cleanup_completed`. A
  non-admin group author changes the original claim to `retracted`. A foreign
  speaker is denied. A group claim's subject records an atomic body-free
  dispute row, changes the original to `disputed`, and suppresses it from
  answer retrieval without relabeling the original as retracted.
- Group `/memories` excludes all private-scope records, including safe
  personalization that remains available to answer generation.
- The persona model sees only `TrustedCommandFacts` and may return one of four
  tone labels. Deterministic code wraps the unchanged fallback. Free-form or
  hallucinated output, empty output, or renderer failure returns the exact
  fallback. Search remains `already_rendered` and bypasses this renderer.

## Verification

Focused affected suite:

```text
python -m unittest tests.test_memory_commands tests.test_command_renderer
  tests.test_product_scope tests.test_user_facing_scope
  tests.test_main_image_flow tests.test_memory_policy tests.test_memory_store
  tests.test_memory_retrieval -v

Ran 155 tests in 13.954s
OK
```

The first bare discovery run failed during imports because this linked
worktree has no `.env` and that process had no required `CHAT_MODELS`; 37
errors had the same `ModelConfigurationError: CHAT_MODELS 不能为空` root cause,
with no failed code assertions. Re-running with isolated test-only values:

```text
$env:CHAT_MODELS='gemini:test-model'
$env:GEMINI_API_KEY='test-key'
$env:MEMORY_MODELS=''
python -m unittest discover -s tests -v

Ran 334 tests in 16.798s
OK
```

Static/scope checks:

```text
python -m py_compile src/commands/__init__.py src/commands/renderer.py
  src/main.py src/memory/policy.py src/memory/store.py
  src/memory/retriever.py
exit 0

git diff --check
exit 0
```

## Changed files

- `src/main.py`
- `src/commands/__init__.py`
- `src/commands/renderer.py`
- `src/memory/policy.py`
- `src/memory/models.py`
- `src/memory/retriever.py`
- `src/memory/store.py`
- `tests/test_main_image_flow.py`
- `tests/test_memory_commands.py`
- `tests/test_command_renderer.py`
- `.superpowers/sdd/2026-07-27-gemini-acceptance-fixes/task-3-report.md`

Explicitly preserved and excluded from staging: `tests/test_model_config.py`,
`docs/plans/`, the real environment, legacy migration, and unrelated queue or
search code.

## Self-review and concerns

- Rechecked every brief bullet against a direct behavior test and inspected the
  complete Task 3 diff. No known Critical or Important correctness issue
  remains.
- The persona layer adds one small model call to non-search commands in the
  real callback path. If it is slow, unavailable, or returns anything except a
  tone label, the exact fallback is returned; mutations already completed.
- Subject-dispute suppression is represented by body-free metadata linked to
  the target with `ON DELETE CASCADE`. The original body remains until a
  privacy/admin deletion because dispute itself is not deletion; no duplicate
  body or FTS row is created.
- Exact `/forget` authorization intentionally considers current
  `active`/`disputed` records. Already retracted/archived records are not
  redisclosed by the command.
