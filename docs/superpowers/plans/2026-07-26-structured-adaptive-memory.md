# Structured Adaptive Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace flat JSON memories with a scoped, attributed, conflict-aware SQLite memory system and load every user-facing reply from the standalone persona file.

**Architecture:** A durable local SQLite claim ledger stores structured claims, evidence, relations, FTS entries, and background jobs. A provider-neutral extractor produces validated candidates; deterministic policy applies privacy, entity, conflict, lifecycle, and sensitive-data rules before storage. Retrieval hard-filters scopes before relevance ranking, while a background service keeps chat responsive and explicit commands wait for confirmed writes.

**Tech Stack:** Python 3.13, standard-library `sqlite3`, `dataclasses`, `threading` and `concurrent.futures`; Flask; existing Gemini native and DeepSeek clients; `unittest`; Markdown configuration.

## Global Constraints

- Preserve chat, web search, image understanding, conversation history, and memory commands; do not add unrelated product capabilities.
- Runtime persona source is only `config/persona.md`; remove `BOT_NAME` and `BOT_PERSONA` from `.env`, `.env.example`, and Python configuration.
- Missing, unreadable, empty, or nameless `config/persona.md` is a startup configuration error.
- Private, each group, and global memories remain distinct; hard permission filtering runs before semantic retrieval or model input.
- Group A data never reaches group B. Other users' private data never reaches a group answer.
- Current-sender private memory may affect group address and style only, never public factual disclosure.
- Only `/globalremember` promotes information to all sessions, and only `ADMIN_QQ_IDS` may execute it.
- Ordinary private messages and accepted group @Bot messages enter learning evaluation; not every message becomes a claim.
- Normal user-facing text follows the persona. Accurate operation or fault status takes precedence over roleplay.
- Automatic learning is reply-first and eventually consistent. Explicit memory commands wait for the write result.
- Raw images, secrets, credentials, and payment data never enter long-term memory or logs.
- Preferred names do not decay. Preferences keep a non-trivial retrieval floor until corrected or retracted.
- Runtime must not read or migrate `qqbot_data/memories/*.json`.
- Add no third-party dependency for storage, FTS, queues, or schema migration.
- Use TDD for every task and commit only the files named by that task.

---

## File Structure

### New runtime modules

- `src/persona.py`: load, validate, cache, and expose the standalone persona.
- `src/memory/__init__.py`: public memory package exports.
- `src/memory/models.py`: immutable contexts, events, candidates, claims, evidence, retrieval results, and command outcomes.
- `src/memory/store.py`: SQLite schema, transactions, FTS synchronization, durable jobs, claim CRUD, relations, and cleanup.
- `src/memory/policy.py`: scope selection, entity resolution, sensitive filtering, confidence class validation, conflict/update decisions, and lifecycle rules.
- `src/memory/extractor.py`: isolated LLM prompt, JSON parsing, one repair attempt, and multimodal candidate extraction.
- `src/memory/retriever.py`: hard permission filter, query hints, FTS/structured ranking, and prompt-safe evidence formatting.
- `src/memory/service.py`: synchronous command operations, durable background queue, retry policy, per-scope ordering, and ephemeral image cache.

### New tests

- `tests/test_persona_file.py`
- `tests/test_memory_store.py`
- `tests/test_memory_policy.py`
- `tests/test_memory_extractor.py`
- `tests/test_memory_retrieval.py`
- `tests/test_memory_service.py`
- `tests/test_memory_commands.py`
- `tests/test_memory_end_to_end.py`

### Existing files changed

- `src/config.py`: remove environment persona fields; parse optional `MEMORY_MODELS`; expose database path.
- `src/model_config.py`: generalize model-chain parsing and validation labels.
- `src/services/llm_client.py`: create a separate fallback client for memory extraction.
- `src/services/onebot_client.py`: resolve replied-message author for background entity resolution.
- `src/chat/prompt.py`: inject persona and filtered structured evidence; remove flat-memory reads.
- `src/chat/chat_service.py`: accept a conversation context, retrieve scoped memory, and render trusted command outcomes in persona.
- `src/main.py`: initialize persona/store/service, build contexts, stage/release learning jobs, and remove legacy-memory startup.
- `src/messaging.py`: attach a monotonic receive sequence without changing per-session FIFO behavior.
- `src/commands/__init__.py`: context-aware scopes, `/memories`, `/forget`, and trusted outcomes.
- `src/commands/help.py`: list the final command set without reading `BOT_NAME`.
- `src/commands/reset.py`: reset history only.
- `src/utils/data_migration.py`: stop reading, copying, or merging legacy memory JSON.
- `.env`: remove only `BOT_NAME` and the complete `BOT_PERSONA` assignment; optionally add `MEMORY_MODELS`.
- `.env.example`: remove persona variables and document optional `MEMORY_MODELS`.
- `README.md`: explain persona files, memory semantics, commands, storage, privacy, and model cost.
- Existing identity, health, branding, product-scope, migration, image-flow, prompt, and concurrency tests: adapt to the new interfaces.

### Removed module

- `src/chat/memory.py`: delete after all callers use `src.memory`.

---

### Task 1: Standalone Persona Loading

**Files:**
- Create: `src/persona.py`
- Add: `config/persona.md`
- Add: `config/persona.example.md`
- Modify: `src/config.py`
- Modify: `src/chat/prompt.py`
- Modify: `src/main.py`
- Modify: `src/commands/help.py`
- Modify: `.env`
- Modify: `.env.example`
- Create: `tests/test_persona_file.py`
- Modify: `tests/test_identity_configuration.py`
- Modify: `tests/test_qqbot_branding.py`
- Modify: `tests/test_health.py`

**Interfaces:**
- Produces: `Persona(name: str, content: str)`.
- Produces: `load_persona(path: Path) -> Persona`.
- Produces: `get_persona() -> Persona`.
- Produces: `Config.persona_path: Path`.
- Consumes: the user-provided `config/persona.md` and template `config/persona.example.md`.

- [ ] **Step 1: Write failing persona loader tests**

```python
class PersonaFileTests(unittest.TestCase):
    def test_loads_name_and_full_markdown(self):
        path = self.root / "persona.md"
        path.write_text("# 角色\n\n- 名字：ATRI\n\n我是高性能机器人。", encoding="utf-8")
        persona = load_persona(path)
        self.assertEqual("ATRI", persona.name)
        self.assertIn("我是高性能机器人。", persona.content)

    def test_rejects_missing_empty_and_nameless_files(self):
        with self.assertRaisesRegex(PersonaConfigurationError, "不存在"):
            load_persona(self.root / "missing.md")
        empty = self.root / "empty.md"
        empty.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(PersonaConfigurationError, "为空"):
            load_persona(empty)
        nameless = self.root / "nameless.md"
        nameless.write_text("# 角色\n\n只有描述", encoding="utf-8")
        with self.assertRaisesRegex(PersonaConfigurationError, "名字"):
            load_persona(nameless)
```

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run: `python -m unittest tests.test_persona_file -v`

Expected: FAIL because `src.persona` does not exist.

- [ ] **Step 3: Implement the persona loader**

```python
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from src.config import config

NAME_PATTERN = re.compile(r"^\s*-\s*名字[：:]\s*(.+?)\s*$", re.MULTILINE)

class PersonaConfigurationError(RuntimeError):
    pass

@dataclass(frozen=True)
class Persona:
    name: str
    content: str

def load_persona(path: Path) -> Persona:
    if not path.is_file():
        raise PersonaConfigurationError(f"角色文件不存在：{path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise PersonaConfigurationError(f"角色文件为空：{path}")
    match = NAME_PATTERN.search(content)
    if not match or not match.group(1).strip():
        raise PersonaConfigurationError("角色文件缺少“- 名字：...”")
    return Persona(name=match.group(1).strip(), content=content)

@lru_cache(maxsize=1)
def get_persona() -> Persona:
    return load_persona(config.persona_path)
```

In `Config`, remove `bot_name`, `bot_persona`, `DEFAULT_BOT_NAME`, and `DEFAULT_BOT_PERSONA`, then add:

```python
persona_path: Path = field(default_factory=lambda: BASE_DIR / "config" / "persona.md")
```

Update the system prompt, startup, `/health`, start log, and help text to call `get_persona()` instead of `config.bot_name` or `config.bot_persona`.

- [ ] **Step 4: Move persona configuration out of dotenv**

Add the already reviewed user persona as `config/persona.md` and the approved template as `config/persona.example.md`.

Use a targeted patch on `.env` that removes only the complete `BOT_NAME=...` assignment and the entire quoted multiline `BOT_PERSONA=...` assignment. Preserve every unrelated line and secret exactly. Remove the two keys from `.env.example`.

- [ ] **Step 5: Run persona, identity, branding, prompt, and health tests**

Run:

```powershell
python -m unittest tests.test_persona_file tests.test_identity_configuration tests.test_qqbot_branding tests.test_health -v
```

Expected: PASS. Tests assert that the full file content appears in the first system message and neither environment variable remains supported.

- [ ] **Step 6: Commit the standalone persona**

```powershell
git add src/persona.py src/config.py src/chat/prompt.py src/main.py src/commands/help.py config/persona.md config/persona.example.md .env.example tests/test_persona_file.py tests/test_identity_configuration.py tests/test_qqbot_branding.py tests/test_health.py
git commit -m "feat: load identity from persona file"
```

Do not stage `.env`.

---

### Task 2: Optional Memory Model Chain

**Files:**
- Modify: `src/model_config.py`
- Modify: `src/config.py`
- Modify: `src/services/llm_client.py`
- Modify: `.env`
- Modify: `.env.example`
- Create: `tests/test_memory_model_configuration.py`
- Modify: `tests/test_model_chain_configuration.py`

**Interfaces:**
- Produces: `parse_model_chain(value: str | None, setting_name: str) -> tuple[ConfiguredModel, ...]`.
- Preserves: `parse_chat_models(value)`.
- Produces: `Config.memory_models: tuple[ConfiguredModel, ...]`.
- Produces: `get_memory_llm_client() -> FallbackLLMClient`.
- Consumes: `CHAT_MODELS`; optional `MEMORY_MODELS`.

- [ ] **Step 1: Write failing model-chain tests**

```python
class MemoryModelConfigurationTests(unittest.TestCase):
    def test_blank_memory_models_reuses_chat_chain(self):
        with patch.dict(os.environ, {
            "CHAT_MODELS": "gemini:chat-main,deepseek:chat-fallback",
            "MEMORY_MODELS": "",
            "GEMINI_API_KEY": "g",
            "DEEPSEEK_API_KEY": "d",
        }, clear=False):
            current = Config()
        self.assertEqual(current.chat_models, current.memory_models)

    def test_explicit_memory_chain_is_independent(self):
        with patch.dict(os.environ, {
            "CHAT_MODELS": "gemini:chat-main",
            "MEMORY_MODELS": "deepseek:memory-cheap",
            "GEMINI_API_KEY": "g",
            "DEEPSEEK_API_KEY": "d",
        }, clear=False):
            current = Config()
        self.assertEqual("memory-cheap", current.memory_models[0].model)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m unittest tests.test_memory_model_configuration -v`

Expected: FAIL because `Config.memory_models` does not exist.

- [ ] **Step 3: Generalize parsing and add `MEMORY_MODELS`**

```python
def parse_model_chain(
    value: str | None,
    setting_name: str,
) -> tuple[ConfiguredModel, ...]:
    raw = str(value or "")
    if not raw.strip():
        raise ModelConfigurationError(f"{setting_name} 不能为空")
    # Preserve the existing item parsing and deduplication, but use
    # setting_name in every error.

def parse_chat_models(value: str | None) -> tuple[ConfiguredModel, ...]:
    return parse_model_chain(value, "CHAT_MODELS")
```

In `Config.__post_init__`, use the chat chain when `_memory_models_raw` is blank; otherwise parse and validate `MEMORY_MODELS` with its own setting label. Validate API keys referenced by either chain.

- [ ] **Step 4: Add the memory fallback client**

```python
def _build_chain(cfg=None, models=None) -> list[LLMModelSpec]:
    cfg = cfg or config
    selected = tuple(models or cfg.chat_models)
    return [
        LLMModelSpec(
            provider=item.provider,
            model=item.model,
            supports_tools=_model_supports_tools(item.provider, item.model),
        )
        for item in selected
    ]

_memory_llm_client: FallbackLLMClient | None = None

def get_memory_llm_client() -> FallbackLLMClient:
    global _memory_llm_client
    if _memory_llm_client is None:
        _memory_llm_client = FallbackLLMClient(
            _build_chain(models=config.memory_models)
        )
    return _memory_llm_client
```

- [ ] **Step 5: Update dotenv examples and run model tests**

Add this optional example without making it required:

```dotenv
# MEMORY_MODELS=gemini:gemini-3.1-flash-lite
```

The ignored `.env` may omit it or use a user-selected chain. Run:

```powershell
python -m unittest tests.test_memory_model_configuration tests.test_model_chain_configuration tests.test_model_config -v
```

Expected: PASS.

- [ ] **Step 6: Commit memory model configuration**

```powershell
git add src/model_config.py src/config.py src/services/llm_client.py .env.example tests/test_memory_model_configuration.py tests/test_model_chain_configuration.py
git commit -m "feat: configure memory extraction models"
```

Do not stage `.env`.

---

### Task 3: SQLite Claim Ledger and Durable Jobs

**Files:**
- Create: `src/memory/__init__.py`
- Create: `src/memory/models.py`
- Create: `src/memory/store.py`
- Modify: `src/config.py`
- Create: `tests/test_memory_store.py`

**Interfaces:**
- Produces: `MemoryContext`, `MemoryEvent`, `CandidateClaim`, `MemoryClaim`, `RetrievedMemory`, `MemoryJob`.
- Produces: `MemoryStore.initialize()`.
- Produces: `MemoryStore.create_job(event) -> tuple[int, bool]`.
- Produces: `MemoryStore.mark_job_ready(job_id)`, `claim_next_job(scope_key)`, `complete_job(job_id)`, `fail_job(job_id, error_type, retry_at)`.
- Produces: claim/evidence/relation CRUD and `search_claims(...)`.
- Consumes: `Config.memory_database_path`.

- [ ] **Step 1: Write failing model and schema tests**

```python
class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.root.name) / "memory.sqlite3")
        self.store.initialize()

    def test_initializes_schema_and_fts(self):
        names = self.store.table_names()
        self.assertTrue({
            "memory_claims", "memory_evidence", "memory_relations",
            "memory_jobs", "schema_version", "memory_fts",
        }.issubset(names))

    def test_duplicate_message_job_is_idempotent(self):
        event = private_event(message_id="42", sequence=1, text="我喜欢跑步")
        first_id, first_created = self.store.create_job(event)
        second_id, second_created = self.store.create_job(event)
        self.assertTrue(first_created)
        self.assertFalse(second_created)
        self.assertEqual(first_id, second_id)
```

- [ ] **Step 2: Run store tests and verify they fail**

Run: `python -m unittest tests.test_memory_store -v`

Expected: FAIL because `src.memory.store` does not exist.

- [ ] **Step 3: Define immutable domain types**

```python
@dataclass(frozen=True)
class MemoryContext:
    user_id: str
    session_key: str
    is_group: bool
    group_id: str | None = None

    @property
    def primary_scope(self) -> tuple[str, str]:
        if self.is_group:
            return "group", str(self.group_id or "")
        return "private", self.user_id

@dataclass(frozen=True)
class MemoryEvent:
    context: MemoryContext
    message_id: str
    sequence: int
    text: str
    image_count: int = 0
    mentioned_qq_ids: tuple[str, ...] = ()
    reply_to_message_id: str | None = None
    reply_to_user_id: str | None = None

@dataclass(frozen=True)
class CandidateClaim:
    subject_ref: str
    predicate: str
    value: str
    memory_type: str
    modality: str
    confidence: str
    operation: str = "add"
    valid_from: str | None = None
    valid_to: str | None = None
```

Define the remaining result types in the same file with explicit string fields and immutable tuples.

- [ ] **Step 4: Implement schema version 1**

Create `memory_claims`, `memory_evidence`, `memory_relations`, `memory_jobs`, `schema_version`, and FTS5 tables in one transaction. Required constraints:

```sql
CREATE TABLE memory_jobs (
    id INTEGER PRIMARY KEY,
    dedupe_key TEXT NOT NULL UNIQUE,
    scope_key TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('staged','ready','running','retry','done','failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    retry_at TEXT,
    error_type TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE memory_relations (
    source_claim_id INTEGER NOT NULL REFERENCES memory_claims(id) ON DELETE CASCADE,
    target_claim_id INTEGER NOT NULL REFERENCES memory_claims(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL CHECK (relation_type IN ('supports','contradicts','supersedes','retracts')),
    created_at TEXT NOT NULL,
    UNIQUE(source_claim_id, target_claim_id, relation_type)
);
```

The claim table must contain every field listed in the design. Store only a minimal `source_excerpt`; never store image data in `payload_json`.

- [ ] **Step 5: Implement thread-safe store operations**

Each public operation opens its own connection, enables foreign keys, uses parameterized SQL, and closes the transaction promptly:

```python
@contextmanager
def _connection(self):
    connection = sqlite3.connect(self.path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
```

Set WAL during `initialize()`. Synchronize FTS rows in the same transaction as claim inserts, updates, and physical deletes.

- [ ] **Step 6: Test CRUD, FTS, relations, retries, and physical deletion**

Add tests that insert two conflicting claims, attach evidence, create a `contradicts` relation, query FTS, mark a job for retry, reopen the database, and physically delete a private claim while leaving a body-free audit record.

Run: `python -m unittest tests.test_memory_store -v`

Expected: PASS.

- [ ] **Step 7: Commit the ledger**

```powershell
git add src/memory/__init__.py src/memory/models.py src/memory/store.py src/config.py tests/test_memory_store.py
git commit -m "feat: add sqlite memory ledger"
```

---

### Task 4: Extraction, Entity Resolution, and Conflict Policy

**Files:**
- Create: `src/memory/extractor.py`
- Create: `src/memory/policy.py`
- Modify: `src/services/onebot_client.py`
- Create: `tests/test_memory_extractor.py`
- Create: `tests/test_memory_policy.py`
- Modify: `tests/test_image_input_service.py`

**Interfaces:**
- Consumes: `MemoryEvent`, ephemeral `image_data_urls`, `get_memory_llm_client()`.
- Produces: `MemoryExtractor.extract(event, image_data_urls=()) -> tuple[CandidateClaim, ...]`.
- Produces: `MemoryPolicy(store: MemoryStore)`.
- Produces: `MemoryPolicy.apply(event, candidates) -> tuple[PolicyDecision, ...]`.
- Produces: `OneBotClient.get_message_author(message_id: str) -> str | None`.
- Consumes: `MemoryStore` queries for active claims matching scope, speaker, subject, and predicate.

- [ ] **Step 1: Write failing extraction tests**

```python
def test_extracts_multiple_attributed_claims(self):
    self.llm.chat.return_value = ChatResponse(content=json.dumps({
        "claims": [
            {
                "subject_ref": "speaker",
                "predicate": "name",
                "value": "夏目安安",
                "memory_type": "identity",
                "modality": "asserted",
                "confidence": "high",
                "operation": "add",
            },
            {
                "subject_ref": "speaker",
                "predicate": "likes",
                "value": "跑步",
                "memory_type": "preference",
                "modality": "asserted",
                "confidence": "high",
                "operation": "add",
            },
        ]
    }, ensure_ascii=False))
    claims = self.extractor.extract(private_event(text="我是夏目安安，我喜欢跑步"))
    self.assertEqual(("name", "likes"), tuple(item.predicate for item in claims))

def test_invalid_json_is_repaired_once(self):
    self.llm.chat.side_effect = [
        ChatResponse(content="not json"),
        ChatResponse(content='{"claims": []}'),
    ]
    self.assertEqual((), self.extractor.extract(private_event(text="晚上好")))
    self.assertEqual(2, self.llm.chat.call_count)
```

- [ ] **Step 2: Write failing policy matrix tests**

Cover these exact cases:

```python
("我是谁？", "question", "no claim")
("这是我的狗", "speaker", "ownership")
("听说A喜欢跑步", "A", "hearsay medium truth")
("我以前喜欢跑步，现在喜欢游泳", "speaker", "close old validity and add new")
("sk-abcdefghijklmnopqrstuvwxyz", "secret", "reject")
```

Also test that `subject_ref="unknown"` is rejected, `subject_ref="qq:123"` resolves exactly, a reply target resolves through `reply_to_user_id`, and a user statement about Bot becomes an opinion without changing the persona.

- [ ] **Step 3: Implement the isolated extraction prompt and parser**

The extractor system prompt must require only this JSON object shape:

```json
{
  "claims": [
    {
      "subject_ref": "speaker|bot|qq:<number>|reply_target|unknown",
      "predicate": "short_snake_case",
      "value": "concise value",
      "memory_type": "identity|preferred_name|preference|opinion|event|plan|relationship|fact",
      "modality": "asserted|uncertain|hearsay|negated",
      "confidence": "low|medium|high",
      "operation": "add|confirm|supersede|retract",
      "valid_from": null,
      "valid_to": null
    }
  ]
}
```

Use temperature `0.0`, no tools, no persona, and no web search. Strip a single Markdown code fence if present, validate every enum and required field, and perform only one repair call with the validation error.

- [ ] **Step 4: Implement deterministic policy**

Required decision order:

```python
def apply(self, event, candidates):
    for candidate in candidates:
        if candidate.confidence == "low":
            continue
        subject = self.resolve_subject(event, candidate.subject_ref)
        if subject is None:
            continue
        if self.contains_hard_secret(candidate.value):
            continue
        scope_type, scope_id = event.context.primary_scope
        if event.context.is_group and self.is_sensitive_personal(candidate):
            continue
        yield self.decide_against_existing(
            event, candidate, subject, scope_type, scope_id
        )
```

`decide_against_existing` must implement confirm, same-speaker supersede/retract, different-speaker dispute, temporal validity closure, and independent evidence support. It must never let one speaker overwrite another.

- [ ] **Step 5: Resolve reply authors without blocking chat**

Add `OneBotClient.get_message_author()` using `/get_msg`. Call it only from the background extraction path. Return `None` on missing/invalid author data and log only the message ID and error type.

- [ ] **Step 6: Verify extractor and policy**

Run:

```powershell
python -m unittest tests.test_memory_extractor tests.test_memory_policy tests.test_image_input_service -v
```

Expected: PASS, including image candidates that require accompanying text for ownership or identity.

- [ ] **Step 7: Commit extraction and policy**

```powershell
git add src/memory/extractor.py src/memory/policy.py src/services/onebot_client.py tests/test_memory_extractor.py tests/test_memory_policy.py tests/test_image_input_service.py
git commit -m "feat: extract and reconcile scoped memories"
```

---

### Task 5: Permission-First Retrieval and Prompt Integration

**Files:**
- Create: `src/memory/retriever.py`
- Modify: `src/chat/prompt.py`
- Modify: `src/chat/chat_service.py`
- Create: `tests/test_memory_retrieval.py`
- Modify: `tests/test_multimodal_chat.py`
- Modify: `tests/test_chat_tool_finalization.py`
- Modify: `tests/test_user_facing_scope.py`

**Interfaces:**
- Produces: `MemoryRetriever.retrieve(context, query, limit=12) -> tuple[RetrievedMemory, ...]`.
- Produces: `format_memory_context(results) -> str`.
- Changes: `generate_reply(context: MemoryContext, text: str, tool_context: str = "", image_data_urls: list[str] | None = None) -> str`.
- Changes: `build_system_prompt(tool_context: str = "") -> str`.
- Changes: `build_untrusted_context(memories, tool_context="") -> str`.

- [ ] **Step 1: Write the privacy regression tests**

Build fixture claims for QQ A, QQ B, group 1, group 2, and global. Assert:

```python
def test_other_users_private_identity_never_enters_group_prompt(self):
    results = retriever.retrieve(group_context(user_id="B", group_id="1"), "A是谁")
    rendered = format_memory_context(results)
    self.assertNotIn("A-private-name", rendered)
    self.assertIn("A-group-1-name", rendered)

def test_global_first_person_is_attributed(self):
    results = retriever.retrieve(private_context(user_id="B"), "夏目安安是谁")
    rendered = format_memory_context(results)
    self.assertIn("发言者=A", rendered)
    self.assertIn("主体=A", rendered)

def test_group_data_does_not_cross_groups(self):
    results = retriever.retrieve(group_context(user_id="A", group_id="2"), "我喜欢什么")
    self.assertNotIn("group-1-running", format_memory_context(results))
```

Add the long-lived preference test by setting an old timestamp and asserting it remains retrievable.

- [ ] **Step 2: Run retrieval tests and verify they fail**

Run: `python -m unittest tests.test_memory_retrieval -v`

Expected: FAIL because `MemoryRetriever` does not exist.

- [ ] **Step 3: Implement hard scope filters**

Construct allowed SQL predicates before FTS:

```python
if context.is_group:
    evidence_scopes = [
        ("group", context.group_id),
        ("global", "global"),
    ]
    personalization_scope = ("private", context.user_id)
else:
    evidence_scopes = [
        ("private", context.user_id),
        ("global", "global"),
    ]
```

For private fallback, query only group claims where both `speaker_qq` and `subject_id` equal the current user and the type is identity, preferred name, or preference. For group personalization, query only the current user's private preferred name and response-style preferences; mark every result `usage="personalization"` so it cannot appear as factual evidence.

- [ ] **Step 4: Implement query hints and ranking**

Map first-person identity patterns such as “我是谁、我叫什么、怎么称呼我” to the current QQ and identity predicates. Resolve explicit aliases through current-group claims before global claims. Combine:

```python
score = (
    exact_subject * 40
    + exact_predicate * 25
    + direct_scope * 20
    + confidence_rank * 10
    + recent_confirmation_bonus
    + fts_rank
)
```

Preferred names receive no age penalty. Preferences receive a bounded recency bonus but no score below the documented floor. Exclude retracted, superseded, archived, expired-current-state, and hard-deleted records.

- [ ] **Step 5: Replace flat prompt injection**

The prompt context must have separate blocks:

```text
[允许使用的记忆证据]
- 作用域=group:1；发言者=A；主体=A；类型=opinion；内容=……
[/允许使用的记忆证据]
[仅用于称呼和表达的个性化信息]
- 主体=当前发言者；首选称呼=安安
禁止把本区内容作为公开身份、经历或关系事实。
[/仅用于称呼和表达的个性化信息]
```

Remove imports and calls to `get_global_memory`, `get_personal_memory`, and `get_memory`. Fix prompt priority to capability/safety, privacy/permissions, persona, and then untrusted evidence.

- [ ] **Step 6: Change chat context plumbing**

Update `generate_reply()` to receive `MemoryContext`, retrieve memory for the current text before building messages, and continue using `context.session_key` for history. Preserve search tool loops and multimodal content unchanged.

- [ ] **Step 7: Run prompt, retrieval, tool, and multimodal tests**

Run:

```powershell
python -m unittest tests.test_memory_retrieval tests.test_multimodal_chat tests.test_chat_tool_finalization tests.test_user_facing_scope -v
```

Expected: PASS. No prompt contains the old labels “个人基础信息” or “当前会话记忆”.

- [ ] **Step 8: Commit retrieval integration**

```powershell
git add src/memory/retriever.py src/chat/prompt.py src/chat/chat_service.py tests/test_memory_retrieval.py tests/test_multimodal_chat.py tests/test_chat_tool_finalization.py tests/test_user_facing_scope.py
git commit -m "feat: retrieve memories with hard scope filters"
```

---

### Task 6: Durable Background Learning and Message Integration

**Files:**
- Create: `src/memory/service.py`
- Modify: `src/main.py`
- Modify: `src/messaging.py`
- Modify: `src/services/onebot_client.py`
- Create: `tests/test_memory_service.py`
- Create: `tests/test_memory_end_to_end.py`
- Modify: `tests/test_messaging.py`
- Modify: `tests/test_main_image_flow.py`

**Interfaces:**
- Produces: `MemoryService.start()`, `stop()`, `stage_event(event) -> int`, `release_job(job_id, image_data_urls=())`, `wait_for_scope(scope_key, timeout)`.
- Produces: `get_memory_service() -> MemoryService`.
- Produces: `MemoryService.store: MemoryStore`.
- Changes: accepted event dictionaries gain internal `_qqbot_sequence: int`.
- Consumes: `MemoryStore`, `MemoryExtractor`, `MemoryPolicy`.

- [ ] **Step 1: Write failing background service tests**

```python
def test_stage_is_durable_but_images_are_ephemeral(self):
    job_id = service.stage_event(event_with_one_image)
    service.release_job(job_id, ["data:image/png;base64,abc"])
    row = store.get_job(job_id)
    self.assertNotIn("base64", row.payload_json)
    self.assertEqual(1, row.image_count)

def test_chat_release_does_not_wait_for_extraction(self):
    extractor.extract.side_effect = threading.Event().wait
    started = time.monotonic()
    service.release_job(service.stage_event(event), ())
    self.assertLess(time.monotonic() - started, 0.1)

def test_restart_retries_ready_text_job(self):
    job_id = store.create_ready_job(event)[0]
    replacement = build_service(store_path=store.path)
    replacement.start()
    self.assertTrue(wait_until(lambda: store.get_job(job_id).state == "done"))
```

- [ ] **Step 2: Run service tests and verify they fail**

Run: `python -m unittest tests.test_memory_service -v`

Expected: FAIL because `MemoryService` does not exist.

- [ ] **Step 3: Implement staged and ready job lifecycle**

`stage_event()` writes only text and metadata with state `staged`. `release_job()` stores image data in an in-memory dictionary keyed by job ID, changes state to `ready`, and returns immediately. A restarted process can retry text and metadata but never reconstruct or persist raw image bytes.

Use per-scope FIFO queues:

```python
scope_key = (
    f"group:{event.context.group_id}"
    if event.context.is_group
    else f"private:{event.context.user_id}"
)
```

Only one worker processes a given scope at a time; different scopes use the configured bounded executor. This preserves group and private memory commit order without serializing unrelated chats.

- [ ] **Step 4: Implement retry and cleanup**

- Retry transient LLM/network failures with delays of 2, 10, and 30 seconds.
- Repair invalid JSON inside the extractor once; a second invalid result is permanent for that attempt.
- Mark a job `failed` after four total attempts; store only the error class name.
- Always remove ephemeral images in `finally`.
- On startup, convert abandoned `running` jobs to `retry`, claim ready/retry jobs in sequence order, and start workers.
- Cleanup completed job payloads and archived source excerpts according to the 90-day design.

- [ ] **Step 5: Assign receive sequence and build `MemoryContext`**

In `MessageQueue.enqueue`, assign a lock-protected monotonic sequence before appending. Preserve the existing session FIFO and dedupe behavior.

In `process_message`, build:

```python
memory_context = MemoryContext(
    user_id=uid,
    session_key=session_key,
    is_group=is_group,
    group_id=str(data.get("group_id")) if is_group else None,
)
```

After parsing an accepted ordinary chat message, stage the event before model generation. In a `finally` path after the reply attempt, release it for learning with the loaded image data. Do not automatically enqueue slash commands; memory commands use synchronous service methods in Task 7.

- [ ] **Step 6: Initialize and recover at startup**

`startup()` must validate persona, initialize SQLite, and start `MemoryService` exactly once under `_startup_lock`. `/health` may report memory worker status and failed-job count, but never content.

- [ ] **Step 7: Test ordering, concurrency, duplicate callbacks, failures, and images**

Required integration assertions:

- Same-user reply order remains FIFO.
- Different private users can reply concurrently.
- Different users in one group cannot commit memory out of receive sequence.
- Duplicate OneBot callbacks create one durable job and one set of claims.
- A blocked extractor does not delay the reply.
- Extractor failure leaves chat successful and a retryable/failed job.
- Ephemeral image data is gone after success and failure.
- Logs contain job IDs and error class names but not message text, claim values, image data, API keys, or QQ-private excerpts.

Run:

```powershell
python -m unittest tests.test_memory_service tests.test_memory_end_to_end tests.test_messaging tests.test_main_image_flow -v
```

Expected: PASS.

- [ ] **Step 8: Commit the background pipeline**

```powershell
git add src/memory/service.py src/main.py src/messaging.py src/services/onebot_client.py tests/test_memory_service.py tests/test_memory_end_to_end.py tests/test_messaging.py tests/test_main_image_flow.py
git commit -m "feat: learn memories in durable background jobs"
```

---

### Task 7: Scoped Memory Commands and Persona Rendering

**Files:**
- Modify: `src/commands/__init__.py`
- Modify: `src/commands/help.py`
- Modify: `src/commands/reset.py`
- Modify: `src/chat/chat_service.py`
- Modify: `src/main.py`
- Create: `tests/test_memory_commands.py`
- Modify: `tests/test_product_scope.py`
- Modify: `tests/test_user_facing_scope.py`

**Interfaces:**
- Changes: `CommandContext` includes `memory_context`, `message_id`, `is_admin`.
- Produces: `CommandOutcome(code: str, facts: tuple[str, ...], fallback_reply: str, already_rendered: bool = False)`.
- Produces: `MemoryService.remember(...)`, `global_remember(...)`, `list_memories(...)`, `forget(...)`.
- Produces: `render_command_outcome(context, outcome) -> str`.

- [ ] **Step 1: Write failing command scope and permission tests**

```python
def test_remember_uses_private_scope_in_private_chat(self):
    outcome = handle("/remember 我喜欢跑步", private_context("A"))
    self.assertEqual(("private", "A"), store.last_claim_scope())
    self.assertEqual("saved", outcome.code)

def test_remember_uses_current_group_scope_in_group(self):
    outcome = handle("/remember 我在这里叫安安", group_context("A", "100"))
    self.assertEqual(("group", "100"), store.last_claim_scope())

def test_globalremember_attributes_first_person_to_admin(self):
    handle("/globalremember 我是夏目安安", private_admin_context("A"))
    claim = store.last_claim()
    self.assertEqual("A", claim.speaker_qq)
    self.assertEqual("A", claim.subject_id)
    self.assertEqual(("global", "global"), claim.scope)
```

Add tests for non-admin rejection, `/memories` scope filtering, own-claim retraction, other-speaker protection, dispute/suppression of a claim about self, administrator physical deletion, and hard-secret rejection.

- [ ] **Step 2: Run command tests and verify they fail**

Run: `python -m unittest tests.test_memory_commands -v`

Expected: FAIL because the new commands and outcomes do not exist.

- [ ] **Step 3: Implement synchronous command service methods**

`/remember` and `/globalremember` call the same extractor and policy as automatic learning but wait for completion. The command source sets high attribution confidence and keeps normal truth-confidence rules.

`/memories` returns only already-filtered summaries and short IDs. `/forget` resolves a short ID first, then a natural-language description if exactly one permitted match exists. Ambiguous descriptions return a factual outcome requesting clarification; they do not delete multiple records.

- [ ] **Step 4: Enforce deletion and moderation rules**

```python
def can_forget(actor, claim, is_admin):
    if is_admin:
        return True
    if claim.scope_type == "private":
        return claim.scope_id == actor
    return claim.speaker_qq == actor
```

If a non-author is the subject of a group claim, allow `dispute_and_suppress` but do not mark the original speaker as retracted. Hard-delete private content and FTS rows on an authorized privacy deletion; retain only body-free audit metadata.

- [ ] **Step 5: Render normal command outcomes in persona**

Command mutation happens before rendering. Send the model a trusted status such as:

```text
操作状态：成功
操作类型：保存当前群记忆
必须准确表达以上状态；不得声称保存到私聊或全局；保持当前角色设定。
```

Search results remain already rendered. For help, reset, unknown commands, and memory outcomes, use `render_command_outcome()`. If the LLM renderer fails, send `fallback_reply` with the exact status and cause.

- [ ] **Step 6: Update command registry and help**

The supported set is:

```python
{
    "help", "h", "search", "s", "reset",
    "remember", "memo",
    "globalremember", "gremember",
    "memories", "forget",
}
```

The Bot never suggests memory commands during ordinary conversation. It explains them only after `/help`, direct feature questions, or explicit use.

- [ ] **Step 7: Run command and user-facing tests**

Run:

```powershell
python -m unittest tests.test_memory_commands tests.test_product_scope tests.test_user_facing_scope -v
```

Expected: PASS.

- [ ] **Step 8: Commit command support**

```powershell
git add src/commands/__init__.py src/commands/help.py src/commands/reset.py src/chat/chat_service.py src/main.py tests/test_memory_commands.py tests/test_product_scope.py tests/test_user_facing_scope.py
git commit -m "feat: manage scoped memories with commands"
```

---

### Task 8: Remove Legacy Memory Paths, Document, and Run Full Acceptance

**Files:**
- Delete: `src/chat/memory.py`
- Modify: `src/utils/data_migration.py`
- Modify: `src/main.py`
- Modify: `src/config.py`
- Modify: `.env`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/test_data_migration.py`
- Modify: `tests/test_readme_guide.py`
- Modify: `tests/test_product_scope.py`
- Modify: all existing tests that import `src.chat.memory`, `config.bot_name`, `config.bot_persona`, or `config.memory_limit`

**Interfaces:**
- Removes: all flat JSON memory functions and `migrate_legacy_memory_files()`.
- Removes: `Config.memory_limit` and `MEMORY_LIMIT`.
- Preserves: history migration and `reset_history(session_key)`.
- Documents: `config/persona.md`, optional `MEMORY_MODELS`, SQLite storage, command scopes, privacy, automatic learning, and costs.

- [ ] **Step 1: Write failing legacy-removal and README tests**

```python
def test_runtime_has_no_legacy_memory_module_imports(self):
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src").rglob("*.py")
    )
    self.assertNotIn("src.chat.memory", source)
    self.assertNotIn("migrate_legacy_memory_files", source)

def test_readme_explains_group_and_private_remember_scopes(self):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    self.assertIn("私聊使用 `/remember`", readme)
    self.assertIn("群聊使用 `/remember`", readme)
    self.assertIn("config/persona.md", readme)
    self.assertIn("MEMORY_MODELS", readme)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
python -m unittest tests.test_data_migration tests.test_readme_guide tests.test_product_scope -v
```

Expected: FAIL because legacy memory imports, migration behavior, and old documentation remain.

- [ ] **Step 3: Remove old runtime and migration behavior**

- Delete `src/chat/memory.py`.
- Remove the startup import and call to `migrate_legacy_memory_files()`.
- Remove memory copying/merging from `migrate_legacy_data`; preserve history and other approved startup migration behavior.
- Remove `memory_limit` parameters from migration signatures and all callers.
- Remove `Config.memory_limit` and `MEMORY_LIMIT`.
- Make `/reset` clear current history only.
- Add a test that places a sentinel JSON file under `qqbot_data/memories/`, starts the app, and verifies the file was neither opened nor imported into SQLite.

- [ ] **Step 4: Rewrite the relevant README and dotenv sections**

Document exactly:

- Copy and edit `config/persona.example.md` as `config/persona.md`.
- `.env` no longer contains the Bot name or persona.
- `CHAT_MODELS` is required; `MEMORY_MODELS` is optional and falls back to it.
- Each learned message can cause an additional background model call and cost.
- Private, current-group, and global scope rules.
- `/remember`, `/globalremember`, `/memories`, `/forget`, `/reset`.
- Automatic learning, eventual consistency, conflicts, sensitive filtering, image handling, and deletion.
- SQLite file location under `DATA_DIR`, backups, and logs that omit content.
- Old JSON memories are unsupported and ignored.

Do not document removed product capabilities as if they still exist.

- [ ] **Step 5: Run static searches**

Run:

```powershell
rg -n "BOT_NAME|BOT_PERSONA|MEMORY_LIMIT|src\\.chat\\.memory|migrate_legacy_memory_files" src tests README.md .env.example
rg -n "ATRI" src .env.example README.md
```

Expected:

- The first command returns only intentional negative assertions in tests, if any.
- The second command returns no hard-coded runtime identity. `config/persona.md` is intentionally excluded because ATRI is the selected replaceable default persona.

- [ ] **Step 6: Run the full test suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 7: Run database and import smoke tests**

Run:

```powershell
python -c "from src.persona import get_persona; from src.memory.service import get_memory_service; print(get_persona().name); print(get_memory_service().store.integrity_check())"
python -c "from src.main import startup; startup(); print('startup-ok')"
```

Expected: the configured persona name, `ok`, and `startup-ok`. No old JSON memory migration log appears.

- [ ] **Step 8: Inspect the final diff for secrets and unrelated changes**

Run:

```powershell
git diff --check
git status --short
git diff --stat
```

Verify that `.env`, `qqbot_data/`, image data, API keys, QQ numbers, and backup directories are not staged. Verify that `config/persona.md` is intentionally tracked as the replaceable default persona requested by the user.

- [ ] **Step 9: Commit cleanup and documentation**

```powershell
git add src/chat/memory.py src/utils/data_migration.py src/main.py src/config.py .env.example README.md tests
git commit -m "docs: finalize adaptive memory migration"
```

Do not stage `.env` or runtime data.

---

## Spec Coverage Matrix

| Design requirement | Implementation task |
| --- | --- |
| Standalone persona, startup validation, role-consistent replies | Tasks 1 and 7 |
| Optional memory extraction model chain | Task 2 |
| Structured claims, evidence, relations, confidence, FTS, lifecycle fields | Task 3 |
| Entity resolution, sensitive filtering, image rules, update and conflict policy | Task 4 |
| Private/group/global isolation, temporary fallback, ranking and prompt safety | Task 5 |
| Reply-first automatic learning, eventual consistency, retries, ordering and dedupe | Task 6 |
| `/remember`, `/globalremember`, `/memories`, `/forget`, deletion permissions | Task 7 |
| No legacy JSON reads, persona/env cleanup, README and full regression suite | Task 8 |
| Preferred-name no-decay and preference retrieval floor | Tasks 4 and 5 |
| 90-day evidence cleanup and immediate privacy deletion | Tasks 3, 6, and 7 |
| Accurate faults, redacted logs, restart recovery | Tasks 6 and 8 |
| All twenty-one design acceptance scenarios | Tasks 4 through 8 plus the final checklist |

---

## Final Acceptance Checklist

- [ ] Private identity isolation passes.
- [ ] Per-group isolation passes.
- [ ] Global first-person attribution passes.
- [ ] Same-speaker supersede and retract pass.
- [ ] Different-speaker conflict preservation passes.
- [ ] Preferred-name and preference retention-floor tests pass.
- [ ] Question, negation, hearsay, joke, and ambiguous-pronoun tests pass.
- [ ] Sensitive-data and image ownership tests pass.
- [ ] Reply FIFO, parallel users, group commit order, and duplicate callback tests pass.
- [ ] Background retry, restart recovery, and ephemeral image cleanup pass.
- [ ] `/remember`, `/globalremember`, `/memories`, and `/forget` permissions pass.
- [ ] Persona is present in all normal user-facing model calls.
- [ ] Fault and command fallback text reports exact status.
- [ ] Old JSON memories are not read or migrated.
- [ ] README and `.env.example` describe the final behavior.
- [ ] Full `unittest` discovery passes.
