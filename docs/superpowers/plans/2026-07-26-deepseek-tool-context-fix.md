# DeepSeek Tool Context Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve DeepSeek V4 reasoning context through tool calls, force the post-limit request to produce text, and stop inferring tool support from model-name substrings.

**Architecture:** Reuse the existing short-lived `ChatResponse.provider_context` bridge. `DeepSeekClient` captures `reasoning_content` only for tool-call responses and restores it only when sending a DeepSeek-owned temporary assistant message; the context never reaches storage or Gemini. The chat loop keeps tool declarations on the final synthesis request but sends `tool_choice="none"`, while the model-chain capability policy returns to its explicit deny-list.

**Tech Stack:** Python 3.11+, `dataclasses`, `unittest`, `unittest.mock`, `requests`.

## Global Constraints

- Do not read, modify, stage, or commit the real `.env`.
- Do not add dependencies or environment variables.
- Preserve native Gemini `generateContent`, thought signatures, function IDs, images, memory, history, search, fallback, and concurrency.
- DeepSeek provider context is temporary and must not be written to history, memory, or JSON files.
- Gemini provider context must not be sent to DeepSeek.
- Use test-driven development: each production change follows a focused failing test observed before implementation.
- Make only the files listed below part of the repair commits.

---

## File Map

- Create `tests/test_deepseek_tool_context.py`: focused DeepSeek reasoning-context protocol tests.
- Create `tests/test_chat_tool_finalization.py`: final synthesis tool-choice regression test.
- Modify `tests/test_model_chain_configuration.py`: capability-policy regression test.
- Modify `tests/test_gemini_native_client.py`: remove obsolete substring-policy assertions.
- Modify `src/services/deepseek_client.py`: capture and restore DeepSeek `reasoning_content`.
- Modify `src/chat/chat_service.py`: force `tool_choice="none"` on final synthesis.
- Modify `src/services/llm_client.py`: remove `reasoner`/`r1` substring inference.

---

### Task 1: Preserve DeepSeek Reasoning Context

**Files:**
- Create: `tests/test_deepseek_tool_context.py`
- Modify: `src/services/deepseek_client.py:31-78`

**Interfaces:**
- Consumes: `ChatResponse.provider_context: dict[str, Any] | None`.
- Produces: DeepSeek tool responses with provider context:

```python
{
    "provider": "deepseek",
    "reasoning_content": str,
}
```

- Produces: outgoing DeepSeek assistant messages with `_provider_context` removed and provider-owned `reasoning_content` restored.

- [ ] **Step 1: Write focused failing tests**

Create `tests/test_deepseek_tool_context.py`:

```python
import unittest
from types import SimpleNamespace
from unittest import mock

from src.services.deepseek_client import DeepSeekClient


class FakeResponse:
    def __init__(self, message):
        self._message = message

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": self._message}]}


def config():
    return SimpleNamespace(
        deepseek_api_key="d-key",
        deepseek_url="https://api.deepseek.com/chat/completions",
        proxies=None,
        request_timeout=18,
    )


TOOL_CALL = {
    "id": "call-1",
    "type": "function",
    "function": {
        "name": "search_web",
        "arguments": '{"query":"测试"}',
    },
}


class DeepSeekToolContextTests(unittest.TestCase):
    def test_tool_response_captures_reasoning_content(self):
        response = FakeResponse(
            {
                "content": "",
                "reasoning_content": "先搜索再整理",
                "tool_calls": [TOOL_CALL],
            }
        )

        with mock.patch(
            "src.services.deepseek_client.try_proxied_post",
            return_value=response,
        ):
            result = DeepSeekClient(config()).chat(
                [{"role": "user", "content": "查一下"}],
                model="deepseek-v4-flash",
                tools=[{"type": "function", "function": {"name": "search_web"}}],
            )

        self.assertEqual(
            {
                "provider": "deepseek",
                "reasoning_content": "先搜索再整理",
            },
            result.provider_context,
        )

    def test_deepseek_context_is_restored_without_private_field(self):
        response = FakeResponse({"content": "整理结果"})
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [TOOL_CALL],
                "_provider_context": {
                    "provider": "deepseek",
                    "reasoning_content": "先搜索再整理",
                },
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search_web",
                "content": "搜索结果",
            },
        ]

        with mock.patch(
            "src.services.deepseek_client.try_proxied_post",
            return_value=response,
        ) as post:
            DeepSeekClient(config()).chat(
                messages,
                model="deepseek-v4-flash",
            )

        sent_assistant = post.call_args.kwargs["json"]["messages"][0]
        self.assertEqual(
            "先搜索再整理",
            sent_assistant["reasoning_content"],
        )
        self.assertNotIn("_provider_context", sent_assistant)

    def test_gemini_context_is_stripped_without_reasoning_conversion(self):
        response = FakeResponse({"content": "整理结果"})
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [TOOL_CALL],
                "_provider_context": {
                    "provider": "gemini",
                    "content": {
                        "role": "model",
                        "parts": [],
                    },
                },
            }
        ]

        with mock.patch(
            "src.services.deepseek_client.try_proxied_post",
            return_value=response,
        ) as post:
            DeepSeekClient(config()).chat(
                messages,
                model="deepseek-v4-flash",
            )

        sent_assistant = post.call_args.kwargs["json"]["messages"][0]
        self.assertNotIn("_provider_context", sent_assistant)
        self.assertNotIn("reasoning_content", sent_assistant)

    def test_text_response_does_not_retain_reasoning_context(self):
        response = FakeResponse(
            {
                "content": "普通回答",
                "reasoning_content": "无需跨轮保存",
                "tool_calls": [],
            }
        )

        with mock.patch(
            "src.services.deepseek_client.try_proxied_post",
            return_value=response,
        ):
            result = DeepSeekClient(config()).chat(
                [{"role": "user", "content": "你好"}],
                model="deepseek-v4-flash",
            )

        self.assertEqual("普通回答", result.content)
        self.assertIsNone(result.provider_context)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_deepseek_tool_context -v
```

Expected:

- `test_tool_response_captures_reasoning_content` fails because `provider_context` is `None`.
- `test_deepseek_context_is_restored_without_private_field` fails because outgoing assistant messages lack `reasoning_content`.
- Gemini stripping and text-only context tests may already pass; do not weaken them.

- [ ] **Step 3: Implement provider-aware DeepSeek message cleaning**

In `src/services/deepseek_client.py`, add this helper above `DeepSeekClient`:

```python
def _clean_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        clean_message = {
            key: value
            for key, value in message.items()
            if key != "_provider_context"
        }
        provider_context = message.get("_provider_context")
        if (
            message.get("role") == "assistant"
            and isinstance(provider_context, dict)
            and provider_context.get("provider") == "deepseek"
        ):
            reasoning_content = provider_context.get(
                "reasoning_content"
            )
            if isinstance(reasoning_content, str):
                clean_message["reasoning_content"] = reasoning_content
        cleaned.append(clean_message)
    return cleaned
```

Replace the inline `clean_messages` comprehension in `DeepSeekClient.chat()` with:

```python
        clean_messages = _clean_messages(messages)
```

Do not restore arbitrary keys from provider context.

- [ ] **Step 4: Capture reasoning content only for tool responses**

Immediately before returning `ChatResponse`, add:

```python
        reasoning_content = message.get("reasoning_content")
        provider_context = None
        if raw_tool_calls and isinstance(reasoning_content, str):
            provider_context = {
                "provider": "deepseek",
                "reasoning_content": reasoning_content,
            }
```

Return:

```python
        return ChatResponse(
            content=(message.get("content") or "").strip(),
            tool_calls=raw_tool_calls,
            provider_context=provider_context,
        )
```

- [ ] **Step 5: Run focused and provider-boundary tests**

Run:

```powershell
python -m unittest tests.test_deepseek_tool_context tests.test_gemini_native_client -v
```

Expected: all tests pass, including existing empty-Key and Gemini-context stripping tests.

- [ ] **Step 6: Compile, inspect, and commit**

Run:

```powershell
python -m compileall -q src/services/deepseek_client.py tests/test_deepseek_tool_context.py
git diff --check
git add src/services/deepseek_client.py tests/test_deepseek_tool_context.py
git diff --cached --check
git diff --cached --name-only
git commit -m "fix: preserve DeepSeek tool reasoning context"
```

Expected staged names:

```text
src/services/deepseek_client.py
tests/test_deepseek_tool_context.py
```

---

### Task 2: Make Final Synthesis Deterministic

**Files:**
- Create: `tests/test_chat_tool_finalization.py`
- Modify: `tests/test_model_chain_configuration.py`
- Modify: `tests/test_gemini_native_client.py`
- Modify: `src/chat/chat_service.py:322-325`
- Modify: `src/services/llm_client.py:36-44`

**Interfaces:**
- Consumes: existing `llm.chat(..., tools=..., tool_choice=...)`.
- Produces: final synthesis request with `tools` retained and `tool_choice="none"`.
- Produces: `_model_supports_tools()` based only on the explicit capability map plus the existing default-true policy.

- [ ] **Step 1: Write the final-synthesis failing test**

Create `tests/test_chat_tool_finalization.py`:

```python
import unittest
from unittest import mock

from src.chat import chat_service
from src.services.llm_types import ChatResponse


def tool_call(call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "search_web",
            "arguments": '{"query":"测试"}',
        },
    }


class SequencedLLM:
    def __init__(self):
        self.calls = []
        self.responses = [
            ChatResponse(tool_calls=[tool_call("call-1")]),
            ChatResponse(tool_calls=[tool_call("call-2")]),
            ChatResponse(content="最终整理结果"),
        ]

    def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FinalSynthesisTests(unittest.TestCase):
    def test_final_synthesis_keeps_tools_but_forbids_more_calls(self):
        fake_llm = SequencedLLM()
        session_key = "final-synthesis-test"

        with (
            mock.patch.object(chat_service, "llm", fake_llm),
            mock.patch.object(
                chat_service,
                "run_tool",
                return_value="搜索结果",
            ),
            mock.patch.object(chat_service, "append_history"),
            mock.patch.object(chat_service, "_ensure_history_loaded"),
        ):
            chat_service.chat_history[session_key] = []
            try:
                reply = chat_service.generate_reply(
                    session_key,
                    "查一下",
                )
            finally:
                chat_service.chat_history.pop(session_key, None)

        self.assertEqual("最终整理结果", reply)
        self.assertEqual(3, len(fake_llm.calls))
        self.assertTrue(fake_llm.calls[-1]["tools"])
        self.assertEqual(
            "none",
            fake_llm.calls[-1]["tool_choice"],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Add the capability-policy failing test**

In `tests/test_model_chain_configuration.py`, import:

```python
from src.services.llm_client import _build_chain, _model_supports_tools
```

Add to `ConfiguredChainTests`:

```python
    def test_tool_support_is_not_inferred_from_model_name_substrings(self):
        self.assertTrue(
            _model_supports_tools(
                "deepseek",
                "custom-reasoner-with-tools",
            )
        )
        self.assertTrue(
            _model_supports_tools(
                "deepseek",
                "custom-r1-with-tools",
            )
        )
        self.assertFalse(
            _model_supports_tools(
                "gemini",
                "gemma-4-26b-a4b-it",
            )
        )
```

- [ ] **Step 3: Run both tests and verify RED**

Run:

```powershell
python -m unittest tests.test_chat_tool_finalization tests.test_model_chain_configuration -v
```

Expected:

- final synthesis test fails because `tool_choice` is absent;
- capability-policy test fails because both custom model names are classified as unsupported.

- [ ] **Step 4: Force text-only final synthesis**

In `src/chat/chat_service.py`, replace the final call with:

```python
            reply = normalize_chat_response(
                llm.chat(
                    messages,
                    temperature=0.75,
                    tools=tools,
                    tool_choice="none",
                )
            ).content
```

Do not add a fourth tool round.

- [ ] **Step 5: Remove the model-name substring inference**

In `src/services/llm_client.py`, delete:

```python
    if "reasoner" in key or "r1" in key:
        return False
```

Keep `_MODEL_CAPABILITIES` and the default `True` return unchanged.

In `tests/test_gemini_native_client.py`, delete the now-invalid assertions:

```python
        self.assertFalse(_model_supports_tools("deepseek", "deepseek-reasoner"))
        self.assertFalse(_model_supports_tools("deepseek", "deepseek-r1"))
```

Then remove the entire `test_model_supports_tools_filters_reasoner_and_r1_models` method because its remaining assertions are covered by `tests/test_model_chain_configuration.py`.

- [ ] **Step 6: Run focused tool-loop and model-chain tests**

Run:

```powershell
python -m unittest tests.test_chat_tool_finalization tests.test_model_chain_configuration tests.test_gemini_native_client tests.test_llm_image_fallback -v
```

Expected: all tests pass.

- [ ] **Step 7: Compile, inspect, and commit**

Run:

```powershell
python -m compileall -q src/chat/chat_service.py src/services/llm_client.py tests/test_chat_tool_finalization.py tests/test_model_chain_configuration.py
git diff --check
git add src/chat/chat_service.py src/services/llm_client.py tests/test_chat_tool_finalization.py tests/test_model_chain_configuration.py tests/test_gemini_native_client.py
git diff --cached --check
git diff --cached --name-only
git commit -m "fix: finalize tool loops without another call"
```

Expected staged names:

```text
src/chat/chat_service.py
src/services/llm_client.py
tests/test_chat_tool_finalization.py
tests/test_gemini_native_client.py
tests/test_model_chain_configuration.py
```

---

### Task 3: Full Verification and Handoff

**Files:**
- Verify all files changed by Tasks 1–2.
- Preserve the real `.env` and local data.

**Interfaces:**
- Final behavior: DeepSeek reasoning context survives tool turns, final synthesis cannot request another tool, and capability inference uses only explicit entries.

- [ ] **Step 1: Run the complete test suite**

Run:

```powershell
python -m unittest discover -s tests -t . -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Run compilation and patch validation**

Run:

```powershell
python -m compileall -q src tests run_bot.py
git diff --check
git status --short --branch
git ls-files .env
```

Expected:

- compilation succeeds;
- patch check is clean;
- `.env` is not tracked;
- only intended files or a clean worktree remain.

- [ ] **Step 3: Run focused protocol verification**

Run:

```powershell
python -m unittest tests.test_deepseek_tool_context tests.test_chat_tool_finalization tests.test_gemini_native_client tests.test_model_chain_configuration -v
python -c "from dotenv import dotenv_values; values=dotenv_values('.env.example'); print('keys', len(values), 'unique', len(set(values)))"
rg -n "reasoner.*in key|r1.*in key" src/services/llm_client.py
```

Expected:

- focused tests pass;
- dotenv totals are equal;
- the substring scan returns no matches.

- [ ] **Step 4: Review the complete repair diff**

Verify:

```text
- DeepSeek reasoning_content is captured only when tool_calls exist;
- only DeepSeek-owned provider context restores reasoning_content;
- _provider_context never enters a remote payload;
- Gemini provider context handling remains unchanged;
- final synthesis keeps tools and sends tool_choice="none";
- model names are not used as broad capability substrings;
- history and memory persistence do not contain provider context;
- no real .env, API Key, URL token, or local data is staged.
```

- [ ] **Step 5: Apply any review correction with a failing test**

For each actionable finding:

1. write or adjust one focused test;
2. run it and observe the expected failure;
3. implement the smallest correction;
4. rerun the focused and complete suites.

Do not add provider settings, thinking-mode configuration, or unrelated refactors.

- [ ] **Step 6: Finish the branch**

Use `superpowers:finishing-a-development-branch`. Do not merge, push, delete a branch, or modify the real `.env` without explicit user authorization.
