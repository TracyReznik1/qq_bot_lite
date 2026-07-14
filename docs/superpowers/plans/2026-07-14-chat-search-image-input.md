# Chat, Search, and Image Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove non-product media and direct-URL remnants while preserving chat, web search, history, and memory, and add user-supplied image understanding as a chat input mode.

**Architecture:** Keep the existing OneBot → Flask → router → chat/search → LLM flow. Add one focused image-input service that parses OneBot image segments and converts bounded image downloads to OpenAI-compatible data URLs; pass those through the existing provider clients, persist only `[图片]` placeholders, and let the current fallback chain continue when a model rejects multimodal input.

**Tech Stack:** Python 3.11+, Flask, requests, OneBot 11 CQ/structured messages, OpenAI-compatible chat-completion payloads, standard-library `unittest` and `unittest.mock`.

## Global Constraints

- Preserve `/remember`, `/globalremember`, `/help`, `/reset`, chat history, and all memory modules.
- Preserve automatic `search_web`, explicit `/search <关键词>`, and internal search-result page fetching.
- Remove `/search <URL>` direct reading, OpenAI media, video, image generation, ComfyUI, and OneBot outbound image sending.
- Accept at most 4 input images per message, each at most 5 MiB, with MIME type JPEG, PNG, WebP, or GIF.
- Never persist image bytes, base64 data, or temporary image URLs.
- Do not modify `.env`, existing history JSON, or memory data.
- Do not add third-party dependencies.
- Use test-first red—green cycles for every behavior change.

## File Map

- `tests/test_product_scope.py`: executable product-boundary checks.
- `tests/test_image_input_service.py`: OneBot parsing, image validation, size limiting, and data-URL conversion.
- `tests/test_multimodal_chat.py`: multimodal message construction and safe history representation.
- `tests/test_main_image_flow.py`: private/group image events through the application entrypoint.
- `tests/test_llm_image_fallback.py`: fallback behavior when providers reject images.
- `tests/test_user_facing_scope.py`: README, help, and system-prompt consistency.
- `src/services/image_input_service.py`: the only image-input parsing/downloading module.
- `src/chat/chat_service.py`: creates multimodal user content and placeholder history.
- `src/main.py`: accepts image-only events and passes loaded images to chat.
- `src/services/llm_client.py`: distinguishes all-model image failures from generic configuration failures.
- `src/config.py`, `src/commands/search.py`, `src/services/onebot_client.py`: residual removal only.
- `README.md`, `src/commands/help.py`, `src/chat/prompt.py`: user-facing scope alignment.

---

### Task 1: Lock the Product Boundary and Remove Residual Surfaces

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_product_scope.py`
- Modify: `.gitignore`
- Modify: `src/config.py:68-143`
- Modify: `src/commands/search.py:1-42`
- Modify: `src/services/onebot_client.py:1-77`

**Interfaces:**
- Consumes: `Config.__dataclass_fields__`, `COMMANDS`, `search_reply(query, session_key, raw_message)`, `OneBotClient`.
- Produces: a text-only outbound OneBot client and a keyword-only `/search` command while preserving internal `search_service.fetch_url`.

- [ ] **Step 1: Stop ignoring tests and add the failing product-boundary test**

Remove `test_*.py` from `.gitignore`, create an empty `tests/__init__.py`, and create `tests/test_product_scope.py`:

```python
import unittest
from types import SimpleNamespace
from unittest import mock

import src.commands.search as search_command
import src.services.search_service as search_service
from src.commands import COMMANDS
from src.config import Config
from src.services.onebot_client import OneBotClient
from src.services.search_service import SearchResult


class ProductScopeTests(unittest.TestCase):
    def test_generation_and_video_configuration_is_absent(self):
        fields = set(Config.__dataclass_fields__)
        residual = sorted(
            name
            for name in fields
            if name == "openai_api_key"
            or name.startswith("video_")
            or name.startswith("image_")
            or name.startswith("comfyui_")
        )
        self.assertEqual([], residual)

    def test_onebot_client_has_no_outbound_image_method(self):
        self.assertFalse(hasattr(OneBotClient, "send_image"))

    def test_expected_commands_are_preserved(self):
        self.assertTrue(
            {"search", "help", "reset", "remember", "globalremember"}.issubset(COMMANDS)
        )

    def test_search_with_url_uses_keyword_search_not_direct_fetch(self):
        with (
            mock.patch.object(
                search_command,
                "extract_first_url",
                return_value="https://example.com/page",
                create=True,
            ),
            mock.patch.object(
                search_command,
                "fetch_url",
                return_value=SimpleNamespace(ok=True, text="direct page"),
                create=True,
            ) as direct_fetch,
            mock.patch.object(
                search_command,
                "search",
                return_value=SearchResult(ok=True, status="success", text="search result"),
            ) as keyword_search,
            mock.patch.object(search_command, "generate_reply", return_value="answer"),
        ):
            result = search_command.search_reply(
                "https://example.com/page", "private:1", "/search https://example.com/page"
            )

        self.assertEqual("answer", result)
        keyword_search.assert_called_once_with("https://example.com/page")
        direct_fetch.assert_not_called()

    def test_search_internal_page_fetch_is_preserved(self):
        self.assertTrue(hasattr(search_service, "fetch_url"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the scope test and verify the red state**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_product_scope -v
```

Expected: failures report residual `openai_api_key`/`video_*`/`image_*`/`comfyui_*` fields, `send_image` still exists, and URL input bypasses `search()`.

- [ ] **Step 3: Remove only the tested residual production surfaces**

In `src/config.py`, delete `openai_api_key`, all `video_*`, all `image_*`, and all `comfyui_*` dataclass fields. Keep LLM, OneBot, search, history, memory, timeout, and reply-size fields unchanged.

Replace `src/commands/search.py` with the keyword-search-only implementation:

```python
from src.chat.chat_service import generate_reply
from src.services.search_service import has_search_results, normalize_search_query, search


def search_reply(query: str, session_key: str, raw_message: str) -> str:
    query = normalize_search_query(query)
    if not query:
        return "想搜什么？比如：/search DeepSeek 最新消息"

    search_result = search(query)
    if not has_search_results(search_result):
        tool_context = (
            "这是 /search 命令的搜索失败结果。请按 ATRI 的角色设定回答用户："
            "说明没有搜到可靠结果，所以不知道或无法确认；不要猜测，不要编造成确定事实。\n"
            f"搜索状态：\n{search_result.text}"
        )
    else:
        tool_context = f"网页搜索结果：\n{search_result.text}"

    return generate_reply(session_key, raw_message, tool_context)
```

In `src/services/onebot_client.py`, delete `from pathlib import Path` and delete the complete `send_image()` method. Do not change `send_msg()`.

- [ ] **Step 4: Run the scope test and verify green**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_product_scope -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit the bounded cleanup**

```powershell
git add .gitignore tests/__init__.py tests/test_product_scope.py src/config.py src/commands/search.py src/services/onebot_client.py
git commit -m "refactor: remove non-product media surfaces"
```

---

### Task 2: Parse and Load User-Supplied Images Safely

**Files:**
- Create: `tests/test_image_input_service.py`
- Create: `src/services/image_input_service.py`

**Interfaces:**
- Consumes: OneBot event dictionaries, CQ image text, `config.proxies`, `config.request_timeout`, and `try_proxied_get()`.
- Produces: `ParsedImageMessage(text: str, image_urls: tuple[str, ...])`, `ImageInputError`, `parse_image_message(data, raw_text)`, and `load_chat_images(image_urls) -> list[str]`.

- [ ] **Step 1: Write the failing parser and downloader tests**

Create `tests/test_image_input_service.py`:

```python
import importlib
import unittest
from unittest import mock


class FakeResponse:
    def __init__(self, chunks, content_type="image/png", content_length=""):
        self._chunks = chunks
        self.headers = {"Content-Type": content_type}
        if content_length:
            self.headers["Content-Length"] = content_length
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=65536):
        yield from self._chunks

    def close(self):
        self.closed = True


class ImageInputServiceTests(unittest.TestCase):
    def service(self):
        try:
            return importlib.import_module("src.services.image_input_service")
        except ModuleNotFoundError as error:
            self.fail(f"image input service is missing: {error}")

    def test_parses_structured_image_and_removes_cq_image_from_text(self):
        service = self.service()
        event = {
            "message": [
                {"type": "text", "data": {"text": "看看 "}},
                {"type": "image", "data": {"url": "https://img.example/a.png"}},
            ]
        }
        parsed = service.parse_image_message(
            event,
            "看看 [CQ:image,file=a.png,url=https://img.example/a.png]",
        )
        self.assertEqual("看看", parsed.text)
        self.assertEqual(("https://img.example/a.png",), parsed.image_urls)

    def test_falls_back_to_cq_url(self):
        service = self.service()
        parsed = service.parse_image_message(
            {}, "[CQ:image,file=a.png,url=https://img.example/a.png]"
        )
        self.assertEqual("", parsed.text)
        self.assertEqual(("https://img.example/a.png",), parsed.image_urls)

    def test_rejects_more_than_four_images(self):
        service = self.service()
        event = {
            "message": [
                {"type": "image", "data": {"url": f"https://img.example/{index}.png"}}
                for index in range(5)
            ]
        }
        with self.assertRaisesRegex(service.ImageInputError, "最多发送 4 张图片"):
            service.parse_image_message(event, "images")

    def test_loads_valid_image_as_data_url_and_closes_response(self):
        service = self.service()
        response = FakeResponse([b"png-bytes"])
        with mock.patch.object(service, "try_proxied_get", return_value=response):
            loaded = service.load_chat_images(["https://img.example/a.png"])
        self.assertEqual(["data:image/png;base64,cG5nLWJ5dGVz"], loaded)
        self.assertTrue(response.closed)

    def test_rejects_non_image_content(self):
        service = self.service()
        response = FakeResponse([b"html"], content_type="text/html")
        with (
            mock.patch.object(service, "try_proxied_get", return_value=response),
            self.assertRaisesRegex(service.ImageInputError, "不是支持的图片格式"),
        ):
            service.load_chat_images(["https://img.example/a"])

    def test_rejects_stream_larger_than_five_mib(self):
        service = self.service()
        response = FakeResponse([b"x" * (5 * 1024 * 1024), b"x"])
        with (
            mock.patch.object(service, "try_proxied_get", return_value=response),
            self.assertRaisesRegex(service.ImageInputError, "不能超过 5 MiB"),
        ):
            service.load_chat_images(["https://img.example/large.png"])

    def test_wraps_download_failure_as_user_facing_error(self):
        service = self.service()
        with (
            mock.patch.object(service, "try_proxied_get", side_effect=OSError("offline")),
            self.assertRaisesRegex(service.ImageInputError, "图片读取失败"),
        ):
            service.load_chat_images(["https://img.example/a.png"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the image service tests and verify the red state**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_image_input_service -v
```

Expected: all tests fail with `image input service is missing`.

- [ ] **Step 3: Implement the focused image-input service**

Create `src/services/image_input_service.py`:

```python
import base64
import html
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from src.config import config
from src.util import try_proxied_get


MAX_CHAT_IMAGES = 4
MAX_CHAT_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
CQ_IMAGE_PATTERN = re.compile(r"\[CQ:image,([^\]]+)\]", re.IGNORECASE)


class ImageInputError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedImageMessage:
    text: str
    image_urls: tuple[str, ...]


def _structured_image_urls(data: dict[str, Any]) -> tuple[bool, list[str]]:
    message = data.get("message")
    if not isinstance(message, list):
        return False, []
    saw_image = False
    urls = []
    for segment in message:
        if not isinstance(segment, dict) or segment.get("type") != "image":
            continue
        saw_image = True
        segment_data = segment.get("data")
        if isinstance(segment_data, dict):
            url = str(segment_data.get("url") or "").strip()
            if url:
                urls.append(url)
    return saw_image, urls


def _cq_image_urls(raw_text: str) -> tuple[bool, list[str]]:
    matches = CQ_IMAGE_PATTERN.findall(str(raw_text or ""))
    urls = []
    for attributes in matches:
        match = re.search(r"(?:^|,)url=([^,]+)", attributes, flags=re.IGNORECASE)
        if match:
            urls.append(html.unescape(match.group(1).strip()))
    return bool(matches), urls


def parse_image_message(data: dict[str, Any], raw_text: str) -> ParsedImageMessage:
    structured_saw_image, structured_urls = _structured_image_urls(data)
    cq_saw_image, cq_urls = _cq_image_urls(raw_text)
    saw_image = structured_saw_image or cq_saw_image
    urls = structured_urls or cq_urls
    urls = list(dict.fromkeys(urls))

    if saw_image and not urls:
        raise ImageInputError("没有取得可读取的图片地址，请重新发送图片。")
    if len(urls) > MAX_CHAT_IMAGES:
        raise ImageInputError(f"每条消息最多发送 {MAX_CHAT_IMAGES} 张图片。")

    text = CQ_IMAGE_PATTERN.sub("", str(raw_text or "")).strip()
    return ParsedImageMessage(text=text, image_urls=tuple(urls))


def _content_type(response) -> str:
    return str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()


def load_chat_images(image_urls: list[str] | tuple[str, ...]) -> list[str]:
    loaded = []
    for url in image_urls:
        if urlparse(url).scheme.lower() not in {"http", "https"}:
            raise ImageInputError("图片地址无效，只支持 http/https 图片。")
        try:
            response = try_proxied_get(
                url,
                proxies=config.proxies,
                timeout=config.request_timeout,
                stream=True,
            )
            try:
                response.raise_for_status()
                mime_type = _content_type(response)
                if mime_type not in ALLOWED_IMAGE_TYPES:
                    raise ImageInputError("收到的内容不是支持的图片格式。")
                content_length = str(response.headers.get("Content-Length") or "").strip()
                if content_length.isdigit() and int(content_length) > MAX_CHAT_IMAGE_BYTES:
                    raise ImageInputError("每张图片不能超过 5 MiB。")

                content = bytearray()
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    content.extend(chunk)
                    if len(content) > MAX_CHAT_IMAGE_BYTES:
                        raise ImageInputError("每张图片不能超过 5 MiB。")
                if not content:
                    raise ImageInputError("图片内容为空，请重新发送。")
                encoded = base64.b64encode(bytes(content)).decode("ascii")
                loaded.append(f"data:{mime_type};base64,{encoded}")
            finally:
                response.close()
        except ImageInputError:
            raise
        except Exception as error:
            raise ImageInputError("图片读取失败，请重新发送。") from error
    return loaded
```

- [ ] **Step 4: Run the image service tests and verify green**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_image_input_service -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit the image-input service**

```powershell
git add tests/test_image_input_service.py src/services/image_input_service.py
git commit -m "feat: parse and load chat image inputs"
```

---

### Task 3: Build Multimodal Model Messages Without Persisting Images

**Files:**
- Create: `tests/test_multimodal_chat.py`
- Modify: `src/chat/chat_service.py:203-276`

**Interfaces:**
- Consumes: `image_data_urls: list[str] | None` from the application entrypoint.
- Produces: `build_user_content(text, image_data_urls)`, `history_user_text(text, image_count)`, and extended `generate_reply(..., image_data_urls=None)`.

- [ ] **Step 1: Write the failing multimodal and history tests**

Create `tests/test_multimodal_chat.py`:

```python
import unittest
from unittest import mock

import src.chat.chat_service as chat_service
from src.services.llm_types import ChatResponse


class MultimodalChatTests(unittest.TestCase):
    def test_builds_openai_compatible_multimodal_content(self):
        content = chat_service.build_user_content(
            "这是什么？", ["data:image/png;base64,cG5n"]
        )
        self.assertEqual(
            [
                {"type": "text", "text": "这是什么？"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}},
            ],
            content,
        )

    def test_image_only_message_gets_a_text_instruction(self):
        content = chat_service.build_user_content("", ["data:image/png;base64,cG5n"])
        self.assertEqual("请识别图片内容并回答。", content[0]["text"])

    def test_history_uses_placeholders_and_never_contains_image_data(self):
        history_text = chat_service.history_user_text(
            "帮我看看", 2
        )
        self.assertEqual("[图片]\n[图片]\n帮我看看", history_text)
        self.assertNotIn("base64", history_text)
        self.assertNotIn("http", history_text)

    def test_text_only_content_remains_a_string(self):
        self.assertEqual("你好", chat_service.build_user_content("你好", []))

    def test_generate_reply_persists_placeholder_not_image_data(self):
        session_key = "test:image-history"
        chat_service.chat_history.pop(session_key, None)
        with (
            mock.patch.object(chat_service, "_ensure_history_loaded"),
            mock.patch.object(chat_service, "_save_history_unlocked"),
            mock.patch.object(
                chat_service.llm,
                "chat",
                return_value=ChatResponse(content="看到了"),
            ),
        ):
            reply = chat_service.generate_reply(
                session_key,
                "帮我看看",
                tool_context="已有上下文",
                image_data_urls=["data:image/png;base64,cG5n"],
            )
        history = chat_service.chat_history.pop(session_key)
        self.assertEqual("看到了", reply)
        self.assertEqual("[图片]\n帮我看看", history[0]["content"])
        self.assertNotIn("base64", str(history))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the multimodal tests and verify the red state**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_multimodal_chat -v
```

Expected: failures report missing `build_user_content` and `history_user_text`.

- [ ] **Step 3: Add the minimal multimodal helpers and wire them into `generate_reply`**

Add before `generate_reply()` in `src/chat/chat_service.py`:

```python
def build_user_content(text: str, image_data_urls: list[str]):
    text = str(text or "").strip()
    if not image_data_urls:
        return text
    content: list[dict[str, Any]] = [
        {"type": "text", "text": text or "请识别图片内容并回答。"}
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_data_url}}
        for image_data_url in image_data_urls
    )
    return content


def history_user_text(text: str, image_count: int) -> str:
    parts = ["[图片]"] * max(image_count, 0)
    text = str(text or "").strip()
    if text:
        parts.append(text)
    return "\n".join(parts)
```

Change the signature and user-message/history lines in `generate_reply()`:

```python
def generate_reply(
    session_key: str,
    text: str,
    tool_context: str = "",
    image_data_urls: list[str] | None = None,
) -> str:
    images = list(image_data_urls or [])
    _ensure_history_loaded(session_key)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(session_key, tool_context)},
        {"role": "user", "content": build_untrusted_context(session_key, tool_context)},
    ]
    with chat_history_lock:
        messages.extend(chat_history.get(session_key, []).copy())
    messages.append({"role": "user", "content": build_user_content(text, images)})
```

Keep the existing tool loop unchanged, but pass `text or "图片内容"` as the fallback query to `build_tool_messages()`. Replace the final append with:

```python
    append_history(session_key, history_user_text(text, len(images)), reply)
```

- [ ] **Step 4: Run the multimodal tests and the existing scope tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_multimodal_chat tests.test_product_scope -v
```

Expected: 10 tests pass.

- [ ] **Step 5: Commit multimodal chat construction**

```powershell
git add tests/test_multimodal_chat.py src/chat/chat_service.py
git commit -m "feat: pass image inputs to chat models"
```

---

### Task 4: Route Private and Group Image Events Through Chat

**Files:**
- Create: `tests/test_main_image_flow.py`
- Modify: `src/main.py:8-204`

**Interfaces:**
- Consumes: `parse_image_message(data, raw_text)`, `load_chat_images(image_urls)`, and extended `generate_reply()`.
- Produces: image-only and image-plus-text processing for private messages and group messages that mention the bot.

- [ ] **Step 1: Write failing private/group image-flow tests**

Create `tests/test_main_image_flow.py`:

```python
import unittest
from unittest import mock

import src.main as main


class MainImageFlowTests(unittest.TestCase):
    def test_private_image_only_message_reaches_multimodal_chat(self):
        event = {
            "post_type": "message",
            "message_type": "private",
            "user_id": 1,
            "self_id": 9,
            "message_id": 10,
            "raw_message": "[CQ:image,file=a.png,url=https://img.example/a.png]",
            "message": [
                {"type": "image", "data": {"url": "https://img.example/a.png"}}
            ],
        }
        with (
            mock.patch.object(main, "load_chat_images", return_value=["data:image/png;base64,cG5n"], create=True),
            mock.patch.object(main, "generate_reply", return_value="看到了") as generate,
            mock.patch.object(main.onebot, "send_msg") as send,
        ):
            main.process_message(event)

        generate.assert_called_once_with(
            "private:1", "", image_data_urls=["data:image/png;base64,cG5n"]
        )
        send.assert_called_once_with("1", "看到了", is_group=False)

    def test_group_image_requires_and_accepts_bot_mention(self):
        event = {
            "post_type": "message",
            "message_type": "group",
            "group_id": 20,
            "user_id": 1,
            "self_id": 9,
            "message_id": 11,
            "raw_message": "[CQ:at,qq=9] [CQ:image,file=a.png,url=https://img.example/a.png]",
            "message": [
                {"type": "at", "data": {"qq": "9"}},
                {"type": "image", "data": {"url": "https://img.example/a.png"}},
            ],
        }
        with (
            mock.patch.object(main, "load_chat_images", return_value=["data:image/png;base64,cG5n"], create=True),
            mock.patch.object(main, "generate_reply", return_value="群图") as generate,
            mock.patch.object(main.onebot, "send_msg") as send,
        ):
            main.process_message(event)

        generate.assert_called_once_with(
            "group:20:1", "", image_data_urls=["data:image/png;base64,cG5n"]
        )
        send.assert_called_once_with(20, "群图", is_group=True)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the flow tests and verify the red state**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_main_image_flow -v
```

Expected: both tests fail because `process_message()` still passes raw CQ text and no `image_data_urls`.

- [ ] **Step 3: Integrate image parsing/loading into `src/main.py`**

Add imports:

```python
from src.services.image_input_service import (
    ImageInputError,
    load_chat_images,
    parse_image_message,
)
```

After group mention handling, parse the message:

```python
    try:
        parsed_message = parse_image_message(data, raw_msg)
    except ImageInputError as error:
        send_reply(target_id, str(error), is_group)
        return

    route_text = parsed_message.text or ("[图片]" if parsed_message.image_urls else "")
    if not route_text:
        return
```

Route using `route_text`. Keep command behavior unchanged. In the chat branch, load images and call:

```python
        image_data_urls = load_chat_images(parsed_message.image_urls)
        reply = generate_reply(
            session_key,
            parsed_message.text,
            image_data_urls=image_data_urls,
        )
```

Add `except ImageInputError as error` before the existing `RuntimeError` handler so download/validation failures are sent directly to the user.

- [ ] **Step 4: Run private/group flow plus parser tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_main_image_flow tests.test_image_input_service -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Commit OneBot image routing**

```powershell
git add tests/test_main_image_flow.py src/main.py
git commit -m "feat: route OneBot images through chat"
```

---

### Task 5: Return a Specific Error When Every Model Rejects Images

**Files:**
- Create: `tests/test_llm_image_fallback.py`
- Modify: `src/services/llm_client.py:45-148`
- Modify: `src/main.py:199-204`

**Interfaces:**
- Consumes: OpenAI-compatible message lists with `image_url` content blocks.
- Produces: `ImageRecognitionUnavailable(RuntimeError)` and preserved generic `RuntimeError` for text-only exhaustion.

- [ ] **Step 1: Write failing image-fallback tests**

Create `tests/test_llm_image_fallback.py`:

```python
import unittest
from unittest import mock

import src.services.llm_client as llm_client
import src.main as main
from src.services.llm_types import LLMModelSpec


class FailingClient:
    def chat(self, *args, **kwargs):
        raise RuntimeError("model does not support image input")


class LlmImageFallbackTests(unittest.TestCase):
    def image_error_type(self):
        error_type = getattr(llm_client, "ImageRecognitionUnavailable", None)
        self.assertIsNotNone(error_type, "ImageRecognitionUnavailable is missing")
        return error_type

    def test_image_exhaustion_uses_specific_error(self):
        error_type = self.image_error_type()
        client = llm_client.FallbackLLMClient(
            [LLMModelSpec(provider="gemini", model="vision-test")]
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看看"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}},
                ],
            }
        ]
        with (
            mock.patch.object(client, "_get_client", return_value=FailingClient()),
            self.assertRaisesRegex(
                error_type,
                "当前模型无法识别该图片",
            ),
        ):
            client.chat(messages)

    def test_text_exhaustion_keeps_generic_error(self):
        client = llm_client.FallbackLLMClient(
            [LLMModelSpec(provider="gemini", model="text-test")]
        )
        with (
            mock.patch.object(client, "_get_client", return_value=FailingClient()),
            self.assertRaisesRegex(RuntimeError, "所有模型暂时不可用"),
        ):
            client.chat([{"role": "user", "content": "你好"}])

    def test_main_sends_image_specific_error_without_config_prefix(self):
        event = {
            "post_type": "message",
            "message_type": "private",
            "user_id": 1,
            "self_id": 9,
            "message_id": 12,
            "raw_message": "[CQ:image,file=a.png,url=https://img.example/a.png]",
            "message": [
                {"type": "image", "data": {"url": "https://img.example/a.png"}}
            ],
        }
        error_type = self.image_error_type()
        with (
            mock.patch.object(main, "load_chat_images", return_value=["data:image/png;base64,cG5n"]),
            mock.patch.object(
                main,
                "generate_reply",
                side_effect=error_type("当前模型无法识别该图片。"),
            ),
            mock.patch.object(main.onebot, "send_msg") as send,
        ):
            main.process_message(event)
        send.assert_called_once_with("1", "当前模型无法识别该图片。", is_group=False)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the fallback tests and verify the red state**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_llm_image_fallback -v
```

Expected: image test fails because `ImageRecognitionUnavailable` is absent; text test passes.

- [ ] **Step 3: Implement image-aware exhaustion without hardcoding provider capability**

Add near the fallback status constants in `src/services/llm_client.py`:

```python
class ImageRecognitionUnavailable(RuntimeError):
    pass


def _messages_have_images(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        if any(
            isinstance(item, dict) and item.get("type") == "image_url"
            for item in content
        ):
            return True
    return False
```

At the start of `FallbackLLMClient.chat()`, set:

```python
        has_images = _messages_have_images(messages)
```

Replace the final exhaustion raise with:

```python
        if has_images:
            raise ImageRecognitionUnavailable("当前模型无法识别该图片。")
        raise RuntimeError("所有模型暂时不可用，请稍后再试。")
```

In `src/main.py`, import `ImageRecognitionUnavailable` and catch it before `RuntimeError`:

```python
    except ImageRecognitionUnavailable as error:
        logger.info("Image recognition unavailable session_key=%s", session_key)
        send_reply(target_id, str(error), is_group)
```

- [ ] **Step 4: Run fallback and application flow tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_llm_image_fallback tests.test_main_image_flow -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit image-aware fallback behavior**

```powershell
git add tests/test_llm_image_fallback.py src/services/llm_client.py src/main.py
git commit -m "feat: report unsupported image recognition"
```

---

### Task 6: Align User-Facing Scope and Run Full Verification

**Files:**
- Create: `tests/test_user_facing_scope.py`
- Modify: `README.md`
- Modify: `src/commands/help.py`
- Modify: `src/chat/prompt.py`

**Interfaces:**
- Consumes: `help_text()` and `build_system_prompt()`.
- Produces: consistent statements that permit image understanding and prohibit direct URL reading, generation, editing, and outbound images.

- [ ] **Step 1: Write the failing user-facing scope test**

Create `tests/test_user_facing_scope.py`:

```python
import unittest
from pathlib import Path

from src.chat.prompt import build_system_prompt
from src.commands.help import help_text


class UserFacingScopeTests(unittest.TestCase):
    def test_help_mentions_image_understanding_not_generation(self):
        text = help_text()
        self.assertIn("发送图片", text)
        self.assertIn("识别", text)
        self.assertIn("不支持图片生成", text)

    def test_system_prompt_allows_input_images_but_forbids_output_images(self):
        prompt = build_system_prompt("private:1")
        self.assertIn("理解用户随消息提供的图片", prompt)
        self.assertIn("不能生成、编辑或主动发送图片", prompt)

    def test_readme_describes_keyword_search_and_image_input(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("图片理解", readme)
        self.assertIn("/search <关键词>", readme)
        self.assertIn("不提供独立 URL 直读", readme)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the user-facing test and verify the red state**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_user_facing_scope -v
```

Expected: failures show image understanding is not yet documented and the prompt still forbids all image capability.

- [ ] **Step 3: Update README, help, and system prompt with exact scope**

In `README.md`:

- Add image understanding to “能做什么”: users may send up to 4 JPEG/PNG/WebP/GIF images, each at most 5 MiB, with optional text.
- State that actual recognition depends on the configured model.
- Change the exclusions to “不提供图片生成、图片编辑、主动发图、视频理解、天气、B站和独立 URL 直读”.
- Clarify `/search <关键词>` is keyword search; search may internally read result pages.
- Keep memory, reset, and model fallback documentation.

In `src/commands/help.py`, include these exact concepts:

```python
"也可以直接发送图片，或发送图片加文字，让支持图片理解的模型回答。\n"
"不支持图片生成、图片编辑、主动发图、视频理解、天气、B站或独立 URL 直读。"
```

In `src/chat/prompt.py`, replace the blanket image prohibition with:

```python
"你可以理解用户随消息提供的图片；图片是否能被识别取决于当前模型能力。\n"
"你不能生成、编辑或主动发送图片，也不能调用视频理解、天气、B站、独立 URL 直读或文件功能。\n"
```

- [ ] **Step 4: Run the full test suite**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -v
```

Expected: all tests pass with no errors or warnings from test code.

- [ ] **Step 5: Run static and scope verification**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -c "import ast,pathlib; files=list(pathlib.Path('.').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print(f'AST_OK={len(files)}')"
git diff --check
rg -n "OPENAI_API_KEY|VIDEO_|COMFYUI_|IMAGE_ENABLE|IMAGE_CHAT_TOOL_ENABLE|send_image" src
git status --short
```

Expected: AST parsing succeeds; `git diff --check` is empty; the residual scan returns no matches; status contains only the planned source, test, and documentation changes. Existing `.env`, `atri_data/history`, and memory data remain unchanged.

- [ ] **Step 6: Commit user-facing scope alignment**

```powershell
git add README.md src/commands/help.py src/chat/prompt.py tests/test_user_facing_scope.py
git commit -m "docs: align chat search and image scope"
```

- [ ] **Step 7: Re-run verification on the committed tree**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -v
git status --short --branch
```

Expected: all tests pass; the working tree is clean; the branch is ahead of `origin/main` only by the intentional local commits.
