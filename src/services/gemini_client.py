"""Gemini Developer API client using native generateContent REST."""

from __future__ import annotations

import base64
import copy
import json
from typing import Any
from urllib.parse import quote

from src.config import Config
from src.services.llm_types import ChatResponse
from src.util import try_proxied_post


DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def _append_content(
    contents: list[dict[str, Any]],
    role: str,
    parts: list[dict[str, Any]],
) -> None:
    if not parts:
        return
    if contents and contents[-1]["role"] == role:
        contents[-1]["parts"].extend(parts)
        return
    contents.append({"role": role, "parts": parts})


def _data_url_part(url: str) -> dict[str, Any]:
    header, separator, encoded = str(url or "").partition(",")
    if (
        not separator
        or not header.startswith("data:")
        or ";base64" not in header.casefold()
    ):
        raise RuntimeError("Gemini 图片必须是 base64 data URL")
    mime_type = header[5:].split(";", 1)[0].strip().lower()
    if not mime_type.startswith("image/"):
        raise RuntimeError("Gemini 图片 data URL 缺少有效图片 MIME 类型")
    try:
        base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise RuntimeError("Gemini 图片 base64 数据无效") from error
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": encoded,
        }
    }


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = str(item.get("text") or "")
            if text:
                parts.append({"text": text})
        elif item_type == "image_url":
            image = item.get("image_url")
            url = image.get("url") if isinstance(image, dict) else ""
            parts.append(_data_url_part(str(url or "")))
    return parts


def _function_call_part(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise RuntimeError("内部工具调用缺少 function")
    name = str(function.get("name") or "").strip()
    arguments = function.get("arguments", "{}")
    try:
        parsed_arguments = (
            json.loads(arguments) if isinstance(arguments, str) else arguments
        )
    except json.JSONDecodeError as error:
        raise RuntimeError("内部工具调用参数不是有效 JSON") from error
    if not name or not isinstance(parsed_arguments, dict):
        raise RuntimeError("内部工具调用名称或参数无效")
    return {
        "functionCall": {
            "name": name,
            "args": parsed_arguments,
        }
    }


def _native_messages(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system_parts: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []
    native_call_ids: dict[str, str] = {}

    for message in messages:
        role = str(message.get("role") or "")
        if role == "system":
            system_parts.extend(_content_parts(message.get("content")))
            continue
        if role == "tool":
            name = str(message.get("name") or "").strip()
            if not name:
                raise RuntimeError("内部工具结果缺少名称")
            function_response: dict[str, Any] = {
                "name": name,
                "response": {
                    "result": str(message.get("content") or "")
                },
            }
            internal_id = str(message.get("tool_call_id") or "")
            native_id = native_call_ids.get(internal_id)
            if native_id:
                function_response["id"] = native_id
            _append_content(
                contents,
                "user",
                [
                    {
                        "functionResponse": function_response
                    }
                ],
            )
            continue

        native_role = "model" if role == "assistant" else "user"
        parts = _content_parts(message.get("content"))
        if role == "assistant":
            raw_calls = message.get("tool_calls") or []
            provider_context = message.get("_provider_context")
            native_content = (
                provider_context.get("content")
                if (
                    isinstance(provider_context, dict)
                    and provider_context.get("provider") == "gemini"
                )
                else None
            )
            if isinstance(native_content, dict):
                native_parts = native_content.get("parts")
                if not isinstance(native_parts, list):
                    raise RuntimeError(
                        "Gemini provider context 缺少原始 parts"
                    )
                internal_calls = (
                    raw_calls if isinstance(raw_calls, list) else []
                )
                returned_calls = [
                    part.get("functionCall")
                    for part in native_parts
                    if (
                        isinstance(part, dict)
                        and isinstance(part.get("functionCall"), dict)
                    )
                ]
                for internal_call, returned_call in zip(
                    internal_calls,
                    returned_calls,
                ):
                    internal_id = str(
                        internal_call.get("id") or ""
                    )
                    returned_id = str(
                        returned_call.get("id") or ""
                    )
                    if internal_id and returned_id:
                        native_call_ids[internal_id] = returned_id
                contents.append(copy.deepcopy(native_content))
                continue

            if isinstance(raw_calls, list):
                call_parts = [
                    _function_call_part(call)
                    for call in raw_calls
                    if isinstance(call, dict)
                ]
                if call_parts:
                    call_parts[0][
                        "thoughtSignature"
                    ] = DUMMY_THOUGHT_SIGNATURE
                    parts.extend(call_parts)
        _append_content(contents, native_role, parts)

    system_instruction = (
        {"parts": system_parts} if system_parts else None
    )
    return system_instruction, contents


def _native_tools(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        declaration = {
            "name": name,
            "description": str(function.get("description") or ""),
            "parameters": function.get("parameters") or {
                "type": "object",
                "properties": {},
            },
        }
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}] if declarations else []


def _native_tool_config(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if tool_choice is None or tool_choice == "auto":
        mode = "AUTO"
        allowed_names: list[str] = []
    elif tool_choice == "none":
        mode = "NONE"
        allowed_names = []
    elif tool_choice in {"required", "any"}:
        mode = "ANY"
        allowed_names = []
    elif isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        name = (
            str(function.get("name") or "").strip()
            if isinstance(function, dict)
            else ""
        )
        mode = "ANY"
        allowed_names = [name] if name else []
    else:
        raise RuntimeError("Gemini 不支持当前 tool_choice")

    config: dict[str, Any] = {"mode": mode}
    if allowed_names:
        config["allowedFunctionNames"] = allowed_names
    return {"functionCallingConfig": config}


def _parse_response(data: dict[str, Any]) -> ChatResponse:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        feedback = data.get("promptFeedback")
        reason = (
            str(feedback.get("blockReason") or "")
            if isinstance(feedback, dict)
            else ""
        )
        suffix = f"：{reason}" if reason else ""
        raise RuntimeError(f"Gemini 未返回候选结果{suffix}")

    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise RuntimeError("Gemini 候选结果缺少 content.parts")

    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for index, part in enumerate(parts, 1):
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
        function_call = part.get("functionCall")
        if not isinstance(function_call, dict):
            continue
        name = str(function_call.get("name") or "").strip()
        arguments = function_call.get("args")
        if not name or not isinstance(arguments, dict):
            continue
        call_id = str(
            function_call.get("id") or f"gemini_call_{index}"
        )
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        )

    result = ChatResponse(
        content="\n".join(texts).strip(),
        tool_calls=tool_calls,
        provider_context={
            "provider": "gemini",
            "content": copy.deepcopy(content),
        },
    )
    if not result.content and not result.tool_calls:
        raise RuntimeError("Gemini 返回空内容且没有函数调用")
    return result


class GeminiClient:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResponse:
        if not self._cfg.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        model_name = str(model or "").strip()
        if not model_name:
            raise RuntimeError("Gemini model is not configured")

        system_instruction, contents = _native_messages(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        native_tools = _native_tools(tools)
        if native_tools:
            payload["tools"] = native_tools
            payload["toolConfig"] = _native_tool_config(tool_choice)

        encoded_model = quote(model_name, safe="")
        url = (
            f"{self._cfg.gemini_url.rstrip('/')}/models/"
            f"{encoded_model}:generateContent"
        )
        response = try_proxied_post(
            url,
            proxies=self._cfg.proxies,
            json=payload,
            headers={
                "x-goog-api-key": self._cfg.gemini_api_key,
                "Content-Type": "application/json",
            },
            timeout=self._cfg.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Gemini 返回的 JSON 不是对象")
        return _parse_response(data)
