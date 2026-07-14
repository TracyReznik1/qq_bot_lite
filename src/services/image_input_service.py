import base64
import html
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from src.config import config
from src.services.url_fetch_service import (
    MAX_REDIRECTS,
    REDIRECT_STATUS_CODES,
    _validate_url,
)
from src.util import try_proxied_get


MAX_CHAT_IMAGES = 4
MAX_CHAT_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
CQ_IMAGE_PATTERN = re.compile(r"\[CQ:image,([^\]]+)\]", re.IGNORECASE)
TRUSTED_ONEBOT_IMAGE_HOSTS = {"multimedia.nt.qq.com.cn"}


class ImageInputError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedImageMessage:
    text: str
    image_urls: tuple[str, ...]
    image_file_ids: tuple[str, ...] = ()


def _structured_image_sources(data: dict[str, Any]) -> list[tuple[str, str]]:
    message = data.get("message")
    if not isinstance(message, list):
        return []
    sources = []
    for segment in message:
        if not isinstance(segment, dict) or segment.get("type") != "image":
            continue
        segment_data = segment.get("data")
        if not isinstance(segment_data, dict):
            sources.append(("", ""))
            continue
        url = str(segment_data.get("url") or "").strip()
        file_id = str(segment_data.get("file") or "").strip()
        sources.append((url, file_id))
    return sources


def _cq_image_sources(raw_text: str) -> list[tuple[str, str]]:
    matches = CQ_IMAGE_PATTERN.findall(str(raw_text or ""))
    sources = []
    for attributes in matches:
        url_match = re.search(
            r"(?:^|,)url=([^,]+)", attributes, flags=re.IGNORECASE
        )
        file_match = re.search(
            r"(?:^|,)file=([^,]+)", attributes, flags=re.IGNORECASE
        )
        url = html.unescape(url_match.group(1).strip()) if url_match else ""
        file_id = (
            html.unescape(file_match.group(1).strip()) if file_match else ""
        )
        sources.append((url, file_id))
    return sources


def parse_image_message(data: dict[str, Any], raw_text: str) -> ParsedImageMessage:
    structured_sources = _structured_image_sources(data)
    cq_sources = _cq_image_sources(raw_text)
    image_count = max(len(structured_sources), len(cq_sources))

    if image_count > MAX_CHAT_IMAGES:
        raise ImageInputError(f"每条消息最多发送 {MAX_CHAT_IMAGES} 张图片。")

    sources = []
    for index in range(image_count):
        structured_url, structured_file = (
            structured_sources[index]
            if index < len(structured_sources)
            else ("", "")
        )
        cq_url, cq_file = (
            cq_sources[index] if index < len(cq_sources) else ("", "")
        )
        url = structured_url or cq_url
        file_id = structured_file or cq_file
        if not url and not file_id:
            raise ImageInputError("没有取得可读取的图片地址，请重新发送图片。")
        sources.append((url, file_id))

    deduped_sources = []
    seen = set()
    for url, file_id in sources:
        key = ("url", url) if url else ("file", file_id)
        if key in seen:
            continue
        seen.add(key)
        deduped_sources.append((url, file_id))

    text = CQ_IMAGE_PATTERN.sub("", str(raw_text or "")).strip()
    return ParsedImageMessage(
        text=text,
        image_urls=tuple(url for url, _file_id in deduped_sources),
        image_file_ids=tuple(file_id for _url, file_id in deduped_sources),
    )


def _content_type(response) -> str:
    return str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()


def _close_response(response) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _is_trusted_onebot_image_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme.lower() == "https"
        and str(parsed.hostname or "").lower() in TRUSTED_ONEBOT_IMAGE_HOSTS
    )


def _fetch_image_response(url: str, *, onebot_resolved: bool = False):
    current_url = url
    for _redirect_index in range(MAX_REDIRECTS + 1):
        parsed = urlparse(current_url)
        proxies = config.proxies
        trusted_url = (
            onebot_resolved
            and _is_trusted_onebot_image_url(current_url)
            and bool(proxies and proxies.get(parsed.scheme.lower()))
        )
        valid, _status, _message = (
            (True, "", "") if trusted_url else _validate_url(current_url)
        )
        if not valid:
            raise ImageInputError("图片地址无效，只支持公网 http/https 图片。")

        request_options = {
            "proxies": proxies,
            "timeout": config.request_timeout,
            "stream": True,
            "allow_redirects": False,
        }
        if trusted_url:
            response = requests.get(current_url, **request_options)
        else:
            response = try_proxied_get(
                current_url,
                hide_url_in_logs=True,
                **request_options,
            )
        if getattr(response, "status_code", 200) not in REDIRECT_STATUS_CODES:
            return response

        headers = getattr(response, "headers", {})
        location = headers.get("Location") if isinstance(headers, Mapping) else None
        _close_response(response)
        if not location:
            raise ImageInputError("图片重定向缺少目标地址，请重新发送。")
        current_url = urljoin(current_url, str(location))

    raise ImageInputError("图片重定向次数过多，请重新发送。")


def load_chat_images(
    image_urls: list[str] | tuple[str, ...],
    *,
    image_file_ids: list[str] | tuple[str, ...] = (),
    image_url_resolver: Callable[[str], str] | None = None,
) -> list[str]:
    file_ids = tuple(image_file_ids) or tuple("" for _url in image_urls)
    if len(file_ids) != len(image_urls):
        raise ImageInputError("图片消息格式无效，请重新发送。")

    loaded = []
    for original_url, file_id in zip(image_urls, file_ids):
        try:
            url = str(original_url or "").strip()
            onebot_resolved = False
            direct_valid = _validate_url(url)[0] if url else False
            if file_id and not direct_valid:
                if image_url_resolver is None:
                    raise ImageInputError("没有取得可读取的图片地址，请重新发送图片。")
                url = str(image_url_resolver(file_id) or "").strip()
                onebot_resolved = True

            response = _fetch_image_response(url, onebot_resolved=onebot_resolved)
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
                _close_response(response)
        except ImageInputError:
            raise
        except Exception as error:
            raise ImageInputError("图片读取失败，请重新发送。") from error
    return loaded
