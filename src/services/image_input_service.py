import base64
import html
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

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


class ImageInputError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedImageMessage:
    text: str
    image_urls: tuple[str, ...]


def _structured_image_urls(data: dict[str, Any]) -> list[str]:
    message = data.get("message")
    if not isinstance(message, list):
        return []
    urls = []
    for segment in message:
        if not isinstance(segment, dict) or segment.get("type") != "image":
            continue
        segment_data = segment.get("data")
        url = (
            str(segment_data.get("url") or "").strip()
            if isinstance(segment_data, dict)
            else ""
        )
        urls.append(url)
    return urls


def _cq_image_urls(raw_text: str) -> list[str]:
    matches = CQ_IMAGE_PATTERN.findall(str(raw_text or ""))
    urls = []
    for attributes in matches:
        match = re.search(r"(?:^|,)url=([^,]+)", attributes, flags=re.IGNORECASE)
        url = html.unescape(match.group(1).strip()) if match else ""
        urls.append(url)
    return urls


def parse_image_message(data: dict[str, Any], raw_text: str) -> ParsedImageMessage:
    structured_urls = _structured_image_urls(data)
    cq_urls = _cq_image_urls(raw_text)
    image_count = max(len(structured_urls), len(cq_urls))

    if image_count > MAX_CHAT_IMAGES:
        raise ImageInputError(f"每条消息最多发送 {MAX_CHAT_IMAGES} 张图片。")

    urls = []
    for index in range(image_count):
        structured_url = structured_urls[index] if index < len(structured_urls) else ""
        cq_url = cq_urls[index] if index < len(cq_urls) else ""
        url = structured_url or cq_url
        if not url:
            raise ImageInputError("没有取得可读取的图片地址，请重新发送图片。")
        urls.append(url)

    urls = list(dict.fromkeys(urls))
    text = CQ_IMAGE_PATTERN.sub("", str(raw_text or "")).strip()
    return ParsedImageMessage(text=text, image_urls=tuple(urls))


def _content_type(response) -> str:
    return str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()


def _close_response(response) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _fetch_image_response(url: str):
    current_url = url
    for _redirect_index in range(MAX_REDIRECTS + 1):
        valid, _status, _message = _validate_url(current_url)
        if not valid:
            raise ImageInputError("图片地址无效，只支持公网 http/https 图片。")

        response = try_proxied_get(
            current_url,
            proxies=config.proxies,
            timeout=config.request_timeout,
            stream=True,
            allow_redirects=False,
            hide_url_in_logs=True,
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


def load_chat_images(image_urls: list[str] | tuple[str, ...]) -> list[str]:
    loaded = []
    for url in image_urls:
        try:
            response = _fetch_image_response(url)
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
