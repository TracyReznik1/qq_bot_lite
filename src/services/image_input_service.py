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


def _structured_image_urls(data: dict[str, Any]) -> tuple[int, list[str]]:
    message = data.get("message")
    if not isinstance(message, list):
        return 0, []
    image_count = 0
    urls = []
    for segment in message:
        if not isinstance(segment, dict) or segment.get("type") != "image":
            continue
        image_count += 1
        segment_data = segment.get("data")
        if isinstance(segment_data, dict):
            url = str(segment_data.get("url") or "").strip()
            if url:
                urls.append(url)
    return image_count, urls


def _cq_image_urls(raw_text: str) -> tuple[int, list[str]]:
    matches = CQ_IMAGE_PATTERN.findall(str(raw_text or ""))
    urls = []
    for attributes in matches:
        match = re.search(r"(?:^|,)url=([^,]+)", attributes, flags=re.IGNORECASE)
        if match:
            urls.append(html.unescape(match.group(1).strip()))
    return len(matches), urls


def parse_image_message(data: dict[str, Any], raw_text: str) -> ParsedImageMessage:
    structured_count, structured_urls = _structured_image_urls(data)
    cq_count, cq_urls = _cq_image_urls(raw_text)
    image_count, urls = (
        (structured_count, structured_urls)
        if structured_urls
        else (cq_count, cq_urls)
    )

    if image_count > MAX_CHAT_IMAGES:
        raise ImageInputError(f"每条消息最多发送 {MAX_CHAT_IMAGES} 张图片。")
    if image_count and not urls:
        raise ImageInputError("没有取得可读取的图片地址，请重新发送图片。")

    urls = list(dict.fromkeys(urls))
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
