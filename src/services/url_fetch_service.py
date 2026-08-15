import ipaddress
import logging
import re
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from src.config import config
from src.util import try_proxied_get


logger = logging.getLogger("qq-bot")

MAX_URL_BYTES = 512 * 1024
MAX_URL_TEXT_CHARS = 6000
MAX_REDIRECTS = 3
URL_FETCH_USER_AGENT = "qqbot-url-fetch/1.0"
URL_PATTERN = re.compile(r"https?://[^\s<>'\"\]]+", re.IGNORECASE)
TRAILING_URL_PUNCTUATION = ".,;:!?，。；：！？)]}）】》"
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
ALLOWED_CONTENT_TYPES = {
    "text/html",
    "text/plain",
    "application/xhtml+xml",
    "application/json",
    "application/ld+json",
    "application/pdf",
    "application/x-pdf",
}

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None


@dataclass(frozen=True)
class UrlFetchResult:
    ok: bool
    status: str
    text: str


@dataclass(frozen=True)
class UrlDocumentResult:
    ok: bool
    status: str
    requested_url: str
    final_url: str
    title: str
    content_type: str
    text: str


class _ReadableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._in_title = False
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, _attrs) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._ignored_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in {"p", "br", "div", "section", "article", "header", "footer", "li", "tr", "h1", "h2", "h3"}:
            self.text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3"}:
            self.text_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        self.text_parts.append(data)


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _truncate_text(text: str, limit: int) -> str:
    text = _collapse_spaces(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip("，,。；;：:") + "..."


def extract_first_url(text: str) -> str:
    match = URL_PATTERN.search(str(text or ""))
    if not match:
        return ""
    return match.group(0).rstrip(TRAILING_URL_PUNCTUATION)


def _format_failure(status: str, url: str, message: str) -> str:
    return (
        f"获取状态：{status}\n"
        "内容类型：url\n"
        f"URL：{url or '无'}\n"
        f"说明：{message}"
    )


def _format_success(url: str, title: str, text: str, content_type: str) -> str:
    title = title or "无标题"
    excerpt = _truncate_text(text, MAX_URL_TEXT_CHARS)
    return (
        "获取状态：success\n"
        "内容类型：url\n"
        f"URL：{url}\n"
        f"标题：{title}\n"
        f"响应类型：{content_type or '未知'}\n"
        f"正文字符数：{len(_collapse_spaces(text))}\n"
        "正文摘录：\n"
        f"{excerpt}"
    )


from src.search.url_policy import evaluate_public_http_url


def _validate_url(url: str) -> tuple[bool, str, str]:
    decision = evaluate_public_http_url(url)
    return decision.allowed, decision.status, decision.message


def _content_type(headers: Mapping) -> str:
    content_type = str(headers.get("Content-Type") or headers.get("content-type") or "")
    return content_type.split(";", 1)[0].strip().lower()


def _content_length(headers: Mapping) -> int:
    raw_value = str(headers.get("Content-Length") or headers.get("content-length") or "").strip()
    if not raw_value:
        return 0
    try:
        return int(raw_value)
    except ValueError:
        return 0


def _extract_readable_text(raw_text: str, content_type: str) -> tuple[str, str]:
    raw_text = str(raw_text or "")
    if content_type in {"text/html", "application/xhtml+xml"} or "<html" in raw_text[:1000].lower():
        parser = _ReadableHtmlParser()
        parser.feed(raw_text)
        title = _collapse_spaces(" ".join(parser.title_parts))
        body = _collapse_spaces(" ".join(parser.text_parts))
        return title, body
    body = _collapse_spaces(raw_text)
    title = ""
    return title, body


def _extract_pdf_text(content: bytes) -> tuple[str, str]:
    if PdfReader is None:
        return "", ""
    try:
        reader = PdfReader(__import__("io").BytesIO(content))
    except Exception:
        return "", ""
    parts: list[str] = []
    try:
        for page in reader.pages:
            page_text = str(getattr(page, "extract_text", lambda: "")() or "")
            if page_text:
                parts.append(page_text)
    except Exception:
        pass
    body = _collapse_spaces(" ".join(parts))
    return "", body[:MAX_URL_TEXT_CHARS]


def _fetch_response(url: str, *, timeout: float | None = None):
    current_url = url
    timeout = timeout if timeout is not None else config.request_timeout
    for _redirect_index in range(MAX_REDIRECTS + 1):
        valid, status, message = _validate_url(current_url)
        if not valid:
            return None, current_url, status, message

        try:
            response = try_proxied_get(
                current_url,
                proxies=config.proxies,
                timeout=timeout,
                headers={
                    "User-Agent": URL_FETCH_USER_AGENT,
                    "Accept": "text/html,text/plain,application/json;q=0.8,*/*;q=0.2",
                },
                allow_redirects=False,
                stream=True,
            )
        except Exception:
            logger.debug("URL fetch request failed: %s", current_url)
            return None, current_url, "request_error", "网页读取失败，可能是网络或站点暂时不可用。"

        if getattr(response, "status_code", 200) in REDIRECT_STATUS_CODES:
            response_headers = getattr(response, "headers", {})
            location = response_headers.get("Location") if isinstance(response_headers, Mapping) else None
            _safe_close(response)
            if not location:
                return None, current_url, "redirect_error", "网页重定向缺少目标地址。"
            current_url = urljoin(current_url, str(location))
            continue

        try:
            response.raise_for_status()
        except Exception:
            _safe_close(response)
            return None, current_url, "http_error", "网页返回了错误状态码。"

        return response, str(getattr(response, "url", "") or current_url), "", ""

    return None, current_url, "too_many_redirects", "网页重定向次数过多。"


def fetch_document(
    url: str,
    *,
    timeout_seconds: float | None = None,
) -> UrlDocumentResult:
    """Fetch one URL into a structured document result with full safety guards."""
    timeout = timeout_seconds if timeout_seconds is not None else config.request_timeout
    url = (str(url or "")).strip()
    if not url:
        return UrlDocumentResult(
            ok=False,
            status="empty_url",
            requested_url=url,
            final_url="",
            title="",
            content_type="",
            text="",
        )

    response, final_url, status, message = _fetch_response(url, timeout=timeout)
    if response is None:
        return UrlDocumentResult(
            ok=False,
            status=status,
            requested_url=url,
            final_url=final_url or url,
            title="",
            content_type="",
            text=message,
        )

    response_headers = getattr(response, "headers", {})
    headers = response_headers if isinstance(response_headers, Mapping) else {}
    content_type = _content_type(headers)
    if content_type and content_type not in ALLOWED_CONTENT_TYPES and not content_type.startswith("text/"):
        _safe_close(response)
        return UrlDocumentResult(
            ok=False,
            status="unsupported_content_type",
            requested_url=url,
            final_url=final_url,
            title="",
            content_type=content_type,
            text="这个链接不是可直接阅读的文本网页。",
        )
    if _content_length(headers) > MAX_URL_BYTES:
        _safe_close(response)
        return UrlDocumentResult(
            ok=False,
            status="too_large",
            requested_url=url,
            final_url=final_url,
            title="",
            content_type=content_type,
            text="网页内容太大，已停止读取。",
        )

    readable, read_status, raw_text, raw_bytes = _read_limited_document(response)
    _safe_close(response)
    if not readable:
        return UrlDocumentResult(
            ok=False,
            status=read_status,
            requested_url=url,
            final_url=final_url,
            title="",
            content_type=content_type,
            text="网页内容太大，已停止读取。",
        )

    if content_type in {"application/pdf", "application/x-pdf"}:
        title, body = _extract_pdf_text(raw_bytes)
        if not body:
            return UrlDocumentResult(
                ok=False,
                status="no_text",
                requested_url=url,
                final_url=final_url,
                title=title,
                content_type=content_type,
                text="没有从 PDF 中提取到可阅读的正文。",
            )
        return UrlDocumentResult(
            ok=True,
            status="success",
            requested_url=url,
            final_url=final_url,
            title=title,
            content_type=content_type,
            text=body,
        )

    title, body = _extract_readable_text(raw_text, content_type)
    if not body:
        return UrlDocumentResult(
            ok=False,
            status="no_text",
            requested_url=url,
            final_url=final_url,
            title=title,
            content_type=content_type,
            text="没有提取到可阅读的正文。",
        )

    return UrlDocumentResult(
        ok=True,
        status="success",
        requested_url=url,
        final_url=final_url,
        title=title,
        content_type=content_type,
        text=body,
    )


def _safe_close(response) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.debug("URL fetch response close failed", exc_info=True)


def _read_limited_document(response) -> tuple[bool, str, str, bytes]:
    """Read a bounded body. For streamed responses, consume ``iter_content``
    into a bounded buffer and never touch ``response.content`` (which would
    drain the full body into memory first)."""
    iter_content = getattr(response, "iter_content", None)
    if callable(iter_content):
        chunks = bytearray()
        for chunk in iter_content(chunk_size=8192):
            if not chunk:
                continue
            if isinstance(chunk, str):
                chunk = chunk.encode(getattr(response, "encoding", None) or "utf-8", errors="replace")
            chunks.extend(chunk)
            if len(chunks) > MAX_URL_BYTES:
                return False, "too_large", "", b""
        raw_bytes = bytes(chunks)
    else:
        content = getattr(response, "content", None)
        if isinstance(content, str):
            content = content.encode(getattr(response, "encoding", None) or "utf-8", errors="replace")
        if content is not None and len(content) > MAX_URL_BYTES:
            return False, "too_large", "", b""
        raw_bytes = content if content is not None else b""

    if not raw_bytes:
        response_text = getattr(response, "text", None)
        if response_text:
            raw_bytes = response_text.encode(
                getattr(response, "encoding", None) or "utf-8",
                errors="replace",
            )

    if not raw_bytes:
        return True, "", "", b""

    encoding = getattr(response, "encoding", None) or getattr(response, "apparent_encoding", None) or "utf-8"
    raw_text = raw_bytes.decode(encoding, errors="replace")
    return True, "", raw_text, raw_bytes


def fetch_url(text: str) -> UrlFetchResult:
    """Compatibility wrapper: extract the first URL and format a user-readable result."""
    url = extract_first_url(text)
    if not url:
        return UrlFetchResult(
            ok=False,
            status="empty_url",
            text=_format_failure("empty_url", "", "没有找到可读取的 URL。"),
        )

    document = fetch_document(url)
    if not document.ok:
        return UrlFetchResult(
            ok=False,
            status=document.status,
            text=_format_failure(document.status, document.final_url or url, document.text),
        )
    return UrlFetchResult(
        ok=True,
        status="success",
        text=_format_success(document.final_url, document.title, document.text, document.content_type),
    )
