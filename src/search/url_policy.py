"""Centralized public HTTP(S) URL safety and canonicalization policy."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlunparse


@dataclass(frozen=True)
class UrlDecision:
    """Outcome of evaluating a public HTTP(S) URL for security and correctness."""

    allowed: bool
    status: str
    canonical_url: str | None
    message: str


def canonicalize_public_http_url(url: str) -> str | None:
    """Return the canonical public HTTP(S) URL, or None if invalid/unsupported."""
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return None

    hostname = parsed.hostname
    if not hostname:
        return None
    hostname = hostname.strip().lower()

    if ":" not in hostname:
        try:
            # Normalize Unicode/IDNA domains
            hostname = hostname.encode("idna").decode("ascii")
        except Exception:
            pass

    try:
        port = parsed.port
    except ValueError:
        return None

    host_for_netloc = f"[{hostname}]" if ":" in hostname else hostname
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        netloc = host_for_netloc
    elif port is not None:
        netloc = f"{host_for_netloc}:{port}"
    else:
        netloc = host_for_netloc

    # Strip fragments completely from canonical form
    return urlunparse((scheme, netloc, parsed.path, parsed.params, parsed.query, ""))


def _is_unsafe_ip(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return True
    return (
        not ip.is_global
        or ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def evaluate_public_http_url(url: str, *, timeout: float | None = None) -> UrlDecision:
    """Evaluate whether a URL is a valid, publicly reachable HTTP(S) endpoint."""
    del timeout
    if not isinstance(url, str) or not url.strip():
        return UrlDecision(False, "empty_url", None, "URL 不能为空。")

    raw_url = url.strip()
    try:
        raw_parsed = urlparse(raw_url)
        raw_scheme = raw_parsed.scheme.lower()
        if raw_scheme not in {"http", "https"}:
            return UrlDecision(False, "unsupported_scheme", None, "只支持 http/https 网页。")
        raw_hostname = raw_parsed.hostname
        if not raw_hostname:
            return UrlDecision(False, "invalid_url", None, "URL 格式无效。")
    except Exception:
        return UrlDecision(False, "invalid_url", None, "URL 格式无效。")

    canonical = canonicalize_public_http_url(raw_url)
    if canonical is None:
        return UrlDecision(False, "invalid_url", None, "URL 格式无效。")

    parsed = urlparse(canonical)
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return UrlDecision(False, "invalid_url", None, "URL 格式无效。")

    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        return UrlDecision(False, "unsafe_url", None, "出于安全原因，不能读取本机或局域网地址。")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return UrlDecision(False, "invalid_url", None, "URL 格式无效。")

    try:
        ip = ipaddress.ip_address(hostname)
        if _is_unsafe_ip(str(ip)):
            return UrlDecision(False, "unsafe_url", None, "出于安全原因，不能读取本机或局域网地址。")
        return UrlDecision(True, "allowed", canonical, "")
    except ValueError:
        pass

    try:
        addr_infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError:
        return UrlDecision(False, "dns_error", None, "域名解析失败。")

    if not addr_infos:
        return UrlDecision(False, "dns_error", None, "域名解析失败。")

    for info in addr_infos:
        ip_str = info[4][0]
        if _is_unsafe_ip(ip_str):
            return UrlDecision(False, "unsafe_url", None, "出于安全原因，不能读取本机或局域网地址。")

    return UrlDecision(True, "allowed", canonical, "")
