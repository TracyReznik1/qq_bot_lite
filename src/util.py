"""Shared HTTP helpers."""

import logging

import requests

logger = logging.getLogger("qq-bot")


def try_proxied_get(url: str, proxies: dict[str, str] | None = None, timeout: float = 18, **kwargs):
    """Try a GET request through proxy first; fall back to direct on connection failure.

    Only connection-level errors (ConnectionError, Timeout) trigger a fallback.
    HTTP-level errors (4xx, 5xx) are raised normally — a working proxy that returns
    an error is not a reason to bypass it.
    """
    if proxies:
        try:
            resp = requests.get(url, proxies=proxies, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            logger.debug("Proxy request to %s failed, retrying without proxy", url)
    return requests.get(url, timeout=timeout, **kwargs)


def try_proxied_post(url: str, proxies: dict[str, str] | None = None, timeout: float = 18, **kwargs):
    """Try a POST request through proxy first; fall back to direct on connection failure.

    Only connection-level errors (ConnectionError, Timeout) trigger a fallback.
    HTTP-level errors (4xx, 5xx) are raised normally — a working proxy that returns
    an error is not a reason to bypass it.
    """
    if proxies:
        try:
            resp = requests.post(url, proxies=proxies, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            logger.debug("Proxy POST to %s failed, retrying without proxy", url)
    return requests.post(url, timeout=timeout, **kwargs)
