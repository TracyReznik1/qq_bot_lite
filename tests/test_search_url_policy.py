import socket
import unittest
from unittest import mock

from src.search.url_policy import (
    UrlDecision,
    canonicalize_public_http_url,
    evaluate_public_http_url,
)


class SearchUrlPolicyTests(unittest.TestCase):
    def test_canonicalize_strips_fragments_and_default_ports(self):
        cases = (
            ("http://example.com:80/path#frag", "http://example.com/path"),
            ("https://example.com:443/path?q=1#frag", "https://example.com/path?q=1"),
            ("HTTP://EXAMPLE.COM/PATH", "http://example.com/PATH"),
            ("https://example.com:8080/test", "https://example.com:8080/test"),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(expected, canonicalize_public_http_url(raw))

    def test_canonicalize_rejects_non_http_and_invalid_urls(self):
        invalid = (
            "",
            None,
            "ftp://example.com/file",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "http://:80",
            "http://example.com:invalid_port",
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                self.assertIsNone(canonicalize_public_http_url(raw))

    def test_evaluate_public_http_url_allowed_on_public_ip(self):
        with mock.patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]):
            decision = evaluate_public_http_url("http://example.com/news")
            self.assertTrue(decision.allowed)
            self.assertEqual("allowed", decision.status)
            self.assertEqual("http://example.com/news", decision.canonical_url)

    def test_evaluate_rejects_localhost_and_private_ips(self):
        reject_cases = (
            "http://localhost/test",
            "http://sub.localhost/test",
            "http://router.local/admin",
            "http://127.0.0.1:8080/api",
            "http://10.0.0.1/",
            "http://192.168.1.100/",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/",
        )
        for raw in reject_cases:
            with self.subTest(raw=raw):
                decision = evaluate_public_http_url(raw)
                self.assertFalse(decision.allowed)
                self.assertEqual("unsafe_url", decision.status)

    def test_evaluate_rejects_dns_resolving_to_private_ip(self):
        with mock.patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]):
            decision = evaluate_public_http_url("http://spoofed.example.com/")
            self.assertFalse(decision.allowed)
            self.assertEqual("unsafe_url", decision.status)

    def test_evaluate_dns_error(self):
        with mock.patch("socket.getaddrinfo", side_effect=socket.gaierror("lookup failed")):
            decision = evaluate_public_http_url("http://nonexistent-domain-1234567.com/")
            self.assertFalse(decision.allowed)
            self.assertEqual("dns_error", decision.status)

    def test_evaluate_empty_and_unsupported_scheme(self):
        self.assertEqual("empty_url", evaluate_public_http_url("").status)
        self.assertEqual("unsupported_scheme", evaluate_public_http_url("ftp://example.com").status)


if __name__ == "__main__":
    unittest.main()
