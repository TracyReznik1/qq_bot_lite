import importlib
import socket
import unittest
from types import SimpleNamespace
from unittest import mock

import requests
import src.util as util


class FakeResponse:
    def __init__(
        self,
        chunks,
        content_type="image/png",
        content_length="",
        status_code=200,
        location="",
    ):
        self._chunks = chunks
        self.headers = {"Content-Type": content_type}
        if content_length:
            self.headers["Content-Length"] = content_length
        if location:
            self.headers["Location"] = location
        self.status_code = status_code
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=65536):
        yield from self._chunks

    def close(self):
        self.closed = True


class ImageInputServiceTests(unittest.TestCase):
    def setUp(self):
        public_dns = mock.patch(
            "src.services.url_fetch_service.socket.getaddrinfo",
            return_value=self.dns_result("93.184.216.34"),
        )
        public_dns.start()
        self.addCleanup(public_dns.stop)

    def service(self):
        try:
            return importlib.import_module("src.services.image_input_service")
        except ModuleNotFoundError as error:
            self.fail(f"image input service is missing: {error}")

    @staticmethod
    def dns_result(ip_address, port=443):
        family = socket.AF_INET6 if ":" in ip_address else socket.AF_INET
        return [
            (
                family,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (ip_address, port),
            )
        ]

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

    def test_falls_back_to_cq_url_when_structured_image_has_no_url(self):
        service = self.service()
        event = {"message": [{"type": "image", "data": {"file": "a.png"}}]}
        try:
            parsed = service.parse_image_message(
                event, "[CQ:image,file=a.png,url=https://img.example/a.png]"
            )
        except service.ImageInputError as error:
            self.fail(f"expected CQ URL fallback, got: {error}")
        self.assertEqual("", parsed.text)
        self.assertEqual(("https://img.example/a.png",), parsed.image_urls)

    def test_fills_each_missing_structured_url_from_matching_cq_position(self):
        service = self.service()
        event = {
            "message": [
                {"type": "image", "data": {"url": "https://img.example/a.png"}},
                {"type": "image", "data": {"file": "b.png"}},
            ]
        }

        parsed = service.parse_image_message(
            event,
            "[CQ:image,file=a.png,url=https://img.example/a.png]"
            "[CQ:image,file=b.png,url=https://img.example/b.png]",
        )

        self.assertEqual(
            ("https://img.example/a.png", "https://img.example/b.png"),
            parsed.image_urls,
        )

    def test_five_structured_segments_cannot_be_hidden_by_one_cq_segment(self):
        service = self.service()
        event = {
            "message": [
                {"type": "image", "data": {"file": f"{index}.png"}}
                for index in range(5)
            ]
        }

        with self.assertRaisesRegex(service.ImageInputError, "最多发送 4 张图片"):
            service.parse_image_message(
                event,
                "[CQ:image,file=0.png,url=https://img.example/0.png]",
            )

    def test_unresolved_logical_image_slot_is_rejected_instead_of_silently_dropped(self):
        service = self.service()
        event = {
            "message": [
                {"type": "image", "data": {"url": "https://img.example/a.png"}},
                {"type": "image", "data": {"file": "b.png"}},
            ]
        }

        with self.assertRaisesRegex(service.ImageInputError, "没有取得可读取的图片地址"):
            service.parse_image_message(
                event,
                "[CQ:image,file=a.png,url=https://img.example/a.png]",
            )

    def test_cq_fallback_counts_duplicate_segments_before_deduplication(self):
        service = self.service()
        event = {"message": [{"type": "image", "data": {"file": "a.png"}}]}
        raw_text = "".join(
            "[CQ:image,file=a.png,url=https://img.example/a.png]" for _ in range(5)
        )
        with self.assertRaisesRegex(service.ImageInputError, "最多发送 4 张图片"):
            service.parse_image_message(event, raw_text)

    def test_cq_fallback_counts_segments_without_urls_before_filtering(self):
        service = self.service()
        event = {"message": [{"type": "image", "data": {"file": "a.png"}}]}
        raw_text = (
            "[CQ:image,file=a.png,url=https://img.example/a.png]"
            "[CQ:image,file=b.png]"
            "[CQ:image,file=c.png,url=https://img.example/c.png]"
            "[CQ:image,file=d.png]"
            "[CQ:image,file=e.png,url=https://img.example/e.png]"
        )
        with self.assertRaisesRegex(service.ImageInputError, "最多发送 4 张图片"):
            service.parse_image_message(event, raw_text)

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

    def test_rejects_five_structured_image_segments_with_duplicate_urls(self):
        service = self.service()
        event = {
            "message": [
                {"type": "image", "data": {"url": "https://img.example/a.png"}}
                for _ in range(5)
            ]
        }
        with self.assertRaisesRegex(service.ImageInputError, "最多发送 4 张图片"):
            service.parse_image_message(event, "images")

    def test_rejects_five_structured_image_segments_with_missing_urls(self):
        service = self.service()
        event = {
            "message": [
                {"type": "image", "data": {"url": "https://img.example/a.png"}},
                {"type": "image", "data": {}},
                {"type": "image", "data": {"url": "https://img.example/b.png"}},
                {"type": "image", "data": {}},
                {"type": "image", "data": {"url": "https://img.example/c.png"}},
            ]
        }
        with self.assertRaisesRegex(service.ImageInputError, "最多发送 4 张图片"):
            service.parse_image_message(event, "images")

    def test_deduplicates_image_urls_after_counting_segments(self):
        service = self.service()
        event = {
            "message": [
                {"type": "image", "data": {"url": "https://img.example/a.png"}},
                {"type": "image", "data": {"url": "https://img.example/a.png"}},
            ]
        }
        parsed = service.parse_image_message(event, "images")
        self.assertEqual(("https://img.example/a.png",), parsed.image_urls)

    def test_rejects_image_segment_without_usable_url(self):
        service = self.service()
        event = {"message": [{"type": "image", "data": {"file": "a.png"}}]}
        with self.assertRaisesRegex(service.ImageInputError, "没有取得可读取的图片地址"):
            service.parse_image_message(event, "[CQ:image,file=a.png]")

    def test_loads_valid_image_as_data_url_and_closes_response(self):
        service = self.service()
        response = FakeResponse([b"png-bytes"])
        with mock.patch.object(service, "try_proxied_get", return_value=response):
            loaded = service.load_chat_images(["https://img.example/a.png"])
        self.assertEqual(["data:image/png;base64,cG5nLWJ5dGVz"], loaded)
        self.assertTrue(response.closed)

    def test_rejects_non_public_ipv4_and_ipv6_targets_before_downloading(self):
        service = self.service()
        unsafe_targets = (
            ("127.0.0.1", "127.0.0.1"),
            ("10.0.0.8", "10.0.0.8"),
            ("100.64.0.1", "100.64.0.1"),
            ("169.254.1.1", "169.254.1.1"),
            ("0.0.0.0", "0.0.0.0"),
            ("224.0.0.1", "224.0.0.1"),
            ("192.0.2.1", "192.0.2.1"),
            ("[::1]", "::1"),
            ("[fc00::1]", "fc00::1"),
            ("[fe80::1]", "fe80::1"),
        )

        for url_host, resolved_ip in unsafe_targets:
            with self.subTest(url_host=url_host):
                with (
                    mock.patch(
                        "src.services.url_fetch_service.socket.getaddrinfo",
                        return_value=self.dns_result(resolved_ip),
                    ),
                    mock.patch.object(service, "try_proxied_get") as download,
                    self.assertRaisesRegex(service.ImageInputError, "图片地址无效"),
                ):
                    service.load_chat_images([f"http://{url_host}/a.png"])
                download.assert_not_called()

    def test_rejects_hostname_when_dns_resolves_to_private_address(self):
        service = self.service()
        with (
            mock.patch(
                "src.services.url_fetch_service.socket.getaddrinfo",
                return_value=self.dns_result("192.168.1.10"),
            ),
            mock.patch.object(service, "try_proxied_get") as download,
            self.assertRaisesRegex(service.ImageInputError, "图片地址无效"),
        ):
            service.load_chat_images(["https://images.example/a.png"])
        download.assert_not_called()

    def test_rejects_hostname_when_any_dns_result_is_not_public(self):
        service = self.service()
        mixed_results = self.dns_result("93.184.216.34") + self.dns_result(
            "192.168.1.10"
        )
        with (
            mock.patch(
                "src.services.url_fetch_service.socket.getaddrinfo",
                return_value=mixed_results,
            ),
            mock.patch.object(service, "try_proxied_get") as download,
            self.assertRaisesRegex(service.ImageInputError, "图片地址无效"),
        ):
            service.load_chat_images(["https://images.example/a.png"])
        download.assert_not_called()

    def test_rejects_public_redirect_to_private_target_and_closes_redirect(self):
        service = self.service()
        redirect = FakeResponse(
            [], status_code=302, location="http://127.0.0.1/private.png"
        )

        def resolve(hostname, port, **_kwargs):
            if hostname == "public.example":
                return self.dns_result("93.184.216.34", port)
            return self.dns_result("127.0.0.1", port)

        with (
            mock.patch(
                "src.services.url_fetch_service.socket.getaddrinfo",
                side_effect=resolve,
            ),
            mock.patch.object(
                service, "try_proxied_get", return_value=redirect
            ) as download,
            self.assertRaisesRegex(service.ImageInputError, "图片地址无效"),
        ):
            service.load_chat_images(["https://public.example/a.png"])

        self.assertEqual(1, download.call_count)
        self.assertFalse(download.call_args.kwargs["allow_redirects"])
        self.assertTrue(redirect.closed)

    def test_follows_public_redirect_explicitly_and_closes_every_response(self):
        service = self.service()
        redirect = FakeResponse(
            [], status_code=302, location="/final.png"
        )
        final = FakeResponse([b"png-bytes"])
        with (
            mock.patch(
                "src.services.url_fetch_service.socket.getaddrinfo",
                return_value=self.dns_result("93.184.216.34"),
            ),
            mock.patch.object(
                service, "try_proxied_get", side_effect=[redirect, final]
            ) as download,
        ):
            loaded = service.load_chat_images(["https://public.example/a.png"])

        self.assertEqual(["data:image/png;base64,cG5nLWJ5dGVz"], loaded)
        self.assertEqual(2, download.call_count)
        self.assertTrue(
            all(not call.kwargs["allow_redirects"] for call in download.call_args_list)
        )
        self.assertTrue(redirect.closed)
        self.assertTrue(final.closed)

    def test_rejects_too_many_redirects_and_closes_every_intermediate_response(self):
        service = self.service()
        redirects = [
            FakeResponse([], status_code=302, location=f"/{index + 1}.png")
            for index in range(service.MAX_REDIRECTS + 1)
        ]
        with (
            mock.patch.object(
                service, "try_proxied_get", side_effect=redirects
            ) as download,
            self.assertRaisesRegex(service.ImageInputError, "重定向次数过多"),
        ):
            service.load_chat_images(["https://public.example/0.png"])

        self.assertEqual(service.MAX_REDIRECTS + 1, download.call_count)
        self.assertTrue(all(response.closed for response in redirects))

    def test_rejects_non_http_image_url_without_downloading(self):
        service = self.service()
        with mock.patch.object(service, "try_proxied_get") as download:
            with self.assertRaisesRegex(service.ImageInputError, "图片地址无效"):
                service.load_chat_images(["file:///tmp/a.png"])
        download.assert_not_called()

    def test_rejects_non_image_content_and_closes_response(self):
        service = self.service()
        response = FakeResponse([b"html"], content_type="text/html")
        with (
            mock.patch.object(service, "try_proxied_get", return_value=response),
            self.assertRaisesRegex(service.ImageInputError, "不是支持的图片格式"),
        ):
            service.load_chat_images(["https://img.example/a"])
        self.assertTrue(response.closed)

    def test_rejects_content_length_larger_than_five_mib_and_closes_response(self):
        service = self.service()
        response = FakeResponse([b"x"], content_length=str(5 * 1024 * 1024 + 1))
        with mock.patch.object(service, "try_proxied_get", return_value=response):
            with self.assertRaisesRegex(service.ImageInputError, "不能超过 5 MiB"):
                service.load_chat_images(["https://img.example/large.png"])
        self.assertTrue(response.closed)

    def test_rejects_stream_larger_than_five_mib_and_closes_response(self):
        service = self.service()
        response = FakeResponse([b"x" * (5 * 1024 * 1024), b"x"])
        with (
            mock.patch.object(service, "try_proxied_get", return_value=response),
            self.assertRaisesRegex(service.ImageInputError, "不能超过 5 MiB"),
        ):
            service.load_chat_images(["https://img.example/large.png"])
        self.assertTrue(response.closed)

    def test_rejects_empty_image_response_and_closes_response(self):
        service = self.service()
        response = FakeResponse([])
        with mock.patch.object(service, "try_proxied_get", return_value=response):
            with self.assertRaisesRegex(service.ImageInputError, "图片内容为空"):
                service.load_chat_images(["https://img.example/empty.png"])
        self.assertTrue(response.closed)

    def test_wraps_download_failure_as_user_facing_error(self):
        service = self.service()
        with (
            mock.patch.object(service, "try_proxied_get", side_effect=OSError("offline")),
            self.assertRaisesRegex(service.ImageInputError, "图片读取失败"),
        ):
            service.load_chat_images(["https://img.example/a.png"])

    def test_proxy_fallback_hides_temporary_image_url_from_debug_logs(self):
        service = self.service()
        image_url = "https://img.example/private/a.png?token=temporary-secret"
        response = FakeResponse([b"png-bytes"])
        proxy_error = requests.exceptions.ConnectionError("proxy unavailable")

        with (
            mock.patch.object(
                service,
                "config",
                SimpleNamespace(
                    proxies={"https": "http://proxy.example:8080"},
                    request_timeout=3,
                ),
            ),
            mock.patch.object(
                util.requests, "get", side_effect=[proxy_error, response]
            ) as request_get,
            mock.patch.object(util.logger, "debug") as debug_log,
        ):
            loaded = service.load_chat_images([image_url])

        self.assertEqual(["data:image/png;base64,cG5nLWJ5dGVz"], loaded)
        self.assertEqual(2, request_get.call_count)
        self.assertNotIn("proxies", request_get.call_args_list[1].kwargs)
        logged = repr(debug_log.call_args_list)
        self.assertNotIn(image_url, logged)
        self.assertNotIn("temporary-secret", logged)
        self.assertNotIn("base64", logged)
        self.assertIn("[hidden URL]", logged)


if __name__ == "__main__":
    unittest.main()
