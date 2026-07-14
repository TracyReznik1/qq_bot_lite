import unittest
from unittest import mock

import requests

import src.util as util


class DummyResponse:
    def __init__(self):
        self.closed = False

    def raise_for_status(self):
        return None

    def close(self):
        self.closed = True


class ProxyHttpErrorResponse(DummyResponse):
    def __init__(self, status_code):
        super().__init__()
        self.status_code = status_code

    def raise_for_status(self):
        raise requests.exceptions.HTTPError(
            f"proxy returned {self.status_code}", response=self
        )


class ProxiedGetCompatibilityTests(unittest.TestCase):
    def test_default_proxy_fallback_keeps_existing_url_logging_and_call_shape(self):
        url = "https://search.example/results?q=compatibility"
        response = DummyResponse()
        proxy_error = requests.exceptions.Timeout("proxy timed out")

        with (
            mock.patch.object(
                util.requests, "get", side_effect=[proxy_error, response]
            ) as request_get,
            mock.patch.object(util.logger, "debug") as debug_log,
        ):
            result = util.try_proxied_get(
                url,
                proxies={"https": "http://proxy.example:8080"},
                timeout=3,
                stream=True,
            )

        self.assertIs(response, result)
        self.assertEqual(2, request_get.call_count)
        self.assertEqual(url, request_get.call_args_list[0].args[0])
        self.assertIn("proxies", request_get.call_args_list[0].kwargs)
        self.assertEqual(url, request_get.call_args_list[1].args[0])
        self.assertNotIn("proxies", request_get.call_args_list[1].kwargs)
        self.assertTrue(request_get.call_args_list[1].kwargs["stream"])
        self.assertFalse(response.closed)
        debug_log.assert_called_once_with(
            "Proxy request to %s failed, retrying without proxy", url
        )

    def test_proxy_http_error_closes_stream_without_direct_fallback(self):
        image_url = "https://img.example/private.png?token=temporary-secret"

        for status_code in (403, 503):
            with self.subTest(status_code=status_code):
                response = ProxyHttpErrorResponse(status_code)
                with (
                    mock.patch.object(
                        util.requests, "get", return_value=response
                    ) as request_get,
                    self.assertRaises(requests.exceptions.HTTPError) as raised,
                ):
                    util.try_proxied_get(
                        image_url,
                        proxies={"https": "http://proxy.example:8080"},
                        timeout=3,
                        stream=True,
                        hide_url_in_logs=True,
                    )

                self.assertTrue(response.closed)
                self.assertIs(response, raised.exception.response)
                request_get.assert_called_once_with(
                    image_url,
                    proxies={"https": "http://proxy.example:8080"},
                    timeout=3,
                    stream=True,
                )


if __name__ == "__main__":
    unittest.main()
