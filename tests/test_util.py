import unittest
from unittest import mock

import requests

import src.util as util


class DummyResponse:
    def raise_for_status(self):
        return None


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
        debug_log.assert_called_once_with(
            "Proxy request to %s failed, retrying without proxy", url
        )


if __name__ == "__main__":
    unittest.main()
