import importlib
import unittest
from unittest import mock


class FakeResponse:
    def __init__(self, chunks, content_type="image/png", content_length=""):
        self._chunks = chunks
        self.headers = {"Content-Type": content_type}
        if content_length:
            self.headers["Content-Length"] = content_length
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=65536):
        yield from self._chunks

    def close(self):
        self.closed = True


class ImageInputServiceTests(unittest.TestCase):
    def service(self):
        try:
            return importlib.import_module("src.services.image_input_service")
        except ModuleNotFoundError as error:
            self.fail(f"image input service is missing: {error}")

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

    def test_loads_valid_image_as_data_url_and_closes_response(self):
        service = self.service()
        response = FakeResponse([b"png-bytes"])
        with mock.patch.object(service, "try_proxied_get", return_value=response):
            loaded = service.load_chat_images(["https://img.example/a.png"])
        self.assertEqual(["data:image/png;base64,cG5nLWJ5dGVz"], loaded)
        self.assertTrue(response.closed)

    def test_rejects_non_image_content(self):
        service = self.service()
        response = FakeResponse([b"html"], content_type="text/html")
        with (
            mock.patch.object(service, "try_proxied_get", return_value=response),
            self.assertRaisesRegex(service.ImageInputError, "不是支持的图片格式"),
        ):
            service.load_chat_images(["https://img.example/a"])

    def test_rejects_stream_larger_than_five_mib(self):
        service = self.service()
        response = FakeResponse([b"x" * (5 * 1024 * 1024), b"x"])
        with (
            mock.patch.object(service, "try_proxied_get", return_value=response),
            self.assertRaisesRegex(service.ImageInputError, "不能超过 5 MiB"),
        ):
            service.load_chat_images(["https://img.example/large.png"])

    def test_wraps_download_failure_as_user_facing_error(self):
        service = self.service()
        with (
            mock.patch.object(service, "try_proxied_get", side_effect=OSError("offline")),
            self.assertRaisesRegex(service.ImageInputError, "图片读取失败"),
        ):
            service.load_chat_images(["https://img.example/a.png"])


if __name__ == "__main__":
    unittest.main()
