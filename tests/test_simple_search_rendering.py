import unittest

from src.search.simple.rendering import split_qq_reply


class QQReplySplittingTests(unittest.TestCase):
    def test_empty_input_returns_no_parts(self):
        self.assertEqual([], split_qq_reply("", 200))

    def test_prefers_newline_and_preserves_every_character(self):
        text = "第一段\n第二段很长\n第三段"
        parts = split_qq_reply(text, 8)
        self.assertTrue(all(1 <= len(part) <= 8 for part in parts))
        self.assertEqual(text, "".join(parts))

    def test_hard_splits_a_single_long_line(self):
        self.assertEqual(["abcd", "efgh", "ij"], split_qq_reply("abcdefghij", 4))

    def test_rejects_nonpositive_limit(self):
        with self.assertRaisesRegex(ValueError, "max_chars"):
            split_qq_reply("text", 0)


if __name__ == "__main__":
    unittest.main()
