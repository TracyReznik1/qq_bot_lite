from types import SimpleNamespace
import unittest

from src.search.simple.answering import AnswerResult, SearchAnswerer
from src.search.simple.models import (
    OutputKind,
    RequestSource,
    SearchFailure,
    SearchMode,
    SearchResult,
    SearchTrace,
)
from src.search.simple.rendering import (
    render_search_answer,
    render_search_failure,
)


class FakeLLM:
    def __init__(self, content="", error=None):
        self.content = content
        self.error = error
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


def results() -> tuple[SearchResult, ...]:
    return (
        SearchResult(
            result_id="R1",
            title="标题",
            url="https://example.com/1",
            excerpt="摘要",
            provider="tavily",
            score=0.9,
        ),
    )


def four_results() -> tuple[SearchResult, ...]:
    return tuple(
        SearchResult(
            result_id=f"R{i}",
            title=f"标题{i}",
            url=f"https://example.com/{i}",
            excerpt=f"摘要{i}",
            provider="tavily",
            score=0.9 - i * 0.1,
        )
        for i in range(1, 5)
    )


def trace() -> SearchTrace:
    return SearchTrace("r1", RequestSource.CHAT, SearchMode.LIGHT)


class SearchAnsweringTests(unittest.TestCase):
    def test_answer_keeps_base_history_and_image_content_but_excludes_urls(self):
        llm = FakeLLM("这是某品牌相机，目前约有三款相关型号。")
        base = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是什么"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAA"},
                    },
                ],
            }
        ]
        answer = SearchAnswerer(llm).answer(
            base_messages=base,
            question="这是什么",
            results=results(),
            timeout_seconds=20,
        )
        self.assertFalse(answer.degraded)
        self.assertIn("image_url", repr(llm.calls[0][0]))
        self.assertNotIn("https://example.com", repr(llm.calls[0][0]))

    def test_answer_exception_returns_deterministic_ranked_summary(self):
        answer = SearchAnswerer(FakeLLM(error=TimeoutError())).answer(
            base_messages=[],
            question="q",
            results=results(),
            timeout_seconds=20,
        )
        self.assertTrue(answer.degraded)
        self.assertEqual("根据搜索结果：\n1. 标题：摘要", answer.text)

    def test_empty_model_content_falls_back(self):
        answer = SearchAnswerer(FakeLLM(content="   ")).answer(
            base_messages=[],
            question="q",
            results=results(),
            timeout_seconds=20,
        )
        self.assertTrue(answer.degraded)
        self.assertEqual("根据搜索结果：\n1. 标题：摘要", answer.text)

    def test_model_returned_urls_are_stripped(self):
        llm = FakeLLM("参考 https://evil.com/leak 和 http://test.com 结论正确。")
        answer = SearchAnswerer(llm).answer(
            base_messages=[],
            question="q",
            results=results(),
            timeout_seconds=20,
        )
        self.assertNotIn("http://", answer.text)
        self.assertNotIn("https://", answer.text)
        self.assertIn("结论正确", answer.text)

    def test_answer_timeout_is_forwarded(self):
        llm = FakeLLM("回答")
        SearchAnswerer(llm).answer(
            base_messages=[],
            question="q",
            results=results(),
            timeout_seconds=15.5,
        )
        self.assertEqual(15.5, llm.calls[0][1]["timeout_seconds"])

    def test_light_render_hides_sources_and_standard_command_shows_three(self):
        light = render_search_answer(
            "回答",
            four_results(),
            warning=None,
            show_sources=False,
            qq_limit=1700,
            trace=trace(),
        )
        standard = render_search_answer(
            "回答",
            four_results(),
            warning=None,
            show_sources=True,
            qq_limit=1700,
            trace=trace(),
        )
        self.assertNotIn("https://", light.text)
        self.assertEqual(3, standard.text.count("https://"))
        self.assertEqual(3, len(standard.sources))

    def test_warning_once_and_output_never_exceeds_qq_limit(self):
        response = render_search_answer(
            "回答" * 1000,
            results(),
            warning="注意：测试",
            show_sources=True,
            qq_limit=200,
            trace=trace(),
        )
        self.assertLessEqual(len(response.text), 200)
        self.assertEqual(1, response.text.count("注意：测试"))

    def test_incomplete_or_reliability_disclaimer_is_suppressed(self):
        suppressed_warning = render_search_answer(
            "这是搜索回答。",
            results(),
            warning="信息可能不完整。",
            show_sources=False,
            trace=trace(),
        )
        self.assertEqual("这是搜索回答。", suppressed_warning.text)
        self.assertNotIn("信息可能不完整", suppressed_warning.text)

        trailing_disclaimer = render_search_answer(
            "这是搜索回答。\n\n注：搜索信息可能不完整，以上内容仅供参考，无法保证完全可靠。",
            results(),
            show_sources=False,
            trace=trace(),
        )
        self.assertEqual("这是搜索回答。", trailing_disclaimer.text)
        self.assertNotIn("仅供参考", trailing_disclaimer.text)
        self.assertNotIn("信息可能不完整", trailing_disclaimer.text)

    def test_failure_rendering_exact_messages_and_trace_kind(self):
        tr1 = trace()
        unavailable = render_search_failure(
            SearchFailure.PROVIDER_UNAVAILABLE, qq_limit=200, trace=tr1
        )
        self.assertEqual("在线搜索暂时不可用，请稍后再试。", unavailable.text)
        self.assertIs(OutputKind.SEARCH_FAILURE, tr1.output_kind)

        tr2 = trace()
        no_res = render_search_failure(
            SearchFailure.NO_RESULTS, qq_limit=200, trace=tr2
        )
        self.assertEqual("没有找到可用的在线搜索结果。", no_res.text)
        self.assertIs(OutputKind.SEARCH_FAILURE, tr2.output_kind)


if __name__ == "__main__":
    unittest.main()
