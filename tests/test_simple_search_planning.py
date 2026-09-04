import unittest
from types import SimpleNamespace

from src.search.simple.models import SearchMode, SearchRequest
from src.search.simple.planning import RoutePlanner


class FakeLLM:
    def __init__(self, content="", error=None):
        self.content, self.error, self.calls = content, error, []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


class RoutePlannerTests(unittest.TestCase):
    def test_force_search_overrides_model_skip_and_caps_three_queries(self):
        llm = FakeLLM('prefix ```json\n{"mode":"skip","queries":["a","b","c","d"]}\n``` suffix')
        plan = RoutePlanner(llm).plan(SearchRequest("question", force_search=True), timeout_seconds=8)
        self.assertEqual(SearchMode.STANDARD, plan.mode)
        self.assertEqual(("a", "b", "c"), tuple(q.text for q in plan.queries))

    def test_light_deduplicates_and_caps_one_query(self):
        llm = FakeLLM('{"mode":"light","queries":["  EDG  ","EDG"]}')
        plan = RoutePlanner(llm).plan(SearchRequest("EDG能去吗"), timeout_seconds=8)
        self.assertEqual(("EDG",), tuple(q.text for q in plan.queries))

    def test_invalid_output_falls_back_to_original_light_query(self):
        plan = RoutePlanner(FakeLLM("not json")).plan(SearchRequest("当前版本是什么"), timeout_seconds=8)
        self.assertEqual(SearchMode.LIGHT, plan.mode)
        self.assertTrue(plan.planner_degraded)
        self.assertEqual("当前版本是什么", plan.queries[0].text)

    def test_invalid_output_skips_obvious_social_chat(self):
        plan = RoutePlanner(FakeLLM(error=TimeoutError())).plan(SearchRequest("你好呀"), timeout_seconds=8)
        self.assertEqual(SearchMode.SKIP, plan.mode)

    def test_planner_error_keeps_current_fact_dependent_report_on_light(self):
        plan = RoutePlanner(FakeLLM(error=RuntimeError("planner unavailable"))).plan(
            SearchRequest("写一份关于 OpenSSL 最新漏洞的报告"), timeout_seconds=8
        )
        self.assertEqual(SearchMode.LIGHT, plan.mode)

    def test_planner_error_keeps_current_news_transform_on_light(self):
        plan = RoutePlanner(FakeLLM(error=RuntimeError("planner unavailable"))).plan(
            SearchRequest("总结一下今天的新闻"), timeout_seconds=8
        )
        self.assertEqual(SearchMode.LIGHT, plan.mode)

    def test_planner_error_skips_self_contained_creative_writing(self):
        plan = RoutePlanner(FakeLLM(error=RuntimeError("planner unavailable"))).plan(
            SearchRequest("写一首关于月光的诗"), timeout_seconds=8
        )
        self.assertEqual(SearchMode.SKIP, plan.mode)

    def test_planner_error_skips_transform_only_when_source_text_is_supplied(self):
        supplied = RoutePlanner(FakeLLM(error=RuntimeError("planner unavailable"))).plan(
            SearchRequest("把以下文字翻译成英文：你好"), timeout_seconds=8
        )
        supplied_current_text = RoutePlanner(
            FakeLLM(error=RuntimeError("planner unavailable"))
        ).plan(SearchRequest("翻译成英文：今天下雨"), timeout_seconds=8)
        unsupplied = RoutePlanner(FakeLLM(error=RuntimeError("planner unavailable"))).plan(
            SearchRequest("把 OpenSSL 最新漏洞报告翻译成英文"), timeout_seconds=8
        )
        self.assertEqual(SearchMode.SKIP, supplied.mode)
        self.assertEqual(SearchMode.SKIP, supplied_current_text.mode)
        self.assertEqual(SearchMode.LIGHT, unsupplied.mode)

    def test_planner_error_skips_only_valid_pure_arithmetic(self):
        for question, expected in (
            ("2 * (3 + 4)", SearchMode.SKIP),
            ("-2 + 3", SearchMode.SKIP),
            ("2026", SearchMode.LIGHT),
            ("-42", SearchMode.LIGHT),
            ("1++", SearchMode.LIGHT),
        ):
            with self.subTest(question=question):
                plan = RoutePlanner(FakeLLM(error=RuntimeError("planner unavailable"))).plan(
                    SearchRequest(question), timeout_seconds=8
                )
                self.assertEqual(expected, plan.mode)

    def test_parser_continues_after_invalid_balanced_object(self):
        llm = FakeLLM('{not valid json} then {"mode":"light","queries":["OpenSSL"]}')
        plan = RoutePlanner(llm).plan(SearchRequest("OpenSSL 最新漏洞"), timeout_seconds=8)
        self.assertEqual(SearchMode.LIGHT, plan.mode)
        self.assertFalse(plan.planner_degraded)
        self.assertEqual(("OpenSSL",), tuple(query.text for query in plan.queries))

    def test_planner_exception_debug_log_excludes_question_and_error_content(self):
        question = "private question sentinel"
        error_content = "private error sentinel"
        with self.assertLogs("qq-bot", level="DEBUG") as captured:
            RoutePlanner(FakeLLM(error=RuntimeError(error_content))).plan(
                SearchRequest(question), timeout_seconds=8
            )
        logged = " ".join(captured.output)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn(question, logged)
        self.assertNotIn(error_content, logged)

    def test_timeout_is_forwarded_to_llm(self):
        llm = FakeLLM('{"mode":"light","queries":["q"]}')
        RoutePlanner(llm).plan(SearchRequest("q"), timeout_seconds=7.5)
        self.assertEqual(7.5, llm.calls[0][1]["timeout_seconds"])


if __name__ == "__main__":
    unittest.main()
