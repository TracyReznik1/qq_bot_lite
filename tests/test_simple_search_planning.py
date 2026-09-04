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

    def test_timeout_is_forwarded_to_llm(self):
        llm = FakeLLM('{"mode":"light","queries":["q"]}')
        RoutePlanner(llm).plan(SearchRequest("q"), timeout_seconds=7.5)
        self.assertEqual(7.5, llm.calls[0][1]["timeout_seconds"])


if __name__ == "__main__":
    unittest.main()
