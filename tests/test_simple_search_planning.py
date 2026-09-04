from pathlib import Path
from types import SimpleNamespace
import unittest

from src.search.simple.models import SearchMode
from src.search.simple.planning import IMAGE_ONLY_FALLBACK_QUERY, QueryPlanner


class FakeLLM:
    def __init__(self, content="", error=None):
        self.content, self.error, self.calls = content, error, []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


class QueryPlannerTests(unittest.TestCase):
    def test_light_ignores_returned_mode_and_keeps_exactly_one_query(self):
        llm = FakeLLM('{"mode":"standard","queries":["  EDG   上海 ","second"]}')
        plan = QueryPlanner(llm).plan(
            mode=SearchMode.LIGHT, text="EDG能去吗", images=(), timeout_seconds=7.5
        )
        self.assertIs(SearchMode.LIGHT, plan.mode)
        self.assertEqual(("EDG 上海",), tuple(item.text for item in plan.queries))
        self.assertEqual(7.5, llm.calls[0][1]["timeout_seconds"])

    def test_standard_finds_first_valid_balanced_object_deduplicates_and_caps_three(self):
        llm = FakeLLM('{bad} prefix {"queries":["a"," a ","b","c","d"],"extra":1}')
        plan = QueryPlanner(llm).plan(
            mode=SearchMode.STANDARD, text="question", images=(), timeout_seconds=8
        )
        self.assertIs(SearchMode.STANDARD, plan.mode)
        self.assertEqual(("a", "b", "c"), tuple(item.text for item in plan.queries))

    def test_image_only_planning_is_multimodal(self):
        llm = FakeLLM('{"queries":["图中的相机型号"]}')
        plan = QueryPlanner(llm).plan(
            mode=SearchMode.LIGHT,
            text="",
            images=("data:image/png;base64,AAA",),
            timeout_seconds=8,
        )
        user_content = llm.calls[0][0][-1]["content"]
        self.assertEqual("image_url", user_content[-1]["type"])
        self.assertEqual("图中的相机型号", plan.queries[0].text)

    def test_malformed_text_falls_back_without_changing_mode(self):
        plan = QueryPlanner(FakeLLM("not json")).plan(
            mode=SearchMode.STANDARD, text="  原始   问题 ", images=(), timeout_seconds=8
        )
        self.assertIs(SearchMode.STANDARD, plan.mode)
        self.assertTrue(plan.planner_degraded)
        self.assertEqual("原始 问题", plan.queries[0].text)

    def test_image_only_exception_uses_fixed_fallback_without_changing_mode(self):
        plan = QueryPlanner(FakeLLM(error=TimeoutError())).plan(
            mode=SearchMode.LIGHT,
            text="",
            images=("data:image/png;base64,AAA",),
            timeout_seconds=8,
        )
        self.assertIs(SearchMode.LIGHT, plan.mode)
        self.assertTrue(plan.planner_degraded)
        self.assertEqual(IMAGE_ONLY_FALLBACK_QUERY, plan.queries[0].text)

    def test_skip_mode_raises(self):
        with self.assertRaises(ValueError):
            QueryPlanner(FakeLLM()).plan(
                mode=SearchMode.SKIP, text="test", images=(), timeout_seconds=8
            )

    def test_source_contains_no_route_planner_or_heuristics(self):
        source = Path("src/search/simple/planning.py").read_text(encoding="utf-8")
        self.assertNotIn("Route" + "Planner", source)
        self.assertNotIn("obviously" + "_no_search", source)
        self.assertNotIn("is_valid" + "_arithmetic", source)


if __name__ == "__main__":
    unittest.main()
