import unittest
from unittest.mock import MagicMock

from src.search.simple.models import SearchMode, SearchRouteDecision
from src.search.simple.router import SearchRouter, _parse_decision, _router_messages
from src.services.llm_types import ChatResponse


class SearchRouterTests(unittest.TestCase):
    def test_casual_banter_routes_to_skip(self):
        fake_llm = MagicMock()
        fake_llm.chat.return_value = ChatResponse(
            content='{"search_mode": "skip", "reason_code": "conversation", "retrieval_topics": []}'
        )
        router = SearchRouter(llm=fake_llm)
        decision = router.route("@ATRI 你是笨蛋吗")

        self.assertEqual(SearchMode.SKIP, decision.mode)
        self.assertEqual("conversation", decision.reason_code)
        self.assertEqual((), decision.retrieval_topics)

    def test_self_contained_task_routes_to_skip(self):
        fake_llm = MagicMock()
        fake_llm.chat.return_value = ChatResponse(
            content='{"search_mode": "skip", "reason_code": "self_contained", "retrieval_topics": []}'
        )
        router = SearchRouter(llm=fake_llm)
        decision = router.route("帮我把这句话翻译成日语：今天天气真好")

        self.assertEqual(SearchMode.SKIP, decision.mode)
        self.assertEqual("self_contained", decision.reason_code)
        self.assertEqual((), decision.retrieval_topics)

    def test_external_factual_query_routes_to_light_with_topics(self):
        fake_llm = MagicMock()
        fake_llm.chat.return_value = ChatResponse(
            content='{"search_mode": "light", "reason_code": "external_fact", "retrieval_topics": ["ANSYS Bonded MPC 定义与接触类型"]}'
        )
        router = SearchRouter(llm=fake_llm)
        decision = router.route("ANSYS 的 Bonded MPC 是什么")

        self.assertEqual(SearchMode.LIGHT, decision.mode)
        self.assertEqual("external_fact", decision.reason_code)
        self.assertEqual(("ANSYS Bonded MPC 定义与接触类型",), decision.retrieval_topics)

    def test_standard_hallucinated_by_model_is_normalized_to_light(self):
        fake_llm = MagicMock()
        fake_llm.chat.return_value = ChatResponse(
            content='{"search_mode": "standard", "reason_code": "broad_research", "retrieval_topics": ["Coding Agent 比较", "主流 Agent 特点"]}'
        )
        router = SearchRouter(llm=fake_llm)
        decision = router.route("比较目前主流的几个 Coding Agent")

        # Router in non-command flow strictly normalizes to LIGHT with at most 1 topic
        self.assertEqual(SearchMode.LIGHT, decision.mode)
        self.assertEqual("broad_research", decision.reason_code)
        self.assertEqual(("Coding Agent 比较",), decision.retrieval_topics)

    def test_malformed_json_degrades_to_light(self):
        fake_llm = MagicMock()
        fake_llm.chat.return_value = ChatResponse(content="Sorry, I cannot format this.")
        router = SearchRouter(llm=fake_llm)
        decision = router.route("EDG 什么时候夺冠的")

        self.assertEqual(SearchMode.LIGHT, decision.mode)
        self.assertEqual("degraded_fallback", decision.reason_code)
        self.assertEqual(("EDG 什么时候夺冠的",), decision.retrieval_topics)

    def test_exception_or_timeout_degrades_to_light(self):
        fake_llm = MagicMock()
        fake_llm.chat.side_effect = TimeoutError("API timeout")
        router = SearchRouter(llm=fake_llm)
        decision = router.route("smoggy 什么冠军")

        self.assertEqual(SearchMode.LIGHT, decision.mode)
        self.assertEqual("degraded_fallback", decision.reason_code)
        self.assertEqual(("smoggy 什么冠军",), decision.retrieval_topics)

    def test_multimodal_images_pass_to_router_messages(self):
        image_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        messages = _router_messages("这是什么花", (image_url,))
        user_msg = messages[-1]["content"]
        self.assertIsInstance(user_msg, list)
        self.assertEqual(2, len(user_msg))
        self.assertEqual("text", user_msg[0]["type"])
        self.assertEqual("image_url", user_msg[1]["type"])


if __name__ == "__main__":
    unittest.main()
