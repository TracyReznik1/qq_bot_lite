import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.chat import chat_service
from src.chat.chat_service import generate_reply
from src.config import config
from src.search.simple.answering import AnswerResult
from src.search.simple.models import (
    OutputKind,
    RequestSource,
    SearchFailure,
    SearchMode,
    SearchOutcome,
    SearchPlan,
    SearchQuery,
    SearchRequest,
    SearchResult,
    SearchTrace,
)


def success_outcome(mode: SearchMode = SearchMode.LIGHT) -> SearchOutcome:
    query = SearchQuery("q1", "最新消息")
    plan = SearchPlan(mode, (query,))
    res = (
        SearchResult(
            result_id="R1",
            title="最新消息标题",
            url="https://example.com/news",
            excerpt="这是最新消息的详细摘要",
            provider="tavily",
            score=0.9,
        ),
    )
    trace = SearchTrace("req1", RequestSource.CHAT, mode)
    return SearchOutcome(plan=plan, results=res, trace=trace)


class FakeEngine:
    def __init__(self, outcome: SearchOutcome):
        self.outcome = outcome
        self.requests: list[SearchRequest] = []

    def run(self, request: SearchRequest) -> SearchOutcome:
        self.requests.append(request)
        return self.outcome


class RaisingEngine:
    def run(self, request: SearchRequest) -> SearchOutcome:
        raise RuntimeError("search crashed")


class FakeLLM:
    def __init__(self, content="plain multimodal reply"):
        self.content = content
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content=self.content)


def run_reply(engine, *, mode: SearchMode, text: str, images=None, fake_llm=None):
    if fake_llm is None:
        fake_llm = FakeLLM("模型生成的搜索回答")
    with patch.object(chat_service, "get_simple_search_pipeline_for_chat", return_value=engine), \
         patch.object(chat_service, "llm", fake_llm):
        return generate_reply("private:1", text, images, mode=mode)


class SimpleSearchChatFlowTests(unittest.TestCase):
    def setUp(self):
        chat_service.reset_history("private:1")

    def tearDown(self):
        chat_service.reset_history("private:1")

    def test_generate_reply_requires_explicit_mode(self):
        with self.assertRaises(TypeError):
            generate_reply("private:1", "你好")

    def test_legacy_force_kwarg_is_rejected(self):
        legacy_kwargs = {"force_" + "search": True}
        with self.assertRaises(TypeError):
            generate_reply("private:1", "你好", mode=SearchMode.LIGHT, **legacy_kwargs)

    def test_light_constructs_light_request_and_hides_sources(self):
        engine = FakeEngine(success_outcome(SearchMode.LIGHT))
        reply = run_reply(engine, mode=SearchMode.LIGHT, text="最新消息")
        self.assertIs(SearchMode.LIGHT, engine.requests[0].mode)
        self.assertNotIn("https://", reply)

    def test_standard_constructs_standard_request_and_shows_sources(self):
        engine = FakeEngine(success_outcome(SearchMode.STANDARD))
        reply = run_reply(engine, mode=SearchMode.STANDARD, text="最新消息")
        self.assertIs(SearchMode.STANDARD, engine.requests[0].mode)
        self.assertIn("https://example.com", reply)

    def test_skip_never_gets_pipeline_or_search_answerer(self):
        fake_llm = FakeLLM("plain multimodal reply")
        with patch.object(chat_service, "get_simple_search_pipeline_for_chat") as factory, \
             patch("src.chat.chat_service.SearchAnswerer") as answerer, \
             patch.object(chat_service, "llm", fake_llm):
            reply = generate_reply(
                "private:1", "看图回答", ["data:image/png;base64,AAA"], mode=SearchMode.SKIP
            )
        factory.assert_not_called()
        answerer.assert_not_called()
        self.assertEqual("plain multimodal reply", reply)

    def test_image_reaches_light_planner_and_search_answer(self):
        engine = FakeEngine(success_outcome(SearchMode.LIGHT))
        fake_llm = FakeLLM("识别到了相机")
        run_reply(engine, mode=SearchMode.LIGHT, text="", images=["data:image/png;base64,AAA"], fake_llm=fake_llm)
        self.assertEqual(("data:image/png;base64,AAA",), engine.requests[0].images)
        self.assertIn("image_url", repr(fake_llm.calls[-1][0]))

    def test_unexpected_dispatch_error_returns_fixed_reply_and_saves_history(self):
        reply = run_reply(RaisingEngine(), mode=SearchMode.LIGHT, text="q")
        self.assertEqual("在线搜索暂时不可用，请稍后再试。", reply)
        self.assertEqual(reply, chat_service.chat_history["private:1"][-1]["content"])

    def test_history_stores_placeholders_not_image_bytes(self):
        fake_llm = FakeLLM("普通回答")
        with patch.object(chat_service, "llm", fake_llm):
            generate_reply(
                "private:1",
                "图片描述",
                ["data:image/png;base64,AAA"],
                mode=SearchMode.SKIP,
            )
        user_msg = chat_service.chat_history["private:1"][-2]["content"]
        self.assertIn("[图片]", user_msg)
        self.assertNotIn("data:image/png", user_msg)

    def test_degraded_answer_does_not_add_incomplete_information_warning(self):
        engine = FakeEngine(success_outcome(SearchMode.LIGHT))
        with patch("src.chat.chat_service.SearchAnswerer") as mock_answerer_cls:
            mock_answerer_instance = MagicMock()
            mock_answerer_instance.answer.return_value = AnswerResult(text="降级回答", degraded=True)
            mock_answerer_cls.return_value = mock_answerer_instance
            with patch.object(chat_service, "get_simple_search_pipeline_for_chat", return_value=engine):
                reply = generate_reply("private:1", "搜索提问", mode=SearchMode.LIGHT)
        self.assertEqual("降级回答", reply)
        self.assertNotIn("信息可能不完整", reply)
        self.assertNotIn("不完整", reply)

    def test_light_search_no_results_falls_back_to_plain_chat(self):
        trace = SearchTrace("req1", RequestSource.CHAT, SearchMode.LIGHT)
        failure_outcome = SearchOutcome(
            plan=SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "你个笨蛋"),)),
            results=(),
            trace=trace,
            failure=SearchFailure.NO_RESULTS,
        )
        engine = FakeEngine(failure_outcome)
        fake_llm = FakeLLM("哼，才不是笨蛋呢！")
        reply = run_reply(engine, mode=SearchMode.LIGHT, text="你个笨蛋", fake_llm=fake_llm)
        self.assertEqual("哼，才不是笨蛋呢！", reply)

    def test_standard_search_no_results_retains_failure_message(self):
        trace = SearchTrace("req1", RequestSource.COMMAND, SearchMode.STANDARD)
        failure_outcome = SearchOutcome(
            plan=SearchPlan(SearchMode.STANDARD, (SearchQuery("q1", "冷门关键词"),)),
            results=(),
            trace=trace,
            failure=SearchFailure.NO_RESULTS,
        )
        engine = FakeEngine(failure_outcome)
        reply = run_reply(engine, mode=SearchMode.STANDARD, text="冷门关键词")
        self.assertEqual("没有找到可用的在线搜索结果。", reply)

    def test_get_recent_dialogue_context(self):
        from src.chat.chat_service import append_history, get_recent_dialogue_context, reset_history

        session = "private:context_test"
        reset_history(session)
        self.assertEqual((), get_recent_dialogue_context(session))

        append_history(session, "你好", "你好！有什么我可以帮你的？")
        append_history(session, "Python和Go哪个好？", "各有特点，Python开发快，Go并发好。")

        # 默认 1 轮（最后 2 条消息）
        last_turn = get_recent_dialogue_context(session, turns=1)
        self.assertEqual(
            (
                ("user", "Python和Go哪个好？"),
                ("assistant", "各有特点，Python开发快，Go并发好。"),
            ),
            last_turn,
        )

        # 2 轮
        last_two_turns = get_recent_dialogue_context(session, turns=2)
        self.assertEqual(4, len(last_two_turns))
        self.assertEqual(("user", "你好"), last_two_turns[0])

        reset_history(session)


if __name__ == "__main__":
    unittest.main()
