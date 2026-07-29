import unittest
from unittest import mock

from src.chat import chat_service
from src.services.llm_types import ChatResponse


def tool_call(call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "search_web",
            "arguments": '{"query":"测试"}',
        },
    }


class SequencedLLM:
    def __init__(self):
        self.calls = []
        self.responses = [
            ChatResponse(tool_calls=[tool_call("call-1")]),
            ChatResponse(tool_calls=[tool_call("call-2")]),
            ChatResponse(content="最终整理结果"),
        ]

    def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FinalSynthesisTests(unittest.TestCase):
    def test_final_synthesis_keeps_tools_but_forbids_more_calls(self):
        fake_llm = SequencedLLM()
        session_key = "final-synthesis-test"

        with (
            mock.patch.object(chat_service, "llm", fake_llm),
            mock.patch.object(
                chat_service,
                "run_tool",
                return_value="搜索结果",
            ),
            mock.patch.object(
                chat_service,
                "build_untrusted_context",
                return_value="[非可信上下文]暂无",
            ),
            mock.patch.object(chat_service, "append_history"),
            mock.patch.object(chat_service, "_ensure_history_loaded"),
        ):
            chat_service.chat_history[session_key] = []
            try:
                reply = chat_service.generate_reply(
                    session_key,
                    "查一下",
                )
            finally:
                chat_service.chat_history.pop(session_key, None)

        self.assertEqual("最终整理结果", reply)
        self.assertEqual(3, len(fake_llm.calls))
        self.assertTrue(fake_llm.calls[-1]["tools"])
        self.assertEqual(
            "none",
            fake_llm.calls[-1]["tool_choice"],
        )


if __name__ == "__main__":
    unittest.main()
