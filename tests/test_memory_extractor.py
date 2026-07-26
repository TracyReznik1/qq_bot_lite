import json
import unittest
from unittest import mock

from src.memory.models import MemoryContext, MemoryEvent
from src.services.llm_types import ChatResponse


def private_event(
    text: str,
    *,
    image_count: int = 0,
) -> MemoryEvent:
    return MemoryEvent(
        context=MemoryContext(
            user_id="10001",
            session_key="private:10001",
            is_group=False,
        ),
        message_id="m-1",
        sequence=1,
        text=text,
        image_count=image_count,
    )


class MemoryExtractorTests(unittest.TestCase):
    def setUp(self):
        try:
            from src.memory.extractor import MemoryExtractor
        except ModuleNotFoundError as error:
            self.fail(f"memory extractor is missing: {error}")
        self.llm = mock.Mock()
        self.extractor = MemoryExtractor(self.llm)

    def test_extracts_multiple_attributed_claims(self):
        self.llm.chat.return_value = ChatResponse(
            content=json.dumps(
                {
                    "claims": [
                        {
                            "subject_ref": "speaker",
                            "predicate": "name",
                            "value": "夏目安安",
                            "memory_type": "identity",
                            "modality": "asserted",
                            "confidence": "high",
                            "operation": "add",
                        },
                        {
                            "subject_ref": "speaker",
                            "predicate": "likes",
                            "value": "跑步",
                            "memory_type": "preference",
                            "modality": "asserted",
                            "confidence": "high",
                            "operation": "add",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        )

        claims = self.extractor.extract(
            private_event("我是夏目安安，我喜欢跑步")
        )

        self.assertEqual(
            ("name", "likes"),
            tuple(item.predicate for item in claims),
        )
        call = self.llm.chat.call_args
        self.assertEqual(0.0, call.kwargs["temperature"])
        self.assertIsNone(call.kwargs["tools"])
        self.assertEqual("none", call.kwargs["tool_choice"])

    def test_invalid_json_is_repaired_once(self):
        self.llm.chat.side_effect = [
            ChatResponse(content="not json"),
            ChatResponse(content='{"claims": []}'),
        ]

        self.assertEqual(
            (),
            self.extractor.extract(private_event("晚上好")),
        )
        self.assertEqual(2, self.llm.chat.call_count)
        repair_messages = self.llm.chat.call_args.args[0]
        self.assertIn("validation_error", repair_messages[-1]["content"])

    def test_second_invalid_response_raises_classifiable_failure(self):
        from src.memory.extractor import MemoryExtractionError

        self.llm.chat.side_effect = [
            ChatResponse(content='{"claims": "invalid"}'),
            ChatResponse(
                content=json.dumps(
                    {
                        "claims": [
                            {
                                "subject_ref": "speaker",
                                "predicate": "name",
                                "value": "安安",
                                "memory_type": "identity",
                                "modality": "asserted",
                                "confidence": "certain",
                                "operation": "add",
                            }
                        ]
                    }
                )
            ),
        ]

        with self.assertRaises(MemoryExtractionError) as raised:
            self.extractor.extract(private_event("我是安安"))

        self.assertEqual("invalid_output", raised.exception.error_type)
        self.assertEqual(2, self.llm.chat.call_count)

    def test_rejects_whole_response_when_one_claim_is_malformed(self):
        from src.memory.extractor import MemoryExtractionError

        response = {
            "claims": [
                {
                    "subject_ref": "speaker",
                    "predicate": "name",
                    "value": "安安",
                    "memory_type": "identity",
                    "modality": "asserted",
                    "confidence": "high",
                    "operation": "add",
                },
                {
                    "subject_ref": "unknown",
                    "predicate": "not valid",
                    "value": "坏候选",
                    "memory_type": "fact",
                    "modality": "asserted",
                    "confidence": "high",
                    "operation": "add",
                },
            ]
        }
        self.llm.chat.side_effect = [
            ChatResponse(content=json.dumps(response, ensure_ascii=False)),
            ChatResponse(content=json.dumps(response, ensure_ascii=False)),
        ]

        with self.assertRaises(MemoryExtractionError):
            self.extractor.extract(private_event("我是安安"))

    def test_strips_one_json_markdown_fence(self):
        self.llm.chat.return_value = ChatResponse(
            content='```json\n{"claims": []}\n```'
        )

        self.assertEqual(
            (),
            self.extractor.extract(private_event("晚上好")),
        )
        self.assertEqual(1, self.llm.chat.call_count)

    def test_non_iso_temporal_value_requires_repair(self):
        invalid = {
            "claims": [
                {
                    "subject_ref": "speaker",
                    "predicate": "likes",
                    "value": "跑步",
                    "memory_type": "preference",
                    "modality": "asserted",
                    "confidence": "high",
                    "operation": "add",
                    "valid_from": "yesterday",
                    "valid_to": None,
                }
            ]
        }
        self.llm.chat.side_effect = [
            ChatResponse(content=json.dumps(invalid, ensure_ascii=False)),
            ChatResponse(content='{"claims": []}'),
        ]

        self.assertEqual(
            (),
            self.extractor.extract(private_event("我昨天开始喜欢跑步")),
        )
        self.assertEqual(2, self.llm.chat.call_count)

    def test_images_are_ephemeral_user_input_not_system_persona(self):
        self.llm.chat.return_value = ChatResponse(content='{"claims": []}')
        image = "data:image/png;base64,cG5n"

        self.extractor.extract(
            private_event("这是我的狗", image_count=1),
            image_data_urls=(image,),
        )

        messages = self.llm.chat.call_args.args[0]
        self.assertIsInstance(messages[1]["content"], list)
        self.assertEqual(
            image,
            messages[1]["content"][-1]["image_url"]["url"],
        )
        self.assertNotIn("data:image", messages[0]["content"])
        self.assertNotIn("角色设定", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
