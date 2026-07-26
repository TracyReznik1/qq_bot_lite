import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from src.memory.models import CandidateClaim, MemoryContext, MemoryEvent
from src.memory.store import MemoryStore


def event(
    text: str,
    *,
    user_id: str = "10001",
    message_id: str = "m-1",
    group_id: str | None = None,
    mentioned_qq_ids: tuple[str, ...] = (),
    reply_to_user_id: str | None = None,
    image_count: int = 0,
) -> MemoryEvent:
    is_group = group_id is not None
    return MemoryEvent(
        context=MemoryContext(
            user_id=user_id,
            session_key=(
                f"group:{group_id}:user:{user_id}"
                if is_group
                else f"private:{user_id}"
            ),
            is_group=is_group,
            group_id=group_id,
        ),
        message_id=message_id,
        sequence=1,
        text=text,
        image_count=image_count,
        mentioned_qq_ids=mentioned_qq_ids,
        reply_to_user_id=reply_to_user_id,
    )


def candidate(
    *,
    subject_ref: str = "speaker",
    predicate: str = "likes",
    value: str = "跑步",
    memory_type: str = "preference",
    modality: str = "asserted",
    confidence: str = "high",
    operation: str = "add",
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> CandidateClaim:
    return CandidateClaim(
        subject_ref=subject_ref,
        predicate=predicate,
        value=value,
        memory_type=memory_type,
        modality=modality,
        confidence=confidence,
        operation=operation,
        valid_from=valid_from,
        valid_to=valid_to,
    )


class MemoryPolicyTests(unittest.TestCase):
    def setUp(self):
        try:
            from src.memory.policy import MemoryPolicy
        except ModuleNotFoundError as error:
            self.fail(f"memory policy is missing: {error}")
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.path = Path(self.root.name) / "memory.sqlite3"
        self.store = MemoryStore(self.path)
        self.store.initialize()
        self.policy = MemoryPolicy(self.store)

    def rows(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            return connection.execute(
                "SELECT * FROM memory_claims ORDER BY id"
            ).fetchall()

    def test_policy_matrix_rejects_question_even_if_candidate_is_asserted(self):
        decisions = self.policy.apply(
            event("我是谁？"),
            (
                candidate(
                    predicate="name",
                    value="安安",
                    memory_type="identity",
                ),
            ),
        )

        self.assertEqual((), decisions)
        self.assertEqual([], self.rows())

    def test_policy_matrix_attributes_ownership_to_real_speaker(self):
        decisions = self.policy.apply(
            event("这是我的狗"),
            (
                candidate(
                    predicate="owns",
                    value="一只狗",
                    memory_type="relationship",
                ),
            ),
        )

        self.assertEqual(1, len(decisions))
        self.assertEqual("created", decisions[0].action)
        self.assertEqual("10001", decisions[0].claim.speaker_qq)
        self.assertEqual("10001", decisions[0].claim.subject_id)
        self.assertEqual("speaker", decisions[0].attribution_source)

    def test_policy_matrix_caps_hearsay_at_medium_truth(self):
        decisions = self.policy.apply(
            event(
                "听说 QQ 123 喜欢跑步",
                mentioned_qq_ids=("123",),
            ),
            (
                candidate(
                    subject_ref="qq:123",
                    modality="hearsay",
                ),
            ),
        )

        self.assertEqual(1, len(decisions))
        self.assertEqual("123", decisions[0].claim.subject_id)
        self.assertEqual("medium", decisions[0].claim.truth_confidence)
        self.assertEqual("mention", decisions[0].attribution_source)

    def test_policy_matrix_keeps_historical_validity_and_adds_current_fact(self):
        decisions = self.policy.apply(
            event("我以前喜欢跑步，现在喜欢游泳"),
            (
                candidate(
                    value="跑步",
                    valid_to="2026-07-26T00:00:00+00:00",
                ),
                candidate(
                    value="游泳",
                    valid_from="2026-07-26T00:00:00+00:00",
                ),
            ),
        )

        self.assertEqual(("archived", "created"), tuple(d.action for d in decisions))
        rows = self.rows()
        self.assertEqual(("archived", "active"), tuple(row["status"] for row in rows))
        self.assertEqual(
            "2026-07-26T00:00:00+00:00",
            rows[0]["valid_to"],
        )

    def test_policy_matrix_rejects_hard_secret(self):
        decisions = self.policy.apply(
            event("我的密钥需要记住"),
            (
                candidate(
                    predicate="api_key",
                    value="sk-abcdefghijklmnopqrstuvwxyz",
                    memory_type="fact",
                ),
            ),
        )

        self.assertEqual((), decisions)
        self.assertEqual([], self.rows())

    def test_rejects_credentials_payment_and_raw_image_from_claim_or_evidence(self):
        cases = (
            ("password is hunter2", "password is hunter2"),
            ("payment_token is pay_private", "payment_token is pay_private"),
            ("银行卡是 4242 4242 4242 4242", "4242 4242 4242 4242"),
            (
                "图像 data:image/png;base64,iVBORw0KGgoAAAA",
                "普通值",
            ),
        )

        for index, (text, value) in enumerate(cases):
            with self.subTest(text=text):
                self.assertEqual(
                    (),
                    self.policy.apply(
                        event(text, message_id=f"secret-{index}"),
                        (
                            candidate(
                                predicate="fact",
                                value=value,
                                memory_type="fact",
                            ),
                        ),
                    ),
                )

        self.assertEqual([], self.rows())

    def test_unknown_subject_is_rejected(self):
        self.assertEqual(
            (),
            self.policy.apply(
                event("某人喜欢跑步"),
                (candidate(subject_ref="unknown"),),
            ),
        )

    def test_explicit_qq_resolves_exactly_and_records_evidence_source(self):
        decisions = self.policy.apply(
            event("QQ 123 的名字是小明"),
            (
                candidate(
                    subject_ref="qq:123",
                    predicate="name",
                    value="小明",
                    memory_type="identity",
                ),
            ),
        )

        self.assertEqual("123", decisions[0].claim.subject_id)
        self.assertEqual("explicit_qq", decisions[0].attribution_source)
        self.assertEqual("message:explicit_qq", decisions[0].claim.source_kind)

    def test_unattested_qq_reference_is_rejected_instead_of_guessed(self):
        self.assertEqual(
            (),
            self.policy.apply(
                event("小明喜欢跑步"),
                (candidate(subject_ref="qq:123"),),
            ),
        )

    def test_unique_scoped_alias_resolves_and_ambiguous_alias_is_rejected(self):
        self.policy.apply(
            event(
                "请叫我安安",
                user_id="123",
                message_id="alias-123",
                group_id="900",
            ),
            (
                candidate(
                    predicate="preferred_name",
                    value="安安",
                    memory_type="preferred_name",
                ),
            ),
        )
        resolved = self.policy.apply(
            event(
                "安安喜欢跑步",
                user_id="999",
                message_id="resolved",
                group_id="900",
            ),
            (candidate(subject_ref="qq:123"),),
        )

        self.assertEqual(1, len(resolved))
        self.assertEqual("123", resolved[0].claim.subject_id)
        self.assertEqual("alias", resolved[0].attribution_source)

        self.policy.apply(
            event(
                "也请叫我安安",
                user_id="456",
                message_id="alias-456",
                group_id="900",
            ),
            (
                candidate(
                    predicate="preferred_name",
                    value="安安",
                    memory_type="preferred_name",
                ),
            ),
        )
        ambiguous = self.policy.apply(
            event(
                "安安喜欢游泳",
                user_id="999",
                message_id="ambiguous",
                group_id="900",
            ),
            (
                candidate(
                    subject_ref="qq:123",
                    value="游泳",
                ),
            ),
        )

        self.assertEqual((), ambiguous)

    def test_reply_target_resolves_from_background_supplied_user_id(self):
        decisions = self.policy.apply(
            event(
                "他喜欢跑步",
                reply_to_user_id="456",
            ),
            (candidate(subject_ref="reply_target"),),
        )

        self.assertEqual("456", decisions[0].claim.subject_id)
        self.assertEqual("reply_target", decisions[0].attribution_source)

    def test_user_statement_about_bot_is_scoped_opinion_not_persona_change(self):
        decisions = self.policy.apply(
            event("我觉得你说话很温柔"),
            (
                candidate(
                    subject_ref="bot",
                    predicate="personality",
                    value="温柔",
                    memory_type="identity",
                ),
            ),
        )

        claim = decisions[0].claim
        self.assertEqual("bot", claim.subject_type)
        self.assertEqual("bot", claim.subject_id)
        self.assertEqual("opinion", claim.memory_type)
        self.assertEqual("private", claim.scope_type)

    def test_private_and_group_scopes_never_cross(self):
        private = self.policy.apply(
            event("我喜欢跑步", message_id="private-1"),
            (candidate(),),
        )[0].claim
        group = self.policy.apply(
            event(
                "我喜欢跑步",
                message_id="group-1",
                group_id="900",
            ),
            (candidate(),),
        )[0].claim

        self.assertEqual(("private", "10001"), (private.scope_type, private.scope_id))
        self.assertEqual(("group", "900"), (group.scope_type, group.scope_id))
        self.assertNotEqual(private.id, group.id)

    def test_different_speakers_conflict_without_overwriting_each_other(self):
        first = self.policy.apply(
            event(
                "QQ 999 的名字是小明",
                user_id="101",
                message_id="first",
                group_id="900",
                mentioned_qq_ids=("999",),
            ),
            (
                candidate(
                    subject_ref="qq:999",
                    predicate="name",
                    value="小明",
                    memory_type="identity",
                ),
            ),
        )[0]
        second = self.policy.apply(
            event(
                "QQ 999 的名字是小王",
                user_id="102",
                message_id="second",
                group_id="900",
                mentioned_qq_ids=("999",),
            ),
            (
                candidate(
                    subject_ref="qq:999",
                    predicate="name",
                    value="小王",
                    memory_type="identity",
                    operation="supersede",
                ),
            ),
        )[0]

        self.assertEqual("disputed", second.action)
        self.assertEqual({"disputed"}, {row["status"] for row in self.rows()})
        relations = self.store.list_relations(second.claim.id)
        self.assertEqual("contradicts", relations[0].relation_type)
        self.assertEqual(first.claim.id, relations[0].target_claim_id)

    def test_same_speaker_explicit_correction_supersedes_old_claim(self):
        old = self.policy.apply(
            event("我叫安安", message_id="old"),
            (
                candidate(
                    predicate="name",
                    value="安安",
                    memory_type="identity",
                ),
            ),
        )[0]
        new = self.policy.apply(
            event("纠正一下，我叫小夏", message_id="new"),
            (
                candidate(
                    predicate="name",
                    value="小夏",
                    memory_type="identity",
                    operation="supersede",
                    valid_from="2026-07-26T00:00:00+00:00",
                ),
            ),
        )[0]

        self.assertEqual("superseded", new.action)
        self.assertEqual("superseded", self.store.get_claim(old.claim.id).status)
        self.assertEqual(
            "2026-07-26T00:00:00+00:00",
            self.store.get_claim(old.claim.id).valid_to,
        )
        self.assertEqual(
            "supersedes",
            self.store.list_relations(new.claim.id)[0].relation_type,
        )

    def test_same_speaker_explicit_retraction_retracts_old_claim(self):
        old = self.policy.apply(
            event("我喜欢跑步", message_id="old"),
            (candidate(),),
        )[0]
        retraction = self.policy.apply(
            event("我收回之前说喜欢跑步的话", message_id="retract"),
            (
                candidate(
                    modality="negated",
                    operation="retract",
                ),
            ),
        )[0]

        self.assertEqual("retracted", retraction.action)
        self.assertEqual("retracted", self.store.get_claim(old.claim.id).status)
        self.assertEqual(
            "retracts",
            self.store.list_relations(retraction.claim.id)[0].relation_type,
        )

    def test_untrusted_operation_cannot_supersede_without_correction_evidence(self):
        old = self.policy.apply(
            event("我叫安安", message_id="old"),
            (
                candidate(
                    predicate="name",
                    value="安安",
                    memory_type="identity",
                ),
            ),
        )[0]
        new = self.policy.apply(
            event("我叫小夏", message_id="unverified"),
            (
                candidate(
                    predicate="name",
                    value="小夏",
                    memory_type="identity",
                    operation="supersede",
                ),
            ),
        )[0]

        self.assertNotEqual("superseded", new.action)
        self.assertNotEqual(
            "superseded",
            self.store.get_claim(old.claim.id).status,
        )

    def test_untrusted_operation_cannot_retract_without_withdrawal_evidence(self):
        old = self.policy.apply(
            event("我喜欢跑步", message_id="old"),
            (candidate(),),
        )[0]
        decisions = self.policy.apply(
            event("我喜欢跑步", message_id="unverified"),
            (
                candidate(
                    modality="negated",
                    operation="retract",
                ),
            ),
        )

        self.assertEqual((), decisions)
        self.assertEqual("active", self.store.get_claim(old.claim.id).status)

    def test_same_claim_confirms_and_independent_speaker_supports(self):
        first = self.policy.apply(
            event(
                "QQ 999 的名字是小明",
                user_id="101",
                message_id="first",
                group_id="900",
                mentioned_qq_ids=("999",),
            ),
            (
                candidate(
                    subject_ref="qq:999",
                    predicate="name",
                    value="小明",
                    memory_type="identity",
                ),
            ),
        )[0]
        confirmed = self.policy.apply(
            event(
                "QQ 999 的名字确实是小明",
                user_id="101",
                message_id="confirm",
                group_id="900",
                mentioned_qq_ids=("999",),
            ),
            (
                candidate(
                    subject_ref="qq:999",
                    predicate="name",
                    value="小明",
                    memory_type="identity",
                    operation="confirm",
                ),
            ),
        )[0]
        supported = self.policy.apply(
            event(
                "QQ 999 的名字是小明",
                user_id="102",
                message_id="support",
                group_id="900",
                mentioned_qq_ids=("999",),
            ),
            (
                candidate(
                    subject_ref="qq:999",
                    predicate="name",
                    value="小明",
                    memory_type="identity",
                ),
            ),
        )[0]

        self.assertEqual("confirmed", confirmed.action)
        self.assertEqual(first.claim.id, confirmed.claim.id)
        self.assertEqual(1, len(self.store.list_evidence(first.claim.id)))
        self.assertEqual("supported", supported.action)
        self.assertEqual(
            "supports",
            self.store.list_relations(supported.claim.id)[0].relation_type,
        )

    def test_lifecycle_rules_keep_preferred_names_and_preference_floor(self):
        preferred = self.policy.apply(
            event("请一直叫我安安", message_id="preferred"),
            (
                candidate(
                    predicate="preferred_name",
                    value="安安",
                    memory_type="preferred_name",
                    confidence="medium",
                    valid_to="2026-07-27T00:00:00+00:00",
                ),
            ),
        )[0].claim
        preference = self.policy.apply(
            event("我可能喜欢跑步", message_id="preference"),
            (
                candidate(
                    confidence="medium",
                    modality="uncertain",
                ),
            ),
        )[0].claim

        self.assertIsNone(preferred.valid_to)
        self.assertEqual("active", preferred.status)
        self.assertEqual("medium", preferred.truth_confidence)
        self.assertEqual("medium", preference.truth_confidence)

    def test_policy_revalidates_untrusted_candidate_shape_and_timestamps(self):
        invalid_candidates = (
            candidate(predicate="not valid"),
            candidate(value=" "),
            candidate(confidence="certain"),
            candidate(operation="merge"),
            candidate(valid_from="sometime"),
        )

        for index, invalid in enumerate(invalid_candidates):
            with self.subTest(candidate=invalid):
                self.assertEqual(
                    (),
                    self.policy.apply(
                        event(
                            "我喜欢跑步",
                            message_id=f"invalid-{index}",
                        ),
                        (invalid,),
                    ),
                )

        self.assertEqual([], self.rows())

    def test_quoted_first_person_is_not_attributed_to_real_sender(self):
        decisions = self.policy.apply(
            event("小明说：“我喜欢跑步”"),
            (candidate(subject_ref="speaker"),),
        )

        self.assertEqual((), decisions)
        self.assertEqual([], self.rows())

    def test_sensitive_group_claim_and_image_only_ownership_are_rejected(self):
        group_sensitive = self.policy.apply(
            event(
                "我的家庭住址是某处",
                group_id="900",
                message_id="sensitive",
            ),
            (
                candidate(
                    predicate="home_address",
                    value="某处",
                    memory_type="identity",
                ),
            ),
        )
        image_only = self.policy.apply(
            event("", image_count=1, message_id="image-only"),
            (
                candidate(
                    predicate="owns",
                    value="图中的狗",
                    memory_type="relationship",
                ),
            ),
        )
        accompanied = self.policy.apply(
            event(
                "这是我的狗",
                image_count=1,
                message_id="image-text",
            ),
            (
                candidate(
                    predicate="owns",
                    value="图中的狗",
                    memory_type="relationship",
                ),
            ),
        )

        self.assertEqual((), group_sensitive)
        self.assertEqual((), image_only)
        self.assertEqual(1, len(accompanied))

    def test_low_confidence_and_joke_do_not_become_claims(self):
        self.assertEqual(
            (),
            self.policy.apply(
                event("听说我可能喜欢跑步"),
                (candidate(confidence="low", modality="hearsay"),),
            ),
        )
        self.assertEqual(
            (),
            self.policy.apply(
                event("我喜欢跑步，开玩笑的"),
                (candidate(),),
            ),
        )

    def test_textual_negation_and_hearsay_override_mislabeled_candidates(self):
        negated = self.policy.apply(
            event("我不喜欢跑步", message_id="negated"),
            (candidate(modality="asserted"),),
        )
        hearsay = self.policy.apply(
            event("听别人说我喜欢游泳", message_id="hearsay"),
            (
                candidate(
                    value="游泳",
                    modality="asserted",
                ),
            ),
        )

        self.assertEqual((), negated)
        self.assertEqual(1, len(hearsay))
        self.assertEqual("medium", hearsay[0].claim.truth_confidence)


if __name__ == "__main__":
    unittest.main()
