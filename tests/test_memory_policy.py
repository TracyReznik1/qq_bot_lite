import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
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


def seed_claim(
    store: MemoryStore,
    *,
    dedupe_key: str,
    scope_type: str = "group",
    scope_id: str = "900",
    speaker_qq: str = "101",
    subject_id: str = "999",
    predicate: str = "likes",
    value: str = "跑步",
    modality: str = "asserted",
    status: str = "active",
):
    return store.create_claim(
        scope_type=scope_type,
        scope_id=scope_id,
        speaker_qq=speaker_qq,
        subject_type="qq_user",
        subject_id=subject_id,
        predicate=predicate,
        value=value,
        memory_type="fact",
        modality=modality,
        source_kind="message:speaker",
        source_message_id=dedupe_key,
        source_excerpt=value,
        extraction_confidence="high",
        attribution_confidence="high",
        truth_confidence="high",
        dedupe_key=dedupe_key,
        status=status,
    )[0]


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

    def test_historical_candidate_closes_existing_claim_and_adds_evidence(self):
        old = self.policy.apply(
            event("我喜欢跑步", message_id="historical-old"),
            (candidate(),),
        )[0].claim

        decisions = self.policy.apply(
            event(
                "我以前喜欢跑步，现在喜欢游泳",
                message_id="historical-replacement",
            ),
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
        self.assertEqual(old.id, decisions[0].claim.id)
        self.assertEqual(2, len(self.rows()))
        archived = self.store.get_claim(old.id)
        self.assertEqual("archived", archived.status)
        self.assertEqual(
            "2026-07-26T00:00:00+00:00",
            archived.valid_to,
        )
        self.assertEqual(1, len(self.store.list_evidence(old.id)))

    def test_earlier_history_stays_separate_from_later_active_claim(self):
        current = self.policy.apply(
            event("我从七月底开始喜欢跑步", message_id="later-current"),
            (
                candidate(
                    valid_from="2026-07-27T00:00:00+00:00",
                ),
            ),
        )[0].claim
        historical_candidate = candidate(
            valid_from="2025-01-01T00:00:00+00:00",
            valid_to="2026-07-26T00:00:00+00:00",
        )

        first_history = self.policy.apply(
            event("我以前喜欢跑步", message_id="earlier-history"),
            (historical_candidate,),
        )[0]
        repeated_history = self.policy.apply(
            event("我以前确实喜欢跑步", message_id="repeat-earlier-history"),
            (historical_candidate,),
        )[0]

        unchanged_current = self.store.get_claim(current.id)
        self.assertEqual("active", unchanged_current.status)
        self.assertEqual(
            "2026-07-27T00:00:00+00:00",
            unchanged_current.valid_from,
        )
        self.assertNotEqual(current.id, first_history.claim.id)
        self.assertEqual(first_history.claim.id, repeated_history.claim.id)
        self.assertEqual("archived", first_history.claim.status)
        self.assertEqual(
            "2025-01-01T00:00:00+00:00",
            first_history.claim.valid_from,
        )
        self.assertEqual(
            "2026-07-26T00:00:00+00:00",
            first_history.claim.valid_to,
        )
        self.assertEqual(2, len(self.rows()))
        self.assertEqual(
            1,
            len(self.store.list_evidence(first_history.claim.id)),
        )

    def test_compatible_active_interval_is_archived_in_place(self):
        current = self.policy.apply(
            event("我从七月开始喜欢跑步", message_id="compatible-current"),
            (
                candidate(
                    valid_from="2026-07-01T00:00:00+00:00",
                ),
            ),
        )[0].claim

        historical = self.policy.apply(
            event("我七月喜欢过跑步", message_id="compatible-history"),
            (
                candidate(
                    valid_from="2026-07-01T00:00:00+00:00",
                    valid_to="2026-07-26T00:00:00+00:00",
                ),
            ),
        )[0].claim

        self.assertEqual(current.id, historical.id)
        self.assertEqual("archived", historical.status)
        self.assertEqual(
            "2026-07-01T00:00:00+00:00",
            historical.valid_from,
        )
        self.assertEqual(
            "2026-07-26T00:00:00+00:00",
            historical.valid_to,
        )
        self.assertEqual(1, len(self.rows()))

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

    def test_bare_number_is_not_an_explicit_qq_marker(self):
        decisions = self.policy.apply(
            event("我有123个苹果"),
            (
                candidate(
                    subject_ref="qq:123",
                    predicate="owns_count",
                    value="123个苹果",
                    memory_type="fact",
                ),
            ),
        )

        self.assertEqual((), decisions)

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

    def test_exact_active_claim_after_fts_limit_is_still_confirmed(self):
        for index in range(512):
            seed_claim(
                self.store,
                dedupe_key=f"filler-active-{index}",
                subject_id=f"filler-{index}",
                value=f"值{index}",
            )
        target = seed_claim(
            self.store,
            dedupe_key="target-after-limit",
        )

        decision = self.policy.apply(
            event(
                "QQ 999 喜欢跑步",
                user_id="101",
                message_id="confirm-after-limit",
                group_id="900",
            ),
            (candidate(subject_ref="qq:999"),),
        )[0]

        self.assertEqual("confirmed", decision.action)
        self.assertEqual(target.id, decision.claim.id)
        self.assertEqual(513, len(self.rows()))

    def test_alias_ambiguity_after_fts_limit_is_not_falsely_unique(self):
        for index in range(512):
            seed_claim(
                self.store,
                dedupe_key=f"same-alias-{index}",
                speaker_qq=f"speaker-{index}",
                subject_id="999",
                predicate="preferred_name",
                value="安安",
            )
        seed_claim(
            self.store,
            dedupe_key="conflicting-alias-after-limit",
            speaker_qq="other-speaker",
            subject_id="888",
            predicate="preferred_name",
            value="安安",
        )

        decisions = self.policy.apply(
            event(
                "安安喜欢跑步",
                user_id="777",
                message_id="ambiguous-after-limit",
                group_id="900",
            ),
            (candidate(subject_ref="qq:999"),),
        )

        self.assertEqual((), decisions)
        self.assertEqual(513, len(self.rows()))

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

    def test_supersede_targets_only_claim_bound_to_correction_clause(self):
        running = self.policy.apply(
            event("我喜欢跑步", message_id="target-running"),
            (candidate(value="跑步"),),
        )[0].claim
        swimming = self.policy.apply(
            event("我喜欢游泳", message_id="target-swimming"),
            (candidate(value="游泳"),),
        )[0].claim

        decision = self.policy.apply(
            event(
                "纠正一下，我不是游泳而是喜欢骑行",
                message_id="targeted-correction",
            ),
            (
                candidate(
                    value="骑行",
                    operation="supersede",
                ),
            ),
        )[0]

        self.assertEqual("superseded", decision.action)
        self.assertEqual("active", self.store.get_claim(running.id).status)
        self.assertEqual(
            "superseded",
            self.store.get_claim(swimming.id).status,
        )

    def test_supersede_without_unique_or_named_target_does_not_close_claims(self):
        running = self.policy.apply(
            event("我喜欢跑步", message_id="ambiguous-running"),
            (candidate(value="跑步"),),
        )[0].claim
        swimming = self.policy.apply(
            event("我喜欢游泳", message_id="ambiguous-swimming"),
            (candidate(value="游泳"),),
        )[0].claim

        decision = self.policy.apply(
            event("纠正一下，我喜欢骑行", message_id="ambiguous-target"),
            (
                candidate(
                    value="骑行",
                    operation="supersede",
                ),
            ),
        )[0]

        self.assertNotEqual("superseded", decision.action)
        self.assertEqual("active", self.store.get_claim(running.id).status)
        self.assertEqual("active", self.store.get_claim(swimming.id).status)

    def test_reconciliation_failure_rolls_back_claim_updates_and_fts(self):
        old = self.policy.apply(
            event("我叫安安", message_id="atomic-old"),
            (
                candidate(
                    predicate="name",
                    value="安安",
                    memory_type="identity",
                ),
            ),
        )[0].claim
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                CREATE TRIGGER fail_memory_relation
                BEFORE INSERT ON memory_relations
                BEGIN
                    SELECT RAISE(ABORT, 'injected relation failure');
                END
                """
            )

        with self.assertRaises(sqlite3.IntegrityError):
            self.policy.apply(
                event("纠正一下，我叫小夏", message_id="atomic-new"),
                (
                    candidate(
                        predicate="name",
                        value="小夏",
                        memory_type="identity",
                        operation="supersede",
                    ),
                ),
            )

        self.assertEqual(1, len(self.rows()))
        unchanged = self.store.get_claim(old.id)
        self.assertEqual("active", unchanged.status)
        self.assertIsNone(unchanged.valid_to)
        self.assertEqual((), self.store.search_claims("小夏"))
        self.assertEqual((), self.store.list_relations(old.id))

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

    def test_unrelated_correction_clause_cannot_supersede_a_candidate(self):
        old = self.policy.apply(
            event("我叫安安", message_id="old-name"),
            (
                candidate(
                    predicate="name",
                    value="安安",
                    memory_type="identity",
                ),
            ),
        )[0]
        decision = self.policy.apply(
            event("其实天气很好；我叫小夏", message_id="unrelated-correction"),
            (
                candidate(
                    predicate="name",
                    value="小夏",
                    memory_type="identity",
                    operation="supersede",
                ),
            ),
        )[0]

        self.assertNotEqual("superseded", decision.action)
        self.assertEqual("active", self.store.get_claim(old.claim.id).status)

    def test_parallel_unrelated_correction_cannot_supersede_name(self):
        old = self.policy.apply(
            event("我叫安安", message_id="parallel-old-name"),
            (
                candidate(
                    predicate="name",
                    value="安安",
                    memory_type="identity",
                ),
            ),
        )[0].claim

        decision = self.policy.apply(
            event(
                "我纠正天气预报同时我叫小夏",
                message_id="parallel-unrelated-correction",
            ),
            (
                candidate(
                    predicate="name",
                    value="小夏",
                    memory_type="identity",
                    operation="supersede",
                ),
            ),
        )[0]

        self.assertNotEqual("superseded", decision.action)
        self.assertEqual("active", self.store.get_claim(old.id).status)

    def test_natural_correction_still_closes_one_current_name(self):
        old = self.policy.apply(
            event("我叫安安", message_id="natural-old-name"),
            (
                candidate(
                    predicate="name",
                    value="安安",
                    memory_type="identity",
                ),
            ),
        )[0].claim

        decision = self.policy.apply(
            event("其实我叫小夏", message_id="natural-correction"),
            (
                candidate(
                    predicate="name",
                    value="小夏",
                    memory_type="identity",
                    operation="supersede",
                ),
            ),
        )[0]

        self.assertEqual("superseded", decision.action)
        self.assertEqual("superseded", self.store.get_claim(old.id).status)

    def test_correction_must_bind_subject_predicate_and_value_clause(self):
        old = self.policy.apply(
            event(
                "QQ 999 的名字是安安",
                user_id="101",
                message_id="old-other-name",
                group_id="900",
                mentioned_qq_ids=("999",),
            ),
            (
                candidate(
                    subject_ref="qq:999",
                    predicate="name",
                    value="安安",
                    memory_type="identity",
                ),
            ),
        )[0]
        decision = self.policy.apply(
            event(
                "其实 QQ 888 喜欢小夏；QQ 999 的名字是小夏",
                user_id="101",
                message_id="wrong-correction-clause",
                group_id="900",
                mentioned_qq_ids=("888", "999"),
            ),
            (
                candidate(
                    subject_ref="qq:999",
                    predicate="name",
                    value="小夏",
                    memory_type="identity",
                    operation="supersede",
                ),
            ),
        )[0]

        self.assertNotEqual("superseded", decision.action)
        self.assertEqual("active", self.store.get_claim(old.claim.id).status)

    def test_unrelated_withdrawal_clause_cannot_retract_a_candidate(self):
        old = self.policy.apply(
            event("我喜欢跑步", message_id="old-like"),
            (candidate(),),
        )[0]
        decisions = self.policy.apply(
            event("我收回天气预报；我喜欢跑步", message_id="wrong-withdrawal"),
            (
                candidate(
                    modality="negated",
                    operation="retract",
                ),
            ),
        )

        self.assertEqual((), decisions)
        self.assertEqual("active", self.store.get_claim(old.claim.id).status)

    def test_parallel_unrelated_withdrawal_cannot_retract_preference(self):
        old = self.policy.apply(
            event("我喜欢跑步", message_id="parallel-old-like"),
            (candidate(),),
        )[0].claim

        decisions = self.policy.apply(
            event(
                "我收回天气预报同时我喜欢跑步",
                message_id="parallel-unrelated-withdrawal",
            ),
            (
                candidate(
                    modality="negated",
                    operation="retract",
                ),
            ),
        )

        self.assertEqual((), decisions)
        self.assertEqual("active", self.store.get_claim(old.id).status)

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

    def test_support_supersession_and_contradiction_are_all_reconciled(self):
        old_same_speaker = self.policy.apply(
            event(
                "QQ 999 的名字是安安",
                user_id="101",
                message_id="mixed-old",
                group_id="900",
                mentioned_qq_ids=("999",),
            ),
            (
                candidate(
                    subject_ref="qq:999",
                    predicate="name",
                    value="安安",
                    memory_type="identity",
                ),
            ),
        )[0].claim
        exact_other = self.policy.apply(
            event(
                "QQ 999 的名字是小夏",
                user_id="102",
                message_id="mixed-support",
                group_id="900",
                mentioned_qq_ids=("999",),
            ),
            (
                candidate(
                    subject_ref="qq:999",
                    predicate="name",
                    value="小夏",
                    memory_type="identity",
                ),
            ),
        )[0].claim
        conflicting_other = self.policy.apply(
            event(
                "QQ 999 的名字是小王",
                user_id="103",
                message_id="mixed-conflict",
                group_id="900",
                mentioned_qq_ids=("999",),
            ),
            (
                candidate(
                    subject_ref="qq:999",
                    predicate="name",
                    value="小王",
                    memory_type="identity",
                ),
            ),
        )[0].claim

        mixed = self.policy.apply(
            event(
                "纠正一下，QQ 999 不是安安而是名字叫小夏",
                user_id="101",
                message_id="mixed-new",
                group_id="900",
                mentioned_qq_ids=("999",),
            ),
            (
                candidate(
                    subject_ref="qq:999",
                    predicate="name",
                    value="小夏",
                    memory_type="identity",
                    operation="supersede",
                ),
            ),
        )[0]

        outgoing = {
            (relation.target_claim_id, relation.relation_type)
            for relation in self.store.list_relations(mixed.claim.id)
            if relation.source_claim_id == mixed.claim.id
        }
        self.assertEqual(
            {
                (old_same_speaker.id, "supersedes"),
                (exact_other.id, "supports"),
                (conflicting_other.id, "contradicts"),
            },
            outgoing,
        )
        self.assertEqual("disputed", mixed.action)
        self.assertEqual(
            "superseded",
            self.store.get_claim(old_same_speaker.id).status,
        )
        self.assertEqual(
            "disputed",
            self.store.get_claim(conflicting_other.id).status,
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

    def test_non_string_candidate_fields_are_rejected_without_side_effects(self):
        invalid_fields = (
            ("subject_ref", 10001),
            ("predicate", ["likes"]),
            ("value", {"text": "跑步"}),
            ("memory_type", None),
            ("modality", 1),
            ("confidence", False),
            ("operation", ("add",)),
            ("valid_from", 1721952000),
            ("valid_to", []),
        )

        for index, (field, invalid_value) in enumerate(invalid_fields):
            with self.subTest(field=field):
                malformed = replace(
                    candidate(),
                    **{field: invalid_value},
                )
                self.assertEqual(
                    (),
                    self.policy.apply(
                        event(
                            "我喜欢跑步",
                            message_id=f"invalid-type-{index}",
                        ),
                        (malformed,),
                    ),
                )

        self.assertEqual([], self.rows())

    def test_temporal_values_require_aware_ordered_timestamps_and_normalize_utc(
        self,
    ):
        invalid_candidates = (
            candidate(valid_from="2026-07-26"),
            candidate(valid_from="2026-07-26T08:00:00"),
            candidate(
                valid_from="2026-07-27T00:00:00+00:00",
                valid_to="2026-07-26T00:00:00+00:00",
            ),
        )
        for index, invalid in enumerate(invalid_candidates):
            with self.subTest(candidate=invalid):
                self.assertEqual(
                    (),
                    self.policy.apply(
                        event("我喜欢跑步", message_id=f"bad-time-{index}"),
                        (invalid,),
                    ),
                )

        normalized = self.policy.apply(
            event("我以前喜欢跑步", message_id="normalized-time"),
            (
                candidate(
                    valid_from="2026-07-26T08:00:00+08:00",
                    valid_to="2026-07-27T08:00:00+08:00",
                ),
            ),
        )[0].claim
        self.assertEqual("2026-07-26T00:00:00+00:00", normalized.valid_from)
        self.assertEqual("2026-07-27T00:00:00+00:00", normalized.valid_to)

    def test_quoted_first_person_is_not_attributed_to_real_sender(self):
        decisions = self.policy.apply(
            event("小明说：“我喜欢跑步”"),
            (candidate(subject_ref="speaker"),),
        )

        self.assertEqual((), decisions)
        self.assertEqual([], self.rows())

    def test_speaker_requires_unquoted_first_person_across_quote_styles(self):
        quoted_or_unattributed = (
            "小明说：‘我喜欢跑步’",
            "小明写道《我喜欢跑步》",
            "`我喜欢跑步`",
            "> 我喜欢跑步",
            "喜欢跑步",
        )
        for index, text in enumerate(quoted_or_unattributed):
            with self.subTest(text=text):
                self.assertEqual(
                    (),
                    self.policy.apply(
                        event(text, message_id=f"quoted-style-{index}"),
                        (candidate(),),
                    ),
                )

        unquoted = self.policy.apply(
            event(
                "小明说“我喜欢跑步”，但我喜欢游泳",
                message_id="quoted-and-own",
            ),
            (candidate(value="游泳"),),
        )
        self.assertEqual(1, len(unquoted))

    def test_unclosed_quote_or_code_boundary_keeps_following_text_quoted(self):
        quoted_or_unattributed = (
            "小明说：“我叫小夏",
            '小明说："我叫小夏',
            "小明说：‘我叫小夏",
            "小明写道《我叫小夏",
            "`我叫小夏",
            "```text\n我叫小夏",
            "> 小明说：\n我叫小夏",
        )
        for index, text in enumerate(quoted_or_unattributed):
            with self.subTest(text=text):
                self.assertEqual(
                    (),
                    self.policy.apply(
                        event(text, message_id=f"unclosed-quote-{index}"),
                        (
                            candidate(
                                predicate="name",
                                value="小夏",
                                memory_type="identity",
                            ),
                        ),
                    ),
                )

        own_statement = self.policy.apply(
            event(
                "小明说：“我叫小夏”，但我叫小冬",
                message_id="closed-then-own",
            ),
            (
                candidate(
                    predicate="name",
                    value="小冬",
                    memory_type="identity",
                ),
            ),
        )
        self.assertEqual(1, len(own_statement))

    def test_qq_sender_requires_marker_or_first_person_speaker_evidence(self):
        rejected = self.policy.apply(
            event("小明叫小夏", message_id="unmarked-current-sender"),
            (
                candidate(
                    subject_ref="qq:10001",
                    predicate="name",
                    value="小夏",
                    memory_type="identity",
                ),
            ),
        )
        marked = self.policy.apply(
            event("QQ 10001 叫小夏", message_id="marked-current-sender"),
            (
                candidate(
                    subject_ref="qq:10001",
                    predicate="name",
                    value="小夏",
                    memory_type="identity",
                ),
            ),
        )
        first_person = self.policy.apply(
            event("我叫小冬", message_id="speaker-current-sender"),
            (
                candidate(
                    subject_ref="qq:10001",
                    predicate="name",
                    value="小冬",
                    memory_type="identity",
                ),
            ),
        )

        self.assertEqual((), rejected)
        self.assertEqual(("explicit_qq",), tuple(d.attribution_source for d in marked))
        self.assertEqual(("speaker",), tuple(d.attribution_source for d in first_person))

    def test_policy_rejects_over_limit_candidate_batches_and_long_qq_refs(self):
        too_many = tuple(
            candidate(
                predicate=f"fact_{index}",
                value=f"值{index}",
                memory_type="fact",
            )
            for index in range(17)
        )
        self.assertEqual(
            (),
            self.policy.apply(
                event("我陈述很多事实", message_id="too-many"),
                too_many,
            ),
        )
        self.assertEqual(
            (),
            self.policy.apply(
                event(
                    "QQ 1234567890123 喜欢跑步",
                    message_id="long-qq",
                ),
                (candidate(subject_ref="qq:1234567890123"),),
            ),
        )

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
