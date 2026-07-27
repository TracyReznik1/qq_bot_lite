import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
from unittest import mock

from src.chat.prompt import build_system_prompt
from src.memory.models import (
    MemoryClaim,
    MemoryContext,
    MemoryEvent,
    RetrievedMemory,
)
from src.memory.store import MemoryStore
from src.memory.policy import MemoryPolicy
from src.memory.retriever import MemoryRetriever, format_memory_context


def _utc_now_offset(seconds: int = 0) -> str:
    dt = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return dt.isoformat()


class MemoryRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()
        self.policy = MemoryPolicy(self.store)
        self.retriever = MemoryRetriever(self.store)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_claim(
        self,
        scope_type: str,
        scope_id: str,
        speaker_qq: str,
        subject_id: str,
        predicate: str,
        value: str,
        memory_type: str = "fact",
        status: str = "active",
        valid_from: str | None = None,
        valid_to: str | None = None,
        created_at_offset: int = 0,
        truth_confidence: str = "high",
    ) -> MemoryClaim:
        event = MemoryEvent(
            context=MemoryContext(
                user_id=speaker_qq,
                session_key=f"{scope_type}:{scope_id}:{speaker_qq}" if scope_type == "group" else f"private:{speaker_qq}",
                is_group=(scope_type == "group"),
                group_id=scope_id if scope_type == "group" else None,
            ),
            message_id="msg_1",
            sequence=1,
            text=f"test message {predicate} {value}",
        )
        claim, _ = self.store.create_claim(
            scope_type=scope_type,
            scope_id=scope_id,
            speaker_qq=speaker_qq,
            subject_type="qq_user" if subject_id.isdigit() else subject_id,
            subject_id=subject_id,
            predicate=predicate,
            value=value,
            memory_type=memory_type,
            modality="asserted",
            source_kind="message:test",
            source_message_id="msg_1",
            source_excerpt=f"test message {predicate} {value}",
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence=truth_confidence,
            dedupe_key=f"test:{scope_type}:{scope_id}:{speaker_qq}:{predicate}:{value}:{created_at_offset}",
            status=status,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        return claim

    def test_default_retriever_uses_config_memory_database_path(self):
        expected_path = Path(self.temp_dir) / "configured" / "memory.sqlite3"
        fake_config = SimpleNamespace(memory_database_path=expected_path)

        with mock.patch("src.memory.retriever.config", fake_config):
            retriever = MemoryRetriever()

        self.assertEqual(expected_path, retriever.store.path)

    def test_private_scope_isolation_user_a_and_user_b(self):
        # User A private claim
        self._create_claim("private", "1001", "1001", "1001", "likes", "苹果", memory_type="preference")
        # User B private claim
        self._create_claim("private", "1002", "1002", "1002", "likes", "香蕉", memory_type="preference")

        ctx_a = MemoryContext(user_id="1001", session_key="private:1001", is_group=False)
        results_a = self.retriever.retrieve(ctx_a, "喜欢")
        claims_a = [r.claim.value for r in results_a]
        self.assertIn("苹果", claims_a)
        self.assertNotIn("香蕉", claims_a)

        ctx_b = MemoryContext(user_id="1002", session_key="private:1002", is_group=False)
        results_b = self.retriever.retrieve(ctx_b, "喜欢")
        claims_b = [r.claim.value for r in results_b]
        self.assertIn("香蕉", claims_b)
        self.assertNotIn("苹果", claims_b)

    def test_group_scope_isolation_group_1_and_group_2(self):
        self._create_claim("group", "2001", "1001", "1001", "project", "Alpha")
        self._create_claim("group", "2002", "1002", "1002", "project", "Beta")

        ctx_g1 = MemoryContext(user_id="1001", session_key="group:2001:1001", is_group=True, group_id="2001")
        results_g1 = self.retriever.retrieve(ctx_g1, "project")
        claims_g1 = [r.claim.value for r in results_g1]
        self.assertIn("Alpha", claims_g1)
        self.assertNotIn("Beta", claims_g1)

        ctx_g2 = MemoryContext(user_id="1002", session_key="group:2002:1002", is_group=True, group_id="2002")
        results_g2 = self.retriever.retrieve(ctx_g2, "project")
        claims_g2 = [r.claim.value for r in results_g2]
        self.assertIn("Beta", claims_g2)
        self.assertNotIn("Alpha", claims_g2)

    def test_global_scope_attribution(self):
        self._create_claim("global", "global", "9999", "9999", "role", "管理员", memory_type="identity")

        ctx_g = MemoryContext(user_id="1001", session_key="group:2001:1001", is_group=True, group_id="2001")
        results = self.retriever.retrieve(ctx_g, "管理员")
        self.assertEqual(1, len(results))
        self.assertEqual("9999", results[0].claim.speaker_qq)
        self.assertEqual("global", results[0].claim.scope_type)

    def test_preferred_names_no_decay_and_preference_floor(self):
        # Create preferred_name claim with old timestamp
        old_time = _utc_now_offset(-86400 * 365)
        claim_name = self._create_claim(
            "private", "1001", "1001", "1001", "preferred_name", "安安",
            memory_type="preferred_name"
        )
        self.store.update_claim(claim_name.id, last_confirmed_at=old_time)

        # Create preference claim with old timestamp
        claim_pref = self._create_claim(
            "private", "1001", "1001", "1001", "response_style", "简洁",
            memory_type="preference"
        )
        self.store.update_claim(claim_pref.id, last_confirmed_at=old_time)

        ctx = MemoryContext(user_id="1001", session_key="private:1001", is_group=False)
        results = self.retriever.retrieve(ctx, "")
        scores = {r.claim.predicate: r.score for r in results}
        self.assertIn("preferred_name", scores)
        self.assertIn("response_style", scores)
        # Preferred name should have high score regardless of age
        self.assertGreaterEqual(scores["preferred_name"], 0.8)
        # Preference should retain score floor
        self.assertGreaterEqual(scores["response_style"], 0.5)

    def test_disputed_claims_preserved_with_attribution(self):
        # Two different speakers in same group make conflicting claims
        self._create_claim("group", "2001", "1001", "bot", "opinion", "好用", status="disputed")
        self._create_claim("group", "2001", "1002", "bot", "opinion", "不好用", status="disputed")

        ctx = MemoryContext(user_id="1001", session_key="group:2001:1001", is_group=True, group_id="2001")
        results = self.retriever.retrieve(ctx, "bot")
        values = [(r.claim.speaker_qq, r.claim.value, r.claim.status) for r in results]
        self.assertEqual(2, len(values))
        self.assertIn(("1001", "好用", "disputed"), values)
        self.assertIn(("1002", "不好用", "disputed"), values)
        formatted = format_memory_context(results)
        self.assertIn("发言者=1001", formatted)
        self.assertIn("发言者=1002", formatted)
        self.assertIn("内容=opinion为好用", formatted)
        self.assertIn("内容=opinion为不好用", formatted)

    def test_excluded_lifecycle_states(self):
        self._create_claim("private", "1001", "1001", "1001", "status1", "retracted_val", status="retracted")
        self._create_claim("private", "1001", "1001", "1001", "status2", "superseded_val", status="superseded")
        self._create_claim("private", "1001", "1001", "1001", "status3", "archived_val", status="archived")
        self._create_claim("private", "1001", "1001", "1001", "status4", "active_val", status="active")

        ctx = MemoryContext(user_id="1001", session_key="private:1001", is_group=False)
        results = self.retriever.retrieve(ctx, "status")
        retrieved_values = [r.claim.value for r in results]
        self.assertIn("active_val", retrieved_values)
        self.assertNotIn("retracted_val", retrieved_values)
        self.assertNotIn("superseded_val", retrieved_values)
        self.assertNotIn("archived_val", retrieved_values)

    def test_private_personalization_in_group(self):
        # Sender's private preferred name and response style
        self._create_claim("private", "1001", "1001", "1001", "preferred_name", "安安", memory_type="preferred_name")
        self._create_claim("private", "1001", "1001", "1001", "response_style", "使用二次元语气", memory_type="preference")
        # Another user's private claim
        self._create_claim("private", "1002", "1002", "1002", "secret_hobby", "私密兴趣", memory_type="preference")

        ctx_group = MemoryContext(user_id="1001", session_key="group:2001:1001", is_group=True, group_id="2001")
        results = self.retriever.retrieve(ctx_group, "")

        evidence_results = [r for r in results if getattr(r, "usage", "evidence") == "evidence"]
        personalization_results = [r for r in results if getattr(r, "usage", "evidence") == "personalization"]

        # User 1002's private data must NOT be in group evidence
        all_values = [r.claim.value for r in results]
        self.assertNotIn("私密兴趣", all_values)

        # Sender's preferred name should be in personalization results
        p_values = [r.claim.value for r in personalization_results]
        self.assertIn("安安", p_values)

    def test_group_prompt_never_exposes_private_identity_as_evidence(self):
        self._create_claim(
            "private",
            "1001",
            "1001",
            "1001",
            "real_name",
            "PRIVATE-NAME",
            memory_type="identity",
        )

        context = MemoryContext(
            user_id="1001",
            session_key="group:2001:1001",
            is_group=True,
            group_id="2001",
        )
        results = self.retriever.retrieve(context, "我是谁")
        formatted = format_memory_context(results)

        self.assertNotIn("PRIVATE-NAME", [result.claim.value for result in results])
        self.assertNotIn("PRIVATE-NAME", formatted)

    def test_group_personalization_fallback_is_current_user_self_attribution_only(self):
        self._create_claim(
            "group",
            "2001",
            "1001",
            "1001",
            "preferred_name",
            "CURRENT-GROUP-NAME",
            memory_type="preferred_name",
        )
        self._create_claim(
            "group",
            "2001",
            "1002",
            "1002",
            "preferred_name",
            "OTHER-USER-NAME",
            memory_type="preferred_name",
        )
        self._create_claim(
            "group",
            "2002",
            "1001",
            "1001",
            "preferred_name",
            "OTHER-GROUP-NAME",
            memory_type="preferred_name",
        )

        context = MemoryContext(
            user_id="1001",
            session_key="group:2001:1001",
            is_group=True,
            group_id="2001",
        )
        results = self.retriever.retrieve(context, "怎么称呼")
        personalization = {
            result.claim.value
            for result in results
            if result.usage == "personalization"
        }

        self.assertEqual({"CURRENT-GROUP-NAME"}, personalization)
        self.assertNotIn(
            "OTHER-GROUP-NAME",
            [result.claim.value for result in results],
        )

    def test_private_preferred_name_prevents_group_personalization_fallback(self):
        self._create_claim(
            "private",
            "1001",
            "1001",
            "1001",
            "preferred_name",
            "PRIVATE-PREFERRED-NAME",
            memory_type="preferred_name",
        )
        self._create_claim(
            "group",
            "2001",
            "1001",
            "1001",
            "preferred_name",
            "GROUP-PREFERRED-NAME",
            memory_type="preferred_name",
        )
        context = MemoryContext(
            user_id="1001",
            session_key="group:2001:1001",
            is_group=True,
            group_id="2001",
        )

        results = self.retriever.retrieve(context, "怎么称呼")
        personalization = {
            result.claim.value
            for result in results
            if result.usage == "personalization"
        }

        self.assertEqual({"PRIVATE-PREFERRED-NAME"}, personalization)

    def test_first_person_identity_query_ranks_current_qq_subject_first(self):
        self._create_claim(
            "private",
            "1001",
            "1001",
            "1001",
            "name",
            "CURRENT-PERSON",
            memory_type="identity",
        )
        for index, query in enumerate(("我是谁", "我叫什么", "怎么称呼"), 1):
            self._create_claim(
                "private",
                "1001",
                "1001",
                str(2000 + index),
                "note",
                query,
                memory_type="fact",
            )
        context = MemoryContext(
            user_id="1001",
            session_key="private:1001",
            is_group=False,
        )

        for query in ("我是谁", "我叫什么", "怎么称呼"):
            with self.subTest(query=query):
                result = self.retriever.retrieve(context, query, limit=1)
                self.assertEqual("CURRENT-PERSON", result[0].claim.value)

    def test_explicit_alias_resolves_current_group_before_global(self):
        self._create_claim(
            "group",
            "2001",
            "1002",
            "1002",
            "name",
            "小明",
            memory_type="identity",
        )
        self._create_claim(
            "group",
            "2001",
            "1002",
            "1002",
            "likes",
            "GROUP-SUBJECT-PREFERENCE",
            memory_type="preference",
        )
        self._create_claim(
            "group",
            "2001",
            "1001",
            "1001",
            "likes",
            "CURRENT-USER-DISTRACTOR",
            memory_type="preference",
        )
        self._create_claim(
            "global",
            "global",
            "9002",
            "9002",
            "name",
            "小明",
            memory_type="identity",
        )
        self._create_claim(
            "global",
            "global",
            "9002",
            "9002",
            "likes",
            "GLOBAL-SUBJECT-PREFERENCE",
            memory_type="preference",
        )
        context = MemoryContext(
            user_id="1001",
            session_key="group:2001:1001",
            is_group=True,
            group_id="2001",
        )

        results = self.retriever.retrieve(context, "小明喜欢什么")
        scores = {result.claim.value: result.score for result in results}

        self.assertGreater(
            scores["GROUP-SUBJECT-PREFERENCE"],
            scores["GLOBAL-SUBJECT-PREFERENCE"],
        )
        self.assertGreater(
            scores["GROUP-SUBJECT-PREFERENCE"],
            scores["CURRENT-USER-DISTRACTOR"],
        )

    def test_truth_confidence_affects_ranking_before_extraction_confidence(self):
        low_truth = self._create_claim(
            "private",
            "1001",
            "1001",
            "1001",
            "likes",
            "LOW-TRUTH",
            memory_type="preference",
            truth_confidence="low",
        )
        high_truth = self._create_claim(
            "private",
            "1001",
            "1001",
            "1001",
            "likes",
            "HIGH-TRUTH",
            memory_type="preference",
            truth_confidence="high",
        )
        self.store.update_claim(low_truth.id, extraction_confidence="high")
        self.store.update_claim(high_truth.id, extraction_confidence="low")
        context = MemoryContext(
            user_id="1001",
            session_key="private:1001",
            is_group=False,
        )

        results = self.retriever.retrieve(context, "我喜欢什么")
        scores = {result.claim.value: result.score for result in results}

        self.assertGreater(scores["HIGH-TRUTH"], scores["LOW-TRUTH"])

    def test_expired_current_claim_is_excluded(self):
        self._create_claim(
            "private",
            "1001",
            "1001",
            "1001",
            "likes",
            "EXPIRED",
            memory_type="preference",
            valid_to=_utc_now_offset(-1),
        )
        self._create_claim(
            "private",
            "1001",
            "1001",
            "1001",
            "likes",
            "CURRENT",
            memory_type="preference",
            valid_to=_utc_now_offset(3600),
        )
        context = MemoryContext(
            user_id="1001",
            session_key="private:1001",
            is_group=False,
        )

        values = [
            result.claim.value
            for result in self.retriever.retrieve(context, "喜欢")
        ]

        self.assertEqual(["CURRENT"], values)

    def test_format_memory_context_output(self):
        c1 = self._create_claim("group", "2001", "1001", "1001", "topic", "AI研究", memory_type="opinion")
        r1 = RetrievedMemory(claim=c1, score=1.0, usage="evidence")
        c2 = self._create_claim("private", "1001", "1001", "1001", "preferred_name", "安安", memory_type="preferred_name")
        r2 = RetrievedMemory(claim=c2, score=1.0, usage="personalization")

        formatted = format_memory_context((r1, r2))
        self.assertIn("[允许使用的记忆证据]", formatted)
        self.assertIn("- 作用域=group:2001；发言者=1001；主体=1001；类型=opinion；内容=topic为AI研究", formatted)
        self.assertIn("[仅用于称呼和表达的个性化信息]", formatted)
        self.assertIn("- 主体=当前发言者；首选称呼=安安", formatted)
        self.assertIn("禁止把本区内容作为公开身份、经历或关系事实。", formatted)

    def test_system_prompt_states_privacy_before_persona_and_evidence(self):
        prompt = build_system_prompt("private:1001")

        self.assertIn(
            "规则优先级：能力与安全边界 > 隐私与权限规则 > 角色人格 > 非可信证据。",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
