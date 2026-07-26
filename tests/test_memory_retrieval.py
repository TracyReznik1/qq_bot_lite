import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import shutil

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
            truth_confidence="high",
            dedupe_key=f"test:{scope_type}:{scope_id}:{speaker_qq}:{predicate}:{value}:{created_at_offset}",
            status=status,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        return claim

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


if __name__ == "__main__":
    unittest.main()
