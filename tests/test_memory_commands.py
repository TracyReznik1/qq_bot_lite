import tempfile
import unittest
from pathlib import Path

from src.chat.chat_service import chat_history
from src.commands import CommandContext, handle_command
from src.config import config
from src.memory.models import CandidateClaim, MemoryContext, MemoryEvent
from src.memory.policy import MemoryPolicy
from src.memory.store import MemoryStore
from src.router import Route


class MemoryCommandsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()
        self.policy = MemoryPolicy(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_context_derives_memory_context_and_admin_status(self):
        ctx = CommandContext(uid="1001", session_key="private:1001", raw_message="/help")
        self.assertIsNotNone(ctx.memory_context)
        self.assertEqual("1001", ctx.memory_context.user_id)
        self.assertFalse(ctx.memory_context.is_group)

        admin_ctx = CommandContext(
            uid=config.admin_qq_ids[0] if config.admin_qq_ids else "99999",
            session_key="private:99999",
            raw_message="/help",
        )
        if config.admin_qq_ids:
            self.assertTrue(admin_ctx.is_admin)

    def test_reset_command_clears_chat_history_only(self):
        chat_history["private:1001"] = [{"role": "user", "parts": ["hi"]}]
        ctx = CommandContext(uid="1001", session_key="private:1001", raw_message="/reset")
        route = Route(handler="command", action="command", command="reset", query="")
        result = handle_command(route, ctx, store=self.store)

        self.assertTrue(result.handled)
        self.assertNotIn("private:1001", chat_history)
        self.assertIn("当前会话上下文已清空", result.reply)

    def test_remember_and_memories_and_forget_flow(self):
        # 1. Store a claim via policy first to test retrieval & forget
        ctx = MemoryContext(user_id="1001", session_key="private:1001", is_group=False)
        event = MemoryEvent(context=ctx, message_id="m1", sequence=1, text="我喜欢吃苹果")
        candidate = CandidateClaim(
            subject_ref="speaker",
            predicate="food_preference",
            value="喜欢吃苹果",
            memory_type="preference",
            modality="asserted",
            confidence="high",
        )
        self.policy.apply(event, [candidate])

        # 2. Query /memories
        cmd_ctx = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message="/memories 苹果",
            memory_context=ctx,
        )
        route_mem = Route(handler="command", action="command", command="memories", query="苹果")
        result_mem = handle_command(route_mem, cmd_ctx, store=self.store)
        self.assertTrue(result_mem.handled)
        self.assertIn("喜欢吃苹果", result_mem.reply)

        # 3. Forget by keyword
        route_forget = Route(handler="command", action="command", command="forget", query="苹果")
        result_forget = handle_command(route_forget, cmd_ctx, store=self.store)
        self.assertTrue(result_forget.handled)
        self.assertIn("已删除", result_forget.reply)

        # 4. Query /memories after forget
        result_mem2 = handle_command(route_mem, cmd_ctx, store=self.store)
        self.assertIn("没有找到", result_mem2.reply)

    def test_globalremember_permission_check(self):
        ctx_user = CommandContext(
            uid="regular_user",
            session_key="private:regular_user",
            raw_message="/globalremember 规则",
            is_admin=False,
        )
        route = Route(handler="command", action="command", command="globalremember", query="规则")
        result = handle_command(route, ctx_user, store=self.store)
        self.assertIn("只能由管理员", result.reply)


if __name__ == "__main__":
    unittest.main()
