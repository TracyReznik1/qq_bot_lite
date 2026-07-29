import json
import logging
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest import mock

from src.chat.chat_service import chat_history
from src.commands import CommandContext, handle_command
from src.config import config
from src.memory.extractor import MemoryExtractionError
from src.memory.models import CandidateClaim, MemoryContext, MemoryEvent
from src.memory.policy import MemoryPolicy
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore
from src.router import Route
from src.services.llm_types import ChatResponse
from tests.runtime import COMPACT_HARD_SECRET_CASES


class MemoryCommandsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "memory.sqlite3"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()
        self.policy = MemoryPolicy(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_claim(
        self,
        *,
        scope_type: str,
        scope_id: str,
        speaker: str,
        subject: str,
        predicate: str,
        value: str,
        status: str = "active",
    ):
        claim, created = self.store.create_claim(
            scope_type=scope_type,
            scope_id=scope_id,
            speaker_qq=speaker,
            subject_type="qq_user",
            subject_id=subject,
            predicate=predicate,
            value=value,
            memory_type="preference",
            modality="asserted",
            source_kind="message:speaker",
            source_message_id=f"source-{scope_type}-{scope_id}-{predicate}-{value}",
            source_excerpt=value,
            extraction_confidence="high",
            attribution_confidence="high",
            truth_confidence="high",
            dedupe_key=f"test-{scope_type}-{scope_id}-{speaker}-{subject}-{predicate}-{value}",
            status=status,
        )
        self.assertTrue(created)
        return claim

    @staticmethod
    def _valid_extractor_response(value: str = "测试值") -> ChatResponse:
        return ChatResponse(
            content=json.dumps(
                {
                    "claims": [
                        {
                            "subject_ref": "speaker",
                            "predicate": "remembered_text",
                            "value": value,
                            "memory_type": "preference",
                            "modality": "asserted",
                            "confidence": "high",
                            "operation": "add",
                            "valid_from": None,
                            "valid_to": None,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

    def _explicit_memory_command(
        self,
        command: str,
        *,
        query: str,
        renderer,
    ):
        is_global = command == "globalremember"
        return handle_command(
            Route(
                handler="command",
                action="command",
                command=command,
                query=query,
            ),
            CommandContext(
                uid="9001" if is_global else "1001",
                session_key=(
                    "private:9001" if is_global else "private:1001"
                ),
                raw_message=f"/{command} {query}",
                message_id=f"{command}-failure-contract",
                is_admin=is_global,
            ),
            store=self.store,
            renderer=renderer,
        )

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

    def test_deleted_remember_command_cannot_replay_same_claim_dedupe(self):
        value = "命令回放测试值"
        extracted = (
            CandidateClaim(
                subject_ref="speaker",
                predicate="remembered_text",
                value=value,
                memory_type="preference",
                modality="asserted",
                confidence="high",
            ),
        )
        replay_context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message=f"/remember {value}",
            message_id="remember-replay-source",
        )

        def remember():
            with mock.patch(
                "src.commands.MemoryExtractor.extract",
                return_value=extracted,
            ):
                return handle_command(
                    Route(
                        handler="command",
                        action="command",
                        command="remember",
                        query=value,
                    ),
                    replay_context,
                    store=self.store,
                )

        first = remember()
        claim_id = int(first.outcome.facts[0].partition("=")[2])
        deleted = handle_command(
            Route(
                handler="command",
                action="command",
                command="forget",
                query=str(claim_id),
            ),
            CommandContext(
                uid="1001",
                session_key="private:1001",
                raw_message=f"/forget {claim_id}",
                message_id="forget-remember-replay-source",
            ),
            store=self.store,
        )
        replay = remember()

        self.assertEqual(
            ("applied", "deleted", "rejected"),
            (
                first.outcome.status,
                deleted.outcome.status,
                replay.outcome.status,
            )
        )
        self.assertIsNone(self.store.get_claim(claim_id))

    def test_memories_never_lists_legacy_private_hard_secret(self):
        secrets = (
            "sk-legacy-private-command-secret-abcdef123456",
            *(
                secret
                for _label, secret, _raw_value
                in COMPACT_HARD_SECRET_CASES
            ),
        )
        targets = tuple(
            self.create_claim(
                scope_type="private",
                scope_id="1001",
                speaker="1001",
                subject="1001",
                predicate="fact",
                value=secret,
            )
            for secret in secrets
        )
        ordinary = self.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker="1001",
            subject="1001",
            predicate="fact",
            value="密码学基础",
        )

        result = handle_command(
            Route(
                handler="command",
                action="command",
                command="memories",
                query="",
            ),
            CommandContext(
                uid="1001",
                session_key="private:1001",
                raw_message="/memories",
                message_id="legacy-private-secret-list",
            ),
            store=self.store,
        )

        self.assertEqual("listed", result.outcome.status)
        self.assertIn(str(ordinary.id), result.outcome.facts)
        for target, secret in zip(targets, secrets):
            with self.subTest(secret=secret):
                self.assertNotIn(str(target.id), result.outcome.facts)
                self.assertNotIn(secret, result.reply)

    def test_space_delimited_secret_command_has_no_output_or_storage_leak(self):
        secret = "PASSWORD_COMMAND_SENTINEL_61E7"

        class RecordingRenderer:
            def __init__(self):
                self.facts = None
                self.fallback = None

            def render(self, facts, fallback):
                self.facts = facts
                self.fallback = fallback
                return f"persona::{fallback}"

        class RecordingHandler(logging.Handler):
            def __init__(self):
                super().__init__()
                self.messages = []

            def emit(self, record):
                self.messages.append(self.format(record))

        renderer = RecordingRenderer()
        log_handler = RecordingHandler()
        command_logger = logging.getLogger("qq-bot")
        command_logger.addHandler(log_handler)
        try:
            with mock.patch(
                "src.commands.MemoryExtractor.extract",
                return_value=(),
            ):
                result = handle_command(
                    Route(
                        handler="command",
                        action="command",
                        command="remember",
                        query=f"密码 {secret}",
                    ),
                    CommandContext(
                        uid="1001",
                        session_key="private:1001",
                        raw_message=f"/remember 密码 {secret}",
                        message_id="space-delimited-secret-command",
                    ),
                    store=self.store,
                    renderer=renderer,
                )
        finally:
            command_logger.removeHandler(log_handler)

        self.assertEqual("rejected", result.outcome.status)
        with closing(sqlite3.connect(self.db_path)) as connection:
            claim_count = connection.execute(
                "SELECT COUNT(*) FROM memory_claims"
            ).fetchone()[0]
            job_count = connection.execute(
                "SELECT COUNT(*) FROM memory_jobs"
            ).fetchone()[0]
        self.assertEqual(0, claim_count)
        self.assertEqual(0, job_count)
        leak_surfaces = {
            "fallback": result.outcome.fallback_reply,
            "outcome facts": repr(result.outcome.facts),
            "final reply": result.reply,
            "renderer facts": repr(renderer.facts),
            "renderer fallback": renderer.fallback,
            "logs": "\n".join(log_handler.messages),
        }
        for surface, text in leak_surfaces.items():
            with self.subTest(surface=surface):
                self.assertNotIn(secret, text)
        sentinel_bytes = secret.encode("utf-8")
        for database_file in (
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            if database_file.exists():
                with self.subTest(database_file=database_file.name):
                    self.assertNotIn(
                        sentinel_bytes,
                        database_file.read_bytes(),
                    )

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

    def test_explicit_memory_failures_are_classified_and_body_free(self):
        secret = "USER-MEMORY-BODY-7b1f33"

        class Renderer:
            def __init__(self):
                self.facts = None

            def render(self, facts, fallback):
                self.facts = facts
                return fallback

        class InvalidExtractorModel:
            def chat(self, _messages, **_kwargs):
                return ChatResponse(content=f"invalid-{secret}")

        class UnavailableProviderModel:
            def chat(self, _messages, **_kwargs):
                raise RuntimeError(f"provider-{secret}")

        class ValidExtractorModel:
            def chat(self, _messages, **_kwargs):
                return MemoryCommandsTests._valid_extractor_response(secret)

        cases = (
            (
                "extractor_unavailable",
                InvalidExtractorModel(),
                None,
            ),
            (
                "provider_unavailable",
                UnavailableProviderModel(),
                None,
            ),
            (
                "store_unavailable",
                ValidExtractorModel(),
                sqlite3.OperationalError("database is locked"),
            ),
        )
        for command in ("remember", "globalremember"):
            for cause, model, store_error in cases:
                with self.subTest(command=command, cause=cause):
                    renderer = Renderer()
                    store_patch = (
                        mock.patch.object(
                            self.store,
                            "reconciliation",
                            side_effect=store_error,
                        )
                        if store_error is not None
                        else mock.patch.object(
                            self.store,
                            "reconciliation",
                            wraps=self.store.reconciliation,
                        )
                    )
                    with (
                        mock.patch(
                            "src.memory.extractor.get_memory_llm_client",
                            return_value=model,
                        ),
                        store_patch,
                    ):
                        result = self._explicit_memory_command(
                            command,
                            query=secret,
                            renderer=renderer,
                        )

                    self.assertEqual("failed", result.outcome.status)
                    self.assertEqual(cause, result.outcome.cause)
                    self.assertEqual(
                        ("retryable=true",),
                        result.outcome.facts,
                    )
                    self.assertEqual("failed", renderer.facts.status)
                    self.assertEqual(cause, renderer.facts.cause)
                    self.assertEqual(
                        ("retryable=true",),
                        renderer.facts.details,
                    )
                    self.assertNotIn(
                        secret,
                        repr(
                            (
                                result.outcome,
                                renderer.facts,
                                result.reply,
                            )
                        ),
                    )

    def test_explicit_memory_success_and_rejection_facts_never_include_value(self):
        secret = "USER-MEMORY-VALUE-8c02e1"

        class Renderer:
            def __init__(self):
                self.facts = None

            def render(self, facts, fallback):
                self.facts = facts
                return fallback

        class ValidExtractorModel:
            def chat(self, _messages, **_kwargs):
                return MemoryCommandsTests._valid_extractor_response(secret)

        for command in ("remember", "globalremember"):
            with self.subTest(command=command, outcome="success"):
                renderer = Renderer()
                with mock.patch(
                    "src.memory.extractor.get_memory_llm_client",
                    return_value=ValidExtractorModel(),
                ):
                    result = self._explicit_memory_command(
                        command,
                        query=secret,
                        renderer=renderer,
                    )
                self.assertEqual("applied", result.outcome.status)
                self.assertNotIn(secret, repr(result.outcome.facts))
                self.assertNotIn(secret, repr(renderer.facts))
                self.assertTrue(
                    all(
                        fact.startswith("claim_id=")
                        for fact in result.outcome.facts
                    )
                )

            with self.subTest(command=command, outcome="missing_message_id"):
                renderer = Renderer()
                is_global = command == "globalremember"
                result = handle_command(
                    Route(
                        handler="command",
                        action="command",
                        command=command,
                        query=secret,
                    ),
                    CommandContext(
                        uid="9001" if is_global else "1001",
                        session_key=(
                            "private:9001"
                            if is_global
                            else "private:1001"
                        ),
                        raw_message=f"/{command} {secret}",
                        is_admin=is_global,
                    ),
                    store=self.store,
                    renderer=renderer,
                )
                self.assertEqual("rejected", result.outcome.status)
                self.assertEqual((), result.outcome.facts)
                self.assertNotIn(secret, repr(renderer.facts))

    def test_remember_uses_private_or_current_group_scope(self):
        candidate = CandidateClaim(
            subject_ref="speaker",
            predicate="response_style",
            value="简洁",
            memory_type="preference",
            modality="asserted",
            confidence="high",
        )
        route = Route(
            handler="command",
            action="command",
            command="remember",
            query="我喜欢简洁回答",
        )

        with mock.patch(
            "src.commands.MemoryExtractor.extract",
            return_value=(candidate,),
        ):
            handle_command(
                route,
                CommandContext(
                    uid="1001",
                    session_key="private:1001",
                    raw_message="/remember 我喜欢简洁回答",
                    message_id="private-command-1",
                ),
                store=self.store,
            )
            handle_command(
                route,
                CommandContext(
                    uid="1001",
                    session_key="group:2001:1001",
                    raw_message="/remember 我喜欢简洁回答",
                    message_id="group-command-1",
                ),
                store=self.store,
            )

        private_claims = self.store.find_claims_exact(
            scope_type="private",
            scope_id="1001",
        )
        group_claims = self.store.find_claims_exact(
            scope_type="group",
            scope_id="2001",
        )
        self.assertEqual(["private-command-1"], [c.source_message_id for c in private_claims])
        self.assertEqual(["group-command-1"], [c.source_message_id for c in group_claims])

    def test_admin_globalremember_creates_global_claim_without_private_claim(self):
        candidate = CandidateClaim(
            subject_ref="speaker",
            predicate="global_fact",
            value="维护窗口是周日",
            memory_type="fact",
            modality="asserted",
            confidence="high",
        )
        route = Route(
            handler="command",
            action="command",
            command="globalremember",
            query="维护窗口是周日",
        )
        context = CommandContext(
            uid="9001",
            session_key="private:9001",
            raw_message="/globalremember 维护窗口是周日",
            message_id="global-command-1",
            is_admin=True,
        )

        with mock.patch(
            "src.commands.MemoryExtractor.extract",
            return_value=(candidate,),
        ):
            result = handle_command(route, context, store=self.store)

        global_claims = self.store.find_claims_exact(
            scope_type="global",
            scope_id="global",
        )
        private_claims = self.store.find_claims_exact(
            scope_type="private",
            scope_id="9001",
        )
        self.assertEqual(1, len(global_claims))
        self.assertEqual("9001", global_claims[0].speaker_qq)
        self.assertEqual("9001", global_claims[0].subject_id)
        self.assertEqual("global-command-1", global_claims[0].source_message_id)
        self.assertEqual((), private_claims)
        self.assertEqual("applied", result.outcome.status)
        self.assertEqual("global:global", result.outcome.scope)

    def test_real_command_message_ids_preserve_distinct_confirmation_evidence(self):
        candidate = CandidateClaim(
            subject_ref="speaker",
            predicate="likes",
            value="苹果",
            memory_type="preference",
            modality="asserted",
            confidence="high",
        )
        route = Route(
            handler="command",
            action="command",
            command="remember",
            query="我喜欢苹果",
        )
        with mock.patch(
            "src.commands.MemoryExtractor.extract",
            return_value=(candidate,),
        ):
            for message_id in ("command-101", "command-102"):
                handle_command(
                    route,
                    CommandContext(
                        uid="1001",
                        session_key="private:1001",
                        raw_message="/remember 我喜欢苹果",
                        message_id=message_id,
                    ),
                    store=self.store,
                )

        claim = self.store.find_claims_exact(
            scope_type="private",
            scope_id="1001",
        )[0]
        evidence_ids = {claim.source_message_id}
        evidence_ids.update(
            evidence.source_message_id
            for evidence in self.store.list_evidence(claim.id)
        )
        self.assertEqual({"command-101", "command-102"}, evidence_ids)

    def test_policy_rejection_is_not_reported_as_success(self):
        rejected = CandidateClaim(
            subject_ref="speaker",
            predicate="likes",
            value="苹果",
            memory_type="preference",
            modality="asserted",
            confidence="low",
        )
        route = Route(
            handler="command",
            action="command",
            command="remember",
            query="我喜欢苹果",
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message="/remember 我喜欢苹果",
            message_id="rejected-command",
        )

        with mock.patch(
            "src.commands.MemoryExtractor.extract",
            return_value=(rejected,),
        ):
            result = handle_command(route, context, store=self.store)

        self.assertEqual("rejected", result.outcome.status)
        self.assertEqual("policy_rejected", result.outcome.cause)
        self.assertNotIn("记住了", result.reply)
        self.assertEqual(
            (),
            self.store.find_claims_exact(
                scope_type="private",
                scope_id="1001",
            ),
        )

    def test_remember_calls_real_extractor_contract_with_memory_event(self):
        class Llm:
            def __init__(self):
                self.messages = None

            def chat(self, messages, **_kwargs):
                self.messages = messages
                return ChatResponse(
                    content=json.dumps(
                        {
                            "claims": [
                                {
                                    "subject_ref": "speaker",
                                    "predicate": "likes",
                                    "value": "苹果",
                                    "memory_type": "preference",
                                    "modality": "asserted",
                                    "confidence": "high",
                                    "operation": "add",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

        llm = Llm()
        route = Route(
            handler="command",
            action="command",
            command="remember",
            query="我喜欢苹果",
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message="/remember 我喜欢苹果",
            message_id="real-extractor-command",
        )

        with mock.patch(
            "src.memory.extractor.get_memory_llm_client",
            return_value=llm,
        ):
            result = handle_command(route, context, store=self.store)

        self.assertEqual("applied", result.outcome.status)
        claim = self.store.find_claims_exact(
            scope_type="private",
            scope_id="1001",
        )[0]
        self.assertEqual("real-extractor-command", claim.source_message_id)
        prompt = "\n".join(str(message["content"]) for message in llm.messages)
        self.assertIn('"sender_qq":"1001"', prompt)

    def test_exact_id_outside_top_twelve_retracts_own_group_claim(self):
        target = self.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker="1001",
            subject="1001",
            predicate="favorite_food",
            value="苹果",
        )
        for index in range(15):
            self.create_claim(
                scope_type="group",
                scope_id="2001",
                speaker="1001",
                subject="1001",
                predicate=f"filler_{index}",
                value=f"填充内容 {index}",
            )
        context = CommandContext(
            uid="1001",
            session_key="group:2001:1001",
            raw_message=f"/forget {target.id}",
            message_id="forget-own-group",
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=str(target.id),
        )

        result = handle_command(route, context, store=self.store)

        retained = self.store.get_claim(target.id)
        self.assertIsNotNone(retained)
        self.assertEqual("retracted", retained.status)
        self.assertEqual("retracted", result.outcome.status)
        self.assertEqual("author_withdrawal", result.outcome.cause)

    def test_different_group_speaker_cannot_retract_foreign_claim(self):
        target = self.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker="1002",
            subject="1002",
            predicate="likes",
            value="游泳",
        )
        context = CommandContext(
            uid="1001",
            session_key="group:2001:1001",
            raw_message=f"/forget {target.id}",
            message_id="forget-foreign",
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=str(target.id),
        )

        result = handle_command(route, context, store=self.store)

        self.assertEqual("active", self.store.get_claim(target.id).status)
        self.assertEqual("forbidden", result.outcome.status)
        self.assertEqual("foreign_author", result.outcome.cause)

    def test_group_claim_subject_creates_answer_suppression_not_retraction(self):
        target = self.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker="1002",
            subject="1001",
            predicate="likes",
            value="游泳",
        )
        memory_context = MemoryContext(
            user_id="1001",
            session_key="group:2001:1001",
            is_group=True,
            group_id="2001",
        )
        context = CommandContext(
            uid="1001",
            session_key=memory_context.session_key,
            raw_message=f"/forget {target.id}",
            memory_context=memory_context,
            message_id="forget-as-subject",
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=str(target.id),
        )

        result = handle_command(route, context, store=self.store)

        original = self.store.get_claim(target.id)
        self.assertEqual("active", original.status)
        self.assertNotEqual("retracted", original.status)
        self.assertEqual("disputed", result.outcome.status)
        self.assertEqual("subject_dispute", result.outcome.cause)
        answer_ids = {
            item.claim.id
            for item in MemoryRetriever(self.store).retrieve(
                memory_context,
                "游泳",
            )
        }
        self.assertNotIn(target.id, answer_ids)

    def test_group_claim_subject_can_suppress_current_disputed_claim(self):
        target = self.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker="1002",
            subject="1001",
            predicate="likes",
            value="已存在其他冲突",
            status="disputed",
        )
        memory_context = MemoryContext(
            user_id="1001",
            session_key="group:2001:1001",
            is_group=True,
            group_id="2001",
        )
        context = CommandContext(
            uid="1001",
            session_key=memory_context.session_key,
            raw_message=f"/forget {target.id}",
            memory_context=memory_context,
            message_id="forget-disputed-as-subject",
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=str(target.id),
        )

        result = handle_command(route, context, store=self.store)

        self.assertEqual("disputed", result.outcome.status)
        self.assertEqual("subject_dispute", result.outcome.cause)
        self.assertEqual("disputed", self.store.get_claim(target.id).status)
        self.assertIn(
            target.id,
            self.store.subject_dispute_suppressed_ids((target.id,)),
        )
        answer_ids = {
            item.claim.id
            for item in MemoryRetriever(self.store).retrieve(
                memory_context,
                "已存在其他冲突",
            )
        }
        self.assertNotIn(target.id, answer_ids)

    def test_subject_dispute_rejects_noncurrent_group_claim_statuses(self):
        for status in ("retracted", "superseded", "archived"):
            with self.subTest(status=status):
                target = self.create_claim(
                    scope_type="group",
                    scope_id="2001",
                    speaker="1002",
                    subject="1001",
                    predicate=f"status-{status}",
                    value=f"不可争议-{status}",
                    status=status,
                )

                disputed = self.store.register_subject_dispute(
                    target.id,
                    actor_qq="1001",
                    group_id="2001",
                    source_message_id=f"dispute-{status}",
                )

                self.assertIsNone(disputed)
                self.assertEqual(status, self.store.get_claim(target.id).status)
                self.assertNotIn(
                    target.id,
                    self.store.subject_dispute_suppressed_ids((target.id,)),
                )

    def test_committed_author_retraction_cannot_be_overwritten_by_subject_dispute(self):
        target = self.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker="1002",
            subject="1001",
            predicate="likes",
            value="先撤回后争议",
        )

        self.assertTrue(
            self.store.retract_group_claim(
                target.id,
                actor_qq="1002",
                group_id="2001",
            )
        )
        disputed = self.store.register_subject_dispute(
            target.id,
            actor_qq="1001",
            group_id="2001",
            source_message_id="late-subject-dispute",
        )

        self.assertIsNone(disputed)
        self.assertEqual("retracted", self.store.get_claim(target.id).status)
        self.assertNotIn(
            target.id,
            self.store.subject_dispute_suppressed_ids((target.id,)),
        )

    def test_author_retraction_and_subject_dispute_race_ends_retracted(self):
        target = self.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker="1002",
            subject="1001",
            predicate="likes",
            value="并发撤回与争议",
        )
        barrier = threading.Barrier(2)

        def retract():
            barrier.wait()
            return self.store.retract_group_claim(
                target.id,
                actor_qq="1002",
                group_id="2001",
            )

        def dispute():
            barrier.wait()
            return self.store.register_subject_dispute(
                target.id,
                actor_qq="1001",
                group_id="2001",
                source_message_id="concurrent-subject-dispute",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            retraction = pool.submit(retract)
            subject_dispute = pool.submit(dispute)
            self.assertTrue(retraction.result())
            subject_dispute.result()

        self.assertEqual("retracted", self.store.get_claim(target.id).status)

    def test_admin_delete_after_subject_dispute_removes_every_body_copy(self):
        marker = "争议后删除正文-marker-918d"
        target = self.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker="1002",
            subject="1001",
            predicate="likes",
            value=marker,
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=str(target.id),
        )
        handle_command(
            route,
            CommandContext(
                uid="1001",
                session_key="group:2001:1001",
                raw_message=f"/forget {target.id}",
                message_id="subject-dispute-before-delete",
            ),
            store=self.store,
        )

        result = handle_command(
            route,
            CommandContext(
                uid="9001",
                session_key="group:2001:9001",
                raw_message=f"/forget {target.id}",
                message_id="admin-delete-after-dispute",
                is_admin=True,
            ),
            store=self.store,
        )

        self.assertEqual("deleted", result.outcome.status)
        self.assertEqual((), self.store.search_claims(marker))
        connection = sqlite3.connect(self.db_path)
        try:
            body_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM memory_claims
                WHERE value = ? OR source_excerpt = ?
                """,
                (marker, marker),
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(0, body_count)

    def test_group_author_retraction_is_one_conditional_store_mutation(self):
        target = self.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker="1001",
            subject="1001",
            predicate="likes",
            value="原子撤回",
        )

        self.assertFalse(
            self.store.retract_group_claim(
                target.id,
                actor_qq="1002",
                group_id="2001",
            )
        )
        self.assertTrue(
            self.store.retract_group_claim(
                target.id,
                actor_qq="1001",
                group_id="2001",
            )
        )
        self.assertFalse(
            self.store.retract_group_claim(
                target.id,
                actor_qq="1001",
                group_id="2001",
            )
        )
        self.assertEqual("retracted", self.store.get_claim(target.id).status)

    def test_private_owner_forget_physically_deletes_body_and_keeps_audit_only(self):
        marker = "private-delete-marker-7f23"
        target = self.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker="1001",
            subject="1001",
            predicate="likes",
            value=marker,
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message=f"/forget {target.id}",
            message_id="forget-private",
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=str(target.id),
        )

        result = handle_command(route, context, store=self.store)

        self.assertIsNone(self.store.get_claim(target.id))
        self.assertEqual((), self.store.search_claims(marker))
        self.assertEqual("deleted", result.outcome.status)
        self.assertEqual("private_privacy_delete", result.outcome.cause)
        connection = sqlite3.connect(self.db_path)
        try:
            audit = connection.execute(
                """
                SELECT claim_id, reason
                FROM memory_deletion_audit
                WHERE claim_id = ?
                """,
                (target.id,),
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual((target.id, "private_privacy_delete"), audit)

    def test_private_owner_exact_id_can_physically_delete_every_lifecycle(self):
        for status in (
            "active",
            "disputed",
            "retracted",
            "superseded",
            "archived",
        ):
            with self.subTest(status=status):
                target = self.create_claim(
                    scope_type="private",
                    scope_id="1001",
                    speaker="1001",
                    subject="1001",
                    predicate=f"delete-{status}",
                    value=f"private lifecycle {status}",
                    status=status,
                )
                result = handle_command(
                    Route(
                        handler="command",
                        action="command",
                        command="forget",
                        query=str(target.id),
                    ),
                    CommandContext(
                        uid="1001",
                        session_key="private:1001",
                        raw_message=f"/forget {target.id}",
                        message_id=f"forget-{status}",
                    ),
                    store=self.store,
                )

                self.assertEqual("deleted", result.outcome.status)
                self.assertIsNone(self.store.get_claim(target.id))

    def test_admin_exact_id_can_delete_archived_claim_outside_current_scope(self):
        target = self.create_claim(
            scope_type="group",
            scope_id="other-group",
            speaker="1002",
            subject="1002",
            predicate="archived-admin-delete",
            value="archived administrator target",
            status="archived",
        )

        result = handle_command(
            Route(
                handler="command",
                action="command",
                command="forget",
                query=str(target.id),
            ),
            CommandContext(
                uid="9001",
                session_key="private:9001",
                raw_message=f"/forget {target.id}",
                message_id="admin-forget-archived",
                is_admin=True,
            ),
            store=self.store,
        )

        self.assertEqual("deleted", result.outcome.status)
        self.assertIsNone(self.store.get_claim(target.id))

    def test_physical_delete_cleanup_failure_reports_partial_and_is_retryable(self):
        target = self.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker="1001",
            subject="1001",
            predicate="likes",
            value="维护失败后不可谎报成功",
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message=f"/forget {target.id}",
            message_id="forget-partial-cleanup",
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=str(target.id),
        )

        with mock.patch.object(
            self.store,
            "_checkpoint_wal",
            side_effect=RuntimeError("active reader"),
        ):
            result = handle_command(route, context, store=self.store)

        self.assertIsNone(self.store.get_claim(target.id))
        self.assertEqual("partial", result.outcome.status)
        self.assertEqual("privacy_cleanup_pending", result.outcome.cause)
        self.assertNotIn("status=deleted", result.reply)
        with closing(sqlite3.connect(self.db_path)) as connection:
            pending = connection.execute(
                """
                SELECT reason, scope_type, scope_id, needs_fts_optimize
                FROM memory_pending_privacy_cleanup
                WHERE claim_id = ?
                """,
                (target.id,),
            ).fetchone()
            pending_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(memory_pending_privacy_cleanup)"
                ).fetchall()
            }
        self.assertEqual(
            ("private_privacy_delete", "private", "1001", 0),
            pending,
        )
        self.assertTrue(
            {"predicate", "value", "source_excerpt"}.isdisjoint(
                pending_columns
            )
        )

        retry_store = MemoryStore(self.db_path)
        retry_store.initialize()
        replacement = self.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker="1001",
            subject="1001",
            predicate="likes",
            value="删除后新建记录不可复用旧ID",
        )
        self.assertNotEqual(target.id, replacement.id)
        foreign_result = handle_command(
            route,
            CommandContext(
                uid="1002",
                session_key="private:1002",
                raw_message=f"/forget {target.id}",
                message_id="foreign-cleanup-retry",
            ),
            store=retry_store,
        )
        self.assertEqual("not_found", foreign_result.outcome.status)

        retry_result = handle_command(route, context, store=retry_store)
        self.assertEqual("cleanup_completed", retry_result.outcome.status)
        self.assertEqual(
            "privacy_cleanup_completed",
            retry_result.outcome.cause,
        )
        self.assertIn("retryable=false", retry_result.outcome.facts)
        self.assertIsNotNone(retry_store.get_claim(replacement.id))
        with closing(sqlite3.connect(self.db_path)) as connection:
            pending_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM memory_pending_privacy_cleanup
                WHERE claim_id = ?
                """,
                (target.id,),
            ).fetchone()[0]
        self.assertEqual(0, pending_count)

    def test_admin_private_partial_cleanup_can_be_retried_by_owner(self):
        target = self.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker="1001",
            subject="1001",
            predicate="likes",
            value="管理员删除后所有者可重试清理",
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=str(target.id),
        )
        admin_context = CommandContext(
            uid="9001",
            session_key="private:9001",
            raw_message=f"/forget {target.id}",
            message_id="admin-private-partial",
            is_admin=True,
        )

        with mock.patch.object(
            self.store,
            "_checkpoint_wal",
            side_effect=RuntimeError("active reader"),
        ):
            admin_result = handle_command(
                route,
                admin_context,
                store=self.store,
            )

        self.assertEqual("partial", admin_result.outcome.status)
        self.assertIsNone(self.store.get_claim(target.id))
        owner_result = handle_command(
            route,
            CommandContext(
                uid="1001",
                session_key="private:1001",
                raw_message=f"/forget {target.id}",
                message_id="owner-private-cleanup-retry",
            ),
            store=MemoryStore(self.db_path),
        )
        self.assertEqual("cleanup_completed", owner_result.outcome.status)
        self.assertEqual(
            "privacy_cleanup_completed",
            owner_result.outcome.cause,
        )

    def test_precommit_database_lock_returns_store_unavailable_outcome(self):
        target = self.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker="1001",
            subject="1001",
            predicate="likes",
            value="提交前锁冲突",
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message=f"/forget {target.id}",
            message_id="forget-precommit-lock",
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=str(target.id),
        )
        locker = sqlite3.connect(self.db_path)
        locker.execute("BEGIN IMMEDIATE")
        original_connect = sqlite3.connect

        def short_timeout_connect(*args, **kwargs):
            kwargs["timeout"] = 0.01
            return original_connect(*args, **kwargs)

        try:
            with mock.patch.object(
                sqlite3,
                "connect",
                side_effect=short_timeout_connect,
            ):
                result = handle_command(route, context, store=self.store)
        finally:
            locker.rollback()
            locker.close()

        self.assertEqual("failed", result.outcome.status)
        self.assertEqual("store_unavailable", result.outcome.cause)
        self.assertIn("retryable=true", result.outcome.facts)
        self.assertIsNotNone(self.store.get_claim(target.id))

    def test_forget_get_claim_error_returns_persona_aware_store_unavailable(self):
        class Renderer:
            def __init__(self):
                self.facts = None

            def render(self, facts, fallback):
                self.facts = facts
                return f"persona::{fallback}"

        renderer = Renderer()
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query="42",
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message="/forget 42",
            message_id="forget-get-claim-error",
        )

        with mock.patch.object(
            self.store,
            "get_claim",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            result = handle_command(
                route,
                context,
                store=self.store,
                renderer=renderer,
            )

        self.assertEqual("failed", result.outcome.status)
        self.assertEqual("store_unavailable", result.outcome.cause)
        self.assertIn("retryable=true", result.outcome.facts)
        self.assertTrue(result.reply.startswith("persona::"))
        self.assertEqual("failed", renderer.facts.status)

    def test_forget_pending_lookup_error_returns_persona_aware_store_unavailable(self):
        class Renderer:
            def __init__(self):
                self.facts = None

            def render(self, facts, fallback):
                self.facts = facts
                return f"persona::{fallback}"

        renderer = Renderer()
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query="4242",
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message="/forget 4242",
            message_id="forget-pending-lookup-error",
        )

        with mock.patch.object(
            self.store,
            "retry_pending_delete_cleanup",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            result = handle_command(
                route,
                context,
                store=self.store,
                renderer=renderer,
            )

        self.assertEqual("failed", result.outcome.status)
        self.assertEqual("store_unavailable", result.outcome.cause)
        self.assertIn("retryable=true", result.outcome.facts)
        self.assertIn(
            "本次未能确认或执行记忆变更",
            result.outcome.fallback_reply,
        )
        self.assertTrue(result.reply.startswith("persona::"))
        self.assertEqual("failed", renderer.facts.status)

    def test_forget_description_lookup_error_returns_persona_aware_store_unavailable(self):
        class Renderer:
            def __init__(self):
                self.facts = None

            def render(self, facts, fallback):
                self.facts = facts
                return f"persona::{fallback}"

        target = self.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker="1001",
            subject="1001",
            predicate="likes",
            value="描述查询故障保护",
        )
        renderer = Renderer()
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query="查询故障",
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message="/forget 查询故障",
            message_id="forget-description-lookup-error",
        )

        with mock.patch.object(
            self.store,
            "find_claims_exact",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            result = handle_command(
                route,
                context,
                store=self.store,
                renderer=renderer,
            )

        self.assertEqual("failed", result.outcome.status)
        self.assertEqual("store_unavailable", result.outcome.cause)
        self.assertIn("retryable=true", result.outcome.facts)
        self.assertTrue(result.reply.startswith("persona::"))
        self.assertEqual("failed", renderer.facts.status)
        self.assertIsNotNone(self.store.get_claim(target.id))

    def test_forget_description_error_never_exposes_raw_target(self):
        secret = "SECRET-FORGET-TARGET-7f3c91"

        class Renderer:
            def __init__(self):
                self.facts = None
                self.fallback = None

            def render(self, facts, fallback):
                self.facts = facts
                self.fallback = fallback
                return f"persona::{fallback}"

        class RecordingHandler(logging.Handler):
            def __init__(self):
                super().__init__()
                self.messages = []

            def emit(self, record):
                self.messages.append(self.format(record))

        renderer = Renderer()
        log_handler = RecordingHandler()
        command_logger = logging.getLogger("qq-bot")
        command_logger.addHandler(log_handler)
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=f"请忘记 {secret}",
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message=f"/forget 请忘记 {secret}",
            message_id="forget-secret-description-error",
        )

        try:
            with mock.patch.object(
                self.store,
                "find_claims_exact",
                side_effect=sqlite3.OperationalError("database is locked"),
            ):
                result = handle_command(
                    route,
                    context,
                    store=self.store,
                    renderer=renderer,
                )
        finally:
            command_logger.removeHandler(log_handler)

        self.assertEqual("failed", result.outcome.status)
        self.assertEqual("store_unavailable", result.outcome.cause)
        self.assertEqual("failed", renderer.facts.status)
        self.assertEqual("private:1001", renderer.facts.scope)
        self.assertEqual("store_unavailable", renderer.facts.cause)
        leak_surfaces = {
            "fallback": result.outcome.fallback_reply,
            "outcome facts": repr(result.outcome.facts),
            "final reply": result.reply,
            "renderer facts": repr(renderer.facts),
            "renderer fallback": renderer.fallback,
            "logs": "\n".join(log_handler.messages),
        }
        for surface, text in leak_surfaces.items():
            with self.subTest(surface=surface):
                self.assertNotIn(secret, text)
        self.assertEqual(("retryable=true",), result.outcome.facts)
        self.assertEqual(("retryable=true",), renderer.facts.details)
        self.assertIn("内容描述", result.outcome.fallback_reply)

    def test_operational_error_after_delete_commit_reports_partial_and_retries(self):
        target = self.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker="1001",
            subject="1001",
            predicate="likes",
            value="提交后数据库清理锁定",
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message=f"/forget {target.id}",
            message_id="forget-operational-cleanup",
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=str(target.id),
        )

        with mock.patch.object(
            self.store,
            "_checkpoint_wal",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            result = handle_command(route, context, store=self.store)

        self.assertIsNone(self.store.get_claim(target.id))
        self.assertEqual("partial", result.outcome.status)
        self.assertEqual("privacy_cleanup_pending", result.outcome.cause)
        self.assertIn("retryable=true", result.outcome.facts)

        retry_store = MemoryStore(self.db_path)
        retry_store.initialize()
        with mock.patch.object(
            retry_store,
            "_checkpoint_wal",
            side_effect=sqlite3.OperationalError("database is still locked"),
        ):
            retry_pending = retry_store.delete_claim_physically_with_outcome(
                target.id,
                reason="private_privacy_delete",
                actor_qq="1001",
                is_admin=False,
            )
        self.assertEqual("partial", retry_pending.status)
        self.assertTrue(retry_pending.row_deleted)
        self.assertTrue(retry_pending.retryable)

        retry_result = handle_command(route, context, store=retry_store)
        self.assertEqual("cleanup_completed", retry_result.outcome.status)
        self.assertEqual(
            "privacy_cleanup_completed",
            retry_result.outcome.cause,
        )
        self.assertIn("retryable=false", retry_result.outcome.facts)

    def test_admin_forget_physically_deletes_group_claim(self):
        target = self.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker="1002",
            subject="1002",
            predicate="likes",
            value="管理员删除目标",
        )
        context = CommandContext(
            uid="9001",
            session_key="group:2001:9001",
            raw_message=f"/forget {target.id}",
            message_id="forget-admin",
            is_admin=True,
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query=str(target.id),
        )

        result = handle_command(route, context, store=self.store)

        self.assertIsNone(self.store.get_claim(target.id))
        self.assertEqual("deleted", result.outcome.status)
        self.assertEqual("administrator_delete", result.outcome.cause)

    def test_natural_description_requires_one_permitted_match(self):
        first = self.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker="1001",
            subject="1001",
            predicate="likes",
            value="苹果派",
        )
        second = self.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker="1001",
            subject="1001",
            predicate="dislikes",
            value="苹果汁",
        )
        context = CommandContext(
            uid="1001",
            session_key="private:1001",
            raw_message="/forget 苹果",
            message_id="forget-ambiguous",
        )
        route = Route(
            handler="command",
            action="command",
            command="forget",
            query="苹果",
        )

        result = handle_command(route, context, store=self.store)

        self.assertEqual("ambiguous", result.outcome.status)
        self.assertEqual("active", self.store.get_claim(first.id).status)
        self.assertEqual("active", self.store.get_claim(second.id).status)

    def test_group_memories_never_lists_private_personalization_value(self):
        private_marker = "私密称呼-不可公开-3d81"
        self.create_claim(
            scope_type="private",
            scope_id="1001",
            speaker="1001",
            subject="1001",
            predicate="preferred_name",
            value=private_marker,
        )
        context = CommandContext(
            uid="1001",
            session_key="group:2001:1001",
            raw_message="/memories",
            message_id="list-group",
        )
        route = Route(
            handler="command",
            action="command",
            command="memories",
            query="",
        )

        result = handle_command(route, context, store=self.store)

        self.assertNotIn(private_marker, result.reply)

    def test_group_memories_listing_is_not_starved_by_private_top_results(self):
        group_claim = self.create_claim(
            scope_type="group",
            scope_id="2001",
            speaker="1002",
            subject="1002",
            predicate="group_fact",
            value="GROUP-LISTING-SURVIVES",
        )
        self.store.update_claim(
            group_claim.id,
            truth_confidence="low",
            last_confirmed_at="2010-01-01T00:00:00+00:00",
        )
        for index in range(20):
            self.create_claim(
                scope_type="private",
                scope_id="1001",
                speaker="1001",
                subject="1001",
                predicate="preferred_name",
                value=f"昵称{index}",
            )
        context = CommandContext(
            uid="1001",
            session_key="group:2001:1001",
            raw_message="/memories",
            message_id="list-not-starved",
        )

        result = handle_command(
            Route(
                handler="command",
                action="command",
                command="memories",
                query="",
            ),
            context,
            store=self.store,
        )

        self.assertEqual("listed", result.outcome.status)
        self.assertIn("GROUP-LISTING-SURVIVES", result.reply)
        self.assertNotIn("昵称0", result.reply)


if __name__ == "__main__":
    unittest.main()
