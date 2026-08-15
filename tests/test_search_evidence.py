"""Evidence tests: relevance-gated admission, dedup, conflicts, sufficiency."""

from __future__ import annotations

import importlib
import json
import unittest
from unittest import mock
from dataclasses import replace
from datetime import date, datetime, timezone
from itertools import permutations

from src.search.models import (
    EvidenceCandidate,
    EvidenceGapAnalysis,
    EvidenceState,
    ExcerptOrigin,
    Factuality,
    FetchedDocument,
    Freshness,
    FreshnessEligibility,
    FreshnessRequirement,
    JudgeBatchResult,
    JudgeBatchStatus,
    JudgeVerdict,
    PlanningStatus,
    ProviderHit,
    QueryPurpose,
    RepairReasonCode,
    RequiredTopic,
    RequestSource,
    RetrievalDecision,
    RetrievalRequest,
    RiskLevel,
    SearchPlan,
    SearchQuery,
    SearchRoundKind,
    SearchTier,
    SourceRelation,
    SourceRequirement,
)
from tests.search_fakes import StaticEvidenceJudge


def evidence_module():
    try:
        return importlib.import_module("src.search.evidence")
    except ModuleNotFoundError:
        raise AssertionError("src.search.evidence must exist") from None


def decision(tier=SearchTier.STANDARD):
    m = __import__("src.search.models", fromlist=["RetrievalDecision"])
    return m.RetrievalDecision(
        route=tier, skip_reason=None, must_search=True, reason_codes=(),
    )


def query(qid="initial-1", text="什么是光合作用", targets=("topic-1",)):
    return SearchQuery(
        qid,
        SearchRoundKind.INITIAL,
        QueryPurpose.DIRECT,
        text,
        query_index=1,
        target_topic_ids=targets,
    )


def plan(required_topics=("定义",), route=SearchTier.STANDARD):
    d = decision(route)
    topics = tuple(
        RequiredTopic(
            f"topic-{index}",
            label,
            True,
            FreshnessRequirement.NOT_REQUIRED,
            source_requirement=SourceRequirement.ANY_RELEVANT,
        )
        for index, label in enumerate(required_topics, 1)
    ) if all(type(value) is str for value in required_topics) else tuple(required_topics)
    material_ids = tuple(value.topic_id for value in topics if value.material)
    return SearchPlan(
        d, "什么是光合作用", __import__("src.search.models", fromlist=["PlanningStatus"]).PlanningStatus.NORMAL,
        ("光合作用",), None, (query(targets=material_ids),), topics,
        frozenset({SourceRelation.PRIMARY, SourceRelation.INDEPENDENT}), (), _budget(route),
    )


def material_topic_plan():
    d = decision(SearchTier.STANDARD)
    topics = (
        RequiredTopic(
            "topic-1",
            "background",
            False,
            FreshnessRequirement.NOT_REQUIRED,
            source_requirement=SourceRequirement.ANY_RELEVANT,
        ),
        RequiredTopic(
            "topic-2",
            "core",
            True,
            FreshnessRequirement.NOT_REQUIRED,
            source_requirement=SourceRequirement.ANY_RELEVANT,
        ),
    )
    direct = SearchQuery(
        "initial-1",
        SearchRoundKind.INITIAL,
        QueryPurpose.DIRECT,
        "比较并发 API",
        query_index=1,
        target_topic_ids=("topic-2",),
    )
    return SearchPlan(
        d,
        "比较并发 API",
        PlanningStatus.NORMAL,
        (),
        None,
        (direct,),
        topics,
        frozenset({SourceRelation.PRIMARY, SourceRelation.INDEPENDENT}),
        (),
        _budget(SearchTier.STANDARD),
    )


def topic(
    topic_id,
    label,
    freshness_requirement=FreshnessRequirement.NOT_REQUIRED,
    *,
    material=True,
    date_from=None,
    date_to=None,
    version_constraint=None,
    source_requirement=SourceRequirement.ANY_RELEVANT,
):
    return RequiredTopic(
        topic_id,
        label,
        material,
        freshness_requirement,
        date_from=date_from,
        date_to=date_to,
        version_constraint=version_constraint,
        source_requirement=source_requirement,
    )


def topic_plan(*topics):
    d = decision(SearchTier.STANDARD)
    material_ids = tuple(item.topic_id for item in topics if item.material)
    direct = SearchQuery(
        "initial-1",
        SearchRoundKind.INITIAL,
        QueryPurpose.DIRECT,
        "比较当前主题",
        query_index=1,
        target_topic_ids=material_ids,
    )
    return SearchPlan(
        d,
        "比较当前主题",
        PlanningStatus.NORMAL,
        (),
        None,
        (direct,),
        tuple(topics),
        frozenset(),
        (),
        _budget(SearchTier.STANDARD),
    )


def corroborated_topic_plan(label="dynamic"):
    return topic_plan(
        topic(
            "topic-1",
            label,
            source_requirement=SourceRequirement.INDEPENDENT_CORROBORATION,
        )
    )


def current_topic_plan(label="current status"):
    return topic_plan(
        topic(
            "topic-1",
            label,
            FreshnessRequirement.CURRENT,
        )
    )


def topic_judge_ok(
    candidate_id="C1",
    *,
    supported_topic_ids=("topic-1",),
    freshness_by_topic=None,
    relation="primary",
    conflict_key=None,
    conflict_value=None,
    conflict_relation=None,
):
    if freshness_by_topic is None:
        freshness_by_topic = {
            topic_id: "not_required" for topic_id in supported_topic_ids
        }
    if conflict_key is not None and conflict_relation is None:
        conflict_relation = "contradicts"
    return {
        "candidate_id": candidate_id,
        "source_relation": relation,
        "publisher_entity_match": relation == "primary",
        "ownership_basis": "publisher matches query entity" if relation == "primary" else None,
        "publisher": None,
        "supported_topic_ids": list(supported_topic_ids),
        "freshness_by_topic": dict(freshness_by_topic),
        "conflict_key": conflict_key,
        "conflict_value": conflict_value,
        "conflict_relation": conflict_relation,
    }


def _budget(route):
    m = __import__("src.search.models", fromlist=["DEFAULT_TIER_BUDGETS"])
    return m.DEFAULT_TIER_BUDGETS[route]


def hit(url="https://example.com/page", title="Title", provider="tavily", content=None, published=None):
    return ProviderHit(
        provider=provider,
        query_id="q1",
        title=title,
        url=url,
        snippet=content,
        score=1.0,
        published_at=published,
        raw_content=None,
        quality_flags=(),
    )


def candidate(
    url="https://example.com/page",
    title="Title",
    provider="tavily",
    content="直接回答光合作用定义的正文。",
    published=None,
):
    return EvidenceCandidate(
        hit=hit(url=url, title=title, provider=provider, content=content, published=published),
        document=None,
        excerpt=content,
        excerpt_origin=ExcerptOrigin.PROVIDER_SNIPPET,
        extraction_status="search_result_snippet",
        safety_flags=(),
        content_reads_consumed=0,
    )


def judge_ok(
    candidate_id="C1",
    relation="primary",
    supported=("topic-1",),
    freshness_by_topic=None,
    conflict_key=None,
    conflict_value=None,
    conflict_relation=None,
    publisher=None,
):
    # Existing test call sites name legacy topic labels.  This fixture emits
    # only closed topic IDs; the two explicit material-topic fixtures retain
    # their stable IDs while all single-topic legacy plans use topic-1.
    topic_ids = tuple(
        dict.fromkeys(
            {"background": "topic-1", "core": "topic-2"}.get(
                supported_topic,
                "topic-1",
            )
            for supported_topic in supported
        )
    ) if supported else ()
    if freshness_by_topic is None:
        freshness_by_topic = {
            topic_id: "not_required" for topic_id in topic_ids
        }
    if conflict_key is not None and conflict_relation is None:
        conflict_relation = "contradicts"
    return {
        "candidate_id": candidate_id,
        "source_relation": relation,
        "publisher_entity_match": relation == "primary",
        "ownership_basis": "publisher matches query entity" if relation == "primary" else None,
        "publisher": publisher,
        "supported_topic_ids": list(topic_ids),
        "freshness_by_topic": dict(freshness_by_topic),
        "conflict_key": conflict_key,
        "conflict_value": conflict_value,
        "conflict_relation": conflict_relation,
    }


class EvidenceAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    def test_topic_support_is_admission_gate_before_source_relation(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", relation="primary", supported=()),
                "C2": judge_ok("C2", relation="independent", supported=("topic-1",)),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        candidates = (
            candidate(url="https://official.example.com", title="Official Homepage"),
            candidate(url="https://independent.example.com/note", title="Independent Release Note"),
        )
        bundle = assembler.assemble(plan(), candidates)
        evidence = bundle.evidence_items
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].url, "https://independent.example.com/note")
        self.assertIs(evidence[0].source_relation, SourceRelation.INDEPENDENT)
        self.assertEqual(("topic-1",), evidence[0].supported_topic_ids)

    def test_primary_official_docs_url_without_support_cannot_be_admitted(self):
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", relation="primary", supported=())}
        )
        assembler = self.module.EvidenceAssembler(judge)
        candidates = (candidate(url="https://unrelated.example/docs", title="Docs"),)
        bundle = assembler.assemble(plan(), candidates)
        self.assertEqual(bundle.evidence_items, ())
        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)

    def test_contradictory_primary_row_is_rejected(self):
        judge = StaticEvidenceJudge(
            {
                "C1": {
                    "candidate_id": "C1",
                    "source_relation": "primary",
                    "publisher_entity_match": False,
                    "ownership_basis": None,
                    "publisher": None,
                    "supported_topic_ids": ["topic-1"],
                    "freshness_by_topic": {"topic-1": "not_required"},
                    "conflict_key": None,
                    "conflict_value": None,
                    "conflict_relation": None,
                }
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(), (candidate(url="https://docs.example.com"),))
        self.assertEqual(bundle.evidence_items, ())

    def test_docs_path_never_primary_by_shape(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", relation="primary")})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(), (candidate(url="https://x.example.com/docs/guide"),))
        self.assertEqual(len(bundle.evidence_items), 1)

    def test_judge_failure_falls_back_deterministically_to_unknown(self):
        assembler = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": "garbage"})
        )
        bundle = assembler.assemble(plan(), (candidate(content="直接相关正文"),))
        self.assertEqual(bundle.evidence_items, ())
        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)

    def test_empty_support_is_not_admitted(self):
        assembler = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": judge_ok("C1", supported=())})
        )
        bundle = assembler.assemble(plan(), (candidate(content="背景相关但不回答问题"),))
        self.assertEqual(bundle.evidence_items, ())

    def test_closed_non_supported_rows_are_not_admitted_but_do_not_fail_parsing(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok(
                    "C1",
                    relation="unknown",
                    supported=(),
                    freshness_by_topic={},
                ),
                "C2": judge_ok(
                    "C2",
                    relation="unknown",
                    supported=(),
                    freshness_by_topic={},
                ),
                "C3": judge_ok("C3", relation="independent", supported=("topic-1",)),
            }
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(),
            (
                candidate(url="https://irrelevant.example/page"),
                candidate(url="https://contextual.example/page"),
                candidate(url="https://direct.example/page"),
            ),
        )

        self.assertEqual(
            ("https://direct.example/page",),
            tuple(item.url for item in bundle.evidence_items),
        )

    def test_judge_receives_required_topics_and_remaining_time(self):
        calls = []

        class RecordingJudge:
            def judge(self, question, candidates, *, required_topics, timeout_seconds):
                calls.append((question, tuple(required_topics), timeout_seconds, len(candidates)))
                return {"C1": judge_ok("C1")}

        assembler = self.module.EvidenceAssembler(RecordingJudge())
        assembler.assemble(plan(required_topics=("定义", "历史")), (candidate(),), timeout_seconds=0.4)

        self.assertEqual(
            calls,
            [
                (
                    "什么是光合作用",
                    (
                        {"topic_id": "topic-1", "label": "定义"},
                        {"topic_id": "topic-2", "label": "历史"},
                    ),
                    0.4,
                    1,
                )
            ],
        )

    def test_partial_direct_judge_row_is_rejected_as_a_whole(self):
        class PartialJudge:
            def judge(self, *_args, **_kwargs):
                return {
                    "C1": {
                        "source_relation": "independent",
                        "supported_topic_ids": ["topic-1"],
                    }
                }

        bundle = self.module.EvidenceAssembler(PartialJudge()).assemble(
            plan(required_topics=("定义",)), (candidate(),)
        )

        self.assertEqual(bundle.evidence_items, ())
        self.assertEqual(bundle.evidence_state, EvidenceState.INSUFFICIENT)

    def test_missing_or_unknown_judge_field_rejects_the_complete_row(self):
        missing = judge_ok("C1")
        missing.pop("source_relation")
        extra = judge_ok("C2")
        extra["unreviewed_field"] = "must not be ignored"

        bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": missing, "C2": extra})
        ).assemble(
            plan(required_topics=("定义",)),
            (
                candidate(url="https://one.example/item"),
                candidate(url="https://two.example/item"),
            ),
        )

        self.assertEqual(bundle.evidence_items, ())

    def test_malformed_or_contradictory_conflict_contract_rejects_row(self):
        missing_value = judge_ok("C1", conflict_key="version", conflict_value=None)
        value_without_key = judge_ok("C2", conflict_key=None, conflict_value="2.0")

        bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": missing_value, "C2": value_without_key})
        ).assemble(
            plan(required_topics=("版本",)),
            (
                candidate(url="https://one.example/version", content="版本 1.0"),
                candidate(url="https://two.example/version", content="版本 2.0"),
            ),
        )

        self.assertEqual(bundle.evidence_items, ())
        self.assertEqual(bundle.conflicts, ())


class EvidenceJudgeSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    @staticmethod
    def _candidates(count: int):
        return tuple(
            candidate(url=f"https://candidate-{index}.example/release")
            for index in range(1, count + 1)
        )

    def _judge(self, content, *, candidate_count: int):
        class StaticLLM:
            def chat(self, *_args, **_kwargs):
                return type("Response", (), {"content": content})()

        return self.module.LLMEvidenceJudge(StaticLLM()).judge(
            "question",
            self._candidates(candidate_count),
            required_topics=({"topic_id": "topic-1", "label": "release"},),
        )

    def test_supported_topic_ids_are_the_only_direct_support_signal(self):
        row = topic_judge_ok("C1", supported_topic_ids=("topic-1",))
        parsed = self._judge(
            json.dumps({"candidates": {"C1": row}, "gap_hints": []}),
            candidate_count=1,
        )
        self.assertEqual(("topic-1",), tuple(parsed["C1"]["supported_topic_ids"]))

    def test_llm_judge_accepts_only_closed_outer_topic_id_schema(self):
        row = topic_judge_ok(
            "C1",
            freshness_by_topic={"topic-1": "satisfied"},
        )
        calls = []

        class StaticLLM:
            def chat(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return type(
                    "Response",
                    (),
                    {
                        "content": json.dumps(
                            {"candidates": {"C1": row}, "gap_hints": []}
                        )
                    },
                )()

        result = self.module.LLMEvidenceJudge(StaticLLM()).judge(
            "question",
            (candidate(),),
            required_topics=({"topic_id": "topic-1", "label": "release"},),
        )

        self.assertEqual({"C1": row}, result)
        payload = json.loads(calls[0][0][1]["content"])
        self.assertEqual(
            [{"topic_id": "topic-1", "label": "release"}],
            payload["required_topics"],
        )

    def test_llm_judge_rejects_extra_or_partial_closed_schema_rows(self):
        valid = topic_judge_ok("C1")
        invalid_payloads = (
            {"candidates": {"C1": valid}, "gap_hints": [], "extra": True},
            {
                "candidates": {"C1": valid | {"unreviewed_field": True}},
                "gap_hints": [],
            },
            {"candidates": {"C1": {key: value for key, value in valid.items() if key != "candidate_id"}}, "gap_hints": []},
            {"candidates": {"C1": {"candidate_id": "C1", "source_relation": "independent"}}, "gap_hints": []},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                class StaticLLM:
                    def chat(self, *_args, **_kwargs):
                        return type(
                            "Response",
                            (),
                            {"content": json.dumps(payload)},
                        )()

                result = self.module.LLMEvidenceJudge(StaticLLM()).judge(
                    "question",
                    (candidate(),),
                    required_topics=({"topic_id": "topic-1", "label": "release"},),
                )
                self.assertEqual({}, result)

    def test_llm_judge_keeps_valid_rows_when_another_row_fails_closed(self):
        valid = topic_judge_ok("C1")
        invalid = topic_judge_ok("C2") | {"unreviewed_field": True}

        class StaticLLM:
            def chat(self, *_args, **_kwargs):
                return type(
                    "Response",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "candidates": {"C1": valid, "C2": invalid},
                                "gap_hints": [],
                            }
                        )
                    },
                )()

        result = self.module.LLMEvidenceJudge(StaticLLM()).judge(
            "question",
            (candidate(), candidate(url="https://two.example/release")),
            required_topics=({"topic_id": "topic-1", "label": "release"},),
        )

        self.assertEqual({"C1": valid}, result)

    def test_llm_judge_preserves_closed_non_direct_rows_without_anomalies(self):
        irrelevant = topic_judge_ok(
            "C1",
            supported_topic_ids=(),
            freshness_by_topic={},
            relation="unknown",
        )
        contextual = topic_judge_ok(
            "C2",
            supported_topic_ids=(),
            freshness_by_topic={},
            relation="unknown",
        )
        direct = topic_judge_ok("C3")

        result = self._judge(
            json.dumps(
                {
                    "candidates": {
                        "C1": irrelevant,
                        "C2": contextual,
                        "C3": direct,
                    },
                    "gap_hints": [],
                }
            ),
            candidate_count=3,
        )

        self.assertEqual(
            {"C1": irrelevant, "C2": contextual, "C3": direct},
            result,
        )
        self.assertEqual((), result.judge_anomaly_codes)
        self.assertEqual(0, result.judge_anomaly_count)

    def test_llm_judge_preserves_a_complete_five_candidate_batch(self):
        rows = {
            f"C{index}": topic_judge_ok(f"C{index}")
            for index in range(1, 6)
        }

        result = self._judge(
            json.dumps({"candidates": rows, "gap_hints": []}),
            candidate_count=5,
        )

        self.assertEqual(rows, result)

    def test_llm_judge_keeps_partial_valid_rows_and_records_missing_candidates(self):
        first = topic_judge_ok("C1")
        third = topic_judge_ok("C3")

        result = self._judge(
            json.dumps(
                {
                    "candidates": {"C1": first, "C3": third},
                    "gap_hints": [],
                }
            ),
            candidate_count=5,
        )

        self.assertEqual(first, result["C1"])
        self.assertEqual(third, result["C3"])
        self.assertEqual(
            (self.module.JudgeAnomalyCode.MISSING_CANDIDATE,),
            result.judge_anomaly_codes,
        )
        self.assertEqual(3, result.judge_anomaly_count)

    def test_llm_judge_discards_unknown_candidate_without_poisoning_valid_rows(self):
        first = topic_judge_ok("C1")
        unknown = topic_judge_ok("C99")

        result = self._judge(
            json.dumps(
                {
                    "candidates": {"C1": first, "C99": unknown},
                    "gap_hints": [],
                }
            ),
            candidate_count=1,
        )

        self.assertEqual(first, result["C1"])
        self.assertEqual(
            (self.module.JudgeAnomalyCode.UNKNOWN_CANDIDATE,),
            result.judge_anomaly_codes,
        )
        self.assertEqual(1, result.judge_anomaly_count)
        self.assertNotIn("C99", result)

    def test_llm_judge_rejects_only_a_malformed_expected_candidate_row(self):
        first = topic_judge_ok("C1")
        malformed = topic_judge_ok("C2") | {"unreviewed_field": True}

        result = self._judge(
            json.dumps(
                {
                    "candidates": {"C1": first, "C2": malformed},
                    "gap_hints": [],
                }
            ),
            candidate_count=2,
        )

        self.assertEqual(first, result["C1"])
        self.assertEqual(
            (self.module.JudgeAnomalyCode.MALFORMED_CANDIDATE,),
            result.judge_anomaly_codes,
        )
        self.assertEqual(1, result.judge_anomaly_count)

    def test_llm_judge_rejects_only_a_duplicate_expected_candidate_id(self):
        first = json.dumps(topic_judge_ok("C1"), ensure_ascii=False)
        second = json.dumps(topic_judge_ok("C2"), ensure_ascii=False)
        content = (
            '{"candidates":{"C1":' + first + ',"C1":' + first
            + ',"C2":' + second + '},"gap_hints":[]}'
        )

        result = self._judge(content, candidate_count=2)

        self.assertEqual(topic_judge_ok("C2"), result["C2"])
        self.assertNotIn("C1", result)
        self.assertEqual(
            (self.module.JudgeAnomalyCode.DUPLICATE_CANDIDATE,),
            result.judge_anomaly_codes,
        )
        self.assertEqual(1, result.judge_anomaly_count)

    def test_llm_judge_rejects_a_duplicate_nested_candidate_field_only_for_that_row(self):
        first = json.dumps(topic_judge_ok("C1"), ensure_ascii=False, separators=(",", ":"))
        malformed = json.dumps(topic_judge_ok("C2"), ensure_ascii=False, separators=(",", ":"))
        malformed = malformed.replace(
            '"source_relation":"primary"',
            '"source_relation":"primary","source_relation":"secondary"',
            1,
        )
        content = (
            '{"candidates":{"C1":' + first + ',"C2":' + malformed
            + '},"gap_hints":[]}'
        )

        result = self._judge(content, candidate_count=2)

        self.assertEqual(topic_judge_ok("C1"), result["C1"])
        self.assertNotIn("C2", result)
        self.assertEqual(
            (self.module.JudgeAnomalyCode.MALFORMED_CANDIDATE,),
            result.judge_anomaly_codes,
        )
        self.assertEqual(1, result.judge_anomaly_count)

    def test_llm_judge_empty_candidates_are_missing_without_a_judgement(self):
        result = self._judge(
            json.dumps({"candidates": {}, "gap_hints": []}),
            candidate_count=2,
        )

        self.assertEqual({}, result)
        self.assertEqual(
            (self.module.JudgeAnomalyCode.MISSING_CANDIDATE,),
            result.judge_anomaly_codes,
        )
        self.assertEqual(2, result.judge_anomaly_count)

    def test_llm_judge_bounds_missing_candidate_anomaly_count(self):
        result = self._judge(
            json.dumps({"candidates": {}, "gap_hints": []}),
            candidate_count=9,
        )

        self.assertEqual(
            (self.module.JudgeAnomalyCode.MISSING_CANDIDATE,),
            result.judge_anomaly_codes,
        )
        self.assertEqual(8, result.judge_anomaly_count)

    def test_empty_support_with_empty_freshness_is_valid_negative(self):
        row = topic_judge_ok("C1", supported_topic_ids=(), freshness_by_topic={}, relation="unknown")
        result = self._judge(
            json.dumps({"candidates": {"C1": row}, "gap_hints": []}),
            candidate_count=1,
        )
        self.assertEqual(row, result["C1"])
        self.assertIs(result.status, self.module.JudgeBatchStatus.COMPLETED)
        self.assertEqual((), result.judge_anomaly_codes)
        self.assertEqual(0, result.judge_anomaly_count)

    def test_damaged_root_or_call_failure_returns_unavailable(self):
        class FailingLLM:
            def chat(self, *_args, **_kwargs):
                raise RuntimeError("LLM request failed")

        result = self.module.LLMEvidenceJudge(FailingLLM()).judge(
            "question",
            self._candidates(2),
            required_topics=({"topic_id": "topic-1", "label": "release"},),
        )
        self.assertIs(result.status, self.module.JudgeBatchStatus.UNAVAILABLE)
        self.assertEqual({}, dict(result))

    def test_all_negative_rows_completed_without_support(self):
        row1 = topic_judge_ok("C1", supported_topic_ids=(), freshness_by_topic={}, relation="unknown")
        row2 = topic_judge_ok("C2", supported_topic_ids=(), freshness_by_topic={}, relation="unknown")
        result = self._judge(
            json.dumps({"candidates": {"C1": row1, "C2": row2}, "gap_hints": []}),
            candidate_count=2,
        )
        self.assertEqual({"C1": row1, "C2": row2}, result)
        self.assertIs(result.status, self.module.JudgeBatchStatus.COMPLETED)
        self.assertEqual((), result.judge_anomaly_codes)

    def test_valid_root_with_all_malformed_rows_is_completed_with_anomalies(self):
        malformed1 = topic_judge_ok("C1") | {"unreviewed_field": True}
        malformed2 = topic_judge_ok("C2") | {"unreviewed_field": True}
        result = self._judge(
            json.dumps({"candidates": {"C1": malformed1, "C2": malformed2}, "gap_hints": []}),
            candidate_count=2,
        )
        self.assertEqual({}, dict(result))
        self.assertIs(result.status, self.module.JudgeBatchStatus.COMPLETED)
        self.assertEqual(
            (self.module.JudgeAnomalyCode.MALFORMED_CANDIDATE,),
            result.judge_anomaly_codes,
        )
        self.assertEqual(2, result.judge_anomaly_count)

    def test_llm_called_exactly_once_for_candidate_batch(self):
        call_count = 0

        class CountingLLM:
            def chat(self, messages, **kwargs):
                nonlocal call_count
                call_count += 1
                return type(
                    "Response",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "candidates": {
                                    "C1": topic_judge_ok("C1"),
                                    "C2": topic_judge_ok("C2"),
                                },
                                "gap_hints": [],
                            }
                        )
                    },
                )()

        result = self.module.LLMEvidenceJudge(CountingLLM()).judge(
            "question",
            self._candidates(2),
            required_topics=({"topic_id": "topic-1", "label": "release"},),
        )
        self.assertEqual(1, call_count)
        self.assertEqual(2, len(result))


class MaterialTopicEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    def test_nonmaterial_background_does_not_block_material_sufficiency_or_gap(self):
        search_plan = material_topic_plan()
        assembler = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": judge_ok("C1", supported=("core",))})
        )

        bundle = assembler.assemble(search_plan, (candidate(),))
        gap = assembler.analyze_gap(search_plan, bundle)

        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual((), bundle.missing_claim_topics)
        self.assertEqual((), gap.missing_topic_ids)

    def test_missing_material_topic_drives_repair_without_background(self):
        search_plan = material_topic_plan()
        assembler = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": judge_ok("C1", supported=())})
        )

        bundle = assembler.assemble(search_plan, (candidate(),))
        gap = assembler.analyze_gap(search_plan, bundle)
        planner = importlib.import_module("src.search.planner").SearchPlanner(object())
        repair = planner.plan_repair(search_plan, gap)

        self.assertEqual(("core",), bundle.missing_claim_topics)
        self.assertEqual(("topic-2",), gap.missing_topic_ids)
        self.assertEqual(("topic-2",), gap.repair_target_topic_ids)
        self.assertTrue(repair.triggered)
        self.assertIn("core", repair.repair_query.text)
        self.assertNotIn("background", repair.repair_query.text)

    def test_nonmaterial_conflicts_do_not_create_gap_or_repair(self):
        search_plan = material_topic_plan()
        assembler = self.module.EvidenceAssembler(StaticEvidenceJudge({
            "C1": judge_ok(
                "C1", supported=("background",),
                conflict_key="background-version", conflict_value="1",
            ),
            "C2": judge_ok(
                "C2", supported=("background",),
                conflict_key="background-version", conflict_value="2",
            ),
            "C3": judge_ok("C3", supported=("core",)),
        }))

        bundle = assembler.assemble(
            search_plan,
            (
                candidate(url="https://one.example/background"),
                candidate(url="https://two.example/background"),
                candidate(url="https://three.example/core"),
            ),
        )
        gap = assembler.analyze_gap(search_plan, bundle)
        repair = importlib.import_module("src.search.planner").SearchPlanner(
            object()
        ).plan_repair(search_plan, gap)

        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual((), bundle.conflict_groups)
        self.assertFalse(gap.repair_eligible)
        self.assertFalse(repair.triggered)

    def test_repair_ignores_a_nonmaterial_only_missing_topic(self):
        search_plan = material_topic_plan()
        gap = EvidenceGapAnalysis(
            ("topic-1",), (), True, (RepairReasonCode.MISSING_TOPIC,), ("topic-1",),
        )

        repair = importlib.import_module("src.search.planner").SearchPlanner(
            object()
        ).plan_repair(search_plan, gap)

        self.assertFalse(repair.triggered)

    def test_repair_keeps_a_label_shared_by_material_and_nonmaterial_topics(self):
        search_plan = replace(
            material_topic_plan(),
            required_topics=(
                RequiredTopic(
                    "topic-1", "shared", False,
                    FreshnessRequirement.NOT_REQUIRED,
                ),
                RequiredTopic(
                    "topic-2", "shared", True,
                    FreshnessRequirement.NOT_REQUIRED,
                ),
            ),
        )
        gap = EvidenceGapAnalysis(
            ("topic-2",), (), True, (RepairReasonCode.MISSING_TOPIC,), ("topic-2",),
        )

        repair = importlib.import_module("src.search.planner").SearchPlanner(
            object()
        ).plan_repair(search_plan, gap)

        self.assertTrue(repair.triggered)
        self.assertIn("shared", repair.repair_query.text)


class TopicFreshnessSufficiencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    def test_two_fresh_topics_and_one_stale_topic_are_partial(self):
        search_plan = topic_plan(
            topic(
                "topic-1",
                "A",
                FreshnessRequirement.CURRENT,
                date_from=date(2026, 8, 1),
            ),
            topic(
                "topic-2",
                "B",
                FreshnessRequirement.CURRENT,
                date_from=date(2026, 8, 1),
            ),
            topic(
                "topic-3",
                "C",
                FreshnessRequirement.CURRENT,
                date_from=date(2026, 8, 1),
            ),
        )
        bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge(
                {
                    "C1": topic_judge_ok(
                        "C1",
                        supported_topic_ids=("topic-1",),
                        freshness_by_topic={"topic-1": "satisfied"},
                    ),
                    "C2": topic_judge_ok(
                        "C2",
                        supported_topic_ids=("topic-2",),
                        freshness_by_topic={"topic-2": "satisfied"},
                    ),
                    "C3": topic_judge_ok(
                        "C3",
                        supported_topic_ids=("topic-3",),
                        freshness_by_topic={"topic-3": "satisfied"},
                    ),
                }
            )
        ).assemble(
            search_plan,
            (
                candidate(url="https://a.example/fresh", published=datetime(2026, 8, 9, tzinfo=timezone.utc)),
                candidate(url="https://b.example/fresh", published=datetime(2026, 8, 9, tzinfo=timezone.utc)),
                candidate(url="https://c.example/stale", published=datetime(2025, 1, 1, tzinfo=timezone.utc)),
            ),
        )

        eligibility = getattr(importlib.import_module("src.search.models"), "FreshnessEligibility")
        self.assertIs(bundle.evidence_state, EvidenceState.PARTIAL)
        self.assertEqual(("topic-1", "topic-2"), bundle.supported_topic_ids)
        self.assertEqual(("topic-3",), bundle.missing_topic_ids)
        self.assertIs(eligibility.STALE, bundle.topic_assessments[2].freshness)

    def test_unknown_timestamp_is_missing_even_when_judge_claims_current(self):
        search_plan = topic_plan(
            topic(
                "topic-1",
                "current release",
                FreshnessRequirement.CURRENT,
                date_from=date(2026, 8, 1),
            )
        )
        bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge(
                {
                    "C1": topic_judge_ok(
                        "C1",
                        freshness_by_topic={"topic-1": "satisfied"},
                    )
                }
            )
        ).assemble(search_plan, (candidate(published=None),))

        eligibility = getattr(importlib.import_module("src.search.models"), "FreshnessEligibility")
        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)
        self.assertEqual(("topic-1",), bundle.missing_topic_ids)
        self.assertIs(eligibility.UNKNOWN, bundle.topic_assessments[0].freshness)

    def test_date_after_topic_upper_bound_is_unknown(self):
        search_plan = topic_plan(
            topic(
                "topic-1",
                "as-of release",
                FreshnessRequirement.AS_OF,
                date_to=date(2026, 8, 8),
            )
        )
        bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge(
                {
                    "C1": topic_judge_ok(
                        "C1",
                        freshness_by_topic={"topic-1": "satisfied"},
                    )
                }
            )
        ).assemble(
            search_plan,
            (candidate(published=datetime(2026, 8, 9, tzinfo=timezone.utc)),),
        )

        eligibility = getattr(importlib.import_module("src.search.models"), "FreshnessEligibility")
        self.assertIs(eligibility.UNKNOWN, bundle.topic_assessments[0].freshness)
        self.assertEqual(("topic-1",), bundle.missing_topic_ids)

    def test_one_relevant_source_can_satisfy_any_relevant_topic(self):
        search_plan = topic_plan(topic("topic-1", "release"))
        bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": topic_judge_ok("C1")})
        ).assemble(search_plan, (candidate(),))

        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual(("topic-1",), bundle.supported_topic_ids)

    def test_one_source_cannot_satisfy_independent_corroboration(self):
        search_plan = topic_plan(
            topic(
                "topic-1",
                "release",
                source_requirement=SourceRequirement.INDEPENDENT_CORROBORATION,
            )
        )
        bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": topic_judge_ok("C1")})
        ).assemble(search_plan, (candidate(),))

        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)
        self.assertEqual(("topic-1",), bundle.missing_topic_ids)

    def test_two_independence_groups_satisfy_independent_corroboration(self):
        search_plan = topic_plan(
            topic(
                "topic-1",
                "release",
                source_requirement=SourceRequirement.INDEPENDENT_CORROBORATION,
            )
        )
        bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge(
                {"C1": topic_judge_ok("C1"), "C2": topic_judge_ok("C2")}
            )
        ).assemble(
            search_plan,
            (
                candidate(url="https://one.example/release", content="one direct release report"),
                candidate(url="https://two.example/release", content="two independent release report"),
            ),
        )

        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual(("topic-1",), bundle.supported_topic_ids)

    def test_same_or_empty_independence_groups_cannot_corroborate(self):
        search_plan = topic_plan(
            topic(
                "topic-1",
                "release",
                source_requirement=SourceRequirement.INDEPENDENT_CORROBORATION,
            )
        )
        judged = StaticEvidenceJudge(
            {"C1": topic_judge_ok("C1"), "C2": topic_judge_ok("C2")}
        )
        same_group = self.module.EvidenceAssembler(judged).assemble(
            search_plan,
            (
                candidate(url="https://same.example/one", content="one direct release report"),
                candidate(url="https://same.example/two", content="two direct release report"),
            ),
        )
        self.assertIs(same_group.evidence_state, EvidenceState.INSUFFICIENT)

        def empty_groups(items, _provenance):
            return [replace(item, independence_group=None) for item in items]

        with mock.patch.object(
            self.module,
            "_assign_independence_groups",
            side_effect=empty_groups,
        ):
            empty_group = self.module.EvidenceAssembler(judged).assemble(
                search_plan,
                (
                    candidate(url="https://one.example/release", content="one direct release report"),
                    candidate(url="https://two.example/release", content="two independent release report"),
                ),
            )
        self.assertIs(empty_group.evidence_state, EvidenceState.INSUFFICIENT)

    def test_material_conflict_precedes_partial_and_edge_conflict_does_not(self):
        material = topic_plan(
            topic("topic-1", "release"),
            topic("topic-2", "history"),
        )
        material_bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge(
                {
                    "C1": topic_judge_ok(
                        "C1",
                        conflict_key="version",
                        conflict_value="1.0",
                    ),
                    "C2": topic_judge_ok(
                        "C2",
                        conflict_key="version",
                        conflict_value="2.0",
                    ),
                    "C3": topic_judge_ok(
                        "C3",
                        supported_topic_ids=("topic-2",),
                    ),
                }
            )
        ).assemble(
            material,
            (
                candidate(url="https://one.example/release"),
                candidate(url="https://two.example/release"),
                candidate(url="https://three.example/history"),
            ),
        )
        self.assertIs(material_bundle.evidence_state, EvidenceState.CONFLICTING)
        self.assertEqual(("topic-2",), material_bundle.supported_topic_ids)

        edge = topic_plan(
            topic("topic-1", "release"),
            topic("topic-2", "background", material=False),
        )
        edge_bundle = self.module.EvidenceAssembler(
            StaticEvidenceJudge(
                {
                    "C1": topic_judge_ok(
                        "C1",
                        supported_topic_ids=("topic-2",),
                        conflict_key="background-version",
                        conflict_value="1.0",
                    ),
                    "C2": topic_judge_ok(
                        "C2",
                        supported_topic_ids=("topic-2",),
                        conflict_key="background-version",
                        conflict_value="2.0",
                    ),
                    "C3": topic_judge_ok("C3"),
                }
            )
        ).assemble(
            edge,
            (
                candidate(url="https://one.example/background"),
                candidate(url="https://two.example/background"),
                candidate(url="https://three.example/release"),
            ),
        )
        self.assertIs(edge_bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual((), edge_bundle.conflict_groups)

    def test_same_conflict_key_on_disjoint_material_topics_does_not_conflict(self):
        search_plan = topic_plan(
            topic("topic-1", "A"),
            topic("topic-2", "B"),
        )
        judge = StaticEvidenceJudge(
            {
                "C1": topic_judge_ok(
                    "C1",
                    supported_topic_ids=("topic-1",),
                    conflict_key="status",
                    conflict_value="alpha",
                ),
                "C2": topic_judge_ok(
                    "C2",
                    supported_topic_ids=("topic-2",),
                    conflict_key="status",
                    conflict_value="beta",
                ),
            }
        )

        bundle = self.module.EvidenceAssembler(judge).assemble(
            search_plan,
            (
                candidate(url="https://a.example/status"),
                candidate(url="https://b.example/status"),
            ),
        )

        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual(("topic-1", "topic-2"), bundle.supported_topic_ids)
        self.assertEqual((), bundle.conflicts)

    def test_conflict_removes_only_the_material_topic_that_actually_conflicts(self):
        search_plan = topic_plan(
            topic("topic-1", "A"),
            topic("topic-2", "B"),
        )
        judge = StaticEvidenceJudge(
            {
                "C1": topic_judge_ok(
                    "C1",
                    supported_topic_ids=("topic-1", "topic-2"),
                    conflict_key="version",
                    conflict_value="1",
                ),
                "C2": topic_judge_ok(
                    "C2",
                    supported_topic_ids=("topic-1",),
                    conflict_key="version",
                    conflict_value="2",
                ),
            }
        )

        bundle = self.module.EvidenceAssembler(judge).assemble(
            search_plan,
            (
                candidate(url="https://a.example/version"),
                candidate(url="https://b.example/version"),
            ),
        )

        self.assertIs(bundle.evidence_state, EvidenceState.CONFLICTING)
        self.assertEqual(("topic-2",), bundle.supported_topic_ids)
        self.assertEqual(("topic-1",), bundle.missing_topic_ids)
        self.assertEqual(("E1",), bundle.topic_assessments[1].supporting_evidence_ids)
        self.assertEqual(
            ("E1", "E2"),
            tuple(member.evidence_id for member in bundle.conflicts[0].members),
        )
        self.assertEqual(("topic-1",), bundle.conflicts[0].topic_ids)
        gap = self.module.EvidenceAssembler(judge).analyze_gap(search_plan, bundle)
        self.assertEqual(("topic-1",), gap.repair_target_topic_ids)

    def test_version_literal_has_matching_mismatching_and_absent_outcomes(self):
        search_plan = topic_plan(
            topic(
                "topic-1",
                "Python version",
                FreshnessRequirement.VERSION,
                version_constraint="3.13",
            )
        )
        eligibility = getattr(importlib.import_module("src.search.models"), "FreshnessEligibility")
        cases = (
            ("Python 3.13 release", "supports 3.13", eligibility.SATISFIED),
            ("Python 3.12 release", "supports 3.12", eligibility.STALE),
            ("Python 13.13 release", "supports 13.13", eligibility.STALE),
            ("Python 3.130 release", "supports 3.130", eligibility.STALE),
            ("Python3.13 release", "product name may precede the exact version", eligibility.SATISFIED),
            ("Python release", "release notes", eligibility.UNKNOWN),
        )
        for title, content, expected in cases:
            with self.subTest(title=title):
                bundle = self.module.EvidenceAssembler(
                    StaticEvidenceJudge(
                        {
                            "C1": topic_judge_ok(
                                "C1",
                                freshness_by_topic={"topic-1": "satisfied"},
                            )
                        }
                    )
                ).assemble(search_plan, (candidate(title=title, content=content),))
                self.assertIs(expected, bundle.topic_assessments[0].freshness)

    def test_judge_rejects_unknown_topic_or_freshness_values_fail_closed(self):
        search_plan = topic_plan(topic("topic-1", "release"))
        invalid_rows = (
            topic_judge_ok("C1", supported_topic_ids=("topic-99",)),
            topic_judge_ok("C1", freshness_by_topic={"topic-1": "future"}),
            {
                "candidate_id": "C1",
                "source_relation": "independent",
                "supported_topic_ids": ["topic-1"],
            },
            topic_judge_ok("C1") | {"unreviewed_field": True},
        )
        for row in invalid_rows:
            with self.subTest(row=row):
                bundle = self.module.EvidenceAssembler(
                    StaticEvidenceJudge({"C1": row})
                ).assemble(search_plan, (candidate(),))
                self.assertEqual((), bundle.evidence_items)
                self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)


class EvidenceDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    def test_url_canonicalization_dedup(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("定义",)),
                "C2": judge_ok("C2", supported=("定义",)),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        candidates = (
            candidate(url="https://example.com/page"),
            candidate(url="https://example.com/page/"),
            candidate(url="https://example.com/page#fragment"),
        )
        bundle = assembler.assemble(plan(), candidates)
        # Redirects/fragments/trailing slash collapse to one canonical URL.
        self.assertEqual(len(bundle.evidence_items), 1)

    def test_syndicated_excerpts_group_into_one_independence_group(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("定义",)),
                "C2": judge_ok("C2", supported=("定义",)),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        text = "同一篇关于光合作用的转载文章内容完全相同。"
        candidates = (
            candidate(url="https://a.example.com/1", content=text),
            candidate(url="https://b.example.com/2", content=text),
        )
        bundle = assembler.assemble(plan(), candidates)
        self.assertEqual(len(bundle.evidence_items), 2)
        self.assertEqual(bundle.evidence_items[0].independence_group, bundle.evidence_items[1].independence_group)
        self.assertIsNotNone(bundle.evidence_items[0].independence_group)

    def test_same_domain_is_not_independent_when_wording_differs(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", relation="primary"),
                "C2": judge_ok("C2", relation="independent"),
            }
        )
        deep_candidates = tuple(
            replace(
                item,
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for item in (
                candidate(url="https://same.example/a", content="正文表述甲：光合作用定义。"),
                candidate(url="https://same.example/b", content="完全不同措辞乙：植物把光转成化学能。"),
            )
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            corroborated_topic_plan("定义"),
            deep_candidates,
        )
        self.assertEqual(
            bundle.evidence_items[0].independence_group,
            bundle.evidence_items[1].independence_group,
        )
        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)

    def test_same_registrable_parent_subdomains_cannot_corroborate(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", relation="primary", supported=("动态",)),
                "C2": judge_ok("C2", relation="independent", supported=("动态",)),
            }
        )
        candidates = tuple(
            replace(
                candidate(url=url, content=text),
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for url, text in (
                ("https://www.vendor.example/status", "厂商主站确认当前状态。"),
                ("https://news.vendor.example/report", "新闻子站使用完全不同的措辞报道。"),
            )
        )

        bundle = self.module.EvidenceAssembler(judge).assemble(
            corroborated_topic_plan("动态"), candidates
        )

        self.assertEqual(
            bundle.evidence_items[0].independence_group,
            bundle.evidence_items[1].independence_group,
        )
        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)

    def test_same_publisher_across_domains_cannot_corroborate(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok(
                    "C1", relation="primary", supported=("动态",), publisher="Example Wire"
                ),
                "C2": judge_ok(
                    "C2", relation="independent", supported=("动态",), publisher="Example Wire"
                ),
            }
        )
        candidates = tuple(
            replace(
                candidate(url=url, content=text),
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for url, text in (
                ("https://authority.example/status", "权威页面陈述动态状态。"),
                ("https://affiliate.invalid/report", "关联出版方采用不同措辞。"),
            )
        )

        bundle = self.module.EvidenceAssembler(judge).assemble(
            corroborated_topic_plan("动态"), candidates
        )

        self.assertEqual(
            bundle.evidence_items[0].publisher,
            bundle.evidence_items[1].publisher,
        )
        self.assertEqual(
            bundle.evidence_items[0].independence_group,
            bundle.evidence_items[1].independence_group,
        )
        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)

    def test_domain_to_publisher_bridge_is_transitive_in_every_order(self):
        specs = (
            ("https://www.vendor.example/status", "Vendor Org", (), "厂商主站确认当前状态。"),
            ("https://news.vendor.example/report", "Affiliate Media", (), "新闻子站采用不同措辞报道。"),
            ("https://affiliate.invalid/report", "Affiliate Media", (), "关联出版方独立撰写另一篇报道。"),
        )

        for ordered_specs in permutations(specs):
            with self.subTest(order=tuple(spec[0] for spec in ordered_specs)):
                bundle = self._deep_bridge_bundle(ordered_specs)
                self.assertEqual(len(bundle.evidence_items), 3)
                self.assertEqual(
                    len({item.independence_group for item in bundle.evidence_items}),
                    1,
                )
                self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)

    def test_publisher_to_canonical_marker_bridge_is_transitive_in_every_order(self):
        specs = (
            ("https://publisher.example/a", "Shared Publisher", (), "来源甲采用独立表述。"),
            (
                "https://bridge.invalid/b",
                "Shared Publisher",
                ("canonical_source:wire-story-42",),
                "桥接来源乙采用另一种表述。",
            ),
            (
                "https://copy.invalid/c",
                "Copy Publisher",
                ("canonical_source:wire-story-42",),
                "转载来源丙采用第三种表述。",
            ),
        )

        for ordered_specs in permutations(specs):
            with self.subTest(order=tuple(spec[0] for spec in ordered_specs)):
                bundle = self._deep_bridge_bundle(ordered_specs)
                self.assertEqual(
                    len({item.independence_group for item in bundle.evidence_items}),
                    1,
                )

    def test_text_to_syndication_bridge_is_transitive_in_every_order(self):
        syndicated_text = "同一篇转载文章逐字说明当前动态状态与发布日期。"
        specs = (
            ("https://origin.example/a", "Origin Publisher", (), syndicated_text),
            (
                "https://bridge.invalid/b",
                "Bridge Publisher",
                ("syndication_source:wire-story-99",),
                syndicated_text,
            ),
            (
                "https://copy.invalid/c",
                "Copy Publisher",
                ("syndication_source:wire-story-99",),
                "转载方针对同一稿件另写摘要。",
            ),
        )

        for ordered_specs in permutations(specs):
            with self.subTest(order=tuple(spec[0] for spec in ordered_specs)):
                bundle = self._deep_bridge_bundle(ordered_specs)
                self.assertEqual(
                    len({item.independence_group for item in bundle.evidence_items}),
                    1,
                )

    def test_nfc_and_nfd_publishers_are_one_provenance_group_in_both_orders(self):
        nfc = "ガイド社"
        nfd = "カ\u3099イト\u3099社"

        for first_publisher, second_publisher in ((nfc, nfd), (nfd, nfc)):
            with self.subTest(first=first_publisher, second=second_publisher):
                bundle = self._deep_bridge_bundle(
                    (
                        (
                            "https://primary.example/report",
                            first_publisher,
                            (),
                            "主来源说明当前动态事实。",
                        ),
                        (
                            "https://independent.invalid/report",
                            second_publisher,
                            (),
                            "第二页面采用完全不同的措辞。",
                        ),
                    )
                )

                self.assertEqual(
                    len({item.independence_group for item in bundle.evidence_items}),
                    1,
                )
                self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)

    def test_publisher_nfc_normalization_preserves_meaningful_distinctions(self):
        distinct_publishers = (
            ("カ\u3099イト\u3099社", "カイト社"),
            ("Ｅｘａｍｐｌｅ Ｏｒｇ", "Example Org"),
        )

        for first_publisher, second_publisher in distinct_publishers:
            with self.subTest(first=first_publisher, second=second_publisher):
                bundle = self._deep_bridge_bundle(
                    (
                        (
                            "https://primary.example/report",
                            first_publisher,
                            (),
                            "主来源说明当前动态事实。",
                        ),
                        (
                            "https://independent.invalid/report",
                            second_publisher,
                            (),
                            "第二页面采用完全不同的措辞。",
                        ),
                    )
                )

                self.assertEqual(
                    len({item.independence_group for item in bundle.evidence_items}),
                    2,
                )
                self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)

    def test_provenance_markers_use_nfc_without_nfkc_compatibility_folding(self):
        nfc = "ガイド社"
        nfd = "カ\u3099イト\u3099社"

        for prefix in ("canonical_source", "syndication_source"):
            with self.subTest(prefix=prefix, equivalence="nfc_nfd"):
                bundle = self._deep_bridge_bundle(
                    (
                        (
                            "https://primary.example/report",
                            "Primary Publisher",
                            (f"{prefix}:{nfc}",),
                            "主来源说明当前动态事实。",
                        ),
                        (
                            "https://independent.invalid/report",
                            "Independent Publisher",
                            (f"{prefix}:{nfd}",),
                            "第二页面采用完全不同的措辞。",
                        ),
                    )
                )
                self.assertEqual(
                    len({item.independence_group for item in bundle.evidence_items}),
                    1,
                )

            with self.subTest(prefix=prefix, equivalence="fullwidth_ascii"):
                bundle = self._deep_bridge_bundle(
                    (
                        (
                            "https://primary.example/report",
                            "Primary Publisher",
                            (f"{prefix}:Ｅｘａｍｐｌｅ Ｏｒｇ",),
                            "主来源说明当前动态事实。",
                        ),
                        (
                            "https://independent.invalid/report",
                            "Independent Publisher",
                            (f"{prefix}:Example Org",),
                            "第二页面采用完全不同的措辞。",
                        ),
                    )
                )
                self.assertEqual(
                    len({item.independence_group for item in bundle.evidence_items}),
                    2,
                )

    def _deep_bridge_bundle(self, specs):
        judged = {
            f"C{index}": judge_ok(
                f"C{index}",
                relation="primary" if index == 1 else "independent",
                supported=("动态",),
                publisher=publisher,
            )
            for index, (_url, publisher, _flags, _text) in enumerate(specs, 1)
        }
        candidates = tuple(
            replace(
                candidate(
                    url=url,
                    content=text,
                ),
                hit=replace(candidate(url=url, content=text).hit, quality_flags=flags),
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for url, _publisher, flags, text in specs
        )
        return self.module.EvidenceAssembler(StaticEvidenceJudge(judged)).assemble(
            corroborated_topic_plan("动态"),
            candidates,
        )


class EvidenceStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    def test_sufficient_when_all_topics_supported(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(required_topics=("定义",)), (candidate(),))
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)

    def test_partial_lists_missing_topics(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(required_topics=("定义", "历史")), (candidate(),))
        self.assertIs(bundle.evidence_state, EvidenceState.PARTIAL)
        self.assertIn("历史", bundle.missing_claim_topics)

    def test_conflict_group_created_from_two_evidence_items(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok(
                    "C1", supported=("版本",), conflict_key="version",
                    conflict_value="1.0",
                ),
                "C2": judge_ok(
                    "C2", supported=("版本",), conflict_key="version",
                    conflict_value="2.0",
                    conflict_relation="claims_supersession",
                ),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(
            plan(required_topics=("版本",)),
            (
                candidate(url="https://a.example.com/v", content="版本 1.0"),
                candidate(url="https://b.example.com/v", content="版本 2.0"),
            ),
        )
        self.assertIs(bundle.evidence_state, EvidenceState.CONFLICTING)
        self.assertTrue(bundle.conflict_groups)
        self.assertEqual(bundle.conflicts[0].conflict_key, "version")
        self.assertEqual(
            [(member.evidence_id, member.value) for member in bundle.conflicts[0].members],
            [("E1", "1.0"), ("E2", "2.0")],
        )
        self.assertEqual(bundle.conflicts[0].members[1].relation, "claims_supersession")

    def test_conflict_members_preserve_each_publication_date(self):
        older = datetime(2025, 1, 1, tzinfo=timezone.utc)
        newer = datetime(2026, 1, 1, tzinfo=timezone.utc)
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("版本",), conflict_key="version", conflict_value="1.0"),
                "C2": judge_ok("C2", supported=("版本",), conflict_key="version", conflict_value="2.0"),
            }
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("版本",)),
            (
                candidate(url="https://a.example/v", content="版本 1.0", published=older),
                candidate(url="https://b.example/v", content="版本 2.0", published=newer),
            ),
        )
        self.assertEqual(
            [member.published_at for member in bundle.conflicts[0].members],
            [older, newer],
        )

    def test_model_memory_statement_does_not_create_conflict(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(), (candidate(),))
        # No second Evidence item exists; no conflict can be formed.
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual(bundle.conflict_groups, ())

    def test_dates_remain_separate(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        published = datetime(2024, 5, 1, tzinfo=timezone.utc)
        bundle = assembler.assemble(plan(), (candidate(published=published),))
        item = bundle.evidence_items[0]
        self.assertEqual(item.published_at, published)
        self.assertIsNotNone(item.retrieved_at)

    def test_single_source_does_not_meet_independent_corroboration(self):
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("动态",), relation="primary")}
        )
        assembler = self.module.EvidenceAssembler(judge)
        authoritative = replace(
            candidate(),
            excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
            extraction_status="page_extract",
            content_reads_consumed=1,
        )
        bundle = assembler.assemble(
            corroborated_topic_plan("动态"),
            (authoritative,),
        )
        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)
        self.assertEqual(("topic-1",), bundle.missing_topic_ids)

    def test_independent_corroboration_accepts_distinct_groups(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("动态",), relation="primary"),
                "C2": judge_ok("C2", supported=("动态",), relation="independent"),
            }
        )
        candidates = tuple(
            replace(
                candidate(url=url, content=text),
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for url, text in (
                ("https://authority.example/status", "权威来源确认动态状态。"),
                ("https://independent.example/report", "独立来源交叉确认该动态状态。"),
            )
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            corroborated_topic_plan("动态"), candidates
        )
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertNotEqual(
            bundle.evidence_items[0].independence_group,
            bundle.evidence_items[1].independence_group,
        )

    def test_independent_secondary_sources_meet_corroboration(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("动态",), relation="independent"),
                "C2": judge_ok("C2", supported=("动态",), relation="independent"),
            }
        )
        candidates = tuple(
            replace(
                candidate(url=url, content="动态页面正文"),
                excerpt_origin=ExcerptOrigin.PAGE_EXTRACT,
                extraction_status="page_extract",
                content_reads_consumed=1,
            )
            for url in ("https://a.example/status", "https://b.example/report")
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            corroborated_topic_plan("动态"), candidates
        )
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)

    def test_current_topic_requires_judged_current_freshness(self):
        judge = StaticEvidenceJudge(
            {
                "C1": topic_judge_ok(
                    "C1",
                    freshness_by_topic={"topic-1": "unknown"},
                )
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(
            current_topic_plan("动态"),
            (
                candidate(
                    provider="ddgs",
                    content="搜索片段",
                    url="https://ddgs.example.com/page",
                ),
            ),
        )
        self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIs(bundle.topic_assessments[0].freshness, FreshnessEligibility.UNKNOWN)

    def test_failed_fetch_snippet_is_not_citable_even_when_current_is_satisfied(self):
        failed_document = FetchedDocument(
            "https://tavily.example/status",
            "https://tavily.example/status",
            None,
            None,
            None,
            "request_error",
            (),
        )
        weak = replace(
            candidate(provider="tavily", content="当前状态片段", url="https://tavily.example/status"),
            document=failed_document,
            extraction_status="search_result_snippet_after_fetch_failure",
            content_reads_consumed=1,
        )
        judge = StaticEvidenceJudge(
            {
                "C1": topic_judge_ok(
                    "C1",
                    freshness_by_topic={"topic-1": "satisfied"},
                )
            }
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            current_topic_plan("动态"), (weak,)
        )
        self.assertIsNot(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertFalse(bundle.evidence_items[0].citable)
        self.assertIs(bundle.topic_assessments[0].freshness, FreshnessEligibility.SATISFIED)

    def test_risk_does_not_change_failed_fetch_snippet_admission(self):
        baseline_plan = topic_plan(topic("topic-1", "剂量"))
        weak = replace(
            candidate(content="药物剂量搜索片段"),
            extraction_status="search_result_snippet_after_fetch_failure",
            content_reads_consumed=1,
        )
        judge = StaticEvidenceJudge(
            {"C1": topic_judge_ok("C1")}
        )

        baseline = self.module.EvidenceAssembler(judge).assemble(baseline_plan, (weak,))

        self.assertIs(baseline.evidence_state, EvidenceState.INSUFFICIENT)
        self.assertFalse(baseline.evidence_items[0].citable)

    def test_two_weak_dynamic_snippets_cannot_form_material_conflict(self):
        weak_candidates = tuple(
            replace(
                candidate(url=url, content=text),
                extraction_status="search_result_snippet_after_fetch_failure",
                content_reads_consumed=1,
            )
            for url, text in (
                ("https://a.example/version", "当前版本 1.0"),
                ("https://b.example/version", "当前版本 2.0"),
            )
        )
        judge = StaticEvidenceJudge(
            {
                "C1": topic_judge_ok(
                    "C1",
                    freshness_by_topic={"topic-1": "satisfied"},
                    conflict_key="version",
                    conflict_value="1.0",
                ),
                "C2": topic_judge_ok(
                    "C2",
                    freshness_by_topic={"topic-1": "satisfied"},
                    conflict_key="version",
                    conflict_value="2.0",
                ),
            }
        )

        version_plan = topic_plan(
            topic(
                "topic-1",
                "当前版本",
                FreshnessRequirement.VERSION,
                version_constraint="2.0",
            )
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            version_plan, weak_candidates
        )

        self.assertNotEqual(bundle.evidence_state, EvidenceState.CONFLICTING)
        self.assertEqual(bundle.conflict_groups, ())
        self.assertEqual(bundle.conflicts, ())
        self.assertIn("当前版本", bundle.missing_claim_topics)

    def test_exact_required_topic_label_is_supported(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok(
                    "C1", supported=("Rust 和 Go 的并发模型差异",), relation="primary"
                )
            }
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("Rust 和 Go 的并发模型差异",)),
            (candidate(content="Rust 和 Go 的并发模型存在明确差异。"),),
        )
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual(bundle.missing_claim_topics, ())

    def test_broad_subtopic_cannot_satisfy_composite_required_topic(self):
        judge = StaticEvidenceJudge(
            {"C1": topic_judge_ok("C1", supported_topic_ids=())}
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("Rust 和 Go 的并发模型差异",)),
            (candidate(content="这里只介绍 Go，不比较 Rust，也未讨论并发模型。"),),
        )

        self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIn("Rust 和 Go 的并发模型差异", bundle.missing_claim_topics)

    def test_overlapping_product_name_is_not_a_topic_alias(self):
        judge = StaticEvidenceJudge(
            {"C1": topic_judge_ok("C1", supported_topic_ids=())}
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("Java",)),
            (candidate(content="本文只讨论 JavaScript。"),),
        )

        self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIn("Java", bundle.missing_claim_topics)

    def test_cjk_narrow_label_cannot_satisfy_full_required_topic(self):
        judge = StaticEvidenceJudge(
            {"C1": topic_judge_ok("C1", supported_topic_ids=())}
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("苹果公司的季度营收",)),
            (candidate(content="这里只介绍苹果这一名称。"),),
        )

        self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIn("苹果公司的季度营收", bundle.missing_claim_topics)

    def test_japanese_narrow_label_cannot_satisfy_full_required_topic(self):
        judge = StaticEvidenceJudge(
            {"C1": topic_judge_ok("C1", supported_topic_ids=())}
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("東京大学の入学要件",)),
            (candidate(content="東京大学という名称だけを紹介する。"),),
        )

        self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertIn("東京大学の入学要件", bundle.missing_claim_topics)

    def test_meaningful_topic_symbols_do_not_create_c_cpp_csharp_aliases(self):
        unequal_labels = (
            ("C", "C++"),
            ("C++", "C"),
            ("C", "C#"),
            ("C#", "C"),
            ("C++", "C#"),
            ("C#", "C++"),
        )

        for required, supported in unequal_labels:
            with self.subTest(required=required, supported=supported):
                judge = StaticEvidenceJudge(
                    {"C1": topic_judge_ok("C1", supported_topic_ids=())}
                )
                bundle = self.module.EvidenceAssembler(judge).assemble(
                    plan(required_topics=(required,)),
                    (candidate(content=f"正文只支持 {supported} 标签。"),),
                )
                self.assertNotEqual(bundle.evidence_state, EvidenceState.SUFFICIENT)
                self.assertIn(required, bundle.missing_claim_topics)

    def test_nfc_and_nfd_topic_labels_are_equivalent_in_both_directions(self):
        nfc = "がん治療ガイド"
        nfd = "か\u3099ん治療カ\u3099イト\u3099"

        for required, supported in ((nfc, nfd), (nfd, nfc)):
            with self.subTest(required=required, supported=supported):
                judge = StaticEvidenceJudge(
                    {"C1": judge_ok("C1", supported=(supported,), relation="primary")}
                )
                bundle = self.module.EvidenceAssembler(judge).assemble(
                    plan(required_topics=(required,)),
                    (candidate(content="がん治療ガイドについて説明する本文。"),),
                )
                self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
                self.assertEqual(bundle.missing_claim_topics, ())

    def test_topic_identity_normalizes_case_and_whitespace(self):
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("rust   言語",), relation="primary")}
        )
        bundle = self.module.EvidenceAssembler(judge).assemble(
            plan(required_topics=("  RUST\t言語  ",)),
            (candidate(content="Rust 言語について説明する本文。"),),
        )

        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        self.assertEqual(bundle.missing_claim_topics, ())


class EvidenceGapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = evidence_module()

    def test_zero_citable_evidence_is_unconditionally_insufficient(self):
        judge = StaticEvidenceJudge(
            {"C1": topic_judge_ok("C1", supported_topic_ids=("topic-1",))}
        )
        assembler = self.module.EvidenceAssembler(judge)
        p = plan(required_topics=("定义",), route=SearchTier.STANDARD)
        # A candidate whose excerpt is empty cannot be citable.
        from src.search.models import EvidenceCandidate, ExcerptOrigin
        empty = EvidenceCandidate(
            candidate(hit=None) if False else __import__("src.search.models", fromlist=["ProviderHit"]).ProviderHit(
                "tavily", "q1", "t", "https://example.com/x", None, None, None, None, (),
            ),
            None, None, None, "no_content", (), 0,
        )
        bundle = assembler.assemble(p, (empty,))
        self.assertIs(bundle.evidence_state, EvidenceState.INSUFFICIENT)
        self.assertTrue(bundle.limitations)

    def test_gap_analysis_reports_missing_topics_and_repair_eligibility(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(required_topics=("定义", "历史")), (candidate(),))
        gap = assembler.analyze_gap(plan(required_topics=("定义", "历史")), bundle)
        self.assertEqual(("topic-2",), gap.missing_topic_ids)
        self.assertEqual((RepairReasonCode.MISSING_TOPIC,), gap.repair_reason_codes)
        self.assertEqual(("topic-2",), gap.repair_target_topic_ids)
        self.assertTrue(gap.repair_eligible)

    def test_empty_gap_is_not_repairable(self):
        judge = StaticEvidenceJudge({"C1": judge_ok("C1", supported=("定义",))})
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(plan(required_topics=("定义",)), (candidate(),))
        gap = assembler.analyze_gap(plan(required_topics=("定义",)), bundle)
        self.assertFalse(gap.repair_eligible)
        self.assertEqual(gap.missing_topic_ids, ())

    def test_repair_assembly_keeps_prior_judge_anomalies_and_combines_counts(self):
        m = importlib.import_module("src.search.models")
        from src.search.evidence import _JudgeParseResult

        class TwoRoundJudge:
            def __init__(self):
                self.calls = 0

            def judge(self, _question, candidates, **_kwargs):
                self.calls += 1
                rows = {
                    f"C{index}": judge_ok(f"C{index}")
                    for index, _candidate in enumerate(candidates, 1)
                }
                if self.calls == 1:
                    return _JudgeParseResult(
                        rows,
                        anomaly_codes=(m.JudgeAnomalyCode.UNKNOWN_CANDIDATE,),
                        anomaly_count=1,
                    )
                return _JudgeParseResult(
                    rows,
                    anomaly_codes=(
                        m.JudgeAnomalyCode.MISSING_CANDIDATE,
                        m.JudgeAnomalyCode.UNKNOWN_CANDIDATE,
                    ),
                    anomaly_count=7,
                )

        assembler = self.module.EvidenceAssembler(TwoRoundJudge())
        search_plan = plan(required_topics=("定义",), route=SearchTier.STANDARD)
        first = assembler.assemble(search_plan, (candidate(),))
        repaired = assembler.assemble(
            search_plan,
            (candidate(), candidate(url="https://two.example/release")),
            previous=first,
        )

        self.assertEqual(
            (
                m.JudgeAnomalyCode.UNKNOWN_CANDIDATE,
                m.JudgeAnomalyCode.MISSING_CANDIDATE,
            ),
            repaired.judge_anomaly_codes,
        )
        self.assertEqual(8, repaired.judge_anomaly_count)
        self.assertEqual(first.evidence_state, repaired.evidence_state)


class JudgeGapHintTests(unittest.TestCase):
    """Task 6: only closed entity/premise hints may enter the gap analysis."""

    def setUp(self) -> None:
        self.module = evidence_module()

    def test_closed_hints_survive_for_unsupported_material_topics(self):
        m = importlib.import_module("src.search.models")
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=()),
                "gap_hints": ({"reason_code": "entity_ambiguity", "target_topic_id": "topic-2"},),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        search_plan = material_topic_plan()
        bundle = assembler.assemble(search_plan, (candidate(),))
        gap = assembler.analyze_gap(search_plan, bundle)
        self.assertIn(m.RepairReasonCode.ENTITY_AMBIGUITY, gap.repair_reason_codes)
        self.assertEqual(("topic-2",), gap.repair_target_topic_ids)
        self.assertTrue(gap.repair_eligible)

    def test_unknown_hint_reason_or_topic_is_discarded(self):
        m = importlib.import_module("src.search.models")
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=()),
                "gap_hints": (
                    {"reason_code": "missing_topic", "target_topic_id": "topic-2"},
                    {"reason_code": "premise_mismatch", "target_topic_id": "topic-9"},
                    {"reason_code": "premise_mismatch", "target_topic_id": "topic-2"},
                ),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        search_plan = material_topic_plan()
        bundle = assembler.assemble(search_plan, (candidate(),))
        gap = assembler.analyze_gap(search_plan, bundle)
        self.assertIn(m.RepairReasonCode.PREMISE_MISMATCH, gap.repair_reason_codes)
        self.assertNotIn(m.RepairReasonCode.ENTITY_AMBIGUITY, gap.repair_reason_codes)
        self.assertNotIn(m.RepairReasonCode.MISSING_TOPIC, gap.repair_reason_codes)
        self.assertEqual(("topic-2",), gap.repair_target_topic_ids)

    def test_hint_for_a_supported_topic_is_discarded(self):
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("core",)),
                "gap_hints": ({"reason_code": "entity_ambiguity", "target_topic_id": "topic-2"},),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        search_plan = material_topic_plan()
        bundle = assembler.assemble(search_plan, (candidate(),))
        self.assertIs(bundle.evidence_state, EvidenceState.SUFFICIENT)
        gap = assembler.analyze_gap(search_plan, bundle)
        self.assertFalse(gap.repair_eligible)
        self.assertEqual((), gap.repair_reason_codes)

    def test_judge_parser_accepts_closed_gap_hints(self):
        class Response:
            content = json.dumps(
                {
                    "candidates": {"C1": judge_ok("C1", supported=())},
                    "gap_hints": [
                        {"reason_code": "entity_ambiguity", "target_topic_id": "topic-2"}
                    ],
                },
                ensure_ascii=False,
            )

        class FakeLLM:
            def chat(self, *_args, **_kwargs):
                return Response()

        judge = self.module.LLMEvidenceJudge(FakeLLM())
        parsed = judge.judge(
            "比较并发 API",
            (candidate(),),
            required_topics=(
                {"topic_id": "topic-1", "label": "background"},
                {"topic_id": "topic-2", "label": "core"},
            ),
        )
        self.assertEqual((("entity_ambiguity", "topic-2"),), parsed["gap_hints"])


class GapAggregationTests(unittest.TestCase):
    """Task 6: every reason comes from its closed producer/target rule."""

    def setUp(self) -> None:
        self.module = evidence_module()

    def test_missing_material_topic_targets_its_topic_id(self):
        m = importlib.import_module("src.search.models")
        search_plan = material_topic_plan()
        assembler = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": judge_ok("C1", supported=())})
        )
        bundle = assembler.assemble(search_plan, (candidate(),))
        gap = assembler.analyze_gap(search_plan, bundle)
        self.assertEqual((m.RepairReasonCode.MISSING_TOPIC,), gap.repair_reason_codes)
        self.assertEqual(("topic-2",), gap.repair_target_topic_ids)
        self.assertEqual(("topic-2",), gap.missing_topic_ids)
        self.assertTrue(gap.repair_eligible)

    def test_stale_freshness_targets_stale_evidence(self):
        m = importlib.import_module("src.search.models")
        stale_topic = topic(
            "topic-2",
            "core",
            FreshnessRequirement.CURRENT,
            source_requirement=SourceRequirement.ANY_RELEVANT,
        )
        search_plan = replace(
            material_topic_plan(),
            required_topics=(material_topic_plan().required_topics[0], stale_topic),
        )
        judge = StaticEvidenceJudge(
            {"C1": judge_ok("C1", supported=("core",), freshness_by_topic={"topic-2": "stale"})}
        )
        assembler = self.module.EvidenceAssembler(judge)
        bundle = assembler.assemble(search_plan, (candidate(),))
        gap = assembler.analyze_gap(search_plan, bundle)
        self.assertIn(m.RepairReasonCode.STALE_EVIDENCE, gap.repair_reason_codes)
        self.assertEqual(("topic-2",), gap.repair_target_topic_ids)

    def test_material_conflict_targets_source_conflict(self):
        m = importlib.import_module("src.search.models")
        judge = StaticEvidenceJudge(
            {
                "C1": judge_ok("C1", supported=("core",), conflict_key="version", conflict_value="1"),
                "C2": judge_ok("C2", supported=("core",), conflict_key="version", conflict_value="2"),
            }
        )
        assembler = self.module.EvidenceAssembler(judge)
        search_plan = material_topic_plan()
        bundle = assembler.assemble(
            search_plan,
            (candidate(url="https://one.example"), candidate(url="https://two.example")),
        )
        gap = assembler.analyze_gap(search_plan, bundle)
        self.assertEqual((m.RepairReasonCode.SOURCE_CONFLICT,), gap.repair_reason_codes)
        self.assertEqual(("topic-2",), gap.repair_target_topic_ids)
        self.assertEqual(("conflict:version",), gap.conflict_group_ids)

    def test_unmet_independent_corroboration_targets_source_quality_gap(self):
        m = importlib.import_module("src.search.models")
        independent = topic(
            "topic-2",
            "core",
            FreshnessRequirement.NOT_REQUIRED,
            source_requirement=SourceRequirement.INDEPENDENT_CORROBORATION,
        )
        search_plan = replace(
            material_topic_plan(),
            required_topics=(material_topic_plan().required_topics[0], independent),
        )
        assembler = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": judge_ok("C1", supported=("core",))})
        )
        bundle = assembler.assemble(search_plan, (candidate(),))
        gap = assembler.analyze_gap(search_plan, bundle)
        self.assertEqual((m.RepairReasonCode.SOURCE_QUALITY_GAP,), gap.repair_reason_codes)
        self.assertEqual(("topic-2",), gap.repair_target_topic_ids)

    def test_content_unreadable_reader_result_is_merged(self):
        m = importlib.import_module("src.search.models")
        search_plan = material_topic_plan()
        assembler = self.module.EvidenceAssembler(
            StaticEvidenceJudge({"C1": judge_ok("C1", supported=())})
        )
        bundle = assembler.assemble(search_plan, (candidate(),))
        gap = assembler.analyze_gap(
            search_plan,
            bundle,
            content_unreadable_topic_ids=("topic-2",),
        )
        self.assertEqual((m.RepairReasonCode.CONTENT_UNREADABLE,), gap.repair_reason_codes)
        self.assertEqual(("topic-2",), gap.repair_target_topic_ids)
        self.assertTrue(gap.repair_eligible)


if __name__ == "__main__":
    unittest.main()
