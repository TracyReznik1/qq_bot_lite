import importlib
import importlib.util
import json
import math
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from decimal import Decimal
from datetime import date


def models():
    spec = model_spec()
    if spec is None:
        raise AssertionError("src.search.models must provide the retrieval contracts")
    return importlib.import_module("src.search.models")


def model_spec():
    try:
        return importlib.util.find_spec("src.search.models")
    except ModuleNotFoundError:
        return None


class SearchModelFixtures:
    """Small valid records used by the exhaustive contract tables."""

    def __init__(self, module):
        self.m = module

    def decision(self):
        m = self.m
        return m.RetrievalDecision(
            route=m.SearchTier.LIGHT,
            skip_reason=None,
            must_search=True,
            reason_codes=(),
        )

    def budget(self):
        return self.m.DEFAULT_TIER_BUDGETS[self.m.SearchTier.LIGHT]

    def query(self):
        m = self.m
        return m.SearchQuery(
            "initial-1", m.SearchRoundKind.INITIAL, m.QueryPurpose.DIRECT, "query text",
            query_index=1, target_topic_ids=("topic-1",),
        )

    def analysis(self):
        m = self.m
        return m.RequestAnalysis(
            m.RetrievalContext(
                True,
                None,
                m.Factuality.FACTUAL,
                True,
                (),
                m.SourceRequirement.ANY_RELEVANT,
            ),
            m.FreshnessContext(
                m.FreshnessRequirement.NOT_REQUIRED,
                None,
                None,
                None,
                None,
            ),
            m.RiskContext(False, False, False),
        )

    def plan(self):
        m = self.m
        return m.SearchPlan(
            self.decision(), "question", m.PlanningStatus.NORMAL, (), None,
            (self.query(),),
            (
                m.RequiredTopic(
                    "topic-1", "question", True,
                    m.FreshnessRequirement.NOT_REQUIRED,
                    source_requirement=m.SourceRequirement.ANY_RELEVANT,
                ),
            ),
            frozenset(), (), self.budget(),
        )

    def repair_plan(self):
        return self.m.RepairPlan(False, (), (), None)

    def hit(self):
        return self.m.ProviderHit(
            "tavily", "q1", "title", "https://example.invalid",
            None, None, None, None, (),
        )

    def evidence_item(self):
        m = self.m
        return m.EvidenceItem(
            "e1", "q1", "tavily", "title", "https://example.invalid",
            None, "example.invalid", "Example", m.SourceRelation.PRIMARY,
            None, None, None, "excerpt", m.ExcerptOrigin.PROVIDER_SNIPPET,
            "ok", 1.0, 1.0, True, m.Freshness.NONE, True, (), ("question",), "source-1",
        )

    def gap_analysis(self):
        return self.m.EvidenceGapAnalysis((), (), False, (), ())

    def bundle(self):
        m = self.m
        plan = self.plan()
        topic = plan.required_topics[0]
        return m.EvidenceBundle(
            "req-1", self.decision(), plan, (), (),
            m.EvidenceGapAnalysis((topic.topic_id,), (), False, (), ()),
            self.repair_plan(), 1, (), m.EvidenceState.INSUFFICIENT,
            (topic.label,), (), (), (),
            topic_assessments=(
                m.TopicAssessment(
                    topic.topic_id,
                    m.FreshnessEligibility.NOT_REQUIRED,
                    (),
                ),
            ),
            supported_topic_ids=(),
            missing_topic_ids=(topic.topic_id,),
        )

    def document(self):
        return self.m.FetchedDocument(
            "https://example.invalid", None, None, None, None, "not_read", ()
        )

    def candidate(self):
        return self.m.EvidenceCandidate(
            self.hit(), None, None, None, "provider_snippet", (), 0
        )

    def claim(self):
        return self.m.Claim("c1", "b1", "claim", True, ())

    def answer_block(self):
        return self.m.AnswerBlock("b1", "paragraph", "answer", ())

    def draft(self):
        return self.m.GroundedDraft((), (), (), (), False)

    def validation_report(self):
        return self.m.ValidationReport(self.draft(), (), (), (), {}, ())

    def rendered_reply(self):
        return self.m.RenderedReply("answer", (), (), (), ())

    def trace_with_value(self, location, value):
        m = self.m
        trace = m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT)
        if location == "request_id":
            trace.request_id = value
        elif location == "executed_queries[].provider":
            trace.executed_queries = (
                m.QueryTraceEntry(
                    1, m.QueryPurpose.DIRECT, m.SearchRoundKind.INITIAL,
                    value, m.ProviderStatus.SUCCESS, 1,
                ),
            )
        elif location == "provider_attempts[].provider":
            attempt = m.ProviderAttempt("tavily", m.ProviderStatus.SUCCESS, 1, 1)
            # The final boundary must stay safe even if an upstream record is
            # populated outside its constructor.
            object.__setattr__(attempt, "provider", value)
            trace.provider_attempts = (attempt,)
        else:
            raise AssertionError(f"unknown Trace location: {location}")
        return trace

    @staticmethod
    def logged_value(logged, location):
        if location == "request_id":
            return logged["request_id"]
        if location == "executed_queries[].provider":
            return logged["executed_queries"][0]["provider"]
        if location == "provider_attempts[].provider":
            return logged["provider_attempts"][0]["provider"]
        raise AssertionError(f"unknown Trace location: {location}")


class RenderStateClosureTests(unittest.TestCase):
    def test_rejects_visible_claim_without_matching_citation_and_citable_source(self):
        m = models()
        block = m.AnswerBlock("B1", "factual", "答案", ("C1",))
        claim = m.Claim("C1", "B1", "答案", True, ("e1",))

        with self.assertRaises(ValueError):
            m.RenderState(
                m.RenderOutcome.ANSWER,
                (block,),
                (claim,),
                {},
                (),
                (),
                (),
                (),
            )

    def test_rejects_duplicate_or_non_contiguous_citation_numbers(self):
        m = models()
        block = m.AnswerBlock("B1", "factual", "答案", ("C1",))
        claim = m.Claim("C1", "B1", "答案", True, ("e1", "e2"))
        fixtures = SearchModelFixtures(m)
        sources = (
            fixtures.evidence_item(),
            replace(fixtures.evidence_item(), evidence_id="e2"),
        )
        for citations in ({"e1": 1, "e2": 1}, {"e1": 1, "e2": 3}):
            with self.subTest(citations=citations), self.assertRaises(ValueError):
                m.RenderState(
                    m.RenderOutcome.ANSWER,
                    (block,),
                    (claim,),
                    citations,
                    sources,
                    (),
                    (),
                    (),
                )


class SearchModelContractTests(unittest.TestCase):
    def test_00_models_module_is_available(self):
        self.assertIsNotNone(model_spec())

    def test_closed_enum_values_reject_unknown_strings(self):
        m = models()

        expected = {
            "SearchTier": {"skip", "light", "standard"},
            "SkipReason": {
                "user_forbid_web", "social_or_emotional", "creative_or_roleplay",
                "provided_text_transform", "provided_content_summary", "pure_math",
                "closed_logic", "closed_context_only",
            },
            "SearchRoundKind": {"initial", "repair"},
            "EvidenceState": {"sufficient", "partial", "conflicting", "insufficient"},
            "RequestSource": {"chat", "command", "compatibility"},
            "TriggerCode": {
                "explicit_no_web", "explicit_search", "explicit_verification",
                "explicit_source_request", "freshness_marker", "dynamic_attribute",
                "regulated_domain_foundation", "high_consequence_action",
                "current_rule_or_policy", "controversy_or_conflict",
                "external_fact_explanation_or_comparison", "recommendation_or_evaluation",
                "ambiguous_entity", "multi_hop_complexity", "mixed_task", "factual_default",
                "classifier_uncertain",
            },
            "BenefitDimension": {
                "accuracy", "freshness", "completeness", "verifiability", "disambiguation",
                "risk_control",
            },
            "Factuality": {"non_factual", "factual", "mixed", "ambiguous"},
            "Freshness": {"none", "low"},
            "FreshnessEligibility": {"not_required", "satisfied", "stale", "unknown"},
            "RiskLevel": {"low", "medium", "high"},
            "Actionability": {"none", "general", "personalized"},
            "PotentialHarm": {"none", "low", "high"},
            "ProviderStatus": {
                "success", "empty", "timeout", "error", "not_configured", "unavailable"
            },
            "CandidateRelevance": {"direct", "contextual", "irrelevant"},
            "QueryPurpose": {
                "direct", "primary", "independent", "time_bounded", "disambiguation",
                "counterevidence", "repair",
            },
            "PlanningStatus": {"normal", "degraded"},
            "ExcerptOrigin": {"provider_snippet", "page_extract", "document_extract"},
            "SourceRelation": {"primary", "independent", "secondary", "community", "unknown"},
            "SupportLabel": {"supported", "partial", "conflict", "unsupported", "unmapped"},
            "SearchFailureCode": {
                "provider_not_configured", "provider_unavailable", "provider_timeout",
                "no_results", "content_unreadable", "insufficient_evidence", "partial_evidence",
                "source_conflict", "validation_failed", "user_forbid_web",
            },
            "JudgeAnomalyCode": {
                "missing_candidate", "unknown_candidate", "malformed_candidate",
                "duplicate_candidate",
            },
        }
        for enum_name, values in expected.items():
            enum_type = getattr(m, enum_name)
            with self.subTest(enum_name=enum_name):
                self.assertEqual(values, {item.value for item in enum_type})
                with self.assertRaises(ValueError):
                    enum_type("invented_value")

    def test_decisions_accept_all_routes_and_expose_conflict_clarification(self):
        m = models()
        skipped = m.RetrievalDecision(
            route=m.SearchTier.SKIP,
            skip_reason=m.SkipReason.PURE_MATH,
            must_search=False,
            reason_codes=(),
        )
        self.assertFalse(skipped.requires_clarification)
        for tier in (m.SearchTier.LIGHT, m.SearchTier.STANDARD):
            with self.subTest(tier=tier):
                decision = m.RetrievalDecision(
                    route=tier,
                    skip_reason=None,
                    must_search=True,
                    reason_codes=(),
                )
                self.assertEqual(tier, decision.route)

        conflict = m.RetrievalDecision(
            route=m.SearchTier.SKIP,
            skip_reason=m.SkipReason.USER_FORBID_WEB,
            must_search=True,
            reason_codes=(),
        )
        self.assertTrue(conflict.requires_clarification)

    def test_decision_rejects_illegal_route_combinations_and_free_text_codes(self):
        m = models()
        invalid = (
            dict(route=m.SearchTier.SKIP, skip_reason=None, must_search=False, reason_codes=()),
            dict(route=m.SearchTier.LIGHT, skip_reason=m.SkipReason.PURE_MATH, must_search=True, reason_codes=()),
            dict(route=m.SearchTier.LIGHT, skip_reason=None, must_search="yes", reason_codes=()),
            dict(route=m.SearchTier.LIGHT, skip_reason=None, must_search=True, reason_codes=("free_text",)),
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    m.RetrievalDecision(**overrides)

    def test_budgets_are_immutable_and_validate_derived_totals(self):
        m = models()

        self.assertEqual(
            (1, 5, 2, 0, 1, 1),
            tuple(m.DEFAULT_TIER_BUDGETS[m.SearchTier.LIGHT].__dict__.values()),
        )
        self.assertEqual(
            (3, 8, 5, 1, 4, 2),
            tuple(m.DEFAULT_TIER_BUDGETS[m.SearchTier.STANDARD].__dict__.values()),
        )
        with self.assertRaises(TypeError):
            m.DEFAULT_TIER_BUDGETS[m.SearchTier.LIGHT] = m.DEFAULT_TIER_BUDGETS[m.SearchTier.LIGHT]
        with self.assertRaises(FrozenInstanceError):
            m.DEFAULT_TIER_BUDGETS[m.SearchTier.LIGHT].max_content_reads = 99
        with self.assertRaises(ValueError):
            m.TierBudget(1, 5, 2, 1, 1, 2)

    def test_provider_runtime_statuses_map_to_closed_failure_codes(self):
        m = models()
        mapping = getattr(m, "PROVIDER_STATUS_FAILURE_CODES", None)
        self.assertIsNotNone(mapping)

        self.assertIsNone(mapping[m.ProviderStatus.SUCCESS])
        self.assertEqual(
            m.SearchFailureCode.NO_RESULTS,
            mapping[m.ProviderStatus.EMPTY],
        )
        self.assertEqual(
            m.SearchFailureCode.PROVIDER_TIMEOUT,
            mapping[m.ProviderStatus.TIMEOUT],
        )
        self.assertEqual(
            m.SearchFailureCode.PROVIDER_UNAVAILABLE,
            mapping[m.ProviderStatus.ERROR],
        )
        self.assertEqual(
            m.SearchFailureCode.PROVIDER_NOT_CONFIGURED,
            mapping[m.ProviderStatus.NOT_CONFIGURED],
        )
        self.assertEqual(
            m.SearchFailureCode.PROVIDER_UNAVAILABLE,
            mapping[m.ProviderStatus.UNAVAILABLE],
        )

    def test_trace_log_is_json_safe_and_excludes_bodies_and_raw_queries(self):
        m = models()
        trace = m.SearchTrace(
            request_id="req-1",
            request_source=m.RequestSource.CHAT,
            route=m.SearchTier.STANDARD,
            skip_reason=None,
            adaptive_repair_round_started=True,
            executed_queries=(
                m.QueryTraceEntry(1, m.QueryPurpose.DIRECT, m.SearchRoundKind.INITIAL, "tavily", m.ProviderStatus.SUCCESS, 3),
                m.QueryTraceEntry(1, m.QueryPurpose.DIRECT, m.SearchRoundKind.INITIAL, "ddgs", m.ProviderStatus.EMPTY, 2),
                m.QueryTraceEntry(4, m.QueryPurpose.REPAIR, m.SearchRoundKind.REPAIR, "tavily", m.ProviderStatus.SUCCESS, 5),
            ),
            provider_attempts=(m.ProviderAttempt("fake", m.ProviderStatus.SUCCESS, 1, 3),),
            evidence_state=m.EvidenceState.SUFFICIENT,
            provider_invocation_started=True,
            content_read_count=2,
        )
        trace.initial_query_redaction_codes = ("cq_control_code", "data_url")
        trace.adaptive_repair_redaction_codes = ("callback_secret",)

        logged = trace.to_log_dict()
        self.assertEqual(2, logged["semantic_query_count"])
        self.assertEqual(1, logged["repair_query_count"])
        self.assertTrue(logged["provider_attempted"])
        self.assertTrue(logged["sufficient_evidence"])
        self.assertEqual(
            [1, 1, 4],
            [row["query_index"] for row in logged["executed_queries"]],
        )
        self.assertEqual(["cq_control_code", "data_url"], logged.get("initial_query_redaction_codes"))
        self.assertEqual(["callback_secret"], logged.get("adaptive_repair_redaction_codes"))
        self.assertNotIn("question", logged)
        self.assertNotIn("answer", logged)
        payload = json.dumps(logged).lower()
        self.assertNotIn("https://example.invalid/private-error", payload)
        self.assertNotIn("raw query text", payload)
        json.dumps(logged)

    def test_trace_serialization_redacts_unknown_entry_provider(self):
        m = models()
        trace = m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.STANDARD)
        trace.executed_queries = (
            m.QueryTraceEntry(1, m.QueryPurpose.DIRECT, m.SearchRoundKind.INITIAL, "private provider", m.ProviderStatus.SUCCESS, 1),
            m.QueryTraceEntry(2, m.QueryPurpose.REPAIR, m.SearchRoundKind.REPAIR, "tavily", m.ProviderStatus.SUCCESS, 1),
        )

        logged = trace.to_log_dict()

        self.assertEqual(
            "[redacted]",
            logged["executed_queries"][0]["provider"],
        )
        self.assertEqual("tavily", logged["executed_queries"][1]["provider"])
        self.assertEqual(2, logged["semantic_query_count"])
        json.dumps(logged)

    def test_judge_anomaly_trace_is_closed_bounded_and_body_free(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        bundle = replace(
            fixtures.bundle(),
            judge_anomaly_codes=(
                m.JudgeAnomalyCode.MISSING_CANDIDATE,
                m.JudgeAnomalyCode.UNKNOWN_CANDIDATE,
            ),
            judge_anomaly_count=3,
        )
        self.assertEqual(3, bundle.judge_anomaly_count)
        with self.assertRaises((TypeError, ValueError)):
            replace(bundle, judge_anomaly_codes=("C99",))
        with self.assertRaises(ValueError):
            replace(bundle, judge_anomaly_count=1)
        with self.assertRaises(ValueError):
            replace(bundle, judge_anomaly_count=9)

        trace = m.SearchTrace(
            "req-1",
            m.RequestSource.CHAT,
            m.SearchTier.LIGHT,
            judge_anomaly_codes=(m.JudgeAnomalyCode.UNKNOWN_CANDIDATE,),
            judge_anomaly_count=1,
        )
        logged = trace.to_log_dict()
        self.assertEqual(["unknown_candidate"], logged["judge_anomaly_codes"])
        self.assertEqual(1, logged["judge_anomaly_count"])

        trace.judge_anomaly_codes = ("C99",)
        trace.judge_anomaly_count = 99
        sanitized = trace.to_log_dict()
        self.assertEqual([], sanitized["judge_anomaly_codes"])
        self.assertEqual(0, sanitized["judge_anomaly_count"])
        self.assertNotIn("C99", json.dumps(sanitized))

    def test_validation_failed_is_legal_for_a_sufficient_bundle(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        evidence = fixtures.evidence_item()
        sufficient = replace(
            fixtures.bundle(),
            initial_evidence_ids=(evidence.evidence_id,),
            evidence_items=(evidence,),
            evidence_state=m.EvidenceState.SUFFICIENT,
            missing_claim_topics=(),
            topic_assessments=(
                m.TopicAssessment(
                    "topic-1",
                    m.FreshnessEligibility.NOT_REQUIRED,
                    (evidence.evidence_id,),
                ),
            ),
            supported_topic_ids=("topic-1",),
            missing_topic_ids=(),
        )
        result = m.SearchPipelineResult(
            fixtures.decision(),
            fixtures.plan(),
            sufficient,
            m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT),
            m.SearchFailureCode.VALIDATION_FAILED,
            analysis=fixtures.analysis(),
        )
        self.assertIs(result.failure_code, m.SearchFailureCode.VALIDATION_FAILED)

    def test_redaction_codes_are_closed_and_hostile_trace_mutation_stays_body_free(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        query = fixtures.query()
        for build in (
            lambda: replace(fixtures.plan(), query_redaction_codes=("sk-1234567890abcdef",)),
            lambda: m.RepairPlan(
                True,
                (m.RepairReasonCode.MISSING_TOPIC,),
                ("topic-2",),
                query,
                ("13800138000",),
            ),
            lambda: m.SearchTrace(
                "req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT,
                initial_query_redaction_codes=("https://private.invalid",),
            ),
            lambda: m.SearchTrace(
                "req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT,
                adaptive_repair_redaction_codes=("13800138000",),
            ),
        ):
            with self.subTest(build=build):
                with self.assertRaises(ValueError):
                    build()

        trace = m.SearchTrace(
            "req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT,
            initial_query_redaction_codes=("cq_control_code",),
            adaptive_repair_redaction_codes=("data_url",),
        )
        trace.initial_query_redaction_codes = ("sk-1234567890abcdef",)
        trace.adaptive_repair_redaction_codes = ("13800138000", "https://private.invalid")
        logged = trace.to_log_dict()
        self.assertEqual(["invalid_redaction_code"], logged["initial_query_redaction_codes"])
        self.assertEqual(["invalid_redaction_code"], logged["adaptive_repair_redaction_codes"])
        payload = json.dumps(logged)
        for secret in ("sk-1234567890abcdef", "13800138000", "https://private.invalid"):
            self.assertNotIn(secret, payload)

    def test_review_contracts_have_exact_public_fields_and_exports(self):
        m = models()
        expected_fields = {
            "ProviderResult": ("provider", "status", "hits", "latency_ms"),
            "FetchedDocument": (
                "requested_url", "final_url", "content_type", "title", "excerpt",
                "fetch_status", "untrusted_content_flags",
            ),
            "EvidenceCandidate": ("hit", "document", "excerpt", "excerpt_origin", "extraction_status", "safety_flags", "content_reads_consumed"),
            "EvidenceGapAnalysis": (
                "missing_topic_ids", "conflict_group_ids", "repair_eligible",
                "repair_reason_codes", "repair_target_topic_ids",
            ),
            "Claim": ("claim_id", "block_id", "text", "material", "evidence_ids"),
            "AnswerBlock": ("block_id", "kind", "text", "claim_ids"),
            "GroundedDraft": ("answer_blocks", "claims", "limitations", "conflict_summary", "used_knowledge_fallback"),
    "ValidationReport": ("draft", "retained_blocks", "retained_claims", "removed_block_ids", "claim_labels", "limitations", "status", "effective_certainty", "effective_claim_scope"),
            "RenderedReply": ("text", "chunks", "used_evidence_ids", "shown_source_urls", "degradation_disclosures"),
        }
        public = importlib.import_module("src.search").__all__
        for name, expected in expected_fields.items():
            with self.subTest(name=name):
                contract = getattr(m, name, None)
                self.assertIsNotNone(contract)
                self.assertEqual(expected, tuple(item.name for item in fields(contract)))
                self.assertIn(name, public)

    def test_gap_analysis_enforces_authoritative_repair_shape(self):
        m = models()
        gap = m.EvidenceGapAnalysis(
            ("topic-2",), (), True, (m.RepairReasonCode.MISSING_TOPIC,), ("topic-2",),
        )
        self.assertEqual(
            ("topic-2",),
            gap.missing_topic_ids,
        )
        self.assertEqual((m.RepairReasonCode.MISSING_TOPIC,), gap.repair_reason_codes)
        for values in (
            (("topic-2",), (), True, (m.RepairReasonCode.MISSING_TOPIC,), ()),
            (("topic-2",), (), False, (m.RepairReasonCode.MISSING_TOPIC,), ("topic-2",)),
            (("topic-2",), (), True, (), ("topic-2",)),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    m.EvidenceGapAnalysis(*values)

    def test_pipeline_result_rejects_ambiguous_search_and_skip_shapes(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        decision = self._search_decision(m)
        trace = m.SearchTrace("r", m.RequestSource.CHAT, m.SearchTier.LIGHT)
        with self.assertRaises(ValueError):
            m.SearchPipelineResult(decision, None, None, trace, analysis=fixtures.analysis())
        skip = m.RetrievalDecision(
            route=m.SearchTier.SKIP,
            skip_reason=m.SkipReason.PURE_MATH,
            must_search=False,
            reason_codes=(),
        )
        with self.assertRaises(ValueError):
            m.SearchPipelineResult(skip, object(), None, trace, analysis=fixtures.analysis())

    def test_pipeline_result_requires_request_analysis(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        with self.assertRaises(TypeError):
            m.SearchPipelineResult(
                fixtures.decision(),
                fixtures.plan(),
                fixtures.bundle(),
                m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT),
                m.SearchFailureCode.INSUFFICIENT_EVIDENCE,
            )

    def test_frozen_contracts_normalize_caller_owned_collections_and_budget_rejects_bools(self):
        m = models()
        reasons = [m.RetrievalComplexityCode.MULTI_FACT]
        decision = m.RetrievalDecision(
            route=m.SearchTier.LIGHT,
            skip_reason=None,
            must_search=True,
            reason_codes=reasons,
        )
        reasons.append(m.RetrievalComplexityCode.MULTI_ENTITY)
        self.assertEqual((m.RetrievalComplexityCode.MULTI_FACT,), decision.reason_codes)
        for index in range(6):
            values = [1, 5, 2, 0, 1, 1]
            values[index] = True
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    m.TierBudget(*values)

    def test_trace_rejects_untyped_sensitive_or_non_json_provider_attempt_metadata(self):
        m = models()
        trace = m.SearchTrace("r", m.RequestSource.CHAT, m.SearchTier.LIGHT)
        trace.provider_attempts = ({"provider": "https://private.invalid", "status": "success", "count": 1, "latency_ms": 1},)
        with self.assertRaises((TypeError, ValueError)):
            trace.to_log_dict()
        with self.assertRaises(ValueError):
            m.ProviderAttempt("api_key=secret", m.ProviderStatus.SUCCESS, 1, 1)
        with self.assertRaises(ValueError):
            m.ProviderAttempt("fake", m.ProviderStatus.SUCCESS, True, 1)
        with self.assertRaises(ValueError):
            m.ProviderAttempt("fake", m.ProviderStatus.SUCCESS, 1, math.inf)
        with self.assertRaises(TypeError):
            m.SearchTrace("r", m.RequestSource.CHAT, m.SearchTier.LIGHT, provider_attempts=(Decimal("1"),)).to_log_dict()

    def test_trace_final_boundary_uses_closed_identifier_and_provider_tables(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        locations = (
            "request_id",
            "executed_queries[].provider",
            "provider_attempts[].provider",
        )
        forbidden_probes = (
            "weather",
            "weather-2026",
            "raw query text",
            "http://example.invalid/path",
            "HTTP://EXAMPLE.INVALID/PATH",
            "ftp://example.invalid/file",
            "www.example.invalid",
            "sk-live-secret",
            "sk-123456789",
            "xoxb-123456789-secret",
            "api_key=secret",
            "apikey-123456789",
            "token=secret",
            "token-123456789",
            "qq=123456789",
            "qq:123456789",
            "qq-123456789",
            "group=987654321",
            "group:987654321",
            "group_id=987654321",
            "group-987654321",
            "data:image/png;base64,abc",
            "DATA:IMAGE/PNG;BASE64,ABC",
            "cq:image,file=secret.png",
            "CQ:IMAGE,FILE=SECRET.PNG",
            "user@example.invalid",
        )
        for location in locations:
            for probe in forbidden_probes:
                with self.subTest(location=location, probe=probe):
                    logged = fixtures.trace_with_value(location, probe).to_log_dict()
                    self.assertEqual(
                        "[redacted]", fixtures.logged_value(logged, location)
                    )
                    payload = json.dumps(logged)
                    self.assertNotIn(probe, payload)

        valid_ids = (
            "req-1",
            "initial-1",
            "repair-1",
            "q1",
            "Q1",
            "0123456789abcdef",
            "0123456789abcdef0123456789abcdef",
            "123e4567-e89b-12d3-a456-426614174000",
        )
        for location in locations[:1]:
            for valid_id in valid_ids:
                with self.subTest(location=location, valid_id=valid_id):
                    logged = fixtures.trace_with_value(location, valid_id).to_log_dict()
                    self.assertEqual(valid_id, fixtures.logged_value(logged, location))
                    json.dumps(logged)
        for provider in ("tavily", "ddgs"):
            for location in locations[1:]:
                with self.subTest(location=location, provider=provider):
                    logged = fixtures.trace_with_value(location, provider).to_log_dict()
                    self.assertEqual(provider, fixtures.logged_value(logged, location))
                    json.dumps(logged)

    def test_scalar_string_collection_fields_reject_all_21_scalar_inputs(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        cases = (
            ("SearchQuery.include_domains", fixtures.query(), "include_domains"),
            ("SearchPlan.entities", fixtures.plan(), "entities"),
            ("SearchPlan.required_topics", fixtures.plan(), "required_topics"),
            ("SearchPlan.query_redaction_codes", fixtures.plan(), "query_redaction_codes"),
            ("RepairPlan.target_topic_ids", fixtures.repair_plan(), "target_topic_ids"),
            ("ProviderHit.quality_flags", fixtures.hit(), "quality_flags"),
            ("EvidenceItem.safety_flags", fixtures.evidence_item(), "safety_flags"),
            ("EvidenceItem.supported_topics", fixtures.evidence_item(), "supported_topics"),
            ("EvidenceGapAnalysis.missing_topic_ids", fixtures.gap_analysis(), "missing_topic_ids"),
            ("EvidenceBundle.initial_evidence_ids", fixtures.bundle(), "initial_evidence_ids"),
            ("EvidenceBundle.limitations", fixtures.bundle(), "limitations"),
            ("FetchedDocument.untrusted_content_flags", fixtures.document(), "untrusted_content_flags"),
            ("EvidenceCandidate.safety_flags", fixtures.candidate(), "safety_flags"),
            ("Claim.evidence_ids", fixtures.claim(), "evidence_ids"),
            ("AnswerBlock.claim_ids", fixtures.answer_block(), "claim_ids"),
            ("GroundedDraft.limitations", fixtures.draft(), "limitations"),
            ("ValidationReport.removed_block_ids", fixtures.validation_report(), "removed_block_ids"),
            ("ValidationReport.limitations", fixtures.validation_report(), "limitations"),
            ("RenderedReply.chunks", fixtures.rendered_reply(), "chunks"),
            ("RenderedReply.used_evidence_ids", fixtures.rendered_reply(), "used_evidence_ids"),
            ("RenderedReply.shown_source_urls", fixtures.rendered_reply(), "shown_source_urls"),
        )
        self.assertEqual(21, len(cases))
        for name, record, field_name in cases:
            with self.subTest(field=name):
                with self.assertRaises(TypeError):
                    replace(record, **{field_name: "abc"})

    def test_nested_record_slots_enforce_all_36_shapes(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        cases = (
            ("SearchPlan.decision", fixtures.plan(), "decision", fixtures.decision(), fixtures.budget(), False),
            ("SearchPlan.budget", fixtures.plan(), "budget", fixtures.budget(), fixtures.decision(), False),
            ("EvidenceBundle.decision", fixtures.bundle(), "decision", fixtures.decision(), fixtures.plan(), False),
            ("EvidenceBundle.plan", fixtures.bundle(), "plan", fixtures.plan(), fixtures.decision(), False),
            ("EvidenceBundle.gap_analysis", fixtures.bundle(), "gap_analysis", fixtures.gap_analysis(), fixtures.plan(), False),
            ("EvidenceBundle.repair_plan", fixtures.bundle(), "repair_plan", fixtures.repair_plan(), fixtures.gap_analysis(), False),
            ("EvidenceCandidate.hit", fixtures.candidate(), "hit", fixtures.hit(), fixtures.document(), False),
            ("EvidenceCandidate.document", fixtures.candidate(), "document", fixtures.document(), fixtures.hit(), True),
            ("ValidationReport.draft", fixtures.validation_report(), "draft", fixtures.draft(), fixtures.answer_block(), False),
        )
        self.assertEqual(9, len(cases))
        for name, record, field_name, correct, wrong, optional in cases:
            shapes = (
                ("dict", {"field": "value"}, False),
                ("wrong_record", wrong, False),
                ("correct_record", correct, True),
                ("none", None, optional),
            )
            for shape, value, accepted in shapes:
                with self.subTest(field=name, shape=shape):
                    if accepted:
                        updated = replace(record, **{field_name: value})
                        self.assertIs(value, getattr(updated, field_name))
                    else:
                        with self.assertRaises(TypeError):
                            replace(record, **{field_name: value})

    def test_evidence_candidate_read_accounting_is_closed(self):
        m = models()
        hit = m.ProviderHit("p", "q", "t", "https://example.com", None, None, None, None, ())
        for consumed in (0, 1):
            candidate = m.EvidenceCandidate(hit, None, None, None, "snippet", (), consumed)
            self.assertEqual(consumed, candidate.content_reads_consumed)
        for consumed in (2, True, -1):
            with self.subTest(consumed=consumed):
                with self.assertRaises(ValueError):
                    m.EvidenceCandidate(hit, None, None, None, "snippet", (), consumed)

    def test_every_failure_code_pipeline_shape_is_closed(self):
        m = models()
        analysis = SearchModelFixtures(m).analysis()
        decision = self._search_decision(m)
        trace = m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT)
        for code in m.SearchFailureCode:
            with self.subTest(code=code):
                if code is m.SearchFailureCode.PROVIDER_NOT_CONFIGURED:
                    m.SearchPipelineResult(decision, object(), None, trace, code, analysis=analysis)
                else:
                    with self.assertRaises(ValueError):
                        m.SearchPipelineResult(decision, object(), None, trace, code, analysis=analysis)
        for code in (m.SearchFailureCode.PROVIDER_UNAVAILABLE, m.SearchFailureCode.PROVIDER_TIMEOUT, m.SearchFailureCode.NO_RESULTS, m.SearchFailureCode.CONTENT_UNREADABLE):
            bundle = type("Bundle", (), {"evidence_state": m.EvidenceState.INSUFFICIENT})()
            m.SearchPipelineResult(decision, object(), bundle, trace, code, analysis=analysis)
        for state, code in ((m.EvidenceState.SUFFICIENT, None), (m.EvidenceState.PARTIAL, m.SearchFailureCode.PARTIAL_EVIDENCE), (m.EvidenceState.CONFLICTING, m.SearchFailureCode.SOURCE_CONFLICT), (m.EvidenceState.INSUFFICIENT, m.SearchFailureCode.INSUFFICIENT_EVIDENCE)):
            m.SearchPipelineResult(decision, object(), type("Bundle", (), {"evidence_state": state})(), trace, code, analysis=analysis)
        for state in (
            m.EvidenceState.SUFFICIENT,
            m.EvidenceState.PARTIAL,
            m.EvidenceState.CONFLICTING,
        ):
            with self.subTest(state=state, code=m.SearchFailureCode.VALIDATION_FAILED):
                m.SearchPipelineResult(
                    decision,
                    object(),
                    type("Bundle", (), {"evidence_state": state})(),
                    trace,
                    m.SearchFailureCode.VALIDATION_FAILED,
                    analysis=analysis,
                )
        with self.assertRaises(ValueError):
            m.SearchPipelineResult(decision, object(), type("Bundle", (), {"evidence_state": m.EvidenceState.INSUFFICIENT})(), trace, m.SearchFailureCode.VALIDATION_FAILED, analysis=analysis)

    def test_provider_readiness_and_result_state_tables(self):
        m = models()
        valid_readiness_states = {
            (False, False, m.SearchFailureCode.PROVIDER_NOT_CONFIGURED),
            (True, False, m.SearchFailureCode.PROVIDER_UNAVAILABLE),
            (True, True, None),
        }
        accepted = 0
        reasons = (None, *tuple(m.SearchFailureCode))
        for configured in (False, True):
            for available in (False, True):
                for reason in reasons:
                    values = (configured, available, reason)
                    with self.subTest(
                        configured=configured, available=available, reason=reason
                    ):
                        if values in valid_readiness_states:
                            m.ProviderReadiness("provider", *values)
                            accepted += 1
                        else:
                            with self.assertRaises((TypeError, ValueError)):
                                m.ProviderReadiness("provider", *values)
        self.assertEqual(44, 4 * len(reasons))
        self.assertEqual(3, accepted)

        hit = m.ProviderHit("p", "q", "title", "https://example.com", None, None, None, None, ())
        m.ProviderResult("p", m.ProviderStatus.SUCCESS, [hit], 1)
        m.ProviderResult("p", m.ProviderStatus.EMPTY, [], 1)
        for status, hits in ((m.ProviderStatus.SUCCESS, []), (m.ProviderStatus.EMPTY, [hit]), (m.ProviderStatus.ERROR, [hit])):
            with self.assertRaises(ValueError):
                m.ProviderResult("p", status, hits, 1)

    def test_collection_element_type_closure_for_shared_contract_patterns(self):
        m = models()
        hit = m.ProviderHit("p", "q", "title", "https://example.com", None, None, None, None, [])
        result = m.ProviderResult("p", m.ProviderStatus.SUCCESS, [hit], 1)
        mutable_hits = [hit]
        result = m.ProviderResult("p", m.ProviderStatus.SUCCESS, mutable_hits, 1)
        mutable_hits.clear()
        self.assertEqual((hit,), result.hits)
        with self.assertRaises(TypeError):
            m.ProviderResult("p", m.ProviderStatus.SUCCESS, [{"title": "not a hit"}], 1)
        query = m.SearchQuery("q", m.SearchRoundKind.INITIAL, m.QueryPurpose.DIRECT, "text", include_domains=["example.com"])
        domains = ["example.com"]
        query = m.SearchQuery("q", m.SearchRoundKind.INITIAL, m.QueryPurpose.DIRECT, "text", include_domains=domains)
        domains.append("changed.example")
        self.assertEqual(("example.com",), query.include_domains)
        with self.assertRaises(TypeError):
            m.SearchQuery("q", m.SearchRoundKind.INITIAL, m.QueryPurpose.DIRECT, "text", include_domains=[{}])

    @staticmethod
    def _search_decision(m):
        return m.RetrievalDecision(
            route=m.SearchTier.LIGHT,
            skip_reason=None,
            must_search=True,
            reason_codes=(),
        )


class RequestAnalysisContextContractTests(unittest.TestCase):
    """Task 3 contexts are closed, immutable, and intentionally independent."""

    @staticmethod
    def _retrieval(m, **overrides):
        values = {
            "must_search": True,
            "skip_reason": None,
            "factuality": m.Factuality.FACTUAL,
            "external_fact_required": True,
            "complexity_codes": (),
            "source_requirement": m.SourceRequirement.ANY_RELEVANT,
        }
        values.update(overrides)
        return m.RetrievalContext(**values)

    @staticmethod
    def _freshness(m, **overrides):
        values = {
            "requirement": m.FreshnessRequirement.NOT_REQUIRED,
            "as_of": None,
            "date_from": None,
            "date_to": None,
            "version_constraint": None,
        }
        values.update(overrides)
        return m.FreshnessContext(**values)

    @staticmethod
    def _risk(m, **overrides):
        values = {
            "high_consequence": False,
            "warning_required": False,
            "fail_closed": False,
        }
        values.update(overrides)
        return m.RiskContext(**values)

    def test_request_analysis_keeps_retrieval_freshness_and_risk_separate(self):
        m = models()
        analysis = m.RequestAnalysis(
            retrieval=self._retrieval(m),
            freshness=self._freshness(
                m,
                requirement=m.FreshnessRequirement.CURRENT,
            ),
            risk=self._risk(
                m,
                high_consequence=True,
                warning_required=True,
                fail_closed=True,
            ),
        )

        self.assertEqual((), analysis.retrieval.complexity_codes)
        self.assertTrue(analysis.risk.warning_required)
        self.assertIs(
            analysis.freshness.requirement,
            m.FreshnessRequirement.CURRENT,
        )
        with self.assertRaises(FrozenInstanceError):
            analysis.risk.warning_required = False

    def test_context_contracts_normalize_collections_and_reject_unknown_values(self):
        m = models()
        codes = [m.RetrievalComplexityCode.COMPARISON]
        retrieval = self._retrieval(m, complexity_codes=codes)
        codes.append(m.RetrievalComplexityCode.RECOMMENDATION)

        self.assertEqual(
            (m.RetrievalComplexityCode.COMPARISON,),
            retrieval.complexity_codes,
        )
        with self.assertRaises((TypeError, ValueError)):
            self._retrieval(m, complexity_codes=("comparison",))
        with self.assertRaises((TypeError, ValueError)):
            self._retrieval(m, source_requirement="any_relevant")
        with self.assertRaises((TypeError, ValueError)):
            self._retrieval(m, must_search="yes")

    def test_freshness_contract_closes_date_and_version_constraints(self):
        m = models()
        with self.assertRaises(ValueError):
            self._freshness(m, as_of=date(2026, 8, 11))
        with self.assertRaises(ValueError):
            self._freshness(
                m,
                requirement=m.FreshnessRequirement.WINDOW,
                date_from=date(2026, 8, 12),
                date_to=date(2026, 8, 11),
            )
        with self.assertRaises(ValueError):
            self._freshness(
                m,
                requirement=m.FreshnessRequirement.VERSION,
                version_constraint=" ",
            )
        version = self._freshness(
            m,
            requirement=m.FreshnessRequirement.VERSION,
            version_constraint="3.13",
        )
        self.assertEqual("3.13", version.version_constraint)

    def test_risk_contract_requires_high_consequence_for_warning(self):
        m = models()
        with self.assertRaises(ValueError):
            self._risk(m, warning_required=True)
        with self.assertRaises((TypeError, ValueError)):
            self._risk(m, high_consequence=1)

    def test_pipeline_result_retains_supplied_analysis_identity(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        analysis = m.RequestAnalysis(
            self._retrieval(m),
            self._freshness(m),
            self._risk(m),
        )
        partial_plan = self._structured_partial_plan(m, fixtures)
        supported_topic, missing_topic = partial_plan.required_topics
        evidence_item = fixtures.evidence_item()
        evidence = replace(
            fixtures.bundle(),
            decision=partial_plan.decision,
            plan=partial_plan,
            initial_evidence_ids=(evidence_item.evidence_id,),
            evidence_items=(evidence_item,),
            evidence_state=m.EvidenceState.PARTIAL,
            missing_claim_topics=(missing_topic.label,),
            topic_assessments=(
                m.TopicAssessment(
                    supported_topic.topic_id,
                    m.FreshnessEligibility.NOT_REQUIRED,
                    (evidence_item.evidence_id,),
                ),
                m.TopicAssessment(
                    missing_topic.topic_id,
                    m.FreshnessEligibility.NOT_REQUIRED,
                    (),
                ),
            ),
            supported_topic_ids=(supported_topic.topic_id,),
            missing_topic_ids=(missing_topic.topic_id,),
        )
        result = m.SearchPipelineResult(
            fixtures.decision(),
            fixtures.plan(),
            evidence,
            m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT),
            m.SearchFailureCode.PARTIAL_EVIDENCE,
            analysis=analysis,
        )

        self.assertIs(analysis, result.analysis)

    @staticmethod
    def _structured_partial_plan(m, fixtures):
        topics = (
            m.RequiredTopic(
                "topic-1", "question", True,
                m.FreshnessRequirement.NOT_REQUIRED,
            ),
            m.RequiredTopic(
                "topic-2", "detail", True,
                m.FreshnessRequirement.NOT_REQUIRED,
            ),
        )
        query = replace(
            fixtures.query(),
            target_topic_ids=("topic-1", "topic-2"),
        )
        return replace(fixtures.plan(), required_topics=topics, initial_queries=(query,))


class RequiredTopicAndQueryPlanContractTests(unittest.TestCase):
    """Task 4 contracts: material topics own query targets and freshness."""

    @staticmethod
    def _decision(m):
        return m.RetrievalDecision(
            route=m.SearchTier.STANDARD,
            skip_reason=None,
            must_search=True,
            reason_codes=(),
        )

    @staticmethod
    def _topic(m, topic_id="topic-1", label="并发 API", *, material=True,
               freshness_requirement=None, date_from=None, date_to=None,
               version_constraint=None, source_requirement=None):
        return m.RequiredTopic(
            topic_id=topic_id,
            label=label,
            material=material,
            freshness_requirement=(
                m.FreshnessRequirement.NOT_REQUIRED
                if freshness_requirement is None else freshness_requirement
            ),
            date_from=date_from,
            date_to=date_to,
            version_constraint=version_constraint,
            source_requirement=(
                m.SourceRequirement.ANY_RELEVANT
                if source_requirement is None else source_requirement
            ),
        )

    @staticmethod
    def _query(m, *, query_id="initial-1", query_index=1,
               purpose=None, targets=("topic-1",), text="比较并发 API"):
        return m.SearchQuery(
            query_id=query_id,
            round_kind=m.SearchRoundKind.INITIAL,
            purpose=m.QueryPurpose.DIRECT if purpose is None else purpose,
            text=text,
            query_index=query_index,
            target_topic_ids=targets,
        )

    def _plan(self, m, *, topics=None, queries=None):
        topics = (self._topic(m),) if topics is None else topics
        queries = (self._query(m),) if queries is None else queries
        return m.SearchPlan(
            decision=self._decision(m),
            original_question="比较并发 API",
            planning_status=m.PlanningStatus.NORMAL,
            entities=(),
            time_window=None,
            initial_queries=queries,
            required_topics=topics,
            required_source_relations=frozenset(),
            query_redaction_codes=(),
            budget=m.DEFAULT_TIER_BUDGETS[m.SearchTier.STANDARD],
        )

    def test_search_plan_rejects_legacy_topic_labels_and_unsealed_queries(self):
        m = models()
        with self.assertRaises(TypeError):
            self._plan(
                m,
                topics=("并发 API",),
                queries=(
                    m.SearchQuery(
                        "q1",
                        m.SearchRoundKind.INITIAL,
                        m.QueryPurpose.DIRECT,
                        "比较并发 API",
                    ),
                ),
            )

    def test_required_topic_validates_closed_freshness_source_and_labels(self):
        m = models()
        version = self._topic(
            m,
            label="  Python 3.13  ",
            freshness_requirement=m.FreshnessRequirement.VERSION,
            version_constraint=" 3.13 ",
            source_requirement=m.SourceRequirement.INDEPENDENT_CORROBORATION,
        )
        self.assertEqual("Python 3.13", version.label)
        self.assertEqual("3.13", version.version_constraint)
        self.assertIs(
            m.SourceRequirement.INDEPENDENT_CORROBORATION,
            version.source_requirement,
        )
        invalid_cases = (
            {"topic_id": " "},
            {"label": "\t"},
            {"material": "yes"},
            {"freshness_requirement": "current"},
            {"source_requirement": "any_relevant"},
            {"date_from": date(2026, 8, 12), "date_to": date(2026, 8, 11)},
            {"date_from": date(2026, 8, 11)},
            {
                "freshness_requirement": m.FreshnessRequirement.VERSION,
                "version_constraint": " ",
            },
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    self._topic(m, **overrides)

    def test_topic_identifiers_are_opaque_monotonic_slots(self):
        m = models()
        factories = (
            lambda: self._topic(m, topic_id="当前价格"),
            lambda: m.TopicAssessment(
                "price",
                m.FreshnessEligibility.NOT_REQUIRED,
                (),
            ),
            lambda: m.TopicFreshnessTraceEntry(
                "topic-0",
                m.FreshnessEligibility.NOT_REQUIRED,
            ),
            lambda: m.EvidenceGapAnalysis(
                ("price",),
                (),
                True,
                (m.RepairReasonCode.MISSING_TOPIC,),
                ("price",),
            ),
        )
        for factory in factories:
            with self.subTest(factory=factory), self.assertRaises(ValueError):
                factory()

    def test_structured_plan_closes_topic_ids_material_targets_and_query_slots(self):
        m = models()
        topics = (
            self._topic(m, "topic-1", "主张一"),
            self._topic(m, "topic-2", "背景", material=False),
            self._topic(m, "topic-3", "主张二"),
        )
        direct = self._query(m, targets=("topic-1", "topic-3"))
        supplement = self._query(
            m,
            query_id="initial-2",
            query_index=2,
            purpose=m.QueryPurpose.PRIMARY,
            targets=("topic-1",),
            text="主张一官方资料",
        )
        plan = self._plan(m, topics=topics, queries=(direct, supplement))
        self.assertEqual(("topic-1", "topic-2", "topic-3"), tuple(topic.topic_id for topic in plan.required_topics))
        self.assertEqual((1, 2), tuple(query.query_index for query in plan.initial_queries))

        invalid_plans = (
            {"topics": (self._topic(m, "topic-2", "跳号"),)},
            {"topics": tuple(self._topic(m, f"topic-{index}", f"主题{index}") for index in range(1, 5))},
            {"topics": (self._topic(m, material=False),)},
            {
                "topics": topics,
                "queries": (self._query(m, targets=("topic-1",)),),
            },
            {
                "topics": topics,
                "queries": (
                    direct,
                    self._query(
                        m,
                        query_id="initial-2",
                        query_index=2,
                        purpose=m.QueryPurpose.PRIMARY,
                        targets=("topic-2",),
                        text="背景资料",
                    ),
                ),
            },
            {
                "queries": (
                    self._query(m, query_index=2),
                ),
            },
        )
        for values in invalid_plans:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    self._plan(m, **values)

        with self.assertRaises((TypeError, ValueError)):
            self._plan(
                m,
                topics=(
                    self._topic(m, "topic-1", "主主题"),
                    self._topic(m, "topic-2", "非材料", material=False),
                ),
                queries=(self._query(m, targets=("topic-1", "topic-2")),),
            )

        with self.assertRaises((TypeError, ValueError)):
            self._query(m, query_index=0)

    def test_structured_plan_rejects_query_ids_outside_the_final_sealed_order(self):
        m = models()
        direct = self._query(m)
        supplement = self._query(
            m,
            query_id="initial-2",
            query_index=2,
            purpose=m.QueryPurpose.PRIMARY,
            text="并发 API 官方资料",
        )
        invalid_queries = (
            (self._query(m, query_id="initial-2"),),
            (direct, self._query(
                m,
                query_id="initial-3",
                query_index=2,
                purpose=m.QueryPurpose.PRIMARY,
                text="跳号资料",
            )),
            (direct, self._query(
                m,
                query_id="initial-1",
                query_index=2,
                purpose=m.QueryPurpose.PRIMARY,
                text="重复资料",
            )),
            (supplement,),
        )
        for queries in invalid_queries:
            with self.subTest(queries=queries):
                with self.assertRaises((TypeError, ValueError)):
                    self._plan(m, queries=queries)

    def test_structured_plan_requires_canonical_material_target_tuples(self):
        m = models()
        topics = (
            self._topic(m, "topic-1", "第一主张"),
            self._topic(m, "topic-2", "第二主张"),
            self._topic(m, "topic-3", "背景", material=False),
        )
        direct = self._query(m, targets=("topic-1", "topic-2"))
        supplement = self._query(
            m,
            query_id="initial-2",
            query_index=2,
            purpose=m.QueryPurpose.PRIMARY,
            targets=("topic-1",),
            text="第一主张官方资料",
        )
        invalid_queries = (
            (self._query(m, targets=("topic-2", "topic-1")),),
            (self._query(m, targets=("topic-1", "topic-2", "topic-1")),),
            (direct, self._query(
                m,
                query_id="initial-2",
                query_index=2,
                purpose=m.QueryPurpose.PRIMARY,
                targets=("topic-1", "topic-1"),
                text="重复目标",
            )),
            (direct, self._query(
                m,
                query_id="initial-2",
                query_index=2,
                purpose=m.QueryPurpose.PRIMARY,
                targets=("topic-2", "topic-1"),
                text="倒序目标",
            )),
        )
        self.assertEqual((direct, supplement), self._plan(
            m, topics=topics, queries=(direct, supplement)
        ).initial_queries)
        for queries in invalid_queries:
            with self.subTest(queries=queries):
                with self.assertRaises((TypeError, ValueError)):
                    self._plan(m, topics=topics, queries=queries)

    def test_legacy_topic_labels_do_not_admit_unsealed_structured_plans(self):
        m = models()
        legacy_query = m.SearchQuery(
            "q1",
            m.SearchRoundKind.INITIAL,
            m.QueryPurpose.DIRECT,
            "legacy query",
        )
        for topics in (
            ("legacy label",),
            (),
            ("legacy one", "legacy two", "legacy three"),
            ("legacy label", self._topic(m)),
        ):
            with self.subTest(topics=topics), self.assertRaises((TypeError, ValueError)):
                self._plan(m, topics=topics, queries=(legacy_query,))


class TopicAssessmentContractTests(unittest.TestCase):
    @staticmethod
    def _plan_with_topic(m, topic):
        decision = SearchModelFixtures(m).decision()
        return m.SearchPlan(
            decision,
            "question",
            m.PlanningStatus.NORMAL,
            (),
            None,
            (
                m.SearchQuery(
                    "initial-1",
                    m.SearchRoundKind.INITIAL,
                    m.QueryPurpose.DIRECT,
                    "question",
                    query_index=1,
                    target_topic_ids=("topic-1",),
                ),
            ),
            (topic,),
            frozenset(),
            (),
            m.DEFAULT_TIER_BUDGETS[m.SearchTier.LIGHT],
        )

    def test_assessment_freshness_cannot_misrepresent_topic_requirement(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        cases = (
            (
                m.RequiredTopic(
                    "topic-1",
                    "current",
                    True,
                    m.FreshnessRequirement.CURRENT,
                ),
                m.FreshnessEligibility.NOT_REQUIRED,
            ),
            (
                m.RequiredTopic(
                    "topic-1",
                    "version",
                    True,
                    m.FreshnessRequirement.VERSION,
                    version_constraint="3.13",
                ),
                m.FreshnessEligibility.NOT_REQUIRED,
            ),
            (
                m.RequiredTopic(
                    "topic-1",
                    "stable",
                    True,
                    m.FreshnessRequirement.NOT_REQUIRED,
                ),
                m.FreshnessEligibility.SATISFIED,
            ),
        )
        for topic, freshness in cases:
            with self.subTest(topic=topic.label, freshness=freshness):
                plan = self._plan_with_topic(m, topic)
                with self.assertRaises(ValueError):
                    replace(
                        fixtures.bundle(),
                        decision=plan.decision,
                        plan=plan,
                        topic_assessments=(
                            m.TopicAssessment("topic-1", freshness, ()),
                        ),
                        supported_topic_ids=(),
                        missing_topic_ids=("topic-1",),
                        missing_claim_topics=(topic.label,),
                    )

    def test_topic_support_requires_citable_relevance_gated_evidence(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        for evidence in (
            replace(fixtures.evidence_item(), citable=False),
            replace(fixtures.evidence_item(), relevance_gate_passed=False),
        ):
            with self.subTest(
                citable=evidence.citable,
                relevance_gate_passed=evidence.relevance_gate_passed,
            ):
                with self.assertRaises(ValueError):
                    replace(
                        fixtures.bundle(),
                        initial_evidence_ids=(evidence.evidence_id,),
            evidence_items=(evidence,),
            evidence_state=m.EvidenceState.SUFFICIENT,
            missing_claim_topics=(),
            topic_assessments=(
                            m.TopicAssessment(
                                "topic-1",
                                m.FreshnessEligibility.NOT_REQUIRED,
                                (evidence.evidence_id,),
                            ),
                        ),
                        supported_topic_ids=("topic-1",),
                        missing_topic_ids=(),
                    )

    def test_supporting_evidence_ids_and_bundle_evidence_ids_must_be_unique(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        with self.assertRaises(ValueError):
            m.TopicAssessment(
                "topic-1",
                m.FreshnessEligibility.NOT_REQUIRED,
                ("e1", "e1"),
            )
        evidence = fixtures.evidence_item()
        with self.assertRaises(ValueError):
            replace(
                fixtures.bundle(),
                evidence_items=(evidence, evidence),
            )

    def test_topic_support_must_match_legacy_evidence_label_projection(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        evidence = replace(
            fixtures.evidence_item(),
            supported_topics=("unrelated label",),
        )
        with self.assertRaises(ValueError):
            replace(
                fixtures.bundle(),
                initial_evidence_ids=(evidence.evidence_id,),
                evidence_items=(evidence,),
                evidence_state=m.EvidenceState.SUFFICIENT,
                missing_claim_topics=(),
                topic_assessments=(
                    m.TopicAssessment(
                        "topic-1",
                        m.FreshnessEligibility.NOT_REQUIRED,
                        (evidence.evidence_id,),
                    ),
                ),
                supported_topic_ids=("topic-1",),
                missing_topic_ids=(),
            )

    def test_legacy_missing_claim_topics_must_project_missing_topic_ids(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        with self.assertRaises(ValueError):
            replace(
                fixtures.bundle(),
                missing_claim_topics=("unrelated legacy label",),
            )

    def test_evidence_conflict_rejects_members_from_only_one_evidence(self):
        m = models()
        members = (
            m.EvidenceConflictMember("E1", "draft", None, "contradicts"),
            m.EvidenceConflictMember("E1", "published", None, "contradicts"),
        )
        with self.assertRaises(ValueError):
            m.EvidenceConflict("conflict:status", "status", members, topic_ids=("topic-1",))

    def test_evidence_conflict_rejects_members_asserting_only_one_value(self):
        m = models()
        for values in (("published", "published"), ("1", " 1 ")):
            with self.subTest(values=values):
                members = (
                    m.EvidenceConflictMember("E1", values[0], None, "contradicts"),
                    m.EvidenceConflictMember("E2", values[1], None, "contradicts"),
                )
                with self.assertRaises(ValueError):
                    m.EvidenceConflict("conflict:status", "status", members, topic_ids=("topic-1",))

    def test_bundle_rejects_state_that_disagrees_with_evidence_priority(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        first = fixtures.evidence_item()
        second = replace(first, evidence_id="E2")
        conflict = m.EvidenceConflict(
            "conflict:version",
            "version",
            (
                m.EvidenceConflictMember("e1", "1", None, "contradicts"),
                m.EvidenceConflictMember("E2", "2", None, "contradicts"),
            ),
            topic_ids=("topic-1",),
        )
        supported = {
            "initial_evidence_ids": ("e1", "E2"),
            "evidence_items": (first, second),
            "missing_claim_topics": (),
            "topic_assessments": (
                m.TopicAssessment(
                    "topic-1",
                    m.FreshnessEligibility.NOT_REQUIRED,
                    ("e1",),
                ),
            ),
            "supported_topic_ids": ("topic-1",),
            "missing_topic_ids": (),
        }
        cases = (
            (
                "sufficient_with_conflict",
                supported
                | {
                    "evidence_state": m.EvidenceState.SUFFICIENT,
                    "conflict_groups": ("conflict:version",),
                    "conflicts": (conflict,),
                },
            ),
            (
                "partial_with_all_topics_supported",
                supported | {"evidence_state": m.EvidenceState.PARTIAL},
            ),
            (
                "insufficient_with_supported_topic",
                supported | {"evidence_state": m.EvidenceState.INSUFFICIENT},
            ),
            (
                "conflicting_without_conflict",
                {"evidence_state": m.EvidenceState.CONFLICTING},
            ),
        )
        for name, overrides in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(fixtures.bundle(), **overrides)

    def test_bundle_conflict_groups_match_unique_conflict_ids_in_order(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        first = fixtures.evidence_item()
        second = replace(first, evidence_id="E2")
        members = (
            m.EvidenceConflictMember("e1", "1", None, "contradicts"),
            m.EvidenceConflictMember("E2", "2", None, "contradicts"),
        )
        first_conflict = m.EvidenceConflict("conflict-1", "version", members, topic_ids=("topic-1",))
        second_conflict = m.EvidenceConflict("conflict-2", "status", members, topic_ids=("topic-1",))
        duplicate_id = m.EvidenceConflict("conflict-1", "status", members, topic_ids=("topic-1",))
        cases = (
            (("wrong-id",), (first_conflict,)),
            (("conflict-2", "conflict-1"), (first_conflict, second_conflict)),
            (("conflict-1", "conflict-1"), (first_conflict, duplicate_id)),
        )
        for conflict_groups, conflicts in cases:
            with self.subTest(conflict_groups=conflict_groups), self.assertRaises(ValueError):
                replace(
                    fixtures.bundle(),
                    initial_evidence_ids=("e1", "E2"),
                    evidence_items=(first, second),
                    evidence_state=m.EvidenceState.CONFLICTING,
                    conflict_groups=conflict_groups,
                    conflicts=conflicts,
                )

    def test_bundle_conflict_members_reference_citable_relevant_evidence(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        first = fixtures.evidence_item()
        second = replace(first, evidence_id="E2")
        cases = (
            (
                (first, second),
                (
                    m.EvidenceConflictMember("missing", "1", None, "contradicts"),
                    m.EvidenceConflictMember("E2", "2", None, "contradicts"),
                ),
            ),
            (
                (replace(first, citable=False), second),
                (
                    m.EvidenceConflictMember("e1", "1", None, "contradicts"),
                    m.EvidenceConflictMember("E2", "2", None, "contradicts"),
                ),
            ),
            (
                (replace(first, relevance_gate_passed=False), second),
                (
                    m.EvidenceConflictMember("e1", "1", None, "contradicts"),
                    m.EvidenceConflictMember("E2", "2", None, "contradicts"),
                ),
            ),
        )
        for evidence_items, members in cases:
            conflict = m.EvidenceConflict("conflict-1", "version", members, topic_ids=("topic-1",))
            with self.subTest(evidence_items=evidence_items), self.assertRaises(ValueError):
                replace(
                    fixtures.bundle(),
                    initial_evidence_ids=tuple(
                        item.evidence_id for item in evidence_items
                    ),
                    evidence_items=evidence_items,
                    evidence_state=m.EvidenceState.CONFLICTING,
                    conflict_groups=("conflict-1",),
                    conflicts=(conflict,),
                )

    def test_bundle_rejects_conflict_topic_without_two_member_topic_supports(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        first = fixtures.evidence_item()
        second = replace(first, evidence_id="E2", supported_topics=())
        conflict = m.EvidenceConflict(
            "conflict-1",
            "version",
            (
                m.EvidenceConflictMember("e1", "1", None, "contradicts"),
                m.EvidenceConflictMember("E2", "2", None, "contradicts"),
            ),
            topic_ids=("topic-1",),
        )

        with self.assertRaisesRegex(
            ValueError,
            "conflict topic must be supported by at least two conflict members",
        ):
            replace(
                fixtures.bundle(),
                initial_evidence_ids=("e1", "E2"),
                evidence_items=(first, second),
                evidence_state=m.EvidenceState.CONFLICTING,
                conflict_groups=("conflict-1",),
                conflicts=(conflict,),
            )

    def test_sufficient_bundle_rejects_stale_or_unknown_material_assessment(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        evidence = fixtures.evidence_item()
        for freshness in (m.FreshnessEligibility.STALE, m.FreshnessEligibility.UNKNOWN):
            with self.subTest(freshness=freshness):
                assessment = m.TopicAssessment("topic-1", freshness, (evidence.evidence_id,))
                with self.assertRaises(ValueError):
                    replace(
                        fixtures.bundle(),
                        evidence_items=(evidence,),
                        evidence_state=m.EvidenceState.SUFFICIENT,
                        topic_assessments=(assessment,),
                        supported_topic_ids=("topic-1",),
                        missing_topic_ids=(),
                    )


class RepairContractTests(unittest.TestCase):
    """Task 6: closed repair reasons, targets, stop reasons, and Trace metadata."""

    def test_repair_reason_codes_are_closed(self):
        m = models()
        expected = {
            "missing_topic",
            "stale_evidence",
            "source_conflict",
            "entity_ambiguity",
            "premise_mismatch",
            "source_quality_gap",
            "content_unreadable",
        }
        self.assertEqual(expected, {item.value for item in m.RepairReasonCode})
        with self.assertRaises(ValueError):
            m.RepairReasonCode("invented_reason")

    def test_retrieval_stop_reasons_are_closed(self):
        m = models()
        expected = {
            "evidence_sufficient",
            "no_repair_benefit",
            "budget_exhausted",
            "post_repair_stop",
        }
        self.assertEqual(expected, {item.value for item in m.RetrievalStopReason})
        with self.assertRaises(ValueError):
            m.RetrievalStopReason("invented_stop")

    def test_gap_analysis_closes_reason_codes_and_target_topic_ids(self):
        m = models()
        gap = m.EvidenceGapAnalysis(
            ("topic-2",),
            ("conflict:version",),
            True,
            (m.RepairReasonCode.MISSING_TOPIC,),
            ("topic-2",),
        )
        self.assertEqual(("topic-2",), gap.missing_topic_ids)
        self.assertEqual(("topic-2",), gap.repair_target_topic_ids)
        self.assertIs(gap.repair_reason_codes[0], m.RepairReasonCode.MISSING_TOPIC)
        with self.assertRaises((TypeError, ValueError)):
            m.EvidenceGapAnalysis(("topic-2",), (), True, ("missing_topic",), ("topic-2",))
        with self.assertRaises(ValueError):
            m.EvidenceGapAnalysis((), (), False, (), ("topic-2",))
        with self.assertRaises(ValueError):
            m.EvidenceGapAnalysis(("topic-2",), (), True, (m.RepairReasonCode.MISSING_TOPIC,), ())

    def test_repair_plan_requires_reason_and_target_only_when_triggered(self):
        m = models()
        query = SearchModelFixtures(m).query()
        with self.assertRaises(ValueError):
            m.RepairPlan(True, (), ("topic-2",), query)
        with self.assertRaises(ValueError):
            m.RepairPlan(True, (m.RepairReasonCode.MISSING_TOPIC,), (), query)
        with self.assertRaises(ValueError):
            m.RepairPlan(False, (m.RepairReasonCode.MISSING_TOPIC,), ("topic-2",), None)
        valid = m.RepairPlan(True, (m.RepairReasonCode.MISSING_TOPIC,), ("topic-2",), query)
        self.assertTrue(valid.triggered)

    def test_query_trace_entry_is_closed_metadata(self):
        m = models()
        entry = m.QueryTraceEntry(
            1,
            m.QueryPurpose.DIRECT,
            m.SearchRoundKind.INITIAL,
            "ddgs",
            m.ProviderStatus.SUCCESS,
            12.5,
        )
        self.assertEqual(1, entry.query_index)
        with self.assertRaises(ValueError):
            m.QueryTraceEntry(
                0, m.QueryPurpose.DIRECT, m.SearchRoundKind.INITIAL,
                "ddgs", m.ProviderStatus.SUCCESS, 12.5,
            )
        with self.assertRaises((TypeError, ValueError)):
            m.QueryTraceEntry(
                1, m.QueryPurpose.DIRECT, m.SearchRoundKind.INITIAL,
                "ddgs", "success", 12.5,
            )

    def test_trace_counts_unique_indexes_across_provider_fallback(self):
        m = models()
        trace = m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.STANDARD)
        trace.executed_queries = (
            m.QueryTraceEntry(1, m.QueryPurpose.DIRECT, m.SearchRoundKind.INITIAL, "ddgs", m.ProviderStatus.EMPTY, 5),
            m.QueryTraceEntry(1, m.QueryPurpose.DIRECT, m.SearchRoundKind.INITIAL, "tavily", m.ProviderStatus.SUCCESS, 9),
            m.QueryTraceEntry(4, m.QueryPurpose.REPAIR, m.SearchRoundKind.REPAIR, "tavily", m.ProviderStatus.SUCCESS, 7),
        )
        self.assertEqual(2, trace.semantic_query_count)
        self.assertEqual(1, trace.repair_query_count)
        logged = trace.to_log_dict()
        self.assertEqual(2, logged["semantic_query_count"])
        self.assertEqual(1, logged["repair_query_count"])
        self.assertEqual([1, 1, 4], [row["query_index"] for row in logged["executed_queries"]])

    def test_trace_stop_reason_is_logged(self):
        m = models()
        trace = m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.STANDARD)
        trace.retrieval_stop_reason = m.RetrievalStopReason.POST_REPAIR_STOP
        self.assertEqual("post_repair_stop", trace.to_log_dict()["retrieval_stop_reason"])

    def test_trace_rejects_raw_query_values(self):
        m = models()
        with self.assertRaises(TypeError):
            m.SearchTrace(
                "req-1",
                m.RequestSource.CHAT,
                m.SearchTier.STANDARD,
                executed_queries=(("initial-1", m.QueryPurpose.DIRECT),),
            )


if __name__ == "__main__":
    unittest.main()
