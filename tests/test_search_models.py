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
            m.SearchTier.LIGHT, None, False, (), frozenset(), m.Factuality.FACTUAL,
            True, m.Freshness.NONE, m.RiskLevel.LOW, m.Actionability.NONE,
            m.PotentialHarm.NONE, m.SearchTier.LIGHT, None, (),
        )

    def budget(self):
        return self.m.DEFAULT_TIER_BUDGETS[self.m.SearchTier.LIGHT]

    def query(self):
        m = self.m
        return m.SearchQuery(
            "q1", m.SearchRoundKind.INITIAL, m.QueryPurpose.DIRECT, "query text"
        )

    def plan(self):
        m = self.m
        return m.SearchPlan(
            self.decision(), "question", m.PlanningStatus.NORMAL, (), None,
            (self.query(),), (), frozenset(), (), self.budget(),
        )

    def repair_plan(self):
        return self.m.RepairPlan(False, (), None)

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
            "ok", 1.0, 1.0, True, m.Freshness.NONE, True, (), (), "source-1",
        )

    def gap_analysis(self):
        return self.m.EvidenceGapAnalysis((), (), False, None, ())

    def bundle(self):
        m = self.m
        return m.EvidenceBundle(
            "req-1", self.decision(), self.plan(), (), (), self.gap_analysis(),
            self.repair_plan(), 1, (), m.EvidenceState.INSUFFICIENT,
            (), (), (), (),
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
        elif location == "adaptive_repair_query.query_id":
            trace.adaptive_repair_query = (value, m.QueryPurpose.REPAIR)
        elif location == "executed_queries[].query_id":
            trace.executed_queries = ((value, m.QueryPurpose.DIRECT),)
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
        if location == "adaptive_repair_query.query_id":
            return logged["adaptive_repair_query"]["query_id"]
        if location == "executed_queries[].query_id":
            return logged["executed_queries"][0]["query_id"]
        if location == "provider_attempts[].provider":
            return logged["provider_attempts"][0]["provider"]
        raise AssertionError(f"unknown Trace location: {location}")


class SearchModelContractTests(unittest.TestCase):
    def test_00_models_module_is_available(self):
        self.assertIsNotNone(model_spec())

    def test_closed_enum_values_reject_unknown_strings(self):
        m = models()

        expected = {
            "SearchTier": {"skip", "light", "standard", "deep"},
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
            "Freshness": {"none", "low", "high"},
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
        }
        for enum_name, values in expected.items():
            enum_type = getattr(m, enum_name)
            with self.subTest(enum_name=enum_name):
                self.assertEqual(values, {item.value for item in enum_type})
                with self.assertRaises(ValueError):
                    enum_type("invented_value")

    def test_decisions_accept_all_routes_and_expose_conflict_clarification(self):
        m = models()
        common = dict(
            forced_search=False,
            trigger_codes=(),
            benefit_dimensions=frozenset(),
            factuality=m.Factuality.NON_FACTUAL,
            external_fact_required=False,
            freshness=m.Freshness.NONE,
            risk=m.RiskLevel.LOW,
            actionability=m.Actionability.NONE,
            potential_harm=m.PotentialHarm.NONE,
            model_recommended_tier=None,
            final_reason_codes=(),
        )

        skipped = m.RetrievalDecision(
            route=m.SearchTier.SKIP,
            skip_reason=m.SkipReason.PURE_MATH,
            program_minimum_tier=None,
            **common,
        )
        self.assertFalse(skipped.requires_clarification)
        for tier in (m.SearchTier.LIGHT, m.SearchTier.STANDARD, m.SearchTier.DEEP):
            with self.subTest(tier=tier):
                decision = m.RetrievalDecision(
                    route=tier,
                    skip_reason=None,
                    program_minimum_tier=tier,
                    **common,
                )
                self.assertEqual(tier, decision.route)

        conflict = m.RetrievalDecision(
            route=m.SearchTier.SKIP,
            skip_reason=m.SkipReason.USER_FORBID_WEB,
            forced_search=True,
            trigger_codes=(m.TriggerCode.EXPLICIT_NO_WEB, m.TriggerCode.EXPLICIT_SEARCH),
            benefit_dimensions=frozenset(),
            factuality=m.Factuality.AMBIGUOUS,
            external_fact_required=False,
            freshness=m.Freshness.NONE,
            risk=m.RiskLevel.LOW,
            actionability=m.Actionability.NONE,
            potential_harm=m.PotentialHarm.NONE,
            program_minimum_tier=None,
            model_recommended_tier=None,
            final_reason_codes=(m.TriggerCode.EXPLICIT_NO_WEB, m.TriggerCode.EXPLICIT_SEARCH),
        )
        self.assertTrue(conflict.requires_clarification)

        # Task 3 carries the force/explicit-search conflict in RetrievalContext.
        # Retained decision reason codes are diagnostic only and must not erase
        # the clarification state when their legacy set is minimal.
        minimal_conflict = m.RetrievalDecision(
            route=m.SearchTier.SKIP,
            skip_reason=m.SkipReason.USER_FORBID_WEB,
            forced_search=True,
            trigger_codes=(m.TriggerCode.EXPLICIT_NO_WEB,),
            benefit_dimensions=frozenset(),
            factuality=m.Factuality.AMBIGUOUS,
            external_fact_required=False,
            freshness=m.Freshness.NONE,
            risk=m.RiskLevel.LOW,
            actionability=m.Actionability.NONE,
            potential_harm=m.PotentialHarm.NONE,
            program_minimum_tier=None,
            model_recommended_tier=None,
            final_reason_codes=(m.TriggerCode.EXPLICIT_NO_WEB,),
        )
        self.assertTrue(minimal_conflict.requires_clarification)

    def test_decision_rejects_illegal_route_combinations_and_free_text_codes(self):
        m = models()
        common = dict(
            forced_search=False,
            trigger_codes=(),
            benefit_dimensions=frozenset(),
            factuality=m.Factuality.FACTUAL,
            external_fact_required=True,
            freshness=m.Freshness.NONE,
            risk=m.RiskLevel.LOW,
            actionability=m.Actionability.NONE,
            potential_harm=m.PotentialHarm.NONE,
            model_recommended_tier=None,
            final_reason_codes=(),
        )
        invalid = (
            dict(route=m.SearchTier.SKIP, skip_reason=None, program_minimum_tier=None),
            dict(route=m.SearchTier.LIGHT, skip_reason=m.SkipReason.PURE_MATH, program_minimum_tier=m.SearchTier.LIGHT),
            dict(route=m.SearchTier.LIGHT, skip_reason=None, program_minimum_tier=None),
            dict(route=m.SearchTier.LIGHT, skip_reason=None, program_minimum_tier=m.SearchTier.STANDARD),
            dict(route=m.SearchTier.LIGHT, skip_reason=None, program_minimum_tier=m.SearchTier.SKIP),
            dict(route=m.SearchTier.SKIP, skip_reason=m.SkipReason.PURE_MATH, program_minimum_tier=None, forced_search=True),
            dict(route=m.SearchTier.LIGHT, skip_reason=None, program_minimum_tier=m.SearchTier.LIGHT, trigger_codes=("free_text",)),
            dict(route=m.SearchTier.LIGHT, skip_reason=None, program_minimum_tier=m.SearchTier.LIGHT, final_reason_codes=("free_text",)),
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                values = common | overrides
                with self.assertRaises((TypeError, ValueError)):
                    m.RetrievalDecision(**values)

    def test_budgets_are_immutable_and_validate_derived_totals(self):
        m = models()

        self.assertEqual(
            (1, 5, 2, 0, 1, 1, 8),
            tuple(m.DEFAULT_TIER_BUDGETS[m.SearchTier.LIGHT].__dict__.values()),
        )
        self.assertEqual(
            (3, 8, 5, 1, 4, 2, 20),
            tuple(m.DEFAULT_TIER_BUDGETS[m.SearchTier.STANDARD].__dict__.values()),
        )
        self.assertEqual(
            (5, 15, 8, 1, 6, 2, 40),
            tuple(m.DEFAULT_TIER_BUDGETS[m.SearchTier.DEEP].__dict__.values()),
        )
        with self.assertRaises(TypeError):
            m.DEFAULT_TIER_BUDGETS[m.SearchTier.LIGHT] = m.DEFAULT_TIER_BUDGETS[m.SearchTier.LIGHT]
        with self.assertRaises(FrozenInstanceError):
            m.DEFAULT_TIER_BUDGETS[m.SearchTier.LIGHT].max_content_reads = 99
        with self.assertRaises(ValueError):
            m.TierBudget(1, 5, 2, 1, 1, 2, 8)

    def test_max_tier_returns_the_higher_closed_tier(self):
        m = models()

        self.assertEqual(m.SearchTier.DEEP, m.max_tier(m.SearchTier.STANDARD, m.SearchTier.DEEP))
        self.assertEqual(m.SearchTier.LIGHT, m.max_tier(m.SearchTier.SKIP, m.SearchTier.LIGHT))

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
            trigger_codes=(m.TriggerCode.FACTUAL_DEFAULT,),
            factuality=m.Factuality.FACTUAL,
            external_fact_required=True,
            program_minimum_tier=m.SearchTier.STANDARD,
            final_tier=m.SearchTier.STANDARD,
            adaptive_repair_round_started=True,
            adaptive_repair_query=("repair-1", m.QueryPurpose.REPAIR),
            executed_queries=(
                ("initial-1", m.QueryPurpose.DIRECT),
                ("initial-1", m.QueryPurpose.DIRECT),
                ("repair-1", m.QueryPurpose.REPAIR),
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
            {"query_id": "repair-1", "purpose": "repair"},
            logged["adaptive_repair_query"],
        )
        self.assertEqual(["cq_control_code", "data_url"], logged.get("initial_query_redaction_codes"))
        self.assertEqual(["callback_secret"], logged.get("adaptive_repair_redaction_codes"))
        self.assertNotIn("question", logged)
        self.assertNotIn("answer", logged)
        payload = json.dumps(logged).lower()
        self.assertNotIn("https://example.invalid/private-error", payload)
        self.assertNotIn("raw query text", payload)
        json.dumps(logged)

    def test_trace_serialization_is_total_for_deeply_malformed_query_metadata(self):
        m = models()
        trace = m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.STANDARD)
        trace.executed_queries = (
            ("initial-1", m.QueryPurpose.DIRECT),
            (("repair-1", m.QueryPurpose.REPAIR),),
            {"query_id": "private query text"},
        )
        trace.adaptive_repair_query = (("repair-1", m.QueryPurpose.REPAIR),)

        logged = trace.to_log_dict()

        self.assertEqual(
            [{"query_id": "initial-1", "purpose": "direct"}],
            logged["executed_queries"],
        )
        self.assertIsNone(logged["adaptive_repair_query"])
        self.assertEqual(1, logged["semantic_query_count"])
        json.dumps(logged)

    def test_validation_failed_is_legal_for_a_sufficient_bundle(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        evidence = fixtures.evidence_item()
        sufficient = replace(
            fixtures.bundle(),
            initial_evidence_ids=(evidence.evidence_id,),
            evidence_items=(evidence,),
            evidence_state=m.EvidenceState.SUFFICIENT,
        )
        result = m.SearchPipelineResult(
            fixtures.decision(),
            fixtures.plan(),
            sufficient,
            m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT),
            m.SearchFailureCode.VALIDATION_FAILED,
        )
        self.assertIs(result.failure_code, m.SearchFailureCode.VALIDATION_FAILED)

    def test_redaction_codes_are_closed_and_hostile_trace_mutation_stays_body_free(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        query = fixtures.query()
        for build in (
            lambda: replace(fixtures.plan(), query_redaction_codes=("sk-1234567890abcdef",)),
            lambda: m.RepairPlan(True, (), query, ("13800138000",)),
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

    def test_trace_legacy_positional_constructor_keeps_retrieval_round_slot(self):
        m = models()
        trace = m.SearchTrace(
            "req-1", m.RequestSource.CHAT, m.SearchTier.STANDARD,
            None, (), None, False, None, None, False, 0, False, False,
            ("repair-1", m.QueryPurpose.REPAIR), 1,
        )
        self.assertEqual(1, trace.retrieval_round_count)
        self.assertEqual((), trace.initial_query_redaction_codes)
        self.assertEqual((), trace.adaptive_repair_redaction_codes)

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
                "missing_claim_topics", "conflict_group_ids", "repair_eligible",
                "repair_purpose", "repair_reason_codes",
            ),
            "Claim": ("claim_id", "block_id", "text", "material", "evidence_ids"),
            "AnswerBlock": ("block_id", "kind", "text", "claim_ids"),
            "GroundedDraft": ("answer_blocks", "claims", "limitations", "conflict_summary", "used_knowledge_fallback"),
            "ValidationReport": ("draft", "retained_blocks", "retained_claims", "removed_block_ids", "claim_labels", "limitations"),
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
        self.assertEqual(
            ("topic",),
            m.EvidenceGapAnalysis(("topic",), [], True, "find source", ["missing"]).missing_claim_topics,
        )
        for values in (
            (("topic",), (), True, "", ("missing",)),
            (("topic",), (), True, "find source", ()),
            (("topic",), (), False, "find source", ()),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    m.EvidenceGapAnalysis(*values)

    def test_pipeline_result_rejects_ambiguous_search_and_skip_shapes(self):
        m = models()
        decision = self._search_decision(m)
        trace = m.SearchTrace("r", m.RequestSource.CHAT, m.SearchTier.LIGHT)
        with self.assertRaises(ValueError):
            m.SearchPipelineResult(decision, None, None, trace)
        skip = m.RetrievalDecision(
            m.SearchTier.SKIP, m.SkipReason.PURE_MATH, False, (), frozenset(),
            m.Factuality.NON_FACTUAL, False, m.Freshness.NONE, m.RiskLevel.LOW,
            m.Actionability.NONE, m.PotentialHarm.NONE, None, None, (),
        )
        with self.assertRaises(ValueError):
            m.SearchPipelineResult(skip, object(), None, trace)

    def test_frozen_contracts_normalize_caller_owned_collections_and_budget_rejects_bools(self):
        m = models()
        triggers = [m.TriggerCode.FACTUAL_DEFAULT]
        benefits = {m.BenefitDimension.ACCURACY}
        decision = m.RetrievalDecision(
            m.SearchTier.LIGHT, None, False, triggers, benefits, m.Factuality.FACTUAL,
            True, m.Freshness.NONE, m.RiskLevel.LOW, m.Actionability.NONE,
            m.PotentialHarm.NONE, m.SearchTier.LIGHT, None, triggers,
        )
        triggers.append(m.TriggerCode.EXPLICIT_SEARCH)
        benefits.add(m.BenefitDimension.FRESHNESS)
        self.assertEqual((m.TriggerCode.FACTUAL_DEFAULT,), decision.trigger_codes)
        self.assertEqual(frozenset({m.BenefitDimension.ACCURACY}), decision.benefit_dimensions)
        for index in range(7):
            values = [1, 5, 2, 0, 1, 1, 8]
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
            "adaptive_repair_query.query_id",
            "executed_queries[].query_id",
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
        for location in locations[:3]:
            for valid_id in valid_ids:
                with self.subTest(location=location, valid_id=valid_id):
                    logged = fixtures.trace_with_value(location, valid_id).to_log_dict()
                    self.assertEqual(valid_id, fixtures.logged_value(logged, location))
                    json.dumps(logged)
        for provider in ("tavily", "ddgs"):
            with self.subTest(location=locations[3], provider=provider):
                logged = fixtures.trace_with_value(locations[3], provider).to_log_dict()
                self.assertEqual(provider, fixtures.logged_value(logged, locations[3]))
                json.dumps(logged)

    def test_scalar_string_collection_fields_reject_all_21_scalar_inputs(self):
        m = models()
        fixtures = SearchModelFixtures(m)
        cases = (
            ("SearchQuery.include_domains", fixtures.query(), "include_domains"),
            ("SearchPlan.entities", fixtures.plan(), "entities"),
            ("SearchPlan.required_topics", fixtures.plan(), "required_topics"),
            ("SearchPlan.query_redaction_codes", fixtures.plan(), "query_redaction_codes"),
            ("RepairPlan.gap_codes", fixtures.repair_plan(), "gap_codes"),
            ("ProviderHit.quality_flags", fixtures.hit(), "quality_flags"),
            ("EvidenceItem.safety_flags", fixtures.evidence_item(), "safety_flags"),
            ("EvidenceItem.supported_topics", fixtures.evidence_item(), "supported_topics"),
            ("EvidenceGapAnalysis.missing_claim_topics", fixtures.gap_analysis(), "missing_claim_topics"),
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
        decision = self._search_decision(m)
        trace = m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT)
        for code in m.SearchFailureCode:
            with self.subTest(code=code):
                if code is m.SearchFailureCode.PROVIDER_NOT_CONFIGURED:
                    m.SearchPipelineResult(decision, object(), None, trace, code)
                else:
                    with self.assertRaises(ValueError):
                        m.SearchPipelineResult(decision, object(), None, trace, code)
        for code in (m.SearchFailureCode.PROVIDER_UNAVAILABLE, m.SearchFailureCode.PROVIDER_TIMEOUT, m.SearchFailureCode.NO_RESULTS, m.SearchFailureCode.CONTENT_UNREADABLE):
            bundle = type("Bundle", (), {"evidence_state": m.EvidenceState.INSUFFICIENT})()
            m.SearchPipelineResult(decision, object(), bundle, trace, code)
        for state, code in ((m.EvidenceState.SUFFICIENT, None), (m.EvidenceState.PARTIAL, m.SearchFailureCode.PARTIAL_EVIDENCE), (m.EvidenceState.CONFLICTING, m.SearchFailureCode.SOURCE_CONFLICT), (m.EvidenceState.INSUFFICIENT, m.SearchFailureCode.INSUFFICIENT_EVIDENCE)):
            m.SearchPipelineResult(decision, object(), type("Bundle", (), {"evidence_state": state})(), trace, code)
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
                )
        with self.assertRaises(ValueError):
            m.SearchPipelineResult(decision, object(), type("Bundle", (), {"evidence_state": m.EvidenceState.INSUFFICIENT})(), trace, m.SearchFailureCode.VALIDATION_FAILED)

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
            m.SearchTier.LIGHT, None, False, (), frozenset(), m.Factuality.FACTUAL,
            True, m.Freshness.NONE, m.RiskLevel.LOW, m.Actionability.NONE,
            m.PotentialHarm.NONE, m.SearchTier.LIGHT, None, (),
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
        evidence = replace(
            fixtures.bundle(),
            evidence_state=m.EvidenceState.PARTIAL,
        )
        result = m.SearchPipelineResult(
            fixtures.decision(),
            fixtures.plan(),
            evidence,
            m.SearchTrace("req-1", m.RequestSource.CHAT, m.SearchTier.LIGHT),
            m.SearchFailureCode.PARTIAL_EVIDENCE,
            analysis,
        )

        self.assertIs(analysis, result.analysis)


class RequiredTopicAndQueryPlanContractTests(unittest.TestCase):
    """Task 4 contracts: material topics own query targets and freshness."""

    @staticmethod
    def _decision(m):
        return m.RetrievalDecision(
            m.SearchTier.STANDARD, None, False, (), frozenset(),
            m.Factuality.FACTUAL, True, m.Freshness.NONE, m.RiskLevel.LOW,
            m.Actionability.NONE, m.PotentialHarm.NONE,
            m.SearchTier.STANDARD, None, (),
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

    def test_legacy_topic_labels_do_not_admit_unsealed_structured_plans(self):
        m = models()
        legacy_query = m.SearchQuery(
            "q1",
            m.SearchRoundKind.INITIAL,
            m.QueryPurpose.DIRECT,
            "legacy query",
        )
        legacy_plan = self._plan(
            m,
            topics=("legacy label",),
            queries=(legacy_query,),
        )
        self.assertEqual(("legacy label",), tuple(
            topic.label for topic in legacy_plan.required_topics
        ))
        self.assertTrue(all(topic.material for topic in legacy_plan.required_topics))
        replace(legacy_plan, original_question="replaced legacy question")

        empty_legacy = self._plan(m, topics=())
        self.assertEqual(1, len(empty_legacy.required_topics))
        self.assertTrue(empty_legacy.required_topics[0].material)

        three_legacy = self._plan(
            m,
            topics=("legacy one", "legacy two", "legacy three"),
            queries=(legacy_query,),
        )
        self.assertEqual(3, len(three_legacy.required_topics))
        with self.assertRaises((TypeError, ValueError)):
            self._plan(
                m,
                topics=("one", "two", "three", "four"),
                queries=(legacy_query,),
            )

        with self.assertRaises((TypeError, ValueError)):
            self._plan(m, topics=(self._topic(m),), queries=(legacy_query,))
        with self.assertRaises((TypeError, ValueError)):
            self._plan(m, topics=("legacy label", self._topic(m)))


if __name__ == "__main__":
    unittest.main()
