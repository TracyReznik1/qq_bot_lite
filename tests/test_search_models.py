import importlib
import importlib.util
import json
import unittest
from dataclasses import FrozenInstanceError


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
            provider_attempts=(
                {
                    "provider": "fake",
                    "status": m.ProviderStatus.SUCCESS,
                    "count": 1,
                    "latency_ms": 3,
                    "exception": "https://example.invalid/private-error",
                    "raw_query": "raw query text",
                },
            ),
            evidence_state=m.EvidenceState.SUFFICIENT,
            provider_invocation_started=True,
            content_read_count=2,
        )

        logged = trace.to_log_dict()
        self.assertEqual(2, logged["semantic_query_count"])
        self.assertEqual(1, logged["repair_query_count"])
        self.assertTrue(logged["provider_attempted"])
        self.assertTrue(logged["sufficient_evidence"])
        self.assertEqual(
            {"query_id": "repair-1", "purpose": "repair"},
            logged["adaptive_repair_query"],
        )
        self.assertNotIn("question", logged)
        self.assertNotIn("answer", logged)
        payload = json.dumps(logged).lower()
        self.assertNotIn("https://example.invalid/private-error", payload)
        self.assertNotIn("raw query text", payload)
        json.dumps(logged)


if __name__ == "__main__":
    unittest.main()
