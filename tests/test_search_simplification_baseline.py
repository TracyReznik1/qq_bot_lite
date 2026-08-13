import inspect
from pathlib import Path
import unittest
from unittest.mock import Mock

from src.search.models import (
    DEFAULT_TIER_BUDGETS,
    ProviderReadiness,
    ProviderStatus,
    SearchTier,
)
from src.search.orchestrator import _repair_gates_pass
from src.search.providers.base import ProviderRegistry
from tests.test_search_providers import _provider_result, query


BASELINE_DOCUMENT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "superpowers"
    / "baselines"
    / "2026-08-09-websearch-simplification.md"
)


class SearchSimplificationBaselineTests(unittest.TestCase):
    def test_baseline_document_records_frozen_scope(self):
        self.assertTrue(
            BASELINE_DOCUMENT.is_file(),
            f"missing baseline artifact: {BASELINE_DOCUMENT}",
        )
        content = BASELINE_DOCUMENT.read_text(encoding="utf-8")
        required_fragments = (
            "Baseline implementation: `8abaa8f`",
            "light `1/5/2/0/1/1/8`; standard `3/8/5/1/4/2/20`",
            "DDGS-first/Tavily conditional fallback",
            "absolute deadline",
            "Reader",
            "Evidence relevance gate",
            "Claim/Citation validation",
            "body-free Trace",
            "operational deep",
            "risk/freshness tier floors",
            "explicit-search standard floor",
            "deep failure/validation branches",
            "140 owner-review rows are incomplete",
            "illegal `potential_harm=medium`",
            "online mode remains not run",
            "Hermetic package-aware suite: `842` tests passed",
            "Commit baseline: `9379da5`",
            "behavior baseline, not a real online quality certification",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, content)

    def test_light_and_standard_budget_caps_are_the_frozen_caps(self):
        light = DEFAULT_TIER_BUDGETS[SearchTier.LIGHT]
        standard = DEFAULT_TIER_BUDGETS[SearchTier.STANDARD]

        self.assertEqual(1, light.max_initial_queries)
        self.assertEqual(5, light.max_candidate_urls)
        self.assertEqual(2, light.max_content_reads)
        self.assertEqual(0, light.max_repair_queries)
        self.assertEqual(1, light.max_total_queries)
        self.assertEqual(1, light.max_retrieval_rounds)
        self.assertEqual(3, standard.max_initial_queries)
        self.assertEqual(8, standard.max_candidate_urls)
        self.assertEqual(5, standard.max_content_reads)
        self.assertEqual(1, standard.max_repair_queries)
        self.assertEqual(4, standard.max_total_queries)
        self.assertEqual(2, standard.max_retrieval_rounds)

    def test_budget_baseline_source_does_not_depend_on_dataclass_field_order(self):
        source = Path(__file__).read_text(encoding="utf-8")

        self.assertNotIn("__dict__" + ".values", source)

    def test_provider_registry_remains_ddgs_first_with_conditional_fallback(self):
        ddgs = Mock(name="ddgs")
        ddgs.name = "ddgs"
        ddgs.readiness.return_value = ProviderReadiness("ddgs", True, True, None)
        ddgs.search.return_value = _provider_result("ddgs")
        tavily = Mock(name="tavily")
        tavily.name = "tavily"
        tavily.readiness.return_value = ProviderReadiness("tavily", True, True, None)
        tavily.search.return_value = _provider_result("tavily")
        registry = ProviderRegistry((tavily, ddgs))

        result = registry.search(
            query(),
            tier=SearchTier.LIGHT,
            max_results=5,
            timeout_seconds=1.0,
        )

        self.assertEqual("ddgs", result.provider)
        ddgs.search.assert_called_once()
        tavily.search.assert_not_called()

        ddgs.reset_mock()
        tavily.reset_mock()
        ddgs.search.return_value = _provider_result(
            "ddgs",
            status=ProviderStatus.ERROR,
        )

        result = registry.search(
            query(),
            tier=SearchTier.LIGHT,
            max_results=5,
            timeout_seconds=1.0,
        )

        self.assertEqual("tavily", result.provider)
        ddgs.search.assert_called_once()
        tavily.search.assert_called_once()

    def test_repair_gate_is_a_program_function_not_an_llm_stage(self):
        self.assertTrue(inspect.isfunction(_repair_gates_pass))
        self.assertFalse(inspect.iscoroutinefunction(_repair_gates_pass))
