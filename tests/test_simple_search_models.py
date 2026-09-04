import math
from pathlib import Path
import unittest

from src.config import Config
from src.search.simple.models import (
    OutputKind,
    RequestSource,
    SearchMode,
    SearchPlan,
    SearchQuery,
    SearchRequest,
    SearchTrace,
)


class SimpleSearchModelTests(unittest.TestCase):
    def test_request_normalizes_text_images_and_owns_mode(self):
        request = SearchRequest(
            mode=SearchMode.STANDARD,
            text="  看看   这个  ",
            images=[" data:image/png;base64,AAA ", ""],
            source=RequestSource.COMMAND,
        )
        self.assertIs(SearchMode.STANDARD, request.mode)
        self.assertEqual("看看 这个", request.text)
        self.assertEqual(("data:image/png;base64,AAA",), request.images)
        self.assertFalse(hasattr(request, "force_" + "search"))
        self.assertFalse(hasattr(request, "has_" + "images"))

    def test_text_or_image_is_required(self):
        with self.assertRaisesRegex(ValueError, "text or images"):
            SearchRequest(mode=SearchMode.LIGHT, text="", images=())

    def test_plan_enforces_fixed_query_counts(self):
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.LIGHT, ())
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.LIGHT, (SearchQuery("q1", "a"), SearchQuery("q2", "b")))
        with self.assertRaises(ValueError):
            SearchPlan(SearchMode.STANDARD, tuple(SearchQuery(f"q{i}", str(i)) for i in range(4)))
        self.assertEqual((), SearchPlan(SearchMode.SKIP, ()).queries)

    def test_safe_trace_has_closed_metadata_and_no_request_content(self):
        trace = SearchTrace(
            "r1",
            source=RequestSource.CHAT,
            mode=SearchMode.LIGHT,
            query_count=1,
        )
        trace.provider_statuses["tavily"] = "error"
        trace.output_kind = OutputKind.SEARCH_FAILURE
        safe = trace.to_safe_dict()
        self.assertEqual("chat", safe["source"])
        self.assertEqual("light", safe["mode"])
        self.assertEqual("error", safe["provider_statuses"]["tavily"])
        self.assertNotIn("text", safe)
        self.assertNotIn("images", safe)
        self.assertNotIn("url", repr(safe).lower())

    def test_all_search_timeouts_are_finite_and_at_least_point_one(self):
        fields = (
            "search_planner_timeout",
            "search_tavily_timeout",
            "search_ddgs_timeout",
            "search_reader_timeout",
            "search_ranker_timeout",
            "search_answer_timeout",
        )
        for field in fields:
            for value in (math.nan, math.inf, -math.inf, 0.0, -1.0):
                with self.subTest(field=field, value=value):
                    self.assertEqual(0.1, getattr(Config(**{field: value}), field))


LEGACY_RUNTIME_PATHS = (
    "src/search/providers/__init__.py",
    "src/search/providers/base.py",
    "src/search/providers/tavily.py",
    "src/search/providers/ddgs.py",
    "src/search/budget.py",
    "src/search/evidence.py",
    "src/search/extraction.py",
    "src/search/models.py",
    "src/search/orchestrator.py",
    "src/search/outcomes.py",
    "src/search/planner.py",
    "src/search/policy.py",
    "src/search/renderer.py",
    "src/search/router.py",
    "src/search/stage_runner.py",
    "src/search/validation.py",
    "tests/search_fakes.py",
    "tests/test_chat_retrieval_flow.py",
    "tests/test_search_blind_acceptance_runner.py",
    "tests/test_search_budget.py",
    "tests/test_search_evidence.py",
    "tests/test_search_extraction.py",
    "tests/test_search_models.py",
    "tests/test_search_orchestrator.py",
    "tests/test_search_outcomes.py",
    "tests/test_search_planner.py",
    "tests/test_search_policy.py",
    "tests/test_search_provider_batches.py",
    "tests/test_search_providers.py",
    "tests/test_search_renderer.py",
    "tests/test_search_router.py",
    "tests/test_search_simplification_baseline.py",
    "tests/test_search_stage_runner.py",
    "tests/test_search_validation.py",
    "tools/run_search_blind_acceptance.py",
)

LEGACY_IMPORTS = (
    "src.search." + "providers",
    "src.search." + "models",
    "src.search." + "orchestrator",
    "src.search." + "planner",
    "src.search." + "router",
    "src.search." + "evidence",
    "src.search." + "validation",
    "src.search." + "policy",
    "src.search." + "renderer",
    "src.search." + "outcomes",
    "src.search." + "budget",
    "src.search." + "stage_runner",
    "src.search." + "extraction",
)

LEGACY_SYMBOLS = (
    "Route" + "Planner",
    "force_" + "search",
    "has_" + "images",
    "Search" + "Tier",
    "Grounded" + "Draft",
    "Claim" + "Discovery",
    "Semantic" + "Verifier",
    "Repair" + "Plan",
    "fail_" + "closed",
    "fresh" + "ness",
    "risk_" + "policy",
)


class LegacyRemovalTests(unittest.TestCase):
    def test_legacy_runtime_paths_are_absent(self):
        for relative in LEGACY_RUNTIME_PATHS:
            self.assertFalse(Path(relative).exists(), relative)

    def test_live_tree_has_no_legacy_imports_or_symbols(self):
        roots = tuple(Path(root) for root in ("src", "tests", "tools"))
        this_test = Path(__file__).resolve()
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for root in roots for path in root.rglob("*.py")
            if path.resolve() != this_test
        )
        for forbidden in LEGACY_IMPORTS + LEGACY_SYMBOLS:
            self.assertNotIn(forbidden, source)
