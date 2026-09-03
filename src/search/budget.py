"""Closed, independent time budgets for the bounded search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from types import MappingProxyType
from typing import Mapping

from src.search.models import SearchTier


@dataclass(frozen=True)
class RouteStageBudget:
    """Per-route stage time caps; unused stage time is never transferable."""

    analysis_route_seconds: int | float
    planner_seconds: int | float
    initial_ddgs_seconds: int | float
    initial_tavily_seconds: int | float
    initial_reader_seconds: int | float
    initial_judge_seconds: int | float
    gap_seconds: int | float
    repair_planner_seconds: int | float
    repair_ddgs_seconds: int | float
    repair_tavily_seconds: int | float
    repair_reader_seconds: int | float
    repair_judge_seconds: int | float
    answer_seconds: int | float
    validator_seconds: int | float
    renderer_seconds: int | float
    scheduling_margin_seconds: int | float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field.name} must be a finite non-negative number")


@dataclass(frozen=True)
class SearchBudgetPolicy:
    """An immutable route-to-stage-budget mapping with derived watchdogs."""

    budgets: Mapping[SearchTier, RouteStageBudget]

    def __post_init__(self) -> None:
        if not isinstance(self.budgets, Mapping):
            raise TypeError("budgets must be a mapping")
        expected_routes = frozenset({SearchTier.LIGHT, SearchTier.STANDARD})
        if frozenset(self.budgets) != expected_routes:
            raise ValueError("budgets must define exactly light and standard routes")
        normalized: dict[SearchTier, RouteStageBudget] = {}
        for route, budget in self.budgets.items():
            if not isinstance(route, SearchTier):
                raise TypeError("budget routes must be SearchTier values")
            if not isinstance(budget, RouteStageBudget):
                raise TypeError("budget values must be RouteStageBudget values")
            normalized[route] = budget
        object.__setattr__(self, "budgets", MappingProxyType(normalized))

    def for_route(self, route: SearchTier) -> RouteStageBudget:
        if not isinstance(route, SearchTier):
            raise TypeError("route must be a SearchTier")
        if route is SearchTier.SKIP:
            raise ValueError("skip has no stage budget")
        return self.budgets[route]

    def maximum_for_budget(self, budget: RouteStageBudget) -> int | float:
        if not isinstance(budget, RouteStageBudget):
            raise TypeError("budget must be a RouteStageBudget")
        return sum(getattr(budget, field.name) for field in fields(RouteStageBudget))

    def maximum_request_seconds(self, route: SearchTier) -> int | float:
        if not isinstance(route, SearchTier):
            raise TypeError("route must be a SearchTier")
        if route is SearchTier.SKIP:
            return 0
        return self.maximum_for_budget(self.for_route(route))


DEFAULT_SEARCH_BUDGET_POLICY = SearchBudgetPolicy(
    {
        SearchTier.LIGHT: RouteStageBudget(
            analysis_route_seconds=3,
            planner_seconds=0,
            initial_ddgs_seconds=30,
            initial_tavily_seconds=6,
            initial_reader_seconds=4,
            initial_judge_seconds=4,
            gap_seconds=0,
            repair_planner_seconds=0,
            repair_ddgs_seconds=0,
            repair_tavily_seconds=0,
            repair_reader_seconds=0,
            repair_judge_seconds=0,
            answer_seconds=4,
            validator_seconds=4,
            renderer_seconds=1,
            scheduling_margin_seconds=2,
        ),
        SearchTier.STANDARD: RouteStageBudget(
            analysis_route_seconds=3,
            planner_seconds=4,
            initial_ddgs_seconds=30,
            initial_tavily_seconds=8,
            initial_reader_seconds=6,
            initial_judge_seconds=5,
            gap_seconds=1,
            repair_planner_seconds=2,
            repair_ddgs_seconds=30,
            repair_tavily_seconds=5,
            repair_reader_seconds=3,
            repair_judge_seconds=4,
            answer_seconds=4,
            validator_seconds=4,
            renderer_seconds=1,
            scheduling_margin_seconds=2,
        ),
    }
)
