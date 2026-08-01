"""Provider-neutral search adapters with an availability fallback."""

from .base import ProviderRegistry, SearchProvider
from .ddgs import DDGSSearchProvider
from .tavily import TavilySearchProvider

__all__ = [
    "DDGSSearchProvider",
    "ProviderRegistry",
    "SearchProvider",
    "TavilySearchProvider",
]
