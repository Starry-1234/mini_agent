"""Tech trend data sources for the CompositeTrendProvider.

Each module in this package exposes:
- A `Source` Protocol-conforming class with `fetch(topic, region="cn") -> SourceResult`
- A `_MockXxxProvider` with deterministic data for offline/test use
- Optionally a `RealXxxProvider` that hits a public HTTP API (added in a
  later phase; this MVP only ships the mock variants)

Adding a new source is a 3-step exercise:
1. Implement the class with `fetch()` returning a `SourceResult`
2. Add it to `CompositeTrendProvider._default_providers()` with a weight
3. (Optional) Extend `BaselineTable` if the source informs the
   `learning_window_weeks` estimate

Why a package: keeps each source's HTTP quirks / failure / scoring
together, easy to add or remove one without touching the aggregator.
"""
from .baseline import BaselineTable
from .composite import CompositeTrendProvider, SourceResult
from .github_trending import GitHubTrendingProvider
from .remotive import RemotiveJobsProvider
from .hn_algolia import HNAlgoliaProvider
from .github_trending_real import RealGitHubTrendingProvider
from .remotive_real import RealRemotiveJobsProvider
from .hn_algolia_real import RealHNAlgoliaProvider

__all__ = [
    "BaselineTable",
    "CompositeTrendProvider",
    "SourceResult",
    "GitHubTrendingProvider",
    "RemotiveJobsProvider",
    "HNAlgoliaProvider",
    "RealGitHubTrendingProvider",
    "RealRemotiveJobsProvider",
    "RealHNAlgoliaProvider",
]