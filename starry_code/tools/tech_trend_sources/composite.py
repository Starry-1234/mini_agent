"""Composite aggregator — combines GitHub + Remotive + HN into one score.

Per external-agent feedback Q2: weights 40/45/15 (GitHub/Remotive/HN) are
correct. Hiring demand (45%) is the highest single weight but not
dominant, so a topic with no jobs but heavy GitHub activity (e.g. an
emerging niche) still surfaces.

R1: real data sources (`Real*Provider`) added. Each one may fail
(network / rate-limit / 5xx). The aggregator:
  1. Calls each provider. On RuntimeError, marks the source as failed.
  2. Computes direction + score from the surviving sources.
  3. If a `MockXxxProvider` is also configured as fallback (set via
     `*_FALLBACK=mock` env), substitutes the failed real one with
     its mock counterpart. This is R1's "graceful degradation": real
     when it works, mock when it doesn't, with explicit `data_source`
     tag in the output so the user can see which path served them.

Failure handling:
- A source returning `direction="unknown"` (no data) is excluded from
  the vote. A source *raising* is excluded from the score average and
  (optionally) replaced by its mock fallback.
- If only 1 source returned a real direction, trust it.
- If all sources failed, return `direction="unknown"`, score=None.
"""
from __future__ import annotations
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

from .baseline import BaselineTable
from .github_trending import GitHubTrendingProvider, SourceResult
from .remotive import RemotiveJobsProvider
from .hn_algolia import HNAlgoliaProvider


# Weights — frozen per design doc. Changing these is a deliberate
# calibration decision, not a tweak.
WEIGHTS = {
    "github":    0.40,
    "remotive":  0.45,
    "hn_algolia": 0.15,
}


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean env var. Accepts '1', 'true', 'yes' (case-insens)."""
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes") if v else default


def _default_sources(real: bool | None = None) -> list:
    """Build the default source list.

    `real=None` means auto-detect from env TREND_PROVIDER:
      - "real" → real providers (HTTP)
      - "mock" or unset → mock providers (deterministic)
    """
    if real is None:
        real = os.environ.get("TREND_PROVIDER", "").strip().lower() == "real"

    if real:
        # Lazy import — real providers need `httpx` (optional dep).
        from .github_trending_real import RealGitHubTrendingProvider
        from .remotive_real import RealRemotiveJobsProvider
        from .hn_algolia_real import RealHNAlgoliaProvider
        return [
            RealGitHubTrendingProvider(),
            RealRemotiveJobsProvider(),
            RealHNAlgoliaProvider(),
        ]
    return [
        GitHubTrendingProvider(),
        RemotiveJobsProvider(),
        HNAlgoliaProvider(),
    ]


class CompositeTrendProvider:
    """Aggregates multiple sources into one TechTrendTool response.

    Drop-in replacement for `_MockTrendProvider`. Same `fetch(topic, region)`
    signature returning a dict that matches the TechTrendTool contract.
    """

    def __init__(self, sources: Iterable | None = None,
                 *, allow_mock_fallback: bool = True) -> None:
        if sources is None:
            sources = _default_sources()
        # Map by name for weighted lookup. Pair real source with its mock
        # so a real failure can fall back to the deterministic mock data
        # (only if TREND_FALLBACK=mock).
        self._sources: dict[str, object] = {s.name: s for s in sources}
        self._fallback_pool: dict[str, object] = {}
        if allow_mock_fallback and _env_flag("TREND_FALLBACK", True):
            for src in _default_sources(real=False):
                self._fallback_pool[src.name] = src

    def fetch(self, topic: str, region: str = "cn") -> dict:
        """Aggregate all sources. Returns the same dict shape as the old
        _MockTrendProvider.fetch so TechTrendTool needs no changes.

        Output keys:
          topic, found, direction, demand_score (0-10), learning_window_weeks,
          sources, region, as_of, data_source ("composite" or "composite+fallback")
        """
        results: dict[str, SourceResult] = {}
        any_fallback_used = False
        for name, src in self._sources.items():
            try:
                results[name] = src.fetch(topic, region=region)
            except Exception as e:  # noqa: BLE001 — provider errors must not crash
                # Real source failed. Substitute mock if available.
                fallback = self._fallback_pool.get(name)
                if fallback is not None:
                    try:
                        results[name] = fallback.fetch(topic, region=region)
                        # Mark so the user knows this part is mock
                        results[name] = SourceResult(
                            score=results[name].score,
                            direction=results[name].direction,
                            notes=results[name].notes + " [FALLBACK]",
                            raw=results[name].raw,
                        )
                        any_fallback_used = True
                    except Exception:
                        results[name] = SourceResult(
                            score=None, direction="unknown",
                            notes=f"{name}: provider raised {type(e).__name__}",
                            error=str(e),
                        )
                else:
                    results[name] = SourceResult(
                        score=None, direction="unknown",
                        notes=f"{name}: provider raised {type(e).__name__}",
                        error=str(e),
                    )

        # ---- direction: majority vote among real (non-unknown) signals ----
        real_directions = [r.direction for r in results.values()
                           if r.direction in ("rising", "stable", "falling")]
        if not real_directions:
            direction = "unknown"
        else:
            most_common = Counter(real_directions).most_common(1)[0][0]
            direction = most_common

        # ---- final_score: weighted average of available scores ----
        weighted_sum = 0.0
        weight_total = 0.0
        for name, r in results.items():
            if r.score is None:
                continue
            w = WEIGHTS.get(name, 0.0)
            weighted_sum += r.score * w
            weight_total += w
        if weight_total == 0:
            demand_score = None
        else:
            demand_score = round(weighted_sum / weight_total, 1)

        # ---- learning_window_weeks: from BaselineTable × final_score ----
        if demand_score is None:
            weeks = BaselineTable.weeks_for(topic, final_score=5.0)
        else:
            weeks = BaselineTable.weeks_for(topic, final_score=demand_score)

        found = any(r.score is not None for r in results.values())
        notes = " | ".join(r.notes for r in results.values())

        # Build per-source output for debugging / show.
        sources_out = {
            name: {
                "score": r.score,
                "direction": r.direction,
                "notes": r.notes,
                "error": r.error,
            }
            for name, r in results.items()
        }

        return {
            "topic": topic,
            "found": found,
            "direction": direction,
            "demand_score": demand_score,
            "learning_window_weeks": weeks,
            "notes": notes,
            "source": "composite+fallback" if any_fallback_used else "composite",
            "region": region,
            "as_of": datetime.now(timezone.utc).date().isoformat(),
            "sources": sources_out,
        }