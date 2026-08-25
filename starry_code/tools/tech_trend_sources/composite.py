"""Composite aggregator — combines GitHub + Remotive + HN into one score.

Per external-agent feedback Q2: weights 40/45/15 (GitHub/Remotive/HN) are
correct. Hiring demand (45%) is the highest single weight but not
dominant, so a topic with no jobs but heavy GitHub activity (e.g. an
emerging niche) still surfaces.

Direction logic: majority vote of the three sources. If 2/3 say rising
→ rising. If jobs_count has been 0 for 60+ days → override to "falling"
regardless (no one's hiring this — even if it's still "interesting" on
GitHub, it's not paying your rent).

Failure handling:
- A source returning `direction="unknown"` is excluded from the vote.
- If only 1 source returned a real direction, trust it (no majority
  needed).
- If all sources failed, return `direction="unknown"` and score=None.
"""
from __future__ import annotations
from collections import Counter
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


class CompositeTrendProvider:
    """Aggregates multiple sources into one TechTrendTool response.

    Drop-in replacement for `_MockTrendProvider`. Same `fetch(topic, region)`
    signature returning a dict that matches the TechTrendTool contract.
    """

    def __init__(self, sources: Iterable | None = None) -> None:
        if sources is None:
            sources = self._default_sources()
        # Map by name for weighted lookup.
        self._sources: dict[str, object] = {s.name: s for s in sources}

    @staticmethod
    def _default_sources():
        return [
            GitHubTrendingProvider(),
            RemotiveJobsProvider(),
            HNAlgoliaProvider(),
        ]

    def fetch(self, topic: str, region: str = "cn") -> dict:
        """Aggregate all sources. Returns the same dict shape as the old
        _MockTrendProvider.fetch so TechTrendTool needs no changes.

        Output keys:
          topic, found, direction, demand_score (0-10), learning_window_weeks,
          sources, region, as_of
        """
        results: dict[str, SourceResult] = {}
        for name, src in self._sources.items():
            try:
                results[name] = src.fetch(topic, region=region)
            except Exception as e:  # noqa: BLE001 — provider errors must not crash
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
            # Fall back to median when we have no aggregate signal.
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

        from datetime import datetime, timezone
        return {
            "topic": topic,
            "found": found,
            "direction": direction,
            "demand_score": demand_score,
            "learning_window_weeks": weeks,
            "notes": notes,
            "source": "composite",
            "region": region,
            "as_of": datetime.now(timezone.utc).date().isoformat(),
            "sources": sources_out,
        }