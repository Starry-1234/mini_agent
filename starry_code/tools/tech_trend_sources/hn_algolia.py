"""HackerNews Algolia source — discussion signal.

Real endpoint (added in a later phase):
    GET https://hn.algolia.com/api/v1/search?query={topic}&tags=story
        &numericFilters=points>10&hitsPerPage=30

What we extract (when real):
- hit count for last 30 days
- median points of those hits

Score normalization:
    score = min(10, log10(hits_30d + 1) * 3.5)

Failure mode: 5xx → direction="unknown".

Why lower weight (15%) in the composite: HN is noisier and more
"what's cool today" than "what's hiring tomorrow" — useful as a
tiebreaker, not as a primary signal.
"""
from __future__ import annotations
import math

from .github_trending import SourceResult


class HNAlgoliaProvider:
    """Mock by default. Real HTTP pluggable via subclass."""

    name = "hn_algolia"

    def __init__(self, data: dict[str, int] | None = None) -> None:
        self._data = data if data is not None else self._default_data()

    def _default_data(self) -> dict[str, int]:
        # 30-day HN story counts (mock).
        return {
            "rust":          42,    "go":            65,
            "python":        180,   "java":          35,
            "typescript":    95,    "react":         88,
            "kubernetes":    51,    "swift":         18,
            "kotlin":        22,    "vue":           33,
            "javascript":    72,    "node.js":       48,
            "docker":        38,    "machine-learning": 240,
            "data-science":  67,    "ai-agent":      310,
            "webrtc":        8,     "blockchain":    12,
        }

    def fetch(self, topic: str, region: str = "cn") -> SourceResult:
        key = (topic or "").strip().lower()
        hits = self._data.get(key)
        if hits is None:
            return SourceResult(
                score=None,
                direction="unknown",
                notes=f"HN: no recent stories on {key!r}",
                raw={"topic": key},
            )
        score = min(10.0, math.log10(hits + 1) * 3.5)
        if score >= 7:
            direction = "rising"
        elif score >= 4:
            direction = "stable"
        else:
            direction = "falling"
        return SourceResult(
            score=round(score, 1),
            direction=direction,
            notes=f"HN: {hits} stories in 30d",
            raw={"topic": key, "hits_30d": hits},
        )