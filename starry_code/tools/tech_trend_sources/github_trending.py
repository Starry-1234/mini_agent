"""GitHub Trending source — adoption velocity proxy.

Real endpoint (added in a later phase):
    GET https://api.github.com/search/repositories
        ?q={topic}+language:{lang}&sort=stars&order=desc&per_page=20

What we extract (when real):
- count of repos pushed in last 30 days with topic in name/description
- median star count of those repos
- 30-day delta of stars added

Score normalization:
    score = min(10, log10(30d_delta_stars + 1) * 4)
    direction:
        rising  if score >= 7
        stable  if 4 <= score < 7
        falling if score < 4

Failure mode: (rate-limited / 5xx) → return `direction="unknown"`,
score=None; aggregator treats as "no signal" (still produces final_score
from other sources).
"""
from __future__ import annotations
import math
from typing import Protocol


class _SourceLike(Protocol):
    name: str
    def fetch(self, topic: str, region: str = "cn") -> "SourceResult": ...


class SourceResult:
    """Per-source normalized output. Aggregator combines several of these."""

    def __init__(self, *, score: float | None, direction: str,
                 notes: str, raw: dict | None = None,
                 error: str | None = None) -> None:
        self.score = score            # 0-10, or None if no signal
        self.direction = direction     # 'rising' | 'stable' | 'falling' | 'unknown'
        self.notes = notes             # one-sentence human-readable
        self.raw = raw or {}
        self.error = error             # str if fetch failed

    def __repr__(self) -> str:
        return (f"SourceResult(score={self.score}, direction={self.direction!r}, "
                f"notes={self.notes!r})")


def _normalize_score(raw_value: float | None) -> tuple[float | None, str]:
    """Map a raw_value (whatever the source gives us) to (0-10, direction)."""
    if raw_value is None:
        return None, "unknown"
    # log10(x+1)*4: 1 → 1.2, 10 → 4.6, 100 → 6.9, 1000 → 9.2
    score = min(10.0, math.log10(max(0, raw_value) + 1) * 4.0)
    if score >= 7:
        return round(score, 1), "rising"
    if score >= 4:
        return round(score, 1), "stable"
    return round(score, 1), "falling"


class GitHubTrendingProvider:
    """Mock by default. Real HTTP pluggable via subclass."""

    name = "github"

    # Topic → 30d delta stars (deterministic mock).
    #MOCK = {
    #    "rust":          380,    "go":            620,
    #    "python":        950,    "java":          310,
    #    "typescript":    540,    "react":         410,
    #    "kubernetes":    220,    "swift":         60,
    #    "kotlin":        140,    "vue":           180,
    #    "ai-agent":      510,    "webrtc":        30,
    #}

    def __init__(self, data: dict[str, float] | None = None) -> None:
        # Allow injection for tests / real source.
        self._data = data if data is not None else self._default_data()

    def _default_data(self) -> dict[str, float]:
        # Curated mock — picked to roughly match real-world GitHub trends
        # as of 2026 (rising stars over 30 days).
        return {
            "rust":            380,
            "go":              620,
            "python":          950,
            "java":            310,
            "typescript":      540,
            "react":           410,
            "kubernetes":      220,
            "swift":            60,
            "kotlin":          140,
            "vue":             180,
            "javascript":      420,
            "node.js":         380,
            "docker":          160,
            "machine-learning": 720,
            "data-science":    290,
            "ai-agent":        510,
            "webrtc":           30,
            "blockchain":       40,
        }

    def fetch(self, topic: str, region: str = "cn") -> SourceResult:
        key = (topic or "").strip().lower()
        delta = self._data.get(key)
        if delta is None:
            return SourceResult(
                score=None,
                direction="unknown",
                notes=f"GitHub: no record of {key!r}",
                raw={"topic": key},
            )
        norm, direction = _normalize_score(delta)
        return SourceResult(
            score=norm,
            direction=direction,
            notes=f"GitHub: {key} +{int(delta)} stars in 30d",
            raw={"topic": key, "delta_30d": delta},
        )