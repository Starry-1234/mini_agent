"""Baseline learning-window table.

Pure-curated estimates of "how many weeks from 0 → ship-able, resume-grade
project" for each well-known topic. There is no API for this — it's
domain knowledge we encode as data so the agent can give concrete
guidance.

Why we ship a table (not an LLM call):
- Cost: one lookup beats one LLM roundtrip per topic.
- Stability: the LLM is inconsistent on "weeks to learn X" depending
  on temperature; the table is fixed.
- Locality: this data is small enough to live in the binary.

To extend: add an entry. The min/median/max triplet maps to the
final_score bucket (high → min, mid → median, low → max).
"""
from __future__ import annotations
from typing import NamedTuple


class WindowEntry(NamedTuple):
    min_weeks: int       # when final_score >= 8 (rich ecosystem, easy path)
    median_weeks: int    # when 4 <= final_score < 8 (default)
    max_weeks: int       # when final_score < 4 (sparse resources)


# Curated from aggregate experience; not derived from data sources.
# WindowEntry is a NamedTuple — use positional (min_weeks, median_weeks, max_weeks).
BASELINE: dict[str, WindowEntry] = {
    # System / low-level
    "rust":           WindowEntry(8,  12, 24),
    "c":              WindowEntry(4,  8,  16),
    "cpp":            WindowEntry(6,  10, 20),
    "go":             WindowEntry(4,  8,  14),

    # Backend web
    "python":         WindowEntry(3,  6,  10),
    "java":           WindowEntry(5,  8,  14),
    "kotlin":         WindowEntry(4,  8,  14),
    "node.js":        WindowEntry(3,  6,  10),
    "php":            WindowEntry(3,  6,  10),
    "ruby":           WindowEntry(3,  6,  10),

    # Frontend
    "javascript":     WindowEntry(3,  6,  10),
    "typescript":     WindowEntry(3,  6,  10),
    "react":          WindowEntry(3,  6,  10),
    "vue":            WindowEntry(2,  4,  8),

    # Mobile
    "swift":          WindowEntry(6,  10, 18),
    "android":        WindowEntry(5,  8,  14),

    # Data / ML
    "data-science":   WindowEntry(8,  14, 24),
    "machine-learning": WindowEntry(10, 16, 28),
    "pytorch":        WindowEntry(4,  8,  14),
    "tensorflow":     WindowEntry(4,  8,  14),

    # Infra
    "docker":         WindowEntry(2,  4,  8),
    "kubernetes":     WindowEntry(6,  10, 18),
    "terraform":      WindowEntry(4,  6,  12),
    "aws":            WindowEntry(6,  10, 18),

    # Data engineering
    "spark":          WindowEntry(6,  10, 18),
    "kafka":          WindowEntry(4,  6,  12),
    "elasticsearch":  WindowEntry(4,  6,  12),

    # Trending
    "ai-agent":       WindowEntry(6,  10, 20),
    "webrtc":         WindowEntry(8,  12, 20),
    "blockchain":     WindowEntry(8,  14, 28),
}
_DEFAULT = WindowEntry(4, 12, 20)


class BaselineTable:
    """Lookup curated learning-window estimates by name.

    All lookups are case-insensitive and tolerate aliases ("nodejs" → "node.js").
    """

    _ALIASES = {
        "nodejs": "node.js",
        "node":   "node.js",
        "reactjs": "react",
        "vuejs":  "vue",
        "ml":     "machine-learning",
        "ai":     "ai-agent",
        "k8s":    "kubernetes",
    }

    @classmethod
    def resolve(cls, topic: str) -> str:
        """Map topic → canonical baseline key."""
        k = (topic or "").strip().lower()
        return cls._ALIASES.get(k, k)

    @classmethod
    def get(cls, topic: str) -> WindowEntry:
        """Return curated WindowEntry for `topic`, or _DEFAULT if unknown."""
        canonical = cls.resolve(topic)
        return BASELINE.get(canonical, _DEFAULT)

    @classmethod
    def weeks_for(cls, topic: str, final_score: float) -> int:
        """Pick min/median/max based on final_score bucket.

        - final_score >= 8 → min (rich ecosystem, easy path)
        - 4 <= final_score < 8 → median
        - final_score < 4 → max (sparse resources, longer haul)
        """
        entry = cls.get(topic)
        if final_score >= 8:
            return entry.min_weeks
        if final_score >= 4:
            return entry.median_weeks
        return entry.max_weeks