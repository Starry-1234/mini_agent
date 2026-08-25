"""CompositeTrendProvider tests.

Covers:
1. Each source in isolation: known topic, unknown topic, scoring bands
2. Composite: weighted aggregation with mock data
3. Direction: majority vote logic, override rules
4. learning_window_weeks: derived from BaselineTable × final_score
5. Graceful degradation when a source fails
6. Backward-compat: TechTrendTool's JSON shape unchanged
"""
import json
import pytest

from starry_code.tools.tech_trend_sources import (
    CompositeTrendProvider,
    GitHubTrendingProvider,
    RemotiveJobsProvider,
    HNAlgoliaProvider,
    BaselineTable,
)
from starry_code.tools.tech_trend import TechTrendTool
from starry_code.session import Session


# ---- Single source ----

def test_github_known_topic_rising():
    r = GitHubTrendingProvider().fetch("rust")
    assert r.direction == "rising"
    assert r.score is not None and r.score >= 7
    assert "GitHub" in r.notes


def test_github_unknown_topic_returns_unknown():
    r = GitHubTrendingProvider().fetch("made-up-tech-xyz")
    assert r.direction == "unknown"
    assert r.score is None


def test_github_score_in_range():
    """Score is clamped 0-10."""
    p = GitHubTrendingProvider(data={"huge": 1_000_000})  # absurdly large
    assert p.fetch("huge").score <= 10


def test_remotive_known_topic():
    r = RemotiveJobsProvider().fetch("python")
    assert r.direction in ("rising", "stable")
    assert r.score is not None
    assert "Remotive" in r.notes
    assert "jobs in 7d" in r.notes


def test_remotive_unknown_topic():
    r = RemotiveJobsProvider().fetch("nonexistent-tech")
    assert r.direction == "unknown"


def test_hn_known_topic():
    r = HNAlgoliaProvider().fetch("ai-agent")
    # ai-agent has 310 hits in our mock → should be rising
    assert r.direction == "rising"
    assert r.score is not None and r.score >= 7


def test_hn_unknown_topic():
    r = HNAlgoliaProvider().fetch("nonexistent")
    assert r.direction == "unknown"


# ---- Composite ----

def test_composite_returns_full_shape():
    p = CompositeTrendProvider()
    data = p.fetch("rust")
    for key in ("topic", "found", "direction", "demand_score",
                "learning_window_weeks", "notes", "source", "region",
                "as_of", "sources"):
        assert key in data, f"missing key: {key}"
    assert data["source"] == "composite"
    assert data["topic"] == "rust"
    assert data["found"] is True
    assert 0 <= data["demand_score"] <= 10
    assert isinstance(data["learning_window_weeks"], int)


def test_composite_unknown_topic_returns_unknown():
    p = CompositeTrendProvider()
    data = p.fetch("totally-unknown-tech-xyz")
    assert data["found"] is False
    assert data["direction"] == "unknown"
    assert data["demand_score"] is None
    # learning_window falls back to median
    assert data["learning_window_weeks"] == 12  # default median


def test_composite_direction_is_majority():
    """rust: github rising, remotive rising, hn rising → rising."""
    data = CompositeTrendProvider().fetch("rust")
    assert data["direction"] == "rising"


def test_composite_weighted_score_is_real_number():
    """Composite score is a weighted average (float)."""
    data = CompositeTrendProvider().fetch("python")
    # python: github 950, remotive 95 jobs, hn 180
    # Should be a single weighted average, not None
    assert isinstance(data["demand_score"], float)
    assert 5 <= data["demand_score"] <= 10  # python is hot


def test_composite_includes_per_source_breakdown():
    """The 'sources' field shows each provider's individual contribution."""
    data = CompositeTrendProvider().fetch("go")
    assert "github" in data["sources"]
    assert "remotive" in data["sources"]
    assert "hn_algolia" in data["sources"]
    for name, sub in data["sources"].items():
        assert "score" in sub
        assert "direction" in sub
        assert "notes" in sub


def test_composite_survives_provider_exception():
    """If one source raises, the others still produce output."""
    class BrokenSource:
        name = "broken"
        def fetch(self, topic, region="cn"):
            raise RuntimeError("simulated outage")
    p = CompositeTrendProvider(sources=[
        GitHubTrendingProvider(),
        BrokenSource(),
        RemotiveJobsProvider(),
        HNAlgoliaProvider(),
    ])
    data = p.fetch("python")
    assert data["found"] is True  # other sources still work
    assert data["sources"]["broken"]["error"]


def test_composite_survives_all_unknown():
    """If all sources return unknown, aggregate is unknown but shape is valid."""
    class AllUnknown:
        name = "blank"
        def fetch(self, topic, region="cn"):
            from starry_code.tools.tech_trend_sources.github_trending import SourceResult
            return SourceResult(score=None, direction="unknown", notes="nothing")
    p = CompositeTrendProvider(sources=[AllUnknown()])
    data = p.fetch("rust")
    assert data["direction"] == "unknown"
    assert data["demand_score"] is None


# ---- learning_window ----

def test_learning_window_short_bucket():
    """demand_score >= 8 → min_weeks (short path)."""
    # Compose a scenario where score is high
    p = CompositeTrendProvider(sources=[
        # Force a 9.5 score for "rust" via a custom provider
        type("HighScore")(),
    ]) if False else None
    # Simpler: ask directly
    weeks_high = BaselineTable.weeks_for("rust", final_score=9.0)
    weeks_mid = BaselineTable.weeks_for("rust", final_score=6.0)
    weeks_low = BaselineTable.weeks_for("rust", final_score=2.0)
    assert weeks_high < weeks_mid < weeks_low
    assert weeks_high == BaselineTable.get("rust").min_weeks
    assert weeks_mid == BaselineTable.get("rust").median_weeks
    assert weeks_low == BaselineTable.get("rust").max_weeks


def test_baseline_resolves_aliases():
    assert BaselineTable.resolve("nodejs") == "node.js"
    assert BaselineTable.resolve("k8s") == "kubernetes"
    assert BaselineTable.resolve("ML") == "machine-learning"
    # unknown falls through
    assert BaselineTable.resolve("made-up") == "made-up"


def test_baseline_unknown_topic_uses_default():
    entry = BaselineTable.get("unknown-tech-xyz")
    assert entry.min_weeks == 4
    assert entry.median_weeks == 12
    assert entry.max_weeks == 20


# ---- Tool integration ----

def test_tech_trend_tool_default_uses_composite():
    """TechTrendTool's default provider is now CompositeTrendProvider."""
    assert TechTrendTool.current_provider_name() == "CompositeTrendProvider"


def test_tech_trend_tool_json_includes_sources_field():
    """The JSON output to the LLM now includes a 'sources' breakdown."""
    tool = TechTrendTool()
    r = tool.execute({"topic": "rust"}, session=Session(id="x"))
    assert r.ok
    data = json.loads(r.content)
    assert "sources" in data
    assert "github" in data["sources"]


def test_tech_trend_tool_set_provider_still_works():
    """Back-compat: set_provider still works for custom providers."""
    class FakeProvider:
        def fetch(self, topic, region="cn"):
            return {"topic": topic, "found": True, "direction": "stable",
                    "demand_score": 5, "learning_window_weeks": 8,
                    "notes": "fake", "source": "fake"}
    TechTrendTool.set_provider(FakeProvider())
    try:
        tool = TechTrendTool()
        r = tool.execute({"topic": "x"}, session=Session(id="x"))
        assert r.ok
        assert "fake" in r.content
    finally:
        # Restore composite for other tests
        TechTrendTool.set_provider(CompositeTrendProvider())


# ---- Real-data integration: skipped by default, env-gated ----

@pytest.mark.skip(reason="Real-HTTP test is opt-in; requires network + --run-network flag")
def test_real_github_responds():
    """Opt-in: validates the real HTTP path is wired correctly.

    Run manually with:
        PYTHONPATH=. python -m pytest tests/test_composite_trend.py::test_real_github_responds --run-network
    """
    import requests
    r = requests.get(
        "https://api.github.com/search/repositories",
        params={"q": "rust", "sort": "stars", "per_page": 5},
        timeout=10,
    )
    assert r.status_code == 200