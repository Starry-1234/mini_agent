"""Real provider tests — exercises Real*Provider via mocked HTTP.

We use httpx.MockTransport to inject canned responses, so the tests
verify the provider's:
- URL / params construction
- response parsing
- score normalization
- error handling (rate limit, 5xx, connection error)

No real network calls. Real API behavior covered separately by the
opt-in `test_real_github_responds` style test.
"""
import json
import pytest
import httpx

from starry_code.tools.tech_trend_sources.github_trending_real import (
    RealGitHubTrendingProvider, _query as gh_query,
)
from starry_code.tools.tech_trend_sources.remotive_real import (
    RealRemotiveJobsProvider,
)
from starry_code.tools.tech_trend_sources.hn_algolia_real import (
    RealHNAlgoliaProvider,
)
from starry_code.tools.tech_trend_sources.composite import (
    CompositeTrendProvider, _env_flag, _default_sources,
)
from starry_code.tools.tech_trend_sources.github_trending import SourceResult


# ---- GitHub ----

def test_github_query_includes_pushed_window():
    q = gh_query("rust", days=30)
    assert "rust" in q
    assert "pushed:>" in q
    assert len(q) > 20  # has a date too


def _patched_httpx_get(monkeypatch, response_json: dict, status: int = 200,
                        text: str = ""):
    """Helper: replace httpx.get globally with a stub that returns a
    MockTransport response. The provider does `import httpx` inside the
    function so this global patch is what it sees.

    Uses `content=` (raw bytes) rather than `json=` because some httpx
    versions serialize differently and the test was getting empty body.
    """
    body = json.dumps(response_json).encode("utf-8") if response_json \
        else text.encode("utf-8")
    fake = httpx.MockTransport(
        lambda req: httpx.Response(status, content=body,
                                   headers={"content-type": "application/json"}))
    client = httpx.Client(transport=fake)
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: client.get(*a, **kw))
    return client


def test_github_fetch_with_mock_data(monkeypatch):
    _patched_httpx_get(monkeypatch, {
        "total_count": 1234,
        "items": [
            {"full_name": "tokio/rs", "stargazers_count": 25000},
            {"full_name": "rust-lang/rust", "stargazers_count": 80000},
            {"full_name": "actix/actix-web", "stargazers_count": 19000},
        ],
    })
    r = RealGitHubTrendingProvider().fetch("rust")
    assert r.direction in ("rising", "stable")
    assert r.score is not None and 0 < r.score <= 10
    assert r.raw["count"] == 3
    assert r.raw["topic"] == "rust"


def test_github_fetch_no_results_unknown(monkeypatch):
    _patched_httpx_get(monkeypatch, {"total_count": 0, "items": []})
    r = RealGitHubTrendingProvider().fetch("nonexistent-xyz")
    assert r.direction == "unknown"
    assert r.score is None


def test_github_fetch_5xx_raises(monkeypatch):
    """A real-server error must propagate so the composite can fall back."""
    _patched_httpx_get(monkeypatch, response_json={}, status=503)
    with pytest.raises(RuntimeError, match="server error"):
        RealGitHubTrendingProvider().fetch("rust")


def test_github_fetch_rate_limit_raises(monkeypatch):
    _patched_httpx_get(monkeypatch, response_json={}, status=403, text="rate limit")
    with pytest.raises(RuntimeError, match="rate limit"):
        RealGitHubTrendingProvider().fetch("rust")


# ---- Remotive ----

def test_remotive_fetch_with_mock_data(monkeypatch):
    # 50 jobs in 7d → log10(51)*3.5 = 5.94 → "stable" / "rising" range
    _patched_httpx_get(monkeypatch, {
        "job-count": 50,
        "jobs": [
            {"title": f"Go Developer {i}",
             "publication_date": "2026-08-22T10:00:00+00:00"}
            for i in range(50)
        ],
    })
    r = RealRemotiveJobsProvider().fetch("go")
    assert r.direction in ("rising", "stable", "falling")
    assert r.score is not None
    assert r.raw["jobs_in_window"] >= 1


def test_remotive_fetch_no_jobs_unknown(monkeypatch):
    _patched_httpx_get(monkeypatch, {"job-count": 0, "jobs": []})
    r = RealRemotiveJobsProvider().fetch("nonexistent")
    assert r.direction == "unknown"


# ---- HN Algolia ----

def test_hn_fetch_with_mock_data(monkeypatch):
    import time
    now = int(time.time())
    # 100 stories on ai-agent → log10(101)*3.5 = 7.04 → "rising"
    _patched_httpx_get(monkeypatch, {
        "nbHits": 100,
        "hits": [
            {"title": f"Story {i}", "points": 50,
             "created_at_i": now - 86400 * (i % 28)}  # spread over 28d
            for i in range(100)
        ],
    })
    r = RealHNAlgoliaProvider().fetch("ai-agent")
    assert r.direction in ("rising", "stable", "falling")
    assert r.raw["hits_in_window"] >= 50  # most are within 30d


# ---- Composite TREND_PROVIDER switching ----

def test_env_flag_truthy_values(monkeypatch):
    for v in ("1", "true", "yes", "TRUE", "Yes"):
        monkeypatch.setenv("TEST_FLAG", v)
        assert _env_flag("TEST_FLAG") is True
    for v in ("0", "false", "no", "", "abc"):
        monkeypatch.setenv("TEST_FLAG", v)
        assert _env_flag("TEST_FLAG") is False
    monkeypatch.delenv("TEST_FLAG", raising=False)
    assert _env_flag("TEST_FLAG") is False  # default


def test_default_sources_real_when_env_set(monkeypatch):
    monkeypatch.setenv("TREND_PROVIDER", "real")
    srcs = _default_sources()
    names = {s.name for s in srcs}
    assert names == {"github", "remotive", "hn_algolia"}
    assert all(getattr(s, "network", False) for s in srcs), \
        "with TREND_PROVIDER=real, all sources should be the real (network) ones"


def test_default_sources_mock_when_env_unset(monkeypatch):
    monkeypatch.delenv("TREND_PROVIDER", raising=False)
    srcs = _default_sources()
    assert all(not getattr(s, "network", False) for s in srcs), \
        "without TREND_PROVIDER=real, sources should be the mock ones"


def test_composite_falls_back_to_mock_on_real_failure(monkeypatch):
    """When the real github source fails (e.g. 503), composite should
    substitute the mock fallback for that source. The output's
    `data_source` field should be 'composite+fallback' (vs just
    'composite' for clean runs)."""
    from starry_code.tools.tech_trend_sources.github_trending_real import (
        RealGitHubTrendingProvider,
    )

    monkeypatch.setenv("TREND_PROVIDER", "real")
    monkeypatch.setenv("TREND_FALLBACK", "true")

    real_gh = RealGitHubTrendingProvider()
    real_gh.fetch = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("simulated 503"))

    from starry_code.tools.tech_trend_sources.remotive import RemotiveJobsProvider
    from starry_code.tools.tech_trend_sources.hn_algolia import HNAlgoliaProvider

    comp = CompositeTrendProvider(
        sources=[real_gh, RemotiveJobsProvider(), HNAlgoliaProvider()],
    )
    data = comp.fetch("rust")
    assert data["source"] == "composite+fallback"
    assert "[FALLBACK]" in data["sources"]["github"]["notes"]
    # Other sources still get real (or in this case mock) data normally
    assert data["sources"]["remotive"]["notes"]  # has content


def test_composite_no_fallback_when_env_off(monkeypatch):
    monkeypatch.setenv("TREND_PROVIDER", "real")
    monkeypatch.setenv("TREND_FALLBACK", "false")

    from starry_code.tools.tech_trend_sources.github_trending_real import (
        RealGitHubTrendingProvider,
    )
    real_gh = RealGitHubTrendingProvider()
    real_gh.fetch = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("simulated 503"))

    from starry_code.tools.tech_trend_sources.remotive import RemotiveJobsProvider
    from starry_code.tools.tech_trend_sources.hn_algolia import HNAlgoliaProvider

    comp = CompositeTrendProvider(
        sources=[real_gh, RemotiveJobsProvider(), HNAlgoliaProvider()],
    )
    data = comp.fetch("rust")
    # GitHub failed and no fallback, so its score is None
    assert data["sources"]["github"]["error"] is not None
    # But the composite still has a score from the other 2 sources
    assert data["demand_score"] is not None
    assert data["source"] == "composite"  # no fallback was used