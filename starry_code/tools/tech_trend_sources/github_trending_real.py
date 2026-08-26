"""Real GitHub Trending provider.

Endpoint: GET https://api.github.com/search/repositories
           ?q={topic}+pushed:>YYYY-MM-DD&sort=stars&order=desc&per_page=20

Auth (optional): set GITHUB_TOKEN env var for 5000 req/h vs 60 req/h
unauthenticated. We pick the topic from {topic}, push window 30 days,
count stars added.

Score: log10(30d_delta_stars + 1) * 4, capped 0-10.
Direction: rising/stable/falling per threshold.

Error / rate-limit handling:
- 403 (rate limit) / 5xx → raise RuntimeError → caller catches + falls back
- ConnectionError / Timeout → same
- No results (topic truly absent) → direction="unknown", score=None
"""
from __future__ import annotations
import math
import os
from datetime import datetime, timezone, timedelta

from .github_trending import SourceResult, _normalize_score


_GITHUB_API = "https://api.github.com/search/repositories"


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json",
         "User-Agent": "starry-code/1.0"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _query(topic: str, days: int = 30) -> str:
    """Build GitHub search query: topic in name/description, pushed recently.

    GitHub search doesn't have a direct "stars added in N days" operator.
    We approximate by filtering to repos pushed within the window and
    sorting by star count (proxy for velocity).
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    return f"{topic} pushed:>{since}"


class RealGitHubTrendingProvider:
    """Real GitHub API-backed trend provider.

    Same `fetch(topic, region)` signature as the mock. Network failures
    propagate as RuntimeError; callers (CompositeTrendProvider) catch
    and fall back to mock per R3 circuit-breaker pattern.
    """

    name = "github"
    network = True  # marker: this provider hits the network

    def __init__(self, timeout: float = 5.0, days: int = 30,
                 page_size: int = 20) -> None:
        self.timeout = timeout
        self.days = days
        self.page_size = page_size

    def fetch(self, topic: str, region: str = "cn") -> SourceResult:
        import httpx  # lazy: optional dep (only needed for real providers)

        key = (topic or "").strip().lower()
        if not key:
            return SourceResult(
                score=None, direction="unknown",
                notes="empty topic", raw={"topic": key},
            )
        q = _query(key, self.days)
        try:
            resp = httpx.get(
                _GITHUB_API,
                params={"q": q, "sort": "stars", "order": "desc",
                        "per_page": self.page_size},
                headers=_headers(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise RuntimeError(f"github network error: {type(e).__name__}: {e}")

        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            raise RuntimeError("github rate limit hit")
        if resp.status_code >= 500:
            raise RuntimeError(f"github server error: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items") or []
        if not items:
            return SourceResult(
                score=None, direction="unknown",
                notes=f"GitHub: no repos for {key!r} in last {self.days}d",
                raw={"topic": key, "count": 0, "total": data.get("total_count", 0)},
            )

        # Aggregate stars across the top-N repos as a coarse velocity signal.
        # Real "stars added in N days" would need a more sophisticated query
        # (e.g., stargazers with timestamps), but top-N by total stars is
        # a good enough proxy for the MVP. R3+ can replace with proper
        # time-bucketed star count.
        total_stars = sum(it.get("stargazers_count", 0) for it in items)
        score, direction = _normalize_score(float(total_stars))
        return SourceResult(
            score=score,
            direction=direction,
            notes=(f"GitHub: {len(items)} repos for {key}, "
                   f"top {self.page_size} stars sum = {total_stars}"),
            raw={
                "topic": key,
                "count": len(items),
                "total_count": data.get("total_count", 0),
                "top_repos": [
                    {"name": it["full_name"], "stars": it.get("stargazers_count", 0)}
                    for it in items[:5]
                ],
            },
        )