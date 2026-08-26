"""Real HackerNews Algolia search provider.

Endpoint: GET https://hn.algolia.com/api/v1/search
            ?query={topic}&tags=story&numericFilters=points>10
            &hitsPerPage=30&page=0

We count hits in a 30-day window using `created_at_i` (epoch seconds).
"""
from __future__ import annotations
import math
import time
from datetime import datetime, timezone, timedelta

from .github_trending import SourceResult


_HN_API = "https://hn.algolia.com/api/v1/search"


def _score(hits_30d: int) -> tuple[float, str]:
    score = min(10.0, math.log10(hits_30d + 1) * 3.5)
    if score >= 7:
        return round(score, 1), "rising"
    if score >= 4:
        return round(score, 1), "stable"
    return round(score, 1), "falling"


class RealHNAlgoliaProvider:
    name = "hn_algolia"
    network = True

    def __init__(self, timeout: float = 5.0, days: int = 30,
                 min_points: int = 10, hits_per_page: int = 30) -> None:
        self.timeout = timeout
        self.days = days
        self.min_points = min_points
        self.hits_per_page = hits_per_page

    def fetch(self, topic: str, region: str = "cn") -> SourceResult:
        import httpx

        key = (topic or "").strip().lower()
        if not key:
            return SourceResult(
                score=None, direction="unknown",
                notes="empty topic", raw={"topic": key},
            )
        try:
            resp = httpx.get(
                _HN_API,
                params={
                    "query": key,
                    "tags": "story",
                    "numericFilters": f"points>{self.min_points}",
                    "hitsPerPage": self.hits_per_page,
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise RuntimeError(f"hn network error: {type(e).__name__}: {e}")

        if resp.status_code >= 500:
            raise RuntimeError(f"hn server error: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()

        hits = data.get("hits") or []
        if not hits:
            return SourceResult(
                score=None, direction="unknown",
                notes=f"HN: no stories for {key!r}",
                raw={"topic": key, "total": data.get("nbHits", 0)},
            )

        # Filter to last N days (Algolia returns `created_at_i` epoch sec)
        cutoff = time.time() - self.days * 86400
        recent = [h for h in hits
                  if isinstance(h.get("created_at_i"), (int, float))
                  and h["created_at_i"] >= cutoff]
        hits_30d = len(recent)
        score, direction = _score(hits_30d)
        return SourceResult(
            score=score,
            direction=direction,
            notes=f"HN: {hits_30d} stories on {key} (>{self.min_points} pts) in {self.days}d",
            raw={
                "topic": key,
                "total_hits": data.get("nbHits", 0),
                "hits_in_window": hits_30d,
                "top_titles": [h.get("title", "") for h in hits[:3] if h.get("title")],
            },
        )