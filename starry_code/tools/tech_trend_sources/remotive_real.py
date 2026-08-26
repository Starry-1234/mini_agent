"""Real Remotive Jobs API provider.

Endpoint: GET https://remotive.com/api/remote-jobs?search={topic}&limit=100

Auth: none.
Response: {"0-legal-notice": ..., "job-count": N, "jobs": [...]}

We aggregate:
- 7-day window (filter jobs where `publication_date` < 7 days ago)
- Top 3 titles for the notes field

Note: Remotive's `limit` caps at ~100. For real backends this is fine
for an MVP — full pagination would be R3+ work.
"""
from __future__ import annotations
import math
import os
from datetime import datetime, timezone, timedelta

from .github_trending import SourceResult


_REMOTIVE_API = "https://remotive.com/api/remote-jobs"


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # Remotive's `publication_date` is ISO 8601 but **without** a
        # timezone suffix (naive datetime, e.g. "2026-08-21T05:54:39").
        # We treat naive values as UTC. If they include "Z" or "+HH:MM",
        # fromisoformat handles them natively.
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _score(jobs_7d: int) -> tuple[float, str]:
    """Same shape as github: log10(jobs+1)*3.5, capped 0-10.

    Lower multiplier (vs github's 4.0): jobs are noisier than stars
    so we dampen the signal.
    """
    score = min(10.0, math.log10(jobs_7d + 1) * 3.5)
    if score >= 7:
        return round(score, 1), "rising"
    if score >= 4:
        return round(score, 1), "stable"
    return round(score, 1), "falling"


class RealRemotiveJobsProvider:
    name = "remotive"
    network = True

    def __init__(self, timeout: float = 5.0, days: int = 7,
                 limit: int = 100) -> None:
        self.timeout = timeout
        self.days = days
        self.limit = limit

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
                _REMOTIVE_API,
                params={"search": key, "limit": self.limit},
                headers={"User-Agent": "starry-code/1.0"},
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise RuntimeError(f"remotive network error: {type(e).__name__}: {e}")

        if resp.status_code >= 500:
            raise RuntimeError(f"remotive server error: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()

        jobs = data.get("jobs") or []
        if not jobs:
            return SourceResult(
                score=None, direction="unknown",
                notes=f"Remotive: no postings for {key!r}",
                raw={"topic": key, "total": data.get("job-count", 0)},
            )

        # Filter to last N days
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.days)
        recent = [j for j in jobs
                  if (pub := _parse_date(j.get("publication_date"))) and pub >= cutoff]

        # Top 3 titles for the notes field
        top_titles = [j.get("title", "") for j in recent[:3] if j.get("title")]

        jobs_7d = len(recent)
        score, direction = _score(jobs_7d)
        return SourceResult(
            score=score,
            direction=direction,
            notes=(f"Remotive: {jobs_7d} jobs in {self.days}d for {key} "
                   f"({', '.join(top_titles[:2]) or 'no titles'})"),
            raw={
                "topic": key,
                "total_listed": data.get("job-count", len(jobs)),
                "jobs_in_window": jobs_7d,
                "top_titles": top_titles,
            },
        )