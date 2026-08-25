"""Remotive Jobs source — hiring signal.

Real endpoint (added in a later phase):
    GET https://remotive.com/api/remote-jobs?search={topic}&limit=50

What we extract (when real):
- total jobs returned for query
- jobs posted in last 7 days
- top 3 job titles for the notes field

Score normalization (same shape as GitHub, but tuned for hiring velocity):
    score = min(10, log10(jobs_last_7d + 1) * 3.5)

Why the multiplier (3.5) differs from GitHub's (4.0): jobs are noisier
than stars (one big posting can swing the count) so we dampen slightly.

Failure mode: empty result for known-good topic → likely taxonomy drift;
return `direction="unknown"`. Don't fabricate a 0.
"""
from __future__ import annotations
import math

from .github_trending import SourceResult


class RemotiveJobsProvider:
    """Mock by default. Real HTTP pluggable via subclass."""

    name = "remotive"

    def __init__(self, data: dict[str, dict] | None = None) -> None:
        self._data = data if data is not None else self._default_data()

    def _default_data(self) -> dict[str, dict]:
        # Curated mock — approximate weekly job counts for known topics.
        return {
            "rust":          {"jobs_7d": 12,  "top_titles": ["Rust Engineer", "Backend Developer"]},
            "go":            {"jobs_7d": 38,  "top_titles": ["Go Developer", "Backend Engineer", "Platform Engineer"]},
            "python":        {"jobs_7d": 95,  "top_titles": ["Python Developer", "Data Engineer", "ML Engineer"]},
            "java":          {"jobs_7d": 47,  "top_titles": ["Java Developer", "Backend Engineer"]},
            "typescript":    {"jobs_7d": 56,  "top_titles": ["Frontend Engineer", "Full-Stack Developer"]},
            "react":         {"jobs_7d": 64,  "top_titles": ["React Developer", "Full-Stack Engineer"]},
            "vue":           {"jobs_7d": 18,  "top_titles": ["Frontend Developer"]},
            "kubernetes":    {"jobs_7d": 22,  "top_titles": ["DevOps Engineer", "SRE"]},
            "swift":         {"jobs_7d": 8,   "top_titles": ["iOS Developer"]},
            "kotlin":        {"jobs_7d": 14,  "top_titles": ["Android Developer"]},
            "javascript":    {"jobs_7d": 52,  "top_titles": ["Frontend Developer"]},
            "node.js":       {"jobs_7d": 41,  "top_titles": ["Backend Developer", "Full-Stack Engineer"]},
            "docker":        {"jobs_7d": 19,  "top_titles": ["DevOps Engineer"]},
            "machine-learning": {"jobs_7d": 31, "top_titles": ["ML Engineer", "Research Engineer"]},
            "data-science":  {"jobs_7d": 22,  "top_titles": ["Data Scientist", "ML Engineer"]},
            "ai-agent":      {"jobs_7d": 17,  "top_titles": ["AI Engineer"]},
            "webrtc":        {"jobs_7d": 4,   "top_titles": ["WebRTC Engineer"]},
            "blockchain":    {"jobs_7d": 6,   "top_titles": ["Solidity Developer"]},
        }

    def fetch(self, topic: str, region: str = "cn") -> SourceResult:
        key = (topic or "").strip().lower()
        info = self._data.get(key)
        if info is None:
            return SourceResult(
                score=None,
                direction="unknown",
                notes=f"Remotive: no postings for {key!r}",
                raw={"topic": key},
            )
        jobs = int(info["jobs_7d"])
        # 3.5 multiplier (vs GitHub's 4.0) — jobs are noisier than stars.
        score = min(10.0, math.log10(jobs + 1) * 3.5)
        if score >= 7:
            direction = "rising"
        elif score >= 4:
            direction = "stable"
        else:
            direction = "falling"
        tops = ", ".join(info.get("top_titles", [])[:3])
        return SourceResult(
            score=round(score, 1),
            direction=direction,
            notes=f"Remotive: {jobs} jobs in 7d ({tops})",
            raw={"topic": key, "jobs_7d": jobs},
        )