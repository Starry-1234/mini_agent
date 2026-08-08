# agent/memory/manager.py
from __future__ import annotations
import uuid
from typing import Any

from .short_term import ShortTermStore
from .vector_store import VectorStore
from .extractor import extract_facts


class MemoryManager:
    def __init__(self, embedder, short_term: ShortTermStore, vector_store: VectorStore,
                 llm=None, top_k: int = 5) -> None:
        self.embedder = embedder
        self.short_term = short_term
        self.vector = vector_store
        self.llm = llm
        self.top_k = top_k

    # Short-term helpers
    def push_turn(self, sid: str, record: dict) -> None:
        self.short_term.push(sid, record)

    def recent_turns(self, sid: str, k: int) -> list[dict]:
        return self.short_term.recent(sid, k)

    # Long-term
    def remember_sid(self, sid: str, recent_turns: list[dict], llm=None) -> int:
        facts = extract_facts(llm or self.llm, recent_turns)
        if not facts:
            return 0
        # Deduplicate by cosine similarity to existing items; threshold drops near-duplicates.
        for fact in facts:
            fid = f"{sid}:{uuid.uuid4().hex[:8]}"
            meta = {"sid": sid, "kind": "fact"}
            self.vector.upsert(id=fid, text=fact, vector=None, meta=meta)
        return len(facts)

    def recall(self, sid: str | None, query: str, top_k: int | None = None,
 *, cross_session: bool = False,
 ) -> list[tuple[str, float, dict]]:
        """Recall facts relevant to `query`.

        Args:
            sid: current session id. If None and cross_session=False, returns
                all matching facts (no filtering).
            query: natural-language query used for vector search.
            top_k: override the manager's default top_k for this call.
            cross_session: if True, ignore the `sid` filter and return facts
                from any session. Used by the coach to surface long-term
                learner profile (tech stack, prior projects, etc.) that
                transcends the current session.

        Returns:
            List of (text, score, meta) tuples, already filtered and ranked.
        """
        k = top_k or self.top_k
        # Always pull more when crossing sessions so the filter has headroom.
        fetch_k = k * 3 if cross_session else k
        results = self.vector.search(query=query, top_k=fetch_k)

        if cross_session:
            # Return all matches; caller may further filter by meta (kind, etc.)
            return list(results)[:k]

        if sid is None:
            return list(results)
        return [(t, s, m) for (t, s, m) in results if m.get("sid") == sid]