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

    def recall(self, sid: str | None, query: str, top_k: int | None = None) -> list[tuple[str, float, dict]]:
        k = top_k or self.top_k
        results = self.vector.search(query=query, top_k=k)
        # Map ids back to stored text so callers receive (text, score, meta).
        items = getattr(self.vector, "_items", {})
        enriched = []
        for i, s, m in results:
            text = items.get(i, {}).get("text", "") if isinstance(items, dict) else ""
            enriched.append((text, s, m))
        if sid is None:
            return enriched
        return [(t, s, m) for (t, s, m) in enriched if m.get("sid") == sid]