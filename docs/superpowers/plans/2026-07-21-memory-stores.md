# Task 9: Embeddings, Short-Term Store, and Vector Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the pluggable three-layer memory backend interfaces and implementations specified by the Task 9 brief, with a zero-infrastructure local default.

**Architecture:** Create an isolated `agent.memory` package with three modules. `embeddings.py` defines deterministic and OpenAI-compatible embedding providers; `short_term.py` defines in-memory and optional Redis conversation records; `vector_store.py` defines a JSONL-backed local store plus optional Qdrant and Chroma adapters. Optional integrations import dependencies only in constructors and raise clear `RuntimeError`s when unavailable.

**Tech Stack:** Python 3, `typing.Protocol`, standard library (`hashlib`, `math`, `json`, `pathlib`, `collections`), optional `openai`, `redis`, `qdrant-client`, and `chromadb`, pytest.

## Global Constraints

- Implement exactly the APIs and code behavior in `F:/dev/AI_Tools/workspace/mini_agent/.superpowers/sdd/task-9-brief.md`.
- Do not modify existing files.
- Use `snake_case.py` filenames and create `agent/memory/__init__.py` as an empty package marker.
- Keep optional dependency imports lazy and guarded with clear `RuntimeError` messages.
- Use the brief’s test cycle: failing tests first, then implementation, then `python -m pytest tests/test_memory_stores.py -v`.
- Commit with the exact conventional message and trailer required by the brief.

---

### Task 1: Add failing memory-store tests

**Files:**
- Create: `F:/dev/AI_Tools/workspace/mini_agent/tests/test_memory_stores.py`

**Interfaces:**
- Consumes: Future `MockEmbedder`, `InMemoryShortTermStore`, and `LocalVectorStore` APIs.
- Produces: Two executable behavior tests proving short-term ordering and local keyword fallback.

- [ ] **Step 1: Write the failing test**

Create the test file with exactly:

```python
import math
from agent.memory.embeddings import MockEmbedder
from agent.memory.short_term import InMemoryShortTermStore
from agent.memory.vector_store import LocalVectorStore


def test_short_term_recent_order():
    s = InMemoryShortTermStore()
    for i in range(5):
        s.push("sid", {"role": "user", "content": str(i)})
    recent = s.recent("sid", 3)
    assert [r["content"] for r in recent] == ["2", "3", "4"]


def test_local_vector_search():
    e = MockEmbedder(dim=8)
    v = LocalVectorStore(embedder=e, path=None)
    v.upsert("a", "apple pie", None, {})
    v.upsert("b", "banana split", None, {})
    v.upsert("c", "cherry tart", None, {})
    v.upsert("d", "grape juice", None, {})
    res = v.search("banana", top_k=2)
    ids = [r[0] for r in res]
    assert "b" in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /f/dev/AI_Tools/workspace/mini_agent && python -m pytest tests/test_memory_stores.py -v
```

Expected: collection fails with `ModuleNotFoundError`/`ImportError` because `agent.memory` does not yet exist. Do not implement production code until this failure is observed.

---

### Task 2: Implement embedding providers

**Files:**
- Create: `F:/dev/AI_Tools/workspace/mini_agent/agent/memory/__init__.py`
- Create: `F:/dev/AI_Tools/workspace/mini_agent/agent/memory/embeddings.py`

**Interfaces:**
- Consumes: None.
- Produces: `Embedder` protocol with `embed(texts: list[str]) -> list[list[float]]`; `MockEmbedder(dim=32)`; `OpenAICompatEmbedder(api_key, base_url, model)`.

- [ ] **Step 1: Create the empty package marker**

Create an empty `agent/memory/__init__.py`.

- [ ] **Step 2: Implement the deterministic hash embedding**

Create `embeddings.py` with:

```python
from __future__ import annotations

import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _hash_vec(text: str, dim: int) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(dim):
        b = h[i % len(h)]
        out.append(((b / 255.0) * 2.0) - 1.0)
    n = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / n for x in out]


class MockEmbedder:
    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vec(t, self.dim) for t in texts]


class OpenAICompatEmbedder:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key, self.base_url, self.model = api_key, base_url, model

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
cd /f/dev/AI_Tools/workspace/mini_agent && python -m pytest tests/test_memory_stores.py -v
```

Expected: short-term/vector imports may still fail because those modules are not implemented; the embedding import itself should resolve. Continue to Task 3.

---

### Task 3: Implement short-term stores

**Files:**
- Create: `F:/dev/AI_Tools/workspace/mini_agent/agent/memory/short_term.py`

**Interfaces:**
- Consumes: None.
- Produces: `ShortTermStore` protocol; `InMemoryShortTermStore(maxlen=200)` with `push`, `recent`, and `clear`; guarded `RedisShortTermStore(url, maxlen=200, key_prefix="st:")` with the same methods.

- [ ] **Step 1: Implement the protocol and in-memory store**

Create `short_term.py` with:

```python
from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol


class ShortTermStore(Protocol):
    def push(self, sid: str, record: dict) -> None: ...
    def recent(self, sid: str, k: int) -> list[dict]: ...
    def clear(self, sid: str) -> None: ...


class InMemoryShortTermStore:
    def __init__(self, maxlen: int = 200) -> None:
        self._buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def push(self, sid: str, record: dict) -> None:
        self._buf[sid].append(record)

    def recent(self, sid: str, k: int) -> list[dict]:
        d = self._buf[sid]
        if k >= len(d):
            return list(d)
        return list(d)[-k:]

    def clear(self, sid: str) -> None:
        self._buf.pop(sid, None)


class RedisShortTermStore:
    """Optional backend. Requires `redis`; import is guarded so tests run without it."""

    def __init__(self, url: str, maxlen: int = 200, key_prefix: str = "st:") -> None:
        try:
            import redis  # type: ignore
        except ImportError as error:
            raise RuntimeError("redis backend requested but `redis` is not installed") from error
        self._r = redis.from_url(url)
        self.maxlen = maxlen
        self.prefix = key_prefix

    def _key(self, sid: str) -> str:
        return f"{self.prefix}{sid}"

    def push(self, sid: str, record: dict) -> None:
        import json

        self._r.rpush(self._key(sid), json.dumps(record, ensure_ascii=False))
        self._r.ltrim(self._key(sid), -self.maxlen, -1)

    def recent(self, sid: str, k: int) -> list[dict]:
        import json

        raw = self._r.lrange(self._key(sid), -k, -1)
        return [json.loads(item) for item in raw]

    def clear(self, sid: str) -> None:
        self._r.delete(self._key(sid))
```

- [ ] **Step 2: Run the short-term test**

Run:

```bash
cd /f/dev/AI_Tools/workspace/mini_agent && python -m pytest tests/test_memory_stores.py::test_short_term_recent_order -v
```

Expected: `1 passed`.

---

### Task 4: Implement local and optional vector stores

**Files:**
- Create: `F:/dev/AI_Tools/workspace/mini_agent/agent/memory/vector_store.py`

**Interfaces:**
- Consumes: Optional `Embedder`-compatible object.
- Produces: `VectorStore` protocol; `LocalVectorStore(embedder=None, path=None)` with `upsert` and `search`; `QdrantVectorStore(url, collection="memory", embedder=None)`; `ChromaVectorStore(path, collection="memory")`.

- [ ] **Step 1: Implement cosine similarity and local JSONL store**

Create `vector_store.py` with the brief’s exact behavior:

```python
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol


class VectorStore(Protocol):
    def upsert(self, id: str, text: str, vector: list[float] | None, meta: dict) -> None: ...
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]: ...


def _cos(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    return dot / (na * nb) if na and nb else 0.0


class LocalVectorStore:
    """JSONL-backed. Uses embedder if vectors are None, else uses provided vectors.
    Search falls back to keyword overlap when embedder is unavailable."""

    def __init__(self, embedder=None, path: Path | None = None) -> None:
        self.embedder = embedder
        self.path = path
        self._items: dict[str, dict] = {}
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                self._items[item["id"]] = item

    def _flush(self) -> None:
        if not self.path:
            return
        self.path.write_text(
            "\n".join(json.dumps(value, ensure_ascii=False) for value in self._items.values()),
            encoding="utf-8",
        )

    def upsert(self, id: str, text: str, vector: list[float] | None, meta: dict) -> None:
        vec = vector
        if vec is None and self.embedder is not None:
            vec = self.embedder.embed([text])[0]
        self._items[id] = {"id": id, "text": text, "vector": vec, "meta": meta}
        self._flush()

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]:
        if self.embedder is not None:
            query_vector = self.embedder.embed([query])[0]
            scored = []
            for item in self._items.values():
                if item["vector"] is None:
                    continue
                scored.append((item["id"], _cos(query_vector, item["vector"]), item["meta"]))
            scored.sort(key=lambda result: result[1], reverse=True)
            return scored[:top_k]

        query_tokens = set(query.lower().split())
        scored = []
        for item in self._items.values():
            text_tokens = set(item["text"].lower().split())
            intersection = len(query_tokens & text_tokens)
            if intersection:
                scored.append((item["id"], intersection / max(len(query_tokens | text_tokens), 1), item["meta"]))
        scored.sort(key=lambda result: result[1], reverse=True)
        return scored[:top_k]
```

- [ ] **Step 2: Implement guarded Qdrant adapter**

Append the Qdrant implementation from the brief. Its constructor must lazily import `QdrantClient`, raise `RuntimeError("qdrant backend requested but `qdrant-client` is not installed")` if absent, require an embedder with `RuntimeError("QdrantVectorStore requires an embedder for vectorisation")`, create a cosine collection based on the embedder dimension when absent, and expose the exact `upsert` and `search` tuple format.

- [ ] **Step 3: Implement guarded Chroma adapter**

Append the Chroma implementation from the brief. Its constructor must lazily import `chromadb`, raise `RuntimeError("chroma backend requested but `chromadb` is not installed")` if absent, use `PersistentClient(path=path)`, and expose the exact `upsert` and `search` behavior from the brief.

- [ ] **Step 4: Run the local vector test**

Run:

```bash
cd /f/dev/AI_Tools/workspace/mini_agent && python -m pytest tests/test_memory_stores.py::test_local_vector_search -v
```

Expected: `1 passed`.

---

### Task 5: Run full verification and self-review

**Files:**
- No additional files.

**Interfaces:**
- Consumes: All Task 9 production modules and tests.
- Produces: Verified implementation with no existing-file modifications.

- [ ] **Step 1: Run the required focused test suite**

```bash
cd /f/dev/AI_Tools/workspace/mini_agent && python -m pytest tests/test_memory_stores.py -v
```

Expected: `2 passed`.

- [ ] **Step 2: Run the existing full test suite**

```bash
cd /f/dev/AI_Tools/workspace/mini_agent && python -m pytest -q
```

Expected: all existing tests and the two new tests pass.

- [ ] **Step 3: Inspect the diff and confirm scope**

```bash
cd /f/dev/AI_Tools/workspace/mini_agent && git diff --check && git status --short
```

Confirm only these paths are new or changed:

- `agent/memory/__init__.py`
- `agent/memory/embeddings.py`
- `agent/memory/short_term.py`
- `agent/memory/vector_store.py`
- `tests/test_memory_stores.py`

- [ ] **Step 4: Commit with the exact required message**

```bash
cd /f/dev/AI_Tools/workspace/mini_agent && git add agent/memory/ tests/test_memory_stores.py && git commit -m "feat: embeddings + short-term + vector stores (local default, optional redis/qdrant/chroma)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [x] All brief requirements map to Tasks 1–5.
- [x] No production code is planned before the failing test.
- [x] Optional imports are lazy and guarded with exact error messages.
- [x] Interfaces and return types match the brief, including the brief’s `search(query: str, ...)` signature despite the introductory shorthand mentioning vectors.
- [x] Existing files remain untouched.
- [x] No placeholders or undefined implementation references remain; adapter requirements reproduce the brief’s exact behavior and constructors.
