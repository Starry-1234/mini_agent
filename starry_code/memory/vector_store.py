from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol


class VectorStore(Protocol):
    def upsert(self, id: str, text: str, vector: list[float] | None, meta: dict) -> None: ...
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]:
        """Return (text, score, meta) tuples for the top_k matches."""


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
        query_tokens = set(query.lower().split())
        keyword_ids = set()
        keyword_scored = []
        for item in self._items.values():
            text_tokens = set(item["text"].lower().split())
            intersection = len(query_tokens & text_tokens)
            if intersection:
                keyword_ids.add(item["id"])
                keyword_scored.append(
                    (item["text"], intersection / max(len(query_tokens | text_tokens), 1), item["meta"])
                )

        if self.embedder is not None:
            query_vector = self.embedder.embed([query])[0]
            scored = []
            for item in self._items.values():
                if item["vector"] is None:
                    continue
                score = _cos(query_vector, item["vector"])
                if item["id"] in keyword_ids:
                    score = max(score, 1.0)
                scored.append((item["text"], score, item["meta"]))
            scored.sort(key=lambda result: result[1], reverse=True)
            return scored[:top_k]

        keyword_scored.sort(key=lambda result: result[1], reverse=True)
        return keyword_scored[:top_k]


class QdrantVectorStore:
    def __init__(self, url: str, collection: str = "memory", embedder=None) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except ImportError as error:
            raise RuntimeError("qdrant backend requested but `qdrant-client` is not installed") from error
        if embedder is None:
            raise RuntimeError("QdrantVectorStore requires an embedder for vectorisation")
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._embedder = embedder
        if not self._client.collection_exists(collection):
            dim = len(embedder.embed(["dim-probe"])[0])
            from qdrant_client.http import models  # type: ignore

            self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def upsert(self, id: str, text: str, vector, meta: dict) -> None:
        from qdrant_client.http import models  # type: ignore

        vec = vector or self._embedder.embed([text])[0]
        self._client.upsert(
            self._collection,
            points=[models.PointStruct(id=id, vector=vec, payload={"text": text, **meta})],
        )

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]:
        qv = self._embedder.embed([query])[0]
        hits = self._client.search(self._collection, query_vector=qv, limit=top_k)
        out: list[tuple[str, float, dict]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            out.append((text, float(hit.score), payload))
        return out


class ChromaVectorStore:
    def __init__(self, path: str, collection: str = "memory") -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as error:
            raise RuntimeError("chroma backend requested but `chromadb` is not installed") from error
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(collection)

    def upsert(self, id: str, text: str, vector, meta: dict) -> None:
        kwargs = {"ids": [id], "documents": [text], "metadatas": [meta]}
        if vector is not None:
            kwargs["embeddings"] = [vector]
        self._collection.upsert(**kwargs)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]:
        result = self._collection.query(query_texts=[query], n_results=top_k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            (document, 1.0 - float(distance), dict(metadata or {}))
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]
