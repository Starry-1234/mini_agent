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
