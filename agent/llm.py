# agent/llm.py
from __future__ import annotations
import hashlib
import math
from typing import Any


def _hash_vec(text: str, dim: int = 16) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(dim):
        byte = h[i % len(h)]
        out.append(((byte / 255.0) * 2.0) - 1.0)
    n = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / n for x in out]


class MockLLMClient:
    """In-memory client for unit tests. Embeds are deterministic by hash."""

    def __init__(self, chat_responses: list[dict] | None = None, embed_dim: int = 16) -> None:
        self._responses = list(chat_responses or [])
        self._idx = 0
        self._dim = embed_dim

    def chat(self, messages, tools=None) -> dict:
        if self._idx >= len(self._responses):
            raise RuntimeError("MockLLMClient: no more scripted responses")
        r = self._responses[self._idx]
        self._idx += 1
        return r

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vec(t, self._dim) for t in texts]


class LLMClient:
    """Thin OpenAI-compatible client. Lazy-initialised SDK; never imported unless used."""

    def __init__(self, api_key: str = "", base_url: str = "", model: str = "",
                 embed_api_key: str = "", embed_base_url: str = "", embed_model: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.embed_api_key = embed_api_key or api_key
        self.embed_base_url = embed_base_url or base_url
        self.embed_model = embed_model
        self._client = None

    def _sdk(self):
        if self._client is None:
            if not self.api_key or not self.base_url:
                raise RuntimeError("LLMClient: api_key and base_url are required for real calls")
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(self, messages, tools=None) -> dict:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = self._sdk().chat.completions.create(**kwargs)
        return resp.model_dump()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.embed_model:
            raise RuntimeError("LLMClient: embed_model not configured")
        from openai import OpenAI
        c = OpenAI(api_key=self.embed_api_key, base_url=self.embed_base_url)
        r = c.embeddings.create(model=self.embed_model, input=texts)
        return [item.embedding for item in r.data]