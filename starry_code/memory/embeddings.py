from __future__ import annotations
import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# Strip lone surrogates (U+D800..U+DFFF) before encoding to UTF-8. Reasoning
# models and Windows console input can both produce strings containing
# surrogate codepoints; sha256 (and any other .encode("utf-8")) would
# raise UnicodeEncodeError on them. Mirrors cli._strip_surrogates so
# the embedder never crashes on bad input.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _strip_surrogates(text: str) -> str:
    if not text:
        return text
    return _SURROGATE_RE.sub("", text)


def _hash_vec(text: str, dim: int) -> list[float]:
    # Sanitize BEFORE hashing — .encode("utf-8") rejects lone surrogates
    # by spec, which would crash every embed() call on dirty input.
    safe = _strip_surrogates(text)
    h = hashlib.sha256(safe.encode("utf-8")).digest()
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
        self.api_key, self.base_url, self.model = model, base_url, api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Sanitize before sending to embedding API: some providers reject
        # lone surrogates (400 Bad Request) which would otherwise crash us.
        safe_texts = [_strip_surrogates(t) for t in texts]
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(model=self.model, input=safe_texts)
        return [item.embedding for item in response.data]