# agent/llm.py
from __future__ import annotations
import hashlib
import math
import re
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


# Heuristic: extract the arithmetic expression the user asked about.
# Match patterns like "2+2", "what is 3 * 4", "compute 10 / 2", etc.
_MATH_QUERY_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?\s*[+\-*/%][\s0-9.+\-*/%()]+)")


class _ScriptedMockLLM(MockLLMClient):
    """Internal: wraps MockLLMClient with naming-prompt routing.

    Routes messages that contain "会话命名助手" (the auto-naming system
    prompt marker) to a fixed Chinese slug so `--mock` exercises the
    rename flow without exhausting the scripted queue.
    """

    _NAMING_RESPONSE = {"choices": [{"message": {"role": "assistant", "content": "测试会话"}}]}

    def __init__(self, responses: list[dict], embed_dim: int = 16) -> None:
        super().__init__(chat_responses=responses, embed_dim=embed_dim)

    def chat(self, messages, tools=None) -> dict:  # type: ignore[override]
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, str) and "会话命名助手" in content:
                return self._NAMING_RESPONSE
        return super().chat(messages, tools=tools)


def make_default_mock_llm(user_message: str, embed_dim: int = 16) -> MockLLMClient:
    """Pre-scripted MockLLMClient for common CLI smoke queries.

    Used only by `cli.py --mock` for offline testing — not for real LLM work.
    The demo script keeps its own scripted flow and is unaffected.

    - Math/arithmetic queries (contains a number and an operator + - * /):
        step 1 -> tool_call to `calculator` with the extracted expression
        step 2 -> final answer "The answer is <result>." (placeholder; the real
                  answer is whatever the calculator tool returns).
        step 3 -> extractor no-facts response.
    - Otherwise: a single direct final answer "(mock) I received your message."
      plus an extractor no-facts response.

    Auto-naming prompts are intercepted and return "测试会话" so `--mock`
    exercises the rename flow without exhausting the queue.
    """
    no_facts = {"choices": [{"message": {"role": "assistant", "content": "[]"}}]}
    expr = _extract_expression(user_message)
    if expr is not None:
        responses = [
            {"choices": [{"message": {
                "role": "assistant",
                "content": "computing",
                "tool_calls": [{"id": "mock_calc", "type": "function",
                                "function": {"name": "calculator",
                                             "arguments": '{"expression": "' + expr + '"}'}}],
            }}]},
            {"choices": [{"message": {
                "role": "assistant",
                "content": "The answer is <result>.",
            }}]},
            no_facts,
        ]
    else:
        responses = [
            {"choices": [{"message": {
                "role": "assistant",
                "content": "(mock) I received your message.",
            }}]},
            no_facts,
        ]
    return _ScriptedMockLLM(responses, embed_dim=embed_dim)


def _extract_expression(user_message: str) -> str | None:
    m = _MATH_QUERY_RE.search(user_message)
    if not m:
        return None
    expr = m.group(1).strip()
    # Strip trailing junk that is unlikely to be part of the expression.
    return expr.rstrip(".,;:?")


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