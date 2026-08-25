"""Text sanitization utilities.

Centralized helpers for fixing up text that flows through the LLM call
boundary (system prompt, user input, tool output, trace records, embedder
input, etc.). Every input/output path should call these instead of
re-implementing its own `_strip_surrogates()`.

Why this lives in `starry_code/text/` (and not a sub-module of one
caller): multiple modules need the same primitives — embeddings, trace,
cli, runtime, llm. Putting it in one place keeps the regex single-sourced
and makes it easy to add new sanitizers (control-char stripping, NFC
normalization, etc.) in one commit.
"""
from __future__ import annotations
import re
from typing import Any

# U+D800..U+DFFF — the UTF-16 surrogate range which UTF-8 strictly
# forbids. Reasoning models (MiniMax-M3, DeepSeek-R1) and Windows console
# input (DELETE key behavior) can both produce strings containing these
# codepoints; every code path that hits `.encode("utf-8")` (sha256 in
# embeddings, json.dumps to disk, httpx POST bodies) crashes on them.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

# C0/C1 control characters that the LLM API may reject or that look like
# junk in the terminal. We keep \t, \n, \r (real whitespace). We strip
# the rest so a stray \x01 in user input doesn't reach the LLM.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_surrogates(text: str) -> str:
    """Remove surrogate codepoints (U+D800..U+DFFF) from `text`.

    Use at every boundary that converts str → bytes (sha256, json.dumps,
    httpx). The empty string round-trips; None-ish input is not expected
    (callers must check before calling).
    """
    if not text:
        return text
    return _SURROGATE_RE.sub("", text)


def strip_control_chars(text: str) -> str:
    """Remove C0/C1 control chars (except \\t \\n \\r) from `text`.

    These rarely appear in real user input but occasionally leak in via
    copy-paste from terminals / log scraping. The LLM API rejects some
    of them silently. Keep this conservative — only strip what is almost
    certainly junk, never modify meaning.
    """
    if not text:
        return text
    return _CONTROL_CHARS_RE.sub("", text)


def sanitize_for_io(text: str) -> str:
    """Combined: strip surrogates + control chars.

    Use this everywhere text gets encoded to UTF-8 (terminal writes,
    file writes, network requests, hashing). Cheap (~linear in length);
    safe to call repeatedly (idempotent).
    """
    if not text:
        return text
    # One pass each — they're disjoint character classes so order doesn't
    # matter, and a single combined regex would be less readable.
    return strip_control_chars(strip_surrogates(text))


def sanitize_field(value: Any) -> Any:
    """Sanitize a single field value, recursing into dicts/lists.

    Used by `trace.event()` and similar boundary code so any string
    nested inside a structured payload is also stripped — prevents
    surrogates hiding inside `args={"text": "you\udce5"}` from slipping
    past a top-level `isinstance(value, str)` check.
    """
    if isinstance(value, str):
        return sanitize_for_io(value)
    if isinstance(value, dict):
        return {k: sanitize_field(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_field(v) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize_field(v) for v in value)
    return value