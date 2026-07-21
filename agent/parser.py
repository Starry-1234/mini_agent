# agent/parser.py
from __future__ import annotations
import json
import re
from dataclasses import dataclass


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ParsedResponse:
    thought: str
    tool_calls: list[ToolCall]
    final_answer: str | None


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_thinking(text: str) -> str | None:
    """Remove <think>...</think> reasoning blocks from a final answer.

    Reasoning models (MiniMax-M3, DeepSeek-R1, GLM-Z1, ...) prefix their
    answer with a `<think>...</think>` block. We keep that in `thought`
    (trace visibility) but must strip it from what the user sees.

    Match is case-insensitive on the tag name, spans newlines, and is
    non-greedy so multiple blocks are each removed. Leading/trailing
    whitespace left behind is stripped. Returns None if nothing remains.
    """
    if not text:
        return None
    cleaned = _THINK_RE.sub("", text).strip()
    return cleaned or None


def parse_response(raw: dict) -> ParsedResponse:
    msg = raw["choices"][0]["message"]
    thought = msg.get("content") or ""
    raw_calls = msg.get("tool_calls") or []
    tool_calls: list[ToolCall] = []
    for c in raw_calls:
        try:
            args = json.loads(c["function"]["arguments"]) if c["function"].get("arguments") else {}
        except Exception:
            args = {"_raw": c["function"].get("arguments", "")}
        tool_calls.append(ToolCall(id=c["id"], name=c["function"]["name"], args=args))

    if tool_calls:
        return ParsedResponse(thought=thought, tool_calls=tool_calls, final_answer=None)

    # Text fallback: try to extract <tool_call>{...}</tool_call> / fenced JSON
    fb = _text_fallback(thought)
    if fb is not None:
        return ParsedResponse(thought=thought, tool_calls=[fb], final_answer=None)
    return ParsedResponse(thought=thought, tool_calls=[], final_answer=_strip_thinking(thought))


def _text_fallback(text: str) -> ToolCall | None:
    m = re.search(r"\{[^{}]*\"name\"[^{}]*\"arguments\"[^{}]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return ToolCall(id="fallback", name=obj["name"], args=obj.get("arguments", {}))
    except Exception:
        return None