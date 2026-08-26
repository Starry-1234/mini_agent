from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .prompts import SYSTEM_PROMPT


def _gen_id() -> str:
    return uuid.uuid4().hex[:8]


# Sentinel: the original Phase 0 default. Persisted sessions that still have
# this string get migrated to the current SYSTEM_PROMPT on load (below).
# User-defined custom system prompts (anything else) are preserved verbatim.
_OLD_SYSTEM_PROMPT = "You are a helpful Agent. Use tools when needed."


def _default_plan_cache() -> dict:
    """Phase 2 hot-cache shape. All fields are strings/ints so asdict()
    round-trips cleanly through JSON. Returned fresh per Session instance
    so no two sessions share state."""
    return {
        "version": 0,         # bumped every time the LLM calls update_plan
        "stage": "",          # current milestone, e.g. "阶段1：基础语法"
        "next_task": "",      # the single next action the user should take
        "long_term_goal": "", # the user's stated goal (set from first turn)
        "last_updated": "",   # ISO 8601 UTC timestamp
    }


@dataclass
class Session:
    id: str
    # Default to the live prompts.SYSTEM_PROMPT so freshly created sessions
    # pick up the coach persona (or whatever role prompts.py declares) without
    # requiring every Session() call site to know about it.
    system_prompt: str = SYSTEM_PROMPT
    messages: list[dict] = field(default_factory=list)
    todos: list[dict] = field(default_factory=list)
    summary: str = ""
    # Phase 2: hot-cache of the active learning plan. Always present (so
    # context.py can read it without getattr defaults). ~50 tokens when
    # injected, so it's cheap to keep in every LLM call.
    plan_cache: dict = field(default_factory=_default_plan_cache)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def add_tool_call(self, *, call_id: str, name: str, args: dict) -> None:
        self.messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }],
        })

    def add_parallel_tool_calls(self, calls: list[tuple[str, str, dict]]) -> None:
        """Record multiple parallel tool calls in ONE assistant message.

        Bug J fix: previously add_tool_call was called once per iteration
        (with only the first tool's id), but tool_results were added for
        every tool. Result: orphan tool_results without a matching
        tool_call in the assistant message → LLM rejects with 400
        "tool call result does not follow tool call".

        All parallel calls must go into a single assistant message so
        every tool_call_id in the subsequent tool_results has a match.
        """
        self.messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            } for call_id, name, args in calls],
        })

    def add_tool_result(self, *, call_id: str, name: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        })


class SessionStore:
    def __init__(self, base: Path):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)

    def path_for(self, sid: str) -> Path:
        return self.base / f"{sid}.json"

    def load(self, sid: str) -> Session:
        p = self.path_for(sid)
        if not p.exists():
            return Session(id=sid)
        data = json.loads(p.read_text(encoding="utf-8"))
        # Migration: any session persisted with the old generic default (or an
        # empty prompt) is upgraded to the current SYSTEM_PROMPT. Sessions with
        # a user-set custom prompt are preserved verbatim.
        stored_prompt = data.get("system_prompt", "")
        if not stored_prompt or stored_prompt == _OLD_SYSTEM_PROMPT:
            stored_prompt = SYSTEM_PROMPT
        # Phase 2 migration: sessions saved before plan_cache existed get
        # the default empty cache. Merge rather than replace so any partial
        # state from a half-written save is preserved.
        plan_cache = data.get("plan_cache") or _default_plan_cache()
        return Session(
            id=data.get("id", sid),
            system_prompt=stored_prompt,
            messages=data.get("messages", []),
            todos=data.get("todos", []),
            summary=data.get("summary", ""),
            plan_cache=plan_cache,
        )

    def save(self, s: Session) -> None:
        p = self.path_for(s.id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)