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
        return Session(
            id=data.get("id", sid),
            system_prompt=stored_prompt,
            messages=data.get("messages", []),
            todos=data.get("todos", []),
            summary=data.get("summary", ""),
        )

    def save(self, s: Session) -> None:
        p = self.path_for(s.id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)