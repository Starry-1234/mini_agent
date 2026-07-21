from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _gen_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Session:
    id: str
    system_prompt: str = "You are a helpful Agent. Use tools when needed."
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
        return Session(
            id=data.get("id", sid),
            system_prompt=data.get("system_prompt", Session.system_prompt),
            messages=data.get("messages", []),
            todos=data.get("todos", []),
            summary=data.get("summary", ""),
        )

    def save(self, s: Session) -> None:
        p = self.path_for(s.id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)