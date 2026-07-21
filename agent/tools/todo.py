from __future__ import annotations
from .base import Tool, ToolResult


class TodoTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="todo",
            description="Manage a per-session todo list. Actions: add (text), list, complete (id).",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "list", "complete"]},
                    "text":   {"type": "string"},
                    "id":     {"type": "integer"},
                },
                "required": ["action"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        if session is None:
            return ToolResult.err("todo tool requires an active session")
        action = (args or {}).get("action")
        todos = session.todos
        if action == "add":
            text = (args or {}).get("text", "").strip()
            if not text:
                return ToolResult.err("text is required for add")
            next_id = (max((t["id"] for t in todos), default=0) + 1)
            todos.append({"id": next_id, "text": text, "done": False})
            return ToolResult.ok(f"added todo #{next_id}: {text}")
        if action == "list":
            if not todos:
                return ToolResult.ok("(no todos)")
            lines = [f"  #{t['id']} [{'x' if t['done'] else ' '}] {t['text']}" for t in todos]
            return ToolResult.ok("todos:\n" + "\n".join(lines))
        if action == "complete":
            tid = args.get("id")
            if tid is None:
                return ToolResult.err("id is required for complete")
            for t in todos:
                if t["id"] == tid:
                    t["done"] = True
                    return ToolResult.ok(f"completed todo #{tid}: {t['text']}")
            return ToolResult.err(f"todo #{tid} not found")
        return ToolResult.err(f"unknown action: {action}")