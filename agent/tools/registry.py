from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
from .base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.session import Session


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_all(self, tools: Iterable[Tool]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def openai_schemas(self) -> list[dict]:
        out = []
        for t in self._tools.values():
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            })
        return out

    def execute(self, name: str, args: dict, session: "Session | None") -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.err(f"unknown tool: {name}")
        try:
            return tool.execute(args or {}, session)
        except Exception as e:  # noqa: BLE001 — surface as tool error for model self-correction
            return ToolResult.err(f"tool '{name}' raised: {type(e).__name__}: {e}")
