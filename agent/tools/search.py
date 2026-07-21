from __future__ import annotations
from .base import Tool, ToolResult


_DATA = {
    "python": [
        ("Python is a high-level, general-purpose programming language.", "https://example.com/python"),
        ("Python emphasizes code readability and supports multiple paradigms.", "https://example.com/python-2"),
    ],
    "agent": [
        ("An agent perceives its environment and takes actions to achieve goals.", "https://example.com/agent"),
        ("LLM-based agents use large language models as their reasoning core.", "https://example.com/llm-agent"),
    ],
    "weather": [
        ("Weather describes atmospheric conditions at a specific time and place.", "https://example.com/weather"),
    ],
}


class SearchTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="search",
            description="Mock web search. Returns up to 3 short results for a query keyword.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        q = (args or {}).get("query", "").strip().lower()
        if not q:
            return ToolResult.err("query is required")
        hits = []
        for key, items in _DATA.items():
            if key in q:
                hits.extend(items)
        if not hits:
            return ToolResult.ok(f"No mock results for '{q}'.")
        lines = []
        for i, (snippet, url) in enumerate(hits[:3], 1):
            lines.append(f"{i}. {snippet}\n   {url}")
        return ToolResult.ok("\n".join(lines))