"""read_artifact — fetch a tool output that ContextBuilder offloaded.

Background
----------
When a tool returns >500 chars, ContextBuilder writes the full payload to
sessions/{sid}/artifacts/{call_id}.json and replaces the message with a
short summary card in the LLM context. The coach can call this tool to
fetch the full text back if it needs to reference specific details.

Security
--------
The path is validated to fall under sessions/{sid}/artifacts/. Anything
outside that root is rejected — without this guard, the LLM could be
tricked into reading arbitrary JSON files on disk via path traversal
(../../etc/passwd style).
"""
from __future__ import annotations
import json
from pathlib import Path
from .base import Tool, ToolResult


class ReadArtifactTool(Tool):
    """Read a previously offloaded tool output."""

    def __init__(self, sessions_dir: Path | None = None) -> None:
        # Default to "sessions" so production (where artifacts live under
        # ./sessions/{sid}/artifacts/) works without setup. Tests pass a
        # tmp_path-equivalent via this constructor.
        self._sessions_dir = Path(sessions_dir) if sessions_dir else Path("sessions")
        super().__init__(
            name="read_artifact",
            description=(
                "读取被 ContextBuilder offload 到 sessions/{sid}/artifacts/ 的完整工具输出。"
                "当 history 里出现 [artifact saved] 卡片，且你需要完整原文时调用。"
                "path 参数就是卡片里 path 字段的值。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "artifact 卡片里的 path 字段绝对路径",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "截断长度，默认 4000；返回过长会被截断",
                        "minimum": 100,
                        "maximum": 50000,
                    },
                },
                "required": ["path"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        if session is None:
            return ToolResult.err("read_artifact requires an active session")
        path_str = (args or {}).get("path", "").strip()
        if not path_str:
            return ToolResult.err("path is required")
        try:
            max_chars = int((args or {}).get("max_chars", 4000))
        except (TypeError, ValueError):
            max_chars = 4000

        p = Path(path_str)
        try:
            p_resolved = p.resolve()
        except (OSError, ValueError):
            return ToolResult.err(f"invalid path: {path_str}")

        # Security: must live under {sessions_dir}/{sid}/artifacts/. We
        # resolve both sides so a symlink or relative traversal can't escape.
        allowed_root = (self._sessions_dir / session.id / "artifacts").resolve()
        try:
            p_resolved.relative_to(allowed_root)
        except ValueError:
            return ToolResult.err(
                f"path not under {allowed_root} (refused for safety)"
            )

        if not p_resolved.exists():
            return ToolResult.err(f"artifact not found: {p_resolved}")

        try:
            data = json.loads(p_resolved.read_text(encoding="utf-8"))
            content = data.get("content", "")
        except json.JSONDecodeError as e:
            return ToolResult.err(f"artifact corrupt: {e}")
        except OSError as e:
            return ToolResult.err(f"read failed: {e}")

        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return ToolResult.ok(content)