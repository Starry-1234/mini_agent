from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


_KIND_COLOR = {
    "user": "\033[97m",
    "thought": "\033[90m",
    "tool_call": "\033[36m",
    "tool_result": "\033[33m",
    "assistant": "\033[92m",
    "error": "\033[91m",
    "summary": "\033[95m",
    "recall": "\033[94m",
}
_RESET = "\033[0m"


class TraceLogger:
    def __init__(self, sessions_dir: Path, session_id: str):
        self.sessions_dir = Path(sessions_dir)
        self.session_id = session_id
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.sessions_dir / f"{session_id}.trace.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def event(self, kind: str, **fields) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **fields}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._print(kind, fields)

    def _print(self, kind: str, fields: dict) -> None:
        color = _KIND_COLOR.get(kind, "")
        head = f"{color}[{kind}]{_RESET}"
        body = " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False)}"
            for key, value in fields.items()
        )
        sys.stderr.write(f"{head} {body}\n")
        sys.stderr.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
