from __future__ import annotations

import json
import re
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

# Local surrogate strip — U+D800..U+DFFF is the UTF-16 surrogate range
# which UTF-8 strictly forbids. Reasoning models (MiniMax-M3, DeepSeek-R1)
# occasionally emit lone surrogates mid-response; without stripping, every
# sys.stderr.write() / json.dumps() crashes with UnicodeEncodeError.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _strip_surrogates(text: str) -> str:
    """Remove surrogate codepoints (U+D800..U+DFFF) from `text`."""
    if not text:
        return text
    return _SURROGATE_RE.sub("", text)


class TraceLogger:
    def __init__(self, sessions_dir: Path, session_id: str):
        self.sessions_dir = Path(sessions_dir)
        self.session_id = session_id
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.sessions_dir / f"{session_id}.trace.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def event(self, kind: str, **fields) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind,
                  **{k: _strip_surrogates(v) if isinstance(v, str) else v
                     for k, v in fields.items()}}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._print(kind, fields)

    def _print(self, kind: str, fields: dict) -> None:
        color = _KIND_COLOR.get(kind, "")
        head = f"{color}[{kind}]{_RESET}"
        # Strip surrogates from every string value before formatting: a stray
        # codepoint anywhere (text=, message=, args=, etc.) would otherwise
        # crash sys.stderr.write() with UnicodeEncodeError.
        body = " ".join(
            f"{key}={json.dumps(_strip_surrogates(value) if isinstance(value, str) else value, ensure_ascii=False)}"
            for key, value in fields.items()
        )
        sys.stderr.write(f"{head} {body}\n")
        sys.stderr.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass