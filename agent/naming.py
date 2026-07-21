# agent/naming.py
from __future__ import annotations

import re
from pathlib import Path

from .llm import LLMClient
from .prompts import NAMING_PROMPT
from .session import Session
from .trace import TraceLogger

# 2-16 chars of CJK ideographs, ASCII letters, digits, underscore, hyphen.
_NAME_RE = re.compile(r"^[一-鿿豈-﫿A-Za-z0-9_-]{2,16}$")
# Characters we are willing to strip from the edges (quotes, punctuation, ws).
_STRIP_CHARS = " \t\r\n\"'“”「」『』`。，、.,:：;；!！?？()（）[]【】{}"


def generate_chinese_name(llm: LLMClient, first_user_msg: str) -> str | None:
    """Return a 2-6 char Chinese slug (or short ASCII slug), or None on failure.

    Strips surrounding punctuation/whitespace. Returns None if the LLM call
    raises, the response is empty, longer than 16 chars, or contains characters
    outside the allowed set (CJK, ASCII alnum, underscore, hyphen).
    """
    if llm is None or not (first_user_msg or "").strip():
        return None
    prompt = NAMING_PROMPT.format(user_msg=first_user_msg.strip())
    try:
        raw = llm.chat([{"role": "user", "content": prompt}], tools=None)
        text = raw["choices"][0]["message"]["content"] or ""
    except Exception:
        return None
    # Take the first non-empty line, strip surrounding punctuation/whitespace.
    name = ""
    for line in text.splitlines():
        line = line.strip().strip(_STRIP_CHARS).strip()
        if line:
            name = line
            break
    if not name or len(name) > 16:
        return None
    if not _NAME_RE.match(name):
        return None
    return name


def rename_session(
    session: Session,
    new_id: str,
    trace: TraceLogger,
    sessions_dir: Path,
) -> bool:
    """Atomically rename a session's .json and .trace.jsonl files.

    - Resolves collisions by appending -2, -3, ... to `new_id`.
    - Closes the trace file, moves both files, updates `session.id` in memory
      and on disk, then reopens the trace file at the new path.
    - Returns True on success, False if no free name could be found.
    """
    sessions_dir = Path(sessions_dir)
    old_id = session.id
    if new_id == old_id:
        return True

    target = _resolve_free_id(sessions_dir, new_id)
    if target is None:
        return False

    old_json = sessions_dir / f"{old_id}.json"
    new_json = sessions_dir / f"{target}.json"
    old_trace = sessions_dir / f"{old_id}.trace.jsonl"
    new_trace = sessions_dir / f"{target}.trace.jsonl"

    # Close the trace handle before moving the file (Windows lock safety).
    trace.close()

    if old_json.exists():
        old_json.replace(new_json)
    if old_trace.exists():
        old_trace.replace(new_trace)

    # Update in-memory id, then rewrite the json so the stored id matches.
    session.id = target
    if new_json.exists():
        data = new_json.read_text(encoding="utf-8")
        try:
            import json
            obj = json.loads(data)
            obj["id"] = target
            tmp = new_json.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(new_json)
        except Exception:
            pass

    # Reopen the trace logger against the new path.
    trace.session_id = target
    trace.path = new_trace
    trace._fh = new_trace.open("a", encoding="utf-8")
    return True


def _resolve_free_id(sessions_dir: Path, new_id: str) -> str | None:
    """Return `new_id` if free, else new_id-2, -3, ... up to -99; None if none."""
    def taken(sid: str) -> bool:
        return (sessions_dir / f"{sid}.json").exists() or (
            sessions_dir / f"{sid}.trace.jsonl"
        ).exists()

    if not taken(new_id):
        return new_id
    for i in range(2, 100):
        candidate = f"{new_id}-{i}"
        if not taken(candidate):
            return candidate
    return None
