import os, subprocess, sys
from pathlib import Path

import pytest


def test_cli_help_runs(tmp_path: Path):
    env = os.environ.copy()
    env["LLM_API_KEY"] = ""
    env["SESSIONS_DIR"] = str(tmp_path)
    r = subprocess.run([sys.executable, "cli.py", "--help"], capture_output=True, text=True, env=env)
    assert r.returncode == 0
    assert "session" in r.stdout.lower()


# ---------------------------------------------------------------------------
# _cleanup_empty_auto_session unit tests
# ---------------------------------------------------------------------------

from starry_code.session import Session
from starry_code.trace import TraceLogger
from cli import _cleanup_empty_auto_session


def _make_trace(tmp_path: Path, sid: str, content: str = "") -> TraceLogger:
    """Create a trace file with optional content; return the TraceLogger."""
    p = tmp_path / f"{sid}.trace.jsonl"
    if content:
        p.write_text(content, encoding="utf-8")
    else:
        p.touch()
    t = TraceLogger(tmp_path, sid)
    return t  # path now points to the existing file


def test_cleanup_deletes_empty_auto_trace(tmp_path: Path):
    sid = "auto-20260722-091207-00b7"
    trace = _make_trace(tmp_path, sid, content="")
    session = Session(id=sid)
    assert (tmp_path / f"{sid}.trace.jsonl").exists()

    _cleanup_empty_auto_session(trace, session, tmp_path)

    assert not (tmp_path / f"{sid}.trace.jsonl").exists(), \
        "empty auto trace should be removed"


def test_cleanup_keeps_trace_when_session_json_exists(tmp_path: Path):
    """If the user actually typed something, .json exists, so we keep the trace."""
    sid = "auto-20260722-091207-00b7"
    (tmp_path / f"{sid}.json").write_text('{"id": "' + sid + '", "messages": []}', encoding="utf-8")
    trace = _make_trace(tmp_path, sid, content="")
    session = Session(id=sid)

    _cleanup_empty_auto_session(trace, session, tmp_path)

    assert (tmp_path / f"{sid}.trace.jsonl").exists(), \
        "trace must be kept when .json was written"
    assert (tmp_path / f"{sid}.json").exists()


def test_cleanup_keeps_nonempty_trace(tmp_path: Path):
    """Even for auto-id with no .json, keep trace if it has events (debug info)."""
    sid = "auto-20260722-091207-00b7"
    trace = _make_trace(
        tmp_path, sid,
        content='{"kind":"user","text":"hi"}\n{"kind":"error","message":"x"}\n',
    )
    session = Session(id=sid)

    _cleanup_empty_auto_session(trace, session, tmp_path)

    assert (tmp_path / f"{sid}.trace.jsonl").exists(), \
        "non-empty trace must be preserved (likely error context)"


def test_cleanup_skips_manually_named_session(tmp_path: Path):
    """User passed --session foo, so we never touch foo.trace.jsonl."""
    sid = "weather"
    trace = _make_trace(tmp_path, sid, content="")
    session = Session(id=sid)

    _cleanup_empty_auto_session(trace, session, tmp_path)

    assert (tmp_path / f"{sid}.trace.jsonl").exists(), \
        "manually-named session trace must never be cleaned"


def test_cleanup_handles_missing_trace_file(tmp_path: Path):
    """If trace.path doesn't exist (edge case), cleanup is a no-op (no raise)."""
    sid = "auto-20260722-091207-00b7"
    # Don't create the file; just point the logger at a non-existent path.
    trace = TraceLogger(tmp_path, sid)
    trace.path = tmp_path / "does-not-exist.trace.jsonl"
    session = Session(id=sid)

    # Should not raise.
    _cleanup_empty_auto_session(trace, session, tmp_path)
