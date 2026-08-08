"""REPL startup rendering: history display + display isolation.

User-reported bugs:
1. -c / --session <id> loads messages but REPL shows nothing — looks like
   a fresh session. Want history to be visible.
2. Starting a new session in the same terminal leaves the previous
   session's output on screen. Want Claude Code-style display isolation.

Both fixes live in cli.py's main() under the REPL branch. To make them
testable, this file covers two helper functions:

  render_repl_startup(session, stream)  -> emits ANSI clear + header + history
  print_session_history(session, stream) -> emits the history block

Tests use pytest's capsys / capfd to capture stdout. ANSI escapes in the
captured output are intentional and asserted on directly.
"""
import io
import pytest

from starry_code.session import Session


# ---- helpers under test (imported from cli.py) ----
from cli import render_repl_startup, print_session_history


def test_render_startup_clears_screen(capsys):
    """Every REPL start must clear the screen so previous content doesn't
    leak across sessions (display isolation, like Claude Code)."""
    s = Session(id="new")  # no messages
    render_repl_startup(s, stream=sys_stdout())
    out = capsys.readouterr().out
    # ANSI: \033[H = home cursor, \033[2J = clear screen, \033[3J = clear scrollback
    assert "\033[H" in out, "expected cursor-home escape"
    assert "\033[2J" in out, "expected clear-screen escape"
    assert "\033[3J" in out, "expected clear-scrollback escape"


def test_render_startup_shows_session_header(capsys):
    """Header must show the session id so the user knows which session
    they're in after a fresh screen."""
    s = Session(id="Java学习指南")
    render_repl_startup(s, stream=sys_stdout())
    out = capsys.readouterr().out
    assert "Java学习指南" in out


def test_render_startup_auto_id_falls_back_to_brand(capsys):
    """Regression: fresh REPL sessions get an auto-xxx id (e.g.
    'auto-20260808-091228-b2ff'). Showing that raw slug in the header is
    ugly. Must fall back to the brand 'Starry Code' — matching what
    _set_terminal_title() does for the window title bar."""
    s = Session(id="auto-20260808-091228-b2ff")
    render_repl_startup(s, stream=sys_stdout())
    out = capsys.readouterr().out
    assert "Starry Code" in out, "auto-id should show brand fallback"
    assert "auto-20260808" not in out, "raw auto slug must not leak"


def test_render_startup_shows_loaded_history(capsys):
    """Regression: when REPL starts with a non-empty session (e.g. via
    -c), the previous conversation must be visible to the user."""
    s = Session(id="resumed")
    s.add_user("怎么学Java")
    s.add_assistant("先学基础语法...")
    s.add_user("还要学什么？")
    s.add_assistant("MySQL / Redis ...")
    render_repl_startup(s, stream=sys_stdout())
    out = capsys.readouterr().out
    assert "怎么学Java" in out, "user turn 1 should be visible"
    assert "先学基础语法" in out, "assistant turn 1 should be visible"
    assert "还要学什么" in out, "user turn 2 should be visible"
    assert "MySQL" in out, "assistant turn 2 should be visible"


def test_render_startup_skips_history_for_empty_session(capsys):
    """New session with no messages should NOT print a history block."""
    s = Session(id="fresh")
    render_repl_startup(s, stream=sys_stdout())
    out = capsys.readouterr().out
    # Header still shown
    assert "fresh" in out
    # But no history markers (we use ── separators)
    assert "────" not in out, "empty session should not render history dividers"


def test_render_startup_omits_tool_messages(capsys):
    """Tool messages are noisy — keep the history block readable."""
    s = Session(id="t")
    s.add_user("compute 1+1")
    s.add_tool_call(call_id="c1", name="calculator", args={"expression": "1+1"})
    s.add_tool_result(call_id="c1", name="calculator", content="2")
    s.add_assistant("the answer is 2")
    render_repl_startup(s, stream=sys_stdout())
    out = capsys.readouterr().out
    assert "compute 1+1" in out
    assert "the answer is 2" in out
    # Tool internals should not leak into the user-facing history
    assert "tool_call_id" not in out


def test_render_startup_truncates_very_long_assistant(capsys):
    """An assistant message over 600 chars must be truncated for display
    (history block must fit on one screen)."""
    s = Session(id="long")
    s.add_user("explain rust")
    s.add_assistant("X" * 5000)
    render_repl_startup(s, stream=sys_stdout())
    out = capsys.readouterr().out
    # Truncation marker present
    assert "..." in out or "…" in out
    # The full 5000-char blob should NOT be in the output
    assert "X" * 1000 not in out


def test_print_session_history_compact(capsys):
    """print_session_history alone (without clear/header) just emits the
    history block — used by tests and any future /history command."""
    s = Session(id="h")
    s.add_user("hi")
    s.add_assistant("hello there friend")
    print_session_history(s, stream=sys_stdout())
    out = capsys.readouterr().out
    assert "hi" in out
    assert "hello there friend" in out


# ---- small helper to pass an explicit stdout stream ----

def sys_stdout():
    """Return the real sys.stdout (capys already captures it)."""
    import sys
    return sys.stdout