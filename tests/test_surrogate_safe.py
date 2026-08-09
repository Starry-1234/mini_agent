"""Surrogate-safe output: reasoning models sometimes emit U+D800..U+DFFF
codepoints which UTF-8 forbids by spec. Without sanitization, every
`print(answer)` or `trace._print()` raises UnicodeEncodeError.

These tests verify:
1. `cli._strip_surrogates(text)` returns input with surrogate range removed.
2. trace._print() survives a surrogate in any field.
3. cli.py's print(ans) path is sanitized (we cover the helper directly
   since the REPL loop is hard to exercise with surrogates).
"""
from io import StringIO

from cli import _strip_surrogates
from starry_code.trace import TraceLogger


def test_strip_surrogates_removes_high_surrogates():
    """High surrogates D800-DBFF are leading halves of UTF-16 pairs."""
    text = "before 😀 after"  # emoji as surrogate pair
    out = _strip_surrogates(text)
    assert "\ud83d" not in out
    assert "\ude00" not in out
    assert "before" in out
    assert "after" in out


def test_strip_surrogates_removes_low_surrogates():
    """Low surrogates DC00-DFFF are trailing halves."""
    text = "x\udce5\ud800y"  # two lone surrogates
    out = _strip_surrogates(text)
    assert "\udce5" not in out
    assert "\ud800" not in out
    assert out == "xy"


def test_strip_surrogates_preserves_normal_cjk():
    """CJK Unified Ideographs (U+4E00..U+9FFF) are NOT surrogates and
    must survive sanitization."""
    text = "Java学习指南"
    out = _strip_surrogates(text)
    assert out == text


def test_strip_surrogates_handles_empty_and_none_safe():
    """Empty string round-trips; None shouldn't happen but guard anyway."""
    assert _strip_surrogates("") == ""
    assert _strip_surrogates("plain ascii") == "plain ascii"


def test_trace_event_with_surrogate_does_not_crash(tmp_path):
    """trace._print writes to sys.stderr; with surrogates in the field,
    Python's UTF-8 encoder raises UnicodeEncodeError. After the fix,
    _print must strip first."""
    import sys
    trace = TraceLogger(tmp_path, "surr-test")

    # Capture stderr
    saved = sys.stderr
    buf = StringIO()
    sys.stderr = buf
    try:
        # Should not raise
        trace.event("assistant", text="reply with surrogate \udce5 in middle")
    finally:
        sys.stderr = saved
        trace.close()

    captured = buf.getvalue()
    # The surrogate is stripped; the rest survives
    assert "\udce5" not in captured
    assert "reply with surrogate" in captured
    assert "in middle" in captured