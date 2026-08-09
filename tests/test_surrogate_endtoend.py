"""End-to-end surrogate scenarios — what happens when user input or LLM
output contains lone surrogates.

Three scenarios to cover:
1. User input has surrogate → must NOT crash trace / LLM call / print
2. LLM response has surrogate → must NOT crash print / trace
3. Auto-namer prints session id even if rename produced weird name
"""
import io
import json
from pathlib import Path

import pytest

from cli import _strip_surrogates
from starry_code.session import Session
from starry_code.trace import TraceLogger
from starry_code.runtime import run_turn
from starry_code.config import Settings
from starry_code.llm import MockLLMClient


def _make_session(tmp_path: Path, sid: str = "surr-e2e") -> Session:
    return Session(id=sid)


def test_user_input_with_surrogate_does_not_crash(tmp_path):
    """User input containing a lone surrogate must not crash:
    - trace.event() (file + stderr write)
    - ContextBuilder.build()
    - LLMClient.chat() (mock)
    - print(answer)
    """
    s = _make_session(tmp_path)
    settings = Settings(sessions_dir=tmp_path)
    trace = TraceLogger(tmp_path, s.id)

    # LLM mock that just returns a fixed string
    llm = MockLLMClient(chat_responses=[
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            {"choices": [{"message": {"role": "assistant", "content": "[]"}}]},
    ])

    from starry_code.runtime import build_memory
    from starry_code.tools.registry import ToolRegistry
    memory = build_memory(settings=settings, llm=llm)
    reg = ToolRegistry()

    # The "user input" — surrogate at position 0 (from a Windows DELETE key)
    bad_input = "\udce5" + "你好世界"

    # Capture stderr so trace._print doesn't pollute test output
    import sys
    saved_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        # This should NOT raise
        answer = run_turn(
            s, bad_input,
            settings=settings, llm=llm, registry=reg, memory=memory, trace=trace,
        )
        # Sanitized print (mimics cli.py:278)
        print(_strip_surrogates(answer))
        # Simulated error-print path (cli.py:281)
        e = UnicodeError("simulated surrogate encoding error \udce5")
        print(f"[error] {_strip_surrogates(str(e))}", file=sys.stderr)
    finally:
        sys.stderr = saved_stderr
        trace.close()

    # The user message stored in session should also be safe (or stripped)
    # depending on where we strip. At minimum, the trace file shouldn't have
    # raw surrogates.
    trace_path = tmp_path / f"{s.id}.trace.jsonl"
    raw = trace_path.read_text(encoding="utf-8")
    # No surrogate codepoints should leak into the file
    for cp in ("\ud800", "\udc00", "\udce5"):
        assert cp not in raw, f"trace file leaked surrogate {cp!r}"


def test_llm_response_with_surrogate_does_not_crash(tmp_path):
    """LLM response containing a surrogate at position 69 (the user's
    reported error position) must not crash anything."""
    s = _make_session(tmp_path)
    settings = Settings(sessions_dir=tmp_path)
    trace = TraceLogger(tmp_path, s.id)

    # LLM response with surrogate at position 69
    bad_response = "a" * 69 + "\udce5" + "b" * 30
    llm = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": bad_response}}]},
        {"choices": [{"message": {"role": "assistant", "content": "[]"}}]},
    ])

    from starry_code.runtime import build_memory
    from starry_code.tools.registry import ToolRegistry
    memory = build_memory(settings=settings, llm=llm)
    reg = ToolRegistry()

    import sys
    saved = sys.stderr
    sys.stderr = io.StringIO()
    try:
        answer = run_turn(
            s, "test",
            settings=settings, llm=llm, registry=reg, memory=memory, trace=trace,
        )
        # If the answer has surrogate, _strip_surrogates must clean it
        clean = _strip_surrogates(answer)
        print(clean)
    finally:
        sys.stderr = saved
        trace.close()

    # Verify no surrogates in trace file
    trace_path = tmp_path / f"{s.id}.trace.jsonl"
    raw = trace_path.read_text(encoding="utf-8")
    for cp in ("\ud800", "\udc00", "\udce5"):
        assert cp not in raw, f"trace leaked {cp!r}"


def test_auto_namer_does_not_print_surrogate(tmp_path, capsys):
    """AutoNamer prints session-id-related messages; if the generated name
    contained a surrogate, the print would crash. _sanitize() in
    naming.py already handles this, but verify the print path itself."""
    from starry_code.naming import AutoNamer
    namer = AutoNamer()

    s = Session(id="auto-test")
    trace = TraceLogger(tmp_path, "auto-namer-test")
    # Don't call llm — directly force a print of failure path
    namer._fired = True
    namer.try_name(llm=None, first_user_msg="hi", session=s, trace=trace,
                   sessions_dir=tmp_path)
    # Capture stdout
    out, _ = capsys.readouterr()
    # Output must not contain raw surrogate
    for cp in ("\ud800", "\udc00"):
        assert cp not in out, f"auto-namer printed surrogate {cp!r}"
    trace.close()