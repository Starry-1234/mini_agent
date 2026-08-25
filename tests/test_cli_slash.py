"""W3a tests — CLI slash commands.

Validates:
1. /help lists all commands
2. /plan shows the current plan_cache
3. /forget clears individual fields or all
4. /history lists sessions (best-effort: depends on cwd, so just smoke)
5. /help on bad command shows error message
"""
import pytest
from io import StringIO
from pathlib import Path

from cli import _handle_slash_command
from starry_code.session import Session, SessionStore
from starry_code.memory.embeddings import MockEmbedder
from starry_code.memory.short_term import InMemoryShortTermStore
from starry_code.memory.vector_store import LocalVectorStore
from starry_code.memory.manager import MemoryManager
from starry_code.trace import TraceLogger


def _make(tmp_path):
    s = Session(id="slash-test")
    store = SessionStore(tmp_path)
    s.plan_cache["stage"] = "阶段1：Go 语法"
    s.plan_cache["next_task"] = "写 5 道 if/else 题"
    s.plan_cache["long_term_goal"] = "3 个月转 Go 后端"
    s.plan_cache["version"] = 3
    memory = MemoryManager(
        embedder=MockEmbedder(),
        short_term=InMemoryShortTermStore(),
        vector_store=LocalVectorStore(embedder=MockEmbedder(), path=None),
    )
    trace = TraceLogger(tmp_path, s.id)
    return s, store, memory, trace


def test_help_lists_commands(capsys, tmp_path):
    s, store, mem, trace = _make(tmp_path)
    _handle_slash_command("/help", s, store, mem, trace)
    out = capsys.readouterr().out
    for cmd in ("/history", "/plan", "/forget"):
        assert cmd in out


def test_plan_shows_current_state(capsys, tmp_path):
    s, store, mem, trace = _make(tmp_path)
    _handle_slash_command("/plan", s, store, mem, trace)
    out = capsys.readouterr().out
    assert "阶段1：Go 语法" in out
    assert "写 5 道 if/else 题" in out
    assert "3 个月转 Go 后端" in out
    assert "v3" in out
    trace.close()


def test_forget_clears_specific_field(capsys, tmp_path):
    s, store, mem, trace = _make(tmp_path)
    _handle_slash_command("/forget next_task", s, store, mem, trace)
    out = capsys.readouterr().out
    assert "cleared" in out.lower() or "清" in out
    # next_task is empty
    assert s.plan_cache["next_task"] == ""
    # Other fields preserved
    assert s.plan_cache["stage"] == "阶段1：Go 语法"
    assert s.plan_cache["long_term_goal"] == "3 个月转 Go 后端"
    # Saved to disk
    s2 = store.load("slash-test")
    assert s2.plan_cache["next_task"] == ""
    trace.close()


def test_forget_all_clears_everything(capsys, tmp_path):
    s, store, mem, trace = _make(tmp_path)
    _handle_slash_command("/forget all", s, store, mem, trace)
    assert s.plan_cache["stage"] == ""
    assert s.plan_cache["next_task"] == ""
    assert s.plan_cache["long_term_goal"] == ""
    assert s.plan_cache["version"] == 0
    trace.close()


def test_forget_already_empty_noop(capsys, tmp_path):
    s, store, mem, trace = _make(tmp_path)
    s.plan_cache["next_task"] = ""  # already empty
    _handle_slash_command("/forget next_task", s, store, mem, trace)
    out = capsys.readouterr().out
    # Should not crash; should report nothing-to-clear or similar
    assert s.plan_cache["next_task"] == ""
    trace.close()


def test_forget_unknown_target(capsys, tmp_path):
    s, store, mem, trace = _make(tmp_path)
    _handle_slash_command("/forget bogus_field", s, store, mem, trace)
    out = capsys.readouterr().out
    assert "unknown" in out.lower() or "try" in out.lower()
    trace.close()


def test_forget_without_arg_shows_usage(capsys, tmp_path):
    s, store, mem, trace = _make(tmp_path)
    _handle_slash_command("/forget", s, store, mem, trace)
    out = capsys.readouterr().out
    assert "usage" in out.lower() or "用法" in out or "/forget" in out
    trace.close()


def test_unknown_slash_command(capsys, tmp_path):
    s, store, mem, trace = _make(tmp_path)
    _handle_slash_command("/nonsense", s, store, mem, trace)
    out = capsys.readouterr().out
    assert "unknown" in out.lower() or "/help" in out
    trace.close()


def test_history_lists_sessions(capsys, tmp_path):
    """Create a couple of sessions then list them."""
    s, store, mem, trace = _make(tmp_path)
    # Create two extra sessions
    s2 = Session(id="another-one")
    s2.add_user("hi")
    s2.add_assistant("hello")
    store.save(s2)
    s3 = Session(id="third-one")
    s3.add_user("test")
    store.save(s3)

    # Pass the actual sessions_dir so /history can scan it
    s.plan_cache["stage"] = "test"
    s.plan_cache["version"] = 0

    _handle_slash_command("/history", s, store, mem, trace)
    out = capsys.readouterr().out
    # Should list at least slash-test and the others
    assert "slash-test" in out or "another-one" in out
    trace.close()


def test_history_with_filter(capsys, tmp_path):
    """Filter by query, only matching sessions show."""
    s, store, mem, trace = _make(tmp_path)
    s2 = Session(id="java-learning")
    store.save(s2)
    s3 = Session(id="rust-experiment")
    store.save(s3)

    _handle_slash_command("/history java", s, store, mem, trace)
    out = capsys.readouterr().out
    # "java-learning" should be in output; "rust-experiment" should not
    assert "java-learning" in out
    assert "rust-experiment" not in out
    trace.close()