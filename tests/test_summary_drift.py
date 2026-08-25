"""W3b tests — summary drift fix.

When the LLM calls `update_plan`, any pre-existing session.summary
describes the OLD plan. Without invalidation, the summary + plan_cache
could disagree for many turns (until natural compression).

The fix: detect update_plan success in the tool loop and clear
session.summary, so the next ContextBuilder.build() regenerates from
current history (which now includes the update_plan tool call).
"""
import pytest

from starry_code.runtime import run_turn, build_default_registry
from starry_code.config import Settings
from starry_code.llm import MockLLMClient
from starry_code.session import Session
from starry_code.trace import TraceLogger
from starry_code.memory.embeddings import MockEmbedder
from starry_code.memory.short_term import InMemoryShortTermStore
from starry_code.memory.vector_store import LocalVectorStore
from starry_code.memory.manager import MemoryManager


def _setup(tmp_path):
    s = Session(id="drift")
    settings = Settings(sessions_dir=tmp_path)
    trace = TraceLogger(tmp_path, s.id)
    reg = build_default_registry()
    memory = MemoryManager(
        embedder=MockEmbedder(),
        short_term=InMemoryShortTermStore(),
        vector_store=LocalVectorStore(embedder=MockEmbedder(), path=None),
    )
    return s, settings, trace, reg, memory


def _update_plan_responses(stage_value):
    """Standard 3-call sequence for an update_plan turn."""
    return [
        {"choices": [{"message": {"role": "assistant", "content": "ok",
                                   "tool_calls": [{
                                       "id": "u1", "type": "function",
                                       "function": {"name": "update_plan",
                                                    "arguments": '{"stage":"' + stage_value + '"}'}
                                   }]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        {"choices": [{"message": {"role": "assistant", "content": "[]"}}]},
    ]


def test_update_plan_clears_existing_summary(tmp_path):
    """If session.summary was already set, update_plan should null it."""
    s, settings, trace, reg, mem = _setup(tmp_path)
    s.summary = "User is learning Python (old plan summary)"

    llm = MockLLMClient(chat_responses=_update_plan_responses("阶段1：Go 语法"))

    run_turn(s, "改学 Go",
             settings=settings, llm=llm, registry=reg,
             memory=mem, trace=trace)

    # summary should be cleared because update_plan succeeded
    assert s.summary == ""
    # plan_cache should have the new plan
    assert s.plan_cache["stage"] == "阶段1：Go 语法"
    assert s.plan_cache["version"] >= 1
    trace.close()


def test_no_update_plan_keeps_summary(tmp_path):
    """Other tools (calculator, etc.) must not invalidate the summary."""
    s, settings, trace, reg, mem = _setup(tmp_path)
    s.summary = "User is on track with current plan"

    llm = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "ok",
                                   "tool_calls": [{
                                       "id": "r1", "type": "function",
                                       "function": {"name": "calculator",
                                                    "arguments": '{"expression":"2+2"}'}
                                   }]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "4"}}]},
        {"choices": [{"message": {"role": "assistant", "content": "[]"}}]},
    ])

    run_turn(s, "2+2",
             settings=settings, llm=llm, registry=reg,
             memory=mem, trace=trace)

    # calculator doesn't invalidate
    assert s.summary == "User is on track with current plan"
    trace.close()


def test_failed_update_plan_keeps_summary(tmp_path):
    """If update_plan rejects (e.g. injection), summary stays."""
    s, settings, trace, reg, mem = _setup(tmp_path)
    s.summary = "Plan looks good"

    llm = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "ok",
                                   "tool_calls": [{
                                       "id": "u1", "type": "function",
                                       "function": {"name": "update_plan",
                                                    "arguments": '{"stage":"<script>bad</script>"}'}
                                   }]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "rejected"}}]},
        {"choices": [{"message": {"role": "assistant", "content": "[]"}}]},
    ])

    run_turn(s, "bad plan",
             settings=settings, llm=llm, registry=reg,
             memory=mem, trace=trace)

    # Rejected update_plan → plan unchanged → keep summary
    assert s.summary == "Plan looks good"
    assert s.plan_cache["version"] == 0
    trace.close()


def test_summary_invalidation_traced(tmp_path):
    """The invalidation is observable via a trace event for debugging."""
    s, settings, trace, reg, mem = _setup(tmp_path)
    s.summary = "stale"

    llm = MockLLMClient(chat_responses=_update_plan_responses("v2"))

    run_turn(s, "update",
             settings=settings, llm=llm, registry=reg,
             memory=mem, trace=trace)

    raw = trace.path.read_text(encoding="utf-8")
    assert "summary_invalidated" in raw
    trace.close()