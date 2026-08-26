"""Phase 2 tests: plan_cache + update_plan + read_artifact + trigger logic.

Covers:
1. Session.plan_cache default shape and persistence round-trip
2. UpdatePlanTool behavior (no-op when nothing changed, version bump
   when at least one field changes)
3. ReadArtifactTool security (rejects paths outside artifacts/)
4. _should_adjust_plan trigger classification (keyword / stale / milestone)
5. _inject_trend_for_adjustment calls tech_trend and writes short-term memory
6. build_default_registry() exposes both new tools
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from starry_code.session import Session, SessionStore
from starry_code.tools.update_plan import UpdatePlanTool
from starry_code.tools.read_artifact import ReadArtifactTool
from starry_code.tools.tech_trend import TechTrendTool
from starry_code.tools.registry import ToolRegistry
from starry_code.runtime import (
    _should_adjust_plan,
    _inject_trend_for_adjustment,
    build_default_registry,
)
from starry_code.memory.embeddings import MockEmbedder
from starry_code.memory.short_term import InMemoryShortTermStore
from starry_code.memory.vector_store import LocalVectorStore
from starry_code.memory.manager import MemoryManager
from starry_code.trace import TraceLogger
from starry_code.config import Settings


# ---- Session.plan_cache ----

def test_session_default_plan_cache():
    """Every fresh Session has the documented plan_cache shape."""
    s = Session(id="x")
    assert s.plan_cache == {
        "version": 0, "stage": "", "next_task": "",
        "long_term_goal": "", "last_updated": "",
    }
    # default_factory returns fresh dict per instance (no shared state)
    s2 = Session(id="y")
    s.plan_cache["stage"] = "changed"
    assert s2.plan_cache["stage"] == ""


def test_session_plan_cache_roundtrip(tmp_path: Path):
    """plan_cache survives save/load."""
    s = Session(id="plan-test")
    s.plan_cache = {
        "version": 3,
        "stage": "阶段1：变量与流程",
        "next_task": "写完 5 个 if/else 小练习",
        "long_term_goal": "3 个月内转 Go 后端",
        "last_updated": "2026-08-09T10:00:00+00:00",
    }
    store = SessionStore(tmp_path)
    store.save(s)
    s2 = store.load("plan-test")
    assert s2.plan_cache == s.plan_cache


def test_load_legacy_session_without_plan_cache(tmp_path: Path):
    """Sessions saved before plan_cache existed get the default cache."""
    legacy = {
        "id": "legacy",
        "system_prompt": "test prompt",
        "messages": [],
        "todos": [],
        "summary": "",
    }
    (tmp_path / "legacy.json").write_text(
        json.dumps(legacy, ensure_ascii=False), encoding="utf-8"
    )
    s = SessionStore(tmp_path).load("legacy")
    assert s.plan_cache["version"] == 0
    assert s.plan_cache["stage"] == ""


# ---- UpdatePlanTool ----

def test_update_plan_noop_when_nothing_changes():
    """Empty args or same values -> no version bump."""
    s = Session(id="p")
    before = s.plan_cache["version"]
    r = UpdatePlanTool().execute({}, session=s)
    assert r.ok
    assert s.plan_cache["version"] == before
    assert "no-op" in r.content.lower()


def test_update_plan_bumps_version_on_real_change():
    s = Session(id="p")
    UpdatePlanTool().execute(
        {"stage": "阶段1", "next_task": "学完变量"},
        session=s,
    )
    assert s.plan_cache["version"] == 1
    assert s.plan_cache["stage"] == "阶段1"
    assert s.plan_cache["next_task"] == "学完变量"
    assert s.plan_cache["last_updated"]  # ISO stamp set


def test_update_plan_only_changed_fields_bump():
    """If only next_task changes, only that field moves; other fields
    preserved (NOT cleared by empty strings)."""
    s = Session(id="p")
    s.plan_cache["stage"] = "阶段1"
    s.plan_cache["next_task"] = "学变量"
    s.plan_cache["long_term_goal"] = "转 Go"
    s.plan_cache["version"] = 5
    r = UpdatePlanTool().execute(
        {"stage": "阶段1", "next_task": "学流程控制", "long_term_goal": "转 Go"},
        session=s,
    )
    assert r.ok
    # only next_task actually changed (stage/long_term_goal identical)
    assert s.plan_cache["next_task"] == "学流程控制"
    assert s.plan_cache["stage"] == "阶段1"
    assert s.plan_cache["long_term_goal"] == "转 Go"
    assert s.plan_cache["version"] == 6


def test_update_plan_rejects_session_none():
    r = UpdatePlanTool().execute({"stage": "x"}, session=None)
    assert not r.ok
    assert "session" in r.content.lower()


# ---- ReadArtifactTool security ----

def test_read_artifact_blocks_path_traversal(tmp_path: Path):
    s = Session(id="safe")
    tool = ReadArtifactTool()
    # 试图读 sessions/../etc/passwd -> 必须被拒
    r = tool.execute({"path": str(tmp_path.parent / "evil.txt")}, session=s)
    assert not r.ok
    assert "not under" in r.content.lower() or "not found" in r.content.lower()


def test_read_artifact_blocks_sibling_session(tmp_path: Path):
    """Cannot read another session's artifacts via absolute path."""
    # Setup: session A's artifacts under tmp_path
    s_a = Session(id="A")
    artifacts = tmp_path / "A" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "c1.json").write_text(
        json.dumps({"tool": "x", "content": "secret data"}),
        encoding="utf-8",
    )
    # session B tries to read A's artifact
    s_b = Session(id="B")
    tool = ReadArtifactTool(sessions_dir=tmp_path)
    r = tool.execute(
        {"path": str(artifacts / "c1.json")},
        session=s_b,
    )
    assert not r.ok


def test_read_artifact_returns_full_content(tmp_path: Path):
    """Happy path: read an artifact that belongs to current session."""
    s = Session(id="happy")
    artifacts = tmp_path / "happy" / "artifacts"
    artifacts.mkdir(parents=True)
    content = "this is a long tool output " * 50  # ~1100 chars
    (artifacts / "c1.json").write_text(
        json.dumps({"tool": "search", "content": content}),
        encoding="utf-8",
    )
    tool = ReadArtifactTool(sessions_dir=tmp_path)
    r = tool.execute({"path": str(artifacts / "c1.json")}, session=s)
    assert r.ok
    assert "this is a long tool output" in r.content


def test_read_artifact_truncates(tmp_path: Path):
    """max_chars cuts off long content."""
    s = Session(id="trunc")
    artifacts = tmp_path / "trunc" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "c1.json").write_text(
        json.dumps({"tool": "x", "content": "x" * 5000}),
        encoding="utf-8",
    )
    tool = ReadArtifactTool(sessions_dir=tmp_path)
    r = tool.execute({"path": str(artifacts / "c1.json"), "max_chars": 100}, session=s)
    assert r.ok
    assert "truncated" in r.content
    assert len(r.content) < 200  # 100 chars + truncation marker


# ---- Trigger logic ----

def test_trigger_keyword_user():
    """User keywords force a recheck."""
    s = Session(id="t")
    for kw in ["现在行情怎么样", "Rust 还值得学吗", "我要重新规划",
               "对一下行情", "该不该换方向"]:
        should, reason = _should_adjust_plan(s, kw)
        assert should and reason == "user_keyword", f"keyword {kw!r} not triggered"


def test_trigger_no_match_normal_question():
    s = Session(id="t")
    should, reason = _should_adjust_plan(s, "今天学 Go 变量")
    assert not should
    assert reason == ""


def test_trigger_stale_7d():
    """Plan older than 7 days triggers stale_7d."""
    s = Session(id="t")
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    s.plan_cache["last_updated"] = old
    should, reason = _should_adjust_plan(s, "今天学点什么")
    assert should and reason == "stale_7d"


def test_trigger_milestone():
    """≥3 todos done but plan still v0/v1 -> milestone trigger."""
    s = Session(id="t")
    s.todos = [{"id": i, "done": True, "text": "x"} for i in range(3)]
    s.plan_cache["version"] = 1
    should, reason = _should_adjust_plan(s, "继续")
    assert should and reason.startswith("milestone_3_done_v1")


def test_trigger_milestone_no_fire_when_plan_updated():
    """If plan is at v2+, milestone trigger is silent."""
    s = Session(id="t")
    s.todos = [{"id": i, "done": True, "text": "x"} for i in range(3)]
    s.plan_cache["version"] = 5
    should, _ = _should_adjust_plan(s, "继续")
    assert not should


# ---- Inject trend ----

def test_inject_trend_calls_tech_trend_and_pushes_memory(tmp_path: Path):
    """When triggered, _inject_trend_for_adjustment calls tech_trend
    with topics extracted from long_term_goal + user_input, and pushes
    the result into short-term memory."""
    s = Session(id="inj")
    s.plan_cache["long_term_goal"] = "想学 Rust 后端"

    registry = ToolRegistry()
    registry.register(TechTrendTool())  # uses default _MockTrendProvider

    settings = Settings(sessions_dir=tmp_path)
    memory = MemoryManager(
        embedder=MockEmbedder(),
        short_term=InMemoryShortTermStore(),
        vector_store=LocalVectorStore(embedder=MockEmbedder(), path=None),
    )
    trace = TraceLogger(tmp_path, s.id)

    _inject_trend_for_adjustment(registry, s, memory, "Rust 行情如何", trace)

    # Memory should have tech_trend_recheck entries. Topic is stored with
    # original case from the regex match; lower() is only used for the
    # tech_trend query itself.
    short = memory.recent_turns(s.id, k=10)
    tool_results = [t for t in short if t.get("name") == "tech_trend_recheck"]
    assert any(t.get("topic", "").lower() == "rust" for t in tool_results)
    trace.close()


# ---- Registry ----

def test_build_default_registry_includes_phase2_tools():
    reg = build_default_registry()
    names = reg.names()
    assert "update_plan" in names
    assert "read_artifact" in names
    assert "tech_trend" in names  # existing
    # Both new tools must have openai schemas
    schemas = reg.openai_schemas()
    plan_schema = next(s for s in schemas if s["function"]["name"] == "update_plan")
    art_schema = next(s for s in schemas if s["function"]["name"] == "read_artifact")
    assert "stage" in plan_schema["function"]["parameters"]["properties"]
    assert "path" in art_schema["function"]["parameters"]["required"]


# ---- Bug G regression: read_artifact must use the SAME sessions_dir
# that ContextBuilder writes to ----

# ---- Bug I regression: offload must keep role="tool" so the
# assistant-tool_call ↔ tool_result pairing survives ----

def test_offload_keeps_tool_role(tmp_path):
    """Bug I: previously offload replaced role="tool" with role="system",
    breaking the assistant-tool_call ↔ tool_result pairing. The LLM API
    rejects with 400 "tool call result does not follow tool call".
    Fix: keep role="tool" and only replace content.
    """
    from starry_code.context import ContextBuilder
    from starry_code.session import Session
    from starry_code.config import Settings
    from starry_code.memory.embeddings import MockEmbedder
    from starry_code.memory.short_term import InMemoryShortTermStore
    from starry_code.memory.vector_store import LocalVectorStore
    from starry_code.memory.manager import MemoryManager

    s = Session(id="bug-i")
    s.add_user("compute")
    s.add_tool_call(call_id="abc", name="calculator", args={"expression": "2+2"})
    s.add_tool_result(call_id="abc", name="calculator", content="x" * 1000)
    s.add_assistant("ok")

    memory = MemoryManager(
        embedder=MockEmbedder(),
        short_term=InMemoryShortTermStore(),
        vector_store=LocalVectorStore(embedder=MockEmbedder(), path=None),
    )
    builder = ContextBuilder(memory=memory, settings=Settings(sessions_dir=tmp_path))
    built = builder.build(s, "next")
    msgs = built.messages

    # Find the offloaded tool message (originally >500 chars content)
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert tool_msgs, "no tool messages after build (offload may have replaced)"

    # Find the offloaded one — content should contain "[artifact saved]"
    offloaded = [m for m in tool_msgs if "[artifact saved]" in (m.get("content") or "")]
    assert offloaded, "offloaded tool message not found"

    # The offloaded message MUST keep role="tool" + tool_call_id unchanged
    m = offloaded[0]
    assert m["role"] == "tool", f"Bug I regression: role changed to {m['role']!r}"
    assert m["tool_call_id"] == "abc", (
        f"tool_call_id changed: {m['tool_call_id']!r}"
    )


def test_read_artifact_uses_custom_sessions_dir(tmp_path):
    """When build_default_registry gets a custom sessions_dir, both
    the offload path (ContextBuilder) and the read path (ReadArtifactTool)
    must agree. Otherwise the coach calls read_artifact on an artifact
    that exists but is in a different directory — silently fails.
    """
    from starry_code.context import ContextBuilder
    from starry_code.session import Session
    from starry_code.memory.embeddings import MockEmbedder
    from starry_code.memory.short_term import InMemoryShortTermStore
    from starry_code.memory.vector_store import LocalVectorStore
    from starry_code.memory.manager import MemoryManager

    sessions_dir = tmp_path / "custom-sessions"
    reg = build_default_registry(sessions_dir=sessions_dir)

    # Setup session + history with a tool message > 500 chars (triggers offload)
    s = Session(id="bug-g")
    s.add_user("compute")
    s.add_tool_call(call_id="abc123", name="calculator", args={"expression": "2+2"})
    s.add_tool_result(call_id="abc123", name="calculator", content="x" * 1000)
    s.add_assistant("answer")

    # ContextBuilder writes artifacts to {sessions_dir}/bug-g/artifacts/
    memory = MemoryManager(
        embedder=MockEmbedder(),
        short_term=InMemoryShortTermStore(),
        vector_store=LocalVectorStore(embedder=MockEmbedder(), path=None),
    )
    builder = ContextBuilder(memory=memory, settings=Settings(sessions_dir=sessions_dir))
    built = builder.build(s, "next question")
    msgs = built.messages
    # Find the [artifact saved] card
    artifact_card = next(m for m in msgs if "[artifact saved]" in (m.get("content") or ""))
    card_text = artifact_card["content"]
    # Extract the path
    import re
    path_match = re.search(r"path: (.+\.json)", card_text)
    assert path_match, f"could not find path in card: {card_text[:200]}"
    actual_path = path_match.group(1)

    # Now read_artifact on that path. With the Bug G fix it should succeed.
    read_tool = reg.get("read_artifact")
    r = read_tool.execute({"path": actual_path}, session=s)
    assert r.ok, f"read_artifact rejected its own artifact: {r.content}"