import json
from pathlib import Path

from agent.naming import generate_chinese_name, rename_session
from agent.session import Session, SessionStore
from agent.trace import TraceLogger


class _ScriptedLLM:
    """Minimal LLM stub returning a fixed content, or raising."""

    def __init__(self, content=None, raises=False):
        self._content = content
        self._raises = raises

    def chat(self, messages, tools=None):
        if self._raises:
            raise RuntimeError("boom")
        return {"choices": [{"message": {"role": "assistant", "content": self._content}}]}


def test_generate_name_parses_cjk():
    llm = _ScriptedLLM("天气查询")
    assert generate_chinese_name(llm, "北京今天天气怎么样") == "天气查询"


def test_generate_name_strips_quotes_and_punct():
    llm = _ScriptedLLM("「周报撰写」。")
    assert generate_chinese_name(llm, "帮我写周报") == "周报撰写"


def test_generate_name_takes_first_line():
    llm = _ScriptedLLM("旅游规划\n（这是我给出的名字）")
    assert generate_chinese_name(llm, "计划去日本旅游") == "旅游规划"


def test_generate_name_ascii_slug_ok():
    llm = _ScriptedLLM("weather-check")
    assert generate_chinese_name(llm, "weather in tokyo") == "weather-check"


def test_generate_name_empty_returns_none():
    assert generate_chinese_name(_ScriptedLLM(""), "hi") is None


def test_generate_name_too_long_returns_none():
    llm = _ScriptedLLM("这是一个非常非常非常长的会话名字超过了十六个字符限制")
    assert generate_chinese_name(llm, "hi") is None


def test_generate_name_invalid_chars_returns_none():
    llm = _ScriptedLLM("hello world!")  # space + punctuation inside
    assert generate_chinese_name(llm, "hi") is None


def test_generate_name_llm_raises_returns_none():
    assert generate_chinese_name(_ScriptedLLM(raises=True), "hi") is None


def test_generate_name_blank_input_returns_none():
    assert generate_chinese_name(_ScriptedLLM("天气"), "   ") is None


def _make_session(tmp_path: Path, sid: str) -> tuple[Session, SessionStore, TraceLogger]:
    store = SessionStore(tmp_path)
    session = Session(id=sid)
    session.add_user("hello")
    store.save(session)
    trace = TraceLogger(tmp_path, sid)
    trace.event("user", text="hello")
    return session, store, trace


def test_rename_moves_both_files_and_updates_id(tmp_path: Path):
    session, store, trace = _make_session(tmp_path, "auto-x")
    ok = rename_session(session, "天气查询", trace, tmp_path)
    assert ok is True
    assert session.id == "天气查询"
    assert (tmp_path / "天气查询.json").exists()
    assert (tmp_path / "天气查询.trace.jsonl").exists()
    assert not (tmp_path / "auto-x.json").exists()
    assert not (tmp_path / "auto-x.trace.jsonl").exists()
    # stored id updated on disk
    data = json.loads((tmp_path / "天气查询.json").read_text(encoding="utf-8"))
    assert data["id"] == "天气查询"


def test_rename_reopens_trace_and_appends(tmp_path: Path):
    session, store, trace = _make_session(tmp_path, "auto-y")
    rename_session(session, "旅游规划", trace, tmp_path)
    trace.event("assistant", text="done")
    trace.close()
    lines = (tmp_path / "旅游规划.trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    # original user event + the appended assistant event
    assert len(lines) == 2
    assert json.loads(lines[-1])["kind"] == "assistant"


def test_rename_collision_appends_suffix(tmp_path: Path):
    # Pre-create a session that would collide.
    SessionStore(tmp_path).save(Session(id="周报撰写"))
    session, store, trace = _make_session(tmp_path, "auto-z")
    ok = rename_session(session, "周报撰写", trace, tmp_path)
    assert ok is True
    assert session.id == "周报撰写-2"
    assert (tmp_path / "周报撰写-2.json").exists()
    # existing session untouched
    assert (tmp_path / "周报撰写.json").exists()


def test_rename_same_id_noop(tmp_path: Path):
    session, store, trace = _make_session(tmp_path, "keep")
    ok = rename_session(session, "keep", trace, tmp_path)
    assert ok is True
    assert session.id == "keep"
    trace.close()


# ---------------------------------------------------------------------------
# AutoNamer
# ---------------------------------------------------------------------------


def test_autonamer_pending_flips_after_fire(tmp_path: Path, capsys):
    from agent.naming import AutoNamer
    session, _store, trace = _make_session(tmp_path, "auto-pending")
    namer = AutoNamer()
    assert namer.pending() is True
    namer.try_name(_ScriptedLLM("天气查询"), "北京今天天气怎么样",
                   session, trace, tmp_path)
    capsys.readouterr()  # swallow the success print
    assert namer.pending() is False
    # Second call is a no-op (does not raise).
    namer.try_name(_ScriptedLLM(raises=True), "msg", session, trace, tmp_path)
    out = capsys.readouterr().out
    assert out == ""  # no extra print
    trace.close()


def test_autonamer_raises_prints_fallback(tmp_path: Path, capsys):
    from agent.naming import AutoNamer
    session, _store, trace = _make_session(tmp_path, "auto-raises")
    old_id = session.id
    namer = AutoNamer()
    namer.try_name(_ScriptedLLM(raises=True), "msg",
                   session, trace, tmp_path)
    out = capsys.readouterr().out
    assert "[could not auto-name" in out
    assert old_id in out
    assert session.id == old_id  # unchanged
    trace.close()


def test_autonamer_success_renames_and_prints_id(tmp_path: Path, capsys):
    from agent.naming import AutoNamer
    session, _store, trace = _make_session(tmp_path, "auto-success")
    namer = AutoNamer()
    namer.try_name(_ScriptedLLM("天气查询"), "北京今天天气怎么样",
                   session, trace, tmp_path)
    out = capsys.readouterr().out
    assert "[session auto-named to: 天气查询" in out
    assert session.id == "天气查询"
    assert (tmp_path / "天气查询.json").exists()
    trace.close()
