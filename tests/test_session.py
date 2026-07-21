import json
from pathlib import Path
from agent.session import Session, SessionStore

def test_session_roundtrip(tmp_path: Path):
    s = Session(id="abc", system_prompt="you are an agent")
    s.add_user("hi")
    s.add_assistant("hello")
    s.add_tool_call(call_id="c1", name="calculator", args={"expression": "1+1"})
    s.add_tool_result(call_id="c1", name="calculator", content="2")
    assert s.messages[-1]["role"] == "tool"
    store = SessionStore(tmp_path)
    store.save(s)
    on_disk = json.loads((tmp_path / "abc.json").read_text(encoding="utf-8"))
    assert on_disk["id"] == "abc"
    s2 = store.load("abc")
    assert s2.messages == s.messages
    assert s2.todos == []

def test_load_missing_returns_fresh(tmp_path: Path):
    store = SessionStore(tmp_path)
    s = store.load("nope")
    assert s.id == "nope" and s.messages == []