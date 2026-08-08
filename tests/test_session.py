import json
from pathlib import Path
from starry_code.session import Session, SessionStore, _OLD_SYSTEM_PROMPT
from starry_code.prompts import SYSTEM_PROMPT

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


# ---- Phase 1 bug fix: system_prompt migration ----

def test_new_session_picks_up_live_system_prompt():
    """Regression: Phase 1 changed prompts.SYSTEM_PROMPT but the dataclass
    default was still the Phase 0 string, so new sessions kept the old
    generic prompt. Fresh Session() must now mirror prompts.SYSTEM_PROMPT."""
    s = Session(id="x")
    assert s.system_prompt == SYSTEM_PROMPT
    # Sanity: the live prompt must NOT be the dead Phase 0 default
    assert s.system_prompt != _OLD_SYSTEM_PROMPT


def test_store_load_migrates_old_default_prompt(tmp_path: Path):
    """A session persisted with the old generic default gets auto-upgraded."""
    legacy = {
        "id": "legacy",
        "system_prompt": _OLD_SYSTEM_PROMPT,
        "messages": [],
        "todos": [],
        "summary": "",
    }
    (tmp_path / "legacy.json").write_text(
        json.dumps(legacy, ensure_ascii=False), encoding="utf-8"
    )
    s = SessionStore(tmp_path).load("legacy")
    assert s.system_prompt == SYSTEM_PROMPT
    assert s.system_prompt != _OLD_SYSTEM_PROMPT


def test_store_load_preserves_custom_user_prompt(tmp_path: Path):
    """User-set custom prompts must NOT be touched by migration."""
    custom = "你是一个专门写 Python 单元测试的助手。"
    payload = {
        "id": "custom",
        "system_prompt": custom,
        "messages": [],
        "todos": [],
        "summary": "",
    }
    (tmp_path / "custom.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    s = SessionStore(tmp_path).load("custom")
    assert s.system_prompt == custom


def test_store_load_migrates_empty_prompt(tmp_path: Path):
    """Empty system_prompt (corrupt or partial JSON) also gets filled."""
    payload = {
        "id": "empty",
        "system_prompt": "",
        "messages": [],
        "todos": [],
        "summary": "",
    }
    (tmp_path / "empty.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    s = SessionStore(tmp_path).load("empty")
    assert s.system_prompt == SYSTEM_PROMPT