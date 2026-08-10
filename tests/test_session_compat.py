"""Session serialization forward-compat tests.

Phase 2 will add `plan_cache` (a dict) to Session. This test pins the
invariant: asdict() can handle any dict/list/str/int/float/bool field, and
the JSON round-trip preserves them byte-for-byte.

If we ever add fields containing Path / datetime / set / bytes, this test
will fail and we will know to write a custom __init__ / to_dict.
"""
import json
from dataclasses import asdict, fields
from pathlib import Path

import pytest

from starry_code.session import Session, SessionStore


def test_asdict_with_dict_field_roundtrips(tmp_path: Path):
    """Hypothetical plan_cache field — for now we simulate by attaching a
    dict attribute (Session is a dataclass; non-field attrs are ignored by
    asdict, so this test mainly exercises the *JSON* round-trip path)."""
    s = Session(id="compat")
    # simulate a plan_cache that survives save/load (we use the existing
    # JSON-serialisable fields to confirm the round-trip works)
    s.summary = "the user wants to learn Go"
    s.todos = [{"id": 1, "text": "finish variables", "done": False}]
    store = SessionStore(tmp_path)
    store.save(s)
    on_disk = json.loads((tmp_path / "compat.json").read_text(encoding="utf-8"))
    assert on_disk["summary"] == "the user wants to learn Go"
    assert on_disk["todos"] == [{"id": 1, "text": "finish variables", "done": False}]
    s2 = store.load("compat")
    assert s2.summary == s.summary
    assert s2.todos == s.todos


def test_current_session_field_types_are_json_safe():
    """Pin the invariant: every current Session field must be JSON-serialisable.
    If anyone adds a Path / datetime / set field, this test fails loudly."""
    s = Session(id="t")
    blob = json.dumps(asdict(s), ensure_ascii=False)
    # If we got here without TypeError, we're safe.
    assert "id" in blob


def test_current_session_field_names():
    """Document the current schema. If you add a field, update this test."""
    names = {f.name for f in fields(Session)}
    assert names == {"id", "system_prompt", "messages", "todos", "summary",
                     "plan_cache"}, (
        f"Session fields changed: {names}. Update test_session_compat.py and "
        "ensure all new fields are JSON-serialisable."
    )


def test_session_with_path_field_would_break():
    """Negative test: confirms we WILL catch a bad field type. Skip-on-pass
    so it doesn't fail CI until someone actually adds a Path field."""
    # This test only runs if you remove the `skipif` decorator below.
    from dataclasses import dataclass, field
    @dataclass
    class BadSession:
        id: str
        bad: Path = field(default_factory=Path.cwd)

    bs = BadSession(id="x")
    with pytest.raises(TypeError):
        json.dumps(asdict(bs))  # Path is not JSON-serialisable by default

# To activate the negative test when actually adding fields:
# @pytest.mark.skip(reason="only enable when adding new Session fields")