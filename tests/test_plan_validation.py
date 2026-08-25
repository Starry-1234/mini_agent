"""W2c tests — plan_cache structure validation.

Per external-agent feedback Q5: light validation only. We block:
- non-string types (LLM might pass int/dict by mistake)
- very long fields (would bloat the per-turn system prompt)
- obvious prompt-injection patterns (<script>, javascript:, event handlers)

We do NOT block:
- markdown, lists, etc. (the LLM uses these legitimately)
- foreign languages (the user wrote them in)
"""
import pytest

from starry_code.tools.update_plan import UpdatePlanTool, _validate_field
from starry_code.session import Session


# ---- Unit: _validate_field ----

def test_validate_accepts_normal_chinese():
    assert _validate_field("stage", "阶段1：Go 语法基础") is None
    assert _validate_field("next_task", "写 5 道 if/else 题") is None
    assert _validate_field("long_term_goal", "3 个月内转 Go 后端") is None


def test_validate_accepts_english():
    assert _validate_field("stage", "Stage 1: Go syntax basics") is None


def test_validate_accepts_with_markdown():
    """Markdown in plan is normal — bullet/numbered lists are common."""
    assert _validate_field("next_task",
                          "1. Read docs\n2. Write 3 exercises\n3. Review") is None


def test_validate_rejects_non_string():
    err = _validate_field("stage", 12345)
    assert err and "must be a string" in err


def test_validate_rejects_too_long():
    err = _validate_field("stage", "x" * 501)
    assert err and "too long" in err
    assert "500" in err


def test_validate_rejects_script_tag():
    """<script> tag in plan is almost always an injection."""
    for payload in (
        "<script>alert(1)</script>",
        "<SCRIPT>alert(1)</SCRIPT>",
        "< script >alert(1)</script>",
        "<script src='evil.js'>",
    ):
        err = _validate_field("stage", payload)
        assert err and "injection" in err.lower(), f"missed: {payload!r}"


def test_validate_rejects_javascript_url():
    err = _validate_field("next_task", "click javascript:alert(1)")
    assert err and "injection" in err.lower()


def test_validate_rejects_event_handler():
    for payload in (
        "<img onerror=alert(1)>",
        "<a onclick=alert(1)>",
    ):
        err = _validate_field("stage", payload)
        assert err and "injection" in err.lower(), f"missed: {payload!r}"


def test_validate_rejects_iframe():
    err = _validate_field("stage", "<iframe src=evil.com></iframe>")
    assert err and "injection" in err.lower()


def test_validate_accepts_at_length_boundary():
    """Exactly 500 chars is OK (boundary)."""
    assert _validate_field("stage", "x" * 500) is None
    assert _validate_field("stage", "x" * 501) is not None


# ---- Integration: UpdatePlanTool.apply() ----

def test_update_plan_rejects_injection_in_one_field_keeps_others():
    """If one field has a malicious payload, reject it but apply the others."""
    s = Session(id="p")
    r = UpdatePlanTool().execute(
        {"stage": "阶段1：正常",
         "next_task": "<script>alert(1)</script>",
         "long_term_goal": "正常 goal"},
        session=s,
    )
    # Stage and long_term_goal applied; next_task rejected
    assert s.plan_cache["stage"] == "阶段1：正常"
    assert s.plan_cache["long_term_goal"] == "正常 goal"
    assert s.plan_cache.get("next_task", "") == ""
    # version bumped (since stage and long_term_goal changed)
    assert s.plan_cache["version"] == 1
    # result mentions rejection
    assert "rejected" in r.content.lower()


def test_update_plan_rejects_all_returns_error():
    """If every field is rejected, return ok=False."""
    s = Session(id="p")
    r = UpdatePlanTool().execute(
        {"stage": "<script>x</script>",
         "next_task": "javascript:bad"},
        session=s,
    )
    assert not r.ok
    assert s.plan_cache["stage"] == ""
    assert s.plan_cache["version"] == 0


def test_update_plan_rejects_too_long_keeps_siblings():
    """One field too long, others still apply."""
    s = Session(id="p")
    r = UpdatePlanTool().execute(
        {"stage": "正常 stage",
         "next_task": "x" * 600},
        session=s,
    )
    assert s.plan_cache["stage"] == "正常 stage"
    # next_task stays at default empty string (rejected before write)
    assert s.plan_cache["next_task"] == ""
    assert s.plan_cache["version"] == 1


def test_update_plan_accepts_500_chars_exactly():
    """Boundary: 500 chars must succeed (not be rejected as too long)."""
    s = Session(id="p")
    r = UpdatePlanTool().execute(
        {"stage": "x" * 500},
        session=s,
    )
    assert r.ok
    assert len(s.plan_cache["stage"]) == 500


def test_update_plan_injection_in_stored_value_not_rejected():
    """W2c validates *new writes only*. Existing values (e.g. legacy JSON)
    are kept verbatim — don't break migration."""
    s = Session(id="p")
    s.plan_cache["stage"] = "<legacy from old session>"
    # Update only next_task — stage is preserved
    r = UpdatePlanTool().execute({"next_task": "新任务"}, session=s)
    assert r.ok
    assert "<legacy from old session>" in s.plan_cache["stage"]