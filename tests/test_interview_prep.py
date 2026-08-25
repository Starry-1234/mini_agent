"""interview_prep tool tests."""
import pytest

from starry_code.tools.interview_prep import InterviewPrepTool
from starry_code.session import Session
from starry_code.runtime import build_default_registry


def test_tool_registered():
    reg = build_default_registry()
    assert "interview_prep" in reg.names()


def test_openai_schema_shape():
    reg = build_default_registry()
    schemas = reg.openai_schemas()
    ip = next(x for x in schemas if x["function"]["name"] == "interview_prep")
    props = ip["function"]["parameters"]["properties"]
    assert "role" in ip["function"]["parameters"]["required"]
    assert "level" in props
    assert props["level"]["enum"] == ["junior", "mid", "senior"]
    assert "n_questions" in props
    assert props["n_questions"]["default"] == 5
    assert "focus" in props
    assert props["focus"]["type"] == "array"


def test_missing_session_returns_err():
    r = InterviewPrepTool().execute({"role": "Go 后端"}, session=None)
    assert not r.ok
    assert "session" in r.content.lower()


def test_missing_role_returns_err():
    r = InterviewPrepTool().execute({}, session=Session(id="x"))
    assert not r.ok
    assert "role" in r.content.lower()


def test_n_questions_below_range_returns_err():
    r = InterviewPrepTool().execute(
        {"role": "Go", "n_questions": 1}, session=Session(id="x")
    )
    assert not r.ok


def test_n_questions_above_range_returns_err():
    r = InterviewPrepTool().execute(
        {"role": "Go", "n_questions": 100}, session=Session(id="x")
    )
    assert not r.ok


def test_n_questions_non_int_returns_err():
    r = InterviewPrepTool().execute(
        {"role": "Go", "n_questions": "5"}, session=Session(id="x")
    )
    assert not r.ok


def test_basic_call_ok():
    r = InterviewPrepTool().execute(
        {"role": "Go 后端工程师", "level": "mid", "n_questions": 5},
        session=Session(id="x"),
    )
    assert r.ok


def test_result_prompt_includes_role_and_level():
    r = InterviewPrepTool().execute(
        {"role": "高级前端工程师", "level": "senior", "n_questions": 7},
        session=Session(id="x"),
    )
    assert "高级前端工程师" in r.content
    assert "senior" in r.content
    assert "7" in r.content


def test_result_prompt_includes_focus():
    r = InterviewPrepTool().execute(
        {"role": "Go", "focus": ["goroutine", "context", "gc"]},
        session=Session(id="x"),
    )
    for topic in ("goroutine", "context", "gc"):
        assert topic in r.content


def test_result_prompt_includes_json_keys():
    r = InterviewPrepTool().execute(
        {"role": "Go 后端"}, session=Session(id="x")
    )
    for key in ("questions", "key_points", "common_pitfall",
                "estimated_prep_time_hours", "study_links"):
        assert key in r.content, f"prompt missing key: {key}"


def test_result_prompt_constraints_match_input():
    """The prompt must explicitly carry the n_questions count so the LLM
    produces exactly that many questions."""
    r = InterviewPrepTool().execute(
        {"role": "Go", "n_questions": 8}, session=Session(id="x")
    )
    assert "= 8" in r.content
    assert "8" in r.content


def test_result_prompt_distinguishes_levels():
    """junior / mid / senior should yield different difficulty framing."""
    r_junior = InterviewPrepTool().execute(
        {"role": "X", "level": "junior"}, session=Session(id="x")
    )
    r_senior = InterviewPrepTool().execute(
        {"role": "X", "level": "senior"}, session=Session(id="x")
    )
    # senior prompt emphasizes "系统设计", junior emphasizes "基础概念"
    assert "系统" in r_senior.content or "system" in r_senior.content.lower()
    assert "基础" in r_junior.content or "basic" in r_junior.content.lower()