"""project_drive tool tests.

Validates the structured-prompt helper that asks the LLM to produce a
project skeleton JSON.
"""
import pytest

from starry_code.tools.project_drive import ProjectDriveTool
from starry_code.session import Session
from starry_code.runtime import build_default_registry


def test_tool_registered_in_default_registry():
    reg = build_default_registry()
    assert "project_drive" in reg.names()


def test_openai_schema_shape():
    reg = build_default_registry()
    schemas = reg.openai_schemas()
    pd = next(x for x in schemas if x["function"]["name"] == "project_drive")
    params = pd["function"]["parameters"]
    required = params["required"]
    assert "goal" in required
    assert "time_budget_hours" in required
    props = params["properties"]
    assert "current_level" in props
    assert "target_resume_section" in props


def test_missing_session_returns_err():
    r = ProjectDriveTool().execute(
        {"goal": "做 1 个 CLI", "time_budget_hours": 40},
        session=None,
    )
    assert not r.ok
    assert "session" in r.content.lower()


def test_missing_goal_returns_err():
    r = ProjectDriveTool().execute(
        {"time_budget_hours": 40},
        session=Session(id="x"),
    )
    assert not r.ok
    assert "goal" in r.content.lower()


def test_missing_hours_returns_err():
    r = ProjectDriveTool().execute(
        {"goal": "做 1 个 CLI"},
        session=Session(id="x"),
    )
    assert not r.ok
    assert "time_budget_hours" in r.content.lower()


def test_hours_too_small_returns_err():
    """Below 8 hours is too short for a real project — refuse."""
    r = ProjectDriveTool().execute(
        {"goal": "做 1 个小项目", "time_budget_hours": 4},
        session=Session(id="x"),
    )
    assert not r.ok
    assert "between 8 and 500" in r.content or "8 and 500" in r.content


def test_hours_too_large_returns_err():
    r = ProjectDriveTool().execute(
        {"goal": "做 1 个大项目", "time_budget_hours": 1000},
        session=Session(id="x"),
    )
    assert not r.ok


def test_hours_non_integer_returns_err():
    r = ProjectDriveTool().execute(
        {"goal": "做项目", "time_budget_hours": "40"},  # string, not int
        session=Session(id="x"),
    )
    assert not r.ok


def test_basic_call_ok():
    r = ProjectDriveTool().execute(
        {"goal": "做 1 个能写进简历的 Go 后端项目",
         "time_budget_hours": 40},
        session=Session(id="x"),
    )
    assert r.ok


def test_result_prompt_includes_goal_and_hours():
    r = ProjectDriveTool().execute(
        {"goal": "做 1 个 Go CLI", "time_budget_hours": 30},
        session=Session(id="x"),
    )
    assert "Go CLI" in r.content
    assert "30" in r.content


def test_result_prompt_specifies_json_keys():
    r = ProjectDriveTool().execute(
        {"goal": "做项目", "time_budget_hours": 40},
        session=Session(id="x"),
    )
    for key in ("must_have_features", "tech_stack_suggested", "milestones",
                "interview_talking_points", "stretch_goals"):
        assert key in r.content, f"prompt missing key: {key}"


def test_result_prompt_includes_constraints():
    """The prompt must remind the LLM about time-budget fit and interview
    readiness — the whole point of the tool is to ship something that
    demonstrates a skill."""
    r = ProjectDriveTool().execute(
        {"goal": "做项目", "time_budget_hours": 40},
        session=Session(id="x"),
    )
    assert "interview" in r.content.lower() or "面试" in r.content
    assert "milestones" in r.content
    # Total hours must equal budget (within 10%)
    assert "hours" in r.content.lower() or "工时" in r.content or "小时" in r.content


def test_default_level_and_section_applied():
    r = ProjectDriveTool().execute(
        {"goal": "做项目", "time_budget_hours": 40},
        session=Session(id="x"),
    )
    # Defaults documented in code: intermediate, projects
    assert "intermediate" in r.content
    assert "projects" in r.content