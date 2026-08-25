"""skill_assess tool tests.

Validates:
1. Schema registration (name, params, required)
2. Validation: requires target_role; needs at least one of
   current_skills or evidence
3. Output: returns a structured prompt that names the target role
   and asks the LLM to produce a gap-analysis JSON
"""
import pytest

from starry_code.tools.skill_assess import SkillAssessTool
from starry_code.session import Session
from starry_code.runtime import build_default_registry


def test_tool_registered_in_default_registry():
    reg = build_default_registry()
    assert "skill_assess" in reg.names()


def test_openai_schema_shape():
    reg = build_default_registry()
    schemas = reg.openai_schemas()
    sa = next(x for x in schemas if x["function"]["name"] == "skill_assess")
    props = sa["function"]["parameters"]["properties"]
    assert "target_role" in sa["function"]["parameters"]["required"]
    assert "target_role" in props
    assert "current_skills" in props
    assert "evidence" in props
    # current_skills should be an array
    assert props["current_skills"]["type"] == "array"


def test_missing_session_returns_err():
    r = SkillAssessTool().execute(
        {"target_role": "Go 后端", "current_skills": ["Python 基础"]},
        session=None,
    )
    assert not r.ok
    assert "session" in r.content.lower()


def test_missing_target_role_returns_err():
    r = SkillAssessTool().execute(
        {"current_skills": ["Python 基础"]},
        session=Session(id="x"),
    )
    assert not r.ok
    assert "target_role" in r.content.lower()


def test_missing_both_skills_and_evidence_returns_err():
    """Without any data, we'd be guessing — refuse."""
    r = SkillAssessTool().execute(
        {"target_role": "Go 后端"},
        session=Session(id="x"),
    )
    assert not r.ok
    assert "current_skills" in r.content or "evidence" in r.content


def test_only_evidence_is_enough():
    """If user provides only evidence (no explicit skills list), still
    workable — the LLM can infer from a project list or bio."""
    r = SkillAssessTool().execute(
        {"target_role": "Go 后端",
         "evidence": "2 年 Java 开发，做过支付系统"},
        session=Session(id="x"),
    )
    assert r.ok


def test_only_skills_is_enough():
    r = SkillAssessTool().execute(
        {"target_role": "Go 后端",
         "current_skills": ["Python 基础", "用过 Docker"]},
        session=Session(id="x"),
    )
    assert r.ok


def test_result_prompt_includes_target_role():
    """The prompt the LLM will see must name the target role."""
    r = SkillAssessTool().execute(
        {"target_role": "高级 React工程师",
         "current_skills": ["JavaScript"]},
        session=Session(id="x"),
    )
    assert r.ok
    assert "高级 React工程师" in r.content


def test_result_prompt_mentions_json_shape():
    """The prompt must instruct the LLM to output JSON in the documented
    categories / gap_summary / recommended_next_action shape."""
    r = SkillAssessTool().execute(
        {"target_role": "Go 后端",
         "current_skills": ["Python"]},
        session=Session(id="x"),
    )
    assert "JSON" in r.content or "json" in r.content
    for field in ("gap_summary", "categories", "核心缺口", "加分项",
                  "recommended_next_action"):
        assert field in r.content, f"prompt missing field: {field}"


def test_result_prompt_lists_current_skills():
    r = SkillAssessTool().execute(
        {"target_role": "Go 后端",
         "current_skills": ["Python 基础", "用过 Flask"]},
        session=Session(id="x"),
    )
    assert "Python 基础" in r.content
    assert "用过 Flask" in r.content


def test_result_prompt_includes_priority_labels():
    """The prompt instructs the LLM to use P0/P1/P2 priority labels."""
    r = SkillAssessTool().execute(
        {"target_role": "Go 后端",
         "current_skills": ["Python"]},
        session=Session(id="x"),
    )
    for p in ("P0", "P1"):
        assert p in r.content


def test_result_prompt_constrains_size():
    """Limit core gaps to 0-5, bonus to 0-3 — keeps the JSON small enough
    to inject back into plan_cache without bloat."""
    r = SkillAssessTool().execute(
        {"target_role": "Go 后端",
         "current_skills": ["Python"]},
        session=Session(id="x"),
    )
    assert "0-5" in r.content  # 核心缺口 0-5 条
    assert "0-3" in r.content  # 加分项 0-3 条