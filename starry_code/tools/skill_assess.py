"""skill_assess tool — structured gap analysis between current skills and target role.

Why this exists
---------------
Before the coach can write a learning plan, it needs to know where the
learner actually is. Without this, the LLM tends to give generic
"learn these things" advice without telling the user what's missing
specifically for their target role.

What it does
------------
The tool is a structured prompt: it tells the LLM "give me a gap analysis
in this exact JSON shape". The LLM does the assessment, returns the
JSON, and the tool hands it back. LLM-prompt-only — no external data.

When to call
------------
- First turn when the user names a target role without showing evidence
- After collecting 画像: turn the profile into a structured gap
- When the user says "我现在学到什么程度了" / "我差什么"
"""
from __future__ import annotations
from .base import Tool, ToolResult


class SkillAssessTool(Tool):
    """Return a structured gap analysis between learner and target role."""

    def __init__(self) -> None:
        super().__init__(
            name="skill_assess",
            description=(
                "对比用户当前技能 vs 目标岗位，输出结构化 gap 表。 "
                "返回三类清单：（1）已掌握；（2）核心缺口（按 P0/P1/P2 优先级排序，"
                "带建议学习周数）；（3）加分项（学了有优势，不学也能找到工作）。 "
                "**只在用户给了目标岗位 + 至少一项当前技能或证据时使用**。"
                "不要对还在画像采集阶段的用户调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "current_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "用户自报技能列表，如 ['Python 基础', '用过 Flask', '懂 SQL 基础']",
                    },
                    "target_role": {
                        "type": "string",
                        "description": "目标岗位，如 'Go 后端工程师'、'高级前端工程师'",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "自由文本：简历片段 / 项目列表 / GitHub 简介 / 工作年限",
                    },
                },
                "required": ["target_role"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        if session is None:
            return ToolResult.err("skill_assess requires an active session")
        # The tool itself doesn't compute anything — the LLM is asked to
        # produce a structured gap analysis. We return the LLM-friendly
        # prompt template; the LLM fills in the JSON.
        target_role = (args or {}).get("target_role", "").strip()
        if not target_role:
            return ToolResult.err("target_role is required")
        current_skills = (args or {}).get("current_skills") or []
        evidence = (args or {}).get("evidence") or ""

        if not current_skills and not evidence:
            return ToolResult.err(
                "need at least one of current_skills or evidence "
                "to assess; otherwise the analysis is unfounded"
            )

        # Build the structured-output instruction. The LLM fills the
        # template; the result is consumed by the coach for plan generation.
        skill_lines = "\n".join(f"  - {s}" for s in current_skills) or "  (未提供)"
        prompt = (
            "请基于用户输入，给出目标岗位的能力 gap 分析。\n\n"
            f"目标岗位：{target_role}\n\n"
            f"当前技能（用户自述）：\n{skill_lines}\n\n"
            f"背景证据：{evidence or '（未提供）'}\n\n"
            "输出严格按以下 JSON 结构（**只输出 JSON，无任何解释**）：\n"
            "{\n"
            '  "target_role": "...",\n'
            '  "gap_summary": "一句话总结 gap 严重程度 + 建议方向",\n'
            '  "categories": [\n'
            '    {"name": "已掌握", "items": ["..."]},\n'
            '    {"name": "核心缺口", "items": [\n'
            '      {"skill": "...", "priority": "P0", "est_weeks": 4, "reason": "为什么必须"},\n'
            '      {"skill": "...", "priority": "P1", "est_weeks": 2, "reason": "..."}\n'
            '    ]},\n'
            '    {"name": "加分项", "items": ["..."]}\n'
            "  ],\n"
            '  "recommended_next_action": "1 件事，用户接下来要做"\n'
            "}\n\n"
            "约束：\n"
            "- 核心缺口 0-5 条；P0 = 没有不能找工作；P1 = 有更好但可绕过；"
            "P2 = 加分但非必需\n"
            "- 加分项 0-3 条\n"
            "- est_weeks 是 0 基础到能写进简历的估计周数\n"
            "- recommended_next_action 必须是 1 件事（不是 5 件）"
        )
        # This tool is LLM-prompt-only; we return the prompt as the
        # "tool result" so the calling LLM knows how to fill it in.
        # (The Coach LLM will then call update_plan with the result.)
        return ToolResult.ok(prompt)