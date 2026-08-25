"""interview_prep tool — generate role+level interview questions.

Why this exists
---------------
When the learner is preparing for interviews, the most actionable thing
is to practice questions tailored to their target role + level. Generic
"LeetCode 500 题" advice is useless. This tool gives them N role-specific
questions with key points + common pitfalls — exactly what they need to
memorize and rehearse.

What it does
------------
LLM-prompt-only. Returns a structured prompt asking the LLM to produce
N questions in a specific JSON shape with key points + pitfalls.

When to call
------------
- After skill_assess has identified the target role
- When the user says "我要面试 X" / "准备面试" / "出几道题"
- Before milestone reviews (test the learner's retention)
"""
from __future__ import annotations
from .base import Tool, ToolResult


class InterviewPrepTool(Tool):
    """Generate N role+level interview questions with key points."""

    def __init__(self) -> None:
        super().__init__(
            name="interview_prep",
            description=(
                "为目标岗位 + 职级生成 N 道面试题。"
                "每道题带 key_points（回答要点）+ common_pitfall（常见踩坑）。"
                "**只在用户目标岗位 + 职级明确时使用**。"
                "不要给泛泛刷题建议。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": "目标岗位，如 'Go 后端工程师'、'前端架构师'",
                    },
                    "level": {
                        "type": "enum",
                        "enum": ["junior", "mid", "senior"],
                        "description": "目标职级",
                    },
                    "n_questions": {
                        "type": "integer",
                        "minimum": 3,
                        "maximum": 20,
                        "default": 5,
                        "description": "生成几道题，3-20 之间，默认 5",
                    },
                    "focus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "重点考察的技术点（如 ['goroutine', 'context', 'gc']）",
                    },
                },
                "required": ["role"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        if session is None:
            return ToolResult.err("interview_prep requires an active session")
        role = (args or {}).get("role", "").strip()
        if not role:
            return ToolResult.err("role is required")
        level = (args or {}).get("level", "mid")
        n = (args or {}).get("n_questions", 5)
        if not isinstance(n, int) or n < 3 or n > 20:
            return ToolResult.err("n_questions must be an integer between 3 and 20")
        focus = (args or {}).get("focus") or []
        focus_lines = "\n".join(f"  - {f}" for f in focus) or "  （未指定）"

        prompt = (
            "请为目标岗位生成面试模拟题。\n\n"
            f"目标岗位：{role}\n"
            f"职级：{level}\n"
            f"题数：{n}\n"
            f"重点考察：\n{focus_lines}\n\n"
            "约束：\n"
            f"- 题数 = {n}\n"
            "- 题目难度匹配职级：junior 偏基础概念；mid 偏实战 + 原理；senior 偏系统 + 决策\n"
            "- 如果给了 focus 列表，至少 60% 的题要覆盖列表里的技术点\n"
            "- 每道题要有 key_points（3-5 条回答要点）和 common_pitfall（常见踩坑）\n\n"
            "输出严格按以下 JSON 结构（**只输出 JSON，无任何解释**）：\n"
            "{\n"
            '  "role": "...", "level": "...",\n'
            '  "questions": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "type": "技术基础 / 编程题 / 系统设计 / 行为面试",\n'
            '      "difficulty": "junior|mid|senior",\n'
            '      "question": "...",\n'
            '      "key_points": ["...", "..."],\n'
            '      "common_pitfall": "..."\n'
            "    }\n"
            "  ],\n"
            '  "study_links": ["https://...", "..."],\n'
            '  "estimated_prep_time_hours": 12\n'
            "}\n\n"
            "硬性要求：\n"
            f"- questions 数组长度 = {n}\n"
            "- key_points 每条 1 行、可执行；common_pitfall 一句话\n"
            "- 不要给链接除非是公认权威源（官方文档 / 经典书 / RFC）\n"
            "- estimated_prep_time_hours 是用户准备这 N 道题需要的总小时数"
        )
        return ToolResult.ok(prompt)