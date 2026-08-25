"""project_drive tool — generate a ship-able, resume-grade project skeleton.

Why this exists
---------------
The coach's mantra is "project-driven learning". When a learner has a
goal + time budget but no concrete ship-able target, give them a project
skeleton: what's the must-have features, the tech stack, the milestones
with acceptance criteria, and the interview talking points they can
memorize later.

What it does
------------
LLM-prompt-only. The tool returns a structured prompt asking the LLM
to produce a JSON project plan matching a specific shape. The LLM
fills it in; the result is consumed by the coach for downstream calls
(update_plan, todo, etc.).

When to call
------------
- After skill_assess, when the user has a clear goal
- When the user says "我应该做什么项目" / "学完 X 做什么"
- When plan_cache.next_task is empty AND goal is set
"""
from __future__ import annotations
from .base import Tool, ToolResult


class ProjectDriveTool(Tool):
    """Generate a project skeleton from goal + time budget + level."""

    def __init__(self) -> None:
        super().__init__(
            name="project_drive",
            description=(
                "根据学习目标 + 时间预算 + 当前水平，生成一个**可以写进简历的项目骨架**。"
                "返回 JSON 结构：项目名 + must_have_features + tech_stack + milestones + "
                "interview_talking_points。**只在用户目标明确且时间窗具体时调用**。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "学习目标 / 方向，如 '做 1 个能写进简历的 Go 后端项目'",
                    },
                    "time_budget_hours": {
                        "type": "integer",
                        "minimum": 8,
                        "maximum": 500,
                        "description": "总投入小时数，8-500 之间",
                    },
                    "current_level": {
                        "type": "enum",
                        "enum": ["beginner", "intermediate", "advanced"],
                        "description": "当前水平",
                    },
                    "target_resume_section": {
                        "type": "enum",
                        "enum": ["projects", "open_source", "side_quest"],
                        "description": "简历的哪个 section 放这个项目（默认 projects）",
                    },
                },
                "required": ["goal", "time_budget_hours"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        if session is None:
            return ToolResult.err("project_drive requires an active session")
        goal = (args or {}).get("goal", "").strip()
        if not goal:
            return ToolResult.err("goal is required")
        hours = (args or {}).get("time_budget_hours")
        if not isinstance(hours, int) or hours < 8 or hours > 500:
            return ToolResult.err(
                "time_budget_hours must be an integer between 8 and 500"
            )

        level = (args or {}).get("current_level", "intermediate")
        section = (args or {}).get("target_resume_section", "projects")

        # Build a prompt that the calling LLM will see as the "tool
        # result". It asks for a specific JSON shape; the LLM complies
        # in its next assistant message.
        prompt = (
            "请基于用户输入，生成一个可以写进简历的项目骨架。\n\n"
            f"目标：{goal}\n"
            f"总投入：{hours} 小时\n"
            f"当前水平：{level}\n"
            f"目标简历 section：{section}\n\n"
            "约束：\n"
            "- 项目必须能在总投入小时内完成（milestones 时间总和 <= hours）\n"
            "- 复杂度匹配用户当前水平（beginner 不要堆微服务）\n"
            "- 必须可写进简历，所以必须有 'interview_talking_points'：用户能背下来回答的问题\n\n"
            "输出严格按以下 JSON 结构（**只输出 JSON，无任何解释**）：\n"
            "{\n"
            '  "project_name": "项目名（简洁、可发音）",\n'
            '  "tagline": "一句话定位（10 字内）",\n'
            '  "must_have_features": ["feature1", "feature2", "feature3", "feature5_max"],\n'
            '  "tech_stack_suggested": ["技术 1", "技术 2", "..."],\n'
            '  "milestones": [\n'
            '    {"week": 1, "deliverable": "...", "acceptance": "判定完成的具体标志",\n'
            '     "hours_estimate": 10}\n'
            "  ],\n"
            '  "interview_talking_points": [\n'
            '    "为什么做这个项目",\n'
            '    "架构选型 / 关键技术决策",\n'
            '    "踩过的坑 / 学到的东西"\n'
            "  ],\n"
            '  "stretch_goals": ["ship 之后可以加的功能"]\n'
            "}\n\n"
            "硬性要求：\n"
            "- must_have_features: 3-5 条，每条用户能 demo演示\n"
            "- milestones: 2-5 个，hours_estimate 总和 = hours（±10%）\n"
            "- interview_talking_points: 3-5 条，能讲 5-10 分钟\n"
            "- 不要包含需要付费 / 需要邀请码 / 需要特殊环境的功能"
        )
        return ToolResult.ok(prompt)