"""update_plan tool — coach's only sanctioned way to mutate the active plan.

Why this exists
---------------
plan_cache (Session.plan_cache) is the coach's working memory: it lives
inside every LLM call as a ~50-token system block so the coach always
knows where the learner is. If we let the LLM "think" about plan changes
and just hope they stick, the cache would drift from reality.

Instead: any plan change MUST go through this tool. Only then does
session.plan_cache get touched. The version counter increments on every
real change, so callers can detect "did the coach actually update?"

When to call
------------
- After collecting enough 画像 to write a real plan (image phase → plan)
- When the learner finishes a milestone (advance stage)
- When the user changes goals (long_term_goal)
- Never speculatively — every call should be backed by something the
  user actually said or did.
"""
from __future__ import annotations
from datetime import datetime, timezone
from .base import Tool, ToolResult


class UpdatePlanTool(Tool):
    """Update session.plan_cache. Only this tool may mutate plan state."""

    def __init__(self) -> None:
        super().__init__(
            name="update_plan",
            description=(
                "更新当前学习计划缓存（plan_cache）。"
                "**只有当计划有实质变化时才调用**（新阶段、下一步任务改变、长期目标调整）。"
                "传入字段为空字符串或与现有值相同则不修改该字段；版本号仅在至少一个字段真"
                "正变化时才会自增。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "description": "当前阶段名，如 '阶段1：基础语法'。空字符串表示不变。",
                    },
                    "next_task": {
                        "type": "string",
                        "description": "用户接下来要做的 1 件事（不是 5 件）。空字符串表示不变。",
                    },
                    "long_term_goal": {
                        "type": "string",
                        "description": "用户的长期目标（岗位 / 方向 / 时间窗口）。空字符串表示不变。",
                    },
                },
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        if session is None:
            return ToolResult.err("update_plan requires an active session")
        pc = session.plan_cache
        if not isinstance(pc, dict):
            # Defensive: if a session was constructed before plan_cache
            # existed and somehow ended up with the wrong type, recover.
            pc = {}
            session.plan_cache = pc

        changed: list[str] = []
        for k in ("stage", "next_task", "long_term_goal"):
            new = (args or {}).get(k)
            # Treat empty string as "no change" (caller didn't intend to
            # clear the field) — use a different sentinel if explicit clear
            # is needed later.
            if new is None or new == "":
                continue
            if new == pc.get(k, ""):
                continue
            pc[k] = new
            changed.append(k)

        if changed:
            pc["version"] = pc.get("version", 0) + 1
            pc["last_updated"] = datetime.now(timezone.utc).isoformat()
            return ToolResult.ok(
                f"plan updated (v{pc['version']}): changed={changed}, "
                f"next_task='{pc.get('next_task', '')}'"
            )
        return ToolResult.ok("no-op: no field actually changed")