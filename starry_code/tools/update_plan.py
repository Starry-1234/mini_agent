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

Validation (W2c)
---------------
Per external-agent feedback Q5: light structure validation only. We
verify type, length, and strip the most obvious injection vectors
(<script>, javascript: URLs, control chars). Heavy-duty content moderation is
left to a future W-phase when we ship multi-user.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from .base import Tool, ToolResult


# Validation constants. Tuned to be permissive enough that the LLM can
# express plans naturally, strict enough to block the most obvious
# prompt-injection payloads.
_MAX_FIELD_LEN = 500        # plan_cache is hot cache; longer strings bloat every call
_INJECTION_PATTERNS = re.compile(
    r"<\s*script\b|javascript\s*:|on\w+\s*=|<\s*iframe\b",
    re.IGNORECASE,
)


def _validate_field(name: str, value: str) -> str | None:
    """Return None if OK, else an error message."""
    if not isinstance(value, str):
        return f"{name} must be a string, got {type(value).__name__}"
    if len(value) > _MAX_FIELD_LEN:
        return (f"{name} too long ({len(value)} chars, max {_MAX_FIELD_LEN}). "
                "Split into multiple update_plan calls instead.")
    if _INJECTION_PATTERNS.search(value):
        return f"{name} contains suspicious pattern (likely prompt injection)"
    return None


class UpdatePlanTool(Tool):
    """Update session.plan_cache. Only this tool may mutate plan state."""

    def __init__(self) -> None:
        super().__init__(
            name="update_plan",
            description=(
                "更新当前学习计划缓存（plan_cache）。"
                "**只有当计划有实质变化时才调用**（新阶段、下一步任务改变、长期目标调整）。"
                "传入字段为空字符串或与现有值相同则不修改该字段；版本号仅在至少一个字段真"
                "正变化时才会自增。每个字段最长 500 字符。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "description": "当前阶段名，如 '阶段1：基础语法'。空字符串表示不变。",
                        "maxLength": _MAX_FIELD_LEN,
                    },
                    "next_task": {
                        "type": "string",
                        "description": "用户接下来要做的 1 件事（不是 5 件）。空字符串表示不变。",
                        "maxLength": _MAX_FIELD_LEN,
                    },
                    "long_term_goal": {
                        "type": "string",
                        "description": "用户的长期目标（岗位 / 方向 / 时间窗口）。空字符串表示不变。",
                        "maxLength": _MAX_FIELD_LEN,
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
            pc = {}
            session.plan_cache = pc

        changed: list[str] = []
        rejected: list[str] = []
        for k in ("stage", "next_task", "long_term_goal"):
            new = (args or {}).get(k)
            # Treat empty string as "no change" (caller didn't intend to
            # clear the field) — use a different sentinel if explicit clear
            # is needed later.
            if new is None or new == "":
                continue
            if new == pc.get(k, ""):
                continue
            # W2c validation: check before applying
            err = _validate_field(k, new)
            if err:
                rejected.append(f"{k}: {err}")
                continue
            pc[k] = new
            changed.append(k)

        if rejected and not changed:
            return ToolResult.err(
                "update_plan rejected: " + "; ".join(rejected)
            )

        if changed:
            pc["version"] = pc.get("version", 0) + 1
            pc["last_updated"] = datetime.now(timezone.utc).isoformat()
            msg = (f"plan updated (v{pc['version']}): changed={changed}, "
                   f"next_task='{pc.get('next_task', '')}'")
            if rejected:
                msg += f" | rejected: {rejected}"
            return ToolResult.ok(msg)
        if rejected:
            return ToolResult.err(
                "no-op (no field changed) but rejected: " + "; ".join(rejected)
            )
        return ToolResult.ok("no-op: no field actually changed")