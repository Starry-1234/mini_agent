from __future__ import annotations
import ast
import operator
from .base import Tool, ToolResult


_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos, ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Paren):  # not used by Python AST, but keep for explicitness
        return _safe_eval(node.value)  # type: ignore[attr-defined]
    raise ValueError(f"expression node not allowed: {type(node).__name__}")


class CalculatorTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description="Evaluate a basic arithmetic expression. Supports + - * / // % ** and parentheses.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        expr = (args or {}).get("expression", "").strip()
        if not expr:
            return ToolResult.err("expression is required")
        try:
            tree = ast.parse(expr, mode="eval")
            value = _safe_eval(tree)
            return ToolResult.ok(str(value))
        except Exception as e:  # noqa: BLE001
            return ToolResult.err(f"not allowed or invalid expression: {e}")