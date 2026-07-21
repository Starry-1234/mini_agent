import pytest
from agent.tools.calculator import CalculatorTool
from agent.tools.base import ToolResult

calc = CalculatorTool()

def run(expr: str):
    return calc.execute({"expression": expr}, session=None)

def test_basic_arithmetic():
    r = run("1 + 2 * 3")
    assert r.ok and r.content.strip() == "7"

def test_parens_and_unary():
    r = run("-(3 - 5) ** 2 / 4")
    assert r.ok and r.content.strip() == "-1.0"

def test_rejects_function_calls():
    r = run("__import__('os').system('echo x')")
    assert not r.ok and "not allowed" in r.content.lower()

def test_rejects_names():
    r = run("foo + 1")
    assert not r.ok