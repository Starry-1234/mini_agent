import pytest
from starry_code.tools.base import Tool, ToolResult
from starry_code.tools.registry import ToolRegistry

def echo_execute(args, session):
    return ToolResult(ok=True, content=f"got {args['x']}")

def test_registry_roundtrip():
    reg = ToolRegistry()
    reg.register(Tool(
        name="echo",
        description="echoes x",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        execute=echo_execute,
    ))
    schemas = reg.openai_schemas()
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "echo"
    assert reg.names() == ["echo"]
    res = reg.execute("echo", {"x": "hi"}, session=None)
    assert res.ok and res.content == "got hi"

def test_unknown_tool_returns_error_result():
    reg = ToolRegistry()
    res = reg.execute("nope", {}, session=None)
    assert not res.ok
    assert "unknown" in res.content.lower()
