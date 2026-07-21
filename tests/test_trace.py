import json
from starry_code.trace import TraceLogger


def test_trace_appends_jsonl_and_terminal(tmp_path, capsys):
    log = TraceLogger(tmp_path, "s1")
    log.event("user", text="hi")
    log.event("thought", text="thinking")
    log.event("tool_call", name="calculator", args={"expression": "1+1"})
    log.event("tool_result", name="calculator", result="2")
    log.event("assistant", text="done")

    lines = (tmp_path / "s1.trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    parsed = [json.loads(line) for line in lines]
    assert [record["kind"] for record in parsed] == [
        "user",
        "thought",
        "tool_call",
        "tool_result",
        "assistant",
    ]
    assert parsed[2]["name"] == "calculator"
    out = capsys.readouterr().err
    assert "user" in out and "calculator" in out and "assistant" in out
