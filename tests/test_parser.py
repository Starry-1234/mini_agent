from starry_code.parser import parse_response, ParsedResponse, _strip_thinking

def test_parses_tool_calls_and_thought():
    raw = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "I should compute.",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'},
                }],
            }
        }]
    }
    p = parse_response(raw)
    assert isinstance(p, ParsedResponse)
    assert p.thought == "I should compute."
    assert p.final_answer is None
    assert len(p.tool_calls) == 1
    assert p.tool_calls[0].name == "calculator"
    assert p.tool_calls[0].args == {"expression": "1+1"}

def test_parses_final_answer():
    raw = {"choices": [{"message": {"role": "assistant", "content": "Hello there."}}]}
    p = parse_response(raw)
    assert p.final_answer == "Hello there."
    assert p.tool_calls == []


def test_final_answer_strips_thinking_block():
    raw = {"choices": [{"message": {
        "role": "assistant",
        "content": "<think>The user is asking a math question.</think>\n\n2 + 2 = **4**",
    }}]}
    p = parse_response(raw)
    assert p.final_answer == "2 + 2 = **4**"
    # thought keeps the reasoning for the trace log
    assert "<think>" in p.thought


def test_strip_thinking_helper():
    # Case-insensitive tag, multiline, multiple blocks.
    text = "<THINK>line1\nline2</THINK>answer<think>more</think> tail"
    assert _strip_thinking(text) == "answer tail"
    # Empty after stripping -> None
    assert _strip_thinking("<think>only reasoning</think>") is None
    assert _strip_thinking("") is None
    # No block -> unchanged (trimmed)
    assert _strip_thinking("  plain  ") == "plain"