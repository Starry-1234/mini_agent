from agent.parser import parse_response, ParsedResponse

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