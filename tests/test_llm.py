import pytest
from agent.llm import LLMClient, MockLLMClient

def test_mock_chat_returns_first_scripted():
    m = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    ])
    out = m.chat([{"role": "user", "content": "x"}], tools=None)
    assert out["choices"][0]["message"]["content"] == "hi"

def test_mock_chat_rotates_and_errors_when_exhausted():
    m = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "a"}}]}
    ])
    m.chat([], tools=None)
    with pytest.raises(RuntimeError):
        m.chat([], tools=None)

def test_mock_embed_deterministic():
    m = MockLLMClient()
    v1 = m.embed(["hello world"])
    v2 = m.embed(["hello world"])
    assert v1 == v2 and len(v1[0]) == 16

def test_real_client_requires_config():
    c = LLMClient(api_key="", base_url="x")
    with pytest.raises(RuntimeError):
        c.chat([], tools=None)