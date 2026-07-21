import pytest

from starry_code.config import Settings
from starry_code.session import Session, SessionStore
from starry_code.llm import MockLLMClient
from starry_code.runtime import run_turn, build_default_registry, build_memory, _should_extract
from starry_code.trace import TraceLogger


@pytest.fixture
def env(tmp_path):
    s = Settings(sessions_dir=tmp_path)
    sess = Session(id="rt")
    store = SessionStore(tmp_path)
    reg = build_default_registry()
    mem = build_memory(settings=s, llm=None)
    trace = TraceLogger(tmp_path, "rt")
    return s, sess, store, reg, mem, trace


def test_run_turn_direct_answer(env):
    settings, sess, store, reg, mem, trace = env
    llm = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "pong"}}]}
    ])
    out = run_turn(
        sess, "ping",
        settings=settings, llm=llm, registry=reg, memory=mem, trace=trace,
    )
    assert out == "pong"
    store.save(sess)
    assert sess.messages[-1]["role"] == "assistant"


def test_run_turn_with_tool_loop(env):
    settings, sess, store, reg, mem, trace = env
    llm = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "computing",
                                   "tool_calls": [{"id": "c1", "type": "function",
                                                   "function": {"name": "calculator",
                                                                "arguments": '{"expression":"1+1"}'}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "the answer is 2"}}]},
    ])
    out = run_turn(
        sess, "what is 1+1",
        settings=settings, llm=llm, registry=reg, memory=mem, trace=trace,
    )
    assert out == "the answer is 2"
    assert any(m["role"] == "tool" for m in sess.messages)


def test_max_iters_forces_finalize(env):
    settings, sess, store, reg, mem, trace = env
    # Always emits a tool call — should hit MAX_TOOL_ITERS and force-finish.
    tool_call = {
        "id": "c", "type": "function",
        "function": {"name": "calculator", "arguments": '{"expression":"0"}'},
    }
    llm = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "loop",
                                   "tool_calls": [tool_call]}}]}
        for _ in range(20)
    ])
    out = run_turn(
        sess, "go",
        settings=settings, llm=llm, registry=reg, memory=mem, trace=trace,
    )
    assert isinstance(out, str)
    assert (
        len([m for m in sess.messages if m.get("role") == "tool"])
        == settings.max_tool_iters
    )


def test_should_extract_skips_trivial_turns():
    # Trivial: below the 80-char combined threshold -> skip extraction.
    assert _should_extract("hi", "hello") is False
    assert _should_extract("what is 2+2?", "2 + 2 = 4") is False


def test_should_extract_allows_substantive_turns():
    user = "Remember that I love green tea in the morning and prefer it iced."
    answer = "Got it, I've noted your preference for iced green tea in the morning."
    assert _should_extract(user, answer) is True