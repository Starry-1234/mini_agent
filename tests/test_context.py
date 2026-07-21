from starry_code.context import ContextBuilder
from starry_code.session import Session
from starry_code.config import Settings
from starry_code.memory.embeddings import MockEmbedder
from starry_code.memory.short_term import InMemoryShortTermStore
from starry_code.memory.vector_store import LocalVectorStore
from starry_code.memory.manager import MemoryManager

def make_cb(tmp_path):
    s = Settings(sessions_dir=tmp_path, context_max_messages=6, recent_keep=2)
    mm = MemoryManager(MockEmbedder(), InMemoryShortTermStore(),
                       LocalVectorStore(embedder=MockEmbedder(), path=None))
    return ContextBuilder(memory=mm, settings=s)

def test_build_basic(tmp_path):
    cb = make_cb(tmp_path)
    sess = Session(id="s")
    # The caller (runtime) is now responsible for adding the user message.
    sess.add_user("what's up?")
    msgs, tools = cb.build(sess, "what's up?")
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "what's up?"}
    assert isinstance(tools, list)
    # The user message exists on the session (added by the test setup, not
    # the builder).
    assert sess.messages[-1] == {"role": "user", "content": "what's up?"}


def test_build_does_not_duplicate_user(tmp_path):
    # Regression for F7: calling build twice for the same turn (as the tool
    # loop does per LLM iteration) must NOT duplicate the user message.
    cb = make_cb(tmp_path)
    sess = Session(id="s")
    sess.add_user("what's up?")
    cb.build(sess, "what's up?")
    cb.build(sess, "what's up?")
    user_msgs = [m for m in sess.messages if m.get("role") == "user"]
    assert user_msgs == [{"role": "user", "content": "what's up?"}]

def test_context_triggers_summary_when_long(tmp_path):
    cb = make_cb(tmp_path)
    sess = Session(id="s")
    for i in range(10):
        sess.add_user(f"u{i}")
        sess.add_assistant(f"a{i}")
    # build with a mock llm; we will pass a stub via the manager
    cb.memory.llm = None  # extractor is irrelevant; summary uses settings-provided LLM via env? Not in builder.
    # Force summary path by setting a no-op LLM via the builder attribute
    from starry_code.llm import MockLLMClient
    cb.summarizer = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "summary text"}}]}
    ])
    # Caller adds the user turn before building.
    sess.add_user("next q")
    msgs, _ = cb.build(sess, "next q")
    # summary appears as a system message after the first system
    sys_msgs = [m for m in msgs if m["role"] == "system"]
    assert any("summary text" in m["content"] for m in sys_msgs)
    # The new user turn is on the session (added by the caller).
    assert sess.messages[-1] == {"role": "user", "content": "next q"}
