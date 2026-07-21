from agent.context import ContextBuilder
from agent.session import Session
from agent.config import Settings
from agent.memory.embeddings import MockEmbedder
from agent.memory.short_term import InMemoryShortTermStore
from agent.memory.vector_store import LocalVectorStore
from agent.memory.manager import MemoryManager

def make_cb(tmp_path):
    s = Settings(sessions_dir=tmp_path, context_max_messages=6, recent_keep=2)
    mm = MemoryManager(MockEmbedder(), InMemoryShortTermStore(),
                       LocalVectorStore(embedder=MockEmbedder(), path=None))
    return ContextBuilder(memory=mm, settings=s)

def test_build_basic(tmp_path):
    cb = make_cb(tmp_path)
    sess = Session(id="s")
    sess.add_user("hi")
    msgs, tools = cb.build(sess, "what's up?")
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "what's up?"}
    assert isinstance(tools, list)

def test_context_triggers_summary_when_long(tmp_path):
    cb = make_cb(tmp_path)
    sess = Session(id="s")
    for i in range(10):
        sess.add_user(f"u{i}")
        sess.add_assistant(f"a{i}")
    # build with a mock llm; we will pass a stub via the manager
    cb.memory.llm = None  # extractor is irrelevant; summary uses settings-provided LLM via env? Not in builder.
    # Force summary path by setting a no-op LLM via the builder attribute
    from agent.llm import MockLLMClient
    cb.summarizer = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "summary text"}}]}
    ])
    msgs, _ = cb.build(sess, "next q")
    # summary appears as a system message after the first system
    sys_msgs = [m for m in msgs if m["role"] == "system"]
    assert any("summary text" in m["content"] for m in sys_msgs)
