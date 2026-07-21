from starry_code.memory.embeddings import MockEmbedder
from starry_code.memory.short_term import InMemoryShortTermStore
from starry_code.memory.vector_store import LocalVectorStore
from starry_code.memory.manager import MemoryManager
from starry_code.llm import MockLLMClient


def test_recall_returns_top_k():
    m = MemoryManager(embedder=MockEmbedder(), short_term=InMemoryShortTermStore(),
                      vector_store=LocalVectorStore(embedder=MockEmbedder(), path=None), top_k=3)
    m.remember_sid("s1", [
        {"role": "user", "content": "I love drinking green tea in the morning."},
        {"role": "assistant", "content": "Noted."},
    ], llm=MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "[\"user likes green tea in the morning\"]"}}]}
    ]))
    hits = m.recall("s1", "what drink does the user like", top_k=2)
    assert any("green tea" in h[0] for h in hits)


def test_recall_empty_when_no_memory():
    m = MemoryManager(embedder=MockEmbedder(), short_term=InMemoryShortTermStore(),
                      vector_store=LocalVectorStore(embedder=MockEmbedder(), path=None))
    assert m.recall("none", "anything", top_k=3) == []