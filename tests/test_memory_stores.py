import math
from starry_code.memory.embeddings import MockEmbedder
from starry_code.memory.short_term import InMemoryShortTermStore
from starry_code.memory.vector_store import LocalVectorStore


def test_short_term_recent_order():
    s = InMemoryShortTermStore()
    for i in range(5):
        s.push("sid", {"role": "user", "content": str(i)})
    recent = s.recent("sid", 3)
    assert [r["content"] for r in recent] == ["2", "3", "4"]


def test_local_vector_search():
    e = MockEmbedder(dim=8)
    v = LocalVectorStore(embedder=e, path=None)
    v.upsert("a", "apple pie", None, {})
    v.upsert("b", "banana split", None, {})
    v.upsert("c", "cherry tart", None, {})
    v.upsert("d", "grape juice", None, {})
    res = v.search("banana", top_k=2)
    # search now returns (text, score, meta) — assert on text, not id
    texts = [r[0] for r in res]
    assert "banana split" in texts
    # tuples must be (str, float, dict)
    for r in res:
        assert isinstance(r[0], str)
        assert isinstance(r[1], float)
        assert isinstance(r[2], dict)
