from agent.tools.search import SearchTool
s = SearchTool()
def test_returns_top_results():
    r = s.execute({"query": "python agent"}, session=None)
    assert r.ok and "1." in r.content
def test_missing_query():
    r = s.execute({}, session=None)
    assert not r.ok