from agent.tools.todo import TodoTool
from agent.session import Session

t = TodoTool()
s = Session(id="x")

def test_add_and_list():
    assert t.execute({"action": "add", "text": "buy milk"}, s).ok
    assert t.execute({"action": "add", "text": "call mom"}, s).ok
    r = t.execute({"action": "list"}, s)
    assert r.ok and "buy milk" in r.content and "call mom" in r.content
    assert s.todos[0]["id"] == 1 and s.todos[1]["id"] == 2

def test_complete():
    r = t.execute({"action": "complete", "id": 1}, s)
    assert r.ok and s.todos[0]["done"] is True

def test_unknown_action():
    r = t.execute({"action": "yolo"}, s)
    assert not r.ok

def test_missing_fields():
    assert not t.execute({"action": "add"}, s).ok
    assert not t.execute({"action": "complete"}, s).ok