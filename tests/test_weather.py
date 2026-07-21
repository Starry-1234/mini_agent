from agent.tools.weather import WeatherTool
w = WeatherTool()
def test_known_city():
    r = w.execute({"city": "beijing"}, session=None)
    assert r.ok and "Beijing" in r.content
def test_unknown_city_falls_back():
    r = w.execute({"city": "Atlantis"}, session=None)
    assert r.ok
def test_missing_city():
    r = w.execute({}, session=None)
    assert not r.ok