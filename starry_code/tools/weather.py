from __future__ import annotations
from .base import Tool, ToolResult


_MOCK = {
    "beijing": ("Beijing", "Sunny", "26°C", "NE 12km/h"),
    "shanghai": ("Shanghai", "Cloudy", "23°C", "E 18km/h"),
    "shenzhen": ("Shenzhen", "Rainy", "28°C", "S 20km/h"),
    "hangzhou": ("Hangzhou", "Partly Cloudy", "25°C", "W 8km/h"),
    "new york": ("New York", "Clear", "18°C", "W 15km/h"),
}


class WeatherTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="weather",
            description="Mock current weather for a city. Returns temperature, condition, and wind.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        city = (args or {}).get("city", "").strip()
        if not city:
            return ToolResult.err("city is required")
        key = city.lower()
        if key in _MOCK:
            name, cond, temp, wind = _MOCK[key]
        else:
            name, cond, temp, wind = city.title(), "Clear", "22°C", "N 5km/h"
        return ToolResult.ok(
            f"Weather in {name}: {cond}, {temp}, wind {wind}. (mock data)"
        )