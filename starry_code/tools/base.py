from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from starry_code.session import Session


class ToolResult:
    """Result returned by a tool's execute().

    Implemented as a plain class (not @dataclass) because the factory
    classmethods named `ok`/`err` shadow the field names of the same
    identifiers when @dataclass is applied — Python 3.12 raises
    "non-default argument follows default argument" because the
    classmethod object becomes the implicit default for the field.
    A plain class with an explicit __init__ avoids the collision
    while keeping the same public API.
    """

    def __init__(self, ok: bool, content: str) -> None:
        self.ok = ok
        self.content = content

    def __repr__(self) -> str:
        return f"ToolResult(ok={self.ok!r}, content={self.content!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolResult):
            return NotImplemented
        return self.ok == other.ok and self.content == other.content

    @classmethod
    def ok(cls, content: str) -> "ToolResult":
        return cls(True, content)

    @classmethod
    def err(cls, content: str) -> "ToolResult":
        return cls(False, content)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    execute: Callable[[dict, "Session"], ToolResult]
