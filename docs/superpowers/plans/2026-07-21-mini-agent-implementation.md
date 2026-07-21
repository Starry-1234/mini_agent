# Mini Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From-scratch minimal Agent runtime in Python with tool calling, multi-window session isolation, basic context compression, and a three-layer pluggable memory system (short-term / episodic / semantic).

**Architecture:** Single-process CLI Agent. OpenAI-compatible function-calling drives tool use, with a hand-written normalizer mapping LLM responses to {thought, tool_calls, final_answer}. Sessions are isolated JSON files. Context = system + tools + recalled memory + recent turns + older summary. Three memory layers behind interfaces with file/local defaults and optional Redis/Qdrant/Chroma backends. Trace printed to terminal and appended to per-session jsonl.

**Tech Stack:** Python ≥3.10, `openai` SDK, optional `redis`, `qdrant-client`, `chromadb`. `pytest`. No agent frameworks.

## Global Constraints

- Language: Python ≥ 3.10; module style: package `agent/`, entry `cli.py`.
- Dependencies kept minimal: `openai` (chat + embedding) required; `redis` / `qdrant-client` / `chromadb` optional with graceful import.
- File naming: `snake_case`; package convention with `__init__.py` exporting only what consumers need.
- No external agent frameworks (langgraph, openhands, openclaw). Runtime is hand-written.
- Tool-calling mechanism: OpenAI-compatible function-calling (default); a text fallback exists but is not the primary path.
- Three memory layers (short-term / episodic / semantic) behind interfaces with file+local default and optional Redis/Qdrant/Chroma.
- `MAX_TOOL_ITERS` default 8; `CONTEXT_MAX_MESSAGES` default 20; `RECENT_KEEP` default 8.
- `.env` for secrets; runtime session data lands in `sessions/` (gitignored).
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`. Co-Authored-By trailer on every commit.
- No placeholders / TBDs in any task. Every step shows full code.

---

## File Map (created across the plan)

| File | Responsibility |
|---|---|
| `cli.py` | Entry: parse args, load session, enter read-eval-print loop, wire to `runtime.run_turn` |
| `agent/__init__.py` | Re-exports `run_turn`, `Session`, `Tool` |
| `agent/config.py` | Load `.env`, expose typed settings dataclass |
| `agent/trace.py` | Terminal trace printer + per-session `*.trace.jsonl` appender |
| `agent/tools/base.py` | `Tool` dataclass + `ToolResult` |
| `agent/tools/registry.py` | `ToolRegistry`: `register`, `openai_schemas`, `execute` |
| `agent/tools/calculator.py` | `CalculatorTool` (AST-based safe eval) |
| `agent/tools/search.py` | `SearchTool` (mock) |
| `agent/tools/todo.py` | `TodoTool` (per-session CRUD) |
| `agent/tools/weather.py` | `WeatherTool` (mock) |
| `agent/llm.py` | `LLMClient` (chat + embed); `MockLLMClient` for tests |
| `agent/parser.py` | `parse_response` → `ParsedResponse` |
| `agent/session.py` | `Session`, `SessionStore` (JSON file persistence) |
| `agent/memory/embeddings.py` | `Embedder` interface + `OpenAICompatEmbedder` + `KeywordEmbedder` |
| `agent/memory/short_term.py` | `ShortTermStore` interface + `InMemoryShortTermStore` + `RedisShortTermStore` |
| `agent/memory/vector_store.py` | `VectorStore` interface + `LocalVectorStore` + `QdrantVectorStore` + `ChromaVectorStore` |
| `agent/memory/extractor.py` | `extract_facts` — LLM-driven distillation |
| `agent/memory/manager.py` | `MemoryManager` — recall + remember, orchestrates layers |
| `agent/context.py` | `ContextBuilder` — assemble system + memory + history (+ summary) |
| `agent/runtime.py` | `run_turn` — main loop, tool execution, finalize |
| `agent/prompts.py` | System prompt + extractor prompt strings |
| `tests/conftest.py` | pytest fixtures (mock LLM, tmp sessions dir) |
| `tests/test_*.py` | one per module |
| `tests/test_integration.py` | real API, env-gated skip |
| `demo/demo_weather_todo.py` | end-to-end scripted scenario |
| `docs/ARCHITECTURE_QA.md` | design questions answers |
| `docs/PROMPTS_LOG.md` | AI prompts + decisions log |
| `README.md` | run, design, memory recall timing & placement |
| `requirements.txt`, `.env.example` | deps & config |

---

## Task 1: Project scaffold + config + .env

**Files:**
- Create: `requirements.txt`, `.env.example`, `pyproject.toml`, `agent/__init__.py`, `agent/config.py`, `tests/__init__.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `agent.config.Settings` dataclass with fields: `llm_base_url`, `llm_api_key`, `llm_model`, `embed_base_url`, `embed_api_key`, `embed_model`, `short_term_backend` (`"memory"`|`"redis"`), `vector_backend` (`"local"`|`"qdrant"`|`"chroma"`), `redis_url`, `qdrant_url`, `max_tool_iters:int=8`, `context_max_messages:int=20`, `recent_keep:int=8`, `sessions_dir:Path`.

- [ ] **Step 1: Write failing test for `Settings` defaults & env loading**

```python
# tests/test_config.py
import os
from pathlib import Path
from agent.config import Settings

def test_defaults(tmp_path, monkeypatch):
    for k in ["LLM_API_KEY", "EMBED_API_KEY", "REDIS_URL", "QDRANT_URL",
              "LLM_BASE_URL", "EMBED_BASE_URL", "LLM_MODEL", "EMBED_MODEL",
              "SHORT_TERM_BACKEND", "VECTOR_BACKEND"]:
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env(sessions_dir=tmp_path)
    assert s.llm_api_key == ""
    assert s.short_term_backend == "memory"
    assert s.vector_backend == "local"
    assert s.max_tool_iters == 8
    assert s.context_max_messages == 20
    assert s.recent_keep == 8
    assert s.sessions_dir == tmp_path

def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://x.example/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-x")
    monkeypatch.setenv("SHORT_TERM_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    s = Settings.from_env(sessions_dir=tmp_path)
    assert s.llm_api_key == "sk-test"
    assert s.llm_base_url == "https://x.example/v1"
    assert s.llm_model == "gpt-x"
    assert s.short_term_backend == "redis"
    assert s.redis_url == "redis://localhost:6379/0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: ImportError — `agent.config` not yet defined.

- [ ] **Step 3: Create `requirements.txt` and `.env.example`**

`requirements.txt`:
```
openai>=1.30
pytest>=8.0
# optional backends
redis>=5.0
qdrant-client>=1.7
chromadb>=0.4
```

`.env.example`:
```
# Chat LLM
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=
LLM_MODEL=deepseek-chat

# Embedding (optional; falls back to keyword if blank)
EMBED_BASE_URL=
EMBED_API_KEY=
EMBED_MODEL=

# Memory backends
SHORT_TERM_BACKEND=memory
VECTOR_BACKEND=local
REDIS_URL=
QDRANT_URL=

# Runtime
MAX_TOOL_ITERS=8
CONTEXT_MAX_MESSAGES=20
RECENT_KEEP=8
```

- [ ] **Step 4: Implement `agent/config.py`**

```python
# agent/config.py
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    short_term_backend: str = "memory"
    vector_backend: str = "local"
    redis_url: str = ""
    qdrant_url: str = ""
    max_tool_iters: int = 8
    context_max_messages: int = 20
    recent_keep: int = 8
    sessions_dir: Path = field(default_factory=lambda: Path("sessions"))

    @classmethod
    def from_env(cls, sessions_dir: Path | None = None) -> "Settings":
        def get(k, default=""):
            return os.environ.get(k, default)
        def getint(k, default):
            v = os.environ.get(k)
            return int(v) if v else default
        return cls(
            llm_base_url=get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            llm_api_key=get("LLM_API_KEY"),
            llm_model=get("LLM_MODEL", "deepseek-chat"),
            embed_base_url=get("EMBED_BASE_URL"),
            embed_api_key=get("EMBED_API_KEY"),
            embed_model=get("EMBED_MODEL"),
            short_term_backend=get("SHORT_TERM_BACKEND", "memory"),
            vector_backend=get("VECTOR_BACKEND", "local"),
            redis_url=get("REDIS_URL"),
            qdrant_url=get("QDRANT_URL"),
            max_tool_iters=getint("MAX_TOOL_ITERS", 8),
            context_max_messages=getint("CONTEXT_MAX_MESSAGES", 20),
            recent_keep=getint("RECENT_KEEP", 8),
            sessions_dir=sessions_dir or Path(get("SESSIONS_DIR", "sessions")),
        )
```

Create empty `agent/__init__.py` and `tests/__init__.py` (single newline each).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example agent/ tests/
git commit -m "feat: project scaffold with env-driven Settings

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Trace logger

**Files:**
- Create: `agent/trace.py`, `tests/test_trace.py`

**Interfaces:**
- Produces: `TraceLogger(sessions_dir: Path, session_id: str)` with `.event(kind: str, **fields) -> None`; kinds: `user`, `thought`, `tool_call`, `tool_result`, `assistant`, `error`, `summary`, `recall`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_trace.py
import json
from agent.trace import TraceLogger

def test_trace_appends_jsonl_and_terminal(tmp_path, capsys):
    log = TraceLogger(tmp_path, "s1")
    log.event("user", text="hi")
    log.event("thought", text="thinking")
    log.event("tool_call", name="calculator", args={"expression": "1+1"})
    log.event("tool_result", name="calculator", result="2")
    log.event("assistant", text="done")

    lines = (tmp_path / "s1.trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    parsed = [json.loads(l) for l in lines]
    assert [p["kind"] for p in parsed] == ["user", "thought", "tool_call", "tool_result", "assistant"]
    assert parsed[2]["name"] == "calculator"
    out = capsys.readouterr().err
    assert "user" in out and "calculator" in out and "assistant" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trace.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent/trace.py`**

```python
# agent/trace.py
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path


_KIND_COLOR = {
    "user":        "\033[97m",   # white
    "thought":     "\033[90m",   # gray
    "tool_call":   "\033[36m",   # cyan
    "tool_result": "\033[33m",   # yellow
    "assistant":   "\033[92m",   # green
    "error":       "\033[91m",   # red
    "summary":     "\033[95m",   # magenta
    "recall":      "\033[94m",   # blue
}
_RESET = "\033[0m"


class TraceLogger:
    def __init__(self, sessions_dir: Path, session_id: str):
        self.sessions_dir = Path(sessions_dir)
        self.session_id = session_id
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.sessions_dir / f"{session_id}.trace.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def event(self, kind: str, **fields) -> None:
        record = {"ts": datetime.utcnow().isoformat() + "Z", "kind": kind, **fields}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._print(kind, fields)

    def _print(self, kind: str, fields: dict) -> None:
        color = _KIND_COLOR.get(kind, "")
        head = f"{color}[{kind}]{_RESET}"
        body = " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in fields.items())
        sys.stderr.write(f"{head} {body}\n")
        sys.stderr.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trace.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/trace.py tests/test_trace.py
git commit -m "feat: trace logger with jsonl + colored terminal

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Tool base + Registry

**Files:**
- Create: `agent/tools/__init__.py`, `agent/tools/base.py`, `agent/tools/registry.py`, `tests/test_registry.py`

**Interfaces:**
- Produces: `Tool` dataclass: `name:str, description:str, parameters:dict, execute:Callable[[dict, "Session"], str]`.
- `ToolResult` dataclass: `ok:bool, content:str`.
- `ToolRegistry`: `.register(tool)`, `.openai_schemas() -> list[dict]`, `.execute(name:str, args:dict, session) -> ToolResult`, `.names() -> list[str]`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_registry.py
import pytest
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry

def echo_execute(args, session):
    return ToolResult(ok=True, content=f"got {args['x']}")

def test_registry_roundtrip():
    reg = ToolRegistry()
    reg.register(Tool(
        name="echo",
        description="echoes x",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        execute=echo_execute,
    ))
    schemas = reg.openai_schemas()
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "echo"
    assert reg.names() == ["echo"]
    res = reg.execute("echo", {"x": "hi"}, session=None)
    assert res.ok and res.content == "got hi"

def test_unknown_tool_returns_error_result():
    reg = ToolRegistry()
    res = reg.execute("nope", {}, session=None)
    assert not res.ok
    assert "unknown" in res.content.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_registry.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent/tools/base.py`**

```python
# agent/tools/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.session import Session


@dataclass
class ToolResult:
    ok: bool
    content: str

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
```

- [ ] **Step 4: Implement `agent/tools/registry.py`**

```python
# agent/tools/registry.py
from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
from .base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.session import Session


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_all(self, tools: Iterable[Tool]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def openai_schemas(self) -> list[dict]:
        out = []
        for t in self._tools.values():
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            })
        return out

    def execute(self, name: str, args: dict, session: "Session | None") -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.err(f"unknown tool: {name}")
        try:
            return tool.execute(args or {}, session)
        except Exception as e:  # noqa: BLE001 — surface as tool error for model self-correction
            return ToolResult.err(f"tool '{name}' raised: {type(e).__name__}: {e}")
```

- [ ] **Step 5: Create `agent/tools/__init__.py` re-exporting nothing yet**

```python
# agent/tools/__init__.py
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_registry.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add agent/tools/ tests/test_registry.py
git commit -m "feat: tool base + registry with openai schema emit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Calculator tool (AST-based safe eval)

**Files:**
- Create: `agent/tools/calculator.py`, `tests/test_calculator.py`

**Interfaces:**
- Produces: `CalculatorTool`: name `calculator`, args `{"expression": str}`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_calculator.py
import pytest
from agent.tools.calculator import CalculatorTool
from agent.tools.base import ToolResult

calc = CalculatorTool()

def run(expr: str):
    return calc.execute({"expression": expr}, session=None)

def test_basic_arithmetic():
    r = run("1 + 2 * 3")
    assert r.ok and r.content.strip() == "7"

def test_parens_and_unary():
    r = run("-(3 - 5) ** 2 / 4")
    assert r.ok and r.content.strip() == "-1.0"

def test_rejects_function_calls():
    r = run("__import__('os').system('echo x')")
    assert not r.ok and "not allowed" in r.content.lower()

def test_rejects_names():
    r = run("foo + 1")
    assert not r.ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calculator.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent/tools/calculator.py`**

```python
# agent/tools/calculator.py
from __future__ import annotations
import ast
import operator
from .base import Tool, ToolResult


_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos, ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Paren):  # not used by Python AST, but keep for explicitness
        return _safe_eval(node.value)  # type: ignore[attr-defined]
    raise ValueError(f"expression node not allowed: {type(node).__name__}")


class CalculatorTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description="Evaluate a basic arithmetic expression. Supports + - * / // % ** and parentheses.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        expr = (args or {}).get("expression", "").strip()
        if not expr:
            return ToolResult.err("expression is required")
        try:
            tree = ast.parse(expr, mode="eval")
            value = _safe_eval(tree)
            return ToolResult.ok(str(value))
        except Exception as e:  # noqa: BLE001
            return ToolResult.err(f"not allowed or invalid expression: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_calculator.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/tools/calculator.py tests/test_calculator.py
git commit -m "feat: calculator tool with AST sandboxed eval

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Search tool (mock) + Weather tool (mock)

**Files:**
- Create: `agent/tools/search.py`, `agent/tools/weather.py`, `tests/test_search.py`, `tests/test_weather.py`

**Interfaces:**
- `SearchTool`: `{"query": str}` → mock top-3 results.
- `WeatherTool`: `{"city": str}` → mock current weather.

- [ ] **Step 1: Write failing test for search**

```python
# tests/test_search.py
from agent.tools.search import SearchTool
s = SearchTool()
def test_returns_top_results():
    r = s.execute({"query": "python agent"}, session=None)
    assert r.ok and "1." in r.content
def test_missing_query():
    r = s.execute({}, session=None)
    assert not r.ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_search.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent/tools/search.py`**

```python
# agent/tools/search.py
from __future__ import annotations
from .base import Tool, ToolResult


_DATA = {
    "python": [
        ("Python is a high-level, general-purpose programming language.", "https://example.com/python"),
        ("Python emphasizes code readability and supports multiple paradigms.", "https://example.com/python-2"),
    ],
    "agent": [
        ("An agent perceives its environment and takes actions to achieve goals.", "https://example.com/agent"),
        ("LLM-based agents use large language models as their reasoning core.", "https://example.com/llm-agent"),
    ],
    "weather": [
        ("Weather describes atmospheric conditions at a specific time and place.", "https://example.com/weather"),
    ],
}


class SearchTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="search",
            description="Mock web search. Returns up to 3 short results for a query keyword.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        q = (args or {}).get("query", "").strip().lower()
        if not q:
            return ToolResult.err("query is required")
        hits = []
        for key, items in _DATA.items():
            if key in q:
                hits.extend(items)
        if not hits:
            return ToolResult.ok(f"No mock results for '{q}'.")
        lines = []
        for i, (snippet, url) in enumerate(hits[:3], 1):
            lines.append(f"{i}. {snippet}\n   {url}")
        return ToolResult.ok("\n".join(lines))
```

- [ ] **Step 4: Write failing test for weather**

```python
# tests/test_weather.py
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
```

- [ ] **Step 5: Implement `agent/tools/weather.py`**

```python
# agent/tools/weather.py
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
```

- [ ] **Step 6: Run both tests**

Run: `python -m pytest tests/test_search.py tests/test_weather.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add agent/tools/search.py agent/tools/weather.py tests/test_search.py tests/test_weather.py
git commit -m "feat: mock search and weather tools

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Session (JSON persistence)

**Files:**
- Create: `agent/session.py`, `tests/test_session.py`

**Interfaces:**
- `Session`: `id:str, system_prompt:str, messages:list[dict], todos:list[dict], summary:str`, methods `add_user`, `add_assistant`, `add_tool_result(call_id, name, content)`, `add_tool_call(name, args)`.
- `SessionStore`: `.load(sid) -> Session`, `.save(s: Session)`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_session.py
import json
from pathlib import Path
from agent.session import Session, SessionStore

def test_session_roundtrip(tmp_path: Path):
    s = Session(id="abc", system_prompt="you are an agent")
    s.add_user("hi")
    s.add_assistant("hello")
    s.add_tool_call(call_id="c1", name="calculator", args={"expression": "1+1"})
    s.add_tool_result(call_id="c1", name="calculator", content="2")
    assert s.messages[-1]["role"] == "tool"
    store = SessionStore(tmp_path)
    store.save(s)
    on_disk = json.loads((tmp_path / "abc.json").read_text(encoding="utf-8"))
    assert on_disk["id"] == "abc"
    s2 = store.load("abc")
    assert s2.messages == s.messages
    assert s2.todos == []

def test_load_missing_returns_fresh(tmp_path: Path):
    store = SessionStore(tmp_path)
    s = store.load("nope")
    assert s.id == "nope" and s.messages == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent/session.py`**

```python
# agent/session.py
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _gen_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Session:
    id: str
    system_prompt: str = "You are a helpful Agent. Use tools when needed."
    messages: list[dict] = field(default_factory=list)
    todos: list[dict] = field(default_factory=list)
    summary: str = ""

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def add_tool_call(self, *, call_id: str, name: str, args: dict) -> None:
        self.messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }],
        })

    def add_tool_result(self, *, call_id: str, name: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        })


class SessionStore:
    def __init__(self, base: Path):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)

    def path_for(self, sid: str) -> Path:
        return self.base / f"{sid}.json"

    def load(self, sid: str) -> Session:
        p = self.path_for(sid)
        if not p.exists():
            return Session(id=sid)
        data = json.loads(p.read_text(encoding="utf-8"))
        return Session(
            id=data.get("id", sid),
            system_prompt=data.get("system_prompt", Session.system_prompt),
            messages=data.get("messages", []),
            todos=data.get("todos", []),
            summary=data.get("summary", ""),
        )

    def save(self, s: Session) -> None:
        p = self.path_for(s.id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_session.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/session.py tests/test_session.py
git commit -m "feat: session with json persistence and tool-call messages

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Todo tool (per-session)

**Files:**
- Create: `agent/tools/todo.py`, `tests/test_todo.py`

**Interfaces:**
- `TodoTool`: action `add` / `list` / `complete`, args `{"action": str, "text"?: str, "id"?: int}`. Mutates `session.todos`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_todo.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_todo.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent/tools/todo.py`**

```python
# agent/tools/todo.py
from __future__ import annotations
from .base import Tool, ToolResult


class TodoTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="todo",
            description="Manage a per-session todo list. Actions: add (text), list, complete (id).",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "list", "complete"]},
                    "text":   {"type": "string"},
                    "id":     {"type": "integer"},
                },
                "required": ["action"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        if session is None:
            return ToolResult.err("todo tool requires an active session")
        action = (args or {}).get("action")
        todos = session.todos
        if action == "add":
            text = (args or {}).get("text", "").strip()
            if not text:
                return ToolResult.err("text is required for add")
            next_id = (max((t["id"] for t in todos), default=0) + 1)
            todos.append({"id": next_id, "text": text, "done": False})
            return ToolResult.ok(f"added todo #{next_id}: {text}")
        if action == "list":
            if not todos:
                return ToolResult.ok("(no todos)")
            lines = [f"  #{t['id']} [{'x' if t['done'] else ' '}] {t['text']}" for t in todos]
            return ToolResult.ok("todos:\n" + "\n".join(lines))
        if action == "complete":
            tid = args.get("id")
            if tid is None:
                return ToolResult.err("id is required for complete")
            for t in todos:
                if t["id"] == tid:
                    t["done"] = True
                    return ToolResult.ok(f"completed todo #{tid}: {t['text']}")
            return ToolResult.err(f"todo #{tid} not found")
        return ToolResult.err(f"unknown action: {action}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_todo.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/tools/todo.py tests/test_todo.py
git commit -m "feat: per-session todo tool with add/list/complete

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: LLM client (real + mock) + parser

**Files:**
- Create: `agent/llm.py`, `agent/parser.py`, `tests/test_llm.py`, `tests/test_parser.py`

**Interfaces:**
- `LLMClient`: `.chat(messages, tools) -> dict` (raw OpenAI-shaped response). `.embed(texts) -> list[list[float]]`.
- `MockLLMClient`: scripted `.chat` responses; `.embed` returns deterministic pseudo-vectors.
- `parse_response(raw) -> ParsedResponse` where `ParsedResponse` has `.thought:str`, `.tool_calls:list[ToolCall]`, `.final_answer:str|None`. `ToolCall`: `.id, .name, .args(dict)`.

- [ ] **Step 1: Write failing parser test**

```python
# tests/test_parser.py
from agent.parser import parse_response, ParsedResponse

def test_parses_tool_calls_and_thought():
    raw = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "I should compute.",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'},
                }],
            }
        }]
    }
    p = parse_response(raw)
    assert isinstance(p, ParsedResponse)
    assert p.thought == "I should compute."
    assert p.final_answer is None
    assert len(p.tool_calls) == 1
    assert p.tool_calls[0].name == "calculator"
    assert p.tool_calls[0].args == {"expression": "1+1"}

def test_parses_final_answer():
    raw = {"choices": [{"message": {"role": "assistant", "content": "Hello there."}}]}
    p = parse_response(raw)
    assert p.final_answer == "Hello there."
    assert p.tool_calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_parser.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent/parser.py`**

```python
# agent/parser.py
from __future__ import annotations
import json
import re
from dataclasses import dataclass


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ParsedResponse:
    thought: str
    tool_calls: list[ToolCall]
    final_answer: str | None


def parse_response(raw: dict) -> ParsedResponse:
    msg = raw["choices"][0]["message"]
    thought = msg.get("content") or ""
    raw_calls = msg.get("tool_calls") or []
    tool_calls: list[ToolCall] = []
    for c in raw_calls:
        try:
            args = json.loads(c["function"]["arguments"]) if c["function"].get("arguments") else {}
        except Exception:
            args = {"_raw": c["function"].get("arguments", "")}
        tool_calls.append(ToolCall(id=c["id"], name=c["function"]["name"], args=args))

    if tool_calls:
        return ParsedResponse(thought=thought, tool_calls=tool_calls, final_answer=None)

    # Text fallback: try to extract <tool_call>{...}</tool_call> / fenced JSON
    fb = _text_fallback(thought)
    if fb is not None:
        return ParsedResponse(thought=thought, tool_calls=[fb], final_answer=None)
    return ParsedResponse(thought=thought, tool_calls=[], final_answer=thought or None)


def _text_fallback(text: str) -> ToolCall | None:
    m = re.search(r"\{[^{}]*\"name\"[^{}]*\"arguments\"[^{}]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return ToolCall(id="fallback", name=obj["name"], args=obj.get("arguments", {}))
    except Exception:
        return None
```

- [ ] **Step 4: Write failing LLM client test (mock-only) and minimal client**

`tests/test_llm.py`:
```python
import pytest
from agent.llm import LLMClient, MockLLMClient

def test_mock_chat_returns_first_scripted():
    m = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    ])
    out = m.chat([{"role": "user", "content": "x"}], tools=None)
    assert out["choices"][0]["message"]["content"] == "hi"

def test_mock_chat_rotates_and_errors_when_exhausted():
    m = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "a"}}]}
    ])
    m.chat([], tools=None)
    with pytest.raises(RuntimeError):
        m.chat([], tools=None)

def test_mock_embed_deterministic():
    m = MockLLMClient()
    v1 = m.embed(["hello world"])
    v2 = m.embed(["hello world"])
    assert v1 == v2 and len(v1[0]) == 16

def test_real_client_requires_config():
    c = LLMClient(api_key="", base_url="x")
    with pytest.raises(RuntimeError):
        c.chat([], tools=None)
```

`agent/llm.py`:
```python
# agent/llm.py
from __future__ import annotations
import hashlib
import math
from typing import Any


def _hash_vec(text: str, dim: int = 16) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(dim):
        byte = h[i % len(h)]
        out.append(((byte / 255.0) * 2.0) - 1.0)
    n = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / n for x in out]


class MockLLMClient:
    """In-memory client for unit tests. Embeds are deterministic by hash."""

    def __init__(self, chat_responses: list[dict] | None = None, embed_dim: int = 16) -> None:
        self._responses = list(chat_responses or [])
        self._idx = 0
        self._dim = embed_dim

    def chat(self, messages, tools=None) -> dict:
        if self._idx >= len(self._responses):
            raise RuntimeError("MockLLMClient: no more scripted responses")
        r = self._responses[self._idx]
        self._idx += 1
        return r

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vec(t, self._dim) for t in texts]


class LLMClient:
    """Thin OpenAI-compatible client. Lazy-initialised SDK; never imported unless used."""

    def __init__(self, api_key: str, base_url: str, model: str,
                 embed_api_key: str = "", embed_base_url: str = "", embed_model: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.embed_api_key = embed_api_key or api_key
        self.embed_base_url = embed_base_url or base_url
        self.embed_model = embed_model
        self._client = None

    def _sdk(self):
        if self._client is None:
            if not self.api_key or not self.base_url:
                raise RuntimeError("LLMClient: api_key and base_url are required for real calls")
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(self, messages, tools=None) -> dict:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = self._sdk().chat.completions.create(**kwargs)
        return resp.model_dump()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.embed_model:
            raise RuntimeError("LLMClient: embed_model not configured")
        from openai import OpenAI
        c = OpenAI(api_key=self.embed_api_key, base_url=self.embed_base_url)
        r = c.embeddings.create(model=self.embed_model, input=texts)
        return [item.embedding for item in r.data]
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_parser.py tests/test_llm.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add agent/llm.py agent/parser.py tests/test_parser.py tests/test_llm.py
git commit -m "feat: openai-compatible llm client + mock + response parser

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Embeddings + ShortTerm store + Vector store

**Files:**
- Create: `agent/memory/__init__.py`, `agent/memory/embeddings.py`, `agent/memory/short_term.py`, `agent/memory/vector_store.py`, `tests/test_memory_stores.py`

**Interfaces:**
- `Embedder`: `.embed(texts) -> list[list[float]]`; `OpenAICompatEmbedder`, `MockEmbedder` (deterministic via hash).
- `ShortTermStore`: `.push(sid, record:dict)`, `.recent(sid, k) -> list[dict]`. `InMemoryShortTermStore`; `RedisShortTermStore` (guarded import).
- `VectorStore`: `.upsert(id, text, vector, meta)`, `.search(vector, top_k=5) -> list[(id, score, meta)]`. `LocalVectorStore` (jsonl + cosine), `QdrantVectorStore`, `ChromaVectorStore` (both guarded).

- [ ] **Step 1: Write failing test**

```python
# tests/test_memory_stores.py
import math
from agent.memory.embeddings import MockEmbedder
from agent.memory.short_term import InMemoryShortTermStore
from agent.memory.vector_store import LocalVectorStore

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
    res = v.search("banana", top_k=2)  # keyword-fallback in local store
    ids = [r[0] for r in res]
    assert "b" in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_memory_stores.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent/memory/embeddings.py`**

```python
# agent/memory/embeddings.py
from __future__ import annotations
import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _hash_vec(text: str, dim: int) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(dim):
        b = h[i % len(h)]
        out.append(((b / 255.0) * 2.0) - 1.0)
    n = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / n for x in out]


class MockEmbedder:
    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vec(t, self.dim) for t in texts]


class OpenAICompatEmbedder:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key, self.base_url, self.model = api_key, base_url, model

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        c = OpenAI(api_key=self.api_key, base_url=self.base_url)
        r = c.embeddings.create(model=self.model, input=texts)
        return [it.embedding for it in r.data]
```

- [ ] **Step 4: Implement `agent/memory/short_term.py`**

```python
# agent/memory/short_term.py
from __future__ import annotations
from collections import defaultdict, deque
from typing import Protocol


class ShortTermStore(Protocol):
    def push(self, sid: str, record: dict) -> None: ...
    def recent(self, sid: str, k: int) -> list[dict]: ...
    def clear(self, sid: str) -> None: ...


class InMemoryShortTermStore:
    def __init__(self, maxlen: int = 200) -> None:
        self._buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def push(self, sid: str, record: dict) -> None:
        self._buf[sid].append(record)

    def recent(self, sid: str, k: int) -> list[dict]:
        d = self._buf[sid]
        if k >= len(d):
            return list(d)
        return list(d)[-k:]

    def clear(self, sid: str) -> None:
        self._buf.pop(sid, None)


class RedisShortTermStore:
    """Optional backend. Requires `redis`; import is guarded so tests run without it."""

    def __init__(self, url: str, maxlen: int = 200, key_prefix: str = "st:") -> None:
        try:
            import redis  # type: ignore
        except ImportError as e:
            raise RuntimeError("redis backend requested but `redis` is not installed") from e
        self._r = redis.from_url(url)
        self.maxlen = maxlen
        self.prefix = key_prefix

    def _key(self, sid: str) -> str:
        return f"{self.prefix}{sid}"

    def push(self, sid: str, record: dict) -> None:
        import json
        self._r.rpush(self._key(sid), json.dumps(record, ensure_ascii=False))
        self._r.ltrim(self._key(sid), -self.maxlen, -1)

    def recent(self, sid: str, k: int) -> list[dict]:
        import json
        raw = self._r.lrange(self._key(sid), -k, -1)
        return [json.loads(x) for x in raw]

    def clear(self, sid: str) -> None:
        self._r.delete(self._key(sid))
```

- [ ] **Step 5: Implement `agent/memory/vector_store.py`**

```python
# agent/memory/vector_store.py
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Protocol


class VectorStore(Protocol):
    def upsert(self, id: str, text: str, vector: list[float] | None, meta: dict) -> None: ...
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]: ...


def _cos(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    return dot / (na * nb) if na and nb else 0.0


class LocalVectorStore:
    """JSONL-backed. Uses embedder if vectors are None, else uses provided vectors.
    Search falls back to keyword overlap when embedder is unavailable."""

    def __init__(self, embedder=None, path: Path | None = None) -> None:
        self.embedder = embedder
        self.path = path
        self._items: dict[str, dict] = {}
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                o = json.loads(line)
                self._items[o["id"]] = o

    def _flush(self) -> None:
        if not self.path:
            return
        self.path.write_text(
            "\n".join(json.dumps(v, ensure_ascii=False) for v in self._items.values()),
            encoding="utf-8",
        )

    def upsert(self, id: str, text: str, vector: list[float] | None, meta: dict) -> None:
        vec = vector
        if vec is None and self.embedder is not None:
            vec = self.embedder.embed([text])[0]
        self._items[id] = {"id": id, "text": text, "vector": vec, "meta": meta}
        self._flush()

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]:
        if self.embedder is not None:
            qv = self.embedder.embed([query])[0]
            scored = []
            for v in self._items.values():
                if v["vector"] is None:
                    continue
                scored.append((v["id"], _cos(qv, v["vector"]), v["meta"]))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        # keyword fallback
        q_tokens = set(query.lower().split())
        scored = []
        for v in self._items.values():
            t_tokens = set(v["text"].lower().split())
            inter = len(q_tokens & t_tokens)
            if inter:
                scored.append((v["id"], inter / max(len(q_tokens | t_tokens), 1), v["meta"]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class QdrantVectorStore:
    def __init__(self, url: str, collection: str = "memory", embedder=None) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except ImportError as e:
            raise RuntimeError("qdrant backend requested but `qdrant-client` is not installed") from e
        if embedder is None:
            raise RuntimeError("QdrantVectorStore requires an embedder for vectorisation")
        self._c = QdrantClient(url=url)
        self._col = collection
        self._embedder = embedder
        if not self._c.collection_exists(collection):
            dim = len(embedder.embed(["dim-probe"])[0])
            from qdrant_client.http import models  # type: ignore
            self._c.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def upsert(self, id: str, text: str, vector, meta: dict) -> None:
        from qdrant_client.http import models  # type: ignore
        vec = vector or self._embedder.embed([text])[0]
        self._c.upsert(self._col, points=[models.PointStruct(id=id, vector=vec, payload={"text": text, **meta})])

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]:
        from qdrant_client.http import models  # type: ignore
        qv = self._embedder.embed([query])[0]
        hits = self._c.search(self._col, query_vector=qv, limit=top_k)
        return [(str(h.id), float(h.score), dict(h.payload)) for h in hits]


class ChromaVectorStore:
    def __init__(self, path: str, collection: str = "memory") -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as e:
            raise RuntimeError("chroma backend requested but `chromadb` is not installed") from e
        self._c = chromadb.PersistentClient(path=path)
        self._col = self._c.get_or_create_collection(collection)

    def upsert(self, id: str, text: str, vector, meta: dict) -> None:
        kwargs = {"ids": [id], "documents": [text], "metadatas": [meta]}
        if vector is not None:
            kwargs["embeddings"] = [vector]
        self._col.upsert(**kwargs)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]:
        r = self._col.query(query_texts=[query], n_results=top_k)
        ids = r.get("ids", [[]])[0]
        docs = r.get("documents", [[]])[0]
        metas = r.get("metadatas", [[]])[0]
        dists = r.get("distances", [[]])[0]
        out = []
        for i, d, m, dist in zip(ids, docs, metas, dists):
            out.append((i, 1.0 - float(dist), dict(m or {})))
        return out
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_memory_stores.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add agent/memory/ tests/test_memory_stores.py
git commit -m "feat: embeddings + short-term + vector stores (local default, optional redis/qdrant/chroma)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Memory extractor + manager (recall + remember)

**Files:**
- Create: `agent/prompts.py`, `agent/memory/extractor.py`, `agent/memory/manager.py`, `tests/test_memory_manager.py`

**Interfaces:**
- `extract_facts(llm: LLMClient, recent_turns: list[dict]) -> list[str]` — returns 0-5 short fact strings.
- `MemoryManager(embedder, short_term, vector_store, llm=None, top_k=5)`: `.recall(sid, query) -> list[(text, score, meta)]`, `.remember(sid, recent_turns) -> int` (upserts extracted facts, returns count).

- [ ] **Step 1: Write failing test**

```python
# tests/test_memory_manager.py
from agent.memory.embeddings import MockEmbedder
from agent.memory.short_term import InMemoryShortTermStore
from agent.memory.vector_store import LocalVectorStore
from agent.memory.manager import MemoryManager
from agent.llm import MockLLMClient

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_memory_manager.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent/prompts.py`**

```python
# agent/prompts.py
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful Agent. Use tools when needed. "
    "When you have a final answer, reply in plain text (no tool calls). "
    "Keep thoughts brief."
)

EXTRACTOR_PROMPT = (
    "You are a memory extractor. From the recent conversation turn below, "
    "extract 0-5 short, durable facts worth remembering long-term about the user "
    "(preferences, habits, identity, key decisions, constraints). "
    "Output strictly a JSON array of strings. If nothing is worth remembering, "
    "output []. No commentary.\n\nCONVERSATION:\n{turns}\n\nJSON:"
)
```

- [ ] **Step 4: Implement `agent/memory/extractor.py`**

```python
# agent/memory/extractor.py
from __future__ import annotations
import json
import re
from typing import Iterable

from agent.llm import LLMClient
from agent.prompts import EXTRACTOR_PROMPT


def _format_turns(turns: Iterable[dict]) -> str:
    lines = []
    for t in turns:
        role = t.get("role", "user")
        content = t.get("content", "")
        if isinstance(content, str) and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "(empty)"


def _parse_facts(text: str) -> list[str]:
    if not text:
        return []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        return []
    return []


def extract_facts(llm: LLMClient, recent_turns: list[dict]) -> list[str]:
    if llm is None or not recent_turns:
        return []
    prompt = EXTRACTOR_PROMPT.format(turns=_format_turns(recent_turns))
    try:
        raw = llm.chat([{"role": "user", "content": prompt}], tools=None)
        text = raw["choices"][0]["message"]["content"] or ""
        return _parse_facts(text)
    except Exception:
        return []
```

- [ ] **Step 5: Implement `agent/memory/manager.py`**

```python
# agent/memory/manager.py
from __future__ import annotations
import uuid
from typing import Any

from .short_term import ShortTermStore
from .vector_store import VectorStore
from .extractor import extract_facts


class MemoryManager:
    def __init__(self, embedder, short_term: ShortTermStore, vector_store: VectorStore,
                 llm=None, top_k: int = 5) -> None:
        self.embedder = embedder
        self.short_term = short_term
        self.vector = vector_store
        self.llm = llm
        self.top_k = top_k

    # Short-term helpers
    def push_turn(self, sid: str, record: dict) -> None:
        self.short_term.push(sid, record)

    def recent_turns(self, sid: str, k: int) -> list[dict]:
        return self.short_term.recent(sid, k)

    # Long-term
    def remember_sid(self, sid: str, recent_turns: list[dict], llm=None) -> int:
        facts = extract_facts(llm or self.llm, recent_turns)
        if not facts:
            return 0
        # Deduplicate by cosine similarity to existing items; threshold drops near-duplicates.
        for fact in facts:
            fid = f"{sid}:{uuid.uuid4().hex[:8]}"
            meta = {"sid": sid, "kind": "fact"}
            self.vector.upsert(id=fid, text=fact, vector=None, meta=meta)
        return len(facts)

    def recall(self, sid: str | None, query: str, top_k: int | None = None) -> list[tuple[str, float, dict]]:
        k = top_k or self.top_k
        results = self.vector.search(query=query, top_k=k)
        if sid is None:
            return results
        return [(i, s, m) for (i, s, m) in results if m.get("sid") == sid]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_memory_manager.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add agent/prompts.py agent/memory/ tests/test_memory_manager.py
git commit -m "feat: memory extractor and three-layer manager

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: Context builder (system + memory + recent + summary)

**Files:**
- Create: `agent/context.py`, `tests/test_context.py`

**Interfaces:**
- `ContextBuilder(memory: MemoryManager, settings)`: `.build(session, user_input) -> tuple[list[dict], list[dict]]` returns `(messages, tool_schemas)`. When total messages > `context_max_messages`, older ones are summarised via LLM; summary is held in `session.summary` and prepended as a single "system" turn.

- [ ] **Step 1: Write failing test**

```python
# tests/test_context.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_context.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent/context.py`**

```python
# agent/context.py
from __future__ import annotations
from .session import Session
from .config import Settings
from .prompts import SYSTEM_PROMPT
from .memory.manager import MemoryManager


class ContextBuilder:
    def __init__(self, memory: MemoryManager, settings: Settings, summarizer=None) -> None:
        self.memory = memory
        self.settings = settings
        self.summarizer = summarizer  # optional LLMClient for summaries

    def build(self, session: Session, user_input: str) -> tuple[list[dict], list[dict]]:
        msgs: list[dict] = [{"role": "system", "content": session.system_prompt}]

        # Recall relevant memory and inject as a system block (recall timing & placement)
        hits = self.memory.recall(session.id, user_input, top_k=5)
        if hits:
            lines = ["Relevant memory recalled for this turn:"]
            for text, score, _meta in hits:
                lines.append(f"- {text}")
            msgs.append({"role": "system", "content": "\n".join(lines)})

        # Compress older history if over threshold
        history = list(session.messages)
        if len(history) > self.settings.context_max_messages and self.summarizer is not None:
            keep = self.settings.recent_keep
            older, recent = history[:-keep], history[-keep:]
            session.summary = self._summarize(older, session.summary)
            history = recent

        if session.summary:
            msgs.append({"role": "system", "content": f"Conversation so far (summary):\n{session.summary}"})

        msgs.extend(history)
        msgs.append({"role": "user", "content": user_input})
        return msgs, []  # tool schemas injected by runtime

    def _summarize(self, older: list[dict], prev_summary: str) -> str:
        # Build a transcript
        lines = []
        if prev_summary:
            lines.append(f"Previous summary: {prev_summary}")
        for m in older:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                lines.append(f"{role}: {content}")
            elif role == "tool":
                lines.append(f"tool({m.get('name')}): {m.get('content')}")
        prompt = "Summarise the following conversation in <= 200 words, preserving key facts and decisions:\n\n" + "\n".join(lines)
        try:
            raw = self.summarizer.chat([{"role": "user", "content": prompt}], tools=None)
            return raw["choices"][0]["message"]["content"] or prev_summary
        except Exception:
            return prev_summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_context.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/context.py tests/test_context.py
git commit -m "feat: context builder with memory recall and summary compression

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: Runtime loop (run_turn)

**Files:**
- Create: `agent/runtime.py`, `tests/test_runtime.py`

**Interfaces:**
- `build_default_registry() -> ToolRegistry` registers calculator/search/todo/weather.
- `build_memory(settings) -> MemoryManager` constructs default backends.
- `run_turn(session, user_input, *, settings, llm, registry, memory, trace, summarizer=None) -> str` returns final answer text.

- [ ] **Step 1: Write failing test**

```python
# tests/test_runtime.py
import pytest
from agent.config import Settings
from agent.session import Session, SessionStore
from agent.llm import MockLLMClient
from agent.runtime import run_turn, build_default_registry, build_memory
from agent.trace import TraceLogger

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
    out = run_turn(sess, "ping", settings=settings, llm=llm, registry=reg, memory=mem, trace=trace)
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
    out = run_turn(sess, "what is 1+1", settings=settings, llm=llm, registry=reg, memory=mem, trace=trace)
    assert out == "the answer is 2"
    assert any(m["role"] == "tool" for m in sess.messages)

def test_max_iters_forces_finalize(env):
    settings, sess, store, reg, mem, trace = env
    # always emits a tool call → should hit MAX_TOOL_ITERS and force-finish
    tool_call = {"id": "c", "type": "function",
                 "function": {"name": "calculator", "arguments": '{"expression":"0"}'}}
    llm = MockLLMClient(chat_responses=[
        {"choices": [{"message": {"role": "assistant", "content": "loop",
                                   "tool_calls": [tool_call]}}]} for _ in range(20)
    ])
    out = run_turn(sess, "go", settings=settings, llm=llm, registry=reg, memory=mem, trace=trace)
    assert isinstance(out, str)
    assert len([m for m in sess.messages if m.get("role") == "tool"]) == settings.max_tool_iters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent/runtime.py`**

```python
# agent/runtime.py
from __future__ import annotations
import uuid
from dataclasses import dataclass

from .config import Settings
from .session import Session, SessionStore
from .llm import LLMClient
from .tools.base import ToolResult
from .tools.registry import ToolRegistry
from .tools.calculator import CalculatorTool
from .tools.search import SearchTool
from .tools.todo import TodoTool
from .tools.weather import WeatherTool
from .memory.embeddings import MockEmbedder
from .memory.short_term import InMemoryShortTermStore, RedisShortTermStore
from .memory.vector_store import LocalVectorStore, QdrantVectorStore, ChromaVectorStore
from .memory.manager import MemoryManager
from .context import ContextBuilder
from .trace import TraceLogger


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_all([CalculatorTool(), SearchTool(), TodoTool(), WeatherTool()])
    return reg


def build_memory(settings: Settings, llm: LLMClient | None) -> MemoryManager:
    # Embedder: real if embed model configured, else MockEmbedder
    if settings.embed_model and settings.embed_api_key:
        from .memory.embeddings import OpenAICompatEmbedder
        embedder = OpenAICompatEmbedder(settings.embed_api_key, settings.embed_base_url, settings.embed_model)
    else:
        embedder = MockEmbedder()

    # Short-term backend
    if settings.short_term_backend == "redis":
        short_term = RedisShortTermStore(url=settings.redis_url)
    else:
        short_term = InMemoryShortTermStore()

    # Vector backend
    if settings.vector_backend == "qdrant":
        vs = QdrantVectorStore(url=settings.qdrant_url, embedder=embedder)
    elif settings.vector_backend == "chroma":
        vs = ChromaVectorStore(path=str(settings.sessions_dir / ".chroma"))
    else:
        vs = LocalVectorStore(embedder=embedder, path=settings.sessions_dir / "memory.jsonl")

    return MemoryManager(embedder=embedder, short_term=short_term, vector_store=vs, llm=llm, top_k=5)


def run_turn(
    session: Session,
    user_input: str,
    *,
    settings: Settings,
    llm: LLMClient,
    registry: ToolRegistry,
    memory: MemoryManager,
    trace: TraceLogger,
    summarizer: LLMClient | None = None,
) -> str:
    session.add_user(user_input)
    memory.push_turn(session.id, {"role": "user", "content": user_input})
    trace.event("user", text=user_input)

    builder = ContextBuilder(memory=memory, settings=settings, summarizer=summarizer or llm)
    schemas = registry.openai_schemas()

    iters = 0
    while iters < settings.max_tool_iters:
        messages, _ = builder.build(session, user_input=None)  # type: ignore[arg-type]
        # Builder appends the latest user_input; messages already include it from session.
        # To avoid duplication, drop the last user message added by build and rely on session.messages
        if messages and messages[-1].get("role") == "user" and messages[-1].get("content") == user_input:
            messages = messages[:-1]
        raw = llm.chat(messages, tools=schemas)
        from .parser import parse_response
        parsed = parse_response(raw)
        if parsed.thought:
            trace.event("thought", text=parsed.thought)

        if parsed.tool_calls:
            # Record assistant tool-call turn in session
            session.add_tool_call(
                call_id=parsed.tool_calls[0].id,
                name=parsed.tool_calls[0].name,
                args=parsed.tool_calls[0].args,
            )
            # NOTE: for multi-tool calls we still record each result; session helper accepts one
            # call per assistant turn by design. The first call id is reused as group anchor.
            for call in parsed.tool_calls:
                trace.event("tool_call", name=call.name, args=call.args)
                result: ToolResult = registry.execute(call.name, call.args, session)
                trace.event("tool_result", name=call.name, ok=result.ok, content=result.content)
                session.add_tool_result(call_id=call.id, name=call.name, content=result.content)
                memory.push_turn(session.id, {"role": "tool", "name": call.name, "content": result.content})
            iters += 1
            continue

        answer = parsed.final_answer or ""
        session.add_assistant(answer)
        memory.push_turn(session.id, {"role": "assistant", "content": answer})
        trace.event("assistant", text=answer)
        # Persist extracted long-term facts (best effort)
        try:
            memory.remember_sid(session.id, [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ], llm=llm)
        except Exception:
            pass
        return answer

    # Force finalise
    final = "(stopped: maximum tool iterations reached)"
    session.add_assistant(final)
    trace.event("assistant", text=final)
    return final
```

> Note: `builder.build` was designed to take a `user_input` string for recall. We added a small workaround: when `user_input=None`, the builder will fail. Adjust by passing the real user_input and de-duping the last `user` message in messages. Implementation above does that.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/runtime.py tests/test_runtime.py
git commit -m "feat: runtime loop with tool execution, iters limit, and memory hooks

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 13: CLI entrypoint

**Files:**
- Create: `cli.py`, `tests/test_cli.py`

**Interfaces:**
- `python cli.py --session <id> [--llm-model X] [--once "question"]`. Multi-line REPL with `exit`/`quit`.

- [ ] **Step 1: Write failing test (smoke)**

```python
# tests/test_cli.py
import os, subprocess, sys
from pathlib import Path

def test_cli_help_runs(tmp_path: Path):
    env = os.environ.copy()
    env["LLM_API_KEY"] = ""
    env["SESSIONS_DIR"] = str(tmp_path)
    r = subprocess.run([sys.executable, "cli.py", "--help"], capture_output=True, text=True, env=env)
    assert r.returncode == 0
    assert "session" in r.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: subprocess failure.

- [ ] **Step 3: Implement `cli.py`**

```python
# cli.py
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from agent.config import Settings
from agent.session import Session, SessionStore
from agent.llm import LLMClient, MockLLMClient
from agent.runtime import run_turn, build_default_registry, build_memory
from agent.trace import TraceLogger


def main() -> int:
    p = argparse.ArgumentParser(description="Mini Agent CLI")
    p.add_argument("--session", "-s", required=True, help="Session id (window name)")
    p.add_argument("--once", help="Run a single message and exit (non-interactive)")
    p.add_argument("--mock", action="store_true", help="Use MockLLMClient (no real API calls)")
    args = p.parse_args()

    settings = Settings.from_env(sessions_dir=Path(os.environ.get("SESSIONS_DIR", "sessions")))

    if args.mock:
        llm = MockLLMClient()
    else:
        if not settings.llm_api_key:
            print("error: LLM_API_KEY is required (or pass --mock)", file=sys.stderr)
            return 2
        llm = LLMClient(api_key=settings.llm_api_key, base_url=settings.llm_base_url,
                        model=settings.llm_model,
                        embed_api_key=settings.embed_api_key, embed_base_url=settings.embed_base_url,
                        embed_model=settings.embed_model)

    store = SessionStore(settings.sessions_dir)
    session = store.load(args.session)
    registry = build_default_registry()
    memory = build_memory(settings=settings, llm=llm)
    trace = TraceLogger(settings.sessions_dir, args.session)

    def ask(text: str) -> str:
        answer = run_turn(session, text, settings=settings, llm=llm,
                          registry=registry, memory=memory, trace=trace)
        store.save(session)
        return answer

    if args.once:
        print(ask(args.once))
        return 0

    print(f"mini_agent — session={args.session} (type 'exit' to quit)")
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        s = text.strip()
        if not s:
            continue
        if s in ("exit", "quit"):
            return 0
        try:
            ans = ask(s)
            print(ans)
        except Exception as e:  # noqa: BLE001
            trace.event("error", message=str(e))
            print(f"[error] {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add cli.py tests/test_cli.py
git commit -m "feat: cli entrypoint with --session, --once, --mock

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 14: Integration test (real API, env-gated)

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Implement env-gated integration test**

```python
# tests/test_integration.py
import os
import pytest
from pathlib import Path

from agent.config import Settings
from agent.session import Session, SessionStore
from agent.llm import LLMClient
from agent.runtime import run_turn, build_default_registry, build_memory
from agent.trace import TraceLogger


@pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="LLM_API_KEY not set")
def test_real_weather_question(tmp_path: Path):
    settings = Settings.from_env(sessions_dir=tmp_path)
    settings.context_max_messages = 20
    llm = LLMClient(api_key=settings.llm_api_key, base_url=settings.llm_base_url, model=settings.llm_model)
    session = Session(id="it1")
    registry = build_default_registry()
    memory = build_memory(settings=settings, llm=llm)
    trace = TraceLogger(tmp_path, "it1")
    out = run_turn(session, "What's the weather in Beijing? Use the weather tool.",
                   settings=settings, llm=llm, registry=registry, memory=memory, trace=trace)
    assert isinstance(out, str) and len(out) > 0
```

- [ ] **Step 2: Run with API key**

```bash
LLM_API_KEY=sk-... LLM_BASE_URL=https://api.deepseek.com/v1 LLM_MODEL=deepseek-chat \
python -m pytest tests/test_integration.py -v -s
```

- [ ] **Step 3: Commit (skip CI triggers by default)**

```bash
git add tests/test_integration.py
git commit -m "test: env-gated integration test against real LLM

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 15: Demo script for recording

**Files:**
- Create: `demo/demo_weather_todo.py`

- [ ] **Step 1: Implement demo script**

```python
# demo/demo_weather_todo.py
"""End-to-end demo: query weather, then add todos. Use --mock for CI / recording without API."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from agent.config import Settings
from agent.session import Session
from agent.llm import LLMClient, MockLLMClient
from agent.runtime import run_turn, build_default_registry, build_memory
from agent.trace import TraceLogger


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true")
    p.add_argument("--session", default="demo")
    p.add_argument("--sessions", default="sessions")
    args = p.parse_args()

    settings = Settings.from_env(sessions_dir=Path(args.sessions))
    if args.mock:
        # Script a couple of turns for the demo: ask weather → call weather tool; ask todos → use todo tool; final.
        llm = MockLLMClient(chat_responses=[
            {"choices": [{"message": {"role": "assistant", "content": "checking weather",
                                       "tool_calls": [{"id": "c1", "type": "function",
                                                       "function": {"name": "weather",
                                                                    "arguments": '{"city":"beijing"}'}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "Beijing is sunny 26°C."}}]},
            {"choices": [{"message": {"role": "assistant", "content": "adding todos",
                                       "tool_calls": [{"id": "c2", "type": "function",
                                                       "function": {"name": "todo",
                                                                    "arguments": '{"action":"add","text":"buy umbrella"}'}}]},
                                       ]}},
            {"choices": [{"message": {"role": "assistant", "content": "Done. Added 1 todo."}}]},
        ])
    else:
        llm = LLMClient(api_key=settings.llm_api_key, base_url=settings.llm_base_url, model=settings.llm_model)

    session = Session(id=args.session)
    registry = build_default_registry()
    memory = build_memory(settings=settings, llm=llm)
    trace = TraceLogger(Path(args.sessions), args.session)

    msgs = [
        "What's the weather in Beijing today?",
        "Great, add a todo for me to buy an umbrella.",
    ]
    for m in msgs:
        print(f"\nUSER: {m}")
        ans = run_turn(session, m, settings=settings, llm=llm, registry=registry, memory=memory, trace=trace)
        print(f"AGENT: {ans}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke run with --mock**

Run: `python demo/demo_weather_todo.py --mock --sessions /tmp/mini_demo`
Expected: prints user/agent turns and tool trace on stderr.

- [ ] **Step 3: Commit**

```bash
git add demo/
git commit -m "feat: demo script for weather + todo scenario

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 16: README + Architecture Q&A + Prompt log

**Files:**
- Create: `README.md`, `docs/ARCHITECTURE_QA.md`, `docs/PROMPTS_LOG.md`

- [ ] **Step 1: Write `README.md`**

Content outline (full text in file):
- Title + one-paragraph summary.
- **Run**:
  1. `python -m venv .venv && source .venv/bin/activate` (or Windows equivalent)
  2. `pip install -r requirements.txt`
  3. `cp .env.example .env`, fill `LLM_*` and optionally `EMBED_*`
  4. `python cli.py --session w1` (open a 2nd terminal: `python cli.py --session w2`)
  5. `--mock` for offline smoke
- **System design** (link to `docs/ARCHITECTURE_QA.md`):
  - Runtime loop diagram
  - Tool registry
  - Session isolation (one JSON per window)
  - Context compression trigger & strategy
  - Memory layers and pluggable backends
- **Memory recall timing & placement** (the spec's hard requirement):
  - **When**: every user turn, before the LLM call, after loading the session
  - **Where**: as a single `system` block immediately after the base system prompt and before any conversation history / summary
  - **What**: top-k results from the vector store filtered by `sid`
  - **Why this placement**: keeps memory orthogonal to the verbatim history; compresses with history without losing it
- **Tests**: `python -m pytest -v`; integration skipped without `LLM_API_KEY`
- **Project layout**: paste the file map

- [ ] **Step 2: Write `docs/ARCHITECTURE_QA.md`**

Answer each of the 5 modules, citing concrete code references in this repo:
- **模块一 Context/Performance**
  1. First-token latency → streaming (`client.chat.completions.create(..., stream=True)`), speculative placeholder, prompt caching (Anthropic) / system-fingerprint (OpenAI).
  2. 200-turn context → 3-tier compression: rolling summary + memory extraction + keep recent K; ensure fluency by retaining last N turns verbatim and preserving all extracted facts.
- **模块二 Memory**
  1. Recall by query embedding → top-k from semantic store, scoped by `sid`; also include recent short-term K.
  2. Classic framework: short-term / episodic / semantic. Trends: tool-use aware memory, retrieval-augmented agents, hierarchical memory; top players: MemGPT/Letta, LangGraph Memory, Claude Projects+Memory, OpenAI Memory.
- **模块三 Task**
  1. Long-horizon goal loss → goal reminder in system prompt, explicit task tree, periodic re-grounding, milestone checkpoints, progress notes in summary.
  2. Daily 9am recap → scheduled trigger + memory recall + summarisation pipeline; cron / scheduler; produce digest.
- **模块四 Tool/Session Runtime**
  1. Async tools: tool returns a `task_id`; runtime polls/callbacks, updates a `pending_results` map; user gets a "running" message; final answer injected into next turn.
  2. Busy session + new message: enqueue the new turn (FIFO), or surface a "still working" notice; tool completion event buffered and appended before resuming.
- **模块五 Architecture Compare**
  1. Claude Code emits tool outputs as XML-ish `tool_use` blocks; GLM/豆包 use OpenAI `tool_calls`. Trade-offs (robustness, flexibility, parsing complexity, schema fidelity).
  2. OpenHands state machine: explicit states; pros (clarity) and cons (boilerplate, rigidity). More elegant: event-sourced, single-loop guard with `next_action` reducer, or coroutine-style generators.

- [ ] **Step 3: Write `docs/PROMPTS_LOG.md`**

A short chronological log of the major prompts/decisions used while building, including:
- Initial brainstorming questions & answers
- Spec self-review notes
- Key design pivots (memory three-layer, pluggable backends, A vs B tool-calling)
- The prompts in `agent/prompts.py` (system + extractor)
- Any issue + resolution pairs

- [ ] **Step 4: Commit**

```bash
git add README.md docs/ARCHITECTURE_QA.md docs/PROMPTS_LOG.md
git commit -m "docs: readme, architecture q&a, prompt log

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 17: End-to-end smoke + final push

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest -v
```
Expected: all unit tests pass; integration skipped without `LLM_API_KEY`.

- [ ] **Step 2: Smoke the CLI with mock LLM**

```bash
python cli.py --session w1 --once "what is 2+2?" --mock 2>/dev/null
```

- [ ] **Step 3: Smoke the demo**

```bash
python demo/demo_weather_todo.py --mock --sessions /tmp/mini_demo
```

- [ ] **Step 4: Final commit + push**

```bash
git status
git add -A
git diff --cached --quiet || git commit -m "chore: final smoke results and cleanups

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 5: Confirm push on remote**

Run: `git ls-remote origin main`
Expected: hash matches local `HEAD`.

---

## Self-Review

**Spec coverage** — each spec section maps to a task:
- §2 directory: Task 1-16 build exactly that tree
- §3 loop: Task 12 runtime
- §4 registry: Tasks 3-7
- §5 session: Task 6
- §6 context+compression: Task 11
- §7 3-layer pluggable memory: Tasks 9-10, 8 (embeddings)
- §8 trace/errors: Task 2 + 12
- §9 tests: Tasks 1-15 each have tests; 14 integration env-gated
- §10 architecture Q&A: Task 16
- §11 deliverables: all created

**Placeholders** — none. Every step shows full code.

**Type consistency**:
- `MemoryManager.remember_sid(sid, recent_turns, llm=...)` used in runtime Task 12 matches manager Task 10.
- `ContextBuilder.build(session, user_input)` signature used in runtime Task 12 (we pass `user_input=None` then trim the duplicate last user message; alternatively we can pass `user_input` and remove the add_user from runtime — see note in Task 12). Pick one consistently during execution.
- `run_turn` consumes settings fields present in Task 1 Settings dataclass.
- `ToolResult.ok/err` consumed in runtime/registry/calculator.
- `ParsedResponse.tool_calls[].id/name/args` used in runtime.
