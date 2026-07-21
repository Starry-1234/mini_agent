# Prompts Log

A chronological record of the major prompts, decisions, and cross-task issues encountered while building Mini Agent. Not a transcript — a curated log of the points where a prompt or a design choice actually steered the implementation.

---

## 0. Brainstorming Q&A (pre-implementation)

The first round of questions established the **scope and shape** of the project before any code was written.

**Q: Why Python, not TypeScript/Go/Rust?**
A: The user wanted to demonstrate a from-scratch agent with no framework. Python is the lingua franca of LLM work, has the cleanest OpenAI SDK, and lets the focus stay on the agent loop rather than on tooling.

**Q: Why OpenAI-compatible, not Anthropic-native?**
A: The user is on 国内 (Chinese) infrastructure — DeepSeek / GLM / 豆包 — which all expose OpenAI-compatible APIs. Going OpenAI-compatible buys provider-agnosticism for free; the cost is losing Anthropic's `tool_use` XML format (handled by a text fallback in `agent/parser.py`).

**Q: CLI or web UI?**
A: CLI. The user wanted a runnable artefact that's easy to record (screen capture) and easy to test (`--mock --once`). No web framework, no front-end. `--session <id>` is the multi-window primitive.

**Q: Why a three-layer memory? Why not just one big vector store?**
A: Three reasons. (1) Recency is cheap and lives in RAM — short-term doesn't need embedding. (2) Durable facts and rolling summaries are different beasts — facts are atomic strings, summaries are paragraphs; storing them under one type loses signal. (3) The classic Atkinson–Shiffrin / MemGPT framework is well-trodden; users (and reviewers) will recognise the split.

**Q: Plug-in Redis/Qdrant/Chroma, or commit to defaults?**
A: Defaults, but **interface-gated**. `ShortTermStore` and `VectorStore` are Protocols; the runtime picks the implementation from env vars. Missing deps → automatic fallback to in-memory / local JSONL. This is YAGNI-compatible: zero-infra for casual users, real backends for production.

---

## 1. Spec self-review notes (during writing `docs/superpowers/specs/2026-07-21-mini-agent-design.md`)

After the first draft of the spec, the user / reviewer flagged:

- **"Memory recall timing & placement" needed to be a hard requirement.** Not "implementation detail", not "TBD" — explicit when/where/what/why. Became §3 of the README and Q1.1/Q2.1 of the architecture Q&A.
- **Compression strategy was vague.** "基础压缩" was the original phrasing; the spec was tightened to "rolling summary + memory extraction + keep last K verbatim" so the implementation has a concrete contract.
- **Async tools were punted.** The spec says "async tools return a `task_id`" but defers the runtime design. Captured in ARCHITECTURE_QA §Q4.1 as a forward design — no code change required for the current 4 sync tools.
- **Test gating.** `test_integration.py` is the only env-gated test (skipped without `LLM_API_KEY`). Everything else runs offline with `MockLLMClient`. This is the only way the suite can be CI-friendly without secrets.

---

## 2. Key design pivots

### 2.1 Three-layer memory (Task 9 / Task 10)

**Original thought:** one big vector store with all conversation history embedded on the fly.
**Pivot:** three layers — short-term (recent deque), episodic (rolling summary), semantic (extracted facts). Why: embedding the whole history on every query is O(N) in tokens and the recall is dominated by recent turns anyway. Splitting the work means short-term is O(1) and semantic recall is O(facts), not O(history).

**What landed:** `agent/memory/short_term.py` (in-memory deque + optional Redis), `agent/memory/vector_store.py` (local JSONL + Qdrant + Chroma), `agent/memory/manager.py` (coordinator), `agent/memory/extractor.py` (LLM distiller).

### 2.2 Pluggable backends (Task 9 / Task 13)

**Original thought:** hardcode the local backends; let production users fork.
**Pivot:** `Protocol` classes (`ShortTermStore`, `VectorStore`, `Embedder`) plus `.env`-driven selection in `agent/runtime.py:build_memory`. Missing dependency → raise with a clear message; missing key → fall back to local. Zero-infra defaults preserved.

**Trade-off accepted:** more code paths to test. Mitigated by `MockEmbedder` (deterministic SHA-256 vectors) and the fact that `tests/test_memory_stores.py` exercises only `LocalVectorStore` (the other two backends require network/installed services).

### 2.3 Tool-calling — option A vs option B

**Option A: native function-calling + normalised parser (with text fallback).** This is what we shipped.
**Option B: pure text-mode, model emits `<tool_call>{...}</tool_call>`, runtime regex-extracts.**

Chose A because:
- OpenAI-compatible providers natively support `tools=[...]` and return `tool_calls` — no extra prompt engineering.
- Argument decoding is the provider's problem, not ours.
- Text fallback (`agent/parser.py:_text_fallback`) covers the rare model that emits JSON in `content` instead, so we don't lose robustness.

The cost is the JSON-in-string parsing at `parser.py:30` — `arguments` comes back as a string. Handled with try/except and a `_raw` escape hatch.

### 2.4 System prompt as the first message, not a chat-template parameter

**Considered:** passing `system` as a separate SDK parameter.
**Chose:** always send it as the first `messages[0]` with `role="system"`.

Why: provider parity. Some OpenAI-compatible providers (notably DeepSeek at the time of writing) handle the dedicated `system=` field inconsistently across SDK versions; the messages-array form is universal and what `Chat Completions` was designed around. Cost: an extra dict in every request; benefit: zero ambiguity.

### 2.5 Session JSON, not SQLite

**Considered:** SQLite for sessions — atomic, queryable, scales to large histories.
**Chose:** one JSON file per session, atomic-write via `tmp.replace`.

Why: the spec is minimum-viable. JSON files are human-readable (great for debugging in `sessions/`), diff-friendly, and trivial to back up. SQLite would have meant a dep, a schema, and a migration story. The atomic-write pattern (`tmp.replace`) is sufficient for single-process CLI use.

### 2.6 Compression trigger — message count, not tokens

**Considered:** count tokens via `tiktoken` and trigger at 8k tokens.
**Chose:** count messages and trigger at `CONTEXT_MAX_MESSAGES=20` (default).

Why: `tiktoken` is an extra dep, and provider-specific tokenisers drift. Message count is a stable proxy. Future work: weight tool messages lower, or estimate tokens via `len(content) // 4`.

---

## 3. Cross-task issues & resolutions

### 3.1 `ToolResult` dataclass + classmethod name collision (Task 3)

The brief wrote:

```python
@dataclass
class ToolResult:
    ok: bool
    content: str
    @classmethod
    def ok(cls, content): ...     # ← name collision
    @classmethod
    def err(cls, content): ...
```

On Python 3.12 the `@dataclass` machinery walks the class body, sees the `ok` classmethod object as a class-level attribute, and treats it as the implicit default for the `ok: bool` field. That makes `content` (no default) "follow" a defaulted field → `TypeError: non-default argument 'content' follows default argument`.

**Resolution:** implemented `ToolResult` as a plain class with explicit `__init__` / `__repr__` / `__eq__`. The public API is unchanged (`ToolResult(True, "x")`, `ToolResult.ok("x")`, `ToolResult.err("x")`). Documented in the class docstring and in the Task 3 report. `Tool` itself stayed a `@dataclass` — no naming collision there.

### 3.2 `VectorStore.search` returning `(id, score, meta)` vs `MemoryManager.recall` expecting `(text, score, meta)` (Task 10 + fix)

The Task 10 brief specified `MemoryManager.recall` returns `list[(text, score, meta)]`, but the existing `VectorStore` Protocol declared `search` returning `(id, score, meta)`. The Task 10 implementer worked around the mismatch by reading the private `_items` dict on `LocalVectorStore` to substitute text for id. This only worked for the local backend — Qdrant/Chroma would silently return empty strings.

**Resolution (Task 10 fix, commit `643f425`):** unified the contract. `VectorStore.search` now returns `(text, score, meta)` across all three backends. `LocalVectorStore.search` returns `item["text"]`. `QdrantVectorStore.search` pops `"text"` out of the payload before returning meta. `ChromaVectorStore.search` returns the document string from `result["documents"]`. `MemoryManager.recall` now unpacks the tuple directly with no private-attr bridge.

This was the most consequential cross-task issue — it would have shipped silently broken in production with any non-local vector backend. Caught by code review of the Task 10 report, fixed in a dedicated follow-up commit.

### 3.3 Demo mock-mode under-scoping (Task 15)

The Task 15 demo scripted 4 mock responses for 2 user turns (1 tool call + 1 final answer each). But `run_turn` calls `memory.remember_sid(...)` after every final answer, which calls `extract_facts(llm, ...) → llm.chat(...)`, consuming one extra mock response per turn.

**Net effect:** in `--mock`, the third scripted response (intended as turn 2's tool call) was silently consumed by turn 1's fact extractor. Turn 2's final answer was consumed as a plain assistant turn with no tool call. The extractor then hit the end of the list and `MockLLMClient` raised `RuntimeError("no more scripted responses")`, swallowed by the `try/except` in `runtime.py`. **The todo tool never fired under mock.**

**Resolution (Task 15 fix, commit `8851339`):** expanded the scripted response list from 4 to 6 entries:
- [0] turn 1 weather tool_call
- [1] turn 1 weather final answer
- [2] turn 1 extractor (`"[]"`)
- [3] turn 2 todo tool_call
- [4] turn 2 todo final answer
- [5] turn 2 extractor (`"[]"`)

The extractor responses return `"[]"` so `extract_facts` returns `[]` and no fact is stored. The trace now correctly shows both `weather` and `todo` tool calls firing.

This was a **demo-vs-runtime contract mismatch** — the demo brief didn't account for the implicit LLM call from memory extraction. A future Task 15 brief would script 1 chat + 1 extractor per turn.

---

## 4. The prompts we ship (`agent/prompts.py`)

Two prompts. Both deliberately terse — long system prompts are a known anti-pattern (the model loses focus on the user's actual question).

### 4.1 `SYSTEM_PROMPT`

```python
SYSTEM_PROMPT = (
    "You are a helpful Agent. Use tools when needed. "
    "When you have a final answer, reply in plain text (no tool calls). "
    "Keep thoughts brief."
)
```

**Why these sentences:**
- *"You are a helpful Agent."* — identity, anchors the model's role.
- *"Use tools when needed."* — permission, not instruction. The model decides when "needed" is.
- *"When you have a final answer, reply in plain text (no tool calls)."* — disambiguation rule. Some models occasionally emit a tool call alongside the final text. Forcing "final answer = plain text" makes `parse_response` deterministic.
- *"Keep thoughts brief."* — the `thought` field is shown to the user via the trace logger; we don't want a 200-word monologue before every tool call.

### 4.2 `EXTRACTOR_PROMPT`

```python
EXTRACTOR_PROMPT = (
    "You are a memory extractor. From the recent conversation turn below, "
    "extract 0-5 short, durable facts worth remembering long-term about the user "
    "(preferences, habits, identity, key decisions, constraints). "
    "Output strictly a JSON array of strings. If nothing is worth remembering, "
    "output []. No commentary.\n\nCONVERSATION:\n{turns}\n\nJSON:"
)
```

**Why this shape:**
- *"0-5 short, durable facts"* — bounded output prevents prompt-blowup on long conversations; "durable" excludes ephemeral chatter ("thanks", "ok", "what time is it").
- *"about the user"* — focuses extraction on user identity / preferences / decisions, not on the agent's own actions.
- *"Output strictly a JSON array of strings."* — `extract_facts._parse_facts` extracts the first `[...]` block and parses it as JSON. The model is told "no commentary" so the JSON is the only output and the parser is robust.
- *"If nothing is worth remembering, output []."* — most turns have nothing durable; an empty array is the right answer, not a forced fabrication.

**Known limitation:** the prompt is in English; a Chinese-speaking user gets Chinese-language turns formatted as `{role}: {content}` lines, and the model is asked to extract English facts. This works (the model translates) but loses nuance. A future improvement: localise the prompt based on detected user language.

---

## 5. Self-review note for Task 16

While writing this log, the realisation: **the spec's hard requirement on memory recall timing & placement is not arbitrary — it's the single thing that makes the architecture survive compression**. If memory were appended to the verbatim history, compression would eat it. If it were placed after the user message, the model would weight it less. Putting it as `system` block #2 — right after the base system prompt and right before the summary — makes it (a) cache-friendly, (b) compression-proof, and (c) structurally distinct from verbatim dialogue. That's the line that the README §3 spells out, and it's the design property the whole architecture rests on.