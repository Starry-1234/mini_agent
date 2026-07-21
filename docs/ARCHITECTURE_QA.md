# Architecture Q&A

Answers to the five architecture design modules from the design spec. Every answer cites concrete code references in this repo so the design decisions can be audited against the actual implementation.

---

## 模块一 Context / Performance

### Q1.1 — How do you keep first-token latency low for chat?

First-token latency (TTFT) is dominated by time-to-first-byte from the upstream chat API. The two complementary levers are **streaming** and **prompt-caching**, both of which the codebase is positioned to use without architectural change.

**Streaming (`stream=True`).** The current `LLMClient.chat` (`agent/llm.py`) calls `client.chat.completions.create(...)` and returns the fully-dumped response dict — i.e. it waits for the entire completion. Switching to streaming is a one-line change: pass `stream=True` to `create`, iterate chunks, accumulate content + `tool_calls` deltas, and return the same dict shape. The downstream `parse_response` (`agent/parser.py`) only reads `msg["content"]` and `msg["tool_calls"]`, so it is already agnostic to streaming-vs-non-streaming. Expected impact: TTFT drops from "full generation" to "first chunk round-trip" — typically 200-800 ms on a hosted model.

**Speculative placeholder.** For UX, the trace logger can emit a `thought` line as soon as the LLM call begins, so the user sees the agent "thinking" before the first real token arrives. `TraceLogger.event("thought", ...)` (`agent/trace.py`) is already non-blocking on stderr; we can call it with `text="(waiting for model…)"` immediately after `llm.chat` is invoked and before the first chunk arrives. This is purely a UX win but it materially changes perceived latency.

**Prompt caching — provider-specific.**
- **Anthropic.** The `messages` array is cached automatically when the leading system block is stable across requests. Because we always send the same `session.system_prompt` and the recall block is keyed on the same `sid`, the cache hit rate for system + memory is high after the first turn. To exploit this, we would route Anthropic through a dedicated client that uses `anthropic.Anthropic().messages.create(system=[...])` with `cache_control={"type": "ephemeral"}` on the system block. The cache TTL is provider-managed; we add nothing to the runtime.
- **OpenAI.** OpenAI does not expose prompt caching as a first-class API yet, but the `system_fingerprint` field on the response (`raw["system_fingerprint"]`) lets us assert that the same model build served the request — useful for reproducibility, not for latency. For caching we rely on either Azure OpenAI's deployment-level caching or third-party proxies.

In all cases, the **recall block sitting in position #2 of the messages array** (see `README.md §3`) is the biggest cacheable region: the model is asked the same questions about the same memory on every turn, so a cached system-prompt-plus-memory block turns every later turn into a "small delta" request.

### Q1.2 — How do you handle a 200-turn context?

A 200-turn conversation, even at 100 tokens per turn, is 20k tokens of verbatim history — well beyond the comfort zone of most chat models. The codebase handles this with a **three-tier compression strategy** (see `agent/context.py:ContextBuilder`):

1. **Rolling summary of older turns.** When `len(session.messages) > CONTEXT_MAX_MESSAGES` (default 20), the builder splits history into `older` (everything except the last `RECENT_KEEP=8` turns) and `recent` (last 8). It calls `self._summarize(older, prev_summary)` which prompts the model: *"Summarise the following conversation in <= 200 words, preserving key facts and decisions:"*. The previous summary is folded in so summaries compound. Result: an arbitrarily long history collapses into a single 200-word paragraph.
2. **Memory extraction of durable facts.** Independently of summary, every turn-end fires `memory.remember_sid(session.id, recent_turns, llm=llm)` (`agent/runtime.py`) → `extract_facts` (`agent/memory/extractor.py`) which uses `EXTRACTOR_PROMPT` to pull 0-5 durable facts as a JSON array. These are upserted into the vector store under `meta={"sid": session.id, "kind": "fact"}` and survive any number of compression cycles because they live outside the session message log.
3. **Verbatim recent window.** The last `RECENT_KEEP=8` messages are always passed verbatim into the LLM context. This is the fluency guarantee: the model can still quote, refer back, and continue mid-thought.

**Why this combination is lossless on durable knowledge.** When the rolling summary throws away the older 192 turns, every fact that mattered has already been distilled into the vector store. Recall re-injects the top-k matches as a labelled `system` block above the summary, so the model sees: durable facts (recall) + gist (summary) + recent verbatim. The only thing lost is conversational filler, which is exactly what should be lost.

**Compress trigger semantics.** The trigger is a **count of messages**, not tokens. This is deliberate — counting tokens requires a tokenizer (extra dep, brittle across providers), and 20 messages is a good proxy for "long enough that summarisation is worth the round-trip". The trade-off is that long tool-call traces inflate the count without adding semantic value; a future improvement would be to weight tool messages at 0.1× or to estimate tokens.

---

## 模块二 Memory

### Q2.1 — How do you recall memory by query embedding, and where do short-term / episodic / semantic live?

**Recall flow (read path).**

```
ContextBuilder.build(session, user_input)                       # agent/context.py
  → MemoryManager.recall(sid=session.id, query=user_input, top_k=5)   # agent/memory/manager.py
      → VectorStore.search(query, top_k)                                # agent/memory/vector_store.py
      → filter results where meta["sid"] == session.id
  → render as one labelled `system` block (the "recall block")
```

The query is the **raw user input** for this turn. The vector store embeds the query using whichever embedder is configured (`OpenAICompatEmbedder` when `EMBED_*` are set, else `MockEmbedder`); results are cosine-ranked; the top-k are kept. The default backend `LocalVectorStore` (`agent/memory/vector_store.py`) stores `{id, text, vector, meta}` per item in `sessions/memory.jsonl`. Optional backends (`QdrantVectorStore`, `ChromaVectorStore`) do the same thing over their respective engines. The `sid` filter is applied in `MemoryManager.recall` so every session's recall is hermetic — window 1 cannot leak facts from window 2.

**Short-term (recent K turns).** `InMemoryShortTermStore` (`agent/memory/short_term.py`) is a per-`sid` deque (`defaultdict(lambda: deque(maxlen=200))`). `memory.push_turn` is called from three places: user input (line 103), tool results (line 144), and assistant final answer (line 153) of `agent/runtime.py:run_turn`. The deque is *not* the source of truth for the LLM context — the `Session.messages` list is — but it backs the `extract_facts` call at turn-end (we pass `recent_turns` to `remember_sid`). When `SHORT_TERM_BACKEND=redis`, `RedisShortTermStore` does the same thing via `rpush` + `ltrim`, TTL-managed by the deque maxlen.

**Episodic (rolling summaries).** The summary is currently stored on the `Session` object itself (`session.summary`, see `agent/session.py`) and persisted to `<sid>.json`. On a long enough horizon this should move into the vector store as well — episodic memory is conceptually "summary chunks indexed by time" — but for the current 200-turn horizon, a single rolling string is sufficient. Future work: split the summary by topic and embed each chunk, then recall on the user query the same way we recall semantic facts.

**Semantic (durable facts).** Stored in the same vector store as episodic would be, but tagged `meta["kind"] == "fact"`. Written by `MemoryManager.remember_sid` → `extract_facts`, which uses `EXTRACTOR_PROMPT` to coerce the model into emitting a JSON array of 0-5 strings. Each string is upserted with a fresh UUID suffix to avoid collisions.

### Q2.2 — The classic framework and where the industry is going

**The classic three-layer model** (short-term / episodic / semantic) traces back to cognitive psychology (Atkinson–Shiffrin memory model, 1968) and was popularised in the LLM-agent literature by MemGPT (Packer et al., 2023). Each layer maps to a different latency / capacity / recall trade-off:

| Layer   | Latency    | Capacity         | Recall signal            | This codebase                      |
|---------|------------|------------------|--------------------------|------------------------------------|
| Short-term | microseconds | bounded (200 turns) | recency              | `InMemoryShortTermStore` / Redis   |
| Episodic   | milliseconds | unbounded       | semantic similarity      | `session.summary` (single string)  |
| Semantic   | milliseconds | unbounded       | semantic similarity      | vector store of extracted facts    |

**Industry trends (2024-2026).**

1. **Tool-use aware memory.** The interesting research direction is making memory recall *condition on the tool the agent is about to call* — e.g. when the model emits `tool_call(todo, add)`, prefer memories related to todos. Letta / MemGPT v0.3 added "memory blocks" the agent can read/write directly via tool calls. Our current code treats all memory equally; a near-term improvement is to tag facts with the tool family they were created under.
2. **Retrieval-augmented agents (RAG-as-memory).** The line between "long-term memory" and "RAG over personal documents" has blurred. Mem0, Letta, and LangGraph Memory all support indexing arbitrary documents and treating retrieval as a memory operation. The codebase's pluggable `VectorStore` is positioned for this — drop in a document indexer that writes into the same store.
3. **Hierarchical memory.** Rather than one flat vector index, recent work splits memory by abstraction level (raw → events → themes → persona). Anthropic's "Memory" feature in Claude Projects is implicitly hierarchical: short-term chat, project-level notes, and a global user profile. Our three-layer split is the minimum viable version of this; a fourth "persona" tier would aggregate semantic facts into a single static "who the user is" prompt.

**Top players and where they sit.**

- **MemGPT / Letta.** The canonical implementation of the three-layer model in agents. Introduces the "virtual context management" idea — the LLM issues memory tools to page facts in and out of its own context window. Strong on persona / role-play agents; weak on long-horizon task tracking.
- **LangGraph Memory.** A graph-state memory model that fits naturally with LangGraph's reducer pattern. Memory is a node in the graph, reducers control how it updates. Good fit for multi-agent graphs; overkill for a single-agent CLI.
- **Anthropic Claude Projects + Memory.** Two-tier: per-project files (uploaded docs, persistent notes) and a global "Memory" bank that follows the user across projects. Notably, memory is **explicitly injected** by the runtime rather than retrieved per query — closer to a system-prompt block than to a vector store. Our codebase mirrors this for the recall block (always injected, always labelled) but adds true semantic retrieval on top.
- **OpenAI Memory (ChatGPT).** A single global store of "facts ChatGPT has learned about you", editable by the user. No per-session scoping. Our `sid` filter gives us what OpenAI lacks: per-window isolation.

---

## 模块三 Task

### Q3.1 — How do you prevent long-horizon goal loss?

Long-horizon agents forget their original goal. Three mechanisms in the current code, plus a roadmap for the rest.

1. **Goal reminder in the system prompt.** `session.system_prompt` is the first message on every LLM call. A user-level instruction like *"You are helping me plan a launch. The launch date is 2026-09-01."* lives in this block and survives compression, summarisation, and tool chatter — because compression only touches `messages`, not the system prompt. (`agent/session.py:Session.system_prompt` defaults to `"You are a helpful Agent. Use tools when needed."` — users override at session creation.)
2. **Explicit task tree (roadmap).** The codebase does not yet have a first-class task tree, but `TodoTool` (`agent/tools/todo.py`) is the substrate: a persistent, per-session list of `{id, text, done}` items. A natural extension is to expose a `plan` action that takes a hierarchical goal and emits nested todos; the runtime would inject the current open todos into the recall block so they are visible to every turn.
3. **Periodic re-grounding.** `TraceLogger` (`agent/trace.py`) appends every event to `<sid>.trace.jsonl`. A future "re-grounding" hook would scan the last N trace events, summarise the agent's drift from the original goal (via a cheap LLM call), and inject the result into the recall block. The architecture supports this — `MemoryManager.recall` takes any `query` string, so a periodic job could call `memory.recall(sid, "<original goal>")` to re-surface relevant facts.
4. **Milestone checkpoints via session summary.** `session.summary` is updated whenever compression fires. Forcing the summary to include an explicit "Open goals:" line — extracted by the same `_summarize` prompt — gives the model a checkpoint to refer back to. Today the summary is freeform; tomorrow the prompt should ask for `OPEN GOALS:` and `PROGRESS:` sections.
5. **Progress notes in summary.** Same mechanism — at every turn-end, the runtime can write a one-line "last action + outcome" into a dedicated memory fact, so the next turn's recall surfaces what the agent just did and the next-next turn can compare.

### Q3.2 — How would you implement a daily 9am recap?

A daily 9am recap is a *scheduled* memory recall + summarisation pipeline. The components:

1. **Scheduler.** A small cron-style loop in a separate process (or an OS cron entry) that fires at 09:00 local time. `apscheduler` or a hand-rolled `while True: sleep_until(09:00); …` works. For a CLI tool, the simplest is an OS-level trigger: a `cron` entry that runs `python cli.py --session recap --once "<prompt>"`.
2. **Memory recall.** The recap prompt triggers `MemoryManager.recall(sid="<user>", query="yesterday's tasks, decisions, and open items", top_k=20)` against a *user-scoped* (not session-scoped) memory index. This means we need a session-level convention: "session id `__user__` is the global recap session" — or, more cleanly, a second `MemoryManager` configured with `sid=None` to skip the per-session filter.
3. **Summarisation.** Pass the recalled facts to the LLM with a recap prompt: *"Produce a morning briefing. Yesterday's activity: <facts>. Today's date is <today>. Output: 3 sections — DONE, IN PROGRESS, BLOCKED."*
4. **Delivery.** Print to stdout for the cron capture, post to a webhook, email, or open a fresh session window with the recap as the first system message — the user's first interaction of the day lands in a primed session.

The codebase already has every building block except the scheduler. The cleanest design: add `agent/jobs/daily_recap.py` that imports `MemoryManager` and `LLMClient`, schedules a job, and exits when done. Cron-friendly, no daemon.

---

## 模块四 Tool / Session Runtime

### Q4.1 — Async tools: what does the runtime look like?

Several real-world tools are intrinsically async — a long database query, a video upload, a model fine-tune. The tool returns a `task_id` immediately; the result lands later. The runtime must handle this without blocking the loop.

**Design (this codebase, future-proofed by `ToolResult`).**

1. **Tool returns a `task_id` synchronously.** A new `AsyncTool` interface returns `ToolResult.ok("task_id=<uuid>, status=running")`. `ToolRegistry.execute` does not change — the result type is the same.
2. **Runtime pushes the pending result into a `pending_results` map.** `runtime.py:run_turn` extends its per-iteration state with `self._pending: dict[task_id, (call, started_at)]`. The user turn does not block.
3. **The agent emits a "still running" message this turn.** After tool execution, if the result indicates async, the runtime injects a `system` message into the session: `{"role": "system", "content": "Tool <name> started task <id>. The result will arrive in a later turn."}` and continues the loop. The model is allowed to produce a final answer that says "I've started the upload; I'll let you know when it's done."
4. **Poller / callback.** A separate thread or asyncio task runs alongside the agent. When a task completes, it writes the result into `pending_results[task_id]`. On the **next user turn**, the runtime drains `pending_results` *before* `ContextBuilder.build`: each pending result is appended as a `tool` message attributed to the original `call_id`, then removed from the map. The model sees the completion as if it had arrived in-band.

The current code doesn't ship async tools — none of the four built-in tools (`calculator`, `search`, `weather`, `todo`) take more than microseconds. But the `ToolResult` data class already carries `ok` / `content`, so adding a third optional `task_id` field is a non-breaking change.

### Q4.2 — What happens when a new message arrives on a busy session?

A session can be mid-tool-call when the user hits Enter. Two behaviours, with a clear preference:

1. **FIFO enqueue.** The runtime owns a per-session queue: `self._queues: dict[sid, deque[user_input]]`. While `run_turn` is executing, new inputs from the CLI append to the queue. When the loop returns, the runtime checks the queue and starts the next turn. Trace logger emits a `queued` event so the user sees the queue depth.
2. **"Still working" notice.** If the queue depth exceeds a threshold (say 3), the CLI prints `[busy: agent is still working on the previous turn; message queued]` instead of accepting more input silently. The user can decide to interrupt (Ctrl-C) or wait.

**Why enqueue, not interrupt.** Interrupting a tool mid-execution is risky — a half-applied database write, a half-uploaded file. Enqueueing is the safe default. The CLI surface already supports `--once` (one prompt, exit) — that mode bypasses the queue entirely, so a user who wants to "send and forget" can.

**Tool completion events while busy.** If the runtime has an async poller (Q4.1) running in a background thread, completion events should be **buffered**, not injected into a running turn. The buffer drains at the start of the next turn: pending tool results are appended to `session.messages` *before* `ContextBuilder.build`, so the model sees them at the same logical position as in-band results. This matches the chat model expectation: tool results are always "this happened since you last asked".

---

## 模块五 Agent Runtime Compare

### Q5.1 — Claude Code vs OpenAI function-calling

The two protocols diverge in *how* the model emits tool calls and how the runtime parses them.

**Claude Code / Anthropic tool_use.** The model emits a structured block inside its assistant turn:

```
<thinking>...</thinking>
<tool_use>
  <name>get_weather</name>
  <input>{"city": "beijing"}</input>
</tool_use>
```

Tool results come back as `<tool_result>` blocks in a `user` role message. The runtime stitches them together by `tool_use_id`. **Strengths:** the model can emit text and tool calls in the same turn (mixed content); the XML-ish delimiters are easy to extract with regex; Anthropic's prompt caching applies cleanly. **Weaknesses:** parsing is two-stage (strip XML, then `json.loads` the input), and model output occasionally hallucinates closing tags; the protocol is Anthropic-specific.

**OpenAI function-calling (`tool_calls`).** The model returns a structured JSON field on the message: `message.tool_calls = [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}]`. Tool results come back as a separate `role: "tool"` message carrying `tool_call_id`. **Strengths:** type-safe at the API boundary; `arguments` is a string the runtime parses once; multiple parallel tool calls in one turn are first-class. **Weaknesses:** text and tool_calls don't mix well on the same turn; argument decoding is JSON-in-a-string (we already saw this in `parser.py:parse_response`).

**Our runtime (`agent/parser.py`) is hybrid.** The primary path handles OpenAI-style `tool_calls`; the fallback path (`_text_fallback`) regex-extracts `{"name": ..., "arguments": ...}` from a `tool_call`-style XML-ish block, so the same `parse_response` will accept either provider's output. This is deliberate — the spec says "OpenAI-compatible 国内模型", and we want the parser to survive a model occasionally falling back to text-mode tool calls.

**Trade-off summary.**

| Aspect                | Claude Code (`tool_use`)               | OpenAI (`tool_calls`)                  |
|-----------------------|----------------------------------------|----------------------------------------|
| Robustness to drift   | high (XML-ish delimiters)              | medium (relies on JSON-in-string)      |
| Flexibility (mixed)   | high (text + tools in one turn)        | lower (separate fields)                |
| Parsing complexity    | medium (regex + JSON)                  | low (direct dict access)               |
| Schema fidelity       | high (XML schema, strict input block)  | high (JSON Schema, provider-validated) |
| Caching opportunities | high (Anthropic prompt caching)        | lower (no first-class cache API)       |

**Verdict.** For OpenAI-compatible providers (DeepSeek, GLM, 豆包), `tool_calls` is the right primary path — it's what the providers natively emit and what they natively parse. The text fallback covers the rare model that emits JSON in the content field instead.

### Q5.2 — OpenHands-style explicit state machine vs alternatives

OpenHands (the open-source Devin clone) runs the agent loop as an **explicit state machine**: states like `INIT`, `PLAN`, `ACT`, `OBSERVE`, `REFLECT`, `DONE`, with explicit transitions and an event log. This is great for *observability* and *correctness proofs* — every state change is a logged event, and you can replay the run end-to-end.

**Pros.**
- **Clarity.** Each state has a clear precondition and postcondition. New contributors can read `state_transitions.py` and understand the loop without reading the LLM call sites.
- **Replayability.** Because every transition is logged, you can re-run a past session deterministically up to any state, then branch.
- **Guards.** State-level guards catch impossible transitions (e.g. `OBSERVE → ACT` without an intervening tool result).

**Cons.**
- **Boilerplate.** Every state needs a transition function. A 5-state machine is ~200 lines of scaffolding before any real logic.
- **Rigidity.** LLM behaviour doesn't fit neatly into discrete states — the same `chat()` call can be a "plan step", a "reflect step", or an "act step" depending on the prompt. Forcing the runtime to label each call reduces the model's autonomy and forces the author to anticipate every transition.
- **Composability.** Multi-agent systems with shared state machines are notoriously hard to compose; two state machines in the same process need an orchestrator state machine.

**Three alternatives we find more elegant.**

1. **Event-sourced loop.** A single loop emits events (`event_user_input`, `event_recall`, `event_chat_request`, `event_chat_response`, `event_tool_call`, `event_tool_result`, `event_assistant_answer`) into an append-only log. There are no "states" — there are events. Replay is "replay the log up to event N". The current `TraceLogger` (`agent/trace.py`) is already 80% of this design — every event is JSONL-written, every event is timestamped, every event is human-readable on stderr. Promoting it from "sidecar logger" to "primary loop API" gives us OpenHands' replayability at a fraction of the code.
2. **Single-loop guard with `next_action` reducer.** Keep one `run_turn` loop, but let each iteration decide `next_action` from a small enum: `CONTINUE` (loop again with tool result), `ANSWER` (return), `STOP` (force-finalise). This is a 1-dimensional state machine — the only "state" is "what should we do next". Bounded iteration (`MAX_TOOL_ITERS`) replaces the explicit STOP transition. The current `run_turn` loop in `agent/runtime.py` is structurally this — `continue` / `return answer` / `return force_finalize` are the three branches. Formalising the enum makes the loop easier to test and reason about.
3. **Coroutine-style generators.** The loop is a Python generator that `yield`s each event to a consumer and receives the next instruction (`yield_event` protocol). The consumer can be a CLI printer, a test harness, or a web UI without modifying the loop. This is the cleanest separation but the highest implementation cost; we would not adopt it for a CLI-first runtime.

**Our choice.** The current `run_turn` is closest to option 2 — a single guarded loop with three exits. We get the simplicity of a coroutine without yielding to an external consumer; the `TraceLogger` already gives us the event log. If we need OpenHands-style observability later, we promote the trace logger to an event bus without rewriting the loop. That is the right ordering of complexity.