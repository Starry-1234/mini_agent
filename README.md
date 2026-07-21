# ✦ Starry Code

```
        +
        |
    ----+----
       /|\
      / | \
     /  |  \
----+---+----
     \  |  /
      \ | /
       \|/
        |
        +
```

A from-scratch, minimum-viable agent runtime in Python. Multi-session CLI, tool-use loop, three-layer pluggable memory, and basic context compression — built on top of any OpenAI-compatible chat API (DeepSeek / GLM / 豆包 / OpenAI) without LangGraph, OpenHands, or any agent framework.

> Spec: [`docs/superpowers/specs/2026-07-21-mini-agent-design.md`](docs/superpowers/specs/2026-07-21-mini-agent-design.md)
> Architecture Q&A: [`docs/ARCHITECTURE_QA.md`](docs/ARCHITECTURE_QA.md)
> Prompt log: [`docs/PROMPTS_LOG.md`](docs/PROMPTS_LOG.md)

---

## 1. Starry Code CLI

### 1.1 Setup

```bash
# 1) Create and activate a virtualenv
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 2) Install dependencies (only `openai` is required; redis/qdrant/chroma are optional)
pip install -r requirements.txt

# 3) Configure environment
cp .env.example .env
# then edit .env and set LLM_API_KEY (and optionally EMBED_* for vector search)
```

`.env.example` defaults point at DeepSeek; switch `LLM_BASE_URL` / `LLM_MODEL` to use GLM, 豆包, OpenAI, or any other OpenAI-compatible endpoint.

### 1.2 Basic commands

```bash
# Interactive session (recommended for chatting)
python cli.py --session w1

# A second terminal opens a different window — completely isolated state
python cli.py --session w2

# One-shot: send a single prompt and exit
python cli.py --session w1 --once "What is 2*(3+4)?"

# Offline / no-API smoke (uses a deterministic MockLLMClient)
python cli.py --session demo --mock --once "ping"
```

### 1.3 End-to-end demo

```bash
python demo/demo_weather_todo.py            # real API (needs LLM_API_KEY)
python demo/demo_weather_todo.py --mock     # offline smoke (recommended for CI / recording)
```

### 1.4 Pluggable backends

Defaults require **zero infrastructure** (in-memory + JSONL files). Switch to real backends by setting `.env`:

| Env var             | Default  | Options             | Notes                                          |
|---------------------|----------|---------------------|------------------------------------------------|
| `LLM_BASE_URL`      | DeepSeek | any OpenAI-compat   | `LLM_MODEL` selects the model                  |
| `LLM_API_KEY`       | (none)   | your key            | required unless `--mock`                       |
| `EMBED_*`           | unset    | OpenAI-compat URL   | unset → keyword/BM25 fallback in vector store  |
| `SHORT_TERM_BACKEND`| `memory` | `memory` / `redis`  | `redis` requires `pip install redis` + `REDIS_URL` |
| `VECTOR_BACKEND`    | `local`  | `local` / `qdrant` / `chroma` | `qdrant`/`chroma` require their clients installed |
| `MAX_TOOL_ITERS`    | `8`      | int                 | iteration cap to break tool loops              |
| `CONTEXT_MAX_MESSAGES` | `20`  | int                 | compression trigger threshold                  |
| `RECENT_KEEP`       | `8`      | int                 | messages kept verbatim after compression       |

Missing dependencies or empty keys automatically fall back to defaults — nothing crashes for a casual user.

---

## 2. System design

### 2.1 Runtime loop

Per user turn, `starry_code.runtime.run_turn` runs the loop below (see `starry_code/runtime.py`):

```
run_turn(session_id, user_input):
  memory.push_turn(session_id, {role:user})     # 1) record to short-term
  while iters < MAX_TOOL_ITERS:
    messages = ContextBuilder.build(session, user_input)
        ├── system prompt
        ├── (NEW) recalled memory block    ← recall timing & placement
        ├── (optional) conversation summary
        └── recent messages
    resp    = llm.chat(messages, tools=schemas)
    parsed  = parse_response(resp)              # {thought, tool_calls, final}
    if parsed.tool_calls:
        for call in parsed.tool_calls:
            result = registry.execute(call.name, call.args, session)
            session.add_tool_result(call.id, result)
            memory.push_turn(session.id, {role:tool})
        iters += 1; continue
    else:
        session.add_assistant(parsed.final)
        memory.remember_sid(session.id, recent_turns, llm=llm)   # distill facts
        SessionStore.save(session)
        return parsed.final
  return "(stopped: maximum tool iterations reached)"   # safety
```

The **bounded `MAX_TOOL_ITERS`** (default 8) protects against pathological loops. If the model never emits a `final_answer`, the runtime force-finalises and returns a guard message.

### 2.2 Tool registry

`starry_code/tools/registry.py` keeps a `dict[name, Tool]`. Each `Tool` carries `name`, `description`, `parameters` (JSON Schema dict), and `execute(args, session) -> ToolResult`. Built-in tools:

| Tool         | What it does                                                          |
|--------------|-----------------------------------------------------------------------|
| `calculator` | AST-whitelisted arithmetic (no `eval`)                                |
| `search`     | Mock keyword search returning canned results                          |
| `todo`       | Per-session todo list: `add` / `list` / `complete`, persists to JSON   |
| `weather`    | Mock weather by city                                                  |

`registry.openai_schemas()` produces the `tools=[…]` payload for OpenAI-style function calling. New tools register with one line in `starry_code/runtime.py:build_default_registry()`.

### 2.3 Session isolation

Each `--session <id>` = one JSON file at `sessions/<id>.json` (see `starry_code/session.py`). Two windows with different ids never share state; re-entering the same id resumes the conversation. Writes are atomic (`tmp.replace`); reads tolerate missing files (return a fresh `Session`).

### 2.4 Context compression

Implemented in `starry_code/context.py:ContextBuilder`. Trigger: when `len(session.messages) > CONTEXT_MAX_MESSAGES`, the builder asks the configured `summarizer` LLM to compress everything except the last `RECENT_KEEP` messages (default: keep last 8 verbatim) into a rolling summary, which is then injected as a `system` block above the verbatim history. **Important invariant:** every fact worth remembering has already been extracted into long-term memory by `memory.remember_sid` before compression can run, so compression is lossless with respect to durable knowledge.

### 2.5 Memory layers

Three layers, all pluggable, with zero-infra defaults (see `starry_code/memory/`):

| Layer       | Stores                              | Default backend | Optional                |
|-------------|-------------------------------------|-----------------|-------------------------|
| Short-term  | last K turns of raw conversation    | in-memory deque | Redis (`SHORT_TERM_BACKEND=redis`) |
| Episodic    | rolling summaries of older dialogue | local JSONL + cosine | Qdrant / Chroma (`VECTOR_BACKEND=…`) |
| Semantic    | distilled facts / user profile      | local JSONL + cosine | Qdrant / Chroma         |

Write path (`starry_code/runtime.py:run_turn` → `starry_code/memory/manager.py:remember_sid` → `starry_code/memory/extractor.py:extract_facts`): at the end of each turn the LLM is asked (via `EXTRACTOR_PROMPT`) to extract 0-5 durable facts as a JSON array; each fact is upserted into the vector store with `meta={"sid": session_id, "kind": "fact"}`.

Read path (`starry_code/context.py:ContextBuilder.build` → `starry_code/memory/manager.py:MemoryManager.recall`): top-k semantic matches against the user's input, scoped by `sid`.

---

## 3. Memory recall — timing and placement

This is the spec's hard requirement. **The placement is not an implementation detail; it shapes the entire prompt structure.**

### 3.1 When — every user turn, **before the LLM call**, **after loading the session**

```
run_turn(session, user_input):
  ...
  while iters < MAX_TOOL_ITERS:
      messages = ContextBuilder.build(session, user_input)   ← recall happens HERE
      resp    = llm.chat(messages, tools=schemas)            ← LLM call comes AFTER
```

The recall lives inside `ContextBuilder.build` (`starry_code/context.py`), which is invoked at the **start of every iteration** of the tool loop — so a tool-heavy turn still gets fresh recall before each LLM call. Recall is gated on the **user input** for that turn, not on intermediate tool results.

### 3.2 Where — single `system` block, immediately after the base system prompt, **before** any conversation history or summary

The order of messages sent to the LLM is, deterministically:

```
[
  {"role": "system", "content": <base system prompt>},                    ← 1
  {"role": "system", "content": "Relevant memory recalled for this turn:\n- ..."},  ← 2 (NEW)
  {"role": "system", "content": "Conversation so far (summary):\n..."},   ← 3 (only if compressed)
  {"role": "user" | "assistant" | "tool", ...},                            ← 4 verbatim history
]
```

In code (`starry_code/context.py:ContextBuilder.build`):

```python
msgs.append({"role": "system", "content": session.system_prompt})   # 1 base system
hits = self.memory.recall(session.id, user_input, top_k=5)          # 2 recall
if hits:
    msgs.append({"role": "system", "content": "Relevant memory recalled ...\n" + ...})
# 3 rolling summary block (only when compressed)
# 4 session.messages (verbatim history, user input last)
```

### 3.3 What — top-k results from the vector store, filtered by `sid`

`MemoryManager.recall` (in `starry_code/memory/manager.py`) calls `vector_store.search(query, top_k)` and filters results by `meta["sid"] == session.id`. The default `top_k=5` is configurable via the constructor. The block is rendered as a single labelled `system` message so the model can distinguish it from the verbatim transcript.

### 3.4 Why this placement

- **Orthogonality to verbatim history.** Memory facts and verbatim dialogue are semantically different objects. Bundling them in the same block would force the model to guess which lines are "facts" and which are "what the user just said", inviting mode confusion.
- **Compresses with history, not with the prompt.** When context compression collapses older turns, the recall block sits above the summary block, so the model still sees durable knowledge even after verbatim history is gone.
- **Cheap to inject, cheap to update.** The block is rebuilt on every turn from the latest vector-store query — no cache invalidation across runs.
- **Model sees it as a single labelled section.** The `system` role signals "background context the model should respect"; labelling it `Relevant memory recalled for this turn:` makes it auditable in the trace.

---

## 4. Tests

```bash
# Run the full suite (verbose)
python -m pytest -v
```

The suite covers:

| File                              | What it exercises                                         |
|-----------------------------------|-----------------------------------------------------------|
| `tests/test_config.py`            | `Settings.from_env` defaults & env overrides              |
| `tests/test_trace.py`             | `TraceLogger` JSONL append + coloured terminal output     |
| `tests/test_registry.py`          | `ToolRegistry` register/lookup/error paths                |
| `tests/test_calculator.py`        | AST sandbox + safe-eval boundaries                        |
| `tests/test_search.py`            | `SearchTool` mock data                                    |
| `tests/test_weather.py`           | `WeatherTool` mock data                                   |
| `tests/test_todo.py`              | `TodoTool` add/list/complete                              |
| `tests/test_session.py`           | `SessionStore` load/save round-trip + isolation           |
| `tests/test_parser.py`            | `parse_response` happy path + text-fallback               |
| `tests/test_llm.py`               | `LLMClient` lazy SDK init + chat dump                     |
| `tests/test_memory_stores.py`     | `LocalVectorStore` upsert/search; keyword fallback        |
| `tests/test_memory_manager.py`    | `MemoryManager` recall-by-sid, extractor                  |
| `tests/test_context.py`           | `ContextBuilder` compression trigger + recall placement   |
| `tests/test_runtime.py`           | `run_turn` loop decisions (mock LLM)                      |
| `tests/test_cli.py`               | CLI `--mock --once` end-to-end                            |
| `tests/test_integration.py`       | Real LLM round-trip — **skipped unless `LLM_API_KEY` is set** |

`tests/test_integration.py` is the only env-gated test; everything else runs offline.

---

## 5. Project layout

```
starry_code/
├── cli.py                         # CLI entrypoint: --session / --once / --mock
├── starry_code/
│   ├── __init__.py
│   ├── config.py                  # env-driven Settings dataclass
│   ├── trace.py                   # colored terminal + JSONL TraceLogger
│   ├── llm.py                     # OpenAI-compatible client + MockLLMClient
│   ├── parser.py                  # parse_response → {thought, tool_calls, final}
│   ├── prompts.py                 # SYSTEM_PROMPT + EXTRACTOR_PROMPT
│   ├── session.py                 # Session dataclass + SessionStore (JSON)
│   ├── context.py                 # ContextBuilder: system + recall + summary + history
│   ├── runtime.py                 # run_turn tool-using loop + build_default_*
│   ├── tools/
│   │   ├── base.py                # Tool + ToolResult
│   │   ├── registry.py            # ToolRegistry
│   │   ├── calculator.py          # AST-sandboxed arithmetic
│   │   ├── search.py              # mock web search
│   │   ├── weather.py             # mock weather by city
│   │   └── todo.py                # per-session todo list
│   └── memory/
│       ├── embeddings.py          # MockEmbedder + OpenAICompatEmbedder
│       ├── short_term.py          # InMemoryShortTermStore + RedisShortTermStore
│       ├── vector_store.py        # LocalVectorStore + QdrantVectorStore + ChromaVectorStore
│       ├── extractor.py           # extract_facts(llm, recent_turns)
│       └── manager.py             # MemoryManager: write-path distillation + read-path recall
├── tests/                         # mirrors starry_code/ structure; *.py unit tests
├── demo/
│   └── demo_weather_todo.py       # end-to-end "weather + todo" scenario
├── docs/
│   ├── ARCHITECTURE_QA.md         # answers to the 5 architecture design modules
│   ├── PROMPTS_LOG.md             # chronological log of decisions & prompts
│   └── superpowers/
│       ├── plans/                 # implementation plan (untracked)
│       └── specs/
│           └── 2026-07-21-mini-agent-design.md
├── sessions/                      # runtime: <sid>.json + <sid>.trace.jsonl (gitignored)
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Dockerfile                     # python:3.12-slim, non-root, tini PID 1
├── docker-compose.yml             # agent service, env_file, sessions volume
├── .dockerignore
└── README.md                      # this file
```

---

## 6. Deploy with Docker

容器化部署是推荐的运行方式。镜像基于 `python:3.12-slim`，非 root 用户运行，会话数据落在挂载卷里。

### 6.1 前置条件

- Docker Desktop（或 Linux 上的 Docker Engine）已安装并运行
- 一个 OpenAI-compatible 的 LLM endpoint（base_url + api_key + model 名）
- 可选：一个 OpenAI-compatible 的 embedding endpoint

### 6.2 配置 `.env`

```bash
cp .env.example .env
# 用编辑器填入：
#   LLM_BASE_URL=https://...
#   LLM_API_KEY=...
#   LLM_MODEL=...
#   EMBED_BASE_URL=...   （可选；不填则用 MockEmbedder）
#   EMBED_API_KEY=...
#   EMBED_MODEL=...
```

`.env` 已在 `.gitignore` 里，**不会被提交**。`docker compose` 会自动加载。

### 6.3 构建并运行

```bash
# 构建镜像（首次或改代码后）
docker compose build

# 方式 A：单次问答
docker compose run --rm agent --session test --once "what is 2+2?"

# 方式 B：交互模式（REPL）
docker compose run --rm agent --session test
> What's the weather in Beijing?
> Add a todo: buy milk
> exit

# 方式 C：自动中文命名（不传 --session）
docker compose run --rm agent
> 查北京天气并记待办
[session auto-named to: 天气查询 — use --session 天气查询 to continue next time]
> exit
```

### 6.4 多窗口隔离

每个 `--session` 是独立 JSON 文件，完全隔离。开多个窗口：

```bash
# 终端 1
docker compose run --rm agent --session 天气

# 终端 2
docker compose run --rm agent --session weekly
```

两个 session 互不串味。退出后 session 文件保留在 `./sessions/`。

### 6.5 持久化

- **会话数据** 落在宿主 `./sessions/`，镜像删除/重建不影响
- **Chroma 向量库**（如果用）也落在宿主 `./sessions/.chroma/`
- 重启容器后，session 历史、todos、长期 memory 全部还在

### 6.6 查看 trace

```bash
# 列出会话文件
ls sessions/

# 看某次会话的 trace（彩色 terminal 输出版）
tail -f sessions/<id>.trace.jsonl | python -m json.tool --no-ensure-ascii
```

trace 包含：user、thought（含 reasoning）、tool_call、tool_result、assistant、recall、error。

### 6.7 离线/演示模式

不连真实 LLM，验证流程是否工作：

```bash
# 跑 demo 脚本
docker compose run --rm --entrypoint python agent demo/demo_weather_todo.py --mock --sessions /app/sessions

# CLI 加 --mock
docker compose run --rm --entrypoint "" agent python cli.py --mock --session demo --once "what is 2+2?"
```

### 6.8 可选后端服务

`docker-compose.yml` 默认只起 `agent` 一个服务。生产/分布式需要时，**取消注释** 即可启用 qdrant（向量库）和 redis（短期记忆）：

```bash
# 编辑 docker-compose.yml，去掉 qdrant / redis 服务的注释
# 同时改 .env:
#   SHORT_TERM_BACKEND=redis
#   REDIS_URL=redis://redis:6379/0
#   VECTOR_BACKEND=qdrant
#   QDRANT_URL=http://qdrant:6333

docker compose up -d agent qdrant redis
```

### 6.9 故障排查

| 症状 | 原因 / 修复 |
|---|---|
| `LLM_API_KEY is required` | `.env` 没填或字段名拼错；确认 `LLM_API_KEY=...`（无空格） |
| `error during connect: ... dockerDesktopLinuxEngine` | Docker Desktop 没启动 → 系统托盘启动 |
| `Connection was reset` / `Failed to connect to github` | 走代理；设 `git config --global http.https://github.com.proxy http://127.0.0.1:<port>` |
| Windows 控制台打印 emoji/CJK 报错 | 镜像已强制 UTF-8；本地是 `cli.py` 直接运行时（不走容器）才需要 `set PYTHONUTF8=1` |
| Chroma 启动时下载 79MB ONNX 模型 | 首次创建 collection 触发，之后缓存到 `~/.cache/chroma/onnx_models/` |

### 6.10 不用 Docker（开发者本地跑）

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python cli.py --session test --once "what is 2+2?"
```

容器是推荐方式；本地直接 python 跑也能用，但 chromadb / 部分二进制依赖在你的本地 Python 环境可能装不上（pip Scripts 锁定等问题）。Docker 是最省心的路径。

---

## 7. License

See `LICENSE`.