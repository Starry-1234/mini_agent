# 架构设计题答案

对设计文档中 5 个架构设计模块的解答。每条都附带本仓库的具体代码引用，让设计决策可以与实际实现一一对照。

---

## 模块一 Context / Performance

### Q1.1 — 如何把首 token 延迟压到最低？

首 token 延迟（TTFT）主要取决于上游 chat API 的 time-to-first-byte。两个互补的杠杆是**流式输出**和**prompt 缓存**，两者都无需改动架构就能用上。

**流式输出（`stream=True`）。** 当前 `LLMClient.chat`（`agent/llm.py`）调用 `client.chat.completions.create(...)` 并返回完整 dump 出来的 dict——也就是说它会等整个 completion 跑完。切换到流式只需要改一行：给 `create` 传 `stream=True`，迭代 chunk，累加 content + `tool_calls` delta，最后返回同形状的 dict。下游的 `parse_response`（`agent/parser.py`）只读 `msg["content"]` 和 `msg["tool_calls"]`，所以它本来就对流式/非流式无感。预期效果：TTFT 从"完整生成"降到"第一个 chunk 的往返"——在托管模型上通常能省 200-800 ms。

**占位提示。** 为了 UX，trace logger 可以在 LLM 调用一发起时就 emit 一个 `thought` 行，让用户在第一个真实 token 到达之前就看到 agent 在"思考"。`TraceLogger.event("thought", ...)`（`agent/trace.py`）本来就是非阻塞地写 stderr；在 `llm.chat` 调用之后、第一个 chunk 到达之前，立刻用 `text="(waiting for model…)"` 调一下就行。纯 UX 收益，但能显著改善感知延迟。

**Prompt 缓存——按 provider 不同。**
- **Anthropic。** `messages` 数组的前缀 system block 跨请求稳定时会被自动缓存。我们每次都发相同的 `session.system_prompt`，并且 recall block 是按相同的 `sid` 锁定的，所以 system + memory 的缓存命中率在第一轮之后就很高。要利用这一点，需要把 Anthropic 走专门的 client：`anthropic.Anthropic().messages.create(system=[...])`，给 system block 加 `cache_control={"type": "ephemeral"}`。缓存 TTL 由 provider 管理，runtime 不需要加东西。
- **OpenAI。** OpenAI 还没有把 prompt 缓存做成第一类 API，但响应里的 `system_fingerprint` 字段（`raw["system_fingerprint"]）可以用来断言同一次模型构建服务了请求——对可复现性有用，对延迟没用。缓存可以靠 Azure OpenAI 的部署级缓存或第三方 proxy。

所有情况下，**recall block 处在 messages 数组的第 2 位**（见 `README.md §3`）都是最大的可缓存区域：模型每一轮都被问同样的、关于同一份 memory 的问题，所以一个缓存好的 system-prompt-加-memory block 让后续每一轮都变成"小增量"请求。

### Q1.2 — 一个 200 轮的 session 怎么处理 context？

200 轮对话，就算每轮 100 token，也是 20k token 的逐字历史——远超大多数 chat 模型的舒适区。代码里用**三层压缩策略**处理（见 `agent/context.py:ContextBuilder`）：

1. **老轮次的滚动摘要。** 当 `len(session.messages) > CONTEXT_MAX_MESSAGES`（默认 20）时，builder 把历史切成 `older`（除最近 `RECENT_KEEP=8` 轮之外的全部）和 `recent`（最近 8 轮）。它调 `self._summarize(older, prev_summary)`，prompt 是：*"Summarise the following conversation in <= 200 words, preserving key facts and decisions:"*。旧摘要会被折进去，所以摘要会累积。结果：任意长的历史坍缩成一段 200 词的摘要。
2. **对持久事实做 memory 抽取。** 独立于 summary，每轮结束会触发 `memory.remember_sid(session.id, recent_turns, llm=llm)`（`agent/runtime.py`）→ `extract_facts`（`agent/memory/extractor.py`），用 `EXTRACTOR_PROMPT` 抽出 0-5 条持久事实，JSON 数组形式。这些以 `meta={"sid": session.id, "kind": "fact"}` upsert 进向量库，因为它们在 session 消息日志之外，所以扛得住任意多轮压缩。
3. **逐字的最近窗口。** 最后 `RECENT_KEEP=8` 条消息永远逐字进 LLM context。这是流畅性保证：模型仍然能引用、回调、续上未说完的话。

**为什么这个组合在持久知识上是无损的。** 当滚动摘要丢掉更老的 192 轮时，所有重要的事实都已经被蒸馏进向量库了。Recall 把它作为带标签的 `system` block 重新注入到摘要之上，所以模型看到的是：持久事实（recall）+ 要点（summary）+ 最近的逐字。唯一丢的是对话的填充语，而这正是该丢的。

**压缩触发语义。** 触发是**消息数**而不是 token 数。这是刻意的——数 token 需要 tokenizer（多一个依赖、跨 provider 脆弱），20 条消息是一个足够好的"该摘要了"的代理。代价是长的工具调用 trace 会让计数虚高但不增加语义价值；未来可以给 tool 消息加权 0.1× 或者估算 token。

---

## 模块二 Memory

### Q2.1 — 如何按 query embedding 召回 memory，short-term / episodic / semantic 三层分别在哪？

**Recall 流（读路径）。**

```
ContextBuilder.build(session, user_input)                       # agent/context.py
  → MemoryManager.recall(sid=session.id, query=user_input, top_k=5)   # agent/memory/manager.py
      → VectorStore.search(query, top_k)                                # agent/memory/vector_store.py
      → 过滤 meta["sid"] == session.id 的结果
  → 渲染成一个带标签的 `system` block（即 "recall block"）
```

query 是这一轮的**原始 user input**。向量库用配置的 embedder 把 query 向量化（配了 `EMBED_*` 走 `OpenAICompatEmbedder`，否则走 `MockEmbedder`）；结果按余弦排序，保留 top-k。默认后端 `LocalVectorStore`（`agent/memory/vector_store.py`）把每条 `{id, text, vector, meta}` 存到 `sessions/memory.jsonl`。可选后端（`QdrantVectorStore`、`ChromaVectorStore`）用各自引擎做同样的事。`sid` 过滤在 `MemoryManager.recall` 里做，所以每个 session 的 recall 都是封闭的——窗口 1 不会泄到窗口 2。

**Short-term（最近 K 轮）。** `InMemoryShortTermStore`（`agent/memory/short_term.py`）是按 `sid` 划分的 deque（`defaultdict(lambda: deque(maxlen=200))`）。`memory.push_turn` 在三个地方调用：`agent/runtime.py:run_turn` 的 user input（第 103 行）、tool result（第 144 行）、assistant final answer（第 153 行）。这个 deque **不是** LLM context 的真实来源——`Session.messages` 才是——但它撑起了轮末的 `extract_facts` 调用（我们把 `recent_turns` 传给 `remember_sid`）。当 `SHORT_TERM_BACKEND=redis` 时，`RedisShortTermStore` 用 `rpush` + `ltrim` 做同样的事，TTL 由 deque maxlen 管理。

**Episodic（滚动摘要）。** 摘要目前存在 `Session` 对象本身（`session.summary`，见 `agent/session.py`），持久化到 `<sid>.json`。时间够长后这应该也搬进向量库——episodic memory 在概念上是"按时间索引的摘要块"——但对于当前的 200 轮 horizon，一段滚动字符串够了。未来工作：按主题拆摘要并各自 embedding，然后像召回语义事实一样按 user query 召回。

**Semantic（持久事实）。** 存在和 episodic 一样的向量库里，但打 `meta["kind"] == "fact"` 标签。由 `MemoryManager.remember_sid` → `extract_facts` 写入，用 `EXTRACTOR_PROMPT` 强迫模型输出 0-5 条字符串的 JSON 数组。每条用新的 UUID 后缀 upsert，避免冲突。

### Q2.2 — 经典框架与行业发展趋势

**经典三层模型**（short-term / episodic / semantic）源自认知心理学（Atkinson–Shiffrin 记忆模型，1968），在 LLM agent 文献中由 MemGPT（Packer et al., 2023）推广。每一层对应不同的延迟 / 容量 / 召回权衡：

| 层 | 延迟 | 容量 | 召回信号 | 本代码库 |
|---|---|---|---|---|
| Short-term | 微秒 | 有界（200 轮） | recency | `InMemoryShortTermStore` / Redis |
| Episodic | 毫秒 | 无限 | 语义相似度 | `session.summary`（单字符串） |
| Semantic | 毫秒 | 无限 | 语义相似度 | 向量库中的抽取事实 |

**行业趋势（2024-2026）。**

1. **Tool-use aware memory。** 有意思的研究方向是让 memory 召回**条件化于 agent 即将调用的工具**——比如当模型 emit `tool_call(todo, add)` 时，优先召回与 todo 相关的记忆。Letta / MemGPT v0.3 加了"memory blocks"，agent 可以通过 tool call 直接读/写。我们现在的代码一视同仁地对待所有 memory；近期改进是给事实打上"创建时属于哪个工具族"的标签。
2. **Retrieval-augmented agents（RAG-as-memory）。** "长期 memory"和"对个人文档的 RAG"之间的界限在模糊。Mem0、Letta、LangGraph Memory 都支持索引任意文档，把检索当作一次 memory 操作。本代码库可插拔的 `VectorStore` 已经为此铺好路——扔个文档索引器进同一个库就行。
3. **Hierarchical memory。** 不再是单个扁平向量索引，近期工作把 memory 按抽象层拆（raw → events → themes → persona）。Anthropic 在 Claude Projects 里的"Memory"功能隐式是分层的：短期对话、项目级 notes、全局用户画像。我们的三层拆分是最小可行版本；再加一层"persona"把 semantic 事实聚合成一个静态的"用户是谁" prompt 即可。

**头部的玩家和定位。**

- **MemGPT / Letta。** 三层模型在 agent 中的标杆实现。提出"virtual context management"——LLM 自己用 memory tool 把事实 page in / page out。角色扮演 agent 上很强；长程任务跟踪较弱。
- **LangGraph Memory。** 图状态 memory 模型，天然契合 LangGraph 的 reducer 模式。Memory 是图里一个节点，reducer 控制更新方式。适合多 agent 图；单机 CLI 用着是杀鸡用牛刀。
- **Anthropic Claude Projects + Memory。** 两层：每个项目的文件（上传的文档、持久 notes）和跨项目的全局"Memory" 库。注意，memory 是**由 runtime 显式注入**而不是按 query 检索——更接近一个 system-prompt block 而不是向量库。我们代码库对 recall block 复刻了这一点（永远注入、永远带标签），但加了真正的语义检索在上面。
- **OpenAI Memory（ChatGPT）。** 一个全局"ChatGPT 学到的关于你的事实"库，用户可编辑。没有按 session 隔离。我们的 `sid` 过滤做到了 OpenAI 没有的：按窗口隔离。

---

## 模块三 Task

### Q3.1 — 长程任务如何防止目标丢失？

长程 agent 会忘记最初的目标。代码里现在有 3 个机制，加上剩下的路线图。

1. **System prompt 里的目标提示。** `session.system_prompt` 是每次 LLM 调用的第一条消息。用户级指令比如 *"You are helping me plan a launch. The launch date is 2026-09-01."* 住在这个 block 里，能扛住压缩、摘要、工具闲聊——因为压缩只动 `messages`，不动 system prompt。（`agent/session.py:Session.system_prompt` 默认是 `"You are a helpful Agent. Use tools when needed."`——用户在创建 session 时覆盖。）
2. **显式任务树（路线图）。** 代码库还没有一等公民的任务树，但 `TodoTool`（`agent/tools/todo.py`）是底座：一个按 session 持久化的 `{id, text, done}` 列表。自然的扩展是暴露一个 `plan` action，接收分层目标并产出嵌套 todo；runtime 会把当前 open 的 todo 注入 recall block，每轮可见。
3. **周期性 re-grounding。** `TraceLogger`（`agent/trace.py`）把每个事件追加到 `<sid>.trace.jsonl`。未来的"re-grounding" hook 会扫最近 N 条 trace 事件，摘要 agent 相对原目标的漂移（用一个便宜的 LLM 调用），把结果注入 recall block。架构上能撑住——`MemoryManager.recall` 接受任意 `query` 字符串，所以周期任务可以 `memory.recall(sid, "<original goal>")` 重新拉相关事实。
4. **用 session summary 做 milestone checkpoint。** `session.summary` 在每次压缩时更新。强制让 summary 包含一行显式的 "Open goals:"——用同一个 `_summarize` prompt 抽取——就给模型一个可回看的 checkpoint。现在 summary 是自由格式；明天 prompt 应当要求 `OPEN GOALS:` 和 `PROGRESS:` 段。
5. **在 summary 里写进度 notes。** 同上机制——每轮结束，runtime 可以往专用 memory fact 写一行"last action + outcome"，这样下一轮的 recall 会把 agent 刚才做的事带回来，下下一轮能比较。

### Q3.2 — 怎么实现"每天早上 9 点做复盘总结"？

每天 9 点的复盘是一个**被调度的** memory 召回 + 摘要流水线。组件：

1. **调度器。** 一个独立进程里的小型 cron 循环（或 OS cron 项），本地时间 09:00 触发。`apscheduler` 或手写的 `while True: sleep_until(09:00); …` 都可以。CLI 工具最简单：OS 级触发——`cron` 项跑 `python cli.py --session recap --once "<prompt>"`。
2. **Memory 召回。** 复盘 prompt 触发 `MemoryManager.recall(sid="<user>", query="yesterday's tasks, decisions, and open items", top_k=20)`，针对**用户级**（而不是 session 级）的 memory 索引。这意味着需要一个 session 级约定："session id `__user__` 是全局复盘 session"——或者更干净地，配第二个 `MemoryManager` 用 `sid=None` 来跳过按 session 过滤。
3. **摘要。** 把召回的事实喂给 LLM，prompt：*"Produce a morning briefing. Yesterday's activity: <facts>. Today's date is <today>. Output: 3 sections — DONE, IN PROGRESS, BLOCKED."*
4. **递送。** 打印到 stdout 给 cron 捕获，或者推到 webhook / email，或者开一个以复盘为首条 system 消息的新 session 窗口——用户一天的第一条对话落在一个已经被 primed 的 session 里。

代码库已经有除调度器之外的所有积木。最干净的设计：加一个 `agent/jobs/daily_recap.py`，import `MemoryManager` 和 `LLMClient`，调度一个 job，结束后退出。cron 友好，无守护进程。

---

## 模块四 Tool / Session Runtime

### Q4.1 — 异步工具：runtime 长什么样？

不少现实工具天然是异步的——长的数据库查询、视频上传、模型微调。工具立刻返回 `task_id`；结果稍后才到。runtime 必须能在不阻塞 loop 的前提下处理。

**设计（本代码库，已经被 `ToolResult` 未来化）。**

1. **工具同步返回 `task_id`。** 一个新的 `AsyncTool` 接口返回 `ToolResult.ok("task_id=<uuid>, status=running")`。`ToolRegistry.execute` 不变——结果类型一样。
2. **Runtime 把 pending 结果推进 `pending_results` map。** `runtime.py:run_turn` 把它每轮的状态扩成 `self._pending: dict[task_id, (call, started_at)]`。这一轮的用户回合不阻塞。
3. **Agent 这一轮 emit "still running" 消息。** 工具执行后，如果结果表示异步，runtime 给 session 注入一条 `system` 消息：`{"role": "system", "content": "Tool <name> started task <id>. The result will arrive in a later turn."}`，然后继续 loop。模型被允许产出一个 final answer 形如 "I've started the upload; I'll let you know when it's done."
4. **轮询器 / 回调。** 一个独立线程或 asyncio 任务和 agent 一起跑。任务完成时，把结果写进 `pending_results[task_id]`。**下一轮用户回合**开始前，runtime 在 `ContextBuilder.build` 之前先排干 `pending_results`：每条 pending 结果作为 `tool` 消息追加，归到原 `call_id`，然后从 map 里删掉。模型看到完成事件就像它是在带内到达的。

现在的代码还没有异步工具——四个内置工具（`calculator`、`search`、`weather`、`todo`）都是微秒级。但 `ToolResult` 数据类已经带着 `ok` / `content`，加一个可选的第三字段 `task_id` 是非破坏性变更。

### Q4.2 — session 忙的时候新消息又来了怎么办？

session 可能在 tool 调用中途时用户按下回车。两种行为，明显的偏好是其中之一：

1. **FIFO 入队。** Runtime 拥有一个按 session 的队列：`self._queues: dict[sid, deque[user_input]]`。`run_turn` 在跑的时候，CLI 来的新输入 append 到队列。loop 返回时，runtime 查队列，开下一轮。Trace logger emit 一个 `queued` 事件，让用户看到队列深度。
2. **"仍在工作" 提示。** 如果队列深度超过阈值（比如 3），CLI 打印 `[busy: agent is still working on the previous turn; message queued]`，而不是默默接收输入。用户可以决定中断（Ctrl-C）或者等。

**为什么入队、而不是中断。** 中断一个执行到一半的工具很危险——半个数据库写、半个文件上传。入队是安全默认。CLI 表面已经支持 `--once`（一条 prompt、退出）——这种模式完全绕过队列，所以想"发完就忘"的用户可以走这条。

**忙时的工具完成事件。** 如果 runtime 有一个异步轮询器（Q4.1）跑在后台线程里，完成事件应该被**缓存**，而不是注入到正在跑的轮次里。缓存在下一轮开始时排干：pending 的工具结果在 `ContextBuilder.build` 之前 append 到 `session.messages`，所以模型看到它们和带内结果在同一个逻辑位置。这匹配 chat 模型的预期：工具结果永远是"自你上次发问以来发生的"。

---

## 模块五 Agent Runtime 架构对比

### Q5.1 — Claude Code vs OpenAI function-calling

两种协议的分歧在于**模型怎么 emit 工具调用**以及 runtime 怎么解析。

**Claude Code / Anthropic tool_use。** 模型在它的 assistant 轮里 emit 一个结构化 block：

```
<thinking>...</thinking>
<tool_use>
  <name>get_weather</name>
  <input>{"city": "beijing"}</input>
</tool_use>
```

工具结果作为 `user` 角色消息里的 `<tool_result>` block 回来。Runtime 按 `tool_use_id` 把它们拼起来。**优点：** 模型能在同一轮里 emit 文本和工具调用（混合 content）；XML-ish 分隔符用正则就能轻松抽出来；Anthropic 的 prompt 缓存能干净地用上。**缺点：** 解析是两步（先剥 XML，再 `json.loads` input），而且模型偶尔会幻觉闭合标签；协议是 Anthropic 特有的。

**OpenAI function-calling（`tool_calls`）。** 模型在 message 上返回一个结构化 JSON 字段：`message.tool_calls = [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}]`。工具结果作为一条独立的 `role: "tool"` 消息带着 `tool_call_id` 回来。**优点：** API 边界上类型安全；`arguments` 是一个字符串，runtime 解析一次；同一轮里的多个并行 tool call 是一等公民。**缺点：** 文本和 tool_calls 同一轮混排不好；参数解码是 string-in-JSON（`parser.py:parse_response` 里已经见识过了）。

**我们的 runtime（`agent/parser.py`）是混合的。** 主路径处理 OpenAI 风格的 `tool_calls`；fallback 路径（`_text_fallback`）用正则从 `tool_call` 风格的 XML-ish block 里抽 `{"name": ..., "arguments": ...}`，所以同一个 `parse_response` 能吃任意一种 provider 的输出。这是刻意的——设计文档说"OpenAI-compatible 国内模型"，我们想让 parser 在模型偶尔 fallback 到文本模式工具调用时也能活。

**权衡总结。**

| 维度 | Claude Code（`tool_use`） | OpenAI（`tool_calls`） |
|---|---|---|
| 对漂移的鲁棒性 | 高（XML-ish 分隔符） | 中（依赖 string-in-JSON） |
| 灵活性（混合） | 高（同一轮文本 + 工具） | 低（字段分开） |
| 解析复杂度 | 中（正则 + JSON） | 低（直接 dict 访问） |
| Schema 保真度 | 高（XML schema、严格 input block） | 高（JSON Schema、provider 校验） |
| 缓存机会 | 高（Anthropic prompt 缓存） | 低（无第一类缓存 API） |

**结论。** 对 OpenAI-compatible 的 provider（DeepSeek、GLM、豆包），`tool_calls` 是正确的主路径——这是 provider 原生 emit 的、原生解析的。文本 fallback 覆盖那种把 JSON 放在 content 字段里的稀有个体。

### Q5.2 — OpenHands 风格的显式状态机 vs 替代方案

OpenHands（开源的 Devin 克隆版）把 agent loop 跑成一个**显式状态机**：状态有 `INIT`、`PLAN`、`ACT`、`OBSERVE`、`REFLECT`、`DONE`，有显式转移和事件日志。这对**可观测性**和**正确性证明**很棒——每次状态变化是日志事件，你可以端到端回放这一轮。

**优点。**
- **清晰。** 每个状态有明确的 pre / post 条件。新贡献者读 `state_transitions.py` 就能理解 loop，不用读 LLM 调用点。
- **可回放。** 既然每次转移都被日志下来，你可以确定性地重跑过去一个 session 到任意状态，然后分叉。
- **守卫。** 状态级守卫能抓住不可能的转移（比如 `OBSERVE → ACT` 中间没夹一条工具结果）。

**缺点。**
- **样板代码。** 每个状态都要一个转移函数。5 个状态的机器就是 ~200 行脚手架，里面没有任何真逻辑。
- **僵硬。** LLM 行为不能干净地装进离散状态——同一个 `chat()` 调用可以是"plan 步"、"reflect 步"、或者"act 步"，看 prompt。强迫 runtime 给每次调用贴标签会削减模型的自主权，强迫作者预想每一种转移。
- **可组合性。** 多 agent 共享状态机臭名昭著地难组合；同一进程里两个状态机需要一个 orchestrator 状态机。

**我们觉得更优雅的三个替代。**

1. **事件溯源的 loop。** 一个 loop 把事件（`event_user_input`、`event_recall`、`event_chat_request`、`event_chat_response`、`event_tool_call`、`event_tool_result`、`event_assistant_answer`）emit 到一个 append-only 日志。没有"状态"——只有事件。回放就是"把日志重放到第 N 个事件"。当前的 `TraceLogger`（`agent/trace.py`）已经做了 80% 这个设计——每个事件都 JSONL 写入、都有时间戳、都在 stderr 上人类可读。把它从"sidecar logger" 升格为"主 loop API"，就能以几分之一的代码拿到 OpenHands 的可回放性。
2. **单 loop 守卫 + `next_action` reducer。** 保持一个 `run_turn` loop，但让每次迭代从一个小的 enum 决定 `next_action`：`CONTINUE`（带着 tool result 再 loop 一次）、`ANSWER`（返回）、`STOP`（强制收尾）。这是一维状态机——唯一的"状态"是"下一步该做什么"。有界迭代（`MAX_TOOL_ITERS`）替代显式 STOP 转移。`agent/runtime.py` 里的当前 `run_turn` loop 在结构上就是这个——`continue` / `return answer` / `return force_finalize` 就是三条分支。把 enum 形式化让 loop 更易测、更好推理。
3. **协程式生成器。** Loop 是一个 Python 生成器，向消费者 `yield` 每个事件，并接收下一条指令（`yield_event` 协议）。消费者可以是 CLI 打印器、测试 harness、或者 Web UI，不动 loop。这是分离最干净的、但实现成本最高；我们不会为 CLI-first 的 runtime 采用它。

**我们的选择。** 当前 `run_turn` 最接近选项 2——一个单守卫 loop、三个出口。我们拿到了 coroutine 的简洁而不用 yield 给外部消费者；`TraceLogger` 已经给了我们事件日志。如果将来需要 OpenHands 风格的可观测性，我们把 trace logger 升格成 event bus 即可，不必重写 loop。这才是复杂度正确的铺排顺序。
