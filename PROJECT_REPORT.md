# Starry Code 项目报告 — 给评估者

> **目的**：让一个完全不知情的 AI agent 读完本文档后，能理解项目目标、当前架构、实施进度、踩过的坑，并给出独立建议。
>
> **阅读顺序**：1（概述）→ 2（目标）→ 3（架构）→ 4（实施进度）→ 5（设计决策）→ 6（已解决问题）→ 7（待解决问题）→ 8（技术债）→ 9（评估问题）

---

## 1. 项目概述

**Starry Code**（仓库 `mini_agent`）是一个从零实现的 Python Agent 运行时 CLI。最初是通用 Agent 框架（4 个工具：calculator/search/todo/weather），后转型为面向程序员的**学习与就业动态规划教练**。

**核心承诺**：
- 多 session 并行（同一终端可开多个会话语境）
- OpenAI-compatible chat API（DeepSeek/GLM/MiniMax/OpenAI 均可）
- 三层可插拔 memory（短时 deque + episodic 摘要 + 向量语义）
- 上下文压缩（消息超阈值自动摘要）
- 真工具调用循环，零框架依赖（不用 LangGraph/OpenHands）
- Docker 一键运行，含 watch 模式热重载

**不承诺**（明确的范围边界）：
- 不是多 agent 协作系统
- 不是带图形界面的产品（仍为 CLI）
- 不是无服务器部署（每个 session 一个本地 JSON 文件）

---

## 2. 转型目标与战略

### 2.1 4 个核心能力（用户原始需求）

| # | 能力 | 当前状态 |
|---|---|---|
| 1 | 根据画像（技术栈/目标/时间）生成项目驱动的个性化学习计划 | ✅ 已实现 |
| 2 | 结合外部信息（技术趋势、招聘需求）动态调整计划 | ⚠️ 框架完成，**数据是 Mock** |
| 3 | 以"项目驱动"为核心，产出可执行的具体步骤 | ✅ coach prompt + `update_plan` 工具强制 |
| 4 | Token 优化（卸载/热缓存/分层加载） | ✅ 上下文卸载 + plan_cache 热缓存完成 |

### 2.2 3 个战略原则（用户原文）

1. **先 MVP 后优化** — 第1周跑通可演示原型，不追求完美
2. **预留升级空间** — 设计时考虑未来迁移到 DeepAgents/LangGraph，核心业务逻辑与框架解耦
3. **Token 优化贯穿始终** — 知识库按需加载，避免全量注入

### 2.3 用户真实使用场景

- 用户会创建多个会话语境（例如"学Java"、"Java学习指南"），通过 `starry -c` 续聊、`starry -resume` TUI 选择
- 用户提出学习方法类问题（"怎么学Java"），coach 应该先问画像，不直接给大纲
- 用户提行情类问题（"Rust 行情怎么样"），coach 强制调 `tech_trend` 工具
- 真实 LLM 模型：阿里云百炼 MiniMax-M3（reasoning model，会输出 `<think>...</think>` 块和偶发 surrogate codepoints）

---

## 3. 当前架构

### 3.1 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│ 用户终端                                                     │
│   ↓                                                          │
│ bin/starry.py (Python launcher)                              │
│   ↓ docker compose run --rm agent                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ cli.py                                                  │ │
│ │  ├─ argparse (--session/--once/--mock)                  │ │
│ │  ├─ SessionStore.load() ← 读 sessions/{sid}.json       │ │
│ │  ├─ build_default_registry() → 7 个 Tool                │ │
│ │  ├─ build_memory() → 三层 memory                        │ │
│ │  ├─ TraceLogger() → sessions/{sid}.trace.jsonl          │ │
│ │  ├─ AutoNamer() (仅 auto-id 会话启用)                   │ │
│ │  ├─ render_repl_startup() — 清屏 + header + 历史回放    │ │
│ │  └─ REPL 循环:                                           │ │
│ │      input("> ") → ask() → run_turn()                    │ │
│ │                                ↓                          │ │
│ │                              [error handler]              │ │
│ │                                ↓                          │ │
│ │                       atexit → _cleanup_empty_auto_session│ │
│ └─────────────────────────────────────────────────────────┘ │
│   ↓                                                          │
│ starry_code/runtime.py:run_turn()                            │
│   ├─ sanitize user_input (strip surrogate)                  │
│   ├─ memory.push_turn()                                      │
│   ├─ trace.event("user", ...)                                │
│   ├─ session.add_user()                                      │
│   ├─ ContextBuilder.build() → BuiltContext (side-effect free)│
│   ├─ llm.chat(messages, tools=schemas)                      │
│   ├─ parse_response() — 剥离 thinking blocks                 │
│   ├─ tool loop (≤ settings.max_tool_iters 次)               │
│   │   ├─ trace.event("tool_call", ...)                       │
│   │   ├─ registry.execute(name, args, session)               │
│   │   └─ trace.event("tool_result", ...)                     │
│   ├─ _should_adjust_plan() — 触发器检查                      │
│   │   └─ _inject_trend_for_adjustment() — 调 tech_trend 写记忆│
│   └─ return answer                                           │
│   ↓                                                          │
│ cli.py: print(_strip_surrogates(answer))                      │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 模块清单（starry_code/）

```
starry_code/
├── cli.py                    # 入口：argparse + REPL + atexit cleanup + surrogate strip
├── prompts.py                # SYSTEM_PROMPT (coach) + EXTRACTOR_PROMPT + NAMING_PROMPT
├── context.py                # ContextBuilder — side-effect free，返回 BuiltContext
├──                           # 实现 plan_cache 注入 + tool-result offload (>500字 → artifacts/)
├── session.py                # Session dataclass + SessionStore (JSON 持久化)
├── runtime.py                # run_turn() 编排 + _should_adjust_plan + _inject_trend
├── llm.py                    # LLMClient (OpenAI SDK) + MockLLMClient
├── trace.py                  # TraceLogger (jsonl 写入 + 染色 stderr 输出)
├── naming.py                 # AutoNamer + _sanitize (surrogate strip)
├── parser.py                 # parse_response (剥离 thinking, 解析 tool_calls)
├── config.py                 # Settings (从 .env 读 LLM/embed/memory 后端配置)
├── memory/
│   ├── manager.py            # MemoryManager (push_turn / remember_sid / recall)
│   │                         # recall 支持 cross_session 模式
│   ├── short_term.py         # InMemory + Redis 两种实现
│   ├── embeddings.py         # MockEmbedder (sha256-hash) + OpenAICompatEmbedder
│   ├── vector_store.py       # Local / Chroma / Qdrant 三种后端
│   └── extractor.py          # LLM-driven fact extraction
└── tools/
    ├── base.py               # Tool + ToolResult Protocol
    ├── registry.py           # ToolRegistry — openai_schemas + execute
    ├── calculator.py         # AST 沙箱计算器
    ├── search.py             # Mock 搜索
    ├── weather.py            # Mock 天气
    ├── todo.py               # 会话内 todo list
    ├── tech_trend.py         # ★ Phase 1 — TrendProvider Protocol + Mock 数据
    ├── update_plan.py        # ★ Phase 2 — 唯一能改 plan_cache 的工具
    └── read_artifact.py      # ★ Phase 2 — 取回被 offload 的完整输出
```

### 3.3 数据流：单次用户 turn

```
1. user_input  (含或不含 surrogate)
   ↓ strip_surrogates
2. memory.push_turn(sid, {role: user, content: input})
   trace.event("user", text=input)
   session.add_user(input)
   ↓
3. ContextBuilder.build(session, user_input)
   ├─ 注入 system_prompt
   ├─ 注入召回记忆 (memory.recall with user_input as query)
   ├─ 注入 plan_cache 热缓存 (~50 tokens)
   ├─ 处理 history：
   │   - tool result >500 字符 → 写入 sessions/{sid}/artifacts/{call_id}.json
   │   - 替换为 [artifact saved] 摘要卡片
   ├─ 如果 history > context_max_messages → 摘要压缩 (前 N 留, 后 K 留)
   └─ 返回 BuiltContext (新摘要由 caller 决定是否写入 session.summary)
   ↓
4. llm.chat(messages, tools=schemas)
   → parsed: {thought, tool_calls OR final_answer}
   ↓
5. 如果 tool_calls:
   for call in tool_calls:
     registry.execute(name, args, session) → ToolResult
     trace.event("tool_call"), trace.event("tool_result")
     session.add_tool_call + add_tool_result
     memory.push_turn({role: tool, name, content})
   循环回到步骤 3 (最多 max_tool_iters=8 次)
   ↓
6. 如果 final_answer:
   session.add_assistant(answer)
   memory.push_turn({role: assistant, content: answer})
   trace.event("assistant", text=answer)
   ↓
7. memory.remember_sid() (LLM-driven fact extraction, 仅当 turn 足够长)
   ↓
8. _should_adjust_plan() — 触发器检查
   └─ _inject_trend_for_adjustment() — 调 tech_trend 把结果写进短时记忆
   ↓
9. return answer
   ↓
10. cli.py: store.save(session) (持久化) + print(strip_surrogates(answer))
```

### 3.4 关键技术决策（详见第 5 节）

| 决策 | 理由 |
|---|---|
| `TrendProvider` Protocol | 数据源可替换，mock 是默认实现 |
| `plan_cache` 注入每次 LLM 调用 | ~50 tokens 热缓存，避免漂移 |
| **唯一** `update_plan` 工具改 plan | 防止 LLM "想想就改了" |
| 上下文卸载阈值 500 字符 | 平衡可见性与 token 经济 |
| `_strip_surrogates()` 多层防御 | reasoning model 偶发输出 \ud800-\udfff |
| `reconfigure(errors="replace")` | 输出层兜底，即使 strip 漏了也不崩 |

---

## 4. 实施进度

### Phase 1：改脑 + 装触角（**100%**）

| 项 | 状态 |
|---|---|
| 重写 `prompts.py` (Coach persona) | ✅ |
| `tech_trend.py` + MockTrendProvider | ✅ |
| 注册到 `build_default_registry()` | ✅ |
| 测试 (13 个) | ✅ |

### Phase 2：强化记忆 + Token 优化（**100%**）

| 项 | 状态 |
|---|---|
| 上下文卸载 (artifacts/) | ✅（在 context.py 内） |
| `Session.plan_cache` 字段 | ✅ |
| `update_plan` 工具 | ✅ |
| `read_artifact` 工具 | ✅ |
| `_should_adjust_plan` 触发器 | ✅ |
| 测试 (18 个) | ✅ |

### Phase 3：真实数据 + 产品化（**5% — 仅设计**）

| 项 | 状态 |
|---|---|
| `CompositeTrendProvider` (GitHub/Remotive/HN) | 📝 设计完成 |
| `skill_assess` 工具 | 📝 接口设计完成 |
| `interview_prep` 工具 | 📝 接口设计完成 |
| `project_drive` 工具 | 📝 接口设计完成 |
| Streamlit MVP | 📝 架构设计完成 |
| FastAPI / React | 📝 路线图设计完成 |

设计文档：`docs/PHASE3_DESIGN.md`（525 行）。

---

## 5. 关键设计决策

### 5.1 Coach Prompt 的"必做/必不做"结构

`prompts.py` 的 `SYSTEM_PROMPT` 采用强约束结构：

- **必做行为 5 条**（画像采集、计划生成、下一步1件事、趋势挂钩、节奏感知）
- **必不做行为 4 条**（不列大纲、不脱离目标、不编数据、不替决策）
- **工具使用准则**（tech_trend 行情类必查，todo 记录下一步）
- **回复风格**（≤200 字，少 bullet，多步骤句）

**实测效果**：真实 LLM (MiniMax-M3) 在接收到 coach prompt 后，能稳定先问画像、不直接给 Java 101 大纲。这是 Phase 1 + bug 修复的成果。

### 5.2 plan_cache 设计

```python
{
    "version": 0,          # update_plan 每次真实变更 +1
    "stage": "",           # 当前阶段
    "next_task": "",       # 下一步要做的 1 件事
    "long_term_goal": "",  # 长期目标
    "last_updated": "",    # ISO 时间戳
}
```

**为什么用 dict 而不是 dataclass**：JSON 序列化（asdict）+ 兼容迁移老 session + 字段可灵活增减。

**为什么不让 LLM "想"了就改**：plan_cache 是 ground truth。如果 LLM 在 thought 里说"我应该更新 next_task"但没调工具，下次 LLM 调用不知道它改过。强制走 `update_plan` 工具确保 version 自增 + last_updated 时间戳更新 → 触发器能基于时间判断 stale。

### 5.3 上下文卸载（offload）

阈值 500 字符（可在 Settings 调整）。工具返回 >500 字符时：

1. 写入 `sessions/{sid}/artifacts/{call_id}.json`（原始内容）
2. 替换为 `[artifact saved]` 摘要卡片（head/tail/summary/path）
3. LLM 看到卡片决定是否调 `read_artifact(path=...)` 取完整内容

**为什么不直接截断**：截断是 lossy，LLM 后续无法恢复。offload + read_artifact 是"按需加载"。

### 5.4 _should_adjust_plan 触发器

三个触发条件（**OR 关系**）：

1. **关键词** — 用户说"行情/趋势/前景/调整/对一下..."
2. **stale_7d** — `plan_cache.last_updated > 7 天前`
3. **milestone** — ≥3 个 todo 完成但 plan 还在 v0/v1

触发后调 `_inject_trend_for_adjustment`：从 `long_term_goal` + `user_input` 抽取技术关键词，调 tech_trend 最多 3 个，结果写入短时记忆。**不直接改 plan_cache**——让 LLM 在下一轮根据新数据决定要不要 `update_plan`。

### 5.5 _strip_surrogates() 多层防御

| 层 | 位置 | 处理 |
|---|---|---|
| 1 | `runtime.py:run_turn` 入口 | strip user_input |
| 2 | `trace.py:event()` | strip 所有 field 后 json.dumps |
| 3 | `trace.py:_print()` | strip 后写 stderr |
| 4 | `cli.py:251 print(ans)` | strip 后 stdout |
| 5 | `embeddings.py:_hash_vec()` | strip 后 .encode("utf-8") |
| 6 | `embeddings.py:OpenAICompatEmbedder.embed()` | strip 后送 API |
| 7 | `cli.py:14 reconfigure(errors="replace")` | 输出层兜底 |

**为什么这么多层**：surrogate 是 reasoning model 的硬伤。`text.encode("utf-8")` 是 Python 最严格的字节转换，**任何**含 `\ud800-\udfff` 的字符串都会 raise。在哪个环节漏掉一个，整个 turn 崩。

### 5.6 read_artifact 安全模型

```python
allowed_root = (sessions_dir / session.id / "artifacts").resolve()
p_resolved.relative_to(allowed_root)  # ValueError → 拒绝
```

防止：
- Path traversal（`../../etc/passwd`）
- 跨 session 读 artifacts（A session 读 B session 的）

---

## 6. 已解决的问题（按发现顺序）

### Bug 0: Docker 镜像过期（最关键）

**症状**：用户跑了所有 Phase 1/2 修复后，CLI 表现"还是跟之前一样"。

**根因**：镜像 2026-07-22 build 的，所有 8 月的代码改动都不在容器里。

**修复**：`docker compose build agent`。

**教训**：Python 源码修改不会自动反映到运行中的容器。MVP 阶段必须有"每次代码改动就 rebuild"的流程纪律，或用 `develop.watch`。

### Bug 1: Session.system_prompt 死代码

**症状**：Phase 1 重写了 `prompts.SYSTEM_PROMPT` 但新 session 还在用旧 prompt。

**根因**：`Session` dataclass 把 `system_prompt` 默认值硬编码到字段定义，**不**从 `prompts.SYSTEM_PROMPT` 拿。

**修复**：
- Session 默认值改为 `SYSTEM_PROMPT`
- `SessionStore.load()` 加迁移（空或老默认值 → 新 prompt）
- `runtime.run_turn()` 加防御性检查

### Bug 2: context.py 隐式 mutate

**症状**：docstring 承诺 side-effect free 但 `build()` 内 `session.summary = ...`。

**修复**：返回 `BuiltContext(new_summary=...)`，caller 显式写入。

### Bug 3: REPL 不显示历史

**症状**：`starry -c` 续 session 时 LLM 有记忆但屏幕上看不到之前的对话。

**修复**：新增 `render_repl_startup()` + `print_session_history()`，在 REPL 入口调用。

### Bug 4: REPL 无显示隔离

**症状**：同一终端开多个 session，上一个的输出还在屏幕上。

**修复**：`render_repl_startup()` 第一行输出 `\033[H\033[2J\033[3J`（清屏 + 清滚动缓冲区）。

### Bug 5: auto-id header 显示丑陋 slug

**症状**：新 session 的 REPL header 显示 `✦ auto-20260808-...`，但 window title 是 `✦ Starry Code`，不一致。

**修复**：`render_repl_startup()` 加 `display_id = "Starry Code" if auto- else session.id`。

### Bug 6: surrogate 字符让 print/trace 崩

**症状**：用户按 DELETE 键后输入含 `\udce5`，`print(ans)` raise UnicodeEncodeError。

**根因（最终定位）**：
```
memory.recall(query=user_input)
  → embedder.embed([query])
  → _hash_vec(text)
  → hashlib.sha256(text.encode("utf-8"))  💥
```

之前以为只是 print 路径问题，加了 print/trace 的 strip。但 end-to-end 测试发现 embedder 才是真正的瓶颈。

**修复**：embeddings.py 在 `.encode("utf-8")` 之前 strip + `reconfigure(errors="replace")` 兜底。

---

## 7. 待解决问题

### 7.1 Phase 3 完全没实现

详见 `docs/PHASE3_DESIGN.md`。核心缺口：

- `tech_trend` 用 Mock 数据（12 行硬编码表），不真实
- `skill_assess` / `interview_prep` / `project_drive` 三个核心工具未实现
- 没有 Web UI（仅 CLI）

### 7.2 Coach prompt 强约束不足

实测发现：

- LLM 偶发会**绕过** coach prompt 的"不列大纲"规则
- LLM 偶发不调工具就回答行情类问题
- 需要更明确的"坏例子"和强约束词

可能方案：在 `SYSTEM_PROMPT` 中加入正反示例（few-shot）。

### 7.3 max_tool_iters=8 可能不够

学习规划场景可能需要：search → tech_trend × N → read_artifact → update_plan。8 步可能不够。

当前未改，标记为风险点。

### 7.4 ContextBuilder 摘要的"漂移"

压缩摘要时只更新 `session.summary`，但如果 LLM 后续看到的是新摘要，可能与原 history 矛盾（特别是 update_plan 改变了 plan 但摘要里没体现）。

未解决。

### 7.5 跨会话记忆的边界

`memory.recall(cross_session=True)` 加了，但**没有"遗忘"机制**。用户可能想"忘掉我之前学的 X 项目"。

未解决。

---

## 8. 技术债

### 8.1 必须避免的债（已经避免）

- ✅ **业务逻辑不绑 LangGraph** — runtime.py 是纯 Python 循环
- ✅ **prompt 不硬编码到代码** — 全部在 `prompts.py`
- ✅ **工具实现不绑特定协议** — `TrendProvider` Protocol

### 8.2 可以接受的债

- ⚠️ **tracing 无结构化** — 用 stderr + jsonl，没有 OTLP。未来如要接 Langfuse 需要重写。
- ⚠️ **MockLLMClient** 还在 — 测试用，但生产代码不应该 import 它（runtime 偶尔会用到）
- ⚠️ **memory 后端没拆包** — Redis/Qdrant/Chroma 都直接 import，没有 lazy loading

### 8.3 风险债

- ⚠️ **MiniMax-M3 reasoning 输出 surrogate** — 已经多层防御，但任何新的输出路径都可能漏
- ⚠️ **embedder 调外部 API** — 在 docker 里需要能联网到 LLM_BASE_URL，否则整套链路断
- ⚠️ **REPL 测试覆盖不全** — 真正的 REPL 循环（input 模拟）没写 unit test，只测了 helper

---

## 9. 关键指标

| 指标 | 值 |
|---|---|
| 远程 HEAD | `81a6246` |
| 测试数 | **140 passed + 1 skipped** |
| 工具数 | 7（calculator/search/todo/weather/tech_trend/update_plan/read_artifact）|
| Prompt 行数 | ~150 行（SYSTEM + EXTRACTOR + NAMING） |
| Context offload 阈值 | 500 字符 |
| Token 经济性 | 卸载节省 70%（5000 字符 → ~200 摘要卡） |
| Docker 镜像大小 | ~780MB（chroma + sqlite） |

---

## 10. 给评估者的具体问题

下列问题**没有标准答案**，希望另一个 agent 能基于以上信息给出独立判断：

### Q1（架构）
Coach prompt 是"必做/必不做"硬约束 + few-shot 软约束。**这够吗？** 还是应该用更结构化的方式（比如 plan-then-execute）让 LLM 严格按教练流程走？

### Q2（数据）
Phase 3 用 GitHub Trending + Remotive + HN 三个公开数据源。**聚合权重 40/45/15 合理吗？** 招聘需求 (jobs) 是不是应该占比更高（毕竟核心目标是"找工作"）？

### Q3（用户体验）
用户用 CLI 时反复提"想看历史"、"窗口标题不对"、"显示混乱"。**这些是 CLI 的根本限制吗？** 还是应该在 Web 上重做？Web MVP 优先级应该比 CompositeTrendProvider 高吗？

### Q4（Token 经济）
当前的 offload 阈值是 500 字符固定。**应该按 token 数还是字符数？** 是否需要根据当前 context window 动态调整？

### Q5（安全性）
`read_artifact` 限制了 path 在 `sessions/{sid}/artifacts/` 下。但如果 LLM 被 prompt injection 攻击，调用 `update_plan` 注入恶意 `next_task` 怎么办？**当前没有 plan_cache 内容的合法性校验**，这是不是应该加上？

### Q6（未来扩展性）
如果 Phase 3 完成后想加多用户（账号、权限、协作），**当前架构哪里会先崩？** 是 Session 命名空间？memory 后端？还是 trace 文件结构？

### Q7（建议）
基于以上 6 个问题，**你建议接下来 4 周的具体 roadmap 是什么？** 我们原计划是 CompositeTrendProvider → 3 个新工具 → Streamlit MVP，你认为这个顺序对吗？还是应该调整？

---

## 附录 A：关键技术名词

| 名词 | 含义 |
|---|---|
| 画像 (huàxiàng) | 中文 = user profile / persona。Coach prompt 中指"用户的技术栈/目标/时间等结构化信息" |
| coach prompt | `prompts.SYSTEM_PROMPT` 的 Starry Coach persona 定义 |
| thinking block | reasoning model 在 answer 前的 `<think>...</think>` 块 |
| surrogate | UTF-16 代理项 (`U+D800-U+DFFF`)，UTF-8 协议禁止 |
| offload | 工具返回 >500 字符时写入文件、内存只保留摘要 |
| plan_cache | 会话级 dict，存当前阶段/下一步/长期目标，热注入每次 LLM 调用 |
| hot cache | 与 plan_cache 同义，强调"常驻 context" |
| trigger | `_should_adjust_plan` 检测关键词/7天/里程碑，决定是否注入最新 trend |

## 附录 B：复现步骤

如果你（评估者）想亲自验证：

```bash
# 1. 准备环境
cp .env.example .env
# 编辑 .env 设 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL

# 2. CLI mock 模式（不需要 key）
python cli.py --mock --once "ping"

# 3. CLI 真 LLM 模式
python cli.py --session 测试 --once "我想转 Go 后端，每天 2 小时"
# 期望: coach 行为 — 不直接给 Go 教程，先问画像

# 4. Docker 模式
docker compose build agent
docker compose run --rm agent --session 测试 --once "Rust 行情"

# 5. 跑全量测试
python -m pytest tests/ -q
# 期望: 140 passed, 1 skipped
```

## 附录 C：本文档不包括的内容

- ❌ 任何 API key / 凭证
- ❌ 个人会话数据（`sessions/` 下的 JSON 文件）
- ❌ 与 LLM 供应商的合同 / 价格
- ❌ 具体的 LLM response 内容（只是 trace.jsonl 格式）

如果要更深的技术细节，请直接读 `docs/PHASE3_DESIGN.md`、`docs/ARCHITECTURE_QA.md`、`README.md`。
