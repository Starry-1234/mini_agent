# Mini Agent —— 从零实现最小可用 Agent（设计文档）

- 日期：2026-07-21
- 状态：已通过 brainstorming，待用户复核
- 语言/运行时：Python
- LLM：OpenAI-compatible 国内模型（DeepSeek / GLM / 豆包），通过 `base_url` + `api_key` 环境变量做 provider-agnostic
- 交互：CLI 终端，`--session <id>` 区分多窗口
- 工具调用机制：**方案 A** —— 原生 function-calling + 自写归一化 parser（保留文本 fallback）

## 1. 目标与非目标

### 目标
- 从零实现 Agent Runtime，不依赖任何 agent 框架（langgraph/openhands/openclaw）。
- 实现基本 loop：接收输入 → 判断直接回复还是调工具 → 调工具 → 判断继续 loop 还是返回。
- 工具注册机制（name / description / JSON-Schema 参数），LLM 基于 Schema 自主决策。
- LLM 输出解析：提取思考过程 / 工具调用 / 最终答案。
- Session 隔离与持久化：多窗口独立、随时续聊。
- Context 管理：最大轮次限制、记住历史、支持纯对话追问与带工具追问、基础压缩。
- 三层可插拔 memory：短期 / 情节 / 语义画像。
- 基本异常处理 + 工具调用 trace/日志。
- 测试用例（mock LLM 单测 + 真实 API 集成 demo）。
- 架构设计题（模块一~五）文字答案。

### 非目标（YAGNI）
- 不实现复杂压缩（如分层语义树），只做基础滚动摘要。
- 不实现完整多模态。
- 不硬绑分布式 infra（Redis/向量库为可选后端）。
- 不做 Web UI（CLI 为主）。

## 2. 目录结构

```
mini_agent/
├── cli.py                    # 入口：python cli.py --session <id>
├── agent/
│   ├── runtime.py            # 核心 loop
│   ├── llm.py                # OpenAI-compatible 客户端封装（重试/超时）
│   ├── parser.py             # 归一化 LLM 响应 → {thought, tool_calls, final}
│   ├── context.py            # 上下文组装 + 轮次限制 + 压缩
│   ├── session.py            # session 隔离 + 持久化
│   ├── config.py             # 环境变量配置
│   ├── trace.py              # 步骤日志（终端 + jsonl）
│   ├── memory/
│   │   ├── manager.py        # 三层协调：写路径(蒸馏/去重) + 读路径(混合检索)
│   │   ├── short_term.py     # ShortTermStore 接口 + 文件默认实现 + Redis 实现
│   │   ├── vector_store.py   # VectorStore 接口 + 本地实现 + Qdrant/Chroma 实现
│   │   ├── extractor.py      # 从对话蒸馏事实/画像
│   │   └── embeddings.py     # embedding provider（可配）+ 关键词 fallback
│   └── tools/
│       ├── registry.py       # 工具注册机制
│       ├── base.py           # Tool 基类
│       ├── calculator.py     # 安全表达式计算
│       ├── search.py         # mock 搜索
│       ├── todo.py           # 待办增/查/完成（按 session 持久化）
│       └── weather.py        # mock 天气
├── tests/
│   ├── conftest.py           # mock LLM client fixture
│   ├── test_tools.py
│   ├── test_parser.py
│   ├── test_registry.py
│   ├── test_runtime.py       # loop 决策（mock LLM）
│   ├── test_session.py       # 隔离
│   ├── test_context.py       # 压缩触发
│   ├── test_memory.py        # 召回/去重
│   └── test_integration.py   # 真实 API，env 缺失时 skip
├── demo/
│   └── demo_weather_todo.py  # 录屏用完整场景脚本
├── docs/
│   ├── ARCHITECTURE_QA.md    # 架构设计题 模块一~五
│   ├── PROMPTS_LOG.md        # AI Prompt 与问题解决记录
│   └── superpowers/specs/2026-07-21-mini-agent-design.md  # 本文
├── sessions/                 # 运行时生成
├── README.md
├── requirements.txt
└── .env.example
```

依赖刻意最小：`openai`（chat + 可选 embedding）；可选 `redis`、`qdrant-client`/`chromadb`（缺失时降级到默认后端）。runtime / registry / schema / 压缩全部自写。

## 3. 核心 Loop（runtime.py）

```
run_turn(session_id, user_input):
  session = SessionStore.load(session_id)
  memory.on_user_input(session, user_input)          # 记短期
  iters = 0
  while iters < MAX_TOOL_ITERS:
    messages = context.build(session, user_input)      # system + tools + 召回memory + 历史(+摘要)
    resp = llm.chat(messages, tools=registry.schemas())
    parsed = parser.normalize(resp)                     # {thought, tool_calls, final_answer}
    trace.log_thought(parsed.thought)
    if parsed.tool_calls:
      for call in parsed.tool_calls:
        result = registry.execute(call.name, call.args, session)  # try/except → 错误回喂
        trace.log_tool(call, result)
        session.append_tool_result(call, result)
      iters += 1
      continue
    else:
      session.append_assistant(parsed.final_answer)
      memory.on_turn_end(session)                       # 异步/同步蒸馏事实入长期库
      SessionStore.save(session)
      return parsed.final_answer
  # 超过迭代上限：强制收尾
  return force_finalize(session)
```

- `MAX_TOOL_ITERS`（默认 8）防死循环。
- 完整历史在 session 中 → 纯对话追问 & 带工具追问都自然支持。

## 4. 工具注册机制（tools/）

- `Tool` 基类字段：`name`、`description`、`parameters`(JSON Schema dict)、`execute(args, session) -> str`。
- `registry`：注册所有工具；`schemas()` 生成 function-calling 的 `tools` 列表；`execute(name, args, session)` 按 name 分发，未知工具/参数错误返回结构化错误。
- 手写 JSON Schema，不引 pydantic，体现"从零"。
- 四个工具（超过最低三个）：
  - `calculator`：安全表达式求值（AST 白名单，非 `eval`）。
  - `search`：mock，返回预置结果。
  - `todo`：`add` / `list` / `complete`，按 session 持久化。
  - `weather`：mock，按城市返回固定天气。

## 5. Session 管理（session.py）

- 每个 `--session <id>` = 一个独立 JSON 文件 `sessions/<id>.json`，存 message 历史 + 元数据 + todos。
- 窗口1、窗口2 = 两个 id，完全隔离；同 id 重进即续聊。
- `SessionStore.load/save`，文件锁避免并发写坏。

## 6. Context 管理 + 压缩（context.py）

- **塞入 context**：system prompt、工具 schema、召回的相关 memory 块、最近 N 条原始消息、更早消息的滚动摘要。
- **轮次限制**：单回合工具迭代上限（防循环）+ 会话消息数/估算 token 超阈值触发压缩。
- **基础压缩**：超阈值时调 LLM 把较早消息压成一段摘要，保留最近 N 条原文。重要事实已进长期 memory，压缩不丢关键信息。

## 7. 三层可插拔 Memory（memory/）

经典 agent memory 分层：

| 层 | 存什么 | 检索 | 默认后端 | 可选后端 |
|---|---|---|---|---|
| 短期 / 工作记忆 | 最近 K 轮原始对话 | 直接取最近 | 文件/内存 | Redis（TTL） |
| 情节记忆 episodic | 早期对话滚动摘要 | 语义 top-k | 本地向量 | Qdrant/Chroma |
| 语义 / 画像记忆 | 用户画像、长期事实、偏好 | 语义 top-k | 本地向量 | Qdrant/Chroma |

- **接口化**：`ShortTermStore`、`VectorStore` 两个接口，`.env` 切换实现；缺依赖/缺 key 自动降级默认后端。
- **写路径（蒸馏）**：回合结束由 `extractor` 用 LLM 抽取值得长期记的事实/画像 → 带 embedding upsert；用语义相似度去重/合并，保持画像紧凑（有界"无限"记忆）。
- **读路径（混合检索）**：每回合组装 context 前，短期取最近 K 轮 + 长期把 query embedding 后取 top-k，合并注入到 system 之后、历史之前的"相关记忆"块。
- **embedding**：provider 独立可配（GLM/豆包有 embedding 接口）；无 embedding 时降级关键词/BM25 检索，保证测试与降级可用。
- **召回时机与放置（README 重点）**：召回发生在"每个用户回合、LLM 调用之前"；放置为独立"相关记忆"块，位于 system prompt 之后、对话历史之前，避免与逐字历史混淆。

## 8. 异常处理 & Trace

- LLM 调用：超时 + 有限次重试（指数退避）。
- 工具执行：try/except，错误作为 tool 结果回喂模型自愈。
- Parser：格式跑偏走文本 fallback（正则/JSON 宽松解析）。
- 迭代上限守卫。
- Trace：每步（用户输入 / thought / 工具名+参数 / 工具结果 / 最终答案）终端彩色打印 + 追加 `sessions/<id>.trace.jsonl`。

## 9. 测试

- **单元（不需 API）**：mock LLM client 返回预设响应序列，测 loop 决策、parser、registry、四工具、session 隔离、压缩触发、memory 召回/去重。
- **集成 demo（真实 API）**：`demo/demo_weather_todo.py` 跑通"查天气 + 记待办"完整场景，供录屏；`test_integration.py` 在缺 env key 时 skip。

## 10. 架构设计题

`docs/ARCHITECTURE_QA.md` 单独成文，回答模块一~五全部问题，并与本实现相互印证（压缩策略、memory 分层与召回、异步工具设计、session busy 处理、Claude Code vs OpenAI function-calling 输出方式对比、OpenHands 状态机）。

## 11. 交付物清单

- [ ] 代码（agent/ + cli.py + tools + memory）
- [ ] 测试用例（tests/）
- [ ] README（运行方式 / 系统设计 / memory 召回时机与放置）
- [ ] `docs/ARCHITECTURE_QA.md`（架构设计题答案）
- [ ] `docs/PROMPTS_LOG.md`（AI Prompt 与问题解决记录）
- [ ] demo 脚本（录屏用）
- [ ] `.env.example` + `requirements.txt`

## 12. 环境变量（.env.example 草案）

```
# Chat LLM (OpenAI-compatible)
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=
LLM_MODEL=deepseek-chat

# Embedding（可选；缺失则关键词 fallback）
EMBED_BASE_URL=
EMBED_API_KEY=
EMBED_MODEL=

# Memory 后端（默认 file/local；可切 redis/qdrant）
SHORT_TERM_BACKEND=file        # file | redis
VECTOR_BACKEND=local           # local | qdrant | chroma
REDIS_URL=
QDRANT_URL=

# Runtime
MAX_TOOL_ITERS=8
CONTEXT_MAX_MESSAGES=20
RECENT_KEEP=8
```
