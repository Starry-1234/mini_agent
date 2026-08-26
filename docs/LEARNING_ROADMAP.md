# 学习 Roadmap — 用 Starry Code 学后端工程能力

> **目标**：用本项目作为学习载体，掌握 AI 应用工程师的工程硬通货：
> Redis、向量数据库、异步、外部 API 集成、降级策略、可观测性
>
> **周期**：6-8 周（每周约 10-15 小时）
>
> **方法**：每个 phase 有三件事——**学什么 / 项目做什么 / 简历写什么**

---

## 📐 整体路径

```
R0 (本文件)        —— 设计
R1 (W1-W2)         —— Redis + 真 API 数据源
R2 (W3-W4)         —— Qdrant 向量数据库
R3 (W5-W6)         —— 异步 + 异常处理/降级
R4 (W7-W8, 可选)  —— 可观测性（OpenTelemetry + Prometheus）

每个 phase 完成后：
 ✅ 能跑的新功能
 ✅ 写入简历的具体成果
 ✅ 真正学会（不是"看过"）的技术
```

---

## 🎯 学习目标对照表

| 技术领域 | 学到什么 | Phase | 项目中的体现 | 简历可写 |
|---|---|---|---|---|
| **Redis** | TTL/Pub-Sub/Pipeline/连接池 | R1 | ShortTermStore 真用 Redis | "为 Agent 设计 Redis 缓存层" |
| **Qdrant** | HNSW/ANN/distance metrics | R2 | 跨会话语义搜索 | "实现 HNSW 向量检索" |
| **Async** | asyncio.gather/TaskGroup/超时 | R3 | 并发 tool 调用 | "用 asyncio 重构外部依赖" |
| **降级** | Circuit breaker / Retry / Fallback | R3 | 单源熔断 + mock fallback | "实现 circuit breaker 模式" |
| **API 集成** | Rate limit / Pagination / 错误码 | R1 | 真 GitHub/Remotive/HN | "集成 3 个真实 API 加降级" |

---

## 📋 Phase 详情

### R1 — Redis + 真数据源（2 周）

#### R1 W1: Redis 入门

**学什么**：
- Redis in Action Ch.1-3（数据结构、TTL、persistence、Pub/Sub）
- `redis-py` Python 客户端文档
- 连接池、pipeline 的基本用法

**项目做什么**：
1. `docker-compose.yml` 解开 `redis` 服务注释
2. `.env` 加 `SHORT_TERM_BACKEND=redis` + `REDIS_URL=redis://redis:6379/0`
3. `runtime.py` 的 `build_memory()` 已有 redis 分支，验证它真跑 Redis
4. 加 session TTL（24 小时过期）
5. fallback 逻辑：Redis 连不上 → 自动用 in-memory + 警告

**关键代码位置**：
- `starry_code/memory/short_term.py` — `RedisShortTermStore` 已实现，需 verify
- `starry_code/runtime.py:73-77` — build_memory 的 redis 分支

**学习验证**：
- [ ] `redis-cli PING` 返回 PONG
- [ ] 创建一个 session，kill docker，重启，验证数据还在（persistence）
- [ ] 故意把 REDIS_URL 改成错的端口，验证 fallback 工作
- [ ] 用 `redis-cli MONITOR` 实时看 agent 在做什么命令

#### R1 W2: 真数据源 + Redis 应用

**学什么**：
- GitHub REST API 文档（search/repositories）
- Remotive Jobs API（pagination、response 结构）
- HN Algolia Search（query syntax、numericFilters）
- HTTP rate limit / backoff / circuit breaker 基础
- API key 管理（不在代码里硬编码）

**项目做什么**：
1. 写真 3 个 `RealXxxProvider`（GitHub / Remotive / HN Algolia）
2. 每个 source 自己的 rate limit 处理：
   - GitHub: 60/h unauthenticated, 5000/h with PAT
   - Remotive: 无限制，但要带 User-Agent
   - HN: 无限制
3. 加 `TREND_PROVIDER=real` env var 切换 mock/real
4. 写真数据失败时**自动 fallback**到 mock（带明显标记）
5. 加 Redis 缓存（每个 topic 缓存 6 小时，避免重复请求）

**关键代码位置**：
- `starry_code/tools/tech_trend_sources/` 新增 `github_trending_real.py` 等
- `composite.py` 加 provider 选择逻辑

**学习验证**：
- [ ] `TREND_PROVIDER=real` 跑 tech_trend，trace 里看到真 API 请求
- [ ] 模拟网络断开（断 docker 桥），circuit breaker 触发 → 自动降级
- [ ] Redis 缓存命中（同一 topic 第二次调 < 100ms）
- [ ] 写真 GitHub 失败时 fallback 到 mock（带 "[MOCK fallback]" 标记）

**简历可写**：
- "为 Python Agent 系统设计 Redis 缓存层（TTL + connection pool + fallback）"
- "集成 GitHub/Remotive/HackerNews 三个真实 API，实现 circuit breaker + 降级"

---

### R2 — Qdrant 向量数据库（2 周）

#### R2 W3: Qdrant 入门

**学什么**：
- Qdrant 文档（tutorials/getting-started）
- ANN 算法基础：HNSW vs IVF vs PQ
- Distance metrics：Cosine / Euclidean / Dot Product
- Embedding 维度和检索质量的关系

**项目做什么**：
1. `docker-compose.yml` 解开 `qdrant` 服务注释
2. `.env` 加 `VECTOR_BACKEND=qdrant QDRANT_URL=http://qdrant:6333`
3. `runtime.py:80-88` 已有 qdrant 分支，verify 真跑
4. 加 `/search` 斜杠命令："我之前聊过 Rust吗？"（语义搜索过去会话）
5. 加 collection 自动创建

**学习验证**：
- [ ] Qdrant dashboard (http://localhost:6333/dashboard) 能看到 collection
- [ ] `/search rust` 找到之前包含 rust 的 session（即使措辞不同）
- [ ] 调 distance threshold，看 recall 变化

#### R2 W4: Vector DB 工程化

**学什么**：
- 批量 upsert 性能
- 索引调参（m, ef_construct, ef）
- Embedding 模型选型（text-embedding-v3 vs bge-large）
- Quantization（减少存储）

**项目做什么**：
1. 用真 embedding（aliyun text-embedding-v3）重新索引历史 session
2. 加 collection metadata（topic, timestamp, session_id）
3. 加 `/similar <session-id>` 斜杠命令
4. 性能基准：1000 个 session 搜索延迟 < 100ms

**简历可写**：
- "实现 HNSW 向量检索 + 跨会话记忆"
- "用 Qdrant 替换 naive JSONL 存储，搜索延迟从 O(n) 降到 O(log n)"

---

### R3 — 异步 + 异常处理（2 周）

#### R3 W5: 异步

**学什么**：
- Python asyncio 官方文档
- `asyncio.gather` / `asyncio.TaskGroup`（3.11+）
- 超时、取消、`shield`
- httpx 异步客户端

**项目做什么**：
1. `composite.py` 改用 `asyncio.gather` 并发 3 个 source fetch
2. 加超时控制（单个 source 最多 5 秒）
3. 加 `httpx.AsyncClient` 写真数据源

**学习验证**：
- [ ] 3 个 source 并发，总延迟 = max(单个延迟) 不是 sum
- [ ] 某个 source 超时，其他仍能返回

#### R3 W6: Circuit breaker + 降级

**学什么**：
- Circuit breaker 模式（Hystrix 论文 / Polly 实现）
- Retry with exponential backoff + jitter
- Graceful degradation 设计
- Bulkhead pattern

**项目做什么**：
1. 写真数据源每个加 circuit breaker（3 次失败 → 1 小时冷却）
2. Circuit open 时**返回 mock 数据 + 标记 "[STALE]"**
3. Retry 策略：3 次退避（1s, 2s, 4s + jitter）
4. 加 `structlog` 结构化日志（每次 tool call 都有 trace_id）

**简历可写**：
- "用 asyncio + circuit breaker 重构外部依赖处理"
- "实现 3 层降级策略（real → mock → error message）"

---

### R4 — 可观测性（1 周，可选）

**学什么**：
- OpenTelemetry Python SDK
- Prometheus client
- Grafana 入门

**项目做什么**：
1. 加 OTEL instrumentation：LLM call / tool exec / memory write 全链路 trace
2. `/metrics` 端点暴露：tool 成功率、LLM 延迟、memory hit rate
3. `docker-compose.yml` 加 Prometheus + Grafana

**简历可写**：
- "用 OpenTelemetry + Prometheus 给 Agent 加可观测性"

---

## 📝 每个 Phase 的工作流

```
1. 看 1-2 篇官方文档/tut
2. 写 5-10 行 demo 代码试一下
3. 在本项目里写真功能（带测试）
4. 跑端到端验证（看 trace + 效果）
5. commit + push
6. 写 1-2 句学到的东西（记到 ROADMAP_LEARNINGS.md）
```

---

## 🎯 不在范围内（避免 scope creep）

- ❌ Kubernetes / Helm（solo 项目用 docker compose 够了）
- ❌ OAuth / 多用户（除非真要做产品）
- ❌ Frontend framework（已有 Streamlit MVP）
- ❌ 模型 fine-tuning（数据不够 + 时间不值）
- ❌ Custom embedding（aliyun text-embedding-v3 够用）

---

## ✅ 完成判据（每周自检）

| Phase | 完成时能 demo 的 |
|---|---|
| R1 W1 | 关 docker 重启 CLI，session 还在；连不上 redis 仍能用 |
| R1 W2 | tech_trend 返回的数据有时是 mock、有时是真实（标记清楚） |
| R2 W3 | `/search` 跨会话语义搜索工作 |
| R2 W4 | 1000 个 session 时搜索 < 100ms |
| R3 W5 | 3 个源并发，总延迟 = 最慢那个 |
| R3 W6 | 拔网线后跑 agent，仍能完成（带 [STALE] 标记） |
| R4 | 在 Grafana 看到 coach 的请求拓扑 |

---

## 📚 推荐阅读

| 主题 | 资源 |
|---|---|
| Redis | 《Redis in Action》 Josiah Carlson（Ch.1-5）|
| Qdrant | https://qdrant.tech/documentation/ + Qdrant 101 tutorial |
| Async | https://docs.python.org/3/library/asyncio.html + "Concurrency in Python" tutorial |
| Circuit breaker | https://martinfowler.com/bliki/CircuitBreaker.html |
| 可观测性 | https://opentelemetry.io/docs/languages/python/ |

---

## 🔄 Roadmap 调整

学完一个 phase 后回来更新：
- 实际用时 vs 估计
- 踩过的坑
- 哪些简历可写点更突出

文件路径：`docs/LEARNING_ROADMAP.md`（本文件）
学习笔记：`docs/ROADMAP_LEARNINGS.md`（待创建）

---

**从 R1 W1 开始**——Redis 入门 + 本项目真用 Redis。下一周写真数据源（穿插 R1 W2）。