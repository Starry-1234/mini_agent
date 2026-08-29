# Redis 验证操作手册

> **目的**：作为用户，验证 Redis 在你的 Starry Code 项目里真的被使用，不只是装样子。
>
> **目标**：5 个 Level，从 30 秒到 5 分钟，按需选用。

---

## Level 1: 30 秒 — 验证 Redis 在跑且被 agent 用

```bash
# 1. Redis 进程健康
docker ps --filter "name=starry-code-redis" --format "table {{.Status}}"

# 2. Redis 里有 agent 写入的 key
docker exec starry-code-redis redis-cli DBSIZE
docker exec starry-code-redis redis-cli --scan --pattern "st:*"

# 3. MONITOR 实时看 agent 写入什么
docker exec starry-code-redis redis-cli MONITOR
# （另一 terminal）docker compose run --rm agent --once "ping"
# 你会看到：RPUSH "st:..." "..." + LTRIM "st:..." "-200" "-1" + PING
```

**通过标准**：MONITOR 看到 agent 写 `RPUSH st:*` 命令。

**已验证**：✅（2026-08-27 commit `8e35769` 之后）

---

## Level 2: 2 分钟 — 验证跨重启持久化

```bash
# 1. agent 写入
docker compose run --rm agent --session persist_test --once "记住：项目用 Redis"

# 2. 看 key 内容
docker exec starry-code-redis redis-cli LRANGE st:persist_test 0 -1 | head -3
# 期望看到 [{"role":"user",...}, {"role":"assistant",...}]

# 3. 重启 Redis 容器
docker restart starry-code-redis

# 4. 数据还在吗？
docker exec starry-code-redis redis-cli DBSIZE
docker exec starry-code-redis redis-cli LRANGE st:persist_test 0 -1 | head -3
# 期望：内容**完全一致**

# 5. 续会测试 LLM 真的能从 Redis 读到
docker compose run --rm agent --session persist_test --once "刚才让你记住什么？"
# 期望：assistant 说出"项目用 Redis"
```

**通过标准**：步骤 4 数据一致 + 步骤 5 LLM 召回原话。

**已验证**：✅

---

## Level 3: 5 分钟 — 性能 + Fallback

### 3.1 LTRIM 上限（maxlen=200）

```bash
# 通过 MONITOR 看 push 是否触发 LTRIM（每次 push 都该有）
docker exec starry-code-redis redis-cli MONITOR
# （另一 terminal）docker compose run --rm agent --once "test"
# 你会看到：RPUSH ... + LTRIM "-200" "-1"  ← 每次都对
```

**通过标准**：每次 RPUSH 后跟 LTRIM。

**已验证**：✅ MONITOR 输出确认。

### 3.2 TTL / 闲置时间

```bash
# 看某个 key 多长时间没被访问（秒）
docker exec starry-code-redis redis-cli OBJECT IDLETIME st:persist_test
```

**通过标准**：数字 > 0（key 确实存在一段时间了）。

### 3.3 Fallback（Redis 挂了仍能用）

```bash
# 1. 关 redis
docker stop starry-code-redis

# 2. 跑 agent
docker compose run --rm agent --once "ping" 2>&1 | grep -i "redis\|fall"

# 期望 stderr: "[starry] redis unreachable (RuntimeError('PING failed'));
#         falling back to in-memory"

# 期望 stdout: agent 仍正常回答

# 3. 启回 redis
docker start starry-code-redis
```

**通过标准**：看到 fallback 警告 + agent 不崩。

**已验证**：✅ `[starry] redis unreachable (RuntimeError('PING failed')); falling back to in-memory` + agent 正常输出 4 个画像问题。

---

## Level 4: 5 分钟 — 多 Session 隔离

```bash
# 1. 跑两个独立 session
docker compose run --rm agent --session alice --once "我是 Alice" 2>&1 | tail -1
docker compose run --rm agent --session bob   --once "我是 Bob" 2>&1 | tail -1

# 2. 看 Redis 里的 key（应有两个独立 list）
docker exec starry-code-redis redis-cli --scan --pattern "st:*"
# 看到 st:alice 和 st:bob

# 3. 各自内容（应互不串数据）
docker exec starry-code-redis redis-cli LRANGE st:alice 0 -1 | head -3
docker exec starry-code-redis redis-cli LRANGE st:bob   0 -1 | head -3
# alice: 含"Alice"
# bob:   含"Bob"
```

**通过标准**：两个 session 数据**完全独立**。

**已验证**：✅ alice session 含 "Alice 你好"，bob session 含 "嗨 Bob"。

---

## Level 5: 高级 — Redis 自身健康

```bash
# 5.1 内存使用
docker exec starry-code-redis redis-cli INFO memory | grep -E "used_memory_human|maxmemory_human"

# 5.2 连接数
docker exec starry-code-redis redis-cli INFO clients | grep connected_clients

# 5.3 操作计数
docker exec starry-code-redis redis-cli Info stats | grep -E "total_commands|instantaneous_ops"

# 5.4 大 key 扫描
docker exec starry-code-redis redis-cli --bigkeys

# 5.5 慢查询日志
docker exec starry-code-redis redis-cli SLOWLOG GET 10

# 5.6 持久化配置（应 appendonly=yes）
docker exec starry-code-redis redis-cli CONFIG GET appendonly
```

---

## 故障排查速查

| 现象 | 查什么 | 修什么 |
|---|---|---|
| `DBSIZE = 0` | MONITOR 看 agent 跑时 Redis 有无反应 | ping() 失败 → 自动 fallback 到 in-memory |
| 续会 LLM "不知道" | `LRANGE st:<sid> 0 -1` | history 为空 → agent 没成功 push |
| Redis 重启后数据没 | `CONFIG GET appendonly` | 应该是 yes（compose 已配 `--appendonly yes`）|
| 端口冲突 | `netstat -an \| grep 6379` | 关掉本地 redis，让出端口 |
| `DBSIZE` 一直不增 | 看 `docker compose logs starry-code-redis` | 检查 AOF 配置 |

---

## 📋 总览：每个 Level 跑过 / 没跑

| Level | 测什么 | 状态 |
|---|---|---|
| 1: MONITOR | Redis 真被 agent 调用 | ✅ 跑过，通过 |
| 2: 跨重启 + 续会 recall | AOF 持久化 + LLM 真读 | ✅ 跑过，通过 |
| 3.1: LTRIM maxlen=200 | push 触发 LTRIM | ✅ MONITOR 跑过，间接证明 |
| 3.2: OBJECT IDLETIME | TTL 工作 | ❌ 没跑（不需要，redis 默认无 TTL） |
| 3.3: Fallback | Redis 挂了不崩 | ✅ 跑过，通过 |
| 4: 多 session 隔离 | session 间不串数据 | ✅ 跑过，通过 |
| 5: Redis 自身健康 | 性能/连接/大key | ❌ 没跑（性能调优用，平时不需要） |

**核心 4 项（Level 1/2/3.1/3.3/4）全部跑过 ✅，符合预期。**