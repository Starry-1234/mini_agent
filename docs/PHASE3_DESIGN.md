# Phase 3 — Real Data + Productization Design

> **Status**: design only. No code changes required.
> **Scope**: replace the `_MockTrendProvider` mock with real data sources,
> add three new coaching tools, plan a CLI→Web UI migration.

## 0. Why this phase

Phase 1 made the coach persona real; Phase 2 gave it memory + plan cache +
artifact offload + dynamic re-triggers. But the data underneath the
`tech_trend` tool is still a 12-row hard-coded table. The user can ask
about Rust or Kubernetes and get a sensible-looking answer, but it
doesn't reflect today's GitHub stars, this week's job postings, or
trending discussion. Before we ship to anyone beyond ourselves, the
data needs to be real.

Phase 3 also widens the coaching surface: right now the coach can
*describe* a path but can't help the learner *evaluate their current
skills* against the target role, can't *generate interview questions*
tailored to the plan, and can't *break a goal down into a concrete
ship-able project*. Three new tools fill those gaps.

Finally, the CLI is great for solo tinkering but not for product
distribution. We lay out a three-stage migration to Web so the coach
can reach people who don't have Python installed.

---

## 1. `tech_trend` real data sources

### 1.1 Source selection

We pick three sources, each covering a different signal:

| Signal                          | Source                        | Why                                     | Compliance / TOS                  |
|---------------------------------|-------------------------------|-----------------------------------------|-----------------------------------|
| **Adoption velocity** (momentum) | **GitHub Trending API** + search | Stars/created-date is the cleanest proxy for "are people actively building with this?"  | Public, unauthenticated, no TOS issues                |
| **Hiring signal**               | **Remotive Jobs API**         | Remote-friendly JSON, indexed by tag; "is there demand for someone who lists this skill?" | Public API, attribute required     |
| **Discussion signal**            | **HackerNews Algolia Search** | Real-time developer chatter; surfaces things GitHub stars lag on | Public, no auth                    |

We deliberately exclude:

- **Lagou / Boss直聘** — TOS prohibit scraping; auth walls; legal grey.
- **Stack Overflow Annual Survey** — yearly snapshot, 12-month-stale on day 1.
- **BuiltIn / Levels.fyi** — auth-gated, no public API for our use case.
- **LinkedIn Jobs** — closed API, requires partner agreement.

### 1.2 Aggregator design

Replace `_MockTrendProvider` with a `CompositeTrendProvider` that fans
out to N sources, normalizes results, and returns a single weighted
score. The `TechTrendTool` interface stays unchanged.

```
TrendProvider (Protocol)             ← already exists
    │
    ├── GitHubTrendProvider         ← new
    ├── RemotiveJobsProvider        ← new
    ├── HNAlgoliaProvider           ← new
    └── CompositeTrendProvider      ← new (aggregator)
            │
            ▼
        TechTrendTool               ← unchanged
```

**Composite scoring** (per `topic`):

```
final_score = 0.40 * github_score   # 0-10, normalized from stars delta
           + 0.45 * jobs_score       # 0-10, normalized from postings count
           + 0.15 * hn_score         # 0-10, normalized from HN points/24h

direction = majority vote of the three sources' direction labels
            (rising / stable / falling) with override: if jobs_count = 0
            for 60+ days → "falling" regardless

learning_window_weeks = heuristic from jobs_score +
                         historical baseline for the topic
                         (e.g. "java" baseline = 8, "rust" = 12)
                         — see §1.4

notes = the top-1 sentence from each source, joined.
```

**Source-specific normalization** lives in each provider; the
composite just multiplies and sums.

### 1.3 Per-source spec

#### GitHubTrendProvider

- **Endpoint**:
  `GET https://api.github.com/search/repositories?q={topic}+language:{lang}&sort=stars&order=desc&per_page=20`
- **Auth**: optional unauthenticated (60 req/h); PAT bumps to 5000/h.
- **What we extract**:
  - count of repos with `pushed:>YYYY-MM-DD` and the topic in name/description
  - median star count of those repos
  - delta: stars added in last 30 days (sum across repos)
- **Score**: `min(10, log10(30d_delta_stars + 1) * 4)` clamped.
- **Cache**: 6 hours (GitHub data is slow-moving).
- **Failure mode**: rate-limited / 5xx → fall back to last cached value
  with a `_stale=True` field; if no cache, return `direction="unknown"`.

#### RemotiveJobsProvider

- **Endpoint**: `GET https://remotive.com/api/remote-jobs?search={topic}&limit=50`
- **Auth**: none.
- **What we extract**:
  - total jobs returned for query
  - jobs posted in last 7 days
  - top 3 job titles (for the `notes` field)
- **Score**: `min(10, log10(jobs_last_7d + 1) * 3)` clamped.
- **Cache**: 2 hours (jobs are time-sensitive).
- **Failure mode**: empty result for known-good topic → likely
  Remotive changed their taxonomy; surface as `direction="unknown"`
  rather than fabricating 0.

#### HNAlgoliaProvider

- **Endpoint**: `GET https://hn.algolia.com/api/v1/search?query={topic}&tags=story&numericFilters=points>10&hitsPerPage=30`
- **Auth**: none.
- **What we extract**:
  - hit count for last 30 days
  - median points of those hits
- **Score**: `min(10, log10(hits_30d + 1) * 3.5)` clamped.
- **Cache**: 1 hour (fast-moving).
- **Failure mode**: 5xx → return `direction="unknown"`.

### 1.4 Learning window heuristic

Real `learning_window_weeks` is not derivable from any data source
directly. Plan: maintain a small **internal baseline table** (json
file, ~50 topics) that we curate from aggregate experience. Examples:

```json
{
  "rust":          {"min_weeks": 8,  "median_weeks": 12, "max_weeks": 24},
  "go":            {"min_weeks": 4,  "median_weeks": 8,  "max_weeks": 14},
  "kubernetes":    {"min_weeks": 6,  "median_weeks": 10, "max_weeks": 18},
  ...
}
```

Adjust by `final_score`:
- score ≥ 8 → use `min_weeks` (easy path, lots of community help)
- 4 ≤ score < 8 → use `median_weeks` (default)
- score < 4 → use `max_weeks` (sparser learning resources)

If topic not in baseline table, default to `median_weeks = 12`.

### 1.5 Compliance notes

- All three sources are public APIs; no auth, no scraping.
- Attribute each result to its source in the `notes` field.
  ("GitHub: 1.2k stars added in 30d. Remotive: 47 jobs in 7d. ...")
- Document the data sources in `docs/DATA_SOURCES.md` so users know
  what they're getting.
- Keep `_MockTrendProvider` as the default for offline/dev mode; user
  must explicitly switch to `CompositeTrendProvider` via env var
  (`TREND_PROVIDER=composite`).

### 1.6 Caching + cost control

- **Per-source HTTP cache** (file-based, ~10MB cap), TTL per source.
- **Per-topic coalescing**: if two questions about the same topic land
  within 60s, dedupe — composite provider sees only one outbound call.
- **Circuit breaker**: after 3 consecutive 5xx, suspend the failing
  source for 1 hour. Coach prompt handles missing data gracefully.

---

## 2. New tools

All three follow the existing `Tool` Protocol (base.py). They reuse
session.plan_cache for state when relevant.

### 2.1 `skill_assess`

Goal: take a learner's self-described current skills + target role,
return a structured gap analysis.

**Parameters** (OpenAI-compatible JSON schema):

```json
{
  "type": "object",
  "properties": {
    "current_skills": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Self-reported skills, e.g. ['Python 基础', '用过 Flask']"
    },
    "target_role": {
      "type": "string",
      "description": "Target job title, e.g. 'Go 后端工程师'"
    },
    "evidence": {
      "type": "string",
      "description": "Free-form: resume snippet / project list / GitHub URL"
    }
  },
  "required": ["target_role", "evidence"]
}
```

**Returns** (ToolResult.ok content, JSON):

```json
{
  "target_role": "Go 后端工程师",
  "gap_summary": "已掌握 X、Y；缺 Z；建议补 W",
  "categories": [
    {
      "name": "已掌握",
      "items": ["Python 基础语法", "REST 概念", "Git"]
    },
    {
      "name": "核心缺口（必须补）",
      "items": [
        {"skill": "Go 语法", "priority": "P0", "est_weeks": 4},
        {"skill": "SQL/MySQL", "priority": "P0", "est_weeks": 3}
      ]
    },
    {
      "name": "加分项（学了有优势）",
      "items": ["k8s 基础", "Docker"]
    }
  ],
  "recommended_next_action": "先学 Go 语法基础，4 周后做 CLI 项目作为第一个 milestone"
}
```

**Implementation note**: the actual scoring logic can stay LLM-driven
(no external data needed). The tool just gives the LLM a structured
shape to fill in. Latency: 1 LLM call (returned in same turn).

### 2.2 `interview_prep`

Goal: given target role + level + optional focus areas, generate N
interview questions + reference answer points.

**Parameters**:

```json
{
  "type": "object",
  "properties": {
    "role": {"type": "string", "description": "目标岗位"},
    "level": {
      "type": "enum",
      "values": ["junior", "mid", "senior"],
      "description": "目标职级"
    },
    "n_questions": {
      "type": "integer", "minimum": 3, "maximum": 20, "default": 5
    },
    "focus": {
      "type": "array",
      "items": {"type": "string"},
      "description": "重点考察的技术点（如 ['goroutine', 'context', 'gc']）"
    }
  },
  "required": ["role"]
}
```

**Returns**:

```json
{
  "questions": [
    {
      "id": 1,
      "type": "技术基础",
      "difficulty": "mid",
      "question": "解释 Go 的 goroutine 调度模型（GPM）。",
      "key_points": ["G = goroutine, P = processor, M = machine (OS thread)",
                     "work-stealing 调度", "抢占式调度 (Go 1.14+)"],
      "common_pitfall": "只回答 'M:N 调度' 但不解释三个角色"
    },
    ...
  ],
  "study_links": ["https://...", "..."],
  "estimated_prep_time_hours": 12
}
```

**Implementation note**: prompt-only. No external data. Latency: 1 LLM
call, returns in same turn. If `focus` overlaps with current
`plan_cache.stage`, weight those questions higher.

### 2.3 `project_drive`

Goal: take a goal + time budget + level, return a project skeleton the
learner can ship and put on a resume.

**Parameters**:

```json
{
  "type": "object",
  "properties": {
    "goal": {"type": "string", "description": "学习目标"},
    "time_budget_hours": {"type": "integer", "minimum": 8, "maximum": 500},
    "current_level": {
      "type": "enum", "values": ["beginner", "intermediate", "advanced"]
    },
    "target_resume_section": {
      "type": "enum",
      "values": ["projects", "open_source", "side_quest"],
      "description": "简历哪个 section 放这个项目"
    }
  },
  "required": ["goal", "time_budget_hours"]
}
```

**Returns**:

```json
{
  "project_name": "Mini CLI Coach",
  "tagline": "20 小时做出能写进简历的 LLM 教练 CLI",
  "must_have_features": [
    "OpenAI-compatible chat (chat loop)",
    "3 tools (calculator + todo + custom)",
    "session persistence (JSON files)",
    "window title + ANSI colors"
  ],
  "tech_stack_suggested": ["Python 3.12", "openai SDK", "click"],
  "milestones": [
    {"week": 1, "deliverable": "MVP: chat + 1 tool", "acceptance": "能在 REPL 里调用 calculator"},
    {"week": 2, "deliverable": "All 3 tools + session save", "acceptance": "退出后重进能续"}
  ],
  "interview_talking_points": [
    "为什么先做 CLI 再做 Web",
    "如何保证 session 持久化的原子性",
    "对 reasoning model 的 thinking-block 处理"
  ],
  "stretch_goals": ["streaming responses", "tool auto-naming", "memory recall"]
}
```

**Implementation note**: prompt-only. Latency: 1 LLM call. The
generated project should be **small enough to ship** and **big enough
to demo a non-trivial skill**.

### 2.4 Tool registration

All three go into `build_default_registry()` alongside the existing 7.
Coach prompt gets a one-line addition telling it when to reach for
each one. We add `skill_assess` and `interview_prep` to the
"必做行为" list, `project_drive` as a stretch goal the LLM may
suggest once `plan_cache.long_term_goal` is set.

---

## 3. Web UI upgrade path

Three stages, each independently shippable. The CLI is the source of
truth for all business logic; the Web UI is a thin transport layer.

### 3.1 Stage 1 — Streamlit MVP (1–2 weeks)

**Why Streamlit first**:
- Pure Python — reuses `run_turn()`, all 7 tools, `MemoryManager`,
  `SessionStore` with zero code changes.
- `st.chat_message()` + `st.chat_input()` give a Claude-Code-style
  chat UI in ~50 lines.
- Local-only deployment: `streamlit run app.py` from the project dir.
- Streaming support via `st.write_stream()`.

**Layout sketch**:
```
┌─────────────────────────────────────────────┐
│ ✦ Starry Coach              [历史] [新会话] │
├─────────────────────────────────────────────┤
│ [user] Rust 行情怎么样                       │
│ [assistant] 行情数据...调用了 tech_trend     │
│   ┌─ tool_call ────────────────────────┐   │
│   │ tech_trend(topic="rust")           │   │
│   │ ↳ {direction: rising, demand: 8}   │   │
│   └────────────────────────────────────┘   │
├─────────────────────────────────────────────┤
│ [发送]                                     │
└─────────────────────────────────────────────┘
```

Side panel: session list, plan_cache display, todo list.

**Code surface**:
- `web/streamlit_app.py` — top-level
- `web/components/sidebar.py` — session picker
- `web/components/chat.py` — message rendering + streaming
- `web/components/plan_panel.py` — plan_cache + todo visualization

**Constraint**: one user per process (Streamlit session state is
per-tab). Fine for solo / friend / beta.

### 3.2 Stage 2 — FastAPI + Jinja2 (3–4 weeks)

**Why this stage**:
- Multi-user: each browser gets its own session id via cookie.
- Async I/O lets us handle many concurrent sessions without threads.
- Server-Sent Events for streaming the LLM response (works in plain
  HTML; no WebSocket gymnastics needed).

**API surface**:
```
GET  /                            → index.html (chat UI)
POST /api/sessions               → create new session
GET  /api/sessions               → list sessions
GET  /api/sessions/{id}          → session detail (history + plan)
POST /api/sessions/{id}/messages → send message, returns SSE stream
GET  /api/sessions/{id}/artifacts/{call_id} → read artifact
```

**Architecture**:
```
┌────────┐  HTTP   ┌──────────┐   in-proc   ┌────────────────┐
│  HTML  │ ←────→ │ FastAPI  │ ←────────→ │ run_turn()     │
│  /JS   │  SSE   │ (async)  │            │ MemoryManager  │
└────────┘        └──────────┘            │ ToolRegistry   │
                                          │ TraceLogger    │
                                          └────────────────┘
```

**Auth**: optional cookie-based session token. Skip OAuth for MVP
(this is a learning tool, not a financial product).

### 3.3 Stage 3 — FastAPI + React/Vue (2–3 months)

**Why this stage**:
- Real-time collaborative features (study groups).
- Better mobile experience (PWA).
- Plugin ecosystem for third-party tool packs.
- Multi-model comparison view (compare answers from 2 LLMs side by side).

**Frontend stack**: React + Vite + TanStack Query + Tailwind.
**State**: server is source of truth; client cache invalidates on SSE.
**Real-time**: WebSocket for streaming + collaborative cursors.
**Storage**: Postgres for sessions, Redis for hot cache, S3 for
artifacts (if we hit disk limits).

**Migration path**: FastAPI backend stays. Old Jinja2 frontend
coexists with React via `/app/*` vs `/v2/*` routes during transition.

### 3.4 Cross-cutting concerns

- **Observability**: structured logs, request tracing (OpenTelemetry),
  per-session LLM cost counter.
- **Cost guardrails**: rate-limit per user; per-session token budget
  with warning at 80%.
- **Privacy**: sessions stored encrypted at rest; "delete my account"
  must cascade to all sessions + memories.
- **Multi-tenancy**: from Stage 2, namespace sessions by user_id.

---

## 4. Migration & risk notes

### 4.1 tech_trend migration

- **Backward compatibility**: `TrendProvider` Protocol stays. `_MockTrendProvider`
  remains the default. Users explicitly opt into real data via env var.
- **Test impact**: existing tech_trend tests use mock; add a new
  `test_composite_provider.py` that mocks each upstream source.
- **Failure mode**: if any source is down, composite gracefully
  degrades (already designed in §1.3). Worst case: same answer as
  today (mock data) — never worse.

### 4.2 New tools risk

- All three are LLM-prompt-only — no new external dependencies.
- Main risk is **hallucinated interview questions / project features**:
  coach prompt should caveat ("verify before relying on this").
- Add a small unit test asserting the JSON shape, not the content.

### 4.3 Web UI risks

- **Streaming authn**: SSE through proxies (nginx) sometimes buffers;
  may need `X-Accel-Buffering: no` header.
- **Long sessions**: chat UI rendering 1000+ messages gets slow; need
  virtualization (Stage 3).
- **Mobile**: Stage 1 is not mobile-friendly. Stage 2 needs responsive
  CSS. Stage 3 has full PWA.

### 4.4 What we explicitly don't do in Phase 3

- **No RAG over the user's own notes** — Phase 4+.
- **No multi-tenant billing** — Phase 5+, if the product actually gets
  paying users.
- **No mobile native** — Stage 3 PWA is sufficient.
- **No voice I/O** — out of scope; not what learners need.

---

## 5. Acceptance criteria for Phase 3

| Item | Done when |
|------|-----------|
| Real `tech_trend` data | `CompositeTrendProvider` returns data from all three sources; weighted score documented; failure-mode tests pass |
| `skill_assess` works | Coach calls it on first turn; returns gap analysis with at least 3 categories |
| `interview_prep` works | Generates N questions matching role + level; includes key points for each |
| `project_drive` works | Returns a project skeleton < `time_budget_hours` worth of work; milestones sum to budget |
| Web UI v1 | Streamlit app runs locally; user can create session, send message, see tool calls, see plan_cache |
| Tests | New tests cover each new tool's JSON shape; existing 140 tests still pass |

If we hit any of these, Phase 3 ships. If not, we cut scope to
"CompositeTrendProvider + 1 new tool + Streamlit MVP" and defer the rest
to Phase 4.

---

## 6. Timeline (suggested)

| Week | Scope |
|------|-------|
| 1    | CompositeTrendProvider (3 sources + scoring) |
| 2    | `skill_assess` + `interview_prep` + `project_drive` |
| 3    | Streamlit MVP shell + sidebar + chat panel |
| 4    | Plan panel + todo viz + basic styling |
| 5+   | FastAPI migration (deferred — when we have beta users) |

Total Phase 3 estimated: **4 weeks** with one developer.
