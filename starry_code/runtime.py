# agent/runtime.py
from __future__ import annotations
import re

from .config import Settings
from .llm import LLMClient
from .session import Session
from .tools.base import ToolResult
from .tools.calculator import CalculatorTool
from .tools.registry import ToolRegistry
from .tools.search import SearchTool
from .tools.tech_trend import TechTrendTool
from .tools.todo import TodoTool
from .tools.weather import WeatherTool
from .tools.read_artifact import ReadArtifactTool
from .tools.update_plan import UpdatePlanTool
from .tools.skill_assess import SkillAssessTool
from .tools.project_drive import ProjectDriveTool
from .tools.interview_prep import InterviewPrepTool
from .memory.embeddings import MockEmbedder
from .memory.manager import MemoryManager
from .memory.short_term import InMemoryShortTermStore, RedisShortTermStore
from .memory.vector_store import (
    ChromaVectorStore,
    LocalVectorStore,
    QdrantVectorStore,
)
from .context import ContextBuilder
from .prompts import SYSTEM_PROMPT
from .session import _OLD_SYSTEM_PROMPT
from .trace import TraceLogger

# Minimum combined content length (user + assistant, thinking stripped) for a
# turn to be considered "substantive enough" to run LLM fact extraction on.
_MIN_EXTRACT_CHARS = 80

# --- Phase 2: dynamic plan-adjustment triggers ---
# Keywords that suggest the user is asking the coach to revisit the plan.
# These are checked in user_input; a hit causes a tech_trend recheck and
# the response naturally reflects fresh data (the coach prompt then steers
# the LLM toward update_plan if anything truly changed).
_TRIGGER_KEYWORDS = re.compile(
    r"(最新|行情|趋势|前景|还值得|还要不要|该(不该|不该)|换.+方向|调整|重新规划|对一下|核对|行情如何|换方向)",
    re.IGNORECASE,
)
_TRIGGER_INTERVAL_DAYS = 7
_TRIGGER_MILESTONE_TODOS = 3


def _should_extract(user_text: str, answer_text: str) -> bool:
    """Guard LLM cost: only extract facts from substantive turns.

    Computes a combined content length of the user message plus the assistant
    answer (both stripped of surrounding whitespace; the answer should already
    have thinking blocks removed by the caller). Trivial turns like "hi" or
    "what is 2+2?" fall under the threshold and are skipped, avoiding a real
    (slow, paid) LLM extraction call for nothing.
    """
    content_len = len((user_text or "").strip()) + len((answer_text or "").strip())
    return content_len >= _MIN_EXTRACT_CHARS


def build_default_registry(sessions_dir: Path | None = None) -> ToolRegistry:
    """Register the built-in tools:
    calculator, search, todo, weather, tech_trend, update_plan, read_artifact,
    skill_assess, project_drive.

    Phase 2 additions:
    - update_plan: coach's only sanctioned way to mutate plan_cache
    - read_artifact: lets the coach pull back full offloaded tool output

    Phase 3 (W2) additions:
    - skill_assess: structured gap analysis between learner and target role
    - project_drive: project skeleton + milestones from goal + time budget

    Phase 3 (W4b) additions:
    - interview_prep: role+level specific interview questions

    `sessions_dir` is passed to ReadArtifactTool so it validates against
    the SAME directory ContextBuilder writes artifacts to (Bug G fix).
    """
    reg = ToolRegistry()
    reg.register_all([
        CalculatorTool(),
        SearchTool(),
        TodoTool(),
        WeatherTool(),
        TechTrendTool(),
        UpdatePlanTool(),
        ReadArtifactTool(sessions_dir=sessions_dir),
        SkillAssessTool(),
        ProjectDriveTool(),
        InterviewPrepTool(),
    ])
    return reg


# ---- Phase 2: dynamic plan-adjustment triggers ----

def _should_adjust_plan(session: Session, user_input: str) -> tuple[bool, str]:
    """Decide whether to re-check tech trends before composing the reply.

    Returns (should, reason). Three trigger classes:
      - user_keyword: user used words like "行情/趋势/前景/调整" → force recheck
      - stale_7d:     plan_cache.last_updated > 7 days ago → recheck
      - milestone:   ≥3 todos completed but plan still at v0/v1 → suggest re-plan

    These are *advisory* — they only inject a tech_trend result into the
    short-term memory so the LLM has fresh data. The LLM decides whether
    to actually call update_plan based on what it sees.
    """
    if _TRIGGER_KEYWORDS.search(user_input or ""):
        return True, "user_keyword"

    pc = getattr(session, "plan_cache", {}) or {}
    last = pc.get("last_updated")
    if last:
        try:
            from datetime import datetime, timezone, timedelta
            last_dt = datetime.fromisoformat(last)
            if datetime.now(timezone.utc) - last_dt > timedelta(days=_TRIGGER_INTERVAL_DAYS):
                return True, "stale_7d"
        except (ValueError, TypeError):
            pass  # malformed timestamp — ignore, don't crash

    done = sum(1 for t in (session.todos or []) if t.get("done"))
    pc_version = pc.get("version", 0)
    if done >= _TRIGGER_MILESTONE_TODOS and pc_version <= 1:
        return True, f"milestone_{done}_done_v{pc_version}"

    return False, ""


def _inject_trend_for_adjustment(
    registry: ToolRegistry,
    session: Session,
    memory: MemoryManager,
    user_input: str,
    trace: TraceLogger,
) -> None:
    """When a trigger fires, refresh trend data and inject it into the
    short-term memory. The next LLM call will see the new facts without us
    having to mutate plan_cache ourselves — the LLM will call update_plan
    if the data warrants a real change.

    Picks candidate topics from plan_cache.long_term_goal + the user_input,
    deduped, and queries up to 3. Failures are swallowed (best-effort).
    """
    pc = getattr(session, "plan_cache", {}) or {}
    candidates: set[str] = set()

    if pc.get("long_term_goal"):
        candidates.update(
            w for w in re.findall(r"[A-Za-z][A-Za-z0-9.+#-]{1,}", pc["long_term_goal"])
        )
    if user_input:
        candidates.update(
            w for w in re.findall(r"[A-Za-z][A-Za-z0-9.+#-]{1,}", user_input)
        )

    for topic in list(candidates)[:3]:
        result = registry.execute("tech_trend", {"topic": topic.lower()}, session)
        trace.event("trend_recheck", topic=topic, ok=result.ok)
        if result.ok:
            memory.push_turn(session.id, {
                "role": "tool",
                "name": "tech_trend_recheck",
                "content": result.content,
                "topic": topic,
            })


def build_memory(settings: Settings, llm: LLMClient | None) -> MemoryManager:
    """Construct the three-layer memory stack from runtime settings.

    - Embedder: real OpenAI-compatible if `embed_model` and `embed_api_key`
      are configured, otherwise the deterministic `MockEmbedder`.
    - Short-term: redis if `short_term_backend == "redis"`, else in-memory.
    - Vector: qdrant / chroma / local based on `vector_backend`.
    """
    # Embedder
    if settings.embed_model and settings.embed_api_key:
        from .memory.embeddings import OpenAICompatEmbedder
        embedder = OpenAICompatEmbedder(
            settings.embed_api_key,
            settings.embed_base_url,
            settings.embed_model,
        )
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
        vs = LocalVectorStore(
            embedder=embedder,
            path=settings.sessions_dir / "memory.jsonl",
        )

    return MemoryManager(
        embedder=embedder,
        short_term=short_term,
        vector_store=vs,
        llm=llm,
        top_k=5,
    )


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
    """Run one user turn: tool-using loop with bounded iterations.

    Responsibility split:
      - This function calls `session.add_user(user_input)` exactly ONCE,
        before the tool loop. The loop may call `ContextBuilder.build()`
        multiple times, but the builder is now side-effect free with respect
        to session history (it uses `user_input` only for memory recall).
    """
    # Defensive: if a Session was constructed without going through
    # SessionStore.load() (e.g. a test or future code path), make sure it
    # still carries the live coach prompt instead of the Phase 0 dead default.
    if not session.system_prompt or session.system_prompt == _OLD_SYSTEM_PROMPT:
        session.system_prompt = SYSTEM_PROMPT

    # Sanitize user_input at the entry point: Windows console input (e.g.
    # DELETE key) can produce lone surrogates (U+D800..U+DFFF) which UTF-8
    # strictly forbids. .encode("utf-8") inside the embedder (and any
    # other sink downstream) would otherwise crash. Stripping here makes
    # every downstream path safe.
    from .text.sanitize import strip_surrogates
    user_input = strip_surrogates(user_input)

    # 1) Push to short-term memory.
    memory.push_turn(session.id, {"role": "user", "content": user_input})
    # 2) Trace the user input.
    trace.event("user", text=user_input)
    # 3) Record the user message on the session ONCE (not per LLM iteration).
    session.add_user(user_input)

    builder = ContextBuilder(memory=memory, settings=settings, summarizer=summarizer or llm)
    schemas = registry.openai_schemas()

    from .parser import parse_response, _strip_thinking  # local import to avoid a top-level cycle

    iters = 0
    while iters < settings.max_tool_iters:
        # Builder is now strictly side-effect free w.r.t. session: it reads
        # history and returns a BuiltContext. If compression happened,
        # BuiltContext.new_summary is non-None and we apply it ONCE here.
        built = builder.build(session, user_input)
        if built.new_summary is not None and built.new_summary != session.summary:
            session.summary = built.new_summary
        messages = built.messages
        raw = llm.chat(messages, tools=schemas)
        parsed = parse_response(raw)
        if parsed.thought:
            trace.event("thought", text=parsed.thought)

        if parsed.tool_calls:
            # Record ONE assistant tool-call turn with the FIRST call_id
            # as the group anchor (matches brief: first call id reused).
            anchor = parsed.tool_calls[0]
            session.add_tool_call(
                call_id=anchor.id,
                name=anchor.name,
                args=anchor.args,
            )
            for call in parsed.tool_calls:
                trace.event("tool_call", name=call.name, args=call.args)
                result: ToolResult = registry.execute(call.name, call.args, session)
                trace.event(
                    "tool_result",
                    name=call.name,
                    ok=result.ok,
                    content=result.content,
                )
                session.add_tool_result(
                    call_id=call.id,
                    name=call.name,
                    content=result.content,
                )
                memory.push_turn(
                    session.id,
                    {"role": "tool", "name": call.name, "content": result.content},
                )
                # W3b: summary drift fix. If update_plan succeeded, the
                # old summary (if any) describes a stale plan. Clear it
                # so the next ContextBuilder.build() regenerates from
                # current history — which now contains the update_plan
                # call itself.
                if call.name == "update_plan" and result.ok:
                    if session.summary:
                        session.summary = ""
                        trace.event("summary_invalidated",
                                    reason="plan updated")
            iters += 1
            continue

        answer = parsed.final_answer or ""
        session.add_assistant(answer)
        memory.push_turn(session.id, {"role": "assistant", "content": answer})
        trace.event("assistant", text=answer)
        # Persist extracted long-term facts (best effort), but only when the
        # turn is substantive enough to be worth a real LLM extraction call.
        # `answer` already has thinking blocks stripped by the parser; strip
        # again defensively in case an upstream path left one in.
        answer_for_len = _strip_thinking(answer) or ""
        if _should_extract(user_input, answer_for_len):
            try:
                memory.remember_sid(
                    session.id,
                    [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": answer},
                    ],
                    llm=llm,
                )
            except Exception:
                pass

        # Phase 2: dynamic plan-adjustment trigger. After a successful turn,
        # check if the user implied "check the trend again" or if the plan
        # is stale. If so, re-query tech_trend and inject results into the
        # short-term memory so the next turn's LLM has fresh data.
        try:
            should, reason = _should_adjust_plan(session, user_input)
            if should:
                trace.event("plan_adjust_trigger", reason=reason)
                _inject_trend_for_adjustment(registry, session, memory, user_input, trace)
        except Exception:
            # Best-effort; never let adjust logic block the response.
            pass

        return answer

    # Force finalise when we exhaust tool iterations.
    final = "(stopped: maximum tool iterations reached)"
    session.add_assistant(final)
    trace.event("assistant", text=final)
    return final