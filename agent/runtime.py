# agent/runtime.py
from __future__ import annotations

from .config import Settings
from .llm import LLMClient
from .session import Session
from .tools.base import ToolResult
from .tools.calculator import CalculatorTool
from .tools.registry import ToolRegistry
from .tools.search import SearchTool
from .tools.todo import TodoTool
from .tools.weather import WeatherTool
from .memory.embeddings import MockEmbedder
from .memory.manager import MemoryManager
from .memory.short_term import InMemoryShortTermStore, RedisShortTermStore
from .memory.vector_store import (
    ChromaVectorStore,
    LocalVectorStore,
    QdrantVectorStore,
)
from .context import ContextBuilder
from .trace import TraceLogger


def build_default_registry() -> ToolRegistry:
    """Register the four built-in tools: calculator, search, todo, weather."""
    reg = ToolRegistry()
    reg.register_all([
        CalculatorTool(),
        SearchTool(),
        TodoTool(),
        WeatherTool(),
    ])
    return reg


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

    Responsibility split (per Task 12 brief + plan review):
      - This function does NOT call `session.add_user(user_input)` itself.
      - The `ContextBuilder.build()` call records the user message on the
        session AND includes it as the last message in the LLM context.
        The builder is the single source of truth for "what's in the LLM
        context this turn".
    """
    # 1) Push to short-term memory.
    memory.push_turn(session.id, {"role": "user", "content": user_input})
    # 2) Trace the user input.
    trace.event("user", text=user_input)

    builder = ContextBuilder(memory=memory, settings=settings, summarizer=summarizer or llm)
    schemas = registry.openai_schemas()

    from .parser import parse_response  # local import to avoid a top-level cycle

    iters = 0
    while iters < settings.max_tool_iters:
        # Builder appends the user input to session and includes it last.
        messages, _ = builder.build(session, user_input)
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
            iters += 1
            continue

        answer = parsed.final_answer or ""
        session.add_assistant(answer)
        memory.push_turn(session.id, {"role": "assistant", "content": answer})
        trace.event("assistant", text=answer)
        # Persist extracted long-term facts (best effort).
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
        return answer

    # Force finalise when we exhaust tool iterations.
    final = "(stopped: maximum tool iterations reached)"
    session.add_assistant(final)
    trace.event("assistant", text=final)
    return final