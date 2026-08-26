# agent/context.py
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .session import Session
from .config import Settings
from .prompts import SYSTEM_PROMPT
from .memory.manager import MemoryManager

if TYPE_CHECKING:
    pass


@dataclass
class BuiltContext:
    """Output of ContextBuilder.build().

    - messages: LLM-ready message list (system + recalled memory + summary + history)
    - tool_schemas: reserved for future use (currently always empty; runtime
      injects schemas directly from the ToolRegistry)
    - new_summary: if non-None, caller SHOULD write this to session.summary.
      None means "no compression happened, keep session.summary as-is".
    """
    messages: list[dict]
    tool_schemas: list[dict]
    new_summary: str | None = None


class ContextBuilder:
    def __init__(self, memory: MemoryManager, settings: Settings, summarizer=None) -> None:
        self.memory = memory
        self.settings = settings
        self.summarizer = summarizer  # optional LLMClient for summaries

    def build(self, session: Session, user_input: str) -> BuiltContext:
        """Build the LLM context for this turn.

        **SIDE-EFFECT FREE** with respect to `session`. This is a hard
        contract — `build()` may be called multiple times per turn (once per
        tool iteration), so any mutation here corrupts later iterations.

        `user_input` is REQUIRED and used as the memory-recall query.

        If compression is triggered, the new summary is returned in
        `BuiltContext.new_summary`; the caller (runtime.run_turn) is
        responsible for writing it to `session.summary`.
        """
        msgs: list[dict] = [{"role": "system", "content": session.system_prompt}]

        # 1) Recall relevant memory and inject as a system block.
        hits = self.memory.recall(session.id, user_input, top_k=5)
        if hits:
            lines = ["Relevant memory recalled for this turn:"]
            for text, _score, _meta in hits:
                lines.append(f"- {text}")
            msgs.append({"role": "system", "content": "\n".join(lines)})

        # 2) Hot-cache: inject plan_cache as a system block if non-empty.
        #    Cheap to read, ~50 tokens, gives the LLM continuity across turns.
        plan_cache = getattr(session, "plan_cache", None) or {}
        if any(plan_cache.get(k) for k in ("stage", "next_task", "long_term_goal")):
            msgs.append({
                "role": "system",
                "content": (
                    f"Current plan (v{plan_cache.get('version', 0)}): "
                    f"stage={plan_cache.get('stage', '?')}, "
                    f"next_task={plan_cache.get('next_task', '?')}\n"
                    f"long_term_goal={plan_cache.get('long_term_goal', '?')}"
                ),
            })

        # 3) History — may offload oversized tool results to artifacts/.
        history = self._maybe_offload_tool_results(session)

        # 4) Compress older history if over threshold. Return new summary to
        #    caller; DO NOT mutate session here.
        new_summary: str | None = None
        if len(history) > self.settings.context_max_messages and self.summarizer is not None:
            keep = self.settings.recent_keep
            older, recent = history[:-keep], history[-keep:]
            new_summary = self._summarize(older, session.summary)
            history = recent

        if session.summary or new_summary:
            summary_text = new_summary if new_summary is not None else session.summary
            msgs.append({
                "role": "system",
                "content": f"Conversation so far (summary):\n{summary_text}",
            })

        msgs.extend(history)
        return BuiltContext(messages=msgs, tool_schemas=[], new_summary=new_summary)

    # ---- offload ----

    _OFFLOAD_THRESHOLD = 500
    _ARTIFACT_CARD_TEMPLATE = (
        "[artifact saved]\n"
        "path: {path}\n"
        "tool: {tool}\n"
        "size: {size} chars\n"
        "summary: {summary}\n"
        "head: {head}\n"
        "tail: {tail}\n"
        "(use read_artifact tool to fetch full content)"
    )

    def _maybe_offload_tool_results(self, session: Session) -> list[dict]:
        """Scan history's tool messages; if any exceed threshold, write to
        sessions/{sid}/artifacts/{call_id}.json and replace with a short
        summary card.

        **Bug I fix**: previously the offload replaced the message role
        from "tool" to "system", which broke the
        assistant-tool_call → tool_result pairing that the LLM API
        requires. The LLM would reject with 400 "tool call result does
        not follow tool call". Fix: keep role="tool" and only replace
        the content with the artifact card. The tool_call_id stays intact.

        Side effect: writes artifact files to disk. Does NOT modify
        `session.messages` — the replacement is local to this call.
        """
        import json
        from datetime import datetime, timezone

        out: list[dict] = []
        artifacts_dir = self.settings.sessions_dir / session.id / "artifacts"
        for m in session.messages:
            if (m.get("role") == "tool"
                    and len(m.get("content") or "") > self._OFFLOAD_THRESHOLD):
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                call_id = m.get("tool_call_id") or "unknown"
                tool_name = m.get("name") or "unknown"
                content = m.get("content") or ""
                artifact_path = artifacts_dir / f"{call_id}.json"
                artifact_path.write_text(
                    json.dumps({
                        "tool": tool_name,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "content": content,
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                head = content[:100].replace("\n", " ")
                tail = content[-100:].replace("\n", " ") if len(content) > 200 else ""
                summary = content[:200].replace("\n", " ")
                # CRITICAL: keep role="tool" so tool_call_id pairing
                # with the assistant's tool_call is preserved. Replace
                # only the content with a short artifact card.
                out.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": self._ARTIFACT_CARD_TEMPLATE.format(
                        path=str(artifact_path),
                        tool=tool_name,
                        size=len(content),
                        summary=summary,
                        head=head,
                        tail=tail,
                    ),
                })
            else:
                out.append(m)
        return out

    # ---- summary ----

    def _summarize(self, older: list[dict], prev_summary: str) -> str:
        """Generate a new rolling summary. Failures fall back to prev_summary
        so a flaky LLM never blanks the context.

        Note: this method is the only place that *talks* to the summarizer
        LLM. It does NOT mutate any session state.
        """
        lines: list[str] = []
        if prev_summary:
            lines.append(f"Previous summary: {prev_summary}")
        for m in older:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                lines.append(f"{role}: {content}")
            elif role == "tool":
                lines.append(f"tool({m.get('name')}): {m.get('content')}")
        prompt = "Summarise the following conversation in <= 200 words, preserving key facts and decisions:\n\n" + "\n".join(lines)
        try:
            raw = self.summarizer.chat([{"role": "user", "content": prompt}], tools=None)
            return raw["choices"][0]["message"]["content"] or prev_summary
        except Exception:
            return prev_summary