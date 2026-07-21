# agent/context.py
from __future__ import annotations
from .session import Session
from .config import Settings
from .prompts import SYSTEM_PROMPT
from .memory.manager import MemoryManager


class ContextBuilder:
    def __init__(self, memory: MemoryManager, settings: Settings, summarizer=None) -> None:
        self.memory = memory
        self.settings = settings
        self.summarizer = summarizer  # optional LLMClient for summaries

    def build(self, session: Session, user_input: str) -> tuple[list[dict], list[dict]]:
        """Build the LLM context for this turn.

        `user_input` is REQUIRED and used as the memory-recall query. The
        builder does NOT mutate the session: the caller (runtime.run_turn) is
        responsible for calling `session.add_user(user_input)` exactly ONCE
        before the tool loop. This method may be called multiple times per
        turn (once per LLM iteration), so it must be side-effect free with
        respect to session history.
        """
        msgs: list[dict] = [{"role": "system", "content": session.system_prompt}]

        # Recall relevant memory and inject as a system block (recall timing & placement)
        hits = self.memory.recall(session.id, user_input, top_k=5)
        if hits:
            lines = ["Relevant memory recalled for this turn:"]
            for text, score, _meta in hits:
                lines.append(f"- {text}")
            msgs.append({"role": "system", "content": "\n".join(lines)})

        # Compress older history if over threshold.
        # The current user message is at the tail, so compress everything
        # except the most recent `recent_keep` entries (which keeps the live
        # user turn intact).
        history = list(session.messages)
        if len(history) > self.settings.context_max_messages and self.summarizer is not None:
            keep = self.settings.recent_keep
            older, recent = history[:-keep], history[-keep:]
            session.summary = self._summarize(older, session.summary)
            history = recent

        if session.summary:
            msgs.append({"role": "system", "content": f"Conversation so far (summary):\n{session.summary}"})

        msgs.extend(history)
        # The last message is the user input (added to session.messages by the
        # caller before the tool loop).
        return msgs, []  # tool schemas injected by runtime

    def _summarize(self, older: list[dict], prev_summary: str) -> str:
        # Build a transcript
        lines = []
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
