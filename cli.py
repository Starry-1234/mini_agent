from __future__ import annotations
import argparse
import os
import re
import secrets
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdio so reasoning models / Chinese / emoji don't crash on
# legacy Windows code pages (GBK / cp936) when they print to the terminal.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass  # Python < 3.7 or already closed

from agent.config import Settings
from agent.session import Session, SessionStore
from agent.llm import LLMClient, MockLLMClient
from agent.runtime import run_turn, build_default_registry, build_memory
from agent.trace import TraceLogger
from agent.naming import generate_chinese_name, rename_session


# Heuristic: extract the arithmetic expression the user asked about.
# Match patterns like "2+2", "what is 3 * 4", "compute 10 / 2", etc.
_MATH_QUERY_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?\s*[+\-*/%][\s0-9.+\-*/%()]+)")


class DefaultMockLLMClient(MockLLMClient):
    """A MockLLMClient pre-scripted for common CLI smoke queries.

    Used only by `cli.py --mock` for offline testing — not for real LLM work.
    The demo script keeps its own scripted flow and is unaffected.

    - Math/arithmetic queries (contains a number and an operator + - * /):
        step 1 -> tool_call to `calculator` with the extracted expression
        step 2 -> final answer "The answer is <result>." (placeholder; the real
                  answer is whatever the calculator tool returns).
    - Otherwise: a single direct final answer "(mock) I received your message."
    """

    _EXTRACTOR_RESPONSE = {"choices": [{"message": {"role": "assistant", "content": "[]"}}]}

    def __init__(self, user_message: str, embed_dim: int = 16) -> None:
        responses: list[dict] = []
        expr = self._extract_expression(user_message)
        if expr is not None:
            responses.append({
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "computing",
                    "tool_calls": [{"id": "mock_calc", "type": "function",
                                    "function": {"name": "calculator",
                                                 "arguments": '{"expression": "' + expr + '"}'}}],
                }}],
            })
            responses.append({
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "The answer is <result>.",
                }}],
            })
            # Extractor call from memory.remember_sid — return no facts.
            responses.append(self._EXTRACTOR_RESPONSE)
        else:
            responses.append({
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "(mock) I received your message.",
                }}],
            })
            responses.append(self._EXTRACTOR_RESPONSE)
        super().__init__(chat_responses=responses, embed_dim=embed_dim)

    def chat(self, messages, tools=None) -> dict:
        # Route the auto-naming prompt to a fixed Chinese slug so `--mock`
        # exercises the rename flow without exhausting the scripted queue.
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, str) and "会话命名助手" in content:
                return {"choices": [{"message": {"role": "assistant", "content": "测试会话"}}]}
        return super().chat(messages, tools=tools)

    @staticmethod
    def _extract_expression(user_message: str) -> str | None:
        m = _MATH_QUERY_RE.search(user_message)
        if not m:
            return None
        expr = m.group(1).strip()
        # Strip trailing junk that is unlikely to be part of the expression.
        return expr.rstrip(".,;:?")


def _gen_auto_id() -> str:
    """Generate a temporary auto session id like `auto-20260722-143012-a1b2`."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"auto-{ts}-{secrets.token_hex(2)}"


def main() -> int:
    p = argparse.ArgumentParser(description="Mini Agent CLI")
    p.add_argument("--session", "-s", nargs="?", default=None,
                   help="Session id (window name). If omitted, an auto id is "
                        "generated and the session is auto-named in Chinese "
                        "after the first turn.")
    p.add_argument("--once", help="Run a single message and exit (non-interactive)")
    p.add_argument("--mock", action="store_true", help="Use MockLLMClient (no real API calls)")
    args = p.parse_args()

    settings = Settings.from_env(sessions_dir=Path(os.environ.get("SESSIONS_DIR", "sessions")))

    if args.mock:
        if args.once:
            llm = DefaultMockLLMClient(args.once)
        else:
            llm = MockLLMClient()
    else:
        if not settings.llm_api_key:
            print("error: LLM_API_KEY is required (or pass --mock)", file=sys.stderr)
            return 2
        llm = LLMClient(api_key=settings.llm_api_key, base_url=settings.llm_base_url,
                        model=settings.llm_model,
                        embed_api_key=settings.embed_api_key, embed_base_url=settings.embed_base_url,
                        embed_model=settings.embed_model)

    # If --session is absent, generate a temporary auto id and mark for renaming.
    auto_named = args.session is None
    session_id = args.session if args.session is not None else _gen_auto_id()

    store = SessionStore(settings.sessions_dir)
    session = store.load(session_id)
    registry = build_default_registry()
    memory = build_memory(settings=settings, llm=llm)
    trace = TraceLogger(settings.sessions_dir, session_id)

    # Mutable flag: True while the session still has an auto-generated id that
    # should be replaced by a Chinese name after the first successful turn.
    state = {"pending_auto_name": auto_named}

    def ask(text: str) -> str:
        answer = run_turn(session, text, settings=settings, llm=llm,
                          registry=registry, memory=memory, trace=trace)
        store.save(session)
        return answer

    def maybe_autoname(first_user_msg: str) -> None:
        """After the first turn, rename an auto session to a Chinese name.

        Never blocks the user: on LLM failure or collision the session keeps
        its auto id and a short note is printed once.
        """
        if not state["pending_auto_name"]:
            return
        state["pending_auto_name"] = False  # only ever fires once
        old_id = session.id
        name = generate_chinese_name(llm, first_user_msg)
        if name is None:
            print(f"[could not auto-name, use --session {old_id} to continue]")
            return
        ok = rename_session(session, name, trace, settings.sessions_dir)
        if not ok:
            print(f"[could not auto-name, use --session {old_id} to continue]")
            return
        print(f"[session auto-named to: {session.id} — use --session {session.id} to continue next time]")

    if args.once:
        print(ask(args.once))
        maybe_autoname(args.once)
        return 0

    print(f"mini_agent — session={session.id} (type 'exit' to quit)")
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        s = text.strip()
        if not s:
            continue
        if s in ("exit", "quit"):
            return 0
        try:
            ans = ask(s)
            print(ans)
            maybe_autoname(s)
        except Exception as e:  # noqa: BLE001
            trace.event("error", message=str(e))
            print(f"[error] {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())