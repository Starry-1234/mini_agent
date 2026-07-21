from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from agent.config import Settings
from agent.session import Session, SessionStore
from agent.llm import LLMClient, MockLLMClient
from agent.runtime import run_turn, build_default_registry, build_memory
from agent.trace import TraceLogger


def main() -> int:
    p = argparse.ArgumentParser(description="Mini Agent CLI")
    p.add_argument("--session", "-s", required=True, help="Session id (window name)")
    p.add_argument("--once", help="Run a single message and exit (non-interactive)")
    p.add_argument("--mock", action="store_true", help="Use MockLLMClient (no real API calls)")
    args = p.parse_args()

    settings = Settings.from_env(sessions_dir=Path(os.environ.get("SESSIONS_DIR", "sessions")))

    if args.mock:
        llm = MockLLMClient()
    else:
        if not settings.llm_api_key:
            print("error: LLM_API_KEY is required (or pass --mock)", file=sys.stderr)
            return 2
        llm = LLMClient(api_key=settings.llm_api_key, base_url=settings.llm_base_url,
                        model=settings.llm_model,
                        embed_api_key=settings.embed_api_key, embed_base_url=settings.embed_base_url,
                        embed_model=settings.embed_model)

    store = SessionStore(settings.sessions_dir)
    session = store.load(args.session)
    registry = build_default_registry()
    memory = build_memory(settings=settings, llm=llm)
    trace = TraceLogger(settings.sessions_dir, args.session)

    def ask(text: str) -> str:
        answer = run_turn(session, text, settings=settings, llm=llm,
                          registry=registry, memory=memory, trace=trace)
        store.save(session)
        return answer

    if args.once:
        print(ask(args.once))
        return 0

    print(f"mini_agent — session={args.session} (type 'exit' to quit)")
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
        except Exception as e:  # noqa: BLE001
            trace.event("error", message=str(e))
            print(f"[error] {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())