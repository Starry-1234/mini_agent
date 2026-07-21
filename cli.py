from __future__ import annotations
import argparse
import os
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
from agent.session import SessionStore
from agent.llm import LLMClient, MockLLMClient, make_default_mock_llm
from agent.runtime import run_turn, build_default_registry, build_memory
from agent.trace import TraceLogger
from agent.naming import AutoNamer


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
            llm = make_default_mock_llm(args.once)
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

    # If --session is absent, generate a temporary auto id and arm the auto-namer.
    auto_named = args.session is None
    session_id = args.session if args.session is not None else _gen_auto_id()

    store = SessionStore(settings.sessions_dir)
    session = store.load(session_id)
    registry = build_default_registry()
    memory = build_memory(settings=settings, llm=llm)
    trace = TraceLogger(settings.sessions_dir, session_id)

    autonamer = AutoNamer() if auto_named else None

    def ask(text: str) -> str:
        answer = run_turn(session, text, settings=settings, llm=llm,
                          registry=registry, memory=memory, trace=trace)
        store.save(session)
        if autonamer is not None and autonamer.pending():
            autonamer.try_name(llm, text, session, trace, settings.sessions_dir)
        return answer

    if args.once:
        print(ask(args.once))
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
        except Exception as e:  # noqa: BLE001
            trace.event("error", message=str(e))
            print(f"[error] {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())