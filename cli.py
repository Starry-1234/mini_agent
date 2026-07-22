from __future__ import annotations
import argparse
import atexit
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

from starry_code.config import Settings
from starry_code.session import Session, SessionStore
from starry_code.llm import LLMClient, MockLLMClient, make_default_mock_llm
from starry_code.runtime import run_turn, build_default_registry, build_memory
from starry_code.trace import TraceLogger
from starry_code.naming import AutoNamer


def _gen_auto_id() -> str:
    """Generate a temporary auto session id like `auto-20260722-143012-a1b2`."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"auto-{ts}-{secrets.token_hex(2)}"


def _set_terminal_title(title: str) -> None:
    """Set the terminal window title via the ANSI OSC 0 escape sequence.

    Supported by Windows Terminal, iTerm2, gnome-terminal, kitty, and
    modern PowerShell/cmd on Windows 10+. No-op on streams that can't be
    written to (e.g. captured/redirected stdio).
    """
    try:
        sys.stdout.write(f"\033]0;{title}\007")
        sys.stdout.flush()
    except Exception:
        pass


def _cleanup_empty_auto_session(trace: TraceLogger, session: "Session", sessions_dir: Path) -> None:
    """Delete the trace file if the auto-id session was never written to.

    Triggered by REPL exit when all of these hold:
      - session.id starts with "auto-" (user did not pass --session)
      - the per-session .json does not exist (ask() never saved any turns)
      - the trace file is 0 bytes (no events were emitted)

    Manually-named sessions and --once mode are skipped (they always leave
    real data). Failures are swallowed; cleanup is best-effort and must
    never raise out of atexit.
    """
    try:
        if not session.id.startswith("auto-"):
            return
        json_path = sessions_dir / f"{session.id}.json"
        if json_path.exists():
            return  # the user actually typed something — keep evidence
        if trace.path is None or not trace.path.exists():
            return
        if trace.path.stat().st_size > 0:
            return  # trace has events (e.g. errors); preserve for debugging
        try:
            trace.close()
        except Exception:
            pass
        try:
            trace.path.unlink()
            print(f"[cleaned up empty auto trace: {trace.path.name}]")
        except FileNotFoundError:
            pass
    except Exception:
        pass  # never let cleanup raise out of atexit


def main() -> int:
    p = argparse.ArgumentParser(description="Starry Code CLI")
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

    # Initial window title: brand only. Matches the printed banner below; the
    # session id is shown nowhere on the REPL until the user has given us
    # something to name after (Claude Code pattern: brand -> auto name).
    _set_terminal_title("✦ Starry Code")

    # In REPL mode, register a cleanup hook that deletes the 0-byte trace file
    # if the user exits before typing anything. Skips silently in --once mode
    # (where ask() is always called) and for manually-named sessions.
    if not args.once:
        atexit.register(_cleanup_empty_auto_session, trace, session, settings.sessions_dir)

    def ask(text: str) -> str:
        answer = run_turn(session, text, settings=settings, llm=llm,
                          registry=registry, memory=memory, trace=trace)
        store.save(session)
        if autonamer is not None and autonamer.pending():
            autonamer.try_name(llm, text, session, trace, settings.sessions_dir)
            # The session id may have changed. Once named, the title
            # drops the brand prefix and shows just the star + the new name
            # (matches Claude Code: "✦ <session-name>").
            _set_terminal_title(f"✦ {session.id}")
        return answer

    if args.once:
        print(ask(args.once))
        return 0

    print("✦ Starry Code")
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