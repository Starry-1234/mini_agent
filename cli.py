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
# `errors="replace"` is critical: without it, UTF-8 strictly forbids surrogate
# codepoints (U+D800-U+DFFF), and reasoning models like MiniMax-M3 sometimes
# emit them — every `print(answer)` would then crash. With "replace" the
# offending bytes become `�` (replacement char) so the user still sees
# output instead of an exception.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass  # Python < 3.7 or already closed


# Re-export the shared sanitizer as `_strip_surrogates` for back-compat
# with the many call sites already in this file. New code should use
# `starry_code.text.sanitize.strip_surrogates` directly.
from starry_code.text.sanitize import strip_surrogates as _strip_surrogates

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


# ---- REPL startup rendering ----
#
# These are split out so they can be unit-tested with pytest's capsys
# without spinning up a real REPL. They handle:
#   1. Display isolation — clear screen + scrollback so the previous
#      session's output doesn't leak into this one (Claude Code style).
#   2. History replay — when resuming a session via --session / -c /
#      -resume, the user can see what they were talking about last time.

# ANSI: \033[H = home cursor, \033[2J = clear screen, \033[3J = clear scrollback.
# Combined so the visible area AND the scrollback buffer are wiped in one go.
_CLEAR_SCREEN = "\033[H\033[2J\033[3J"
# History block dimensions — kept short so it fits on a typical terminal.
_HISTORY_DIVIDER = "─" * 60
_ASSISTANT_PREVIEW_CHARS = 600
_USER_PREVIEW_CHARS = 400


def _truncate(text: str, max_chars: int) -> str:
    """Truncate `text` to `max_chars`, appending an ellipsis if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def print_session_history(session: "Session", stream=None) -> None:
    """Print the session's prior conversation as a compact history block.

    Skips tool-call / tool-result messages (noisy and not useful for the
    human reading back). Truncates very long assistant messages.

    Pure write to `stream`; no side effects on session.
    """
    out = stream or sys.stdout
    if not session.messages:
        return
    out.write(_HISTORY_DIVIDER + "\n")
    out.write("[history]\n")
    out.write(_HISTORY_DIVIDER + "\n")
    for m in session.messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        if not content:
            continue
        if role == "user":
            preview = _truncate(content, _USER_PREVIEW_CHARS)
            out.write(f"\033[36m> {preview}\033[0m\n")
        elif role == "assistant":
            preview = _truncate(content, _ASSISTANT_PREVIEW_CHARS)
            out.write(f"\033[32m{preview}\033[0m\n")
        # tool / system / unknown: skip — too noisy for a glance-back block
    out.write(_HISTORY_DIVIDER + "\n")


def render_repl_startup(session: "Session", stream=None) -> None:
    """Render the REPL startup frame: clear screen, header, optional history.

    Called once when the REPL enters its input loop, both for fresh
    sessions and for resumed sessions via --session / -c / -resume.

    Side effects:
      - Writes ANSI escape sequences to `stream` (or sys.stdout).
      - Does NOT touch session state — pure presentation.
    """
    out = stream or sys.stdout
    # 1) Display isolation: wipe both the visible screen and the scrollback
    #    so content from a previous REPL session in the same terminal
    #    doesn't bleed into this one. Standard ANSI CSI sequences; ignored
    #    on streams that can't interpret them.
    out.write(_CLEAR_SCREEN)
    # 2) Header: which session am I in? Auto-id sessions show the brand
    #    "Starry Code" instead of the ugly auto-20260808-...-xxxx slug —
    #    matches what _set_terminal_title() does for the window title bar.
    display_id = "Starry Code" if session.id.startswith("auto-") else session.id
    out.write(f"\033[1m✦ {display_id}\033[0m\n")
    # 3) If resuming a session with messages, replay them so the user has
    #    immediate context. Skip for fresh sessions (no history to show).
    if session.messages:
        out.write("\n")
        print_session_history(session, stream=out)
        out.write("\n")


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

    # Initial window title:
    #   - For auto-id sessions (no --session), show the brand only; the
    #     session gets a Chinese name after the first turn and the title
    #     drops the brand prefix in ask() below.
    #   - For pre-named sessions (--session foo, or -c / -resume picking one),
    #     show the session name immediately so the resumed session's title
    #     shows up at startup instead of the generic brand.
    if session.id.startswith("auto-"):
        _set_terminal_title("✦ Starry Code")
    else:
        _set_terminal_title(f"✦ {session.id}")

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

    # REPL startup frame: clear screen + header + (optional) history replay.
    # See render_repl_startup() docstring for the rationale.
    render_repl_startup(session)
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
            # Strip surrogates before printing — reasoning models (MiniMax-M3,
            # DeepSeek-R1) sometimes emit U+D800..U+DFFF which UTF-8 rejects
            # with UnicodeEncodeError. See _strip_surrogates() for context.
            print(_strip_surrogates(ans))
        except Exception as e:  # noqa: BLE001
            trace.event("error", message=str(e))
            print(f"[error] {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())