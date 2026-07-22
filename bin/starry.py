#!/usr/bin/env python3
"""Starry Code launcher."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SESSIONS_DIR = PROJECT_ROOT / "sessions"
RECENT_FILE_NAME = ".starry-recent"
MAX_RECENTS = 20


# --- Recents persistence ---

def _recent_path() -> Path:
    return Path.cwd() / RECENT_FILE_NAME

def load_recents() -> list[str]:
    p = _recent_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [s for s in data if isinstance(s, str)]
    except Exception:
        return []

def save_recents(recents: list[str]) -> None:
    p = _recent_path()
    try:
        p.write_text(json.dumps(recents[:MAX_RECENTS], ensure_ascii=False, indent=2),
                     encoding="utf-8")
    except OSError:
        pass  # best-effort; cwd might be read-only

def add_recent(session_id: str) -> None:
    recents = load_recents()
    recents = [s for s in recents if s != session_id]
    recents.insert(0, session_id)
    save_recents(recents)


# --- Session listing ---

def list_sessions() -> list[tuple[str, float, int]]:
    """Return [(session_id, mtime, size_bytes), ...] newest first."""
    if not SESSIONS_DIR.exists():
        return []
    out: list[tuple[str, float, int]] = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            st = f.stat()
            out.append((f.stem, st.st_mtime, st.st_size))
        except OSError:
            continue
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def session_has_turns(sid: str) -> bool:
    """True if the session has at least one user message."""
    p = SESSIONS_DIR / f"{sid}.json"
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return bool(data.get("messages"))
    except Exception:
        return False

def fmt_time_ago(mtime: float) -> str:
    delta = time.time() - mtime
    if delta < 60: return "just now"
    if delta < 3600: return f"{int(delta/60)}m ago"
    if delta < 86400: return f"{int(delta/3600)}h ago"
    if delta < 86400*7: return f"{int(delta/86400)}d ago"
    if delta < 86400*30: return f"{int(delta/86400/7)}w ago"
    return "long ago"


# --- Cross-platform keyboard ---

class _KeyReader:
    """Read one key at a time. Cross-platform. Non-blocking (returns None if no key)."""
    def __init__(self) -> None:
        self._msvcrt = None
        self._tty = None
        self._termios = None
        self._old = None
        try:
            import msvcrt  # type: ignore
            self._msvcrt = msvcrt
        except ImportError:
            import tty, termios  # type: ignore
            self._tty = tty
            self._termios = termios
            self._fd = sys.stdin.fileno()

    def __enter__(self) -> "_KeyReader":
        if self._tty:
            self._old = self._termios.tcgetattr(self._fd)
            self._tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        if self._tty and self._old is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSAFLUSH, self._old)

    def read(self) -> str | None:
        if self._msvcrt is not None:
            if not self._msvcrt.kbhit():
                return None
            ch = self._msvcrt.getch()
            # Windows arrow keys come as b'\xe0' + b'H' (up) / b'P' (down)
            if ch == b'\xe0':
                ch2 = self._msvcrt.getch()
                return {'H': 'up', 'P': 'down'}.get(ch2.decode('latin-1', 'ignore'))
            if ch in (b'\r', b'\n'):
                return 'enter'
            if ch in (b'\x1b', b'\x03'):  # ESC or Ctrl-C
                return 'esc'
            if ch in (b'q', b'Q'):
                return 'quit'
            return ch.decode('utf-8', 'ignore')
        # POSIX
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not r:
            return None
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            # could be ESC alone or start of an escape sequence
            r2, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r2:
                return 'esc'
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r3:
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A': return 'up'
                    if ch3 == 'B': return 'down'
            return 'esc'
        if ch in ('\r', '\n'):
            return 'enter'
        if ch in ('q', 'Q'):
            return 'quit'
        return ch

    def read_blocking(self) -> str:
        """Block until a key is pressed. Cross-platform.

        Polls the non-blocking ``read()`` with a short sleep between attempts
        so we don't burn 100% CPU. Used by the picker so we only re-render
        on actual keypresses (no 50ms tick redraws → no flicker on
        Windows Terminal / PowerShell).
        """
        import time as _t
        while True:
            k = self.read()
            if k is not None:
                return k
            _t.sleep(0.02)  # don't spin at 100% CPU


# --- TUI picker ---

# Constants kept module-level so render_picker can be hoisted for testing.
BOX_WIDTH = 56
PICKER_VISIBLE = 15


def render_picker(sessions: list[tuple[str, float, int]], idx: int,
                  box_width: int = BOX_WIDTH, visible: int = PICKER_VISIBLE) -> str:
    """Build the full picker frame as a single string.

    Pure function: same input → same output. Hoisted to module level so it
    can be tested without spinning up a real terminal.
    """
    out: list[str] = []
    # Position cursor at top, then clear from cursor down (so re-renders in
    # the SAME window are flicker-free on Windows Terminal).
    out.append("\033[H\033[J")
    out.append("┌─ Starry Code — pick a session " + "─" * (box_width - 32) + "┐")
    out.append("│   session" + " " * (box_width - 11) + " last used  │")
    out.append("│" + "─" * (box_width + 2) + "│")
    shown = sessions[:visible]
    for k, (sid, mtime, _size) in enumerate(shown):
        ago = fmt_time_ago(mtime)
        marker = ">" if k == idx else " "
        name = sid
        max_name = box_width - len(ago) - 7
        if len(name) > max_name:
            name = name[:max_name - 1] + "…"
        line = f"│ {marker} {name}  {ago}"
        out.append(line.ljust(box_width + 1) + "│")
    if len(sessions) > visible:
        more = f"  (… {len(sessions) - visible} more, use ↑/↓ to scroll)"
        out.append(more.ljust(box_width + 2) + "│")
    out.append("│" + "─" * (box_width + 2) + "│")
    out.append("│ ↑/↓ to move · enter to open · esc / q to quit " + " " * 6 + "│")
    out.append("└" + "─" * (box_width + 2) + "┘")
    return "\n".join(out) + "\n"


def pick_session(sessions: list[tuple[str, float, int]]) -> str | None:
    """Interactive picker. Returns selected session id or None on cancel."""
    if not sessions:
        return None
    if not sys.stdout.isatty():
        # No TTY — fall back to first session (don't loop forever in CI)
        return sessions[0][0]

    idx = 0
    shown_count = min(len(sessions), PICKER_VISIBLE)

    with _KeyReader() as kr:
        # Hide the terminal cursor while the picker is on screen.
        sys.stdout.write("\033[?25l")
        try:
            sys.stdout.write(render_picker(sessions, idx))
            sys.stdout.flush()
            while True:
                key = kr.read_blocking()
                if key == 'up':
                    idx = max(0, idx - 1)
                elif key == 'down':
                    idx = min(shown_count - 1, idx + 1)
                elif key == 'enter':
                    return sessions[idx][0]
                elif key in ('esc', 'quit'):
                    return None
                else:
                    # Unknown key — don't waste a redraw.
                    continue
                # Re-render ONLY after a real keypress that changed state.
                sys.stdout.write(render_picker(sessions, idx))
                sys.stdout.flush()
        finally:
            # Always restore the cursor, even on exception.
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()


# --- Record-after-exit ---

def record_most_recent_session() -> None:
    """Scan sessions/ for the newest .json with at least 1 turn; record in cwd."""
    if not SESSIONS_DIR.exists():
        return
    best: tuple[float, str] | None = None
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            st = f.stat()
            data = json.loads(f.read_text(encoding="utf-8"))
            if not data.get("messages"):
                continue
            if best is None or st.st_mtime > best[0]:
                best = (st.st_mtime, f.stem)
        except (OSError, json.JSONDecodeError):
            continue
    if best is not None:
        add_recent(best[1])


# --- Main ---

def main() -> int:
    parser = argparse.ArgumentParser(prog="starry", add_help=False)
    parser.add_argument("-c", action="store_true", help="Continue most recent session in cwd")
    parser.add_argument("-resume", action="store_true", help="Open interactive session picker")
    parser.add_argument("-s", "--session", help="Pin session id (skip auto-name)")
    parser.add_argument("--once", help="One-shot message")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    args, passthrough = parser.parse_known_args()

    if args.help:
        print(__doc__)
        return 0

    # Resolve session id from flags. Capture caller_cwd BEFORE doing anything
    # that mutates cwd — we need to write .starry-recent back where the user
    # invoked starry from, NOT in the project root.
    caller_cwd = Path.cwd()
    record_recent_after = True

    session_id = args.session
    if args.c and not session_id:
        # load_recents() uses Path.cwd(), so chdir to caller_cwd first.
        if Path.cwd() != caller_cwd:
            os.chdir(caller_cwd)
        recents = load_recents()
        for sid in recents:
            if (SESSIONS_DIR / f"{sid}.json").exists():
                session_id = sid
                break
        if not session_id:
            print("[starry] no recent session for this directory; starting new")

    if args.resume:
        sessions = list_sessions()
        if not sessions:
            print("[starry] no sessions to pick from")
            return 1
        picked = pick_session(sessions)
        if not picked:
            print("[starry] cancelled")
            return 1
        session_id = picked

    # Build docker compose command
    cmd = ["docker", "compose", "run", "--rm", "agent"]
    if session_id:
        cmd += ["--session", session_id]
    if args.once:
        cmd += ["--once", args.once]
    if args.mock:
        cmd.append("--mock")
    cmd += passthrough

    # Always run from the project root so docker compose finds the file
    os.chdir(PROJECT_ROOT)

    # Run docker compose as a subprocess (rather than os.execvp) so that after
    # the container exits we can scan sessions/ and update the caller's
    # .starry-recent for `starry -c` on the next invocation.
    import subprocess
    try:
        rc = subprocess.call(cmd)
    except FileNotFoundError:
        print("[starry] docker not found in PATH", file=sys.stderr)
        return 127

    if record_recent_after:
        # Restore caller's cwd before recording recents.
        os.chdir(caller_cwd)
        record_most_recent_session()
    return rc


if __name__ == "__main__":
    sys.exit(main())
