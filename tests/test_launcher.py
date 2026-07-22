"""Unit tests for the bin/starry.py Python launcher (data layer only).

We do NOT test the TUI picker itself (it requires a real terminal). These
tests cover the helpers around persistence, session discovery, and the
after-exit record step.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

# Import the launcher as a library by adding bin/ to sys.path.
import sys

BIN_DIR = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN_DIR))

import starry  # noqa: E402


# ---------------------------------------------------------------------------
# Recents persistence
# ---------------------------------------------------------------------------

def test_recents_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """save_recents then load_recents returns the same list."""
    monkeypatch.chdir(tmp_path)
    starry.save_recents(["alpha", "beta", "gamma"])
    assert starry.load_recents() == ["alpha", "beta", "gamma"]


def test_recents_load_missing_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    assert starry.load_recents() == []


def test_recents_load_garbage_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Bad JSON should never crash load_recents; only string entries survive."""
    monkeypatch.chdir(tmp_path)
    starry._recent_path().write_text("not json at all", encoding="utf-8")
    assert starry.load_recents() == []

    starry._recent_path().write_text('["ok", 123, "also-ok"]', encoding="utf-8")
    # The spec filter is `[s for s in data if isinstance(s, str)]` — ints are
    # dropped, strings survive.
    assert starry.load_recents() == ["ok", "also-ok"]


def test_add_recent_deduplicates_and_bumps_to_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Adding an existing id moves it to the head rather than duplicating."""
    monkeypatch.chdir(tmp_path)
    starry.save_recents(["alpha", "beta", "gamma"])
    starry.add_recent("beta")
    assert starry.load_recents() == ["beta", "alpha", "gamma"]

    starry.add_recent("delta")
    assert starry.load_recents() == ["delta", "beta", "alpha", "gamma"]


def test_add_recent_caps_at_max(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    # Seed beyond the cap, then add one more — cap should hold.
    big = [f"s{i}" for i in range(starry.MAX_RECENTS + 5)]
    starry.save_recents(big)
    starry.add_recent("FRESH")
    result = starry.load_recents()
    assert len(result) == starry.MAX_RECENTS
    assert result[0] == "FRESH"


def test_save_recents_swallows_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A read-only cwd should not raise out of save_recents."""
    monkeypatch.chdir(tmp_path)

    def boom(_self, _data, **_kw):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(starry.Path, "write_text", boom)
    # Should not raise:
    starry.save_recents(["x", "y"])


# ---------------------------------------------------------------------------
# Session listing
# ---------------------------------------------------------------------------

def test_list_sessions_returns_newest_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """We point starry.SESSIONS_DIR at a fake dir and verify ordering."""
    fake_sessions = tmp_path / "sessions"
    fake_sessions.mkdir()
    # Three files, each with a distinct mtime. We need a true mtime gap because
    # some filesystems only support 1-second resolution.
    (fake_sessions / "old.json").write_text('{"messages": ["a"]}', encoding="utf-8")
    time.sleep(1.05)
    (fake_sessions / "mid.json").write_text('{"messages": ["b"]}', encoding="utf-8")
    time.sleep(1.05)
    (fake_sessions / "new.json").write_text('{"messages": ["c"]}', encoding="utf-8")

    monkeypatch.setattr(starry, "SESSIONS_DIR", fake_sessions)
    result = starry.list_sessions()
    ids = [sid for sid, _mt, _sz in result]
    assert ids == ["new", "mid", "old"]


def test_list_sessions_empty_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(starry, "SESSIONS_DIR", tmp_path / "does_not_exist")
    assert starry.list_sessions() == []


def test_session_has_turns_true_and_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_sessions = tmp_path / "sessions"
    fake_sessions.mkdir()
    (fake_sessions / "empty.json").write_text('{"messages": []}', encoding="utf-8")
    (fake_sessions / "full.json").write_text('{"messages": [{"role": "user", "content": "hi"}]}',
                                             encoding="utf-8")
    (fake_sessions / "missing.json")  # no file

    monkeypatch.setattr(starry, "SESSIONS_DIR", fake_sessions)
    assert starry.session_has_turns("full") is True
    assert starry.session_has_turns("empty") is False
    assert starry.session_has_turns("missing") is False


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "delta_seconds, expected_substring",
    [
        (10, "just now"),
        (120, "m ago"),
        (3600, "h ago"),
        (86400, "d ago"),
        (86400 * 3, "d ago"),
        (86400 * 14, "w ago"),
        (86400 * 60, "long ago"),
    ],
)
def test_fmt_time_ago(delta_seconds: int, expected_substring: str,
                      monkeypatch: pytest.MonkeyPatch):
    fake_now = 1_000_000.0
    monkeypatch.setattr(starry.time, "time", lambda: fake_now)
    assert expected_substring in starry.fmt_time_ago(fake_now - delta_seconds)


# ---------------------------------------------------------------------------
# Record after exit
# ---------------------------------------------------------------------------

def test_record_most_recent_session_picks_newest_with_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """record_most_recent_session should pick newest mtime among non-empty sessions."""
    fake_sessions = tmp_path / "sessions"
    fake_sessions.mkdir()
    # Empty messages — should be ignored:
    (fake_sessions / "empty.json").write_text('{"messages": []}', encoding="utf-8")
    # Two non-empty, mtime distinct so newest wins.
    (fake_sessions / "older.json").write_text(
        '{"messages": [{"role": "user", "content": "x"}]}', encoding="utf-8"
    )
    time.sleep(1.05)
    (fake_sessions / "winner.json").write_text(
        '{"messages": [{"role": "user", "content": "y"}]}', encoding="utf-8"
    )

    monkeypatch.setattr(starry, "SESSIONS_DIR", fake_sessions)
    monkeypatch.chdir(tmp_path)  # recents file lands here

    starry.record_most_recent_session()
    assert starry.load_recents() == ["winner"]


def test_record_most_recent_session_skips_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """If the only .json files have empty messages, recents stays empty."""
    fake_sessions = tmp_path / "sessions"
    fake_sessions.mkdir()
    (fake_sessions / "a.json").write_text('{"messages": []}', encoding="utf-8")
    (fake_sessions / "b.json").write_text('{"messages": []}', encoding="utf-8")

    monkeypatch.setattr(starry, "SESSIONS_DIR", fake_sessions)
    monkeypatch.chdir(tmp_path)

    starry.record_most_recent_session()
    assert starry.load_recents() == []


def test_record_most_recent_session_no_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Missing sessions/ dir is a no-op, not an error."""
    monkeypatch.setattr(starry, "SESSIONS_DIR", tmp_path / "does_not_exist")
    monkeypatch.chdir(tmp_path)
    starry.record_most_recent_session()  # should not raise
    assert starry.load_recents() == []


def test_record_most_recent_session_skips_corrupt_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A .json that doesn't parse should be skipped, not crash."""
    fake_sessions = tmp_path / "sessions"
    fake_sessions.mkdir()
    (fake_sessions / "bad.json").write_text("{not parseable", encoding="utf-8")
    (fake_sessions / "good.json").write_text(
        '{"messages": [{"role": "user", "content": "ok"}]}', encoding="utf-8"
    )

    monkeypatch.setattr(starry, "SESSIONS_DIR", fake_sessions)
    monkeypatch.chdir(tmp_path)

    starry.record_most_recent_session()
    assert starry.load_recents() == ["good"]
