"""Sidebar session picker — list + select prior sessions."""
from __future__ import annotations
import secrets
from datetime import datetime
from pathlib import Path
from typing import Callable

import streamlit as st

from starry_code.session import SessionStore


def new_session_id() -> str:
    """Generate an auto-id matching cli.py's format."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"auto-{ts}-{secrets.token_hex(2)}"


def render_session_picker(store: SessionStore, current_id: str | None) -> str | None:
    """Render a Streamlit selectbox of prior sessions; return picked id or None.

    Caller should:
      - on a picked value != current_id, call st.rerun()
      - on None (nothing picked yet), just continue
    """
    if not store.base.exists():
        return None

    # Collect sessions (newest first)
    sessions = []
    for p in sorted(store.base.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            st_mtime = p.stat().st_mtime
            ago = int((datetime.now().timestamp() - st_mtime) / 60)
            if ago < 60:
                ago_str = f"{ago}m"
            elif ago < 1440:
                ago_str = f"{ago // 60}h"
            else:
                ago_str = f"{ago // 1440}d"
            sessions.append((p.stem, ago_str))
        except OSError:
            continue

    if not sessions:
        st.caption("(no sessions)")
        return None

    labels = [f"{sid} — {ago}" for sid, ago in sessions]
    # Default selection: current session, or most recent
    default_idx = 0
    if current_id:
        for i, (sid, _) in enumerate(sessions):
            if sid == current_id:
                default_idx = i
                break

    picked_label = st.selectbox(
        "Pick a session", labels, index=default_idx,
        label_visibility="collapsed",
    )
    if picked_label:
        # Extract session id from "sid — ago"
        picked_sid = picked_label.split(" — ")[0]
        if picked_sid != current_id:
            return picked_sid
    return None