"""Chat history renderer + input box.

Pure presentation: reads from `session.messages` and renders via
st.chat_message. The user input goes through st.chat_input which Streamlit
manages internally (no need for input() like in the CLI).
"""
from __future__ import annotations

import streamlit as st

from starry_code.session import Session
from starry_code.text.sanitize import strip_surrogates


def render_chat_history(session: Session) -> None:
    """Render the session's messages as chat bubbles.

    Tool messages are hidden (noisy); only user + assistant show.
    Long assistant messages are truncated for the chat preview.
    """
    for m in session.messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                preview = strip_surrogates(content)
                if len(preview) > 800:
                    preview = preview[:800] + "…"
                st.write(preview)


def render_user_input() -> str | None:
    """Render the chat_input box; return text if user submitted, else None.

    Streamlit's chat_input naturally clears the box on submit — we just
    need to capture the value.
    """
    return st.chat_input("Send a message to Starry Coach…")