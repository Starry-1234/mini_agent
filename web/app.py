"""Streamlit web UI — Phase 3 W4a MVP.

Quick-start:
    pip install streamlit
    streamlit run web/app.py

Architecture:
- Reuses every Starry Code runtime component unchanged:
    run_turn(), build_default_registry(), build_memory(), SessionStore,
    TraceLogger, AutoNamer, llm_client factory.
- Streamlit manages UI state (current_session_id, chat messages,
  sidebar visibility). The UI is a transport — the same business
  logic as the CLI, just with st.chat_message() instead of print().

What's NOT here (deliberately):
- Authentication / multi-user (single-user, no auth — local app)
- Streaming (chat round-trip is fast enough; can be added via
  st.write_stream() once we add streaming to run_turn)
- Mobile-specific layout (Streamlit responsive CSS is good enough)
"""
from __future__ import annotations
import sys
from pathlib import Path

# Make `starry_code` importable when launched as `streamlit run web/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402 — must come after sys.path tweak

from starry_code.config import Settings
from starry_code.session import Session, SessionStore
from starry_code.llm import LLMClient, MockLLMClient, make_default_mock_llm
from starry_code.runtime import run_turn, build_default_registry, build_memory
from starry_code.trace import TraceLogger
from starry_code.naming import AutoNamer
from starry_code.text.sanitize import strip_surrogates

from .components.session_picker import render_session_picker, new_session_id
from .components.plan_panel import render_plan_panel
from .components.chat import render_chat_history, render_user_input


# ---- One-time bootstrap (per Streamlit session) ----

@st.cache_resource(show_spinner=False)
def bootstrap():
    """Set up settings + llm + registry + memory + trace once per session.

    Cached so repeated Streamlit reruns don't re-init.
    """
    settings = Settings.from_env(
        sessions_dir=Path("sessions"),
        load_dotenv=True,
    )
    if settings.llm_api_key:
        llm = LLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
    else:
        # No key — fall back to mock so the UI still works for
        # demoing without burning API credits.
        llm = MockLLMClient()

    registry = build_default_registry()
    memory = build_memory(settings=settings, llm=llm)
    return settings, llm, registry, memory


def _ask(text: str, session: Session, store: SessionStore,
         trace: TraceLogger, settings, llm, registry, memory,
         autonamer: AutoNamer | None) -> str:
    """Run one turn (mirrors cli.py:ask but without stdout printing)."""
    answer = run_turn(
        session, text,
        settings=settings, llm=llm, registry=registry,
        memory=memory, trace=trace,
    )
    store.save(session)
    if autonamer is not None and autonamer._fired is False:
        autonamer.try_name(llm, text, session, trace, settings.sessions_dir)
    return strip_surrogates(answer)


def main():
    st.set_page_config(
        page_title="Starry Coach",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("✦ Starry Coach")
    st.caption("程序员学习与就业规划教练 — W4a MVP (Streamlit)")

    settings, llm, registry, memory = bootstrap()

    # ---- Session state init ----
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "autonamer" not in st.session_state:
        st.session_state.autonamer = None
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list of (role, content) for display

    store = SessionStore(settings.sessions_dir)

    # ---- Sidebar: session picker ----
    with st.sidebar:
        st.subheader("Sessions")
        picked = render_session_picker(store, st.session_state.current_session_id)
        if picked:
            st.session_state.current_session_id = picked
            st.session_state.chat = []  # reset chat display on switch
            st.rerun()

        if st.button("➕ New session"):
            new_id = new_session_id()
            st.session_state.current_session_id = new_id
            st.session_state.chat = []
            st.session_state.autonamer = AutoNamer()
            st.rerun()

        st.divider()
        if st.session_state.current_session_id:
            st.caption(f"current: **{st.session_state.current_session_id}**")

    # ---- Load current session ----
    sid = st.session_state.current_session_id
    if not sid:
        st.info("Pick a session in the sidebar or click ➕ New session.")
        return

    session = store.load(sid)
    autonamer = st.session_state.autonamer if sid.startswith("auto-") else None
    trace = TraceLogger(settings.sessions_dir, sid)

    # ---- Main: chat + plan panel ----
    col_chat, col_plan = st.columns([3, 1])
    with col_chat:
        render_chat_history(session)
        user_msg = render_user_input()
        if user_msg:
            with st.spinner("Coach is thinking…"):
                answer = _ask(
                    user_msg, session, store, trace, settings, llm,
                    registry, memory, autonamer,
                )
            # Refresh session name if autonamer fired
            if autonamer is not None and session.id != sid:
                st.session_state.current_session_id = session.id
                st.session_state.autonamer = None
                st.rerun()
            else:
                st.rerun()

    with col_plan:
        render_plan_panel(session)


if __name__ == "__main__":
    main()