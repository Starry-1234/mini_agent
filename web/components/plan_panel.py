"""Right-side plan panel — show the live plan_cache + todo list.

Whyvs the a side panel matters:
- The plan_cache is the coach's working memory. Showing it gives users
  a sense of "where am I in this plan" without having to ask.
- The todo list is the immediate next step — useful when the user
  comes back after a day and forgets what they were doing.
"""
from __future__ import annotations

import streamlit as st

from starry_code.session import Session


def render_plan_panel(session: Session) -> None:
    """Render the plan + todo sidebar block. Always visible when a
    session is loaded.
    """
    st.subheader("Plan")

    pc = session.plan_cache
    if pc.get("long_term_goal"):
        st.markdown(f"**Goal:** {pc['long_term_goal']}")
    if pc.get("stage"):
        st.markdown(f"**Stage:** {pc['stage']} _(v{pc.get('version', 0)})_")
    if pc.get("next_task"):
        st.markdown(f"**Next:** {pc['next_task']}")

    if not any(pc.get(k) for k in ("stage", "next_task", "long_term_goal")):
        st.caption("(plan not set — coach will set it after first turn)")

    st.divider()
    st.subheader("Todos")
    if not session.todos:
        st.caption("(no todos)")
    else:
        for t in session.todos:
            mark = "x" if t.get("done") else " "
            sid = t.get("id", "?")
            text = t.get("text", "")
            st.markdown(f"- [{mark}] #{sid} {text}")