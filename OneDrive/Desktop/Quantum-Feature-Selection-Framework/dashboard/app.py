"""
app.py
======
Main entry point for the Quantum Feature Selection Framework dashboard.

Run from the project root with:

    streamlit run dashboard/app.py

This file configures the page, loads shared session state, and lands the
user on the Home page. All actual page content lives under
`dashboard/pages/`, which Streamlit's native multipage-app mechanism
automatically lists in the sidebar.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root (parent of `dashboard/`) is importable as `src.*`
# regardless of the working directory Streamlit was launched from.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui_components import (
    configure_page,
    init_session_state,
    render_sidebar_brand,
    render_pipeline_status,
)

configure_page("Home", page_icon="🧬")
init_session_state()
render_sidebar_brand()
render_pipeline_status()

# Streamlit's multipage mechanism renders `pages/1_Home.py` etc. in the
# sidebar automatically. As soon as this app boots we hand off to the
# Home page so the user never sees a bare/blank entry screen.
_HOME_PAGE = os.path.join("pages", "1_Home.py")

if hasattr(st, "switch_page"):
    try:
        st.switch_page(_HOME_PAGE)
    except Exception:
        # st.switch_page requires the target to be a registered page; if
        # this Streamlit version/layout doesn't support it, fall back to
        # a simple landing message with manual navigation instructions.
        st.title("🧬 Quantum Feature Selection Framework")
        st.info("Use the sidebar to navigate to **1 Home** to get started.")
else:
    st.title("🧬 Quantum Feature Selection Framework")
    st.info("Use the sidebar to navigate to **1 Home** to get started.")
