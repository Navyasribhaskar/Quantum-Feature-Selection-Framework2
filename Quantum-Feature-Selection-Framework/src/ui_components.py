"""
ui_components.py
==================
Reusable Streamlit UI building blocks shared across every dashboard page:
custom CSS injection, the hero header, metric/feature cards, step-pill
progress indicators, callout boxes, and centralized session-state
initialization.

Keeping these in one place is what gives the multi-page app a consistent,
"product" feel rather than nine independently-styled pages.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from src.utils import ASSETS_DIR, APP_TITLE, APP_SUBTITLE, SKeys


# --------------------------------------------------------------------------
# CSS / page config
# --------------------------------------------------------------------------


def load_css(css_filename: str = "style.css") -> None:
    """Inject the shared stylesheet into the current Streamlit page."""
    css_path = os.path.join(ASSETS_DIR, css_filename)
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def configure_page(page_title: str, page_icon: str = "🧬") -> None:
    """Standard st.set_page_config call, kept identical across all pages."""
    st.set_page_config(
        page_title=f"{page_title} | {APP_TITLE}",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------


def init_session_state() -> None:
    """Initialize every session_state key the app relies on, exactly once.

    Pages call this at import time so that reading `st.session_state[...]`
    never raises a KeyError, even if the user jumps directly to a later
    page via the sidebar without visiting earlier ones first.
    """
    defaults = {
        SKeys.RAW_DF: None,
        SKeys.DATASET_NAME: None,
        SKeys.DATASET_INFO: None,
        SKeys.METADATA_COLUMNS: None,
        SKeys.TARGET_COLUMN: None,
        SKeys.PREPROCESS_DONE: False,
        SKeys.FEATURE_SELECTION_DONE: False,
        SKeys.QUANTUM_ANALYSIS_DONE: False,
        SKeys.CLASSIFICATION_DONE: False,
        SKeys.TOP_K: None,
        SKeys.N_QUBITS: None,
        SKeys.SHOTS: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def render_sidebar_brand() -> None:
    """Render the branding block at the top of the sidebar (all pages)."""
    with st.sidebar:
        st.markdown(
            """
            <div class="qfsf-sidebar-title">🧬 Quantum Feature Selection</div>
            <div class="qfsf-sidebar-sub">Hybrid Quantum-Classical ML Framework</div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()


def render_pipeline_status() -> None:
    """Render a compact checklist of pipeline stage completion in the sidebar."""
    stages = [
        ("Dataset Loaded", st.session_state.get(SKeys.RAW_DF) is not None),
        ("Preprocessing", st.session_state.get(SKeys.PREPROCESS_DONE, False)),
        ("Feature Selection", st.session_state.get(SKeys.FEATURE_SELECTION_DONE, False)),
        ("Quantum Analysis", st.session_state.get(SKeys.QUANTUM_ANALYSIS_DONE, False)),
        ("Classification", st.session_state.get(SKeys.CLASSIFICATION_DONE, False)),
    ]
    with st.sidebar:
        st.caption("PIPELINE STATUS")
        for label, done in stages:
            icon = "✅" if done else "⬜"
            st.markdown(f"{icon} {label}")
        st.divider()
        st.caption("Quantum Feature Selection Framework v1.0")


# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------


def render_hero(title: str, subtitle: str, badges: Optional[list[str]] = None) -> None:
    badges = badges or []
    badge_html = "".join(f'<span class="qfsf-badge">{b}</span>' for b in badges)
    st.markdown(
        f"""
        <div class="qfsf-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="qfsf-badges">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Cards
# --------------------------------------------------------------------------


def render_feature_card(icon: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="qfsf-card">
            <span class="qfsf-icon">{icon}</span>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, sub: str = "", variant: str = "") -> None:
    st.markdown(
        f"""
        <div class="qfsf-metric {variant}">
            <div class="qfsf-metric-label">{label}</div>
            <div class="qfsf-metric-value">{value}</div>
            <div class="qfsf-metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(icon: str, title: str) -> None:
    st.markdown(f'<div class="qfsf-section-title">{icon} {title}</div>', unsafe_allow_html=True)


def render_callout(message: str, kind: str = "info") -> None:
    """kind: 'info' | 'success' | 'warning'"""
    st.markdown(f'<div class="qfsf-callout {kind}">{message}</div>', unsafe_allow_html=True)


def render_step_pills(steps: list, done_upto: int = -1) -> None:
    """Render a horizontal row of step-name pills, marking the first
    `done_upto + 1` as completed.
    """
    html = ""
    for i, step in enumerate(steps):
        cls = "qfsf-step-pill done" if i <= done_upto else "qfsf-step-pill"
        mark = "✓ " if i <= done_upto else ""
        html += f'<span class="{cls}">{mark}{step}</span>'
    st.markdown(html, unsafe_allow_html=True)


def render_flow_diagram(steps: list[dict]) -> None:
    """Render a horizontal workflow diagram from a list of
    {'icon': str, 'label': str, 'sub': str} dicts, connected by arrows.

    Implemented as plain HTML/CSS (rather than graphviz) so it renders
    reliably with zero extra system dependencies.
    """
    boxes = []
    for step in steps:
        boxes.append(
            f"""
            <div style="
                background: white; border: 1px solid #E2E8F0; border-radius: 12px;
                padding: 0.9rem 1rem; min-width: 150px; text-align: center;
                box-shadow: 0 1px 3px rgba(15,23,42,0.06); flex: 1;">
                <div style="font-size: 1.6rem;">{step['icon']}</div>
                <div style="font-weight: 700; font-size: 0.88rem; color:#1E293B; margin-top: 0.3rem;">{step['label']}</div>
                <div style="font-size: 0.75rem; color:#64748B; margin-top: 0.15rem;">{step.get('sub', '')}</div>
            </div>
            """
        )
    arrow = (
        '<div style="display:flex; align-items:center; color:#8B5CF6; '
        'font-size:1.3rem; padding: 0 0.3rem;">&rarr;</div>'
    )
    row = arrow.join(boxes)
    st.markdown(
        f'<div style="display:flex; align-items:stretch; gap:0; overflow-x:auto; padding: 0.5rem 0;">{row}</div>',
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="qfsf-footer">
            Quantum Feature Selection Framework for Gene Expression Analysis &middot;
            Built with Streamlit, Qiskit &amp; scikit-learn
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Guard helper for out-of-order navigation
# --------------------------------------------------------------------------


def require_stage(done_flag_key: str, message: str, redirect_page_label: str) -> bool:
    """Check a prerequisite session_state flag; render a friendly warning
    and return False if the prerequisite stage hasn't been completed yet.

    Usage:
        if not require_stage(SKeys.PREPROCESS_DONE, "...", "Preprocessing"):
            st.stop()
    """
    if not st.session_state.get(done_flag_key, False):
        render_callout(
            f"⚠️ {message} Please complete the <b>{redirect_page_label}</b> step first "
            f"using the sidebar navigation.",
            kind="warning",
        )
        return False
    return True
