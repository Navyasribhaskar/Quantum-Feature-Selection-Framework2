"""
1_Home.py
=========
Landing page: project title/description, workflow diagram, tech stack,
architecture overview, features, advantages, and a call-to-action to
start the pipeline.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui_components import (
    configure_page,
    init_session_state,
    render_sidebar_brand,
    render_pipeline_status,
    render_hero,
    render_feature_card,
    render_section_title,
    render_flow_diagram,
    render_footer,
)

configure_page("Home", page_icon="🏠")
init_session_state()
render_sidebar_brand()
render_pipeline_status()

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------

render_hero(
    title="🧬 Quantum Feature Selection Framework",
    subtitle=(
        "A hybrid quantum-classical machine learning platform for gene "
        "expression analysis — combining ANOVA-based classical feature "
        "selection with quantum angle-encoding circuits and SVM "
        "classification, built on Qiskit and scikit-learn."
    ),
    badges=["Qiskit Aer", "scikit-learn", "ANOVA F-test", "4-Qubit Encoding", "Streamlit"],
)

# --------------------------------------------------------------------------
# Project description
# --------------------------------------------------------------------------

render_section_title("📖", "About This Project")
st.markdown(
    """
This framework operationalizes a research pipeline for classifying **acute
leukemia subtypes** (ALL-B, ALL-T, and AML) from high-dimensional gene
expression microarray data (the Golub et al. leukemia dataset, 7,129
genes across 72 patient samples).

The pipeline is *hybrid*: it first uses classical statistics (an ANOVA
F-test) to reduce thousands of genes down to the most informative
handful, then encodes a subset of those genes into a parameterized
quantum circuit using **angle encoding**, entangles the resulting qubits,
and measures the circuit to derive a new, quantum-native feature
representation. Both the purely classical features and the
quantum-derived features are then classified with a Support Vector
Machine, so their predictive performance can be directly compared.
"""
)

# --------------------------------------------------------------------------
# Workflow diagram
# --------------------------------------------------------------------------

render_section_title("🔄", "Pipeline Workflow")
render_flow_diagram(
    [
        {"icon": "📂", "label": "Dataset", "sub": "Upload / load CSV"},
        {"icon": "🧹", "label": "Preprocessing", "sub": "Clean, encode, scale"},
        {"icon": "🎯", "label": "Feature Selection", "sub": "ANOVA top-K genes"},
        {"icon": "⚛️", "label": "Quantum Encoding", "sub": "Angle encoding + entanglement"},
        {"icon": "🧠", "label": "Classification", "sub": "Classical & Quantum SVM"},
        {"icon": "📊", "label": "Results", "sub": "Compare & export"},
    ]
)

# --------------------------------------------------------------------------
# Architecture overview
# --------------------------------------------------------------------------

render_section_title("🏗️", "Architecture Overview")
col1, col2 = st.columns([1.2, 1])
with col1:
    st.markdown(
        """
**Software architecture**

```
dashboard/          Streamlit UI (app.py + 9 pages)
src/
 ├─ data_loader.py       dataset ingestion & validation
 ├─ preprocessing.py     cleaning, encoding, scaling, splitting
 ├─ feature_selection.py ANOVA (SelectKBest) feature selection
 ├─ quantum_pipeline.py  Qiskit circuits, simulation, feature extraction
 ├─ classifier.py        SVM training & evaluation
 ├─ visualization.py     Plotly chart builders
 ├─ ui_components.py     shared Streamlit UI building blocks
 └─ utils.py             constants, session-state keys, helpers
data/                bundled + uploaded datasets
outputs/             exported results
```
        """
    )
with col2:
    st.markdown(
        """
**Design principles**
- Clear separation between ML logic (`src/`) and presentation (`dashboard/`)
- Every pipeline stage is independently re-runnable and cacheable
- Session state persists results across all 9 pages
- No notebook code duplicated — the notebook's exact algorithm is preserved
        """
    )

# --------------------------------------------------------------------------
# Technology stack
# --------------------------------------------------------------------------

render_section_title("🛠️", "Technology Stack")
tech_cols = st.columns(4)
tech_items = [
    ("🐍", "Python 3.10+", "Core language & scientific stack"),
    ("⚛️", "Qiskit + Aer", "Quantum circuit design & simulation"),
    ("🤖", "scikit-learn", "Preprocessing, ANOVA, SVM classification"),
    ("🎨", "Streamlit", "Interactive multi-page web application"),
    ("📈", "Plotly", "Interactive charts & dashboards"),
    ("🐼", "pandas / NumPy", "Data manipulation & numerical computing"),
    ("📊", "Matplotlib", "Circuit diagrams & Bloch sphere rendering"),
    ("🧪", "SciPy stats (via sklearn)", "ANOVA F-test statistical scoring"),
]
for i, (icon, name, desc) in enumerate(tech_items):
    with tech_cols[i % 4]:
        render_feature_card(icon, name, desc)

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

render_section_title("✨", "Key Features")
feat_cols = st.columns(3)
features = [
    ("📤", "Flexible Dataset Upload", "Bring your own gene-expression CSV, or explore the bundled Golub leukemia dataset."),
    ("🧹", "Transparent Preprocessing", "Every cleaning, encoding, and scaling step is logged and displayed step-by-step."),
    ("🎯", "ANOVA Feature Selection", "Statistically rank and select the most discriminative genes with adjustable top-K."),
    ("⚛️", "Live Quantum Circuits", "Visualize the exact angle-encoding + entanglement circuit built for each sample."),
    ("🧠", "Dual Classification", "Train and evaluate both a classical SVM and a quantum-enhanced SVM."),
    ("📊", "Rich Comparisons", "Confusion matrices, ROC curves, and classical-vs-quantum accuracy comparisons."),
    ("📥", "One-Click Exports", "Download predictions, metrics, reports, and graphs for your own write-ups."),
    ("🖥️", "Research-Grade UI", "A polished interface suitable for demos, coursework, and stakeholder presentations."),
    ("🔒", "Robust Error Handling", "Friendly guidance for invalid files, missing columns, or execution failures."),
]
for i, (icon, title, desc) in enumerate(features):
    with feat_cols[i % 3]:
        render_feature_card(icon, title, desc)

# --------------------------------------------------------------------------
# Advantages
# --------------------------------------------------------------------------

render_section_title("🏆", "Why a Hybrid Quantum-Classical Approach?")
adv_col1, adv_col2 = st.columns(2)
with adv_col1:
    st.markdown(
        """
**Advantages explored by this framework**
- Classical ANOVA selection is fast, interpretable, and scales to thousands of genes
- Quantum angle encoding + entanglement can capture non-linear feature interactions classical scaling alone may miss
- Comparing both approaches on the same data offers an empirical, reproducible way to evaluate quantum feature maps for biomedical data
- Runs entirely on a local simulator — no quantum hardware access required
        """
    )
with adv_col2:
    st.markdown(
        """
**Honest limitations**
- Current NISQ-era simulators/hardware limit practical qubit counts (4 qubits here)
- Quantum feature extraction is comparatively slow — expect seconds-to-minutes for the full dataset
- On this dataset, the classical baseline currently outperforms the quantum-derived features (see Results page) — the point of this framework is to make that comparison rigorous and repeatable, not to claim quantum advantage
        """
    )

# --------------------------------------------------------------------------
# Call to action
# --------------------------------------------------------------------------

render_section_title("🚀", "Get Started")
st.markdown("Ready to run the pipeline on your own gene expression data, or explore the bundled dataset?")
cta_col1, cta_col2, _ = st.columns([1, 1, 2])
with cta_col1:
    if st.button("📂 Go to Dataset Page", use_container_width=True):
        if hasattr(st, "switch_page"):
            st.switch_page("pages/2_Dataset.py")
with cta_col2:
    if st.button("ℹ️ Learn More (About)", use_container_width=True):
        if hasattr(st, "switch_page"):
            st.switch_page("pages/9_About.py")

render_footer()
