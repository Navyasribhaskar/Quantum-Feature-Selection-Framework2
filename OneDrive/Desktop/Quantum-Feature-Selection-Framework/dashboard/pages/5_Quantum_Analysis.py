"""
5_Quantum_Analysis.py
=======================
Visualizes the quantum encoding pipeline: circuit diagram, gate/qubit
metadata, Bloch-sphere state view, measurement histogram, and the
derived quantum feature vector for a chosen sample. Also runs (and
caches) the full-batch quantum feature extraction used by the
Classification page.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.quantum_pipeline import (
    build_encoding_circuit,
    counts_to_vector,
    get_circuit_metadata,
    get_statevector,
    quantum_feature_extractor,
    transform_to_quantum_features,
)
from src.ui_components import (
    configure_page,
    init_session_state,
    render_callout,
    render_flow_diagram,
    render_footer,
    render_hero,
    render_metric_card,
    render_pipeline_status,
    render_section_title,
    render_sidebar_brand,
    require_stage,
)
from src.utils import DEFAULT_N_QUBITS, DEFAULT_SHOTS, SKeys
from src.visualization import plot_quantum_feature_vector, plot_quantum_histogram

configure_page("Quantum Analysis", page_icon="⚛️")
init_session_state()
render_sidebar_brand()
render_pipeline_status()

render_hero(
    title="⚛️ Quantum Analysis",
    subtitle="Angle-encode selected genes into a parameterized quantum circuit, entangle the qubits, and derive quantum feature vectors.",
)

if not require_stage(SKeys.FEATURE_SELECTION_DONE, "Feature selection hasn't been run yet.", "Feature Selection"):
    render_footer()
    st.stop()

render_section_title("🔄", "Quantum Encoding Workflow")
render_flow_diagram(
    [
        {"icon": "🧬", "label": "Top-4 Genes", "sub": "First 4 ANOVA-selected features"},
        {"icon": "🔄", "label": "Angle Encoding", "sub": "RY(feature) per qubit"},
        {"icon": "🔗", "label": "Entanglement", "sub": "Linear CX chain"},
        {"icon": "📏", "label": "Measurement", "sub": "AerSimulator, N shots"},
        {"icon": "🧮", "label": "Feature Vector", "sub": "16-dim count vector"},
    ]
)

X_train_selected = st.session_state[SKeys.X_TRAIN_SELECTED]
X_test_selected = st.session_state[SKeys.X_TEST_SELECTED]

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

render_section_title("⚙️", "Quantum Configuration")
c1, c2, c3 = st.columns(3)
with c1:
    n_qubits = st.slider("Number of qubits", min_value=2, max_value=min(6, X_train_selected.shape[1]), value=DEFAULT_N_QUBITS)
with c2:
    shots = st.select_slider("Shots per circuit", options=[64, 128, 256, 512, 1024], value=DEFAULT_SHOTS)
with c3:
    sample_source = st.selectbox("Preview sample from", options=["Training set", "Testing set"])

preview_pool = X_train_selected if sample_source == "Training set" else X_test_selected
sample_idx = st.slider("Sample index to visualize", min_value=0, max_value=preview_pool.shape[0] - 1, value=0)
sample = preview_pool[sample_idx, :n_qubits]

st.session_state[SKeys.N_QUBITS] = n_qubits
st.session_state[SKeys.SHOTS] = shots

# --------------------------------------------------------------------------
# Circuit diagram
# --------------------------------------------------------------------------

render_section_title("🔌", f"Encoding Circuit for Sample #{sample_idx}")
st.caption(f"Encoded feature values (first {n_qubits} selected genes, standardized): {np.round(sample, 4).tolist()}")

qc = build_encoding_circuit(sample, n_qubits=n_qubits, add_measurement=True)

try:
    fig = qc.draw(output="mpl", style={"backgroundcolor": "#FFFFFF"})
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
except Exception:
    render_callout(
        "ℹ️ Matplotlib circuit rendering unavailable in this environment; showing text diagram instead.",
        kind="info",
    )
    st.code(str(qc.draw(output="text")), language=None)

meta = get_circuit_metadata(qc)
m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Qubits", f"{meta['n_qubits']}")
with m2:
    render_metric_card("Circuit Depth", f"{meta['depth']}", variant="accent")
with m3:
    render_metric_card("Total Gates", f"{meta['total_gates']}", variant="secondary")
with m4:
    render_metric_card("Transpiled Depth", f"{meta['transpiled_depth']}", variant="warning")

st.markdown("**Gate counts:** " + ", ".join(f"`{gate}`: {count}" for gate, count in meta["gate_counts"].items()))

# --------------------------------------------------------------------------
# Bloch sphere
# --------------------------------------------------------------------------

render_section_title("🌐", "Bloch Sphere (Pre-Measurement State)")
try:
    from qiskit.visualization import plot_bloch_multivector

    statevector = get_statevector(sample, n_qubits=n_qubits)
    bloch_fig = plot_bloch_multivector(statevector)
    st.pyplot(bloch_fig, use_container_width=True)
    plt.close(bloch_fig)
except Exception as exc:  # noqa: BLE001
    render_callout(f"ℹ️ Bloch sphere visualization unavailable: {exc}", kind="info")

# --------------------------------------------------------------------------
# Measurement + feature vector for the previewed sample
# --------------------------------------------------------------------------

render_section_title("📏", "Measurement Results")
with st.spinner("Executing circuit on AerSimulator..."):
    counts = quantum_feature_extractor(sample, n_qubits=n_qubits, shots=shots)
st.session_state[SKeys.SAMPLE_CIRCUIT_COUNTS] = counts

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(plot_quantum_histogram(counts, title="Measured Basis-State Counts"), use_container_width=True)
with col_b:
    vector = counts_to_vector(counts, n_qubits=n_qubits)
    st.plotly_chart(plot_quantum_feature_vector(vector, title="Quantum Feature Vector (this sample)"), use_container_width=True)

with st.expander("🔢 Raw measurement counts"):
    st.json(counts)

# --------------------------------------------------------------------------
# Full-batch quantum feature extraction
# --------------------------------------------------------------------------

render_section_title("🚀", "Full-Batch Quantum Feature Extraction")
render_callout(
    "This runs the quantum encoding + simulation pipeline across every training and testing "
    "sample, producing the quantum feature matrices used on the Classification page. "
    "Runtime scales with the number of samples × shots.",
    kind="info",
)

run_batch = st.button("⚛️ Run Full Quantum Feature Extraction", type="primary")

if run_batch:
    n_train = X_train_selected.shape[0]
    n_test = X_test_selected.shape[0]
    total = n_train + n_test
    progress_bar = st.progress(0, text="Starting quantum feature extraction...")

    def _make_callback(offset: int):
        def _cb(current: int, total_local: int):
            done = offset + current
            progress_bar.progress(
                min(done / total, 1.0),
                text=f"Encoding sample {done}/{total} on {n_qubits}-qubit circuit...",
            )
        return _cb

    try:
        X_train_quantum = transform_to_quantum_features(
            X_train_selected, n_qubits=n_qubits, shots=shots, progress_callback=_make_callback(0)
        )
        X_test_quantum = transform_to_quantum_features(
            X_test_selected, n_qubits=n_qubits, shots=shots, progress_callback=_make_callback(n_train)
        )
    except Exception as exc:  # noqa: BLE001
        progress_bar.empty()
        render_callout(f"❌ Quantum feature extraction failed: {exc}", kind="warning")
        st.stop()

    progress_bar.progress(1.0, text="Quantum feature extraction complete!")

    st.session_state[SKeys.X_TRAIN_QUANTUM_RAW] = X_train_selected[:, :n_qubits]
    st.session_state[SKeys.X_TEST_QUANTUM_RAW] = X_test_selected[:, :n_qubits]
    st.session_state[SKeys.X_TRAIN_QUANTUM_FEATURES] = X_train_quantum
    st.session_state[SKeys.X_TEST_QUANTUM_FEATURES] = X_test_quantum
    st.session_state[SKeys.QUANTUM_ANALYSIS_DONE] = True
    st.session_state[SKeys.CLASSIFICATION_DONE] = False
    st.success(
        f"✅ Generated quantum feature vectors: training {X_train_quantum.shape}, testing {X_test_quantum.shape}"
    )

if st.session_state.get(SKeys.QUANTUM_ANALYSIS_DONE):
    render_section_title("✅", "Quantum Feature Matrices Ready")
    m1, m2 = st.columns(2)
    with m1:
        render_metric_card(
            "Training Quantum Features",
            f"{st.session_state[SKeys.X_TRAIN_QUANTUM_FEATURES].shape}",
            "samples × 2^n_qubits",
            variant="secondary",
        )
    with m2:
        render_metric_card(
            "Testing Quantum Features",
            f"{st.session_state[SKeys.X_TEST_QUANTUM_FEATURES].shape}",
            "samples × 2^n_qubits",
            variant="accent",
        )

    st.divider()
    next_col1, _ = st.columns([1, 3])
    with next_col1:
        if st.button("➡️ Proceed to Classification", type="primary", use_container_width=True):
            if hasattr(st, "switch_page"):
                st.switch_page("pages/6_Classification.py")

render_footer()
