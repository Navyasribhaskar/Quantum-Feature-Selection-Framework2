"""
9_About.py
===========
Static informational page: motivation, methodology, technology stack,
a quantum-computing primer, references, future scope, author info, and
acknowledgements.
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
    render_feature_card,
    render_footer,
    render_hero,
    render_pipeline_status,
    render_section_title,
    render_sidebar_brand,
)

configure_page("About", page_icon="ℹ️")
init_session_state()
render_sidebar_brand()
render_pipeline_status()

render_hero(
    title="ℹ️ About This Project",
    subtitle="Motivation, methodology, technology, and the research context behind the Quantum Feature Selection Framework.",
)

# --------------------------------------------------------------------------
# Motivation
# --------------------------------------------------------------------------

render_section_title("💡", "Project Motivation")
st.markdown(
    """
Gene expression microarray datasets are notoriously high-dimensional and
low-sample: the bundled Golub et al. leukemia dataset, for instance,
contains **7,129 gene-expression measurements** across just **72 patient
samples**. This "curse of dimensionality" makes classical machine
learning prone to overfitting and demands aggressive, principled feature
reduction before any classifier can generalize well.

At the same time, quantum computing has emerged as a candidate tool for
representing high-dimensional data in richer, entangled feature spaces
that may expose patterns invisible to purely classical models. This
project was motivated by a simple, practical question: **on a real
biomedical classification task, how does a quantum-encoded feature
representation compare to a classical, statistically-selected one, when
evaluated under an identical experimental protocol?**
"""
)

# --------------------------------------------------------------------------
# Methodology
# --------------------------------------------------------------------------

render_section_title("🔬", "Methodology")
st.markdown(
    """
1. **Data Cleaning** — Clinical metadata columns (patient ID, gender,
   sample source, tissue type) are removed, retaining only gene-expression
   measurements and the target label.
2. **Encoding & Scaling** — The target (cancer subtype) is label-encoded;
   all gene-expression features are standardized to zero mean and unit
   variance.
3. **Train/Test Split** — A stratified 80/20 split preserves class
   proportions across training and testing sets.
4. **Classical Feature Selection** — An ANOVA F-test (`SelectKBest` with
   `f_classif`) ranks every gene by how well it statistically separates
   the classes; the top-K are retained (100 by default).
5. **Classical Classification** — A linear-kernel Support Vector Machine
   is trained directly on the top-K selected genes.
6. **Quantum Encoding** — The first *N* (default 4) selected genes for
   each sample are angle-encoded into an *N*-qubit circuit via RY
   rotations, then entangled with a linear chain of CNOT gates.
7. **Quantum Simulation** — Each circuit is measured on Qiskit Aer's
   `AerSimulator`; the resulting measurement-count distribution
   (a `2^N`-dimensional vector) becomes that sample's *quantum feature
   vector*.
8. **Quantum-Enhanced Classification** — An identical linear-kernel SVM
   is trained on the quantum feature vectors instead of the raw genes.
9. **Comparison** — Both models are evaluated on the same held-out test
   set using accuracy, precision, recall, F1-score, confusion matrices,
   and ROC curves.
"""
)

# --------------------------------------------------------------------------
# Technology stack
# --------------------------------------------------------------------------

render_section_title("🛠️", "Technology Stack")
tech_cols = st.columns(3)
tech_items = [
    ("⚛️", "Qiskit + Qiskit Aer", "Quantum circuit construction and local statevector/measurement simulation."),
    ("🤖", "scikit-learn", "StandardScaler, LabelEncoder, SelectKBest/f_classif, SVC, and evaluation metrics."),
    ("🎨", "Streamlit", "The multi-page interactive web application framework powering this dashboard."),
    ("📈", "Plotly", "Interactive, publication-quality charts throughout the app."),
    ("🐼", "pandas & NumPy", "Tabular data handling and numerical array operations."),
    ("📄", "ReportLab", "Server-side PDF report generation for the Download page."),
]
for i, (icon, name, desc) in enumerate(tech_items):
    with tech_cols[i % 3]:
        render_feature_card(icon, name, desc)

# --------------------------------------------------------------------------
# Quantum computing overview
# --------------------------------------------------------------------------

render_section_title("⚛️", "Quantum Computing Overview")
st.markdown(
    """
**Qubits & superposition.** Unlike a classical bit (0 or 1), a qubit can
exist in a superposition of both states simultaneously. A single-qubit
Hadamard gate, for example, places a qubit into an equal superposition —
measuring it yields 0 or 1 with roughly 50/50 probability, as shown by
the introductory demo circuit referenced in the source notebook.

**Angle encoding.** This framework encodes each classical feature value
as a *rotation angle*. An `RY(θ)` gate rotates a qubit's state around the
Y-axis of the Bloch sphere by angle θ, where θ is set directly from a
(standardized) gene-expression value. This maps continuous classical
data into quantum amplitudes.

**Entanglement.** After encoding, a linear chain of CNOT gates entangles
neighboring qubits, so that the qubits' joint measurement outcomes
depend on *combinations* of the input features, not just each feature in
isolation — this is the mechanism by which a quantum feature map can, in
principle, expose interactions a linear classical scaling cannot.

**Measurement & feature vectors.** Measuring an *N*-qubit circuit
collapses it to one of `2^N` possible classical bitstrings. Running the
circuit many times ("shots") and tabulating how often each bitstring
occurs produces a probability-like distribution — this distribution,
as a fixed-length vector, is the "quantum feature vector" used for
classification in this framework.
"""
)

# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------

render_section_title("📚", "References")
st.markdown(
    """
- Golub, T. R., et al. (1999). *Molecular Classification of Cancer: Class
  Discovery and Class Prediction by Gene Expression Monitoring.* Science,
  286(5439), 531–537.
- Qiskit Development Team. *Qiskit: An Open-source Framework for Quantum
  Computing.* https://www.ibm.com/quantum/qiskit
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in
  Python.* Journal of Machine Learning Research, 12, 2825–2830.
- Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum
  Computers.* Springer.
- Havlíček, V., et al. (2019). *Supervised learning with quantum-enhanced
  feature spaces.* Nature, 567(7747), 209–212.
"""
)

# --------------------------------------------------------------------------
# Future scope
# --------------------------------------------------------------------------

render_section_title("🚀", "Future Scope")
st.markdown(
    """
- **More qubits & advanced feature maps** — Explore larger quantum
  circuits and structured feature maps such as `ZZFeatureMap` or
  `PauliFeatureMap` rather than simple angle encoding.
- **Quantum kernel methods** — Evaluate a Quantum Kernel Estimation (QKE)
  approach / Quantum Support Vector Machine using a quantum kernel matrix
  directly, rather than treating measurement counts as classical
  features.
- **Real quantum hardware** — Execute the pipeline on IBM Quantum
  hardware (via Qiskit Runtime) instead of a noiseless local simulator,
  and study the effect of hardware noise.
- **Larger, multi-omics datasets** — Extend beyond the 72-sample Golub
  dataset to larger cohorts and additional omics data types (proteomics,
  methylation) for more statistically robust conclusions.
- **Hybrid quantum neural networks** — Investigate variational quantum
  circuits trained end-to-end (e.g. with parameter-shift gradients) as an
  alternative to the current fixed-encoding + classical-SVM approach.
"""
)

# --------------------------------------------------------------------------
# Author & acknowledgements
# --------------------------------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    render_section_title("👤", "Author Information")
    st.markdown(
        """
        **Capstone Project**
        Quantum Feature Selection Framework for Gene Expression Analysis
        Using Hybrid Quantum-Classical Machine Learning

        This dashboard was engineered from the author's original Jupyter
        Notebook research pipeline, preserving the notebook's algorithmic
        design while restructuring it into a modular, production-quality
        software application.
        """
    )
with col2:
    render_section_title("🙏", "Acknowledgements")
    st.markdown(
        """
        - The Qiskit and scikit-learn open-source communities
        - Golub et al. for the publicly available leukemia gene-expression dataset
        - Streamlit for the rapid application framework that powers this dashboard
        """
    )

render_footer()
