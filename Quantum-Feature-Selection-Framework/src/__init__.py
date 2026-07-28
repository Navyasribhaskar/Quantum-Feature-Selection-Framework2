"""
src
===
Core, UI-independent modules for the Quantum Feature Selection Framework:
data loading, preprocessing, feature selection, the quantum pipeline,
classical/quantum classification, visualization, and shared utilities.

These modules contain no Streamlit-specific logic (aside from
`ui_components.py`, which is the dashboard's shared UI layer) so they can
be unit-tested or reused outside the dashboard (e.g. from a notebook or
a CLI script).
"""
