"""
quantum_pipeline.py
=====================
Quantum feature encoding and simulation, extracted from the source
notebook's quantum section.

Pipeline (per sample, unchanged from the notebook):
    1. Take the first `n_qubits` selected (ANOVA) features of a sample.
    2. Angle-encode each feature into a qubit via an RY rotation.
    3. Entangle the qubits with a linear chain of CX gates
       (0->1, 1->2, 2->3, ...).
    4. Measure all qubits.
    5. Execute on Qiskit Aer's AerSimulator.
    6. Convert the resulting measurement-count dictionary into a fixed
       length (2**n_qubits) count vector -> this is the "quantum feature
       vector" used for classification.

A few additional helpers (not in the notebook) are provided for the
dashboard's "Quantum Analysis" page: a demo single-qubit Hadamard
circuit (mirrors the notebook's introductory cell), a statevector /
Bloch-sphere helper, and circuit metadata (depth, qubit count, gate
counts) for display.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from src.utils import DEFAULT_N_QUBITS, DEFAULT_SHOTS, get_logger

log = get_logger(__name__)

# A single shared simulator instance is reused across calls for efficiency.
_SIMULATOR = AerSimulator()


# --------------------------------------------------------------------------
# Circuit construction
# --------------------------------------------------------------------------


def build_encoding_circuit(
    sample: np.ndarray,
    n_qubits: int = DEFAULT_N_QUBITS,
    add_measurement: bool = True,
) -> QuantumCircuit:
    """Build the angle-encoding + entanglement circuit for one sample.

    Mirrors the notebook exactly:
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.ry(sample[i], i)
        qc.cx(0, 1); qc.cx(1, 2); qc.cx(2, 3); ...
        qc.measure_all()   # if add_measurement

    Parameters
    ----------
    sample:
        1D array of at least `n_qubits` numeric feature values (typically
        the first `n_qubits` ANOVA-selected, standardized features).
    n_qubits:
        Number of qubits (and features) to encode.
    add_measurement:
        Whether to append `measure_all()`. Set to False when a statevector
        (e.g. for a Bloch-sphere plot) is needed instead of measurement
        counts, since measurement collapses the state.
    """
    if len(sample) < n_qubits:
        raise ValueError(
            f"Sample has only {len(sample)} feature(s); at least {n_qubits} "
            f"are required for {n_qubits}-qubit angle encoding."
        )

    qc = QuantumCircuit(n_qubits)

    # Angle encoding: one RY rotation per feature/qubit.
    for i in range(n_qubits):
        qc.ry(float(sample[i]), i)

    # Entanglement: linear chain of CNOTs.
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    if add_measurement:
        qc.measure_all()

    return qc


def build_demo_hadamard_circuit() -> QuantumCircuit:
    """Build the notebook's introductory 1-qubit Hadamard demo circuit.

    Used on the Quantum Analysis / Home pages to give users a friendly,
    minimal illustration of superposition before showing the full
    4-qubit encoding circuit.
    """
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


# --------------------------------------------------------------------------
# Simulation / execution
# --------------------------------------------------------------------------


def run_circuit(qc: QuantumCircuit, shots: int = DEFAULT_SHOTS) -> dict[str, int]:
    """Transpile and execute a circuit on the Aer simulator, returning counts."""
    compiled = transpile(qc, _SIMULATOR)
    job = _SIMULATOR.run(compiled, shots=shots)
    result = job.result()
    return result.get_counts()


def get_statevector(sample: np.ndarray, n_qubits: int = DEFAULT_N_QUBITS) -> Statevector:
    """Return the exact statevector for a sample's encoding circuit
    (no measurement, no sampling noise) -- used for the Bloch-sphere view.
    """
    qc = build_encoding_circuit(sample, n_qubits=n_qubits, add_measurement=False)
    return Statevector.from_instruction(qc)


# --------------------------------------------------------------------------
# Counts <-> feature vector conversion
# --------------------------------------------------------------------------


def counts_to_vector(counts: dict[str, int], n_qubits: int = DEFAULT_N_QUBITS) -> list[int]:
    """Convert a measurement-count dictionary into a fixed-length vector.

    Mirrors the notebook's `counts_to_vector`: enumerates every possible
    n_qubits-bit bitstring (2**n_qubits of them) in ascending order and
    looks up its count (0 if that state was never measured). This keeps
    the resulting feature vector's length and ordering identical across
    samples, which is required for downstream classification.
    """
    n_states = 2 ** n_qubits
    states = [format(i, f"0{n_qubits}b") for i in range(n_states)]
    return [counts.get(state, 0) for state in states]


def quantum_feature_extractor(
    sample: np.ndarray,
    n_qubits: int = DEFAULT_N_QUBITS,
    shots: int = DEFAULT_SHOTS,
) -> dict[str, int]:
    """Build, execute, and return measurement counts for one sample.

    Direct modularization of the notebook's `quantum_feature_extractor`.
    """
    qc = build_encoding_circuit(sample, n_qubits=n_qubits, add_measurement=True)
    return run_circuit(qc, shots=shots)


def transform_to_quantum_features(
    X: np.ndarray,
    n_qubits: int = DEFAULT_N_QUBITS,
    shots: int = DEFAULT_SHOTS,
    progress_callback: Optional[callable] = None,
) -> np.ndarray:
    """Transform a batch of classical samples into quantum feature vectors.

    For each row of X (using only its first `n_qubits` columns), builds
    the encoding circuit, simulates it, and converts the resulting counts
    into a `2**n_qubits`-length count vector. Equivalent to the notebook's
    per-sample loop that built `X_train_quantum_features` /
    `X_test_quantum_features`.

    Parameters
    ----------
    X:
        Array of shape (n_samples, >= n_qubits). Only the first n_qubits
        columns of each row are used for encoding.
    progress_callback:
        Optional callable(current_index: int, total: int) invoked after
        each sample, so a Streamlit page can render a live progress bar.
    """
    n_samples = X.shape[0]
    features = np.zeros((n_samples, 2 ** n_qubits), dtype=float)

    for idx in range(n_samples):
        sample = X[idx, :n_qubits]
        counts = quantum_feature_extractor(sample, n_qubits=n_qubits, shots=shots)
        features[idx] = counts_to_vector(counts, n_qubits=n_qubits)
        if progress_callback is not None:
            progress_callback(idx + 1, n_samples)

    log.info("Transformed %d samples into %d-dim quantum feature vectors", n_samples, 2 ** n_qubits)
    return features


# --------------------------------------------------------------------------
# Circuit metadata (for the Quantum Analysis page)
# --------------------------------------------------------------------------


def get_circuit_metadata(qc: QuantumCircuit) -> dict:
    """Return descriptive metadata about a circuit for UI display."""
    compiled = transpile(qc, _SIMULATOR)
    op_counts = dict(qc.count_ops())
    return {
        "n_qubits": qc.num_qubits,
        "n_clbits": qc.num_clbits,
        "depth": qc.depth(),
        "transpiled_depth": compiled.depth(),
        "gate_counts": op_counts,
        "total_gates": sum(op_counts.values()),
    }


def circuit_to_ascii(qc: QuantumCircuit) -> str:
    """Render a circuit as ASCII text (safe fallback when matplotlib/
    pylatexenc rendering is unavailable, e.g. on minimal deployments).
    """
    return str(qc.draw(output="text"))
