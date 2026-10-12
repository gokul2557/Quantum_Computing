"""
quantum/circuit.py
------------------
Parameterized Quantum Circuit architecture for Variational Quantum Classification.
Combines Angle Encoding, CNOT entanglement layers, and variational rotation gates.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
try:
    from quantum.encoding import build_angle_encoding_circuit
except (ModuleNotFoundError, ImportError):
    from encoding import build_angle_encoding_circuit

def build_zz_feature_map(feature_dimension: int = 4, reps: int = 2) -> QuantumCircuit:
    """
    Constructs the exact ZZFeatureMap used in QSVM (Slide 7):
    ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
    Encodes feature interactions via Rz and CNOT-Rz-CNOT (ZZ phase) gates.
    """
    fmap = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement="linear", insert_barriers=True)
    fmap.name = "ZZFeatureMap"
    return fmap

def build_variational_ansatz(
    num_qubits: int = 4,
    reps: int = 1,
    entanglement: str = "linear"
) -> QuantumCircuit:
    """
    Constructs a hardware-efficient parameterized variational circuit.
    Uses alternating layers of Ry rotations and linear CNOT entanglement gates:
      q0 ──Ry(w0)──●─────────────Ry(w4)──
                   │
      q1 ──Ry(w1)──X──●──────────Ry(w5)──
                      │
      q2 ──Ry(w2)─────X──●───────Ry(w6)──
                         │
      q3 ──Ry(w3)────────X───────Ry(w7)──
    """
    ansatz = RealAmplitudes(
        num_qubits=num_qubits,
        reps=reps,
        entanglement=entanglement,
        insert_barriers=True
    )
    ansatz.name = "VariationalAnsatz"
    return ansatz

def build_full_quantum_circuit(
    num_qubits: int = 4,
    reps: int = 1,
    entanglement: str = "linear"
) -> QuantumCircuit:
    """
    Assembles the full end-to-end quantum classification circuit:
    Input State |0000> -> Feature Map (Ry Angle Encoding) -> Variational Ansatz (Ry + CNOT).
    """
    feature_map = build_angle_encoding_circuit(num_qubits=num_qubits)
    ansatz = build_variational_ansatz(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
    
    full_circuit = QuantumCircuit(num_qubits, name="FraudDetection_VQC")
    full_circuit.compose(feature_map, inplace=True)
    full_circuit.barrier()
    full_circuit.compose(ansatz, inplace=True)
    return full_circuit

def get_circuit_summary(qc: QuantumCircuit) -> dict:
    """
    Extracts structural information from a QuantumCircuit.
    """
    gate_counts = dict(qc.count_ops())
    return {
        "num_qubits": qc.num_qubits,
        "circuit_depth": qc.depth(),
        "num_parameters": qc.num_parameters,
        "gate_counts": gate_counts,
        "total_gates": sum(gate_counts.values())
    }

def draw_circuit_figure(qc: QuantumCircuit, style: str = "clifford") -> plt.Figure:
    """
    Generates a high-resolution Matplotlib diagram of the genuine quantum circuit.
    """
    fig = qc.draw(output="mpl", style=style)
    return fig

if __name__ == "__main__":
    qc = build_full_quantum_circuit(4, reps=1)
    summary = get_circuit_summary(qc)
    print("Circuit summary:", summary)
