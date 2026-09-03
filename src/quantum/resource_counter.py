from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ansatz import CircuitSpec
from .observables import Observable


@dataclass(frozen=True)
class ResourceCounts:
    qubits: int
    layers: int
    parameters: int
    k: int
    one_qubit_gates: int
    two_qubit_gates: int
    circuit_depth: int
    j_m: int
    g_m: int


def circuit_resources(spec: CircuitSpec, observable: Observable) -> ResourceCounts:
    one = spec.n_qubits + spec.n_qubits * spec.layers
    two = max(0, spec.n_qubits - 1) * spec.layers
    depth = 1 + 2 * spec.layers
    return ResourceCounts(spec.n_qubits, spec.layers, spec.n_parameters, spec.k_methodology, one, two, depth, observable.j_m, observable.g_m)


def analytic_shots(n_train: int, spec: CircuitSpec, observable: Observable, epochs: int, m_f: int, m_d: int, m_q: int) -> dict[str, int]:
    d = spec.n_parameters
    dm_epoch = n_train * (observable.g_m * m_f + 2 * observable.g_m * d * m_d)
    q_epoch = n_train * (observable.g_m * m_f + d * m_q)
    return {
        "dm_gradient_per_sample": int(2 * observable.g_m * d * m_d),
        "qndm_gradient_per_sample": int(d * m_q),
        "dm_epoch": int(dm_epoch),
        "qndm_epoch": int(q_epoch),
        "dm_total": int(epochs * dm_epoch),
        "qndm_total": int(epochs * q_epoch),
    }


def resource_frame(spec: CircuitSpec, observable: Observable, regime: str) -> pd.DataFrame:
    r = circuit_resources(spec, observable)
    return pd.DataFrame([{
        "qubits": r.qubits,
        "layers": r.layers,
        "parameters_d": r.parameters,
        "K": r.k,
        "one_qubit_gates": r.one_qubit_gates,
        "two_qubit_gates": r.two_qubit_gates,
        "circuit_depth": r.circuit_depth,
        "J_M": r.j_m,
        "G_M": r.g_m,
        "regime": regime,
    }])

