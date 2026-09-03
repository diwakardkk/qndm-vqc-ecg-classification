from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircuitSpec:
    n_qubits: int
    layers: int

    @property
    def n_parameters(self) -> int:
        return self.n_qubits * self.layers

    @property
    def k_methodology(self) -> int:
        return self.n_qubits + (2 * self.n_qubits - 1) * self.layers


def ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def dry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return 0.5 * np.array([[-s, -c], [c, -s]], dtype=complex)


def apply_one_qubit(state: np.ndarray, gate: np.ndarray, wire: int, n_wires: int) -> np.ndarray:
    tensor = state.reshape([2] * n_wires)
    moved = np.moveaxis(tensor, wire, 0)
    updated = np.tensordot(gate, moved, axes=([1], [0]))
    updated = np.moveaxis(updated, 0, wire)
    return updated.reshape(-1)


def apply_cnot(state: np.ndarray, control: int, target: int, n_wires: int) -> np.ndarray:
    tensor = state.reshape([2] * n_wires)
    out = tensor.copy()
    idx0 = [slice(None)] * n_wires
    idx1 = [slice(None)] * n_wires
    idx0[control] = 1
    idx0[target] = 0
    idx1[control] = 1
    idx1[target] = 1
    tmp = out[tuple(idx0)].copy()
    out[tuple(idx0)] = out[tuple(idx1)]
    out[tuple(idx1)] = tmp
    return out.reshape(-1)


def zero_state(n_wires: int) -> np.ndarray:
    state = np.zeros(2**n_wires, dtype=complex)
    state[0] = 1.0
    return state


def encode(x: np.ndarray) -> np.ndarray:
    n = len(x)
    state = zero_state(n)
    for q, value in enumerate(x):
        state = apply_one_qubit(state, ry(float(value)), q, n)
    return state


def apply_variational(state: np.ndarray, theta: np.ndarray, spec: CircuitSpec) -> np.ndarray:
    n = spec.n_qubits
    out = state
    params = theta.reshape(spec.layers, n)
    for layer in range(spec.layers):
        for q in range(n):
            out = apply_one_qubit(out, ry(float(params[layer, q])), q, n)
        for q in range(n - 1):
            out = apply_cnot(out, q, q + 1, n)
    return out


def apply_variational_dagger(state: np.ndarray, theta: np.ndarray, spec: CircuitSpec, offset: int = 0, n_wires: int | None = None) -> np.ndarray:
    total = spec.n_qubits if n_wires is None else n_wires
    n = spec.n_qubits
    out = state
    params = theta.reshape(spec.layers, n)
    for layer in reversed(range(spec.layers)):
        for q in reversed(range(n - 1)):
            out = apply_cnot(out, offset + q, offset + q + 1, total)
        for q in reversed(range(n)):
            out = apply_one_qubit(out, ry(float(-params[layer, q])), offset + q, total)
    return out


def apply_variational_full(state: np.ndarray, theta: np.ndarray, spec: CircuitSpec, offset: int = 0, dagger: bool = False) -> np.ndarray:
    if dagger:
        return apply_variational_dagger(state, theta, spec, offset=offset, n_wires=spec.n_qubits + offset)
    total = spec.n_qubits + offset
    n = spec.n_qubits
    out = state
    params = theta.reshape(spec.layers, n)
    for layer in range(spec.layers):
        for q in range(n):
            out = apply_one_qubit(out, ry(float(params[layer, q])), offset + q, total)
        for q in range(n - 1):
            out = apply_cnot(out, offset + q, offset + q + 1, total)
    return out


def forward_state(x: np.ndarray, theta: np.ndarray, spec: CircuitSpec) -> np.ndarray:
    return apply_variational(encode(x), theta, spec)


def derivative_state(x: np.ndarray, theta: np.ndarray, spec: CircuitSpec, param_index: int) -> np.ndarray:
    n = spec.n_qubits
    params = theta.reshape(spec.layers, n)
    state = encode(x)
    for layer in range(spec.layers):
        for q in range(n):
            gate = dry(float(params[layer, q])) if layer * n + q == param_index else ry(float(params[layer, q]))
            state = apply_one_qubit(state, gate, q, n)
        for q in range(n - 1):
            state = apply_cnot(state, q, q + 1, n)
    return state

