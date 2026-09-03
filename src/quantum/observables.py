from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PauliTerm:
    coeff: float
    ops: str


@dataclass(frozen=True)
class Observable:
    name: str
    terms: tuple[PauliTerm, ...]
    groups: tuple[tuple[int, ...], ...]

    @property
    def j_m(self) -> int:
        return len(self.terms)

    @property
    def g_m(self) -> int:
        return len(self.groups)


def observable_for_regime(regime: str, n_qubits: int) -> Observable:
    regime = regime.upper()
    if regime == "A":
        return Observable("A", (PauliTerm(1.0, "Z" + "I" * (n_qubits - 1)),), ((0,),))
    if regime == "B":
        raw = [PauliTerm(1.0, "I" * i + "Z" + "I" * (n_qubits - i - 1)) for i in range(n_qubits)]
        raw += [
            PauliTerm(0.5, "I" * i + "ZZ" + "I" * (n_qubits - i - 2))
            for i in range(max(0, n_qubits - 1))
        ]
        norm = sum(abs(t.coeff) for t in raw)
        return Observable("B", tuple(PauliTerm(t.coeff / norm, t.ops) for t in raw), (tuple(range(len(raw))),))
    if regime == "C":
        ops = ["X" + "I" * (n_qubits - 1), "Y" + "I" * (n_qubits - 1), "Z" + "I" * (n_qubits - 1)]
        if n_qubits > 1:
            ops += ["XX" + "I" * (n_qubits - 2), "ZZ" + "I" * (n_qubits - 2)]
        coeff = 1.0 / len(ops)
        terms = tuple(PauliTerm(coeff, op) for op in ops)
        return Observable("C", terms, tuple((i,) for i in range(len(terms))))
    raise ValueError(f"Unknown readout regime {regime!r}")


def pauli_commutes(a: str, b: str) -> bool:
    anti = 0
    for x, y in zip(a, b):
        if x != "I" and y != "I" and x != y:
            anti += 1
    return anti % 2 == 0


def apply_pauli(state: np.ndarray, ops: str, offset: int = 0, total_wires: int | None = None) -> np.ndarray:
    vec = np.asarray(state, dtype=complex)
    n = int(np.log2(vec.size)) if total_wires is None else total_wires
    tensor = vec.reshape([2] * n)
    out = tensor.copy()
    for q, op in enumerate(ops):
        axis = offset + q
        if op == "I":
            continue
        out = np.moveaxis(out, axis, 0)
        if op == "X":
            out = out[[1, 0], ...]
        elif op == "Y":
            out = np.stack([-1j * out[1, ...], 1j * out[0, ...]], axis=0)
        elif op == "Z":
            out = np.stack([out[0, ...], -out[1, ...]], axis=0)
        else:
            raise ValueError(f"Unsupported Pauli operator {op!r}")
        out = np.moveaxis(out, 0, axis)
    return out.reshape(-1)


def expectation(state: np.ndarray, observable: Observable) -> float:
    val = 0.0 + 0.0j
    for term in observable.terms:
        val += term.coeff * np.vdot(state, apply_pauli(state, term.ops))
    return float(np.real_if_close(val))


def sample_expectation(true_value: float, shots: int, rng: np.random.Generator) -> float:
    p_plus = float(np.clip((1.0 + true_value) / 2.0, 0.0, 1.0))
    draws = rng.binomial(shots, p_plus)
    return 2.0 * draws / shots - 1.0

