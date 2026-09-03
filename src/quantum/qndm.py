from __future__ import annotations

import numpy as np

from .ansatz import CircuitSpec, apply_one_qubit, apply_variational_full, encode, zero_state
from .observables import Observable, apply_pauli, sample_expectation


HADAMARD = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)


def detector_initial_state(x: np.ndarray) -> np.ndarray:
    system = encode(x)
    detector = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    return np.kron(detector, system)


def apply_detector_interaction(state: np.ndarray, observable: Observable, lam: float, n_system: int, sign: int) -> np.ndarray:
    total = n_system + 1
    out = state
    for term in observable.terms:
        ops = "Z" + term.ops
        # The detector coherence compares the two Z_a eigenvalue branches, so
        # its phase response is twice the Hamiltonian angle. This half-angle
        # convention makes the public lambda satisfy the identity used in the
        # experiment protocol: -i dG/dlambda = 2 sin(s) g_j.
        angle = sign * lam * term.coeff / 2.0
        p_state = apply_pauli(out, ops, offset=0, total_wires=total)
        out = np.cos(angle) * out + 1j * np.sin(angle) * p_state
    return out


def qndm_final_state(
    x: np.ndarray,
    theta: np.ndarray,
    spec: CircuitSpec,
    observable: Observable,
    param_index: int,
    lam: float,
    shift: float = np.pi / 2,
) -> np.ndarray:
    plus = theta.copy()
    minus = theta.copy()
    plus[param_index] += shift
    minus[param_index] -= shift
    state = detector_initial_state(x)
    state = apply_variational_full(state, minus, spec, offset=1, dagger=False)
    state = apply_detector_interaction(state, observable, lam, spec.n_qubits, sign=-1)
    state = apply_variational_full(state, minus, spec, offset=1, dagger=True)
    state = apply_variational_full(state, plus, spec, offset=1, dagger=False)
    state = apply_detector_interaction(state, observable, lam, spec.n_qubits, sign=+1)
    return state


def detector_coherence_from_state(state: np.ndarray, n_system: int) -> complex:
    reshaped = state.reshape(2, 2**n_system)
    return complex(np.sum(reshaped[0, :] * np.conjugate(reshaped[1, :])))


def detector_xy_from_state(state: np.ndarray, n_system: int) -> tuple[float, float]:
    rho01 = detector_coherence_from_state(state, n_system)
    x_exp = 2.0 * rho01.real
    y_exp = -2.0 * rho01.imag
    return float(x_exp), float(y_exp)


def coherence_from_xy(x_exp: float, y_exp: float) -> complex:
    return (x_exp - 1j * y_exp) / 2.0


def qndm_g(
    x: np.ndarray,
    theta: np.ndarray,
    spec: CircuitSpec,
    observable: Observable,
    param_index: int,
    lam: float,
    shift: float = np.pi / 2,
    shots: int | None = None,
    rng: np.random.Generator | None = None,
) -> complex:
    state = qndm_final_state(x, theta, spec, observable, param_index, lam, shift=shift)
    x_exp, y_exp = detector_xy_from_state(state, spec.n_qubits)
    if shots is not None:
        if rng is None:
            raise ValueError("Finite-shot QNDM requires an RNG.")
        x_exp = sample_expectation(x_exp, shots, rng)
        y_exp = sample_expectation(y_exp, shots, rng)
    rho01 = coherence_from_xy(x_exp, y_exp)
    return rho01 / 0.5


def qndm_sample_gradient(
    x: np.ndarray,
    theta: np.ndarray,
    spec: CircuitSpec,
    observable: Observable,
    lam: float,
    shift: float = np.pi / 2,
    shots: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    grads = []
    for j in range(spec.n_parameters):
        gp = qndm_g(x, theta, spec, observable, j, abs(lam), shift=shift, shots=shots, rng=rng)
        gm = qndm_g(x, theta, spec, observable, j, -abs(lam), shift=shift, shots=shots, rng=rng)
        deriv = (gp - gm) / (2.0 * abs(lam))
        value = (-1j * deriv) / (2.0 * np.sin(shift))
        grads.append(float(np.real_if_close(value)))
    return np.asarray(grads)


def qndm_uses_detector_qubit(x: np.ndarray, theta: np.ndarray, spec: CircuitSpec, observable: Observable) -> bool:
    state = qndm_final_state(x, theta, spec, observable, 0, 0.01)
    return state.size == 2 ** (spec.n_qubits + 1)
