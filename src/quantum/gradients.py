from __future__ import annotations

import numpy as np

from .ansatz import CircuitSpec, derivative_state, forward_state
from .observables import Observable, expectation, sample_expectation


def predict_one(x: np.ndarray, theta: np.ndarray, spec: CircuitSpec, observable: Observable) -> float:
    return expectation(forward_state(x, theta, spec), observable)


def analytic_sample_gradient(x: np.ndarray, theta: np.ndarray, spec: CircuitSpec, observable: Observable) -> np.ndarray:
    psi = forward_state(x, theta, spec)
    grads = []
    for j in range(spec.n_parameters):
        dpsi = derivative_state(x, theta, spec, j)
        total = 0.0 + 0.0j
        for term in observable.terms:
            from .observables import apply_pauli

            total += term.coeff * 2.0 * np.vdot(dpsi, apply_pauli(psi, term.ops))
        grads.append(float(np.real_if_close(total)))
    return np.asarray(grads)


def parameter_shift_sample_gradient(
    x: np.ndarray,
    theta: np.ndarray,
    spec: CircuitSpec,
    observable: Observable,
    shift: float = np.pi / 2,
    shots: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    grads = []
    denom = 2.0 * np.sin(shift)
    for j in range(spec.n_parameters):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += shift
        minus[j] -= shift
        fp = predict_one(x, plus, spec, observable)
        fm = predict_one(x, minus, spec, observable)
        if shots is not None:
            if rng is None:
                raise ValueError("Finite-shot parameter shift requires an RNG.")
            fp = sample_expectation(fp, shots, rng)
            fm = sample_expectation(fm, shots, rng)
        grads.append((fp - fm) / denom)
    return np.asarray(grads)


def batch_loss_and_gradient(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    spec: CircuitSpec,
    observable: Observable,
    method: str,
    qndm_lambda: float = 0.01,
    shift: float = np.pi / 2,
    shots: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    preds = np.asarray([predict_one(row, theta, spec, observable) for row in x])
    loss = float(0.5 * np.mean((preds - y) ** 2))
    grad = np.zeros_like(theta)
    for row, target, pred in zip(x, y, preds):
        if method == "analytic":
            g = analytic_sample_gradient(row, theta, spec, observable)
        elif method == "parameter_shift":
            g = parameter_shift_sample_gradient(row, theta, spec, observable, shift=shift, shots=shots, rng=rng)
        elif method == "qndm":
            from .qndm import qndm_sample_gradient

            g = qndm_sample_gradient(row, theta, spec, observable, lam=qndm_lambda, shift=shift, shots=shots, rng=rng)
        else:
            raise ValueError(f"Unknown gradient method {method}")
        grad += (pred - target) * g / len(x)
    return loss, grad, preds

