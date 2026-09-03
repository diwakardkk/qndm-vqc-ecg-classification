from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.statistics import gradient_metrics
from quantum.ansatz import CircuitSpec
from quantum.gradients import analytic_sample_gradient, parameter_shift_sample_gradient
from quantum.observables import Observable
from quantum.qndm import qndm_sample_gradient


def run_gradient_validation(
    x: np.ndarray,
    theta: np.ndarray,
    spec: CircuitSpec,
    observable: Observable,
    lam: float,
    shift: float,
    max_samples: int = 8,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows = []
    for sample_idx, row in enumerate(tqdm(x[:max_samples], desc=f"{spec.n_qubits}q gradient validation", leave=False)):
        exact = analytic_sample_gradient(row, theta, spec, observable)
        ps = parameter_shift_sample_gradient(row, theta, spec, observable, shift=shift)
        qndm = qndm_sample_gradient(row, theta, spec, observable, lam=lam, shift=shift)
        for method, estimate in [("parameter_shift", ps), ("qndm_finite_lambda", qndm)]:
            metrics = gradient_metrics(exact, estimate)
            rows.append({"sample_index": sample_idx, "method": method, "lambda": lam if "qndm" in method else np.nan, **metrics})
    frame = pd.DataFrame(rows)
    ps_ok = frame.loc[frame["method"] == "parameter_shift", "mse"].max() < 1e-10
    qndm_ok = frame.loc[frame["method"] == "qndm_finite_lambda", "mse"].max() < 1e-5
    return frame, {"Parameter-shift identity": bool(ps_ok), "QNDM small-lambda validation": bool(qndm_ok)}
