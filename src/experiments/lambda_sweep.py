from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.statistics import gradient_metrics, summarize_by
from quantum.ansatz import CircuitSpec
from quantum.gradients import analytic_sample_gradient
from quantum.observables import Observable
from quantum.qndm import qndm_sample_gradient


def run_lambda_sweep(
    x: np.ndarray,
    theta: np.ndarray,
    spec: CircuitSpec,
    observable: Observable,
    lambdas: list[float],
    shift: float,
    max_samples: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    total = len(lambdas) * min(max_samples, len(x))
    with tqdm(total=total, desc=f"{spec.n_qubits}q lambda sweep", leave=False) as progress:
        for lam in lambdas:
            for sample_idx, row in enumerate(x[:max_samples]):
                exact = analytic_sample_gradient(row, theta, spec, observable)
                est = qndm_sample_gradient(row, theta, spec, observable, lam=lam, shift=shift)
                rows.append({"lambda": lam, "sample_index": sample_idx, "method": "qndm_finite_lambda_noiseless", **gradient_metrics(exact, est)})
                progress.update(1)
    raw = pd.DataFrame(rows)
    summary = summarize_by(raw, ["lambda", "method"], ["mse", "mae", "bias", "cosine_similarity", "relative_l2_error"])
    return raw, summary
