from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.statistics import gradient_metrics, summarize_by
from quantum.ansatz import CircuitSpec
from quantum.gradients import analytic_sample_gradient, parameter_shift_sample_gradient
from quantum.observables import Observable
from quantum.qndm import qndm_sample_gradient


def run_shot_sweep(
    x: np.ndarray,
    theta: np.ndarray,
    spec: CircuitSpec,
    observable: Observable,
    shots: list[int],
    repeats: int,
    lam: float,
    shift: float,
    seed: int,
    max_samples: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    total = len(shots) * repeats * min(max_samples, len(x))
    with tqdm(total=total, desc=f"{spec.n_qubits}q shot sweep", leave=False) as progress:
        for shot in shots:
            for repeat in range(repeats):
                rng = np.random.default_rng(seed + 1000 * repeat + shot)
                for sample_idx, row in enumerate(x[:max_samples]):
                    exact = analytic_sample_gradient(row, theta, spec, observable)
                    ps = parameter_shift_sample_gradient(row, theta, spec, observable, shift=shift, shots=shot, rng=rng)
                    qn = qndm_sample_gradient(row, theta, spec, observable, lam=lam, shift=shift, shots=shot, rng=rng)
                    rows.append({"shots": shot, "repeat": repeat, "sample_index": sample_idx, "method": "finite_shot_parameter_shift", **gradient_metrics(exact, ps)})
                    rows.append({"shots": shot, "repeat": repeat, "sample_index": sample_idx, "method": "finite_shot_qndm", "lambda": lam, **gradient_metrics(exact, qn)})
                    progress.update(1)
    raw = pd.DataFrame(rows)
    summary = summarize_by(raw, ["shots", "method"], ["mse", "mae", "bias", "cosine_similarity"])
    return raw, summary
