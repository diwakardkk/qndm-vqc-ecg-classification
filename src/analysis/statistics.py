from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def gradient_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=float).ravel()
    est = np.asarray(estimate, dtype=float).ravel()
    diff = est - ref
    ref_norm = np.linalg.norm(ref)
    est_norm = np.linalg.norm(est)
    denom = ref_norm * est_norm
    out = {
        "bias": float(np.mean(diff)),
        "absolute_bias": float(abs(np.mean(diff))),
        "mse": float(np.mean(diff**2)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "l1_error": float(np.sum(np.abs(diff))),
        "l2_error": float(np.linalg.norm(diff)),
        "relative_l2_error": float(np.linalg.norm(diff) / ref_norm) if ref_norm else float("nan"),
        "cosine_similarity": float(np.dot(ref, est) / denom) if denom else float("nan"),
        "sign_agreement": float(np.mean(np.sign(ref) == np.sign(est))),
    }
    if len(ref) > 1 and np.std(ref) > 0 and np.std(est) > 0:
        out["pearson"] = float(pearsonr(ref, est).statistic)
        out["spearman"] = float(spearmanr(ref, est).statistic)
    else:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
    return out


def summarize_by(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in value_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_sd"] = group[col].std(ddof=1) if len(group) > 1 else 0.0
            row[f"{col}_ci95_low"] = group[col].mean() - 1.96 * group[col].sem() if len(group) > 1 else group[col].mean()
            row[f"{col}_ci95_high"] = group[col].mean() + 1.96 * group[col].sem() if len(group) > 1 else group[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)

