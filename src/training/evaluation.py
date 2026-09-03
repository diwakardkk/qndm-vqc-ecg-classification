from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.0) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        "npv": npv,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    try:
        out["auroc"] = roc_auc_score(y_true, scores)
    except ValueError:
        out["auroc"] = float("nan")
    try:
        out["average_precision"] = average_precision_score(y_true, scores)
    except ValueError:
        out["average_precision"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def patient_bootstrap_ci(
    metadata: pd.DataFrame,
    scores: np.ndarray,
    metric: str,
    threshold: float,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if replicates <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    records = metadata["record"].unique()
    vals = []
    for _ in range(replicates):
        sampled = rng.choice(records, size=len(records), replace=True)
        mask = np.concatenate([metadata.index[metadata["record"] == record].to_numpy() for record in sampled])
        y = metadata.loc[mask, "label"].to_numpy()
        s = scores[mask]
        vals.append(classification_metrics(y, s, threshold).get(metric, np.nan))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

