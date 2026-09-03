from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.io import ensure_dir, write_csv


def balanced_indices(metadata: pd.DataFrame, split: str, per_class: int | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idxs = []
    for label in [0, 1]:
        pool = metadata.index[(metadata["split"] == split) & (metadata["label"] == label)].to_numpy()
        if per_class is None:
            idxs.append(pool)
        else:
            take = min(int(per_class), len(pool))
            idxs.append(rng.choice(pool, size=take, replace=False) if take else np.array([], dtype=int))
    if not idxs:
        return np.array([], dtype=int)
    out = np.concatenate(idxs)
    rng.shuffle(out)
    return out


def fit_pca_features(
    x_windows: np.ndarray,
    metadata: pd.DataFrame,
    n_components: int,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    train_idx = metadata.index[metadata["split"] == "train"].to_numpy()
    if len(train_idx) < n_components:
        raise ValueError(f"PCA dimension {n_components} exceeds number of training beats {len(train_idx)}.")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_windows[train_idx])
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(x_train_scaled)
    all_scaled = scaler.transform(x_windows)
    z = pca.transform(all_scaled)
    train_z = z[train_idx]
    q = float(config["features"].get("angle_clip_quantile", 0.99))
    denom = np.quantile(np.abs(train_z), q, axis=0)
    denom[denom < 1e-8] = 1.0
    angle_range = float(config["features"].get("angle_range", np.pi))
    angles = np.clip(z / denom, -1.0, 1.0) * angle_range
    np.save(out / f"features_{n_components}q.npy", angles)
    joblib.dump({"scaler": scaler, "pca": pca, "angle_scale": denom}, out / f"pca_scaler_{n_components}q.joblib")
    ev = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    write_csv(out / f"pca_explained_variance_{n_components}q.csv", ev)
    write_csv(out / f"pca_components_{n_components}q.csv", pd.DataFrame(pca.components_))
    write_csv(
        out / f"feature_distribution_{n_components}q.csv",
        pd.DataFrame(angles, columns=[f"x{i}" for i in range(n_components)]).assign(split=metadata["split"].to_numpy()),
    )
    return {"features": angles, "scaler": scaler, "pca": pca, "angle_scale": denom, "explained": ev}

