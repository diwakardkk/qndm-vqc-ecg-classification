from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from utils.io import ensure_dir


def save_figure(fig: plt.Figure, base_path: str | Path) -> None:
    base = Path(base_path)
    ensure_dir(base.parent)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(metadata: pd.DataFrame, out: str | Path) -> None:
    counts = metadata.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "N", 1: "V"})
    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_xlabel("Split")
    ax.set_ylabel("Heartbeat count")
    ax.legend(title="Class")
    save_figure(fig, out)


def plot_raw_filtered(example: dict, out: str | Path) -> None:
    fs = example["fs"]
    t = np.arange(len(example["raw"])) / fs
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    ax.plot(t, example["raw"], lw=0.8, label="Raw")
    ax.plot(t, example["filtered"], lw=0.8, label="Filtered")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECG (mV)")
    ax.legend()
    save_figure(fig, out)


def plot_representative_beats(x: np.ndarray, metadata: pd.DataFrame, label: int, out: str | Path) -> None:
    idx = metadata.index[metadata["label"] == label].to_numpy()[:12]
    if len(idx) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    for i in idx:
        ax.plot(x[i], lw=0.7, alpha=0.7)
    ax.set_xlabel("Sample within heartbeat window")
    ax.set_ylabel("Filtered ECG")
    save_figure(fig, out)


def plot_pca(ev: pd.DataFrame, out: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    ax.bar(ev["component"], ev["explained_variance_ratio"], color="#54A24B", label="Per component")
    ax.plot(ev["component"], ev["cumulative_variance"], marker="o", color="#B279A2", label="Cumulative")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0, 1.05)
    ax.legend()
    save_figure(fig, out)


def plot_metric_line(frame: pd.DataFrame, x: str, y: str, out: str | Path, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if frame.empty or x not in frame or y not in frame:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    ax.plot(frame[x], frame[y], marker="o", color="#4C78A8")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if frame[x].min() > 0 and frame[x].max() / frame[x].min() > 20:
        ax.set_xscale("log")
    save_figure(fig, out)


def plot_training(history: pd.DataFrame, out: str | Path, metric: str = "validation_loss") -> None:
    if history.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.3), constrained_layout=True)
    for method, group in history.groupby("method"):
        ax.plot(group["epoch"], group[metric], marker="o", label=method)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend()
    save_figure(fig, out)


def plot_confusion(tn: int, fp: int, fn: int, tp: int, out: str | Path) -> None:
    mat = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(3.2, 3.2), constrained_layout=True)
    ConfusionMatrixDisplay(mat, display_labels=["N", "V"]).plot(ax=ax, colorbar=False, cmap="Blues")
    save_figure(fig, out)
