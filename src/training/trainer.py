from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from quantum.ansatz import CircuitSpec
from quantum.gradients import batch_loss_and_gradient, predict_one
from quantum.observables import Observable
from quantum.resource_counter import analytic_shots
from training.evaluation import classification_metrics
from training.optimizer import make_optimizer


@dataclass
class TrainingResult:
    theta: np.ndarray
    history: pd.DataFrame
    validation_scores: np.ndarray


def predict_many(x: np.ndarray, theta: np.ndarray, spec: CircuitSpec, observable: Observable) -> np.ndarray:
    return np.asarray([predict_one(row, theta, spec, observable) for row in x])


def train_vqc(
    x_train: np.ndarray,
    y_train_pm: np.ndarray,
    y_train_label: np.ndarray,
    x_val: np.ndarray,
    y_val_pm: np.ndarray,
    y_val_label: np.ndarray,
    theta0: np.ndarray,
    spec: CircuitSpec,
    observable: Observable,
    method: str,
    config: dict,
    seed: int,
    qndm_lambda: float,
    shots: int | None,
) -> TrainingResult:
    rng = np.random.default_rng(seed)
    theta = theta0.copy()
    opt = make_optimizer(config["training"]["optimizer"], float(config["training"]["learning_rate"]))
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    rows = []
    cumulative_shots = 0
    cumulative_gates = 0
    epoch_bar = tqdm(range(epochs), desc=f"{spec.n_qubits}q {method} seed {seed} epochs", leave=True)
    for epoch in epoch_bar:
        start = time.perf_counter()
        order = rng.permutation(len(x_train))
        grad_norms = []
        update_norms = []
        losses = []
        batch_ranges = range(0, len(order), batch_size)
        for start_idx in tqdm(
            batch_ranges,
            desc=f"{spec.n_qubits}q {method} epoch {epoch + 1}/{epochs} batches",
            leave=False,
        ):
            batch = order[start_idx : start_idx + batch_size]
            loss, grad, _ = batch_loss_and_gradient(
                x_train[batch],
                y_train_pm[batch],
                theta,
                spec,
                observable,
                method=method,
                qndm_lambda=qndm_lambda,
                shots=shots if method in {"parameter_shift", "qndm"} else None,
                rng=rng,
            )
            theta, update = opt.step(theta, grad)
            losses.append(loss)
            grad_norms.append(float(np.linalg.norm(grad)))
            update_norms.append(float(np.linalg.norm(update)))
            if method == "parameter_shift":
                cumulative_shots += len(batch) * 2 * observable.g_m * spec.n_parameters * int(shots or 0)
            elif method == "qndm":
                cumulative_shots += len(batch) * spec.n_parameters * int(shots or 0) * 2 * 2
            cumulative_gates += len(batch) * spec.k_methodology * spec.n_parameters
        train_scores = predict_many(x_train, theta, spec, observable)
        val_scores = predict_many(x_val, theta, spec, observable)
        train_metrics = classification_metrics(y_train_label, train_scores)
        val_metrics = classification_metrics(y_val_label, val_scores)
        row = {
            "epoch": epoch + 1,
            "method": method,
            "training_loss": float(np.mean(losses)),
            "validation_loss": float(0.5 * np.mean((val_scores - y_val_pm) ** 2)),
            "training_accuracy": train_metrics["accuracy"],
            "validation_accuracy": val_metrics["accuracy"],
            "training_f1": train_metrics["f1"],
            "validation_f1": val_metrics["f1"],
            "gradient_norm": float(np.mean(grad_norms)),
            "parameter_update_norm": float(np.mean(update_norms)),
            "runtime": float(time.perf_counter() - start),
            "cumulative_shots": cumulative_shots,
            "cumulative_circuit_executions": cumulative_shots,
            "cumulative_logical_gates": cumulative_gates,
            "lambda": qndm_lambda if method == "qndm" else np.nan,
            "shots": shots if shots is not None else 0,
            "execution_success": True,
        }
        rows.append(row)
        epoch_bar.set_postfix(
            train_loss=f"{row['training_loss']:.4g}",
            val_loss=f"{row['validation_loss']:.4g}",
            val_f1=f"{row['validation_f1']:.3f}",
        )
        print(
            f"[TRAIN] {spec.n_qubits}q method={method} seed={seed} epoch={epoch + 1}/{epochs} "
            f"train_loss={row['training_loss']:.6g} val_loss={row['validation_loss']:.6g} "
            f"val_acc={row['validation_accuracy']:.4f} val_f1={row['validation_f1']:.4f} "
            f"runtime={row['runtime']:.2f}s cumulative_shots={cumulative_shots}",
            flush=True,
        )
    return TrainingResult(theta=theta, history=pd.DataFrame(rows), validation_scores=val_scores)
