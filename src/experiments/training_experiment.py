from __future__ import annotations

import numpy as np
import pandas as pd

from quantum.ansatz import CircuitSpec
from quantum.observables import Observable
from training.evaluation import classification_metrics
from training.trainer import predict_many, train_vqc


def run_training_methods(
    x_train: np.ndarray,
    meta_train: pd.DataFrame,
    x_val: np.ndarray,
    meta_val: pd.DataFrame,
    x_test: np.ndarray,
    meta_test: pd.DataFrame,
    spec: CircuitSpec,
    observable: Observable,
    config: dict,
    seed: int,
    qndm_lambda: float,
    shots: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(0.0, 0.05, size=spec.n_parameters)
    histories = []
    metrics = []
    predictions = []
    models = {}
    for method in config["training"]["methods"]:
        result = train_vqc(
            x_train,
            meta_train["target"].to_numpy(),
            meta_train["label"].to_numpy(),
            x_val,
            meta_val["target"].to_numpy(),
            meta_val["label"].to_numpy(),
            theta0,
            spec,
            observable,
            method,
            config,
            seed,
            qndm_lambda=qndm_lambda,
            shots=shots,
        )
        histories.append(result.history.assign(seed=seed))
        test_scores = predict_many(x_test, result.theta, spec, observable)
        m = classification_metrics(meta_test["label"].to_numpy(), test_scores, float(config["training"].get("threshold", 0.0)))
        metrics.append({"method": method, "seed": seed, "qubits": spec.n_qubits, "layers": spec.layers, "lambda": qndm_lambda if method == "qndm" else np.nan, "shots": shots if method in {"qndm", "parameter_shift"} else 0, **m})
        pred = meta_test[["record", "beat_sample", "label"]].copy()
        pred["true_label"] = pred.pop("label")
        pred["raw_quantum_score"] = test_scores
        pred["predicted_label"] = (test_scores >= float(config["training"].get("threshold", 0.0))).astype(int)
        pred["method"] = method
        pred["qubits"] = spec.n_qubits
        pred["seed"] = seed
        predictions.append(pred)
        models[method] = result.theta
    return pd.concat(histories, ignore_index=True), pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), models

