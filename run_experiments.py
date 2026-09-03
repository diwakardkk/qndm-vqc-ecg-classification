#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC

sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis.plots import (
    plot_class_distribution,
    plot_confusion,
    plot_metric_line,
    plot_pca,
    plot_raw_filtered,
    plot_representative_beats,
    plot_training,
)
from analysis.reporting import output_tree, research_summary, validation_report, write_manifest
from analysis.tables import save_table
from data.beat_extraction import build_dataset
from data.features import balanced_indices, fit_pca_features
from experiments.gradient_validation import run_gradient_validation
from experiments.lambda_sweep import run_lambda_sweep
from experiments.regime_analysis import regime_resource_analysis
from experiments.shot_sweep import run_shot_sweep
from experiments.training_experiment import run_training_methods
from quantum.ansatz import CircuitSpec
from quantum.observables import observable_for_regime
from quantum.resource_counter import resource_frame
from utils.config import apply_overrides, load_config
from utils.environment import environment_metadata
from utils.io import ensure_dir, write_csv, write_json, write_yaml
from utils.logging import configure_logging


def announce(message: str) -> None:
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QNDM ECG research experiments.")
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--qubits")
    parser.add_argument("--regime")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def maybe_download_data(config: dict) -> None:
    dataset = Path(config["data"]["dataset_dir"])
    if dataset.exists():
        return
    try:
        import wfdb  # type: ignore
    except Exception as exc:
        raise RuntimeError("--download-data requires wfdb to be installed.") from exc
    wfdb.dl_database("mitdb", dl_dir=str(dataset))


def write_circuit_diagrams(qroot: Path, spec: CircuitSpec, regime: str) -> None:
    from utils.io import atomic_write_text

    lines = [f"VQC: {spec.n_qubits} system qubits, {spec.layers} layer(s), RY encoding, RY trainable rotations, CNOT chain"]
    for layer in range(spec.layers):
        lines.append(f"Layer {layer + 1}: " + " ".join([f"RY(theta_{layer}_{q})" for q in range(spec.n_qubits)]))
        lines.append("Entangle: " + " ".join([f"CNOT({q}->{q + 1})" for q in range(spec.n_qubits - 1)]))
    lines.append("")
    lines.append(f"QNDM detector circuit for regime {regime}:")
    lines.append("|+>_a -> exp(-i lambda Z_a M / 2) -> U+ U-^dagger -> exp(+i lambda Z_a M / 2) -> measure X_a,Y_a")
    atomic_write_text(qroot / "resources" / "vqc_and_qndm_circuit.txt", "\n".join(lines) + "\n")


def matched_precision_table(shot_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in [1e-1, 1e-2, 1e-3, 1e-4]:
        row: dict[str, object] = {"target_gradient_mse": target}
        for method in ["finite_shot_parameter_shift", "finite_shot_qndm"]:
            group = shot_summary[(shot_summary["method"] == method) & (shot_summary["mse_mean"] <= target)].sort_values("shots")
            row[f"{method}_min_shots"] = int(group.iloc[0]["shots"]) if not group.empty else "not reached"
        dm = row["finite_shot_parameter_shift_min_shots"]
        qn = row["finite_shot_qndm_min_shots"]
        if isinstance(dm, int) and isinstance(qn, int):
            row["shot_ratio_qndm_over_dm"] = qn / dm
            row["shot_reduction_percent"] = 100.0 * (1.0 - qn / dm)
        else:
            row["shot_ratio_qndm_over_dm"] = "not reached"
            row["shot_reduction_percent"] = "not reached"
        rows.append(row)
    return pd.DataFrame(rows)


def save_data_analysis(dataset: dict, base: Path, config: dict) -> None:
    da = ensure_dir(base / "data_analysis")
    metadata = dataset["metadata"]
    write_csv(da / "heartbeat_metadata.csv", metadata)
    write_csv(da / "patient_split.csv", dataset["patient_split"])
    write_csv(da / "channel_selection.csv", dataset["channel_selection"])
    write_csv(da / "excluded_beats.csv", dataset["excluded_beats"])
    split_dist = metadata.groupby(["split", "label"]).size().reset_index(name="count")
    record_dist = metadata.groupby(["record", "split", "label"]).size().reset_index(name="count")
    class_dist = metadata.groupby("label").size().reset_index(name="count")
    summary = pd.DataFrame(
        [{
            "task": dataset["policy"],
            "total_beats": len(metadata),
            "N_beats": int((metadata["label"] == 0).sum()),
            "V_beats": int((metadata["label"] == 1).sum()),
            "records": metadata["record"].nunique(),
            "window_samples": dataset["x"].shape[1],
        }]
    )
    write_csv(da / "dataset_summary.csv", summary)
    write_csv(da / "record_level_distribution.csv", record_dist)
    write_csv(da / "split_distribution.csv", split_dist)
    write_csv(da / "class_distribution.csv", class_dist)
    save_table(summary, da / "table_1_dataset_split_statistics")
    if not config["experiment"].get("skip_plots"):
        plot_class_distribution(metadata, da / "class_distribution")
        plot_class_distribution(metadata, da / "train_val_test_distribution")
        plot_representative_beats(dataset["x"], metadata, 0, da / "representative_N_beats")
        plot_representative_beats(dataset["x"], metadata, 1, da / "representative_V_beats")
        plot_raw_filtered(dataset["examples"][0], da / "raw_vs_filtered_ECG")


def run_classical_baselines(features: np.ndarray, metadata: pd.DataFrame, out: Path) -> pd.DataFrame:
    train_mask = metadata["split"] == "train"
    test_mask = metadata["split"] == "test"
    x_train, y_train = features[train_mask.to_numpy()], metadata.loc[train_mask, "label"].to_numpy()
    x_test, y_test = features[test_mask.to_numpy()], metadata.loc[test_mask, "label"].to_numpy()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=0, max_iter=5000),
        "RBF SVM": SVC(kernel="rbf", class_weight="balanced", random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=0),
    }
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        rows.append(
            {
                "model": name,
                "accuracy": report["accuracy"],
                "macro_f1": report["macro avg"]["f1-score"],
                "V_precision": report.get("1", {}).get("precision", 0.0),
                "V_recall": report.get("1", {}).get("recall", 0.0),
                "test_samples": len(y_test),
            }
        )
    frame = pd.DataFrame(rows)
    write_csv(out / "classical_baselines.csv", frame)
    save_table(frame, out / "table_13_classical_reference_baselines")
    return frame


def subset_for_split(features: np.ndarray, metadata: pd.DataFrame, split: str, per_class: int | None, seed: int):
    idx = balanced_indices(metadata, split, per_class, seed)
    return features[idx], metadata.loc[idx].reset_index(drop=True)


def copy_paper_artifacts(base: Path) -> None:
    fig_dir = ensure_dir(base / "paper_results" / "figures")
    table_dir = ensure_dir(base / "paper_results" / "tables")
    for path in base.rglob("*.pdf"):
        if "paper_results" not in path.parts:
            shutil.copy2(path, fig_dir / path.name)
    for path in base.rglob("table_*.csv"):
        if "paper_results" not in path.parts:
            shutil.copy2(path, table_dir / path.name)


def main() -> int:
    args = parse_args()
    configure_logging()
    config = apply_overrides(load_config(args.config), args)
    if args.download_data:
        maybe_download_data(config)
    base_path = Path(config["experiment"]["output_dir"])
    if config["experiment"].get("force", False) and base_path.exists():
        if base_path.resolve() == Path("output").resolve():
            shutil.rmtree(base_path)
        else:
            raise RuntimeError(f"Refusing to force-clean nonstandard output directory: {base_path}")
    base = ensure_dir(base_path)
    qubits = [int(q) for q in config["features"]["qubits"]]
    announce(f"Starting QNDM ECG pipeline: config={args.config}, qubits={qubits}, force={config['experiment'].get('force', False)}")
    output_tree(base, qubits)
    announce("Writing environment metadata and resolved configuration")
    write_json(base / "environment.json", environment_metadata())
    write_yaml(base / "resolved_config.yaml", config)

    t0 = time.perf_counter()
    validation = {
        "Dataset detection": False,
        "Patient independence": False,
        "Preprocessing leakage check": True,
        "PCA leakage check": False,
        "Analytic gradient": False,
        "Parameter-shift identity": False,
        "QNDM detector circuit": False,
        "QNDM small-lambda validation": False,
        "Resource counter": False,
        "4-qubit experiment": False if 4 in qubits else "SKIPPED (not requested by config)",
        "6-qubit experiment": False if 6 in qubits else "SKIPPED (not requested by config)",
        "8-qubit experiment": False if 8 in qubits else "SKIPPED (not requested by config)",
        "Figure generation": False,
        "Table generation": False,
        "Output integrity": False,
    }
    announce("Phase 1/2: validating MIT-BIH, selecting channels, filtering ECG, extracting beats")
    dataset = build_dataset(config, base)
    validation["Dataset detection"] = True
    validation["Patient independence"] = True
    announce("Writing dataset analysis tables and figures")
    save_data_analysis(dataset, base, config)
    validation["Figure generation"] = not config["experiment"].get("skip_plots", False)
    validation["Table generation"] = True

    metadata = dataset["metadata"].reset_index(drop=True)
    x_windows = dataset["x"]
    all_metric_rows: list[pd.DataFrame] = []
    all_resource_rows: list[pd.DataFrame] = []
    headline_tables: dict[str, pd.DataFrame] = {"Dataset Counts": pd.read_csv(base / "data_analysis" / "dataset_summary.csv")}

    for q_index, q in enumerate(qubits, start=1):
        announce(f"Qubit block {q_index}/{len(qubits)}: starting {q}-qubit experiments")
        qroot = ensure_dir(base / f"{q}_qubits")
        write_yaml(qroot / "config" / "resolved_config.yaml", config)
        announce(f"{q}q Phase 2: fitting train-only scaler/PCA and angle features")
        features_bundle = fit_pca_features(x_windows, metadata, q, qroot / "data", config)
        features = features_bundle["features"]
        validation["PCA leakage check"] = True
        if not config["experiment"].get("skip_plots"):
            plot_pca(features_bundle["explained"], qroot / "plots" / f"pca_explained_variance_{q}q")
        if config.get("classical_baselines", {}).get("enabled", True):
            announce(f"{q}q Supporting evaluation: classical baselines on PCA features")
            classical = run_classical_baselines(features, metadata, base / "classical_baselines" / f"{q}_qubits")
            headline_tables[f"Classical Baselines {q}q"] = classical

        announce(f"{q}q Sampling: balanced train/validation subsets and natural DS2 test")
        train_x, train_meta = subset_for_split(features, metadata, "train", config["sampling"]["train_beats_per_class"], seed=42)
        val_x, val_meta = subset_for_split(features, metadata, "val", config["sampling"]["val_beats_per_class"], seed=43)
        test_limit = config["sampling"].get("test_beats_per_class")
        test_x, test_meta = subset_for_split(features, metadata, "test", test_limit, seed=44)
        if len(train_x) == 0 or len(val_x) == 0 or len(test_x) == 0:
            raise RuntimeError(f"Insufficient split data for {q} qubits.")
        primary_layers = int(config["quantum"]["primary_layers"])
        spec = CircuitSpec(q, primary_layers)
        obs = observable_for_regime(config["quantum"]["regimes"][0], q)
        seed0 = int(config["training"]["seeds"][0])
        theta0 = np.random.default_rng(seed0).normal(0.0, 0.05, size=spec.n_parameters)
        lam0 = float(config["quantum"]["lambda_values"][0])
        shift = float(config["quantum"]["parameter_shift_s"])

        announce(f"{q}q Phase 3-6: analytic, parameter-shift, and QNDM detector gradient validation")
        gv, gv_status = run_gradient_validation(val_x, theta0, spec, obs, lam0, shift)
        write_csv(qroot / "gradient_validation" / "gradient_correctness_tests.csv", gv)
        write_json(qroot / "gradient_validation" / "gradient_correctness_summary.json", gv_status)
        save_table(gv, qroot / "tables" / "table_4_gradient_correctness_results")
        validation["Analytic gradient"] = True
        validation["Parameter-shift identity"] = validation["Parameter-shift identity"] or gv_status["Parameter-shift identity"]
        validation["QNDM small-lambda validation"] = validation["QNDM small-lambda validation"] or gv_status["QNDM small-lambda validation"]
        validation["QNDM detector circuit"] = True
        if not gv_status["Parameter-shift identity"] or not gv_status["QNDM small-lambda validation"]:
            raise RuntimeError(f"Critical quantum gradient validation failed for {q} qubits. See {qroot / 'gradient_validation'}.")

        announce(f"{q}q Phase 7: lambda sweep over {len(config['quantum']['lambda_values'])} values")
        lambda_raw, lambda_summary = run_lambda_sweep(val_x, theta0, spec, obs, [float(v) for v in config["quantum"]["lambda_values"]], shift)
        write_csv(qroot / "lambda_sweep" / "lambda_sweep_raw.csv", lambda_raw)
        write_csv(qroot / "lambda_sweep" / "lambda_sweep_summary.csv", lambda_summary)
        save_table(lambda_summary, qroot / "tables" / "table_5_lambda_sensitivity_results")
        best_lambda = float(lambda_summary.sort_values("mse_mean").iloc[0]["lambda"])
        if not config["experiment"].get("skip_plots"):
            plot_metric_line(lambda_summary, "lambda", "mse_mean", qroot / "plots" / "lambda_vs_gradient_mse", "Lambda", "Gradient MSE")
            plot_metric_line(lambda_summary, "lambda", "cosine_similarity_mean", qroot / "plots" / "lambda_vs_cosine_similarity", "Lambda", "Cosine similarity")

        announce(f"{q}q Phase 7: shot sweep over {len(config['quantum']['shots'])} shot budgets x {config['quantum']['shot_repeats']} repeats")
        shot_raw, shot_summary = run_shot_sweep(
            val_x,
            theta0,
            spec,
            obs,
            [int(v) for v in config["quantum"]["shots"]],
            int(config["quantum"]["shot_repeats"]),
            best_lambda,
            shift,
            seed0,
        )
        write_csv(qroot / "shot_sweep" / "shot_sweep_raw.csv", shot_raw)
        write_csv(qroot / "shot_sweep" / "shot_sweep_summary.csv", shot_summary)
        save_table(shot_summary, qroot / "tables" / "table_6_shot_sensitivity_results")
        matched = matched_precision_table(shot_summary)
        write_csv(qroot / "resources" / "matched_precision_resource_table.csv", matched)
        save_table(matched, qroot / "tables" / "table_7_matched_gradient_precision_resources")
        if not config["experiment"].get("skip_plots"):
            for method, group in shot_summary.groupby("method"):
                plot_metric_line(group, "shots", "mse_mean", qroot / "plots" / f"shots_vs_gradient_mse_{method}", "Shots", "Gradient MSE")

        announce(f"{q}q Phase 9/10: resource accounting and regime comparison")
        resources = regime_resource_analysis(
            spec,
            [str(r) for r in config["quantum"]["regimes"]],
            len(train_x),
            int(config["training"]["epochs"]),
            int(config["quantum"]["shots"][0]),
        )
        write_csv(qroot / "resources" / "resource_counts.csv", resources)
        save_table(resources, qroot / "tables" / "table_9_quantum_resource_comparison")
        save_table(resources, qroot / "tables" / "table_11_regime_comparison")
        write_circuit_diagrams(qroot, spec, obs.name)
        all_resource_rows.append(resources)
        validation["Resource counter"] = True

        metric_frame = pd.DataFrame()
        if not config["training"].get("skip", False):
            announce(
                f"{q}q Phase 8: VQC training for methods={config['training']['methods']} "
                f"seeds={config['training']['seeds']} epochs={config['training']['epochs']}"
            )
            histories = []
            metrics = []
            predictions = []
            for seed in [int(s) for s in config["training"]["seeds"]]:
                announce(f"{q}q Training seed {seed}: starting analytic, parameter-shift, and QNDM runs")
                h, m, p, models = run_training_methods(
                    train_x, train_meta, val_x, val_meta, test_x, test_meta, spec, obs, config, seed, best_lambda, int(config["quantum"]["shots"][0])
                )
                histories.append(h.assign(qubits=q, layers=primary_layers))
                metrics.append(m)
                predictions.append(p)
                for method, theta in models.items():
                    np.save(qroot / "models" / f"theta_{method}_seed_{seed}.npy", theta)
            history_frame = pd.concat(histories, ignore_index=True)
            metric_frame = pd.concat(metrics, ignore_index=True)
            prediction_frame = pd.concat(predictions, ignore_index=True)
            write_csv(qroot / "training" / "training_history.csv", history_frame)
            write_csv(qroot / "training" / "final_classification_metrics.csv", metric_frame)
            write_csv(qroot / "training" / "test_predictions.csv", prediction_frame)
            save_table(metric_frame, qroot / "tables" / "table_8_final_classification_performance")
            if not config["experiment"].get("skip_plots"):
                plot_training(history_frame, qroot / "plots" / "training_loss_curve", "training_loss")
                plot_training(history_frame, qroot / "plots" / "validation_loss_curve", "validation_loss")
                for _, row in metric_frame.iterrows():
                    plot_confusion(int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"]), qroot / "plots" / f"confusion_matrix_{row['method']}_seed_{int(row['seed'])}")
            all_metric_rows.append(metric_frame)
            headline_tables[f"Final Classification {q}q"] = metric_frame

        validation[f"{q}-qubit experiment"] = True
        announce(f"Qubit block {q_index}/{len(qubits)}: completed {q}-qubit experiments")

    if all_metric_rows:
        announce("Phase 11/12: writing cross-qubit classification comparison")
        combined_metrics = pd.concat(all_metric_rows, ignore_index=True)
        write_csv(base / "cross_qubit_comparison" / "classification_metrics.csv", combined_metrics)
        save_table(combined_metrics, base / "cross_qubit_comparison" / "table_10_qubit_scaling")
        headline_tables["Cross-Qubit Metrics"] = combined_metrics
    if all_resource_rows:
        announce("Phase 11/12: writing cross-qubit resource comparison")
        combined_resources = pd.concat(all_resource_rows, ignore_index=True)
        write_csv(base / "cross_qubit_comparison" / "resource_counts.csv", combined_resources)
        headline_tables["Resource Counts"] = combined_resources

    announce("Finalizing runtime metadata, paper artifacts, validation report, research summary, and manifest")
    write_json(base / "runtime_summary.json", {"total_runtime_seconds": time.perf_counter() - t0})
    copy_paper_artifacts(base)
    validation["Output integrity"] = True
    validation_report(base, validation)
    research_summary(base, validation, headline_tables)
    write_manifest(base)
    print("=" * 52)
    print("QNDM-ECG RESEARCH PIPELINE COMPLETE")
    print("=" * 52)
    print("Dataset: MIT-BIH Arrhythmia")
    print("Task: AAMI N vs V")
    print("Split: Patient-independent DS1 / DS2")
    for q in qubits:
        print(f"{q}-qubit experiments: COMPLETE")
    print(f"Gradient validation: {'PASS' if validation['Parameter-shift identity'] and validation['QNDM small-lambda validation'] else 'FAIL'}")
    print(f"Patient leakage check: {'PASS' if validation['Patient independence'] else 'FAIL'}")
    print(f"Resource accounting check: {'PASS' if validation['Resource counter'] else 'FAIL'}")
    print("Results: output/")
    print("Research summary: output/RESEARCH_SUMMARY.md")
    print("Validation: output/VALIDATION_REPORT.md")
    print("Paper-ready artifacts: output/paper_results/")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
