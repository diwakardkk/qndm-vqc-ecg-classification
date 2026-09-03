from __future__ import annotations

import json
import math
import re
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
PACKAGE = OUTPUT / "latex_package"
FIGURES = PACKAGE / "figures"
ZIP_PATH = OUTPUT / "qndm_ecg_latex_package.zip"

METHOD_LABELS = {
    "analytic": "Analytic",
    "parameter_shift": "Parameter shift",
    "qndm": "QNDM",
    "finite_shot_parameter_shift": "Finite-shot parameter shift",
    "finite_shot_qndm": "Finite-shot QNDM",
    "qndm_finite_lambda": "QNDM finite lambda",
    "qndm_finite_lambda_noiseless": "QNDM noiseless",
}

COLUMN_LABELS = {
    "qubits": "Qubits",
    "method": "Method",
    "seed": "Seed",
    "seeds": "Seeds",
    "layers": "Layers",
    "lambda": "Lambda",
    "shots": "Shots",
    "accuracy": "Acc.",
    "balanced_accuracy": "Bal. acc.",
    "precision": "Prec.",
    "recall_sensitivity": "Recall",
    "specificity": "Spec.",
    "f1": "F1",
    "mcc": "MCC",
    "npv": "NPV",
    "tn": "TN",
    "fp": "FP",
    "fn": "FN",
    "tp": "TP",
    "auroc": "AUROC",
    "average_precision": "AP",
    "training_loss": "Train loss",
    "validation_loss": "Val. loss",
    "training_accuracy": "Train acc.",
    "validation_accuracy": "Val. acc.",
    "training_f1": "Train F1",
    "validation_f1": "Val. F1",
    "cumulative_shots": "Shots used",
    "runtime": "Runtime (s)",
    "record_id": "Record",
    "record": "Record",
    "split": "Split",
    "number_N": "N beats",
    "number_V": "V beats",
    "total_beats": "Total",
    "available_channels": "Available",
    "selected_channel": "Selected",
    "fallback_used": "Fallback",
    "component": "PC",
    "explained_variance_ratio": "Expl. var.",
    "cumulative_variance": "Cum. var.",
    "target_gradient_mse": "Target MSE",
    "finite_shot_parameter_shift_min_shots": "PS shots",
    "finite_shot_qndm_min_shots": "QNDM shots",
    "shot_ratio_qndm_over_dm": "QNDM/PS",
    "shot_reduction_percent": "Reduction",
    "parameters_d": "Params",
    "K": "K",
    "one_qubit_gates": "1q gates",
    "two_qubit_gates": "2q gates",
    "circuit_depth": "Depth",
    "J_M": "J",
    "G_M": "G",
    "regime": "Regime",
    "dm_gradient_per_sample": "PS/sample",
    "qndm_gradient_per_sample": "QNDM/sample",
    "dm_epoch": "PS/epoch",
    "qndm_epoch": "QNDM/epoch",
    "dm_total": "PS total",
    "qndm_total": "QNDM total",
    "sample_index": "Sample",
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "relative_l2_error": "Rel. L2",
    "cosine_similarity": "Cosine",
    "sign_agreement": "Sign",
    "pearson": "Pearson",
    "spearman": "Spearman",
    "mse_mean": "MSE mean",
    "mse_sd": "MSE SD",
    "mae_mean": "MAE mean",
    "mae_sd": "MAE SD",
    "bias_mean": "Bias mean",
    "cosine_similarity_mean": "Cosine mean",
    "relative_l2_error_mean": "Rel. L2 mean",
    "best_lambda_noiseless": "Best lambda",
    "qndm_noiseless_mse": "QNDM MSE",
    "qndm_noiseless_cosine": "QNDM cosine",
    "qndm_noiseless_relative_l2": "QNDM rel. L2",
    "best_shots_in_grid": "Best shots",
    "best_mse": "Best MSE",
    "best_mae": "Best MAE",
    "best_cosine": "Best cosine",
    "classifier": "Classifier",
    "model": "Model",
    "macro_f1": "Macro F1",
    "v_precision": "V prec.",
    "v_recall": "V recall",
    "V_precision": "V prec.",
    "V_recall": "V recall",
}


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def method_label(value) -> str:
    return METHOD_LABELS.get(str(value), str(value).replace("_", " ").title())


def esc(value) -> str:
    if value is None:
        return ""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer() and abs(value) < 1e9:
            return str(int(value))
        if abs(value) >= 10000 or (0 < abs(value) < 0.001):
            return f"{value:.3e}"
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def chunked(data: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    return [data.iloc[start : start + size] for start in range(0, len(data), size)]


def display_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    data = df.copy()
    if columns is not None:
        data = data[[col for col in columns if col in data.columns]]
    if "method" in data.columns:
        data["method"] = data["method"].map(method_label)
    data = data.rename(columns={col: COLUMN_LABELS.get(col, col.replace("_", " ").title()) for col in data.columns})
    for col in data.columns:
        data[col] = data[col].map(fmt)
    return data


def latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    *,
    columns: list[str] | None = None,
    landscape: bool = False,
    font_size: str = "",
    rows_per_table: int | None = None,
    max_height: str = r"0.70\textheight",
) -> str:
    data = display_table(df, columns)
    if data.empty:
        return ""
    if rows_per_table is None:
        rows_per_table = 14 if len(data.columns) > 8 else 24

    parts: list[str] = []
    groups = chunked(data, rows_per_table)
    for part_idx, part in enumerate(groups, start=1):
        suffix = f" (part {part_idx} of {len(groups)})" if len(groups) > 1 else ""
        part_label = label if part_idx == 1 else f"{label}-{part_idx}"
        align = "l" * len(part.columns)
        tabular = [
            r"\begin{tabular}{" + align + "}",
            r"\toprule",
            " & ".join(esc(c) for c in part.columns) + r" \\",
            r"\midrule",
        ]
        for row in part.itertuples(index=False, name=None):
            tabular.append(" & ".join(esc(v) for v in row) + r" \\")
        tabular += [r"\bottomrule", r"\end{tabular}"]

        table_lines = [
            r"\begin{table}[!p]",
            r"\centering",
            font_size,
            r"\setlength{\tabcolsep}{2pt}",
            r"{\renewcommand{\arraystretch}{0.86}",
            rf"\caption{{{esc(caption + suffix)}}}",
            rf"\label{{{part_label}}}",
            rf"\begin{{adjustbox}}{{max width=0.96\textwidth,max height={max_height},center}}",
            *tabular,
            r"\end{adjustbox}",
            r"}",
            r"\end{table}",
        ]
        parts.extend([*table_lines, ""])
    return "\n".join(parts)


def validation_table(markdown: str) -> pd.DataFrame:
    rows = []
    for line in markdown.splitlines():
        match = re.match(r"-\s*(.+?):\s*(.+)", line.strip())
        if match:
            rows.append({"Check": match.group(1), "Status": match.group(2)})
    return pd.DataFrame(rows)


def metric_cell(mean: float, sd: float) -> str:
    return f"{mean:.4f} ({sd:.4f})"


def summarize_quantum_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["qubits", "method"])
        .agg(
            seeds=("seed", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_sd=("mcc", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_sd=("average_precision", "std"),
        )
        .reset_index()
    )
    rows = []
    order = {"analytic": 0, "parameter_shift": 1, "qndm": 2}
    grouped["method_order"] = grouped["method"].map(order).fillna(99)
    grouped = grouped.sort_values(["qubits", "method_order"])
    for row in grouped.itertuples(index=False):
        rows.append(
            {
                "qubits": row.qubits,
                "method": row.method,
                "seeds": row.seeds,
                "accuracy": metric_cell(row.accuracy_mean, row.accuracy_sd),
                "balanced_accuracy": metric_cell(row.balanced_accuracy_mean, row.balanced_accuracy_sd),
                "f1": metric_cell(row.f1_mean, row.f1_sd),
                "mcc": metric_cell(row.mcc_mean, row.mcc_sd),
                "auroc": metric_cell(row.auroc_mean, row.auroc_sd),
                "average_precision": metric_cell(row.average_precision_mean, row.average_precision_sd),
            }
        )
    return pd.DataFrame(rows)


def final_epoch_summary() -> pd.DataFrame:
    rows = []
    for q in [4, 6, 8]:
        path = OUTPUT / f"{q}_qubits" / "training" / "training_history.csv"
        if not path.exists():
            continue
        df = read_csv(path)
        idx = df.groupby(["qubits", "method", "seed"])["epoch"].idxmax()
        rows.append(df.loc[idx])
    if not rows:
        return pd.DataFrame()
    cols = [
        "qubits",
        "method",
        "seed",
        "epoch",
        "training_loss",
        "validation_loss",
        "training_accuracy",
        "validation_accuracy",
        "training_f1",
        "validation_f1",
        "cumulative_shots",
        "runtime",
    ]
    return pd.concat(rows, ignore_index=True)[cols].round(6)


def best_gradient_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    lambda_rows = []
    shot_rows = []
    for q in [4, 6, 8]:
        lam = read_csv(OUTPUT / f"{q}_qubits" / "lambda_sweep" / "lambda_sweep_summary.csv")
        best_lam = lam.sort_values("mse_mean").iloc[0].to_dict()
        lambda_rows.append(
            {
                "qubits": q,
                "best_lambda_noiseless": best_lam["lambda"],
                "qndm_noiseless_mse": best_lam["mse_mean"],
                "qndm_noiseless_cosine": best_lam["cosine_similarity_mean"],
                "qndm_noiseless_relative_l2": best_lam["relative_l2_error_mean"],
            }
        )
        shot = read_csv(OUTPUT / f"{q}_qubits" / "shot_sweep" / "shot_sweep_summary.csv")
        for method, group in shot.groupby("method"):
            best = group.sort_values("mse_mean").iloc[0].to_dict()
            shot_rows.append(
                {
                    "qubits": q,
                    "method": method,
                    "best_shots_in_grid": int(best["shots"]),
                    "best_mse": best["mse_mean"],
                    "best_mae": best["mae_mean"],
                    "best_cosine": best["cosine_similarity_mean"],
                }
            )
    return pd.DataFrame(lambda_rows), pd.DataFrame(shot_rows)


def copy_figures() -> list[tuple[str, str]]:
    if FIGURES.exists():
        shutil.rmtree(FIGURES)
    FIGURES.mkdir(parents=True, exist_ok=True)
    copied = []
    seen = set()
    for path in sorted(OUTPUT.rglob("*.pdf")):
        if "latex_package" in path.parts or "paper_results" in path.parts:
            continue
        rel = path.relative_to(OUTPUT)
        prefix = "__".join(rel.parts[:-1])
        target_name = f"{prefix}__{path.name}".replace(" ", "_")
        if target_name in seen:
            continue
        seen.add(target_name)
        shutil.copy2(path, FIGURES / target_name)
        copied.append((target_name, str(rel)))
    return copied


def figure_label(source: str) -> str:
    path = Path(source)
    stem = path.stem
    q_match = re.match(r"(\d+)_qubits", source)
    q_label = f"{q_match.group(1)} qubits" if q_match else ""
    if stem.startswith("confusion_matrix_"):
        body = stem.replace("confusion_matrix_", "")
        seed_match = re.search(r"_seed_(\d+)$", body)
        seed = seed_match.group(1) if seed_match else ""
        method = body[: seed_match.start()] if seed_match else body
        return ", ".join(part for part in [q_label, method_label(method), f"seed {seed}" if seed else ""] if part)
    labels = {
        "lambda_vs_cosine_similarity": "Lambda sweep: cosine similarity",
        "lambda_vs_gradient_mse": "Lambda sweep: gradient MSE",
        "training_loss_curve": "Training loss",
        "validation_loss_curve": "Validation loss",
        "class_distribution": "Class distribution",
        "raw_vs_filtered_ECG": "Raw and filtered ECG",
        "representative_N_beats": "Representative N beats",
        "representative_V_beats": "Representative V beats",
        "train_val_test_distribution": "Train/validation/test distribution",
    }
    if stem.startswith("pca_explained_variance"):
        return f"{q_label}, PCA explained variance" if q_label else "PCA explained variance"
    if stem.startswith("shots_vs_gradient_mse"):
        method = stem.replace("shots_vs_gradient_mse_", "")
        return f"{q_label}, shot sweep: {method_label(method)}" if q_label else f"Shot sweep: {method_label(method)}"
    return labels.get(stem, stem.replace("_", " ").title())


def figure_grid(
    items: list[tuple[str, str]],
    caption: str,
    label_prefix: str,
    *,
    per_page: int = 4,
    width: str = r"0.46\textwidth",
    height: str = r"0.25\textheight",
) -> list[str]:
    lines: list[str] = []
    pages = [items[i : i + per_page] for i in range(0, len(items), per_page)]
    for page_idx, group in enumerate(pages, start=1):
        lines += [
            r"\begin{figure}[!p]",
            r"\centering",
            r"\setlength{\abovecaptionskip}{3pt}",
            r"\setlength{\belowcaptionskip}{0pt}",
        ]
        for row_start in range(0, len(group), 2):
            row = group[row_start : row_start + 2]
            for item_idx, (name, source) in enumerate(row):
                lines += [
                    rf"\begin{{minipage}}[t]{{{width}}}",
                    r"\centering",
                    rf"\begin{{adjustbox}}{{max width=\linewidth,max height={height},center}}",
                    rf"\includegraphics{{\detokenize{{figures/{name}}}}}",
                    r"\end{adjustbox}",
                    rf"{{{esc(figure_label(source))}\par}}",
                    r"\end{minipage}",
                ]
                if item_idx == 0 and len(row) > 1:
                    lines.append(r"\hfill")
            if row_start + 2 < len(group):
                lines.append(r"\par\vspace{0.5em}")
        part = f" Part {page_idx} of {len(pages)}." if len(pages) > 1 else ""
        lines += [
            rf"\caption{{{esc(caption + part)}}}",
            rf"\label{{{label_prefix}-{page_idx}}}",
            r"\end{figure}",
            "",
        ]
    return lines


def figure_section(copied: Iterable[tuple[str, str]]) -> str:
    items = list(copied)
    confusion = [item for item in items if "confusion_matrix" in item[0]]
    other = [item for item in items if "confusion_matrix" not in item[0]]
    lines = [r"\clearpage", r"\section{Figures}", ""]
    lines += figure_grid(
        confusion,
        "Confusion matrices grouped four per page.",
        "fig:confusion-grid",
        per_page=4,
        height=r"0.245\textheight",
    )
    lines += figure_grid(
        other,
        "Experiment and data-analysis plots grouped four per page.",
        "fig:plot-grid",
        per_page=4,
        height=r"0.25\textheight",
    )
    return "\n".join(lines)


def build() -> None:
    if not OUTPUT.exists():
        raise SystemExit("output/ does not exist; run experiments first.")
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True, exist_ok=True)

    validation = (OUTPUT / "VALIDATION_REPORT.md").read_text(encoding="utf-8")
    runtime = json.loads((OUTPUT / "runtime_summary.json").read_text(encoding="utf-8"))
    dataset = read_csv(OUTPUT / "data_analysis" / "dataset_summary.csv")
    split = read_csv(OUTPUT / "data_analysis" / "split_distribution.csv")
    patient_split = read_csv(OUTPUT / "data_analysis" / "patient_split.csv")
    channel = read_csv(OUTPUT / "data_analysis" / "channel_selection.csv")
    metrics = read_csv(OUTPUT / "cross_qubit_comparison" / "classification_metrics.csv")
    resources = read_csv(OUTPUT / "cross_qubit_comparison" / "resource_counts.csv")
    metric_summary = summarize_quantum_metrics(metrics)
    lambda_best, shot_best = best_gradient_tables()
    epoch_summary = final_epoch_summary()

    classical_tables = []
    pca_tables = []
    matched_tables = []
    lambda_tables = []
    shot_tables = []
    gradient_tables = []
    for q in [4, 6, 8]:
        classical_tables.append(read_csv(OUTPUT / "classical_baselines" / f"{q}_qubits" / "classical_baselines.csv").assign(qubits=q))
        pca_tables.append(read_csv(OUTPUT / f"{q}_qubits" / "data" / f"pca_explained_variance_{q}q.csv").assign(qubits=q))
        matched_tables.append(read_csv(OUTPUT / f"{q}_qubits" / "resources" / "matched_precision_resource_table.csv").assign(qubits=q))
        lambda_tables.append(read_csv(OUTPUT / f"{q}_qubits" / "lambda_sweep" / "lambda_sweep_summary.csv").assign(qubits=q))
        shot_tables.append(read_csv(OUTPUT / f"{q}_qubits" / "shot_sweep" / "shot_sweep_summary.csv").assign(qubits=q))
        gradient_tables.append(read_csv(OUTPUT / f"{q}_qubits" / "gradient_validation" / "gradient_correctness_tests.csv").assign(qubits=q))

    classical = pd.concat(classical_tables, ignore_index=True)
    pca = pd.concat(pca_tables, ignore_index=True)
    matched = pd.concat(matched_tables, ignore_index=True)
    lambdas = pd.concat(lambda_tables, ignore_index=True)
    shots = pd.concat(shot_tables, ignore_index=True)
    gradients = pd.concat(gradient_tables, ignore_index=True)
    copied_figures = copy_figures()

    verdict = (
        "The completed run is not publishable as evidence of a QNDM resource or training advantage. "
        "It is suitable only as a negative or diagnostic computational study unless the finite-shot QNDM "
        "training protocol is redesigned. The noiseless QNDM detector-gradient validation passes, but the "
        "finite-shot QNDM classifiers fail across 4, 6, and 8 qubits, while parameter-shift training remains "
        "close to the analytic-gradient reference."
    )

    metric_columns = [
        "qubits",
        "method",
        "seed",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "mcc",
        "auroc",
        "average_precision",
    ]
    count_columns = ["qubits", "method", "seed", "tn", "fp", "fn", "tp"]
    lambda_columns = [
        "qubits",
        "lambda",
        "mse_mean",
        "mse_sd",
        "mae_mean",
        "mae_sd",
        "bias_mean",
        "cosine_similarity_mean",
        "relative_l2_error_mean",
    ]
    shot_columns = [
        "qubits",
        "method",
        "shots",
        "mse_mean",
        "mse_sd",
        "mae_mean",
        "mae_sd",
        "bias_mean",
        "cosine_similarity_mean",
    ]
    gradient_columns = [
        "qubits",
        "sample_index",
        "method",
        "lambda",
        "mse",
        "mae",
        "relative_l2_error",
        "cosine_similarity",
        "sign_agreement",
        "pearson",
        "spearman",
    ]

    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,margin=0.62in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{adjustbox}",
        r"\usepackage{caption}",
        r"\usepackage{float}",
        r"\usepackage{amsmath}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\captionsetup{labelfont=bf}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.45em}",
        r"\sloppy",
        r"\begin{document}",
        r"\title{\vspace{-1.5em}QNDM-Enhanced Variational Quantum ECG Classification\\Full Results Report}",
        r"\author{Generated from completed experiment outputs}",
        r"\date{2026-09-02}",
        r"\maketitle",
        r"\section{Verdict}",
        esc(verdict),
        "",
        r"\section{Experimental Scope}",
        (
            "The full run evaluated 4, 6, and 8 qubits with two variational layers, five random seeds, "
            "analytic-gradient VQC, finite-shot parameter-shift VQC, and finite-shot QNDM-gradient VQC. "
            "The ECG task was AAMI N-type versus V-type heartbeat classification using a patient-independent "
            "DS1/DS2 split. The complete runtime was "
            f"{runtime.get('total_runtime_seconds', float('nan')) / 3600:.3f} hours."
        ),
        "",
        r"\section{Main Findings}",
        r"\begin{itemize}",
        r"\item All validation checks passed, including dataset detection, patient independence, preprocessing leakage checks, PCA leakage checks, parameter-shift identity checks, QNDM detector checks, and output integrity checks.",
        r"\item Noiseless QNDM reproduced exact gradients at small lambda, with cosine similarity approximately equal to 1.",
        r"\item Finite-shot QNDM was unstable at the selected training setting of lambda 0.0001 and 100 shots, producing poor classification results across all qubit counts.",
        r"\item Finite-shot parameter shift reached all tested matched-gradient-MSE thresholds in the shot grid; finite-shot QNDM did not reach any tested threshold.",
        r"\item The present evidence does not support a publishable positive QNDM advantage claim.",
        r"\end{itemize}",
        "",
        r"\section{Tables}",
        latex_table(validation_table(validation), "Validation checklist.", "tab:validation", rows_per_table=30),
        latex_table(dataset, "Dataset summary.", "tab:dataset"),
        latex_table(split, "Split-level class counts.", "tab:split"),
        latex_table(patient_split, "Patient-independent record split and class counts.", "tab:patient-split", rows_per_table=30),
        latex_table(channel, "Selected ECG channel by record.", "tab:channels", rows_per_table=30),
        latex_table(pca.round(6), "PCA explained variance by qubit count.", "tab:pca"),
        latex_table(
            classical.round(6),
            "Classical reference baselines on PCA features.",
            "tab:classical",
            columns=["qubits", "model", "accuracy", "macro_f1", "V_precision", "V_recall"],
        ),
        latex_table(metric_summary, "Final quantum classification summary across seeds. Values are mean with standard deviation in parentheses.", "tab:quantum-summary"),
        latex_table(metrics.round(6), "Final quantum classification metrics for every seed.", "tab:quantum-all", columns=metric_columns, rows_per_table=14),
        latex_table(metrics.round(0), "Confusion-matrix counts for every quantum test run.", "tab:quantum-counts", columns=count_columns, rows_per_table=30),
        latex_table(epoch_summary.round(6), "Final epoch training metrics by qubit, method, and seed.", "tab:epoch-summary", rows_per_table=14),
        latex_table(lambda_best.round(8), "Best noiseless QNDM lambda from validation gradient MSE.", "tab:best-lambda"),
        latex_table(lambdas.round(8), "Noiseless lambda sweep summaries.", "tab:lambda-all", columns=lambda_columns, rows_per_table=14),
        latex_table(shot_best.round(8), "Best finite-shot gradient-MSE result in the shot grid.", "tab:best-shot"),
        latex_table(shots.round(8), "Finite-shot sweep summaries.", "tab:shot-all", columns=shot_columns, rows_per_table=14),
        latex_table(matched, "Matched-gradient-precision resource thresholds.", "tab:matched"),
        latex_table(resources, "Analytic resource counts and regime comparison.", "tab:resources"),
        latex_table(gradients.round(8), "Gradient correctness tests for analytic, parameter-shift, and QNDM estimators.", "tab:gradients", columns=gradient_columns, rows_per_table=14),
        figure_section(copied_figures),
        r"\end{document}",
        "",
    ]

    (PACKAGE / "main.tex").write_text("\n".join(lines), encoding="utf-8")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(PACKAGE))
    print(f"Wrote {ZIP_PATH}")
    print(f"Package folder: {PACKAGE}")
    print(f"Figures copied: {len(copied_figures)}")


if __name__ == "__main__":
    build()
