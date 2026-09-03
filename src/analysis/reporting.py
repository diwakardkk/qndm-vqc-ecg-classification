from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.io import atomic_write_text, ensure_dir, write_csv


def output_tree(base: str | Path, qubits: list[int]) -> None:
    base = ensure_dir(base)
    ensure_dir(base / "data_analysis")
    ensure_dir(base / "classical_baselines")
    ensure_dir(base / "cross_qubit_comparison")
    ensure_dir(base / "paper_results" / "figures")
    ensure_dir(base / "paper_results" / "tables")
    for q in qubits:
        root = ensure_dir(base / f"{q}_qubits")
        for sub in [
            "config", "data", "gradient_validation", "lambda_sweep", "shot_sweep", "training",
            "regime_A", "regime_B", "regime_C", "resources", "statistics", "tables", "plots",
            "models", "logs", "checkpoints",
        ]:
            ensure_dir(root / sub)


def write_manifest(base: str | Path) -> None:
    base = Path(base)
    rows = []
    for path in sorted(base.rglob("*")):
        if path.is_file():
            rows.append({
                "artifact": str(path.relative_to(base)),
                "type": path.suffix.lstrip(".") or "file",
                "qubits": next((part.split("_")[0] for part in path.parts if part.endswith("_qubits")), ""),
                "regime": next((part for part in path.parts if part.startswith("regime_")), ""),
                "method": "",
                "seed": "",
                "description": "",
                "source_csv": "",
            })
    write_csv(base / "result_manifest.csv", pd.DataFrame(rows))


def _status(value) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return str(value)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "No rows were produced."
    small = frame.head(max_rows)
    cols = [str(c) for c in small.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in small.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if pd.isna(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def research_summary(base: str | Path, validation: dict, headline_tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# QNDM ECG Research Summary",
        "",
        "This summary is generated only from computed pipeline outputs. It does not assert quantum advantage unless a matched resource metric supports it.",
        "",
        "## Validation",
    ]
    for key, ok in validation.items():
        lines.append(f"- {key}: {_status(ok)}")
    for name, frame in headline_tables.items():
        lines += ["", f"## {name}", ""]
        if frame.empty:
            lines.append("No rows were produced.")
        else:
            lines.append(_markdown_table(frame))
    atomic_write_text(Path(base) / "RESEARCH_SUMMARY.md", "\n".join(lines) + "\n")


def validation_report(base: str | Path, validation: dict) -> None:
    lines = ["# Validation Report", ""]
    for key, ok in validation.items():
        lines.append(f"- {key}: {_status(ok)}")
    atomic_write_text(Path(base) / "VALIDATION_REPORT.md", "\n".join(lines) + "\n")
