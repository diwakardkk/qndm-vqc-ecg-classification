from __future__ import annotations

import json
import os
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True))


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_latex_table(path: str | Path, frame: pd.DataFrame) -> None:
    def esc(value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            value = f"{value:.5g}"
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

    cols = list(frame.columns)
    lines = [
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\toprule",
        " & ".join(esc(col) for col in cols) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(esc(value) for value in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    text = "\n".join(lines)
    atomic_write_text(path, text)
