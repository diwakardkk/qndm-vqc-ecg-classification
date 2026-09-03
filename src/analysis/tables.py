from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.io import write_csv, write_latex_table


def save_table(frame: pd.DataFrame, path_base: str | Path) -> None:
    base = Path(path_base)
    write_csv(base.with_suffix(".csv"), frame)
    write_latex_table(base.with_suffix(".tex"), frame)

