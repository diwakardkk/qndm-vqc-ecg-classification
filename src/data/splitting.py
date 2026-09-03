from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

AAMI_N = {"N", "L", "R", "e", "j"}
AAMI_V = {"V", "E"}
STRICT_N = {"N"}
STRICT_V = {"V"}

DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
]
DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
]
PACED_RECORDS = ["102", "104", "107", "217"]


@dataclass(frozen=True)
class LabelPolicy:
    name: str
    n_symbols: set[str]
    v_symbols: set[str]

    def classify(self, symbol: str) -> tuple[int, int] | None:
        if symbol in self.n_symbols:
            return 0, -1
        if symbol in self.v_symbols:
            return 1, 1
        return None


def label_policy(strict: bool = False) -> LabelPolicy:
    if strict:
        return LabelPolicy("strict_N_vs_V", STRICT_N, STRICT_V)
    return LabelPolicy("AAMI_N_vs_V", AAMI_N, AAMI_V)


def configured_records(config: dict, quick: bool = False) -> tuple[list[str], list[str]]:
    data_cfg = config["data"]
    if quick and data_cfg.get("quick_ds1_records"):
        ds1 = [str(x) for x in data_cfg["quick_ds1_records"]]
        ds2 = [str(x) for x in data_cfg.get("quick_ds2_records", data_cfg["ds2_records"])]
    else:
        ds1 = [str(x) for x in data_cfg["ds1_records"]]
        ds2 = [str(x) for x in data_cfg["ds2_records"]]
    excluded = {str(x) for x in data_cfg.get("exclude_records", PACED_RECORDS)}
    return [r for r in ds1 if r not in excluded], [r for r in ds2 if r not in excluded]


def assign_patient_splits(records_ds1: Iterable[str], validation_records: Iterable[str]) -> dict[str, str]:
    val = {str(r) for r in validation_records}
    split = {str(r): ("val" if str(r) in val else "train") for r in records_ds1}
    if not any(s == "train" for s in split.values()) or not any(s == "val" for s in split.values()):
        raise ValueError("DS1 train/validation split must contain at least one record in each split.")
    return split


def assert_no_leakage(metadata: pd.DataFrame) -> None:
    split_by_record = metadata.groupby("record")["split"].nunique()
    leaked = split_by_record[split_by_record > 1]
    if not leaked.empty:
        raise AssertionError(f"Patient leakage detected for records: {list(leaked.index)}")
    train = set(metadata.loc[metadata["split"] == "train", "record"])
    val = set(metadata.loc[metadata["split"] == "val", "record"])
    test = set(metadata.loc[metadata["split"] == "test", "record"])
    if train & val:
        raise AssertionError(f"Train/validation overlap: {sorted(train & val)}")
    if (train | val) & test:
        raise AssertionError(f"DS1/DS2 overlap: {sorted((train | val) & test)}")


def split_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record, group in metadata.groupby("record"):
        rows.append(
            {
                "record_id": record,
                "split": group["split"].iloc[0],
                "number_N": int((group["label"] == 0).sum()),
                "number_V": int((group["label"] == 1).sum()),
                "total_beats": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "record_id"])

