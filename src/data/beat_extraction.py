from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .loader import read_annotations, read_signal, select_channel, validate_dataset
from .preprocessing import bandpass
from .splitting import assert_no_leakage, assign_patient_splits, label_policy, split_summary


def extract_record_beats(
    dataset_dir: str | Path,
    record: str,
    split: str,
    config: dict[str, Any],
    policy,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    signal, header = read_signal(dataset_dir, record)
    channel_idx, channel_row = select_channel(header, config["data"]["channel_preference"])
    raw = signal[:, channel_idx]
    filtered = bandpass(
        raw,
        fs=header.fs,
        low_hz=config["data"]["bandpass_low_hz"],
        high_hz=config["data"]["bandpass_high_hz"],
        order=config["data"]["filter_order"],
    )
    before = int(round(config["data"]["window_before_sec"] * header.fs))
    after = int(round(config["data"]["window_after_sec"] * header.fs))
    annotations = read_annotations(dataset_dir, record)
    max_per_class = config["data"].get("max_beats_per_record_per_class")
    counts = {0: 0, 1: 0}
    windows = []
    rows = []
    excluded: dict[str, int] = {}
    for ann in annotations.itertuples(index=False):
        label = policy.classify(ann.symbol)
        if label is None:
            excluded[ann.symbol] = excluded.get(ann.symbol, 0) + 1
            continue
        conventional, squared = label
        if max_per_class is not None and counts[conventional] >= int(max_per_class):
            excluded[f"{ann.symbol}_sampling_cap"] = excluded.get(f"{ann.symbol}_sampling_cap", 0) + 1
            continue
        start = int(ann.sample) - before
        stop = int(ann.sample) + after
        if start < 0 or stop > len(filtered):
            excluded["boundary"] = excluded.get("boundary", 0) + 1
            continue
        segment = filtered[start:stop].astype(float)
        if not np.isfinite(segment).all():
            excluded["nonfinite"] = excluded.get("nonfinite", 0) + 1
            continue
        windows.append(segment)
        counts[conventional] += 1
        rows.append(
            {
                "record": record,
                "split": split,
                "beat_sample": int(ann.sample),
                "symbol": ann.symbol,
                "label": conventional,
                "target": squared,
                "fs": header.fs,
                "channel": channel_row["selected_channel"],
            }
        )
    excluded_frame = pd.DataFrame(
        [{"record": record, "symbol_or_reason": k, "count": v} for k, v in sorted(excluded.items())]
    )
    example = {"record": record, "raw": raw[:2500], "filtered": filtered[:2500], "fs": header.fs}
    if not windows:
        return np.empty((0, before + after)), pd.DataFrame(rows), channel_row, excluded_frame, example
    return np.vstack(windows), pd.DataFrame(rows), channel_row, excluded_frame, example


def build_dataset(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    data_cfg = config["data"]
    quick = config["experiment"]["name"] == "quick"
    ds1_records = [str(r) for r in (data_cfg.get("quick_ds1_records") if quick else data_cfg["ds1_records"])]
    ds2_records = [str(r) for r in (data_cfg.get("quick_ds2_records") if quick else data_cfg["ds2_records"])]
    excluded_records = {str(r) for r in data_cfg.get("exclude_records", [])}
    ds1_records = [r for r in ds1_records if r not in excluded_records]
    ds2_records = [r for r in ds2_records if r not in excluded_records]
    validate_dataset(data_cfg["dataset_dir"], sorted(set(ds1_records + ds2_records)))
    split_map = assign_patient_splits(ds1_records, [str(r) for r in data_cfg["validation_records"]])
    policy = label_policy(bool(data_cfg.get("strict_labels", False)))
    arrays = []
    meta = []
    channels = []
    excluded = []
    examples = []
    for record in tqdm(ds1_records + ds2_records, desc="Extracting MIT-BIH beats"):
        split = split_map.get(record, "test")
        x, m, channel, ex, example = extract_record_beats(data_cfg["dataset_dir"], record, split, config, policy)
        if len(m):
            arrays.append(x)
            meta.append(m)
        channels.append(channel)
        if len(ex):
            excluded.append(ex)
        examples.append(example)
    if not arrays:
        raise RuntimeError("No AAMI N/V beats were extracted. Check label policy and annotations.")
    x_all = np.vstack(arrays)
    metadata = pd.concat(meta, ignore_index=True)
    assert_no_leakage(metadata)
    channel_df = pd.DataFrame(channels)
    excluded_df = pd.concat(excluded, ignore_index=True) if excluded else pd.DataFrame()
    split_df = split_summary(metadata)
    return {
        "x": x_all,
        "metadata": metadata,
        "channel_selection": channel_df,
        "excluded_beats": excluded_df,
        "patient_split": split_df,
        "policy": policy.name,
        "examples": examples,
    }

