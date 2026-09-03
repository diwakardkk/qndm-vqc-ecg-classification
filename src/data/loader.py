from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import wfdb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    wfdb = None


ANN_SYMBOLS = {
    1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S",
    10: "E", 11: "j", 12: "/", 13: "Q", 14: "~", 16: "|", 18: "s", 19: "T",
    20: "*", 21: "D", 22: '"', 23: "=", 24: "p", 25: "B", 26: "^", 27: "t",
    28: "+", 29: "u", 30: "?", 31: "!", 32: "[", 33: "]", 34: "e", 35: "n",
    36: "@", 37: "x", 38: "f", 39: "(", 40: ")", 41: "r",
}


@dataclass(frozen=True)
class Header:
    record: str
    fs: float
    nsig: int
    sig_len: int | None
    signal_names: list[str]
    fmt: list[int]
    gain: list[float]
    baseline: list[float]


def find_dataset(root: str | Path) -> Path:
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset path missing: {path}. Place MIT-BIH under 'MIT-BIH Arrhythmia/' "
            "or rerun with --download-data."
        )
    return path


def validate_dataset(path: str | Path, records: list[str]) -> pd.DataFrame:
    dataset = find_dataset(path)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in records:
        row = {"record": record}
        for suffix in [".dat", ".hea", ".atr"]:
            exists = (dataset / f"{record}{suffix}").exists()
            row[suffix[1:]] = exists
            if not exists:
                missing.append(f"{record}{suffix}")
        rows.append(row)
    if missing:
        raise FileNotFoundError(f"Missing expected MIT-BIH files: {', '.join(missing[:20])}")
    return pd.DataFrame(rows)


def read_header(dataset: str | Path, record: str) -> Header:
    path = Path(dataset) / f"{record}.hea"
    lines = path.read_text(encoding="latin-1").splitlines()
    head = lines[0].split()
    nsig = int(head[1])
    fs = float(head[2].split("/")[0])
    sig_len = int(head[3]) if len(head) > 3 and head[3].isdigit() else None
    names: list[str] = []
    fmts: list[int] = []
    gains: list[float] = []
    baselines: list[float] = []
    for line in lines[1 : 1 + nsig]:
        parts = line.split()
        fmts.append(int(parts[1].split("+")[0]))
        gain_text = parts[2].split("(")[0].split("/")[0]
        gains.append(float(gain_text) if gain_text else 200.0)
        baselines.append(float(parts[4]) if len(parts) > 4 else 0.0)
        names.append(parts[-1])
    return Header(record, fs, nsig, sig_len, names, fmts, gains, baselines)


def select_channel(header: Header, preference: list[str]) -> tuple[int, dict[str, Any]]:
    normalized = [name.upper() for name in header.signal_names]
    for preferred in preference:
        if preferred.upper() in normalized:
            idx = normalized.index(preferred.upper())
            return idx, {
                "record": header.record,
                "available_channels": "|".join(header.signal_names),
                "selected_channel": header.signal_names[idx],
                "fallback_used": preferred.upper() != "MLII",
            }
    return 0, {
        "record": header.record,
        "available_channels": "|".join(header.signal_names),
        "selected_channel": header.signal_names[0],
        "fallback_used": True,
    }


def _decode_212(raw: bytes, nsig: int) -> np.ndarray:
    if nsig != 2:
        raise ValueError("Native reader currently supports MIT-BIH format 212 with two signals.")
    usable = len(raw) - (len(raw) % 3)
    data = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(-1, 3).astype(np.int16)
    s0 = data[:, 0] | ((data[:, 1] & 0x0F) << 8)
    s1 = data[:, 2] | ((data[:, 1] & 0xF0) << 4)
    s0 = np.where(s0 >= 2048, s0 - 4096, s0)
    s1 = np.where(s1 >= 2048, s1 - 4096, s1)
    return np.column_stack([s0, s1]).astype(float)


def read_signal(dataset: str | Path, record: str) -> tuple[np.ndarray, Header]:
    header = read_header(dataset, record)
    if wfdb is not None:
        rec = wfdb.rdrecord(str(Path(dataset) / record))
        return np.asarray(rec.p_signal, dtype=float), header
    if any(fmt != 212 for fmt in header.fmt):
        raise ValueError(f"Record {record} uses unsupported format(s) {header.fmt} without wfdb installed.")
    digital = _decode_212((Path(dataset) / f"{record}.dat").read_bytes(), header.nsig)
    gain = np.asarray(header.gain, dtype=float)
    baseline = np.asarray(header.baseline, dtype=float)
    physical = (digital - baseline) / gain
    return physical, header


def read_annotations(dataset: str | Path, record: str) -> pd.DataFrame:
    if wfdb is not None:
        ann = wfdb.rdann(str(Path(dataset) / record), "atr")
        return pd.DataFrame({"sample": ann.sample.astype(int), "symbol": ann.symbol})
    raw = (Path(dataset) / f"{record}.atr").read_bytes()
    i = 0
    sample = 0
    rows: list[dict[str, Any]] = []
    while i + 1 < len(raw):
        word = raw[i] + (raw[i + 1] << 8)
        i += 2
        ann_type = word >> 10
        delta = word & 0x03FF
        if ann_type == 0:
            break
        if ann_type == 59:
            if i + 3 >= len(raw):
                break
            sample += int.from_bytes(raw[i : i + 4], byteorder="little", signed=True)
            i += 4
            continue
        sample += delta
        if ann_type == 63:
            i += delta + (delta % 2)
            continue
        symbol = ANN_SYMBOLS.get(ann_type)
        if symbol is not None:
            rows.append({"sample": int(sample), "symbol": symbol})
    return pd.DataFrame(rows)

