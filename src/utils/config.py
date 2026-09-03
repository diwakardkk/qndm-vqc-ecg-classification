from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(config)
    if args.qubits:
        cfg.setdefault("features", {})["qubits"] = [int(q) for q in args.qubits.split(",")]
    if args.regime:
        cfg.setdefault("quantum", {})["regimes"] = [args.regime]
    if args.seed is not None:
        cfg.setdefault("training", {})["seeds"] = [args.seed]
    if args.force:
        cfg.setdefault("experiment", {})["force"] = True
    if args.skip_training:
        cfg.setdefault("training", {})["skip"] = True
    if args.skip_plots:
        cfg.setdefault("experiment", {})["skip_plots"] = True
    return cfg

