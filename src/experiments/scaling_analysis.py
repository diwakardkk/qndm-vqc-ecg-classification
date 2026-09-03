from __future__ import annotations

import pandas as pd


def cross_qubit_table(resource_rows: list[pd.DataFrame], metric_rows: list[pd.DataFrame]) -> pd.DataFrame:
    resources = pd.concat(resource_rows, ignore_index=True) if resource_rows else pd.DataFrame()
    metrics = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    if resources.empty:
        return metrics
    if metrics.empty:
        return resources
    return metrics.merge(resources, on=["qubits", "layers"], how="left")

