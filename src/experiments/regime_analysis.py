from __future__ import annotations

import pandas as pd

from quantum.ansatz import CircuitSpec
from quantum.observables import observable_for_regime
from quantum.resource_counter import analytic_shots, resource_frame


def regime_resource_analysis(spec: CircuitSpec, regimes: list[str], n_train: int, epochs: int, shots: int) -> pd.DataFrame:
    rows = []
    for regime in regimes:
        obs = observable_for_regime(regime, spec.n_qubits)
        base = resource_frame(spec, obs, regime).iloc[0].to_dict()
        shot_counts = analytic_shots(n_train, spec, obs, epochs, m_f=shots, m_d=shots, m_q=shots)
        rows.append({**base, **shot_counts})
    return pd.DataFrame(rows)

