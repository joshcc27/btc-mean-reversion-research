"""Parameter grid search for the z-score signal."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import pandas as pd

from .backtest import run_backtest
from .metrics import summarize_backtest
from .signals import rolling_zscore_signal


def grid_search(
    df: pd.DataFrame,
    eval_splits: Mapping[str, pd.Index],
    windows: Sequence[int],
    z_entries: Sequence[float],
    exit_bandwidths: Sequence[float | None] = (None,),
    cost_per_unit_turnover: float = 0.001,
) -> pd.DataFrame:
    """Evaluate every (window, z_entry, exit_bandwidth) combination on named index ranges.

    The signal for each combination is computed once on all of ``df`` (it uses
    only past bars) and then backtested separately on each split, starting flat.
    Pass only the splits you are allowed to look at: selection code should never
    receive the test index. Combinations with exit_bandwidth >= z_entry are skipped.

    Returns:
        One row per combination with ``<split>_<metric>`` columns from
        ``summarize_backtest``, sorted by the first split's net Sharpe (descending).
    """
    records = []
    for w, z, eb in itertools.product(windows, z_entries, exit_bandwidths):
        if eb is not None and eb >= z:
            continue
        sig = rolling_zscore_signal(df, window=w, z_entry=z, exit_bandwidth=eb)
        row: dict[str, float | None] = {"window": w, "z_entry": z, "exit_bandwidth": eb}
        for name, idx in eval_splits.items():
            bt = run_backtest(sig.loc[idx], cost_per_unit_turnover=cost_per_unit_turnover)
            row.update({f"{name}_{k}": v for k, v in summarize_backtest(bt).items()})
        records.append(row)

    key = f"{next(iter(eval_splits))}_sharpe_net"
    return pd.DataFrame(records).sort_values(key, ascending=False).reset_index(drop=True)
