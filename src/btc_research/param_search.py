"""Parameter grid search for the z-score signal."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import pandas as pd

from .backtest import run_backtest
from .metrics import summarize_backtest
from .signals import Direction, rolling_zscore_signal


def grid_search(
    df: pd.DataFrame,
    eval_splits: Mapping[str, pd.Index],
    windows: Sequence[int] = (10, 20, 40),
    z_entries: Sequence[float] = (1.0, 1.5, 2.0),
    exit_bandwidths: Sequence[float | None] = (None,),
    cost_per_unit_turnover: float = 0.001,
    direction: Direction = "mean_reversion",
    sort_by: str | None = None,
) -> pd.DataFrame:
    """Evaluate every (window, z_entry, exit_bandwidth) combination on named index ranges.

    The signal for each combination is computed once on all of ``df`` (it uses
    only past bars) and then backtested separately on each split. Pass only the
    splits you are allowed to look at: selection code should never receive the
    test index.

    Args:
        df: Price frame covering every split.
        eval_splits: Mapping such as {"train": train.index}. Each split is
            backtested on its own, starting flat.
        windows, z_entries, exit_bandwidths: Grid values. Combinations with
            exit_bandwidth >= z_entry are skipped.
        cost_per_unit_turnover: Cost per unit |position change|.
        direction: Signal direction.
        sort_by: Column to sort by (descending). Defaults to the first split's
            net Sharpe.

    Returns:
        One row per combination with ``<split>_<metric>`` columns from
        ``summarize_backtest``.
    """
    records = []
    for w, z, eb in itertools.product(windows, z_entries, exit_bandwidths):
        if eb is not None and eb >= z:
            continue
        sig = rolling_zscore_signal(df, window=w, z_entry=z, exit_bandwidth=eb, direction=direction)
        row: dict[str, float | None] = {"window": w, "z_entry": z, "exit_bandwidth": eb}
        for name, idx in eval_splits.items():
            bt = run_backtest(sig.loc[idx], cost_per_unit_turnover=cost_per_unit_turnover)
            row.update({f"{name}_{k}": v for k, v in summarize_backtest(bt).items()})
        records.append(row)

    out = pd.DataFrame(records)
    if out.empty:
        return out
    key = sort_by or f"{next(iter(eval_splits))}_sharpe_net"
    return out.sort_values(key, ascending=False).reset_index(drop=True)


__all__ = ["grid_search"]
