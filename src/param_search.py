"""Parameter grid search for the z-score signal."""

from __future__ import annotations

import itertools
from typing import Sequence

import pandas as pd

from .backtest import run_backtest_with_costs, run_basic_backtest
from .metrics import max_drawdown, sharpe_ratio, trade_count
from .signals import rolling_zscore_signal


def grid_search(
    df_train: pd.DataFrame,
    windows: Sequence[int] = (10, 20, 40),
    z_entries: Sequence[float] = (1.0, 1.5, 2.0),
    exit_bandwidths: Sequence[float | None] = (None,),
    cost_per_unit_change: float = 0.0005,
    df_test: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Sweep a parameter grid over the training set and optionally evaluate on a test set.

    For each (window, z_entry, exit_bandwidth) combination the function computes
    both a cost-free and a cost-aware backtest so callers can compare gross edge
    against net profitability in a single call.

    Args:
        df_train: Training slice (used for signal fitting and selection).
        windows: Rolling window sizes to sweep.
        z_entries: Entry z-score thresholds to sweep.
        exit_bandwidths: Exit thresholds for the hysteresis signal. None means
            the simple threshold signal (exit when |z| drops below z_entry).
        cost_per_unit_change: Transaction cost per unit of position change (e.g.,
            0.0005 = 5 bps).
        df_test: Optional held-out test slice. When provided, each parameter
            combo is also evaluated on df_test and the results are merged into
            the returned DataFrame.

    Returns:
        DataFrame with one row per (window, z_entry, exit_bandwidth) combination
        and columns for train/test Sharpe, max drawdown, trade count, and cost.
        Sorted by train_sharpe_with_cost descending.
    """
    records = []

    for w, z, eb in itertools.product(windows, z_entries, exit_bandwidths):
        if eb is not None and eb >= z:
            continue  # invalid: exit_bandwidth must be < bandwidth

        train_sig = rolling_zscore_signal(
            df_train, window=w, z_entry=z, exit_bandwidth=eb
        )

        bt_nc = run_basic_backtest(train_sig)
        bt_c = run_backtest_with_costs(train_sig, cost_per_unit_change=cost_per_unit_change)

        row: dict = {
            "window": w,
            "z_entry": z,
            "exit_bandwidth": eb,
            "train_sharpe_no_cost": sharpe_ratio(bt_nc.strategy_returns),
            "train_sharpe_with_cost": sharpe_ratio(bt_c.df["strategy_return_net"]),
            "train_max_dd_no_cost": max_drawdown(bt_nc.equity_curve),
            "train_max_dd_with_cost": max_drawdown(bt_c.df["equity_curve_net"]),
            "train_trades": trade_count(bt_c.df["position"]),
            "train_total_cost": bt_c.extras["total_cost"],
            "train_net_return": bt_c.net_return,
        }

        if df_test is not None:
            test_sig = rolling_zscore_signal(
                df_test, window=w, z_entry=z, exit_bandwidth=eb
            )
            bt_test_nc = run_basic_backtest(test_sig)
            bt_test_c = run_backtest_with_costs(
                test_sig, cost_per_unit_change=cost_per_unit_change
            )
            row.update(
                {
                    "test_sharpe_no_cost": sharpe_ratio(bt_test_nc.strategy_returns),
                    "test_sharpe_with_cost": sharpe_ratio(bt_test_c.df["strategy_return_net"]),
                    "test_max_dd_no_cost": max_drawdown(bt_test_nc.equity_curve),
                    "test_max_dd_with_cost": max_drawdown(bt_test_c.df["equity_curve_net"]),
                    "test_trades": trade_count(bt_test_c.df["position"]),
                    "test_total_cost": bt_test_c.extras["total_cost"],
                    "test_net_return": bt_test_c.net_return,
                }
            )

        records.append(row)

    df_out = pd.DataFrame(records)
    return df_out.sort_values("train_sharpe_with_cost", ascending=False).reset_index(drop=True)


__all__ = ["grid_search"]
