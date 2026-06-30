"""Rolling walk-forward utilities for time-series backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import itertools

import numpy as np
import pandas as pd

from .backtest import run_backtest_with_costs
from .metrics import max_drawdown, sharpe_ratio, trade_count
from .signals import rolling_zscore_signal


DEFAULT_WINDOW: int = 20
DEFAULT_Z_ENTRY: float = 1.5
DEFAULT_COST_PER_UNIT: float = 0.0005

def best_params_on_train(
    df_train: pd.DataFrame,
    windows: Sequence[int],
    z_entries: Sequence[float],
    cost_per_unit_change: float,
) -> Tuple[int, float]:
    """Grid-search the given train slice and return the best (window, z_entry)."""

    best_sharpe = float("-inf")
    best_params: Tuple[int, float] | None = None

    for w, z in itertools.product(windows, z_entries):
        df_sig = rolling_zscore_signal(df_train, window=w, z_entry=z)
        bt = run_backtest_with_costs(
            df_sig,
            cost_per_unit_change=cost_per_unit_change,
        )
        ret = bt.df["strategy_return_net"].dropna()
        if ret.empty:
            continue
        current_sharpe = sharpe_ratio(ret)
        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            best_params = (w, z)

    if best_params is None:
        raise ValueError("Unable to identify best parameters on training window.")

    return best_params


@dataclass(frozen=True)
class WindowSpec:
    """Lightweight container for a single train/test window."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_rolling_windows(
    df: pd.DataFrame,
    train_days: int = 60,
    test_days: int = 10,
) -> List[WindowSpec]:
    """Generate rolling train/test windows over a datetime-indexed DataFrame."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    if df.empty:
        return []
    if train_days <= 0 or test_days <= 0:
        raise ValueError("train_days and test_days must be positive.")

    idx = df.index.sort_values()
    start_date = idx.min().normalize()
    end_date = idx.max().normalize()

    windows: List[WindowSpec] = []
    current_train_start = start_date

    train_delta = pd.Timedelta(days=train_days)
    test_delta = pd.Timedelta(days=test_days)
    one_second = pd.Timedelta(seconds=1)

    while True:
        train_end = current_train_start + train_delta - one_second
        test_start = train_end + one_second
        test_end = test_start + test_delta - one_second

        if test_start > end_date:
            break

        windows.append(
            WindowSpec(
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        current_train_start = current_train_start + test_delta
        if current_train_start + train_delta > end_date:
            break

    return windows


def walk_forward_fixed_params(
    df: pd.DataFrame,
    windows: List[WindowSpec],
    window_param: int = DEFAULT_WINDOW,
    z_entry_param: float = DEFAULT_Z_ENTRY,
    cost_per_unit_change: float = DEFAULT_COST_PER_UNIT,
    price_column: str = "close",
) -> Dict[str, Any]:
    """
    Run a fixed-parameter walk-forward evaluation over a sequence of windows.

    Args:
        df: Input DataFrame containing prices/log returns.
        windows: List of WindowSpec instances defining train/test slices.
        window_param: Rolling window for the z-score signal.
        z_entry_param: Z-score threshold for entries.
        cost_per_unit_change: Transaction cost per unit of position change.
        price_column: Column name used to compute log returns in the signal helper.

    Returns:
        Dictionary with concatenated test returns, equity curve, and per-window stats.
    """

    if not windows:
        raise ValueError("windows must contain at least one window.")

    all_test_returns: List[pd.Series] = []
    all_test_equity: List[pd.Series] = []
    per_window_metrics: List[Dict[str, Any]] = []

    equity = 1.0

    for spec in windows:
        df_slice = df.loc[spec.train_start : spec.test_end].copy()
        df_train = df_slice.loc[spec.train_start : spec.train_end]
        df_test = df_slice.loc[spec.test_start : spec.test_end]

        if df_train.empty or df_test.empty:
            continue

        df_signal = rolling_zscore_signal(df_slice, window=window_param, z_entry=z_entry_param)
        df_test_sig = df_signal.loc[spec.test_start : spec.test_end]

        bt_test = run_backtest_with_costs(
            df_test_sig,
            cost_per_unit_change=cost_per_unit_change,
        )

        df_bt = bt_test.df
        ret = df_bt["strategy_return_net"].dropna()
        if ret.empty:
            continue

        local_equity = equity * np.exp(ret.cumsum())

        all_test_returns.append(ret)
        all_test_equity.append(local_equity)
        equity = float(local_equity.iloc[-1])

        per_window_metrics.append(
            {
                "train_start": spec.train_start,
                "train_end": spec.train_end,
                "test_start": spec.test_start,
                "test_end": spec.test_end,
                "test_sharpe": sharpe_ratio(ret),
                "test_max_dd": max_drawdown(local_equity),
                "test_net_return": float(ret.sum()),
                "test_cum_return": float(local_equity.iloc[-1] / local_equity.iloc[0] - 1.0),
                "test_trades": trade_count(df_bt["position"]),
                "test_total_cost": float(bt_test.extras.get("total_cost", 0.0)),
            }
        )

    if not all_test_returns:
        raise ValueError("No non-empty test segments in walk-forward.")

    test_returns_concat = pd.concat(all_test_returns).sort_index()
    test_equity_concat = pd.concat(all_test_equity).sort_index()

    return {
        "test_returns": test_returns_concat,
        "test_equity_curve": test_equity_concat,
        "per_window_metrics": per_window_metrics,
    }


def walk_forward_retrained_params(
    df: pd.DataFrame,
    windows: List[WindowSpec],
    candidate_windows: Sequence[int],
    candidate_z_entries: Sequence[float],
    cost_per_unit_change: float = DEFAULT_COST_PER_UNIT,
) -> Dict[str, Any]:
    """
    Walk-forward evaluation that re-selects parameters on each training window.

    Args:
        df: Full DataFrame with prices/returns.
        windows: Rolling train/test windows.
        candidate_windows: Rolling window sizes to sweep.
        candidate_z_entries: Z thresholds to sweep.
        cost_per_unit_change: Transaction cost per unit of position change.

    Returns:
        Dictionary with concatenated test returns/equity and per-window metrics.
    """

    if not windows:
        raise ValueError("windows must contain at least one window.")

    all_test_returns: List[pd.Series] = []
    all_test_equity: List[pd.Series] = []
    per_window_metrics: List[Dict[str, Any]] = []

    equity = 1.0

    for spec in windows:
        df_train = df.loc[spec.train_start : spec.train_end]
        df_test = df.loc[spec.test_start : spec.test_end]

        if df_train.empty or df_test.empty:
            continue

        w_best, z_best = best_params_on_train(
            df_train,
            windows=candidate_windows,
            z_entries=candidate_z_entries,
            cost_per_unit_change=cost_per_unit_change,
        )

        df_test_sig = rolling_zscore_signal(df, window=w_best, z_entry=z_best).loc[
            spec.test_start : spec.test_end
        ]

        bt_test = run_backtest_with_costs(
            df_test_sig,
            cost_per_unit_change=cost_per_unit_change,
        )
        df_bt = bt_test.df
        ret = df_bt["strategy_return_net"].dropna()
        if ret.empty:
            continue

        local_equity = equity * np.exp(ret.cumsum())
        all_test_returns.append(ret)
        all_test_equity.append(local_equity)
        equity = float(local_equity.iloc[-1])

        per_window_metrics.append(
            {
                "train_start": spec.train_start,
                "train_end": spec.train_end,
                "test_start": spec.test_start,
                "test_end": spec.test_end,
                "chosen_window": w_best,
                "chosen_z_entry": z_best,
                "test_sharpe": sharpe_ratio(ret),
                "test_max_dd": max_drawdown(local_equity),
                "test_net_return": float(ret.sum()),
                "test_cum_return": float(local_equity.iloc[-1] / local_equity.iloc[0] - 1.0),
                "test_trades": trade_count(df_bt["position"]),
                "test_total_cost": float(bt_test.extras.get("total_cost", 0.0)),
            }
        )

    if not all_test_returns:
        raise ValueError("No non-empty test segments in walk-forward.")

    test_returns_concat = pd.concat(all_test_returns).sort_index()
    test_equity_concat = pd.concat(all_test_equity).sort_index()

    return {
        "test_returns": test_returns_concat,
        "test_equity_curve": test_equity_concat,
        "per_window_metrics": per_window_metrics,
    }


__all__ = [
    "DEFAULT_WINDOW",
    "DEFAULT_Z_ENTRY",
    "DEFAULT_COST_PER_UNIT",
    "WindowSpec",
    "generate_rolling_windows",
    "walk_forward_fixed_params",
    "walk_forward_retrained_params",
    "best_params_on_train",
]
