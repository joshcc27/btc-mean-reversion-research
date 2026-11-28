"""Rolling walk-forward evaluation.

Each window has a training block followed by a test block. Parameters are
chosen from the training block only (or fixed in advance), the signal is then
computed on data up to the end of the test block, and only the test block is
backtested. Test blocks do not overlap, so their returns can be concatenated
into one out-of-sample series.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .metrics import count_entries, max_drawdown, sharpe_ratio
from .param_search import grid_search
from .signals import rolling_zscore_signal

DEFAULT_COST_PER_UNIT: float = 0.001

ParamSelector = Callable[[pd.DataFrame], tuple[int, float]]


@dataclass(frozen=True)
class WindowSpec:
    """One train/test window (inclusive bounds)."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_rolling_windows(
    df: pd.DataFrame, train_days: int = 60, test_days: int = 10
) -> list[WindowSpec]:
    """Train blocks of ``train_days`` followed by test blocks of ``test_days``, rolled by ``test_days``."""
    start = df.index.min().normalize()
    end = df.index.max().normalize()
    train_delta, test_delta = pd.Timedelta(days=train_days), pd.Timedelta(days=test_days)
    eps = pd.Timedelta(seconds=1)

    windows: list[WindowSpec] = []
    train_start = start
    while True:
        test_start = train_start + train_delta
        if test_start > end:
            break
        windows.append(
            WindowSpec(train_start, test_start - eps, test_start, test_start + test_delta - eps)
        )
        train_start += test_delta
        if train_start + train_delta > end:
            break
    return windows


def best_params_on_train(
    df_train: pd.DataFrame,
    windows: Sequence[int],
    z_entries: Sequence[float],
    cost_per_unit_turnover: float = DEFAULT_COST_PER_UNIT,
) -> tuple[int, float]:
    """Return the (window, z_entry) with the highest net Sharpe on ``df_train``."""
    grid = grid_search(
        df_train,
        {"train": df_train.index},
        windows=windows,
        z_entries=z_entries,
        cost_per_unit_turnover=cost_per_unit_turnover,
    )
    best = grid.iloc[0]
    return int(best["window"]), float(best["z_entry"])


def run_walk_forward(
    df: pd.DataFrame,
    windows: Sequence[WindowSpec],
    select_params: ParamSelector,
    cost_per_unit_turnover: float = DEFAULT_COST_PER_UNIT,
) -> dict[str, Any]:
    """Run a walk-forward test with a caller-supplied parameter selector.

    ``select_params`` receives only the training block of each window.

    Returns:
        Dict with ``test_returns`` (concatenated net returns), ``test_equity_curve``
        and ``per_window_metrics`` (list of dicts).
    """
    returns: list[pd.Series] = []
    equities: list[pd.Series] = []
    rows: list[dict[str, Any]] = []
    equity = 1.0

    for spec in windows:
        w, z = select_params(df.loc[spec.train_start : spec.train_end])
        sig = rolling_zscore_signal(df.loc[: spec.test_end], window=w, z_entry=z)
        bt = run_backtest(
            sig.loc[spec.test_start : spec.test_end], cost_per_unit_turnover=cost_per_unit_turnover
        )
        ret = bt.df["strategy_return_net"]
        local_equity = equity * np.exp(ret.cumsum())
        equity = float(local_equity.iloc[-1])
        returns.append(ret)
        equities.append(local_equity)
        rows.append(
            {
                "train_start": spec.train_start,
                "train_end": spec.train_end,
                "test_start": spec.test_start,
                "test_end": spec.test_end,
                "chosen_window": w,
                "chosen_z_entry": z,
                "test_sharpe_net": sharpe_ratio(ret),
                "test_gross_log_return": bt.gross_return,
                "test_net_log_return": bt.net_return,
                "test_turnover": bt.turnover,
                "test_entries": count_entries(bt.df["position"]),
                "test_max_dd": max_drawdown(local_equity),
            }
        )

    return {
        "test_returns": pd.concat(returns).sort_index(),
        "test_equity_curve": pd.concat(equities).sort_index(),
        "per_window_metrics": rows,
    }


def walk_forward_fixed_params(
    df: pd.DataFrame,
    windows: Sequence[WindowSpec],
    window_param: int = 20,
    z_entry_param: float = 1.5,
    cost_per_unit_turnover: float = DEFAULT_COST_PER_UNIT,
) -> dict[str, Any]:
    """Walk-forward with the same parameters in every window."""
    return run_walk_forward(
        df, windows, lambda _train: (window_param, z_entry_param), cost_per_unit_turnover
    )


def walk_forward_retrained_params(
    df: pd.DataFrame,
    windows: Sequence[WindowSpec],
    candidate_windows: Sequence[int],
    candidate_z_entries: Sequence[float],
    cost_per_unit_turnover: float = DEFAULT_COST_PER_UNIT,
) -> dict[str, Any]:
    """Walk-forward that re-selects (window, z_entry) by net Sharpe on each training block."""
    selector = partial(
        best_params_on_train,
        windows=candidate_windows,
        z_entries=candidate_z_entries,
        cost_per_unit_turnover=cost_per_unit_turnover,
    )
    return run_walk_forward(df, windows, selector, cost_per_unit_turnover)
