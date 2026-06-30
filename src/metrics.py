"""Performance statistics for evaluating backtests."""

from __future__ import annotations

from math import sqrt
from typing import Final, Tuple

import numpy as np
import pandas as pd

TRADING_PERIODS_PER_YEAR: Final[int] = 365 * (24 * 60 // 5)  # 5-minute bars


def scaled_sharpe(
    returns: pd.Series,
    periods_per_year: int | None = TRADING_PERIODS_PER_YEAR,
) -> float:
    """
    Compute per-period or annualised Sharpe ratio depending on ``periods_per_year``.
    """

    returns = returns.dropna()
    if returns.empty:
        return 0.0

    mean = returns.mean()
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0

    ratio = mean / std
    if periods_per_year is None:
        return float(ratio)
    return float(ratio * sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int | None = TRADING_PERIODS_PER_YEAR,
) -> float:
    """Alias for ``scaled_sharpe`` to match notebook instructions."""

    return scaled_sharpe(returns, periods_per_year)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Return the minimum drawdown (as a negative value)."""

    if equity_curve.empty:
        return 0.0

    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1.0
    return float(drawdowns.min())


def count_trades(positions: pd.Series) -> int:
    """
    Count the number of entries/exits as sign changes in the position series.
    """

    pos = positions.fillna(0.0)
    changes = pos.diff().fillna(0.0)
    return int((changes != 0.0).sum())


def trade_count(positions: pd.Series) -> int:
    """Alias for ``count_trades``."""

    return count_trades(positions)


def summarize_performance(
    strategy_returns: pd.Series,
    equity_curve: pd.Series | None = None,
    positions: pd.Series | None = None,
    periods_per_year: int | None = TRADING_PERIODS_PER_YEAR,
) -> dict[str, float]:
    """
    Produce a lightweight dictionary of metrics for convenience.
    """

    if equity_curve is None:
        equity_curve = np.exp(strategy_returns.fillna(0.0).cumsum())

    summary: dict[str, float] = {
        "sharpe": scaled_sharpe(strategy_returns, periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "total_return": float(equity_curve.iloc[-1] - 1.0) if not equity_curve.empty else 0.0,
    }

    if positions is not None:
        summary["trade_count"] = float(count_trades(positions))

    return summary


def sharpe_confidence_interval(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    confidence: float = 0.95,
    periods_per_year: int | None = TRADING_PERIODS_PER_YEAR,
) -> Tuple[float, float]:
    """
    Stationary block bootstrap confidence interval for the annualised Sharpe ratio.

    Uses non-overlapping blocks drawn with replacement. Block size defaults to the
    cube-root-of-n rule, which is standard for weakly-dependent time series.

    Args:
        returns: Per-bar strategy returns.
        n_bootstrap: Number of bootstrap resamples.
        block_size: Block length in bars. Defaults to max(1, int(n ** (1/3))).
        confidence: Width of the interval (e.g., 0.95 for 95%).
        periods_per_year: Annualisation factor.

    Returns:
        Tuple of (lower, upper) Sharpe bounds at the requested confidence level.
    """

    returns = returns.dropna()
    n = len(returns)
    if n < 2:
        return (0.0, 0.0)

    if block_size is None:
        block_size = max(1, int(n ** (1 / 3)))

    arr = returns.to_numpy()
    rng = np.random.default_rng(seed=42)
    bootstrap_sharpes = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        n_blocks = max(1, n // block_size)
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([arr[s : s + block_size] for s in starts])[:n]
        std = sample.std(ddof=1)
        if std == 0:
            bootstrap_sharpes[i] = 0.0
        else:
            sr = sample.mean() / std
            bootstrap_sharpes[i] = sr * sqrt(periods_per_year) if periods_per_year else sr

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(bootstrap_sharpes, alpha))
    upper = float(np.quantile(bootstrap_sharpes, 1.0 - alpha))
    return (lower, upper)


def buy_and_hold_metrics(
    df: pd.DataFrame,
    return_col: str = "log_return",
    periods_per_year: int | None = TRADING_PERIODS_PER_YEAR,
) -> dict[str, float]:
    """
    Compute Sharpe, max drawdown, and total return for a passive buy-and-hold position.

    Args:
        df: DataFrame containing the return series.
        return_col: Column of per-bar log returns.
        periods_per_year: Annualisation factor.

    Returns:
        Dictionary with keys: sharpe, max_drawdown, total_return.
    """

    if return_col not in df.columns:
        raise KeyError(f"Column '{return_col}' not found in DataFrame.")

    bh_returns = df[return_col].dropna()
    equity = np.exp(bh_returns.cumsum())

    return {
        "sharpe": scaled_sharpe(bh_returns, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "total_return": float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0,
    }


def returns_by_hour_of_day(strategy_returns: pd.Series) -> pd.Series:
    """
    Compute mean per-bar strategy return grouped by UTC hour of day.

    Useful for identifying intraday seasonality — whether the signal's edge
    (or lack thereof) is concentrated at specific times.

    Args:
        strategy_returns: Per-bar strategy returns with a DatetimeIndex.

    Returns:
        Series indexed 0–23 (UTC hour) with mean return for each hour.
    """

    if not isinstance(strategy_returns.index, pd.DatetimeIndex):
        raise TypeError("strategy_returns must have a DatetimeIndex.")

    return strategy_returns.groupby(strategy_returns.index.hour).mean().rename("mean_return_by_hour")


__all__ = [
    "TRADING_PERIODS_PER_YEAR",
    "scaled_sharpe",
    "sharpe_ratio",
    "max_drawdown",
    "count_trades",
    "trade_count",
    "summarize_performance",
    "sharpe_confidence_interval",
    "buy_and_hold_metrics",
    "returns_by_hour_of_day",
]
