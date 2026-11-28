"""Performance statistics for backtests on 5-minute bars."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .backtest import BacktestResult

BARS_PER_DAY: Final[int] = 24 * 60 // 5
BARS_PER_YEAR: Final[int] = 365 * BARS_PER_DAY  # crypto trades every day: 105,120 bars


def sharpe_ratio(returns: pd.Series, periods_per_year: int | None = BARS_PER_YEAR) -> float:
    """Mean over sample std (ddof=1) of per-bar returns, times sqrt(periods_per_year).

    Pass ``periods_per_year=None`` for the per-bar ratio. Returns 0.0 for an
    empty or constant series.
    """
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    std = returns.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return 0.0
    ratio = returns.mean() / std
    return float(ratio if periods_per_year is None else ratio * sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline of an equity curve, as a negative fraction."""
    if equity_curve.empty:
        return 0.0
    return float((equity_curve / equity_curve.cummax() - 1.0).min())


def turnover(positions: pd.Series) -> float:
    """Sum of |position_t - position_{t-1}|, starting from flat.

    This is the quantity transaction costs are charged on: entering a full
    position adds 1, flipping from -1 to +1 adds 2, and resizing from 1.0 to
    0.8 adds 0.2.
    """
    pos = positions.fillna(0.0)
    return float(pos.diff().fillna(pos).abs().sum())


def count_entries(positions: pd.Series) -> int:
    """Number of bars on which a new directional position starts.

    Counts moves from flat to long/short and direct flips between long and
    short. Resizing within the same direction and exits to flat are not counted.
    """
    sign = np.sign(positions.fillna(0.0))
    prev = sign.shift(1).fillna(0.0)
    return int(((sign != 0) & (sign != prev)).sum())


def break_even_cost(gross_log_return: float, total_turnover: float) -> float:
    """Cost per unit turnover (fraction of notional) at which net return is zero.

    Net = gross - c * turnover, so c* = gross / turnover. Negative when the gross
    return is already negative. NaN if there was no trading.
    """
    if total_turnover <= 0:
        return float("nan")
    return gross_log_return / total_turnover


def sharpe_confidence_interval(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    confidence: float = 0.95,
    periods_per_year: int | None = BARS_PER_YEAR,
    seed: int = 42,
) -> tuple[float, float]:
    """Moving-block bootstrap percentile interval for the annualised Sharpe ratio.

    Each resample concatenates blocks of ``block_size`` consecutive bars whose start
    points are drawn uniformly with replacement from all n - block_size + 1
    positions (blocks may overlap), then truncates to n bars. Blocks keep
    short-range autocorrelation and volatility clustering inside each block.
    ``block_size`` defaults to the n^(1/3) rule, and ``periods_per_year=None``
    gives the per-bar Sharpe ratio.
    """
    arr = returns.dropna().to_numpy()
    n = len(arr)
    if n < 2:
        return (0.0, 0.0)
    block = block_size or max(1, int(n ** (1 / 3)))
    n_blocks = -(-n // block)  # ceiling division
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    scale = sqrt(periods_per_year) if periods_per_year else 1.0

    sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = arr[(starts[:, None] + offsets).ravel()[:n]]
        std = sample.std(ddof=1)
        sharpes[i] = 0.0 if std == 0 else sample.mean() / std * scale

    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(sharpes, alpha)), float(np.quantile(sharpes, 1.0 - alpha)))


def buy_and_hold_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Sharpe, max drawdown, total log return and total simple return of holding the asset."""
    r = df["log_return"].dropna()
    equity = np.exp(r.cumsum())
    return {
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(equity),
        "log_return": float(r.sum()),
        "total_return": float(equity.iloc[-1] - 1.0),
    }


def returns_by_hour_of_day(strategy_returns: pd.Series) -> pd.Series:
    """Mean per-bar return grouped by UTC hour (index 0-23)."""
    return strategy_returns.groupby(strategy_returns.index.hour).mean().rename("mean_return")


def summarize_backtest(result: BacktestResult, with_ci: bool = False) -> dict[str, float]:
    """Headline statistics for one backtest run.

    Returns gross and net log return, turnover, number of entries, break-even
    cost in basis points per unit turnover, gross and net annualised Sharpe,
    and net max drawdown. With ``with_ci`` it adds moving-block bootstrap 95%
    intervals for both Sharpe ratios.
    """
    df = result.df
    gross, net = result.gross_return, result.net_return
    to = result.turnover
    out = {
        "gross_log_return": gross,
        "net_log_return": net,
        "turnover": to,
        "entries": float(count_entries(df["position"])),
        "break_even_bp": break_even_cost(gross, to) * 1e4,
        "sharpe_gross": sharpe_ratio(df["strategy_return_gross"]),
        "sharpe_net": sharpe_ratio(df["strategy_return_net"]),
        "max_drawdown_net": max_drawdown(df["equity_curve_net"]),
    }
    if result.share_of_bars_at_cap is not None:
        out["share_of_bars_at_cap"] = result.share_of_bars_at_cap
    if with_ci:
        out["sharpe_gross_ci_low"], out["sharpe_gross_ci_high"] = sharpe_confidence_interval(
            df["strategy_return_gross"]
        )
        out["sharpe_net_ci_low"], out["sharpe_net_ci_high"] = sharpe_confidence_interval(
            df["strategy_return_net"]
        )
    return out
