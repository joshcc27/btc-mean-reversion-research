"""Vectorised single-asset backtest with linear transaction costs and optional overlays.

Timing convention (all quantities per bar t, log returns):
    position_t   = signal_{t-1} * vol_scale_{t-1}       (decided at the close of t-1)
    gross_t      = position_t * r_t
    cost_t       = c * |position_t - position_{t-1}|     (c = cost per unit turnover)
    net_t        = gross_t - cost_t

``c`` is a fraction of notional per unit of absolute position change, so going
from flat to fully long costs c, and flipping from -1 to +1 costs 2c. A value of
0.001 corresponds to a 10 bp fee on each traded notional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Lower bound on the rolling volatility estimate, so the scaler stays finite.
VOLATILITY_FLOOR = 1e-5


@dataclass
class BacktestResult:
    """Per-bar backtest output.

    ``df`` holds the input columns plus position, position_change,
    strategy_return_gross, transaction_cost, strategy_return_net,
    equity_curve_gross and equity_curve_net (and vol_scale when volatility
    targeting is on). ``share_of_bars_at_cap`` is the share of bars on which the
    volatility scaler sits at the position cap, or None without volatility targeting.
    """

    df: pd.DataFrame
    share_of_bars_at_cap: float | None = None

    @property
    def gross_return(self) -> float:
        """Sum of per-bar gross log returns."""
        return float(self.df["strategy_return_gross"].sum())

    @property
    def net_return(self) -> float:
        """Sum of per-bar net log returns."""
        return float(self.df["strategy_return_net"].sum())

    @property
    def turnover(self) -> float:
        """Sum of |position change| over the run, including the first entry."""
        return float(self.df["position_change"].sum())

    @property
    def total_cost(self) -> float:
        """Sum of transaction costs (log-return units)."""
        return float(self.df["transaction_cost"].sum())


def volatility_scale(
    returns: pd.Series, target_volatility: float, lookback: int, cap: float
) -> pd.Series:
    """Scaling factor ``target / sigma`` known at the start of each bar.

    sigma is the equal-weighted rolling standard deviation of the previous
    ``lookback`` returns (bars t-lookback .. t-1). The current bar's return is
    excluded, because it is the return the scaled position earns. Values are
    capped at ``cap``; warm-up bars get NaN.
    """
    sigma = returns.rolling(lookback, min_periods=lookback).std(ddof=0).clip(lower=VOLATILITY_FLOOR)
    return (target_volatility / sigma).clip(upper=cap).shift(1)


def apply_daily_stop(
    positions: pd.Series,
    returns: pd.Series,
    daily_stop_loss: float,
    cost_per_unit_turnover: float,
) -> pd.Series:
    """Flatten for the rest of the UTC day once that day's cumulative net P&L <= -stop.

    Net P&L includes transaction costs, so a day that loses only through fees still
    triggers the stop. The bar on which the threshold is crossed keeps its
    position; the position is zero from the next bar until the end of the day.
    """
    pos = positions.to_numpy(dtype=float)
    ret = returns.reindex(positions.index).fillna(0.0).to_numpy(dtype=float)
    day = positions.index.normalize().asi8

    out = np.empty_like(pos)
    prev_pos = 0.0
    day_pnl = 0.0
    stopped = False
    for i in range(len(pos)):
        if i == 0 or day[i] != day[i - 1]:
            day_pnl = 0.0
            stopped = False
        p = 0.0 if stopped else pos[i]
        day_pnl += p * ret[i] - cost_per_unit_turnover * abs(p - prev_pos)
        if day_pnl <= -daily_stop_loss:
            stopped = True
        out[i] = p
        prev_pos = p
    return pd.Series(out, index=positions.index)


def run_backtest(
    df: pd.DataFrame,
    cost_per_unit_turnover: float = 0.001,
    max_position: float = 1.0,
    target_volatility: float | None = None,
    volatility_lookback: int = 96,
    daily_stop_loss: float | None = None,
) -> BacktestResult:
    """Backtest the ``signal_raw`` column against the ``log_return`` column.

    The signal is lagged one bar, then scaled by the volatility target, capped,
    and finally passed through the daily stop.

    Args:
        df: Frame with an unlagged ``signal_raw`` column and per-bar ``log_return``.
        cost_per_unit_turnover: Cost as a fraction of notional per unit of
            |position change| (0.001 = 10 bp).
        max_position: Cap on |position|. Also caps the volatility scaler.
        target_volatility: Per-bar volatility target; None disables scaling.
        volatility_lookback: Bars in the rolling volatility estimate.
        daily_stop_loss: Flatten for the rest of the UTC day once that day's
            cumulative net log return is at or below minus this value.
    """
    returns = df["log_return"].fillna(0.0)
    positions = df["signal_raw"].fillna(0.0).shift(1).fillna(0.0)

    scale = None
    if target_volatility is not None:
        scale = volatility_scale(returns, target_volatility, volatility_lookback, max_position)
        positions = (positions * scale).fillna(0.0)

    positions = positions.clip(lower=-max_position, upper=max_position)

    if daily_stop_loss is not None:
        positions = apply_daily_stop(positions, returns, daily_stop_loss, cost_per_unit_turnover)

    out = df.copy()
    out["position"] = positions
    if scale is not None:
        out["vol_scale"] = scale
    out["position_change"] = positions.diff().fillna(positions).abs()
    out["strategy_return_gross"] = positions * returns
    out["transaction_cost"] = cost_per_unit_turnover * out["position_change"]
    out["strategy_return_net"] = out["strategy_return_gross"] - out["transaction_cost"]
    out["equity_curve_gross"] = np.exp(out["strategy_return_gross"].cumsum())
    out["equity_curve_net"] = np.exp(out["strategy_return_net"].cumsum())

    share_at_cap = None
    if scale is not None:
        share_at_cap = float((scale.dropna() >= max_position - 1e-12).mean())
    return BacktestResult(df=out, share_of_bars_at_cap=share_at_cap)
