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

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Per-bar backtest output plus run metadata.

    ``df`` holds the input columns plus position, position_change,
    strategy_return_gross, transaction_cost, strategy_return_net,
    equity_curve_gross and equity_curve_net (and vol_scale when volatility
    targeting is on).
    """

    df: pd.DataFrame
    extras: dict[str, Any] = field(default_factory=dict)

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


def _validate_inputs(
    max_position: float,
    target_volatility: float | None,
    volatility_lookback: int,
    volatility_floor: float,
    daily_stop_loss: float | None,
    cost_per_unit_turnover: float,
) -> None:
    if max_position <= 0:
        raise ValueError("max_position must be positive.")
    if target_volatility is not None and target_volatility <= 0:
        raise ValueError("target_volatility must be positive when provided.")
    if volatility_lookback < 2:
        raise ValueError("volatility_lookback must be at least 2.")
    if volatility_floor <= 0:
        raise ValueError("volatility_floor must be positive.")
    if daily_stop_loss is not None and daily_stop_loss <= 0:
        raise ValueError("daily_stop_loss must be positive when provided.")
    if cost_per_unit_turnover < 0:
        raise ValueError("cost_per_unit_turnover must be non-negative.")


def volatility_scale(
    returns: pd.Series,
    target_volatility: float,
    lookback: int,
    floor: float,
    cap: float,
) -> pd.Series:
    """Scaling factor ``target / sigma`` known at the start of each bar.

    sigma is the equal-weighted rolling standard deviation of the previous
    ``lookback`` returns (bars t-lookback .. t-1). The current bar's return is
    excluded, because it is the return the scaled position earns. Values are
    capped at ``cap``; warm-up bars get NaN.
    """
    sigma = returns.rolling(lookback, min_periods=lookback).std(ddof=0).clip(lower=floor)
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
    if positions.empty:
        return positions
    if not isinstance(positions.index, pd.DatetimeIndex):
        raise TypeError("daily_stop_loss requires a DatetimeIndex.")

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


def compute_positions(
    df: pd.DataFrame,
    signal_column: str = "signal_raw",
    return_column: str = "log_return",
    shift_positions: bool = True,
    max_position: float = 1.0,
    target_volatility: float | None = None,
    volatility_lookback: int = 96,
    volatility_floor: float = 1e-5,
    daily_stop_loss: float | None = None,
    cost_per_unit_turnover: float = 0.0,
) -> tuple[pd.Series, pd.Series | None]:
    """Turn a signal into positions: lag, volatility scaling, cap, then daily stop.

    Returns:
        (positions, vol_scale) where vol_scale is None without volatility targeting.
    """
    _validate_inputs(
        max_position,
        target_volatility,
        volatility_lookback,
        volatility_floor,
        daily_stop_loss,
        cost_per_unit_turnover,
    )
    if signal_column not in df.columns:
        raise KeyError(f"Signal column '{signal_column}' not found in DataFrame.")
    if return_column not in df.columns:
        raise KeyError(f"Return column '{return_column}' not found in DataFrame.")

    returns = df[return_column].fillna(0.0)
    signal = df[signal_column].fillna(0.0)
    positions = (signal.shift(1) if shift_positions else signal).fillna(0.0)

    scale = None
    if target_volatility is not None:
        scale = volatility_scale(
            returns, target_volatility, volatility_lookback, volatility_floor, max_position
        )
        positions = (positions * scale).fillna(0.0)

    positions = positions.clip(lower=-max_position, upper=max_position)

    if daily_stop_loss is not None:
        positions = apply_daily_stop(positions, returns, daily_stop_loss, cost_per_unit_turnover)

    return positions.rename("position"), scale


def run_backtest(
    df: pd.DataFrame,
    cost_per_unit_turnover: float = 0.001,
    signal_column: str = "signal_raw",
    return_column: str = "log_return",
    shift_positions: bool = True,
    max_position: float = 1.0,
    target_volatility: float | None = None,
    volatility_lookback: int = 96,
    volatility_floor: float = 1e-5,
    daily_stop_loss: float | None = None,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """Backtest a signal column with linear costs and optional risk overlays.

    Args:
        df: Frame with a signal column and a per-bar log-return column.
        cost_per_unit_turnover: Cost as a fraction of notional per unit of
            |position change| (0.001 = 10 bp).
        signal_column: Column holding the unlagged signal.
        return_column: Column holding per-bar log returns.
        shift_positions: Trade on the bar after the signal (True for any real test).
        max_position: Cap on |position|. Also caps the volatility scaler.
        target_volatility: Per-bar volatility target; None disables scaling.
        volatility_lookback: Bars in the rolling volatility estimate.
        volatility_floor: Lower bound on the volatility estimate.
        daily_stop_loss: Flatten for the rest of the UTC day once that day's
            cumulative net log return is at or below minus this value.
        initial_capital: Starting equity.
    """
    positions, scale = compute_positions(
        df,
        signal_column=signal_column,
        return_column=return_column,
        shift_positions=shift_positions,
        max_position=max_position,
        target_volatility=target_volatility,
        volatility_lookback=volatility_lookback,
        volatility_floor=volatility_floor,
        daily_stop_loss=daily_stop_loss,
        cost_per_unit_turnover=cost_per_unit_turnover,
    )
    returns = df[return_column].fillna(0.0)

    out = df.copy()
    out["position"] = positions
    if scale is not None:
        out["vol_scale"] = scale
    out["position_change"] = positions.diff().fillna(positions).abs()
    out["strategy_return_gross"] = positions * returns
    out["transaction_cost"] = cost_per_unit_turnover * out["position_change"]
    out["strategy_return_net"] = out["strategy_return_gross"] - out["transaction_cost"]
    out["equity_curve_gross"] = initial_capital * np.exp(out["strategy_return_gross"].cumsum())
    out["equity_curve_net"] = initial_capital * np.exp(out["strategy_return_net"].cumsum())

    extras: dict[str, Any] = {"cost_per_unit_turnover": cost_per_unit_turnover}
    if scale is not None:
        valid = scale.dropna()
        extras["share_of_bars_at_cap"] = (
            float((valid >= max_position - 1e-12).mean()) if not valid.empty else float("nan")
        )
    return BacktestResult(df=out, extras=extras)


__all__ = [
    "BacktestResult",
    "apply_daily_stop",
    "compute_positions",
    "run_backtest",
    "volatility_scale",
]
