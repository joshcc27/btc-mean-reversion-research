"""Basic vectorized backtesting utilities (no transaction costs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """
    Container for the main artifacts of a simple backtest.

    Attributes:
        positions: Series of target exposures (aligned with the input index).
        strategy_returns: Per-bar returns (log scale) from applying the positions.
        equity_curve: Growth of 1 unit of capital implied by the strategy.
    """

    positions: pd.Series
    strategy_returns: pd.Series
    equity_curve: pd.Series

    def to_frame(self) -> pd.DataFrame:
        """Return positions, strategy returns, and equity curve as a DataFrame."""

        return pd.concat(
            [
                self.positions.rename("position"),
                self.strategy_returns.rename("strategy_return"),
                self.equity_curve.rename("equity"),
            ],
            axis=1,
        )

    @property
    def df(self) -> pd.DataFrame:
        """Alias used by notebooks (matches instructions)."""

        return self.to_frame()


@dataclass
class CostBacktestResult:
    """
    Result container for cost-aware backtests.

    Attributes:
        df: DataFrame with positions, returns, costs, and equity curve.
        gross_return: Sum of strategy returns before costs.
        net_return: Sum of strategy returns after costs.
        cum_return: Final cumulative net return relative to initial capital.
        extras: Optional metadata (e.g., total cost).
    """

    df: pd.DataFrame
    gross_return: float
    net_return: float
    cum_return: float
    extras: Optional[Dict[str, Any]] = None


def _default_returns(df: pd.DataFrame, price_column: str, return_column: str | None) -> pd.Series:
    """
    Retrieve a return series, prioritizing an explicit column then falling back to log returns.

    Args:
        df: Input DataFrame containing either prices or returns.
        price_column: Column name for prices.
        return_column: Column name for returns if already computed.

    Returns:
        Log return series aligned with the DataFrame index.
    """

    if return_column and return_column in df.columns:
        return df[return_column].fillna(0.0)

    if price_column not in df.columns:
        raise KeyError(
            f"Expected either '{return_column}' or '{price_column}' in the DataFrame."
        )

    log_returns = np.log(df[price_column]).diff().fillna(0.0)
    return log_returns.rename("log_return")


def _compute_positions(
    df: pd.DataFrame,
    signal_column: str,
    asset_returns: pd.Series,
    shift_positions: bool,
    max_position_size: float,
    target_volatility: float | None,
    volatility_lookback: int,
    volatility_floor: float,
    max_leverage: float | None,
    daily_stop_loss: float | None,
) -> pd.Series:
    """
    Apply shifting, volatility scaling, clipping, and optional stop-outs to build exposures.
    """

    if signal_column not in df.columns:
        raise KeyError(f"Signal column '{signal_column}' not found in DataFrame.")

    if max_position_size <= 0:
        raise ValueError("max_position_size must be positive.")
    if target_volatility is not None and target_volatility <= 0:
        raise ValueError("target_volatility must be positive when provided.")
    if volatility_lookback <= 1:
        raise ValueError("volatility_lookback must be at least 2.")
    if volatility_floor <= 0:
        raise ValueError("volatility_floor must be positive.")
    if max_leverage is not None and max_leverage <= 0:
        raise ValueError("max_leverage must be positive when provided.")
    if daily_stop_loss is not None and daily_stop_loss <= 0:
        raise ValueError("daily_stop_loss must be positive when provided.")

    raw_signal = df[signal_column].fillna(0.0)
    positions = raw_signal.shift(1) if shift_positions else raw_signal
    positions = positions.fillna(0.0)

    if target_volatility is not None:
        rolling_vol = (
            asset_returns.rolling(volatility_lookback, min_periods=volatility_lookback)
            .std(ddof=0)
            .clip(lower=volatility_floor)
        )
        scaling = target_volatility / rolling_vol
        if max_leverage is not None:
            scaling = scaling.clip(upper=max_leverage)
        positions = (positions * scaling).fillna(0.0)

    positions = positions.clip(lower=-max_position_size, upper=max_position_size)

    if daily_stop_loss is not None:
        positions = _apply_daily_stop_out(
            positions=positions,
            returns=asset_returns,
            daily_stop_loss=daily_stop_loss,
        )

    return positions.rename("position")


def run_basic_backtest(
    df: pd.DataFrame,
    signal_column: str = "signal_raw",
    price_column: str = "close",
    return_column: str | None = "log_return",
    shift_positions: bool = True,
    max_position_size: float = 1.0,
    target_volatility: float | None = None,
    volatility_lookback: int = 50,
    volatility_floor: float = 1e-5,
    max_leverage: float | None = None,
    daily_stop_loss: float | None = None,
) -> BacktestResult:
    """
    Run a minimal backtest using signal-derived positions.

    Steps:
        1. Positions = signal (optionally lagged by one bar to avoid lookahead).
        2. Strategy returns = positions * asset log returns.
        3. Equity = exp(cumulative sum of log strategy returns) to capture compounding.

    Args:
        df: DataFrame with prices/returns and a signal column.
        signal_column: Column containing {-1, 0, 1} signals.
        price_column: Column with prices if log returns need to be derived.
        return_column: Column with log returns (optional).
        shift_positions: If True, trades occur on the next bar (lag the signal).
        max_position_size: Absolute cap on exposures after overlays.
        target_volatility: Optional per-bar volatility target for scaling the signal.
        volatility_lookback: Window (in bars) for realised volatility.
        volatility_floor: Minimum volatility to avoid divide-by-zero blow-ups.
        max_leverage: Optional cap applied to the volatility scaler before clipping.
        daily_stop_loss: Flatten exposure for the rest of the day once losses exceed this level.

    Returns:
        BacktestResult with positions, strategy returns, and equity curve.
    """

    asset_returns = _default_returns(df, price_column, return_column)
    positions = _compute_positions(
        df=df,
        signal_column=signal_column,
        asset_returns=asset_returns,
        shift_positions=shift_positions,
        max_position_size=max_position_size,
        target_volatility=target_volatility,
        volatility_lookback=volatility_lookback,
        volatility_floor=volatility_floor,
        max_leverage=max_leverage,
        daily_stop_loss=daily_stop_loss,
    )

    strategy_returns = (positions * asset_returns).rename("strategy_return")

    equity_curve = np.exp(strategy_returns.cumsum()).rename("equity_curve")

    return BacktestResult(
        positions=positions.rename("position"),
        strategy_returns=strategy_returns,
        equity_curve=equity_curve,
    )


def run_backtest_naive(df: pd.DataFrame, **kwargs) -> BacktestResult:
    """Backward compatibility alias for instructional notebooks."""

    return run_basic_backtest(df=df, **kwargs)


def _apply_daily_stop_out(
    positions: pd.Series,
    returns: pd.Series,
    daily_stop_loss: float,
) -> pd.Series:
    """
    Flatten exposures for the remainder of a session once daily loss exceeds the threshold.

    Args:
        positions: Series of target positions (already shifted/lagged as needed).
        returns: Asset return series aligned with `positions`.
        daily_stop_loss: Loss threshold (e.g., 0.02 = -2%) measured on cumulative daily PnL.

    Returns:
        Series of adjusted positions with intra-day stop-outs applied.
    """

    if positions.empty:
        return positions

    if not isinstance(positions.index, pd.DatetimeIndex):
        raise TypeError("daily_stop_loss requires a DatetimeIndex index.")

    adjusted = positions.copy()
    grouped = positions.groupby(positions.index.normalize())
    for day_idx in grouped.groups.values():
        day_positions = positions.loc[day_idx]
        day_returns = returns.loc[day_positions.index].fillna(0.0)
        day_pnl = (day_positions * day_returns).cumsum()

        stop_trigger = day_pnl <= -daily_stop_loss
        if not stop_trigger.any():
            continue

        first_trigger_idx = np.argmax(stop_trigger.values)
        stop_slice = day_positions.index[first_trigger_idx + 1 :]
        if not stop_slice.empty:
            adjusted.loc[stop_slice] = 0.0

    return adjusted


def run_backtest_with_costs(
    df_with_signal: pd.DataFrame,
    return_col: str = "log_return",
    signal_col: str = "signal_raw",
    initial_capital: float = 1.0,
    cost_per_unit_change: float = 0.0005,
    shift_positions: bool = True,
    max_position_size: float = 1.0,
    target_volatility: float | None = None,
    volatility_lookback: int = 50,
    volatility_floor: float = 1e-5,
    max_leverage: float | None = None,
    daily_stop_loss: float | None = None,
) -> CostBacktestResult:
    """
    Run a backtest including a simple transaction cost model.

    - Positions are derived from ``signal_col`` using the same overlays as ``run_basic_backtest``.
    - Gross PnL per bar: position_t * return_t (log scale).
    - Transaction cost per bar: cost_per_unit_change * |position_t - position_{t-1}|.
    - Net PnL subtracts the cost before compounding the equity curve.
    """

    if return_col not in df_with_signal.columns:
        raise KeyError(f"Return column '{return_col}' not found in DataFrame.")
    if signal_col not in df_with_signal.columns:
        raise KeyError(f"Signal column '{signal_col}' not found in DataFrame.")

    df = df_with_signal.copy()
    returns = df[return_col].fillna(0.0)
    positions = _compute_positions(
        df=df,
        signal_column=signal_col,
        asset_returns=returns,
        shift_positions=shift_positions,
        max_position_size=max_position_size,
        target_volatility=target_volatility,
        volatility_lookback=volatility_lookback,
        volatility_floor=volatility_floor,
        max_leverage=max_leverage,
        daily_stop_loss=daily_stop_loss,
    )

    df["position"] = positions
    df["strategy_return_gross"] = positions * returns

    position_change = positions.diff().fillna(positions).abs()
    df["position_change"] = position_change
    df["transaction_cost"] = cost_per_unit_change * position_change

    df["strategy_return_net"] = df["strategy_return_gross"] - df["transaction_cost"]
    df["equity_curve_net"] = initial_capital * np.exp(df["strategy_return_net"].cumsum())

    gross_return = float(df["strategy_return_gross"].sum())
    net_return = float(df["strategy_return_net"].sum())
    cum_return = float(df["equity_curve_net"].iloc[-1] / initial_capital - 1.0)

    costs = df.loc[df["transaction_cost"] > 0, "transaction_cost"]
    extras = {
        "total_cost": float(df["transaction_cost"].sum()),
        "avg_cost_per_trade": float(costs.mean()) if not costs.empty else 0.0,
    }

    return CostBacktestResult(
        df=df,
        gross_return=gross_return,
        net_return=net_return,
        cum_return=cum_return,
        extras=extras,
    )


__all__ = [
    "BacktestResult",
    "run_basic_backtest",
    "run_backtest_naive",
    "CostBacktestResult",
    "run_backtest_with_costs",
]
