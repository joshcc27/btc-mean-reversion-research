"""Simple return-based signal generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

DEFAULT_PRICE_COLUMN: Final[str] = "close"


@dataclass(frozen=True)
class ZScoreSignalConfig:
    """
    Configuration for the rolling z-score breakout signal.

    Attributes:
        window: Number of bars used for rolling mean/std of log returns.
        price_column: Column used to compute log returns.
        bandwidth: Z-score entry threshold (e.g., 2.0 = ±2 std).
        exit_bandwidth: Z-score exit threshold; must be strictly less than bandwidth
            when provided. The signal persists until |z| drops below this level,
            which reduces turnover by holding through shallow mean-reversion. None
            uses the simple threshold signal (exit when |z| < bandwidth).
        active_hours: UTC hours (0–23) during which the signal may fire. Bars
            outside these hours have signal forced to 0. None means trade all day.
    """

    window: int = 20
    price_column: str = DEFAULT_PRICE_COLUMN
    bandwidth: float = 1.0
    exit_bandwidth: float | None = None
    active_hours: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2.")
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        if self.exit_bandwidth is not None:
            if self.exit_bandwidth < 0:
                raise ValueError("exit_bandwidth must be non-negative.")
            if self.exit_bandwidth >= self.bandwidth:
                raise ValueError("exit_bandwidth must be strictly less than bandwidth.")
        if self.active_hours is not None:
            invalid = {h for h in self.active_hours if not (0 <= h <= 23)}
            if invalid:
                raise ValueError(f"active_hours contains invalid UTC hours: {invalid}")


def _compute_log_returns(price: pd.Series) -> pd.Series:
    """Compute log returns for the provided price series."""
    log_return = np.log(price).diff()
    return log_return.rename("log_return")


def _hysteresis_signal(
    z_arr: np.ndarray,
    bandwidth: float,
    exit_bandwidth: float,
) -> np.ndarray:
    """
    Stateful signal: enter when |z| > bandwidth, exit when |z| < exit_bandwidth.

    Compared to the simple threshold signal this holds positions through
    shallow z-score reversions, directly cutting the number of round-trips and
    therefore transaction costs.

    Args:
        z_arr: Array of z-scores (NaN propagates as flat).
        bandwidth: Entry threshold.
        exit_bandwidth: Exit threshold (< bandwidth).

    Returns:
        Integer signal array {-1, 0, +1}.
    """
    n = len(z_arr)
    out = np.zeros(n)
    pos = 0  # current position: -1, 0, or +1

    for i in range(n):
        z = z_arr[i]
        if np.isnan(z):
            pos = 0
            out[i] = 0.0
            continue

        if pos == 0:
            if z > bandwidth:
                pos = 1
            elif z < -bandwidth:
                pos = -1
        elif pos == 1:
            if z < -bandwidth:        # momentum reversal
                pos = -1
            elif z < exit_bandwidth:  # z mean-reverted to exit level
                pos = 0
        else:  # pos == -1
            if z > bandwidth:          # momentum reversal
                pos = 1
            elif z > -exit_bandwidth:  # z mean-reverted to exit level
                pos = 0

        out[i] = float(pos)

    return out


def zscore_breakout_signal(
    df: pd.DataFrame,
    config: ZScoreSignalConfig | None = None,
) -> pd.DataFrame:
    """
    Generate a breakout signal based on rolling mean/std of log returns.

    Signal definition (simple mode, exit_bandwidth=None):
        z_t = (r_t - rolling_mean_t) / rolling_std_t
        signal_raw_t = +1 if z_t > bandwidth
                       -1 if z_t < -bandwidth
                        0 otherwise

    With exit_bandwidth set, the signal is stateful: it enters when |z| crosses
    the bandwidth and holds until |z| drops below exit_bandwidth, which reduces
    turnover. See _hysteresis_signal for details.

    With active_hours set, the signal is forced to 0 for any bar whose UTC hour
    is not in the set. Intraday session filtering is applied after the main
    signal computation so the hysteresis state persists across gaps.

    Args:
        df: Price DataFrame indexed by timestamp.
        config: Optional ZScoreSignalConfig; defaults are used when None.

    Returns:
        Copy of df with rolling statistics and signal_raw added.
    """

    if config is None:
        config = ZScoreSignalConfig()

    price_col = config.price_column
    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not found in DataFrame.")

    output = df.copy()
    log_return = _compute_log_returns(output[price_col])

    rolling = log_return.rolling(config.window, min_periods=config.window)
    rolling_mean = rolling.mean().rename("rolling_mean")
    # ddof=0 for rolling z-score (population std is standard for windowed normalisation)
    rolling_std = rolling.std(ddof=0).rename("rolling_std")

    # NaN where rolling_std is zero or not yet computed — avoids false neutral signals
    # during flat/low-data periods. The backtest fills NaN positions with 0.
    z_score = pd.Series(np.nan, index=output.index, dtype=float, name="z_score")
    valid = rolling_std.notna() & (rolling_std != 0)
    z_score.loc[valid] = (
        (log_return.loc[valid] - rolling_mean.loc[valid]) / rolling_std.loc[valid]
    )

    band = config.bandwidth
    if config.exit_bandwidth is not None:
        raw_signal_arr = _hysteresis_signal(
            z_score.to_numpy(), bandwidth=band, exit_bandwidth=config.exit_bandwidth
        )
        signal_raw = pd.Series(raw_signal_arr, index=z_score.index, name="signal_raw")
    else:
        z_for_signal = z_score.fillna(0.0)
        signal_raw = pd.Series(
            np.select(
                [z_for_signal > band, z_for_signal < -band],
                [1.0, -1.0],
                default=0.0,
            ),
            index=z_score.index,
            name="signal_raw",
        )

    if config.active_hours is not None:
        if not isinstance(output.index, pd.DatetimeIndex):
            raise TypeError("active_hours filtering requires a DatetimeIndex.")
        in_session = pd.Series(output.index.hour, index=output.index).isin(config.active_hours)
        signal_raw = signal_raw.where(in_session, 0.0)

    # signal_raw is unlagged; the backtest applies the 1-bar lag via shift_positions=True.
    output = output.assign(
        log_return=log_return,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        z_score=z_score,
        signal_raw=signal_raw,
    )

    return output


def rolling_zscore_signal(
    df: pd.DataFrame,
    window: int = 20,
    z_entry: float = 1.0,
    price_column: str = DEFAULT_PRICE_COLUMN,
    exit_bandwidth: float | None = None,
    active_hours: frozenset[int] | None = None,
) -> pd.DataFrame:
    """
    Notebook-friendly wrapper for zscore_breakout_signal.

    Args:
        df: Price DataFrame indexed by timestamp.
        window: Rolling window for mean/std computation.
        z_entry: Entry threshold (bandwidth).
        price_column: Column used to derive log returns.
        exit_bandwidth: If set, hold positions until |z| drops below this level.
        active_hours: UTC hours during which the signal may fire.
    """
    config = ZScoreSignalConfig(
        window=window,
        price_column=price_column,
        bandwidth=z_entry,
        exit_bandwidth=exit_bandwidth,
        active_hours=active_hours,
    )
    return zscore_breakout_signal(df, config)


__all__ = [
    "DEFAULT_PRICE_COLUMN",
    "ZScoreSignalConfig",
    "zscore_breakout_signal",
    "rolling_zscore_signal",
]
