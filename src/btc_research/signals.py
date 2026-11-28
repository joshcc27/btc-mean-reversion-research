"""Rolling z-score signal on single-bar log returns.

The z-score of bar t compares that bar's log return with the mean and standard
deviation of the last ``window`` returns (bars t-window+1 .. t), so it is known
at the close of bar t. The backtest trades on the next bar.

The mean-reversion signal goes long after an unusually large down move and short
after an unusually large up move: +1 if z < -z_entry, -1 if z > z_entry, 0
otherwise. The momentum signal is its exact negative, so with identical turnover
its gross P&L is the negative of the mean-reversion gross P&L.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Direction = Literal["mean_reversion", "momentum"]


def _threshold_signal(z: np.ndarray, z_entry: float) -> np.ndarray:
    """Mean-reversion signal without memory: long below -z_entry, short above +z_entry."""
    z = np.nan_to_num(z, nan=0.0)
    return np.select([z < -z_entry, z > z_entry], [1.0, -1.0], default=0.0)


def _hysteresis_signal(z: np.ndarray, z_entry: float, exit_bandwidth: float) -> np.ndarray:
    """Mean-reversion signal with an exit band.

    Flat:  enter long if z < -z_entry, short if z > +z_entry.
    Long:  flip short if z > +z_entry; otherwise exit once z > -exit_bandwidth.
    Short: flip long if z < -z_entry; otherwise exit once z < +exit_bandwidth.
    A NaN z-score (warm-up or zero variance) forces the position flat.
    """
    out = np.zeros(len(z))
    pos = 0.0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            pos = 0.0
        elif pos == 0.0:
            if zi < -z_entry:
                pos = 1.0
            elif zi > z_entry:
                pos = -1.0
        elif pos == 1.0:
            if zi > z_entry:
                pos = -1.0
            elif zi > -exit_bandwidth:
                pos = 0.0
        else:
            if zi < -z_entry:
                pos = 1.0
            elif zi < exit_bandwidth:
                pos = 0.0
        out[i] = pos
    return out


def rolling_zscore_signal(
    df: pd.DataFrame,
    window: int = 20,
    z_entry: float = 1.0,
    exit_bandwidth: float | None = None,
    active_hours: frozenset[int] | None = None,
    direction: Direction = "mean_reversion",
) -> pd.DataFrame:
    """Add ``log_return``, ``rolling_mean``, ``rolling_std``, ``z_score`` and ``signal_raw``.

    ``signal_raw`` is not lagged; the backtest applies the one-bar lag. Compute the
    signal on the full series and split afterwards: every value depends only on
    past and current bars, and splitting first would leave the first ``window``
    bars of each later split without a signal.

    Args:
        df: Frame with a ``close`` column and a DatetimeIndex.
        window: Bars in the rolling mean and standard deviation of log returns.
        z_entry: Entry threshold in standard deviations.
        exit_bandwidth: Optional exit threshold with 0 <= exit_bandwidth < z_entry.
            A position is then held until the z-score relaxes past the exit level,
            instead of closing as soon as |z| falls back inside the entry band.
        active_hours: Optional set of UTC hours in which the signal may be non-zero.
    """
    if window < 2:
        raise ValueError("window must be at least 2.")
    if z_entry <= 0:
        raise ValueError("z_entry must be positive.")
    if exit_bandwidth is not None and not 0 <= exit_bandwidth < z_entry:
        raise ValueError("exit_bandwidth must satisfy 0 <= exit_bandwidth < z_entry.")
    if active_hours is not None and not all(0 <= h <= 23 for h in active_hours):
        raise ValueError(f"active_hours contains invalid UTC hours: {sorted(active_hours)}")
    if direction not in ("mean_reversion", "momentum"):
        raise ValueError(f"Unknown direction: {direction!r}")

    log_return = np.log(df["close"]).diff()
    rolling = log_return.rolling(window, min_periods=window)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std(ddof=0)

    # NaN (not 0) where the std is undefined or zero, so flat periods stay flat.
    z_score = ((log_return - rolling_mean) / rolling_std.where(rolling_std > 0)).astype(float)

    z = z_score.to_numpy()
    if exit_bandwidth is None:
        signal = _threshold_signal(z, z_entry)
    else:
        signal = _hysteresis_signal(z, z_entry, exit_bandwidth)
    if direction == "momentum":
        signal = -signal
    signal_raw = pd.Series(signal + 0.0, index=df.index)  # + 0.0 turns -0.0 into 0.0

    if active_hours is not None:
        signal_raw = signal_raw.where(df.index.hour.isin(list(active_hours)), 0.0)

    return df.assign(
        log_return=log_return,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        z_score=z_score,
        signal_raw=signal_raw,
    )
