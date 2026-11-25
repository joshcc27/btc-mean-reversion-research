"""Rolling z-score signal on single-bar log returns.

The z-score of bar t compares that bar's log return with the mean and standard
deviation of the last ``window`` returns (bars t-window+1 .. t). It uses only
information available at the close of bar t. The backtest then trades on the
next bar.

Direction:
    mean_reversion (the hypothesis under test): after an unusually large down
        move, go long; after an unusually large up move, go short.
            +1 if z < -bandwidth,  -1 if z > +bandwidth,  0 otherwise.
    momentum: the exact negative of the mean-reversion signal. It serves as the
        mirror test: with identical turnover, its gross P&L is the negative of the
        mean-reversion gross P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

DEFAULT_PRICE_COLUMN: Final[str] = "close"
Direction = Literal["mean_reversion", "momentum"]
DIRECTIONS: Final[tuple[Direction, ...]] = ("mean_reversion", "momentum")


@dataclass(frozen=True)
class ZScoreSignalConfig:
    """Parameters of the z-score signal.

    Attributes:
        window: Bars in the rolling mean/std of log returns.
        bandwidth: Entry threshold in standard deviations.
        exit_bandwidth: Optional exit threshold (0 <= exit < bandwidth). When set,
            a position is held until the z-score relaxes past the exit level instead
            of being closed as soon as |z| falls back inside the entry band.
        active_hours: Optional set of UTC hours in which the signal may be non-zero.
        direction: "mean_reversion" (default) or "momentum".
        price_column: Column used to compute log returns.
    """

    window: int = 20
    bandwidth: float = 1.0
    exit_bandwidth: float | None = None
    active_hours: frozenset[int] | None = None
    direction: Direction = "mean_reversion"
    price_column: str = DEFAULT_PRICE_COLUMN

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2.")
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        if self.exit_bandwidth is not None and not 0 <= self.exit_bandwidth < self.bandwidth:
            raise ValueError("exit_bandwidth must satisfy 0 <= exit_bandwidth < bandwidth.")
        if self.active_hours is not None:
            invalid = {h for h in self.active_hours if not 0 <= h <= 23}
            if invalid:
                raise ValueError(f"active_hours contains invalid UTC hours: {invalid}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}.")


def _threshold_signal(z: np.ndarray, bandwidth: float) -> np.ndarray:
    """Mean-reversion signal without memory: long below -band, short above +band."""
    z = np.nan_to_num(z, nan=0.0)
    return np.select([z < -bandwidth, z > bandwidth], [1.0, -1.0], default=0.0)


def _hysteresis_signal(z: np.ndarray, bandwidth: float, exit_bandwidth: float) -> np.ndarray:
    """Mean-reversion signal with an exit band.

    Flat:  enter long if z < -bandwidth, short if z > +bandwidth.
    Long:  flip short if z > +bandwidth; otherwise exit once z > -exit_bandwidth.
    Short: flip long if z < -bandwidth; otherwise exit once z < +exit_bandwidth.
    A NaN z-score (warm-up or zero variance) forces the position flat.
    """
    out = np.zeros(len(z))
    pos = 0.0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            pos = 0.0
        elif pos == 0.0:
            if zi < -bandwidth:
                pos = 1.0
            elif zi > bandwidth:
                pos = -1.0
        elif pos == 1.0:
            if zi > bandwidth:
                pos = -1.0
            elif zi > -exit_bandwidth:
                pos = 0.0
        else:
            if zi < -bandwidth:
                pos = 1.0
            elif zi < exit_bandwidth:
                pos = 0.0
        out[i] = pos
    return out


def zscore_signal(df: pd.DataFrame, config: ZScoreSignalConfig | None = None) -> pd.DataFrame:
    """Add ``log_return``, ``rolling_mean``, ``rolling_std``, ``z_score`` and ``signal_raw``.

    ``signal_raw`` is not lagged; the backtest applies the one-bar lag. Compute the
    signal on the full series and split afterwards: every value depends only on
    past and current bars, and splitting first would leave the first ``window``
    bars of each later split without a signal.
    """
    config = config or ZScoreSignalConfig()
    if config.price_column not in df.columns:
        raise KeyError(f"Price column '{config.price_column}' not found in DataFrame.")

    out = df.copy()
    log_return = np.log(out[config.price_column]).diff()
    rolling = log_return.rolling(config.window, min_periods=config.window)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std(ddof=0)

    # NaN (not 0) where the std is undefined or zero, so flat periods stay flat.
    z_score = ((log_return - rolling_mean) / rolling_std.where(rolling_std > 0)).astype(float)

    z = z_score.to_numpy()
    if config.exit_bandwidth is None:
        signal = _threshold_signal(z, config.bandwidth)
    else:
        signal = _hysteresis_signal(z, config.bandwidth, config.exit_bandwidth)
    if config.direction == "momentum":
        signal = -signal
    signal_raw = pd.Series(signal + 0.0, index=out.index)  # + 0.0 turns -0.0 into 0.0

    if config.active_hours is not None:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise TypeError("active_hours filtering requires a DatetimeIndex.")
        signal_raw = signal_raw.where(out.index.hour.isin(list(config.active_hours)), 0.0)

    return out.assign(
        log_return=log_return,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        z_score=z_score,
        signal_raw=signal_raw,
    )


def rolling_zscore_signal(
    df: pd.DataFrame,
    window: int = 20,
    z_entry: float = 1.0,
    exit_bandwidth: float | None = None,
    active_hours: frozenset[int] | None = None,
    direction: Direction = "mean_reversion",
    price_column: str = DEFAULT_PRICE_COLUMN,
) -> pd.DataFrame:
    """Keyword-argument form of ``zscore_signal``."""
    config = ZScoreSignalConfig(
        window=window,
        bandwidth=z_entry,
        exit_bandwidth=exit_bandwidth,
        active_hours=active_hours,
        direction=direction,
        price_column=price_column,
    )
    return zscore_signal(df, config)


__all__ = [
    "DEFAULT_PRICE_COLUMN",
    "DIRECTIONS",
    "Direction",
    "ZScoreSignalConfig",
    "rolling_zscore_signal",
    "zscore_signal",
]
