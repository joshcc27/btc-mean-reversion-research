"""Sanity checks for OHLCV frames."""

from __future__ import annotations

import pandas as pd


def validate_ohlcv(
    df: pd.DataFrame,
    expected_freq: str | None = "5min",
    max_gap_multiplier: float = 3.0,
) -> None:
    """Raise ValueError if the DataFrame fails basic OHLCV sanity checks.

    Checks:
        1. Index is a DatetimeIndex, sorted ascending, without duplicates.
        2. Columns open, high, low, close, volume are present.
        3. close > 0, high >= low, volume >= 0 on every row.
        4. If ``expected_freq`` is given, no gap between consecutive bars exceeds
           ``max_gap_multiplier`` times the bar width.

    Called by ``load_raw_ohlcv`` (without the gap check) and by
    ``build_dataset`` on the resampled 5-minute bars.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("validate_ohlcv: index must be a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("validate_ohlcv: index is not sorted ascending.")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"validate_ohlcv: missing required columns: {sorted(missing)}.")

    n_dupes = int(df.index.duplicated().sum())
    if n_dupes:
        raise ValueError(f"validate_ohlcv: {n_dupes} duplicate timestamps found.")

    n_bad_close = int((df["close"] <= 0).sum())
    if n_bad_close:
        raise ValueError(f"validate_ohlcv: {n_bad_close} rows with close <= 0.")

    n_bad_hl = int((df["high"] < df["low"]).sum())
    if n_bad_hl:
        raise ValueError(f"validate_ohlcv: {n_bad_hl} rows where high < low.")

    n_neg_vol = int((df["volume"] < 0).sum())
    if n_neg_vol:
        raise ValueError(f"validate_ohlcv: {n_neg_vol} rows with negative volume.")

    if expected_freq is not None and len(df) >= 2:
        bar = pd.Timedelta(pd.tseries.frequencies.to_offset(expected_freq))
        gaps = df.index.to_series().diff().dropna()
        too_long = gaps > max_gap_multiplier * bar
        if too_long.any():
            raise ValueError(
                f"validate_ohlcv: {int(too_long.sum())} gap(s) exceed {max_gap_multiplier}x "
                f"the expected bar width. Largest gap: {gaps.max()} ending {gaps.idxmax()}."
            )


__all__ = ["validate_ohlcv"]
