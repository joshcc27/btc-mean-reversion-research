"""Preprocessing utilities for BTC/USDT OHLCV data."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Tuple

import numpy as np
import pandas as pd

from .data_loader import load_raw_ohlcv

FIVE_MINUTES: Final[str] = "5min"
OHLCV_AGG: Final[dict[str, str]] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(
    df_raw: pd.DataFrame,
    rule: str = FIVE_MINUTES,
) -> pd.DataFrame:
    """
    Resample higher-frequency OHLCV data to a coarser bar size and compute returns.

    Args:
        df_raw: Input OHLCV data indexed by timestamp.
        rule: Pandas offset alias defining the resample interval (e.g., '5min').

    Returns:
        Resampled DataFrame containing OHLCV fields plus ``return`` and ``log_return``.

    Raises:
        ValueError: If the index is not a ``DatetimeIndex``.
    """

    if not isinstance(df_raw.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame index must be a DatetimeIndex.")

    df_resampled = df_raw.resample(rule).agg(OHLCV_AGG).dropna(subset=["close"])

    close = df_resampled["close"]
    simple_return = close.pct_change()
    log_return = np.log(close / close.shift(1))
    df_resampled = df_resampled.assign(**{"return": simple_return, "log_return": log_return})
    return df_resampled


def resample_ohlcv_to_5min(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper that resamples to the default 5-minute bars."""

    return resample_ohlcv(df_raw, rule=FIVE_MINUTES)


def clean_and_resample_ohlcv(
    path: str | Path,
    resample_rule: str = FIVE_MINUTES,
    date_start: str | pd.Timestamp | None = None,
    date_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load raw OHLCV data from disk, resample it, and optionally trim to a date window.

    Args:
        path: Filesystem path pointing to a raw OHLCV CSV extracted from the exchange.
        resample_rule: Pandas offset alias used when aggregating into bars.
        date_start: Optional inclusive lower bound timestamp.
        date_end: Optional inclusive upper bound timestamp.

    Returns:
        Cleaned and resampled OHLCV data with return fields.
    """

    df_raw = load_raw_ohlcv(path)
    df = resample_ohlcv(df_raw, rule=resample_rule)

    if date_start is not None:
        df = df.loc[df.index >= pd.Timestamp(date_start, tz="UTC")]
    if date_end is not None:
        df = df.loc[df.index <= pd.Timestamp(date_end, tz="UTC")]

    return df


def load_processed_ohlcv(
    path: str | Path,
    date_start: str | pd.Timestamp | None = None,
    date_end: str | pd.Timestamp | None = None,
    price_column: str = "close",
) -> pd.DataFrame:
    """
    Load a processed OHLCV file (Parquet or CSV) and guarantee required columns.

    The helper enforces:
        * DatetimeIndex sorted in ascending order.
        * ``log_return`` column (computed from ``price_column`` if missing).
        * Optional inclusive date filtering via ``date_start`` / ``date_end``.
        * NaN rows removed from ``log_return`` to simplify downstream usage.

    Args:
        path: Filesystem path to the processed dataset.
        date_start: Optional inclusive start timestamp (string or Timestamp).
        date_end: Optional inclusive end timestamp.
        price_column: Column to use when deriving log returns.

    Returns:
        Cleaned DataFrame ready for signal generation/backtesting.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If ``price_column`` is absent while computing log returns.
    """

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")

    if data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = "timestamp" if "timestamp" in df.columns else None
        if ts_col is None:
            raise ValueError("Processed data must include a timestamp index/column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    df = df.sort_index().copy()

    if date_start is not None:
        df = df.loc[df.index >= pd.Timestamp(date_start, tz="UTC")]
    if date_end is not None:
        df = df.loc[df.index <= pd.Timestamp(date_end, tz="UTC")]

    if "log_return" not in df.columns:
        if price_column not in df.columns:
            raise KeyError(f"Expected '{price_column}' column to derive log returns.")
        df["log_return"] = np.log(df[price_column]).diff()

    df = df.dropna(subset=["log_return"]).copy()
    return df


def chronological_train_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame into chronological train/test segments.

    Args:
        df: Input DataFrame with a DatetimeIndex sorted ascending.
        train_fraction: Proportion of rows to allocate to the train slice.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If ``df`` is empty or ``train_fraction`` is not (0, 1).
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty; cannot split.")
    if len(df) < 2:
        raise ValueError("Need at least two rows to create a split.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie between 0 and 1.")

    split_idx = int(len(df) * train_fraction)
    split_idx = max(1, min(len(df) - 1, split_idx))
    split_timestamp = df.index[split_idx]

    train = df.loc[df.index <= split_timestamp].copy()
    test = df.loc[df.index > split_timestamp].copy()
    return train, test


def validate_ohlcv(
    df: pd.DataFrame,
    expected_freq: str | None = FIVE_MINUTES,
    max_gap_multiplier: float = 3.0,
) -> None:
    """
    Raise ValueError if the DataFrame fails basic OHLCV sanity checks.

    Checks performed:
        1. Index is a DatetimeIndex.
        2. Required columns (open, high, low, close, volume) are present.
        3. No duplicate timestamps.
        4. close > 0 everywhere.
        5. high >= low everywhere.
        6. volume >= 0 everywhere.
        7. No price gaps larger than max_gap_multiplier × expected bar width
           (only checked when expected_freq is provided).

    Args:
        df: Processed OHLCV DataFrame.
        expected_freq: Pandas offset alias for the expected bar frequency (e.g.,
            '5min'). Used only for gap detection. Pass None to skip.
        max_gap_multiplier: How many consecutive missing bars count as a gap.

    Raises:
        ValueError: On any failed check, with a descriptive message.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("validate_ohlcv: index must be a DatetimeIndex.")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"validate_ohlcv: missing required columns: {missing}.")

    n_dupes = df.index.duplicated().sum()
    if n_dupes:
        raise ValueError(f"validate_ohlcv: {n_dupes} duplicate timestamps found.")

    n_bad_close = (df["close"] <= 0).sum()
    if n_bad_close:
        raise ValueError(f"validate_ohlcv: {n_bad_close} rows with close <= 0.")

    n_bad_hl = (df["high"] < df["low"]).sum()
    if n_bad_hl:
        raise ValueError(f"validate_ohlcv: {n_bad_hl} rows where high < low.")

    n_neg_vol = (df["volume"] < 0).sum()
    if n_neg_vol:
        raise ValueError(f"validate_ohlcv: {n_neg_vol} rows with negative volume.")

    if expected_freq is not None and len(df) >= 2:
        expected_delta = pd.tseries.frequencies.to_offset(expected_freq)
        if expected_delta is not None:
            expected_ns = int(expected_delta.nanos)
            gaps = df.index[1:] - df.index[:-1]
            gap_ns = gaps.view("int64")
            threshold_ns = int(max_gap_multiplier * expected_ns)
            n_gaps = (gap_ns > threshold_ns).sum()
            if n_gaps:
                worst_ns = int(gap_ns.max())
                worst_td = pd.Timedelta(worst_ns)
                raise ValueError(
                    f"validate_ohlcv: {n_gaps} gap(s) exceed {max_gap_multiplier}× "
                    f"the expected bar width. Largest gap: {worst_td}."
                )


def save_processed_ohlcv(df: pd.DataFrame, path: str) -> None:
    """
    Persist the processed OHLCV data to disk in Parquet format.

    Args:
        df: DataFrame to write.
        path: Destination file path.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)


if __name__ == "__main__":
    RAW_CSV = "data/raw/BTCUSDT/btc_2024_raw.csv"
    PROCESSED_PARQUET = "data/processed/btcusdt_5m.parquet"

    raw_df = load_raw_ohlcv(RAW_CSV)
    processed_df = resample_ohlcv_to_5min(raw_df)
    save_processed_ohlcv(processed_df, PROCESSED_PARQUET)


__all__ = [
    "FIVE_MINUTES",
    "resample_ohlcv",
    "load_processed_ohlcv",
    "clean_and_resample_ohlcv",
    "resample_ohlcv_to_5min",
    "chronological_train_test_split",
    "save_processed_ohlcv",
    "validate_ohlcv",
]
