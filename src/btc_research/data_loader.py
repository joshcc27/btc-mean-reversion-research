"""Load Binance spot kline files (the public data.binance.vision format) into pandas."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .validation import validate_ohlcv

# Column layout of the official kline CSVs. The files have no header row.
KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# Epoch timestamps above this are microseconds (Binance switched spot files from
# milliseconds to microseconds starting with 2025 data). 1e14 ms would be year 5138.
_MICROSECOND_THRESHOLD = 10**14


def epoch_to_datetime(values: pd.Series) -> pd.DatetimeIndex:
    """Convert epoch integers in milliseconds or microseconds to a UTC DatetimeIndex.

    The unit is inferred per value, so a frame that mixes 2024 (ms) and 2025 (µs)
    files converts correctly.
    """
    raw = pd.to_numeric(values, errors="raise").astype("int64")
    as_us = raw.where(raw >= _MICROSECOND_THRESHOLD, raw * 1000)
    return pd.DatetimeIndex(pd.to_datetime(as_us, unit="us", utc=True), name="timestamp")


def _read_kline_file(path: Path) -> pd.DataFrame:
    """Read one kline CSV (plain or zipped). Tolerates an optional header row."""
    frame = pd.read_csv(path, header=None, dtype=str)
    first = str(frame.iloc[0, 0]).strip()
    if not first.isdigit():  # a header line, e.g. "open_time,open,..."
        frame = frame.iloc[1:]
    if frame.shape[1] != len(KLINE_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(KLINE_COLUMNS)} columns, found {frame.shape[1]}."
        )
    frame.columns = list(KLINE_COLUMNS)
    return frame


def find_kline_files(raw_dir: str | Path, symbol: str = "BTCUSDT", interval: str = "1m") -> list[Path]:
    """Return monthly kline archives/CSVs in ``raw_dir``, sorted by name (i.e. by month)."""
    raw_path = Path(raw_dir)
    files = sorted(
        p
        for p in raw_path.glob(f"{symbol}-{interval}-*")
        if p.suffix.lower() in {".zip", ".csv"}
    )
    if not files:
        raise FileNotFoundError(
            f"No {symbol}-{interval}-*.zip/.csv files in {raw_path}. "
            "Run `uv run python scripts/download_data.py` first."
        )
    return files


def load_raw_ohlcv(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Load Binance kline files into an OHLCV DataFrame indexed by UTC open time.

    Args:
        paths: A directory containing monthly ``<SYMBOL>-1m-YYYY-MM.zip`` files, a
            single file, or an iterable of files. Files may be zipped or plain CSV.

    Returns:
        DataFrame with float columns open, high, low, close, volume, sorted by
        time with duplicate timestamps removed. It passes ``validate_ohlcv``
        (gap check disabled, since exchange outages leave short holes in 1m data).
    """

    if isinstance(paths, (str, Path)):
        path = Path(paths)
        files = find_kline_files(path) if path.is_dir() else [path]
    else:
        files = [Path(p) for p in paths]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Kline file not found: {f}")

    raw = pd.concat([_read_kline_file(f) for f in files], ignore_index=True)
    index = epoch_to_datetime(raw["open_time"])
    df = raw.loc[:, list(OHLCV_COLUMNS)].astype(float).set_index(index)
    df = df.sort_index(kind="mergesort")
    df = df[~df.index.duplicated(keep="first")]

    if df.empty:
        raise ValueError("No rows loaded; check the input files.")
    validate_ohlcv(df, expected_freq=None)
    return df


__all__ = [
    "KLINE_COLUMNS",
    "OHLCV_COLUMNS",
    "epoch_to_datetime",
    "find_kline_files",
    "load_raw_ohlcv",
]
