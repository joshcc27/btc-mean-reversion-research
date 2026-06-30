"""Utilities for loading BTC/USDT OHLCV data from disk."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

TIMESTAMP_CANDIDATES: tuple[str, ...] = (
    "timestamp",
    "time",
    "datetime",
    "open_time",
    "open-time",
    "open time",
    "date",
)


def _normalize_column(name: str) -> str:
    """Normalize column names to snake_case lower-case form."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _find_timestamp_column(columns: Iterable[str]) -> str | None:
    """Return the first column name that matches the timestamp aliases (if any)."""

    for col in columns:
        if col in TIMESTAMP_CANDIDATES:
            return col
    return None


def merge_monthly_trading_csv(
    year: int,
    base_dir: str | Path = "data/raw/BTCUSDT",
) -> Path:
    """
    Concatenate monthly ``trading_data.csv`` files for a given year.

    Args:
        year: Calendar year (e.g. 2024) to merge.
        base_dir: Root directory containing year subdirectories.

    Returns:
        Path to the merged CSV file.

    Raises:
        FileNotFoundError: If the year directory or source CSVs are missing.
    """

    base_path = Path(base_dir)
    year_path = base_path / str(year)
    if not year_path.exists():
        raise FileNotFoundError(f"Year directory not found: {year_path}")

    month_dirs = sorted(
        [p for p in year_path.iterdir() if p.is_dir()],
        key=lambda p: int(p.name),
    )
    csv_paths = [month / "trading_data.csv" for month in month_dirs]
    csv_paths = [p for p in csv_paths if p.exists()]

    if not csv_paths:
        raise FileNotFoundError(
            f"No monthly trading CSVs found under {year_path}"
        )

    frames = [pd.read_csv(csv_path) for csv_path in csv_paths]
    merged = pd.concat(frames, ignore_index=True)

    output_path = base_path / f"btc_{year}_raw.csv"
    merged.to_csv(output_path, index=False)

    print(
        f"[merge_monthly_trading_csv] Merged {len(merged):,} rows "
        f"into {output_path}"
    )

    return output_path


def load_raw_ohlcv(path: str) -> pd.DataFrame:
    """
    Load a raw OHLCV CSV file into a clean pandas DataFrame.

    The CSV must contain a ``timestamp`` column alongside OHLCV fields.
    The timestamp can be either milliseconds since epoch or an ISO8601 string.
    Rows are sorted chronologically, duplicate timestamps are dropped, and the
    resulting DataFrame uses a UTC ``DatetimeIndex``.

    Args:
        path: Filesystem path to the raw CSV.

    Returns:
        Cleaned OHLCV data indexed by timestamp.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If a timestamp column is missing or cannot be parsed.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [_normalize_column(col) for col in df.columns]

    ts_column = _find_timestamp_column(df.columns)
    if ts_column is None:
        raise ValueError(
            "Input CSV must contain a timestamp column (e.g. 'timestamp' or 'open time')."
        )
    if ts_column != "timestamp":
        df = df.rename(columns={ts_column: "timestamp"})

    ts_raw = df["timestamp"]

    numeric_ts = pd.to_numeric(ts_raw, errors="coerce")
    if numeric_ts.notna().all():
        timestamp = pd.to_datetime(numeric_ts, unit="ms", utc=True)
    else:
        timestamp = pd.to_datetime(ts_raw, utc=True, errors="coerce")

    if timestamp.isna().any():
        raise ValueError(
            "Unable to parse all timestamps. Ensure they are either ISO strings "
            "or millisecond epoch values."
        )

    df = (
        df.assign(timestamp=timestamp)
        .sort_values("timestamp", kind="mergesort")
        .drop_duplicates(subset="timestamp", keep="first")
        .set_index("timestamp")
    )

    if df.empty:
        raise ValueError("No rows remain after cleaning; check the input data.")

    print(f"[load_raw_ohlcv] Loaded {len(df):,} rows from {csv_path}")
    print(f"[load_raw_ohlcv] First timestamp: {df.index[0]}")
    print(f"[load_raw_ohlcv] Last timestamp: {df.index[-1]}")

    return df


def main() -> None:
    """CLI entry point for merging monthly Binance CSV exports into a single file."""

    parser = argparse.ArgumentParser(
        description="Merge monthly BTC/USDT trading CSV files for a given year."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Calendar year to merge (default: 2024).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/raw/BTCUSDT",
        help="Root directory containing year/month subfolders.",
    )
    args = parser.parse_args()

    output = merge_monthly_trading_csv(args.year, args.base_dir)
    print(f"Created merged CSV at {output}")


if __name__ == "__main__":
    main()
