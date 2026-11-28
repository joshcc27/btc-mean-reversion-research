"""Resample raw 1-minute klines to 5-minute bars, persist them, and split chronologically."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from .data_loader import load_raw_ohlcv
from .validation import validate_ohlcv

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "btcusdt_5m.parquet"

FIVE_MINUTES: Final[str] = "5min"
OHLCV_AGG: Final[dict[str, str]] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(df_raw: pd.DataFrame, rule: str = FIVE_MINUTES) -> pd.DataFrame:
    """Aggregate OHLCV bars to ``rule`` and add simple and log close-to-close returns.

    Bars are labelled by their open time (pandas default ``label='left'``), so the
    close of the bar stamped 00:05 is the last trade before 00:10.
    """
    out = df_raw.resample(rule).agg(OHLCV_AGG).dropna(subset=["close"])
    close = out["close"]
    return out.assign(
        **{"return": close.pct_change(), "log_return": np.log(close / close.shift(1))}
    )


def build_dataset(
    raw: str | Path = RAW_DIR,
    rule: str = FIVE_MINUTES,
    output_path: str | Path | None = PROCESSED_PATH,
) -> pd.DataFrame:
    """Load raw klines, resample, validate the result, and optionally save it as Parquet."""
    df = resample_ohlcv(load_raw_ohlcv(raw), rule=rule)
    validate_ohlcv(df, expected_freq=rule)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
    return df


def load_processed_ohlcv(path: str | Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the processed 5-minute Parquet file, building it from raw data if missing.

    The first bar (no prior close, so no return) is dropped.
    """
    data_path = Path(path)
    df = pd.read_parquet(data_path) if data_path.exists() else build_dataset(output_path=data_path)
    return df.sort_index().dropna(subset=["log_return"]).copy()


def chronological_split(
    df: pd.DataFrame,
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
) -> list[pd.DataFrame]:
    """Split a time-ordered frame into consecutive blocks by row count.

    ``fractions`` gives the share of rows in each block and should sum to 1.
    """
    bounds = np.round(np.cumsum([0.0, *fractions]) * len(df)).astype(int)
    return [df.iloc[a:b].copy() for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
