from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from btc_research.data_loader import epoch_to_datetime, load_raw_ohlcv
from btc_research.preprocess import chronological_split, resample_ohlcv

JAN_2024_MS = 1704067200000  # 2024-01-01 00:00:00 UTC
JAN_2025_MS = 1735689600000  # 2025-01-01 00:00:00 UTC


def _rows(start_ms: int, scale: int, n: int = 10) -> list[str]:
    """Headerless Binance kline rows. scale=1 writes milliseconds, scale=1000 microseconds."""
    rows = []
    for i in range(n):
        t = (start_ms + i * 60_000) * scale
        p = 42000 + i
        rows.append(
            f"{t},{p},{p + 5},{p - 5},{p + 1},1.5,{t + 59_999 * scale},63000,100,0.7,29400,0"
        )
    return rows


def _write(path: Path, rows: list[str], zipped: bool = False) -> Path:
    text = "\n".join(rows) + "\n"
    if not zipped:
        path.write_text(text)
        return path
    zpath = path.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(path.name, text)
    return zpath


def test_loads_millisecond_csv(tmp_path: Path) -> None:
    df = load_raw_ohlcv(_write(tmp_path / "BTCUSDT-1m-2024-01.csv", _rows(JAN_2024_MS, 1)))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert df.index[-1] == pd.Timestamp("2024-01-01 00:09", tz="UTC")
    assert df["close"].iloc[0] == 42001.0


def test_loads_microsecond_zip(tmp_path: Path) -> None:
    f = _write(tmp_path / "BTCUSDT-1m-2025-01.csv", _rows(JAN_2025_MS, 1000), zipped=True)
    df = load_raw_ohlcv(f)
    assert df.index[0] == pd.Timestamp("2025-01-01 00:00", tz="UTC")
    assert len(df) == 10


def test_directory_with_mixed_units_and_overlap(tmp_path: Path) -> None:
    dec = _rows(JAN_2025_MS - 5 * 60_000, 1, n=6)  # last row duplicates the first January row
    _write(tmp_path / "BTCUSDT-1m-2024-12.csv", dec, zipped=True)
    _write(tmp_path / "BTCUSDT-1m-2025-01.csv", _rows(JAN_2025_MS, 1000, n=5), zipped=True)
    df = load_raw_ohlcv(tmp_path)
    assert len(df) == 10
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("1min")).all()


def test_tolerates_header_row(tmp_path: Path) -> None:
    header = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore"
    )
    f = _write(tmp_path / "BTCUSDT-1m-2024-01.csv", [header, *_rows(JAN_2024_MS, 1)])
    assert len(load_raw_ohlcv(f)) == 10


def test_loader_runs_validation(tmp_path: Path) -> None:
    rows = _rows(JAN_2024_MS, 1)
    parts = rows[3].split(",")
    parts[2], parts[3] = "1", "2"  # high < low
    rows[3] = ",".join(parts)
    with pytest.raises(ValueError, match="high < low"):
        load_raw_ohlcv(_write(tmp_path / "BTCUSDT-1m-2024-01.csv", rows))


def test_epoch_units() -> None:
    idx = epoch_to_datetime(pd.Series([JAN_2024_MS, JAN_2024_MS * 1000]))
    assert idx[0] == idx[1] == pd.Timestamp("2024-01-01", tz="UTC")


def test_resample_and_split(tmp_path: Path) -> None:
    raw = load_raw_ohlcv(_write(tmp_path / "BTCUSDT-1m-2024-01.csv", _rows(JAN_2024_MS, 1)))
    df = resample_ohlcv(raw)
    assert len(df) == 2
    assert df["high"].iloc[0] == 42004 + 5
    assert df["close"].iloc[1] == 42009 + 1
    parts = chronological_split(pd.DataFrame({"x": range(10)}), (0.6, 0.2, 0.2))
    assert [len(p) for p in parts] == [6, 2, 2]
