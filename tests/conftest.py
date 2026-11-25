"""Shared fixtures and helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def random_walk() -> pd.DataFrame:
    """Five days of 5-minute bars from a seeded random walk."""
    idx = pd.date_range("2024-01-01", periods=5 * 288, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(idx))))
    df = pd.DataFrame({"close": close}, index=idx)
    df["log_return"] = np.log(df["close"]).diff()
    return df


def make_frame(log_returns: list[float], signal: list[float]) -> pd.DataFrame:
    """Frame with given per-bar log returns and signal, 5-minute UTC index."""
    idx = pd.date_range("2024-01-01", periods=len(log_returns), freq="5min", tz="UTC")
    return pd.DataFrame({"close": 100.0, "log_return": log_returns, "signal_raw": signal}, index=idx)
