from __future__ import annotations

import numpy as np
import pandas as pd

from btc_research.walk_forward import (
    WindowSpec,
    generate_rolling_windows,
    run_walk_forward,
    walk_forward_retrained_params,
)


def _df(days: int = 12, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=days * 288, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx))))
    df = pd.DataFrame({"close": close}, index=idx)
    df["log_return"] = np.log(df["close"]).diff()
    return df


def test_windows_are_ordered_and_test_blocks_do_not_overlap() -> None:
    windows = generate_rolling_windows(_df(), train_days=3, test_days=2)
    assert windows and isinstance(windows[0], WindowSpec)
    for w in windows:
        assert w.train_start < w.train_end < w.test_start <= w.test_end
    for a, b in zip(windows, windows[1:], strict=False):
        assert a.test_end < b.test_start


def test_selector_only_sees_training_block() -> None:
    df = _df()
    windows = generate_rolling_windows(df, train_days=3, test_days=2)
    last_seen: list[pd.Timestamp] = []

    def spy(train: pd.DataFrame) -> tuple[int, float]:
        last_seen.append(train.index.max())
        return 20, 1.0

    run_walk_forward(df, windows, spy)
    assert len(last_seen) == len(windows)
    for seen, spec in zip(last_seen, windows, strict=True):
        assert seen <= spec.train_end


def test_future_data_does_not_change_earlier_windows() -> None:
    df = _df()
    windows = generate_rolling_windows(df, train_days=3, test_days=2)
    first = windows[0]
    altered = df.copy()
    future = altered.index > first.test_end
    altered.loc[future, "close"] *= np.exp(np.linspace(0, 0.5, int(future.sum())))
    altered["log_return"] = np.log(altered["close"]).diff()

    kw = {"candidate_windows": [10, 20], "candidate_z_entries": [1.0, 1.5]}
    a = walk_forward_retrained_params(df, windows, **kw)
    b = walk_forward_retrained_params(altered, windows, **kw)
    pd.testing.assert_series_equal(
        a["test_returns"].loc[first.test_start : first.test_end],
        b["test_returns"].loc[first.test_start : first.test_end],
    )
    assert (
        a["per_window_metrics"][0]["chosen_window"] == b["per_window_metrics"][0]["chosen_window"]
    )
