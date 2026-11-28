from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_research.signals import _hysteresis_signal, rolling_zscore_signal


def _prices_with_jump(jump: float) -> pd.DataFrame:
    """Small alternating returns, then one large return on the last bar."""
    n = 30
    rets = np.array([0.001 if i % 2 else -0.001 for i in range(n)])
    rets[-1] = jump
    idx = pd.date_range("2024-01-01", periods=n + 1, freq="5min", tz="UTC")
    close = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
    return pd.DataFrame({"close": close}, index=idx)


@pytest.mark.parametrize("exit_bw", [None, 0.5])
def test_mean_reversion_goes_long_after_large_drop(exit_bw: float | None) -> None:
    out = rolling_zscore_signal(
        _prices_with_jump(-0.02), window=10, z_entry=2.0, exit_bandwidth=exit_bw
    )
    assert out["z_score"].iloc[-1] < -2.0
    assert out["signal_raw"].iloc[-1] == 1.0


@pytest.mark.parametrize("exit_bw", [None, 0.5])
def test_mean_reversion_goes_short_after_large_rise(exit_bw: float | None) -> None:
    out = rolling_zscore_signal(
        _prices_with_jump(0.02), window=10, z_entry=2.0, exit_bandwidth=exit_bw
    )
    assert out["z_score"].iloc[-1] > 2.0
    assert out["signal_raw"].iloc[-1] == -1.0


@pytest.mark.parametrize("exit_bw", [None, 0.5, 0.0])
def test_momentum_is_exact_negative(random_walk: pd.DataFrame, exit_bw: float | None) -> None:
    mr = rolling_zscore_signal(random_walk, window=20, z_entry=1.0, exit_bandwidth=exit_bw)
    mom = rolling_zscore_signal(
        random_walk, window=20, z_entry=1.0, exit_bandwidth=exit_bw, direction="momentum"
    )
    assert (mr["signal_raw"] != 0).sum() > 0
    np.testing.assert_array_equal(mom["signal_raw"].to_numpy(), -mr["signal_raw"].to_numpy())


def test_hysteresis_holds_until_exit_band() -> None:
    z = np.array([0.0, -2.5, -1.0, -0.6, -0.4, 0.0, 2.5, 1.0, 0.4])
    out = _hysteresis_signal(z, z_entry=2.0, exit_bandwidth=0.5)
    # Long at -2.5, held while z <= -0.5, closed at -0.4. Short at 2.5, held at 1.0, closed at 0.4.
    np.testing.assert_array_equal(out, [0, 1, 1, 1, 0, 0, -1, -1, 0])


def test_zscore_is_nan_for_flat_price() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="5min", tz="UTC")
    out = rolling_zscore_signal(pd.DataFrame({"close": [100.0] * 30}, index=idx), window=10)
    assert out["z_score"].isna().all()
    assert (out["signal_raw"] == 0).all()


def test_signal_uses_only_past_data(random_walk: pd.DataFrame) -> None:
    full = rolling_zscore_signal(random_walk, window=20, z_entry=1.0, exit_bandwidth=0.5)
    cut = rolling_zscore_signal(random_walk.iloc[:700], window=20, z_entry=1.0, exit_bandwidth=0.5)
    pd.testing.assert_series_equal(full["signal_raw"].iloc[:700], cut["signal_raw"])


def test_invalid_parameters(random_walk: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        rolling_zscore_signal(random_walk, z_entry=1.0, exit_bandwidth=1.0)
    with pytest.raises(ValueError):
        rolling_zscore_signal(random_walk, direction="sideways")  # type: ignore[arg-type]
