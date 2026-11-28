from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_frame

from btc_research.backtest import run_backtest
from btc_research.signals import rolling_zscore_signal


def test_position_lag_is_one_bar() -> None:
    df = make_frame([0.001] * 6, [0.0, 1.0, -1.0, 1.0, 0.0, 1.0])
    bt = run_backtest(df, cost_per_unit_turnover=0.0)
    np.testing.assert_array_equal(bt.df["position"], [0.0, 0.0, 1.0, -1.0, 1.0, 0.0])


def test_cost_arithmetic_and_compounding() -> None:
    df = make_frame([0.001, 0.002, -0.001, 0.0], [1.0, -1.0, 0.0, 0.0])
    bt = run_backtest(df, cost_per_unit_turnover=0.001)
    # positions 0, 1, -1, 0 -> |change| 0, 1, 2, 1
    np.testing.assert_allclose(bt.df["position_change"], [0, 1, 2, 1])
    np.testing.assert_allclose(bt.df["transaction_cost"], [0, 0.001, 0.002, 0.001])
    np.testing.assert_allclose(bt.df["strategy_return_gross"], [0, 0.002, 0.001, 0.0])
    np.testing.assert_allclose(
        bt.df["equity_curve_net"], np.exp(bt.df["strategy_return_net"].cumsum())
    )
    assert np.isclose(bt.turnover, 4.0)


def test_vol_scaling_does_not_use_current_bar_return() -> None:
    """A return shock at bar k must not change the position held during bar k."""
    n, k = 200, 150
    rets = [0.001 if i % 2 else -0.001 for i in range(n)]
    shocked_rets = list(rets)
    shocked_rets[k] = 0.05
    kwargs = {"cost_per_unit_turnover": 0.0, "target_volatility": 0.0005, "volatility_lookback": 20}

    p_base = run_backtest(make_frame(rets, [1.0] * n), **kwargs).df["position"]
    p_shock = run_backtest(make_frame(shocked_rets, [1.0] * n), **kwargs).df["position"]

    np.testing.assert_allclose(p_shock.iloc[: k + 1], p_base.iloc[: k + 1])
    assert p_shock.iloc[k + 1] < p_base.iloc[k + 1]


def test_vol_scaling_never_exceeds_cap(random_walk: pd.DataFrame) -> None:
    sig = rolling_zscore_signal(random_walk, window=20, z_entry=1.0)
    bt = run_backtest(sig, target_volatility=0.01, max_position=1.0)
    assert bt.df["position"].abs().max() <= 1.0
    assert bt.share_of_bars_at_cap == 1.0


def test_daily_stop_triggers_on_cost_only_losses() -> None:
    """Zero asset returns and constant flipping: gross P&L is 0 but fees add up."""
    n = 12
    signal = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
    bt = run_backtest(
        make_frame([0.0] * n, signal), cost_per_unit_turnover=0.001, daily_stop_loss=0.004
    )
    pos = bt.df["position"].to_numpy()
    # Cumulative net P&L: 0, -0.001, -0.003, -0.005 -> stop crossed on bar 3, flat from bar 4.
    assert (bt.df["strategy_return_gross"] == 0).all()
    assert pos[3] != 0.0
    assert (pos[4:] == 0.0).all()


def test_daily_stop_resets_next_day() -> None:
    n = 288 + 10
    signal = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
    pos = run_backtest(
        make_frame([0.0] * n, signal), cost_per_unit_turnover=0.001, daily_stop_loss=0.004
    ).df["position"]
    assert (pos.iloc[10:288] == 0).all()
    assert (pos.iloc[288:] != 0).any()


def test_mirror_negates_gross_with_identical_costs(random_walk: pd.DataFrame) -> None:
    mr = rolling_zscore_signal(random_walk, window=20, z_entry=1.0)
    mom = rolling_zscore_signal(random_walk, window=20, z_entry=1.0, direction="momentum")
    a = run_backtest(mr, cost_per_unit_turnover=0.001, target_volatility=0.0015)
    b = run_backtest(mom, cost_per_unit_turnover=0.001, target_volatility=0.0015)
    assert np.isclose(a.gross_return, -b.gross_return)
    assert np.isclose(a.total_cost, b.total_cost)
