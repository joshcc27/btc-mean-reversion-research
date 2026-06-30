"""Lightweight smoke tests for core utilities."""

from __future__ import annotations

import unittest
from math import sqrt

import numpy as np
import pandas as pd

from src.backtest import run_backtest_with_costs, run_basic_backtest
from src.metrics import max_drawdown, sharpe_ratio
from src.signals import rolling_zscore_signal
from src.walk_forward import WindowSpec, generate_rolling_windows


class MetricsSmokeTest(unittest.TestCase):
    def test_sharpe_of_zero_returns_is_zero(self) -> None:
        returns = pd.Series([0.0] * 5)
        self.assertEqual(sharpe_ratio(returns), 0.0)

    def test_max_drawdown_of_increasing_curve_is_zero(self) -> None:
        equity = pd.Series(np.linspace(1.0, 2.0, 10))
        self.assertEqual(max_drawdown(equity), 0.0)

    def test_sharpe_uses_ddof1(self) -> None:
        """sharpe_ratio must use ddof=1, matching the standard financial definition."""
        returns = pd.Series([0.01, -0.005, 0.003, 0.008, -0.002])
        expected = returns.mean() / returns.std(ddof=1) * sqrt(365 * 288)
        self.assertAlmostEqual(sharpe_ratio(returns), expected, places=8)


class WalkForwardWindowTest(unittest.TestCase):
    def test_generate_rolling_windows_basic(self) -> None:
        idx = pd.date_range("2024-01-01", periods=24 * 6, freq="h", tz="UTC")
        df = pd.DataFrame({"close": np.arange(len(idx))}, index=idx)
        windows = generate_rolling_windows(df, train_days=1, test_days=1)
        self.assertTrue(len(windows) > 0)
        first = windows[0]
        self.assertIsInstance(first, WindowSpec)
        self.assertLess(first.train_start, first.train_end)
        self.assertLess(first.train_end, first.test_start)


class BacktestCompoundingTest(unittest.TestCase):
    def _make_df(self, log_returns: list[float], signal: list[float]) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=len(log_returns), freq="5min", tz="UTC")
        return pd.DataFrame(
            {"close": 100.0, "log_return": log_returns, "signal_raw": signal},
            index=idx,
        )

    def test_equity_curve_uses_exp_cumsum(self) -> None:
        """run_backtest_with_costs must compound log returns via exp(cumsum), not (1+r).cumprod."""
        log_rets = [0.01, -0.005, 0.008, -0.003, 0.006]
        signals = [1.0, 1.0, 1.0, 1.0, 1.0]
        df = self._make_df(log_rets, signals)

        result = run_backtest_with_costs(df, cost_per_unit_change=0.0, shift_positions=False)
        net_rets = result.df["strategy_return_net"]
        expected_equity = np.exp(net_rets.cumsum())
        np.testing.assert_allclose(
            result.df["equity_curve_net"].values,
            expected_equity.values,
            rtol=1e-9,
        )

    def test_basic_and_cost_backtest_equity_agree_without_costs(self) -> None:
        """With zero costs and no shift, both backtest functions must produce the same equity."""
        log_rets = [0.005, -0.003, 0.007, 0.001, -0.004]
        signals = [1.0, -1.0, 1.0, 0.0, 1.0]
        df = self._make_df(log_rets, signals)

        basic = run_basic_backtest(df, shift_positions=False)
        cost = run_backtest_with_costs(df, cost_per_unit_change=0.0, shift_positions=False)

        np.testing.assert_allclose(
            basic.equity_curve.values,
            cost.df["equity_curve_net"].values,
            rtol=1e-9,
        )

    def test_position_lag_is_exactly_one_bar(self) -> None:
        """With shift_positions=True, position at bar t must equal signal_raw at bar t-1."""
        log_rets = [0.001] * 6
        signals = [0.0, 1.0, -1.0, 1.0, 0.0, 1.0]
        df = self._make_df(log_rets, signals)

        result = run_backtest_with_costs(df, cost_per_unit_change=0.0, shift_positions=True)
        positions = result.df["position"].values
        expected = np.array([0.0, 0.0, 1.0, -1.0, 1.0, 0.0])
        np.testing.assert_array_equal(positions, expected)

    def test_cost_arithmetic(self) -> None:
        """transaction_cost must equal cost_rate * |position.diff()|, first bar uses position[0]."""
        cost_rate = 0.0005
        log_rets = [0.001, 0.001, 0.001]
        signals = [0.0, 1.0, -1.0]
        df = self._make_df(log_rets, signals)

        result = run_backtest_with_costs(df, cost_per_unit_change=cost_rate, shift_positions=False)
        positions = result.df["position"]
        pos_change = positions.diff().fillna(positions).abs()
        expected_cost = cost_rate * pos_change

        np.testing.assert_allclose(
            result.df["transaction_cost"].values,
            expected_cost.values,
            rtol=1e-12,
        )


class SignalTest(unittest.TestCase):
    def test_zscore_is_nan_for_flat_price(self) -> None:
        """A completely flat price series has rolling_std=0; z_score must be NaN, not 0."""
        idx = pd.date_range("2024-01-01", periods=30, freq="5min", tz="UTC")
        df = pd.DataFrame({"close": [100.0] * 30}, index=idx)
        out = rolling_zscore_signal(df, window=10, z_entry=1.0)
        z_valid = out["z_score"].dropna()
        self.assertEqual(len(z_valid), 0, "z_score should be all-NaN for a flat price series")

    def test_signal_raw_no_longer_contains_prelagged_signal_column(self) -> None:
        """zscore_breakout_signal must not write a pre-shifted 'signal' column."""
        idx = pd.date_range("2024-01-01", periods=30, freq="5min", tz="UTC")
        prices = 100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.1, 30))
        df = pd.DataFrame({"close": prices}, index=idx)
        out = rolling_zscore_signal(df, window=10, z_entry=1.0)
        self.assertNotIn("signal", out.columns, "Dead pre-shifted 'signal' column must be removed")
        self.assertIn("signal_raw", out.columns)


if __name__ == "__main__":
    unittest.main()
