from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from btc_research.metrics import (
    BARS_PER_YEAR,
    break_even_cost,
    count_entries,
    max_drawdown,
    sharpe_confidence_interval,
    sharpe_ratio,
    turnover,
)


def test_sharpe_of_zero_returns_is_zero() -> None:
    assert sharpe_ratio(pd.Series([0.0] * 5)) == 0.0


def test_sharpe_uses_ddof1_and_annualises() -> None:
    r = pd.Series([0.01, -0.005, 0.003, 0.008, -0.002])
    assert np.isclose(sharpe_ratio(r), r.mean() / r.std(ddof=1) * sqrt(BARS_PER_YEAR))


def test_max_drawdown() -> None:
    assert max_drawdown(pd.Series(np.linspace(1.0, 2.0, 10))) == 0.0
    assert np.isclose(max_drawdown(pd.Series([1.0, 2.0, 1.0, 1.5])), -0.5)


def test_turnover_and_entries_are_different_quantities() -> None:
    pos = pd.Series([0.0, 1.0, 1.0, 0.8, -1.0, -1.0, 0.0, 1.0])
    # 1 + 0 + 0.2 + 1.8 + 0 + 1 + 1
    assert np.isclose(turnover(pos), 5.0)
    # entries: 0 -> 1, 0.8 -> -1 (flip), 0 -> 1. Resizing and exits are not entries.
    assert count_entries(pos) == 3


def test_break_even_cost() -> None:
    assert np.isclose(break_even_cost(0.02, 100.0), 0.0002)  # 2 bp per unit turnover
    assert break_even_cost(-0.01, 100.0) < 0
    assert np.isnan(break_even_cost(0.01, 0.0))


def test_bootstrap_interval_contains_point_estimate_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0001, 0.002, 5000))
    lo, hi = sharpe_confidence_interval(r, n_bootstrap=300)
    assert lo < sharpe_ratio(r) < hi
    assert (lo, hi) == sharpe_confidence_interval(r, n_bootstrap=300)
