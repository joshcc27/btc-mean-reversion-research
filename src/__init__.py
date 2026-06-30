"""Quant research utilities for BTC/USDT analysis."""

from .data_loader import load_raw_ohlcv, merge_monthly_trading_csv
from .preprocess import (
    resample_ohlcv,
    resample_ohlcv_to_5min,
    clean_and_resample_ohlcv,
    load_processed_ohlcv,
    chronological_train_test_split,
    save_processed_ohlcv,
)
from .signals import (
    DEFAULT_PRICE_COLUMN,
    ZScoreSignalConfig,
    zscore_breakout_signal,
    rolling_zscore_signal,
)
from .backtest import (
    BacktestResult,
    CostBacktestResult,
    run_basic_backtest,
    run_backtest_naive,
    run_backtest_with_costs,
)
from .metrics import (
    TRADING_PERIODS_PER_YEAR,
    scaled_sharpe,
    sharpe_ratio,
    max_drawdown,
    count_trades,
    trade_count,
    summarize_performance,
)

__all__ = [
    "load_raw_ohlcv",
    "merge_monthly_trading_csv",
    "resample_ohlcv",
    "resample_ohlcv_to_5min",
    "clean_and_resample_ohlcv",
    "load_processed_ohlcv",
    "chronological_train_test_split",
    "save_processed_ohlcv",
    "DEFAULT_PRICE_COLUMN",
    "ZScoreSignalConfig",
    "zscore_breakout_signal",
    "rolling_zscore_signal",
    "BacktestResult",
    "CostBacktestResult",
    "run_basic_backtest",
    "run_backtest_naive",
    "run_backtest_with_costs",
    "TRADING_PERIODS_PER_YEAR",
    "scaled_sharpe",
    "sharpe_ratio",
    "max_drawdown",
    "count_trades",
    "trade_count",
    "summarize_performance",
]
