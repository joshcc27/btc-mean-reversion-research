"""Mean-reversion study on 5-minute BTC/USDT bars: data loading, signal, backtest, metrics."""

from .backtest import BacktestResult, run_backtest
from .data_loader import load_raw_ohlcv
from .metrics import (
    break_even_cost,
    count_entries,
    max_drawdown,
    sharpe_confidence_interval,
    sharpe_ratio,
    summarize_backtest,
    turnover,
)
from .preprocess import build_dataset, chronological_split, load_processed_ohlcv
from .signals import ZScoreSignalConfig, rolling_zscore_signal, zscore_signal
from .validation import validate_ohlcv

__all__ = [
    "BacktestResult",
    "ZScoreSignalConfig",
    "break_even_cost",
    "build_dataset",
    "chronological_split",
    "count_entries",
    "load_processed_ohlcv",
    "load_raw_ohlcv",
    "max_drawdown",
    "rolling_zscore_signal",
    "run_backtest",
    "sharpe_confidence_interval",
    "sharpe_ratio",
    "summarize_backtest",
    "turnover",
    "validate_ohlcv",
    "zscore_signal",
]
