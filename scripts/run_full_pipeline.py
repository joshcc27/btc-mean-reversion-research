#!/usr/bin/env python3
"""CLI entry point for the raw-data -> signal -> backtest pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocess import clean_and_resample_ohlcv  # noqa: E402
from src.signals import rolling_zscore_signal  # noqa: E402
from src.backtest import run_backtest_with_costs  # noqa: E402
from src.metrics import summarize_performance  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run the full preprocessing -> signal -> cost-aware backtest pipeline."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "BTCUSDT" / "btc_2024_raw.csv",
        help="Path to the raw 1-minute OHLCV CSV.",
    )
    parser.add_argument(
        "--resample-rule",
        type=str,
        default="5min",
        help="Pandas offset alias for the resample interval (default: 5min).",
    )
    parser.add_argument(
        "--date-start",
        type=str,
        default=None,
        help="Optional inclusive start timestamp (e.g., 2024-01-01).",
    )
    parser.add_argument(
        "--date-end",
        type=str,
        default=None,
        help="Optional inclusive end timestamp (e.g., 2024-03-01).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window for the z-score signal.",
    )
    parser.add_argument(
        "--z-entry",
        type=float,
        default=1.0,
        help="Z-score threshold for entering/exiting positions.",
    )
    parser.add_argument(
        "--cost-per-unit",
        type=float,
        default=0.0005,
        help="Transaction cost per unit of position change.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_prices = clean_and_resample_ohlcv(
        args.raw_path,
        resample_rule=args.resample_rule,
        date_start=args.date_start,
        date_end=args.date_end,
    )
    df_signal = rolling_zscore_signal(df_prices, window=args.window, z_entry=args.z_entry)

    bt_result = run_backtest_with_costs(
        df_signal,
        cost_per_unit_change=args.cost_per_unit,
    )
    df_bt = bt_result.df

    summary = summarize_performance(
        strategy_returns=df_bt["strategy_return_net"],
        equity_curve=df_bt["equity_curve_net"],
        positions=df_bt["position"],
    )
    summary["net_return"] = bt_result.net_return
    summary["total_cost"] = bt_result.extras.get("total_cost", 0.0) if bt_result.extras else 0.0

    print("=== Backtest Summary ===")
    print(f"Rows: {len(df_bt):,}, Date range: {df_bt.index.min()} -> {df_bt.index.max()}")
    print(f"Window / z-entry: {args.window} / {args.z_entry}")
    print(f"Sharpe (net): {summary['sharpe']:.3f}")
    print(f"Max drawdown (net): {summary['max_drawdown']:.2%}")
    print(f"Total return (net): {summary['total_return']:.2%}")
    print(f"Net log-return sum: {summary['net_return']:.4f}")
    print(f"Trades: {int(summary.get('trade_count', 0)):,}")
    print(f"Total cost debited: {summary['total_cost']:.4f}")


if __name__ == "__main__":
    main()
