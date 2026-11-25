#!/usr/bin/env python3
"""Raw klines -> 5-minute bars -> selection on train/validation -> one test evaluation.

Rebuilds data/processed/btcusdt_5m.parquet from data/raw/ and prints the selected
configuration and the test-set results for both signal directions.

Usage:
    uv run python scripts/download_data.py      # once
    uv run python scripts/run_full_pipeline.py
"""

from __future__ import annotations

import argparse

import pandas as pd

from btc_research import study
from btc_research.preprocess import PROCESSED_PATH, RAW_DIR, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument(
        "--cost-bp", type=float, default=study.BASE_COST * 1e4, help="Cost per unit turnover, bp."
    )
    args = parser.parse_args()

    df = build_dataset(args.raw_dir, output_path=PROCESSED_PATH)
    print(f"5-minute bars: {len(df):,} ({df.index[0]} to {df.index[-1]}) -> {PROCESSED_PATH}")

    res = study.run_study(PROCESSED_PATH, cost=args.cost_bp / 1e4)
    print("\nSplits:\n", res.splits.describe().to_string())
    print("\nStep 1, train  -> signal:", res.signal_config.describe())
    print("Step 2, train  -> session hours:", sorted(res.session_hours))
    print("Step 3, valid. -> final:", res.final_config.describe())

    cols = {
        "gross_log_return": "gross",
        "net_log_return": "net",
        "turnover": "turnover",
        "entries": "entries",
        "break_even_bp": "break-even bp",
        "sharpe_gross": "gross Sharpe",
        "sharpe_gross_ci_low": "CI low",
        "sharpe_gross_ci_high": "CI high",
        "sharpe_net": "net Sharpe",
    }
    pd.set_option("display.width", 200)
    print(f"\nStep 4, test (cost {args.cost_bp:g} bp per unit turnover):")
    print(res.results[list(cols)].rename(columns=cols).to_string(float_format=lambda v: f"{v:.4g}"))
    bh = res.buy_and_hold["test"]
    print(f"\nBuy and hold, test: log return {bh['log_return']:.4f}, Sharpe {bh['sharpe']:.2f}")


if __name__ == "__main__":
    main()
