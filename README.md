# Intraday BTC/USDT Breakout-Reversal Research

Intraday research sandbox focused on breakout-reversal trades in BTC/USDT using five-minute bars from 2024. The core idea is a rolling z-score signal (go long/short when the bar's z-score crosses ±threshold) evaluated in a vectorised backtester with transaction costs, volatility targeting, and daily stop-out overlays so you can walk through the entire workflow during interviews.

## Project Overview

- **Asset & frequency**: BTC/USDT spot, aggregated to 5-minute OHLCV bars from Binance 1-minute exports (2024-01-01 → 2024-12-31).
- **Goal**: Prototype a simple rolling z-score mean-reversion strategy, split data chronologically, and evaluate performance under realistic trading frictions.
- **Backtesting**: Signals feed into `src/backtest.py`, which lags positions one bar, caps leverage (±1 by default), supports volatility targeting + daily stop-outs, and charges 5 bps per unit of position change for costs.

## Key Features

- Data pipeline: raw CSVs → cleaned 5-minute OHLCV with returns (`src/data_loader.py`, `src/preprocess.py`).
- Rolling z-score mean-reversion signal (`src/signals.py`) with configurable window/z-entry.
- Backtests with/without transaction costs plus risk overlays (`src/backtest.py`).
- Metrics + parameter sweeps (`src/metrics.py`, `notebooks/03_costs_and_param_sweep.ipynb`).
- CLI helpers: `scripts/run_full_pipeline.py` (raw -> backtest) and `scripts/make_summary_report.py` (plots + `reports/summary.pdf`).

## Repo Structure

```
├─ src/         # data_loader, preprocess, signals, backtest, metrics, param_search
├─ notebooks/   # exploration/backtest notebooks (plots, reporting)
├─ notes/       # weekly summaries
├─ scripts/     # automation scripts (pipelines, reporting)
├─ data/        # raw Binance CSVs + processed parquet files
├─ tests/       # lightweight smoke tests
└─ reports/     # generated summary PDF + figures
```

Raw Binance CSVs are too large for GitHub, so place them under `data/raw/BTCUSDT/…` following the structure described in `notes/week1_summary.md`, then run the preprocessing script/notebook to regenerate `data/processed/btcusdt_5m.parquet`.

## How to Run

```powershell
uv sync
```

```bash
# End-to-end pipeline: load raw CSV, resample, backtest with costs
python scripts/run_full_pipeline.py --raw-path data/raw/BTCUSDT/btc_2024_raw.csv

# Generate/update the PDF summary
python scripts/make_summary_report.py

# Run the lightweight smoke tests
python -m pytest tests/
```

## Results Snapshot

| Variant                             | Train Sharpe (no cost / with cost) | Test Sharpe (no cost / with cost) | Trades (train / test) | Cost debited (train / test) |
|-------------------------------------|------------------------------------|-----------------------------------|-----------------------|-----------------------------|
| Raw ±1 signal (window=20, z=1.0)    | 0.235 / **-73.9**                  | 0.361 / **-82.6**                 | 33k / 14.5k           | 18.5 / 8.1                  |
| Vol target 0.30% + 1% daily stop    | 0.198 / **-74.7**                  | 0.284 / **-75.6**                 | 20.6k / 9.8k          | 15.8 / 7.6                  |
| Vol target 0.35% + 0.75% daily stop | 0.181 / **-75.0**                  | 0.266 / **-73.5**                 | 19.4k / 9.3k          | 15.5 / 7.6                  |

*Without costs the best test Sharpe is ≈0.36; once 5 bps per unit of turnover is charged, Sharpe plunges to ~-75 on both train and test. Risk overlays reduce turnover and daily drawdowns but cannot overcome the assumed friction, so the emphasis is on the process (data → signal → risk → reporting) rather than claiming a live-tradable edge. A narrative + figures live in `reports/summary.pdf`.*

![Net equity curves](reports/equity_curves.png)

### Walk-forward / pseudo-live evaluation

- Data is partitioned into rolling train/test windows (default 60-day train, 10-day test). Each window is backtested sequentially to mimic a production process where models are refreshed periodically.
- Parameters can be **fixed** (e.g., window=20, z=1.5) or **re-optimised** inside each training chunk using the small grid from Week 3 (windows ∈ {10, 20, 40}, z-entry ∈ {1.0, 1.5, 2.0}).
- For every window, strategy returns on the subsequent test slice are recorded and concatenated into a pseudo-live equity curve. This highlights how fragile the signal becomes once costs and periodic re-tuning are considered (fixed-parameter walk-forward Sharpe ≈ -50; re-optimised ≈ -27 for the 2024 sample).
- See `notebooks/05_walk_forward.ipynb` for the end-to-end workflow and `src/walk_forward.py` for reusable helpers.

### Limitations (honest constraints)

- Single asset (BTC/USDT) and a single intraday horizon; no cross-sectional diversification.
- Cost model is transaction-cost-per-unit only; no detailed impact/latency/slippage beyond the 5 bps assumption.
- No production plumbing (order-routing, borrow, position limits) or intraday risk management beyond the simple overlays; results are strictly research/educational prototypes.

> **Disclaimer**: This repository is for research/educational purposes only and is not trading advice.
