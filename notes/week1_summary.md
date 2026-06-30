# Week 1 Summary

## Focus
- Ingested Binance BTC/USDT monthly spot CSVs, standardized timestamp handling, and built a year-level merged file (`data/raw/BTCUSDT/btc_2024_raw.csv`).
- Implemented a preprocessing pipeline (`src/preprocess.py`) to resample 1-minute OHLCV to 5-minute bars, compute returns, and persist `data/processed/btcusdt_5m.parquet`.
- Created `notebooks/01_exploration.ipynb` to inspect dataset shape, date boundaries, descriptive stats, price trajectory, log-return distribution, and short-lag autocorrelations.

## Files Produced
- `data/raw/BTCUSDT/btc_2024_raw.csv`
- `data/processed/btcusdt_5m.parquet`
- Updated `src/data_loader.py`, `src/preprocess.py`, `notebooks/01_exploration.ipynb`, and documentation (`README.md`, `week1_summary.md`).

## Observations
- 2024 data contains 105,408 five-minute bars (full year coverage) with price levels centered around 64k USDT and a broad 14k standard deviation.
- Volume distribution is skewed (mean ≈ 123 BTC per 5-minute bar; occasional spikes above 5k).
- Log returns are tightly clustered around zero with weak short-lag autocorrelations (|ρ| < 0.02 through lag 10), consistent with noisy, near-efficient intraday behavior.
