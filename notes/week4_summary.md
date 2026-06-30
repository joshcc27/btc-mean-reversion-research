# Week 4 Summary

## Focus
- Extend the cost-aware backtester with reusable risk overlays:
  - **Volatility targeting**: scale positions using a rolling realised-vol estimate (default 0.30% per 5‑minute bar, lookback 96 bars, leverage capped at 1.5×).
  - **Daily stop-out**: flatten exposure for the rest of the UTC session once cumulative intra-day net PnL falls below −1%.
- Wire the overlays into `run_backtest_with_costs` so notebooks and scripts can simply pass parameter dictionaries rather than re-implementing the logic.

## Baseline vs. risk overlays (train/test split, COST = 0.0005)

| Variant | Train Sharpe (net) | Train Max DD | Test Sharpe (net) | Test Max DD | Trades (test) | Total cost (test) |
| --- | --- | --- | --- | --- | --- | --- |
| Costs only (window=20, z=1.5) | -74.7 | -100% | -75.6 | -100% | 16,768 | 8.8 |
| + Vol targeting (0.30%, max 1.5×) | -69.8 | -99.9% | -72.4 | -99.9% | 9,835 | 7.6 |
| + Vol targeting & 1% daily stop | -67.1 | -99.7% | -70.3 | -99.5% | 9,282 | 7.6 |

- Overlays reduce turnover by ~40% (≈17k → 10k trades test) and shave a couple of “cost units” because the strategy trades smaller size when realised vol is high.
- Daily stop-outs slightly improve worst-case drawdowns (no more instant −100% in a single day) but the equity still trends toward zero; there is no meaningful improvement in Sharpe.

## Final “Version 1.0” (documented in `notebooks/04_risk_management_and_final_strategy.ipynb`)
- Parameters: `window=20`, `z_entry=1.5`, `COST_PER_UNIT=0.0005`.
- Risk controls: volatility target 0.30% per bar (96-bar lookback, leverage cap 1.5×) plus daily stop-out at −1%.
- Train metrics: Sharpe −67.1, max DD −99.7%, trades 20.6k, total cost 15.8.
- Test metrics: Sharpe −70.3, max DD −99.5%, trades 9.3k, total cost 7.6.
- Conclusion: risk overlays make the strategy more “resume-ready” (positions sized consistently, basic loss limits) but do not recover profitability—the real issue remains the high turnover relative to assumed 5 bps costs.
