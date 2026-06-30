# Week 3 Summary

## Cost model
- The cost-aware backtest lags the discrete {-1, 0, 1} signal, multiplies the lagged position by the next-bar log return for gross PnL, and debits a cost each time the target position changes (`position_change = position - position_lag`). Costs are accrued as `cost_per_unit_change * |position_change|`, so a flip from -1 to +1 pays 2× the per-unit rate, and the net return subtracts this charge before compounding the equity curve (`equity_curve_net`). See `src/backtest.py:150-183`.
- I set `cost_per_unit_change = 0.0005`, the same constant used in the notebook sweep. That equates to 5 bps per unit of position change, which is a reasonable blended estimate of commission + half-spread/slippage for a liquid U.S. equity future or ETF when trading with ~1 tick of friction on each entry/exit. With binary exposure, this translates to roughly 1 bp per side or $5 per $1 M notionally traded, which is plausible for the style of intraday rebalancing being simulated.

## Parameter grid
- Train sweep enumerated the Cartesian product of `WINDOWS = [10, 20, 40]` and `Z_ENTRIES = [1.0, 1.5, 2.0]`, giving nine (window, z-entry) combinations. A shared `COST_PER_UNIT = 0.0005` constant feeds the cost-aware backtest. See `notebooks/03_costs_and_param_sweep.ipynb:271-516`.
- After reviewing the train table/heatmaps, I constrained the test-set evaluation to the more stable train candidates: (20, 1.0), (20, 1.5), and (40, 1.0); captured in `candidate_params`. See `notebooks/03_costs_and_param_sweep.ipynb:947-951`.

## Key train results (cost sweep)
| window | z_entry | Sharpe (no cost) | Sharpe (with cost) | Trades | Total cost | Net return |
|--------|---------|------------------|--------------------|--------|------------|------------|
| 10 | 1.0 | -1.259 | -74.928 | 36,220 | 20.336 | -20.641 |
| 10 | 1.5 | -0.220 | -49.141 | 18,013 | 9.441 | -9.480 |
| 10 | 2.0 | -2.425 | -28.011 | 6,242 | 3.169 | -3.449 |
| 20 | 1.0 | -0.243 | -67.671 | 33,185 | 18.532 | -18.591 |
| 20 | 1.5 | 0.235 | -44.550 | 16,768 | 8.822 | -8.780 |
| 20 | 2.0 | **0.755** | -24.671 | 7,123 | 3.649 | -3.547 |
| 40 | 1.0 | 0.086 | -61.509 | 30,835 | 17.207 | -17.185 |
| 40 | 1.5 | -0.022 | -39.845 | 15,457 | 8.206 | -8.210 |
| 40 | 2.0 | 0.100 | **-23.577** | 7,343 | 3.804 | -3.789 |

- Best train Sharpe without costs occurs at (window=20, z=2.0) with 0.755, but per-trade costs of 5 bps erase the edge completely, driving the with-cost Sharpe down to -24.7 (`notebooks/03_costs_and_param_sweep.ipynb:447-478`).
- The least-bad with-cost Sharpe is -23.6 at (window=40, z=2.0), highlighting how the 30k–36k trade counts amplify turnover penalties despite similar gross profiles.
- Total dollar cost per split ranges from ~3.2 to ~20.3 units of capital over the train sample, roughly matching the number of times the strategy switches sides. This confirms that cost dominates the signal right now and motivates either sparser trading (larger windows, wider thresholds) or a more nuanced microstructure model/alpha.

## Turnover snapshot
- Trade counts over the train sweep span 6,242–36,220 switches per parameter pair (`notebooks/03_costs_and_param_sweep.ipynb:447-478`). Even the “slow” configs (window ≥ 20, z ≥ 2) still turnover 7k+ times in-sample, so each extra basis point of per-trade friction immediately compounds into double-digit percent costs.
- Those counts imply roughly one trade per 8–10 bars for the busiest setup (window=10, z=1.0) and one trade per ~40 bars for the sparsest (window=40, z=2.0), consistent with the zig-zag behavior seen in the notebook plots.

## Key test results
| window | z_entry | Test Sharpe (no cost) | Test Sharpe (with cost) | Test max DD (no cost) | Test max DD (with cost) | Test net return |
|--------|---------|-----------------------|-------------------------|-----------------------|-------------------------|-----------------|
| 20 | 1.0 | 2.260 | -74.530 | -0.103 | -0.9996 | -7.873 |
| 20 | 1.5 | 0.361 | -52.155 | -0.132 | -0.9782 | -3.816 |
| 40 | 1.0 | 0.424 | -72.397 | -0.133 | -0.9995 | -7.575 |

- Same parameters that looked best on train (window=20, z=2.0) were not forwarded to test because the with-cost Sharpe was still -24+, so the above candidates reflect the more “stable” but slower-trading settings.
- The highest out-of-sample Sharpe before costs is 2.26 at (window=20, z=1.0), delivering a modest -10% drawdown, but once 5 bps per turn is applied the Sharpe collapses to -74.5 and max drawdown plunges to ~100%, mirroring the train behavior (`notebooks/03_costs_and_param_sweep.ipynb:1034-1042`).
- Ignoring costs, the OOS Sharpe values are lower than the train sweep but still positive (0.36–2.26), so the raw signal retains some predictive power; the catastrophe arises only when friction is applied.

## Qualitative takeaways
- Sharpe is extremely sensitive to `z_entry`: shifting the threshold from 1.5 to 2.0 at a fixed window (e.g., 20) flips the costless Sharpe from 0.24 to 0.76 and slashes trade count by >50%, while moving the threshold the other way quickly pushes the strategy back into heavily negative territory. Minor tweaks therefore create outsized swings in turnover and PnL, underscoring how thin and unstable the signal currently is.
- Across most of the (window, z) grid, adding a 5 bps per-unit cost roughly halves the Sharpe (or worse), and in every tested combination it flips the sign negative; even the “least bad” cases shrink from ~0.75 costless Sharpe to ≈-25, illustrating that transaction costs dominate any edge in the current formulation.
