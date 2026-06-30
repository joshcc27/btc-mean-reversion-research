# Week 2 Summary

## Parameters
- Window (W): 20 bars (~100 minutes)
- Z-entry threshold: 1.0 standard deviations
- Signal: rolling z-score on log returns, long if z < -1, short if z > +1, flat otherwise with next-bar execution.

## Performance Snapshot
| Split | Sharpe | Max Drawdown | Trades |
|-------|--------|--------------|--------|
| Train | TBD    | TBD          | TBD    |
| Test  | TBD    | TBD          | TBD    |

## Notes
- Strategy shows TBD on train, TBD on test using the baseline configuration.
- Results ignore transaction costs/slippage; expect degradation once costs are included.
- Next: add trading costs, sweep parameters, and perform robustness checks.

