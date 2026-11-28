# Methodology notes

These notes expand on the method section of the README. Code references are to modules under `src/btc_research/`.

## Data

The input is Binance's monthly archive of BTCUSDT spot 1-minute klines (`data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/`). Each file is a zipped CSV without a header and with twelve columns: open time, open, high, low, close, volume, close time, quote volume, trade count, taker buy base volume, taker buy quote volume, and an unused field. Open times are epoch milliseconds up to the end of 2024 and epoch microseconds from January 2025. `data_loader.epoch_to_datetime` decides the unit per value (anything at or above $10^{14}$ is treated as microseconds), so a directory mixing both formats loads correctly.

2024 has 527,040 one-minute bars with no gaps, which resample to 105,408 five-minute bars. Bars are labelled by their open time; the close of the bar stamped 00:05 is the last trade before 00:10. Returns are close-to-close log returns $r_t = \log(P_t / P_{t-1})$, and equity curves compound them as $\exp\left(\sum_{s \le t} r_s\right)$.

`validation.validate_ohlcv` checks that timestamps are sorted and unique, close is positive, high is at least low, and volume is non-negative. The loader runs it on the 1-minute data. `preprocess.build_dataset` runs it again on 5-minute bars with a gap check (no gap longer than three bars).

Over the year the close ranged from about 38,600 to 108,300 USDT. Five-minute log returns have a standard deviation of 0.166% and excess kurtosis of about 32, and the median 8-hour rolling volatility is 0.13% per bar. Autocorrelation at lags 1 and 2 is $-0.013$ and $-0.018$, and at most $0.011$ in absolute value at lags 3, 5, 10 and 20.

## Timing

All quantities are per bar $t$:

- $z_t$ uses returns $r_{t-W+1}, \dots, r_t$, so it is known at the close of bar $t$.
- The signal $s_t$ is computed from $z_t$ at the close of bar $t$.
- The position held during bar $t+1$ is $p_{t+1} = s_t\, m_t$, where $m_t$ is the volatility scaler ($m_t = 1$ when volatility targeting is off). $m_t$ uses returns up to and including $r_t$, never $r_{t+1}$.
- Gross return for bar $t+1$ is $p_{t+1}\, r_{t+1}$, and cost is $c\,\lvert p_{t+1} - p_t \rvert$.

`tests/test_backtest.py::test_vol_scaling_does_not_use_current_bar_return` checks the volatility-scaling lag directly: a return shock on bar $k$ leaves the position on bar $k$ unchanged and only reduces the position from bar $k+1$.

## Cost arithmetic

Cost is charged per unit of turnover, where turnover is $\sum_t \lvert p_t - p_{t-1} \rvert$. At $c = 10$ bp, trading \$1M of notional costs \$1,000; at 5 bp it costs \$500. Entering a full position is one unit of turnover, and a direct flip from long to short is two.

Net return is gross return $G$ minus $c$ times turnover $\Theta$, so the break-even cost is the fee at which net return would be zero:

$$
c^{\ast} = \frac{G}{\Theta}.
$$

For a signal that trades this often, it's more useful than Sharpe. On the test block the mean-reversion signal breaks even at $-0.14$ bp and the momentum mirror at $+0.14$ bp, against fees of about 10 bp.

The metric "entries" (`metrics.count_entries`) counts bars on which a new long or short position starts, including direct flips. It is reported next to turnover because the two differ once positions are scaled: resizing from 1.0 to 0.8 adds 0.2 to turnover but is not an entry.

## Why the selection protocol has three blocks

Train is used for the signal grid and the session-hour ranking, validation for the overlay choice, and test only once. Every choice is made on net Sharpe at 10 bp.

At that fee level, net Sharpe depends almost entirely on how much a configuration trades. Across the 27-point training grid, net Sharpe and turnover have a correlation of $-0.98$, so the procedure ends up picking the slowest configuration. I report break-even cost for every grid point because it shows the gross edge separately from the ranking.

The momentum direction isn't a separate hypothesis. Without a daily stop, its gross P&L is exactly the negative of the mean-reversion gross P&L, at the same cost. If one direction had an edge before costs, the other would show a matching loss. Showing both makes it easy to see that the gross numbers are small and sit either side of zero.

## Sharpe intervals

`metrics.sharpe_confidence_interval` is a moving-block bootstrap. Blocks are $b = n^{1/3}$ bars long (27 bars for the test block), with start points drawn uniformly with replacement from all $n - b + 1$ positions. The blocks are joined, cut to $n$ bars, and resampled 1,000 times, and the interval is the 2.5th to 97.5th percentile. Using blocks keeps short-range dependence and volatility clustering intact.

The intervals are wide. A gross Sharpe of $\pm 0.6$ on ten weeks of data comes with an interval of roughly $\pm 4$.

## Walk-forward

`walk_forward.generate_rolling_windows` builds 60-day training windows each followed by a 10-day test window and rolls forward by 10 days. For 2024 this gives 31 windows, the first trained on Jan 1 to Feb 29 and tested on Mar 1 to Mar 10. `walk_forward.run_walk_forward` passes only the training slice to the parameter selector, computes the signal on data up to the end of the test window, and backtests the test window alone, starting flat. Test windows do not overlap, so their net returns are concatenated into one out-of-sample series. `tests/test_walk_forward.py` checks that the selector never sees bars after the training window and that changing data after a window does not change that window's result.
