## Week 5 Summary – Walk-Forward Testing

### Window construction
- Source data: full 2024 BTC/USDT 5-minute series (105,408 bars).
- Rolling windows: 60 calendar days for training, followed by 10 calendar days for testing.
- Total windows generated: 31 (first window spans 2024‑01‑01 → 2024‑02‑29 for training and 2024‑03‑01 → 2024‑03‑10 for testing).

### Pseudo-live performance (net of 5 bps costs)

| Method | Sharpe | Max DD | Cumulative return |
| --- | --- | --- | --- |
| Fixed params (window=20, z=1.5) | -50.46 | -99.997% | -99.997% |
| Re-optimised per window (grid: window ∈ {10,20,40}, z ∈ {1.0,1.5,2.0}) | -27.03 | -98.70% | -98.70% |

- Fixed-parameter walk-forward aligns with the single train/test split story: costs immediately drive equity to zero.
- Allowing re-optimisation improves the pseudo-live Sharpe slightly (less negative) because later windows overwhelmingly pick slower, wider parameters that trade less often, but the strategy still collapses when friction is applied.
- Chosen parameters across windows: 40/2.0 (19 windows), 20/2.0 (10 windows), 10/2.0 (2 windows). No window ever selected z-entry < 2 once costs were considered.

### Comparison vs. earlier static split
- Static 70/30 split (Week 3) delivered Sharpe -73.9 (train) / -82.6 (test) with max drawdowns ≈ -100%.
- Walk-forward results are similarly catastrophic; per-window optimisation merely delays the wipe-out slightly due to lower turnover but does not change the conclusion that 5 bps costs overwhelm the short-term signal.

### Takeaways
- Walk-forward testing reinforces the lesson that high-turnover intraday mean reversion is extraordinarily fragile under realistic fees.
- Re-optimising frequently risks overfitting (note how parameters drift toward the slowest/least active settings) without producing a viable edge.
- The real value is the process: rolling evaluation surfaces fragility earlier and provides a defensible narrative for recruiters/interviewers.
