# Backtesting Protocol — Standard Operating Procedure

## Purpose
Ensure all strategy backtests produce valid, unbiased performance metrics that can be trusted for capital allocation decisions.

---

## Pre-Backtest Checklist

- [ ] Data free of survivorship bias (use point-in-time constituent lists)
- [ ] No look-ahead bias in feature computation (features use only data ≤ t)
- [ ] Transaction costs modeled (commission + slippage in basis points)
- [ ] Proper train/validation/test split with temporal ordering
- [ ] Walk-forward window defined (e.g., 252d train / 63d test, rolling)

---

## Walk-Forward Validation Protocol

```
For each fold i in 1..K:
  1. Train window: [t_start_i, t_end_i]
  2. Embargo period: [t_end_i, t_end_i + embargo]  # purge leakage
  3. Test window: [t_end_i + embargo, t_test_end_i]
  4. Fit strategy on train window
  5. Generate signals on test window
  6. Record out-of-sample returns
```

### Purged K-Fold Parameters
- **Embargo**: min 5 trading days between train and test
- **Fold overlap**: ZERO — no observation used in both train and test
- **Rolling vs Expanding**: Document which approach and why

---

## Required Performance Metrics

| Metric | Minimum Threshold | Notes |
|--------|------------------|-------|
| Sharpe Ratio (OOS) | > 0.5 | Out-of-sample, net of costs |
| Probabilistic Sharpe (PSR) | > 95% | vs benchmark SR=0 |
| Max Drawdown | < 25% | Absolute peak-to-trough |
| Win Rate | > 45% | For trend-following |
| Profit Factor | > 1.2 | Gross profit / gross loss |
| Calmar Ratio | > 0.5 | Annual return / max DD |

---

## Statistical Validation

### Deflated Sharpe Ratio (DSR)
When reporting Sharpe Ratio, always compute DSR to account for:
- Number of strategies tested (multiple comparisons)
- Skewness and kurtosis of returns
- Sample length

```python
# Use: alpha_engine.evaluation.deflated_sharpe_ratio(sr, n_trials, T, skew, kurt)
# Only deploy if DSR > 95% confidence
```

### Minimum Track Record Length
Before declaring a strategy "works", verify:
```
min_TRL = (Z_α / (SR_hat - SR_benchmark))^2
```
If observed track record < min_TRL → NOT statistically significant.

---

## Anti-Patterns (NEVER DO)

1. **Optimizing on full dataset** — Always hold out final 20% as untouched test
2. **Cherry-picking parameters** — Report results for ALL parameter combos tried
3. **Ignoring market regimes** — Test across bull, bear, and sideways separately
4. **Zero transaction costs** — ALWAYS include realistic costs (min 10bps roundtrip)
5. **Reporting in-sample Sharpe** — Only OOS metrics count
6. **Single backtest run** — Use Monte Carlo permutation to assess robustness

---

## Deliverables After Each Backtest

1. `results/{strategy_name}/performance.json` — All metrics in JSON
2. `results/{strategy_name}/equity_curve.png` — With drawdown overlay
3. `results/{strategy_name}/monthly_returns.png` — Heatmap
4. `results/{strategy_name}/report.md` — Written analysis with conclusions
5. Git commit with conventional message: `test(backtest): {strategy} walk-forward results`
