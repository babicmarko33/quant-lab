# Research Methodology — Standard Operating Procedure

## Purpose
Define a rigorous, reproducible research workflow that prevents data snooping, overfitting, and false discoveries in quantitative strategy development.

---

## Research Pipeline

```
1. Hypothesis → 2. Literature → 3. Data Prep → 4. Feature Eng → 5. Model/Strategy
       ↓                                                                    ↓
6. Walk-Forward Test → 7. Statistical Validation → 8. Paper Trade → 9. Deploy/Reject
```

---

## Step 1: Hypothesis Formation

Every research project MUST start with a written hypothesis:

```markdown
## Hypothesis: [Name]
**Claim**: [Specific, testable statement]
**Null Hypothesis**: [What we're testing against — usually "no edge"]
**Economic Rationale**: [WHY would this work? Behavioral/structural reason]
**Expected Sharpe**: [Prior estimate based on literature]
**Universe**: [Which assets, what timeframe]
**Data Period**: [Train: YYYY-YYYY, Test: YYYY-YYYY]
```

### Rules
- Hypothesis MUST be written BEFORE looking at data
- Economic rationale MUST exist (no pure data mining)
- If you can't explain WHY it works in one paragraph → probably overfitting

---

## Step 2: Literature Review

Before implementing any strategy:
1. Search academic papers (SSRN, arXiv q-fin, Journal of Finance)
2. Check if alpha has been published → likely decayed
3. Document prior Sharpe estimates from literature
4. Note any known regime dependencies

---

## Step 3: Data Preparation

Follow `data-handling.md` SOP. Additionally:
- Define train/validation/test splits BEFORE touching data
- **Train**: 60% of data (fit parameters)
- **Validation**: 20% (hyperparameter selection)
- **Test**: 20% (final evaluation, TOUCH ONLY ONCE)

---

## Step 4: Feature Engineering

Follow `quantcore/features/pipeline.py` patterns:
- All features must be point-in-time (no look-ahead)
- Document feature rationale
- Check for multicollinearity (VIF > 10 → remove)
- Normalize/standardize using ONLY training data statistics

---

## Step 5: Model/Strategy Development

- Start simple (linear model, single rule) → add complexity only if justified
- Use information coefficient (IC) to evaluate feature predictiveness
- Document all parameters tried (for DSR computation later)

---

## Step 6: Walk-Forward Testing

Follow `backtesting-protocol.md` SOP for:
- Purged k-fold cross-validation
- Expanding or rolling window
- Proper embargo periods

---

## Step 7: Statistical Validation

### Required Tests
| Test | Purpose | Threshold |
|------|---------|-----------|
| PSR | Is SR statistically > 0? | > 95% |
| DSR | Accounting for multiple testing | > 95% |
| Min TRL | Enough data for significance? | Track record ≥ min_TRL |
| Regime robustness | Works in bull + bear? | SR > 0 in both |
| Turnover analysis | Realistic execution? | < 500% annual |

### Multiple Testing Correction
```
N_trials = total strategies/parameters tested
Apply Bonferroni or BH correction
Report BOTH raw and corrected p-values
```

---

## Step 8: Paper Trading

Before any real capital:
1. Run strategy in paper trading for minimum 3 months
2. Compare live fills vs. backtest assumptions
3. Measure execution slippage
4. Confirm P&L matches backtest within 2 standard deviations

---

## Step 9: Decision Gate

### Deploy if ALL of:
- [ ] DSR > 95%
- [ ] Paper trading matches backtest (within 2σ)
- [ ] Max drawdown acceptable for portfolio allocation
- [ ] Execution feasible (turnover, liquidity)
- [ ] No upcoming known regime change

### Reject if ANY of:
- [ ] DSR < 80%
- [ ] Only works in one regime
- [ ] Requires unrealistic execution
- [ ] No economic rationale survives scrutiny

---

## Research Notebook Template

Every research project gets a notebook in `notebooks/research/`:

```
notebooks/research/
├── 001_momentum_12_1.ipynb       # Numbered for ordering
├── 002_mean_reversion_pairs.ipynb
└── README.md                      # Index of all research
```

Each notebook MUST contain:
1. Hypothesis section (copy from above template)
2. Data description and splits
3. Feature analysis (IC, correlation)
4. Backtest results with all metrics
5. Statistical validation (PSR, DSR)
6. Conclusion: DEPLOY / REJECT / NEEDS MORE DATA
