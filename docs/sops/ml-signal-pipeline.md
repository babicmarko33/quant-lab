# ML Signal Pipeline — Standard Operating Procedure

## Purpose
Define a rigorous, reproducible workflow for developing, validating, and deploying machine learning-based trading signals using `alpha_ml`.

---

## Module Structure

```
src/alpha_ml/
├── features/
│   └── feature_store.py    — OHLCV → winsorized, z-scored feature matrix
├── models/
│   ├── base.py             — ModelTrainer ABC
│   └── xgboost_model.py    — XGBoost binary classifier
├── validation/
│   ├── purged_kfold.py     — Purged k-fold CV (López de Prado ch7)
│   └── cross_validate.py   — Full OOS evaluation → CVResult
└── pipeline.py             — End-to-end OHLCV → signals → BacktestResult
```

---

## Data Leakage Prevention (Critical)

### Purged K-Fold — Why it Matters

Standard k-fold **leaks** in financial time series because adjacent observations are correlated (autocorrelated returns, overlapping labels).

`PurgedKFold` fixes this by:
1. **No overlap**: train indices have zero overlap with test indices
2. **Embargo**: `purge_window` trading days are removed from the boundary between train/test
3. **Temporal ordering**: all train observations precede all test observations

```python
from alpha_ml.validation.purged_kfold import PurgedKFold

cv = PurgedKFold(n_splits=5, purge_window=21)  # 21-day embargo
for train_idx, test_idx in cv.split(X):
    assert train_idx.max() < test_idx.min()  # No temporal overlap
```

**Never use `sklearn.model_selection.KFold` or `StratifiedKFold`** on financial data.

---

## Feature Engineering Rules

```python
from alpha_ml.features.feature_store import FeatureStore

store = FeatureStore(winsorize_pct=0.01)  # Clip 1% tails
X, y = store.build(df, horizon=5, target_type="direction")
# X: winsorized + z-scored, no NaN, no future data
# y: binary (1 = up >= 0, 0 = down) at t+horizon
```

### Rules
1. All features computed from data at time `t` — NEVER use `t+1` or later
2. Winsorize before z-scoring to prevent extreme values dominating
3. Target label uses `t+horizon` close — this introduces overlap → use purged CV
4. Drop NaN rows **after** feature construction (not before — you lose warmup info)

---

## Model Training Workflow

```python
from alpha_ml.pipeline import MLSignalPipeline
from alpha_ml.models.xgboost_model import XGBoostModel

model = XGBoostModel(n_estimators=200, max_depth=4, learning_rate=0.05)
pipeline = MLSignalPipeline(
    model=model,
    train_ratio=0.6,    # 60% train, 40% OOS
    horizon=5,          # 5-day forward return label
    long_threshold=0.55,
    short_threshold=0.45,
)
result = pipeline.run(df)  # → BacktestResult (OOS only)
importance = model.feature_importance()
```

---

## Performance Thresholds

A signal is **deployable** only if ALL of these hold on out-of-sample data:

| Metric | Minimum |
|--------|---------|
| OOS Sharpe | > 0.5 |
| OOS Accuracy | > 52% |
| Max Drawdown | < 20% |
| Feature overlap (max importance) | < 50% single feature |

---

## Anti-Patterns

1. **Training on full dataset then reporting test accuracy** — Always split temporally first
2. **Using standard k-fold on overlapping labels** — Use `PurgedKFold`
3. **Optimising hyperparameters on test set** — Hyperparams belong to validation set only
4. **Not winsorizing** — Financial data has fat tails; extreme values blow up z-scores
5. **Ignoring feature importance** — If one feature dominates (>60% gain), model is fragile
6. **Reporting in-sample results** — Only OOS metrics appear in research reports

---

## Extending the Framework

To add a new model:
1. Subclass `alpha_ml.models.base.ModelTrainer`
2. Implement `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `feature_importance() -> pd.Series`
3. Write tests: RED → GREEN, test `RuntimeError` before fit, test feature_importance sums to 1
4. Integrate into `MLSignalPipeline` via model substitution
5. Run `scripts/run_backtest.py` for smoke test

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Ch 7 (Purged K-Fold)
- Chen & Guestrin (2016). XGBoost. *KDD 2016*
- Bailey & López de Prado (2012). The Sharpe Ratio Efficient Frontier. *JOIS*
