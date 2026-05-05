# AlphaML Phase 2 — ML Prediction Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a supervised ML pipeline that generates return-prediction signals with XGBoost, validated via López de Prado purged k-fold cross-validation — providing bias-free IS/OOS performance comparison.

**Architecture:** Feature engineering feeds into a `ModelTrainer` abstraction (XGBoost baseline). Purged k-fold (no leakage from overlapping labels) splits data for CV. Predictions are fed back into the existing `run_backtest()` engine as signals.

**Tech Stack:** scikit-learn, xgboost, pandas, numpy, quantcore.features.pipeline (existing)

---

## Prerequisites

Install ML extras (already in pyproject.toml `[ml]` optional):
```powershell
.\.venv\Scripts\activate; pip install scikit-learn xgboost
```
Check version: `python -c "import xgboost; print(xgboost.__version__)"`

---

### Task 1: Purged K-Fold Cross-Validator

**Why first:** Every subsequent ML task depends on bias-free CV. Build the validation tool before any model.

**Files:**
- Create: `src/alpha_ml/validation/purged_kfold.py`
- Create: `src/alpha_ml/validation/__init__.py`
- Test: `tests/test_alpha_ml/test_purged_kfold.py`

**Core concept:** In financial time series, adjacent observations have overlapping labels (e.g., 5-day forward return). Standard k-fold leaks information. Purging removes train samples whose label period overlaps with the test period. Embargoing removes samples immediately after the test fold.

**Step 1: Write failing tests**

```python
# tests/test_alpha_ml/test_purged_kfold.py
from alpha_ml.validation.purged_kfold import PurgedKFold

def test_n_splits_correct():
    pkf = PurgedKFold(n_splits=5)
    assert pkf.n_splits == 5

def test_produces_n_folds(sample_index):
    pkf = PurgedKFold(n_splits=5)
    folds = list(pkf.split(sample_index))
    assert len(folds) == 5

def test_test_indices_non_overlapping(sample_index):
    pkf = PurgedKFold(n_splits=5)
    all_test = []
    for _, test_idx in pkf.split(sample_index):
        all_test.extend(list(test_idx))
    assert len(all_test) == len(set(all_test))  # no duplicates

def test_purging_removes_overlap(sample_index):
    pkf = PurgedKFold(n_splits=5, purge_window=5)
    for train_idx, test_idx in pkf.split(sample_index):
        test_start = min(test_idx)
        # Purging: train samples near test boundary are removed
        near_boundary = [i for i in train_idx if abs(i - test_start) < 5]
        # With purge_window=5, no train idx should be within 5 of test_start
        assert all(test_start - i >= 5 for i in near_boundary), "Purging failed"
```

**Step 2:** `pytest tests/test_alpha_ml/test_purged_kfold.py -q` → FAIL (import error)

**Step 3: Implement**

```python
# src/alpha_ml/validation/purged_kfold.py
class PurgedKFold:
    """Purged k-fold cross-validation for financial time series.
    
    Reference: López de Prado (2018) AFML, Chapter 7.
    
    Parameters
    ----------
    n_splits : int
        Number of folds.
    purge_window : int
        Bars to purge from train set adjacent to test fold start.
    embargo : int
        Bars to exclude after test fold end.
    """
    def __init__(self, n_splits=5, purge_window=0, embargo=0):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo = embargo

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // self.n_splits
        indices = np.arange(n)
        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = test_start + fold_size if k < self.n_splits - 1 else n
            test_idx = indices[test_start:test_end]
            # Purge: exclude train samples within purge_window of test_start
            # Embargo: exclude train samples within embargo of test_end
            purge_start = max(0, test_start - self.purge_window)
            embargo_end = min(n, test_end + self.embargo)
            train_idx = np.concatenate([
                indices[:purge_start],
                indices[embargo_end:]
            ])
            yield train_idx, test_idx
```

**Step 4:** `pytest tests/test_alpha_ml/test_purged_kfold.py -v` → ALL PASS

**Step 5: Commit**
```
git commit -m "feat(alpha-ml): purged k-fold cross-validator (López de Prado ch7)"
```

---

### Task 2: Feature Store — ML-Ready Feature Matrix

**Why:** Raw OHLCV → clean feature matrix with target labels. Wraps existing `quantcore.features.pipeline` and adds ML-specific transforms (winsorization, z-scoring, NaN handling).

**Files:**
- Create: `src/alpha_ml/features/feature_store.py`
- Create: `src/alpha_ml/features/__init__.py`
- Test: `tests/test_alpha_ml/test_feature_store.py`

**Step 1: Write failing tests**

```python
from alpha_ml.features.feature_store import FeatureStore

def test_returns_dataframe(sample_ohlcv):
    fs = FeatureStore()
    X, y = fs.build(sample_ohlcv, horizon=5)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_no_nan_in_X(sample_ohlcv):
    fs = FeatureStore()
    X, y = fs.build(sample_ohlcv, horizon=5)
    assert not X.isna().any().any()

def test_no_look_ahead_in_X(sample_ohlcv):
    """Feature matrix must not contain future information."""
    fs = FeatureStore()
    X, y = fs.build(sample_ohlcv, horizon=5)
    # X and y must be aligned — X[i] predicts y[i]
    assert X.index.equals(y.index)

def test_target_is_binary_direction(sample_ohlcv):
    fs = FeatureStore()
    X, y = fs.build(sample_ohlcv, horizon=5, target_type="direction")
    assert set(y.dropna().unique()).issubset({0, 1})

def test_winsorization_removes_extreme_outliers(sample_ohlcv):
    fs = FeatureStore(winsorize_pct=0.01)
    X, _ = fs.build(sample_ohlcv)
    # After winsorization, no feature value should be beyond 3-sigma
    z_scores = (X - X.mean()) / X.std()
    assert (z_scores.abs() <= 5).all().all()
```

**Step 2:** `pytest tests/test_alpha_ml/test_feature_store.py -q` → FAIL

**Step 3: Implement `FeatureStore`**
- Call `quantcore.features.pipeline.build_features()` and `add_target()`
- Winsorize using `scipy.stats.mstats.winsorize`
- Z-score features
- Drop NaN rows from both X and y consistently

**Step 4:** `pytest tests/test_alpha_ml/test_feature_store.py -v` → ALL PASS

**Step 5: Commit**
```
git commit -m "feat(alpha-ml): FeatureStore — ML-ready feature matrix with winsorization + z-score"
```

---

### Task 3: ModelTrainer ABC + XGBoost Implementation

**Why:** Abstract interface allows swapping models (XGBoost → LSTM → ensemble) without changing calling code.

**Files:**
- Create: `src/alpha_ml/models/base.py`
- Create: `src/alpha_ml/models/xgboost_model.py`
- Create: `src/alpha_ml/models/__init__.py`
- Test: `tests/test_alpha_ml/test_xgboost_model.py`

**Step 1: Write failing tests**

```python
from alpha_ml.models.base import ModelTrainer
from alpha_ml.models.xgboost_model import XGBoostPredictor

def test_model_trainer_is_abstract():
    with pytest.raises(TypeError):
        ModelTrainer()

def test_xgboost_fit_predict(sample_X, sample_y):
    model = XGBoostPredictor(n_estimators=10, max_depth=3)
    model.fit(sample_X, sample_y)
    preds = model.predict(sample_X)
    assert len(preds) == len(sample_X)
    assert set(np.unique(preds)).issubset({0, 1})

def test_predict_proba_in_01(sample_X, sample_y):
    model = XGBoostPredictor(n_estimators=10)
    model.fit(sample_X, sample_y)
    proba = model.predict_proba(sample_X)
    assert proba.shape == (len(sample_X), 2)
    assert (proba >= 0).all() and (proba <= 1).all()

def test_feature_importance_available(sample_X, sample_y):
    model = XGBoostPredictor(n_estimators=10)
    model.fit(sample_X, sample_y)
    importance = model.feature_importance()
    assert isinstance(importance, pd.Series)
    assert len(importance) == sample_X.shape[1]

def test_not_fitted_raises(sample_X):
    model = XGBoostPredictor()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(sample_X)
```

**Step 2:** `pytest tests/test_alpha_ml/test_xgboost_model.py -q` → FAIL

**Step 3: Implement ABC + XGBoost**

```python
# ModelTrainer ABC
class ModelTrainer(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    @abstractmethod
    def feature_importance(self) -> pd.Series: ...

# XGBoostPredictor wraps xgboost.XGBClassifier
# Raises RuntimeError("Model not fitted") if predict called before fit
```

**Step 4:** `pytest tests/test_alpha_ml/test_xgboost_model.py -v` → ALL PASS

**Step 5: Commit**
```
git commit -m "feat(alpha-ml): ModelTrainer ABC + XGBoostPredictor with feature importance"
```

---

### Task 4: Cross-Validated Model Evaluation

**Why:** Purged k-fold CV is the core bias-free evaluation. Validates that IS Sharpe > OOS Sharpe detection works (overfitting flag).

**Files:**
- Create: `src/alpha_ml/validation/cross_validate.py`
- Test: `tests/test_alpha_ml/test_cross_validate.py`

**Step 1: Write failing tests**

```python
from alpha_ml.validation.cross_validate import cross_val_predict_purged, CVResult

def test_returns_cv_result(sample_X, sample_y):
    model = XGBoostPredictor(n_estimators=5)
    result = cross_val_predict_purged(model, sample_X, sample_y)
    assert isinstance(result, CVResult)

def test_oos_predictions_cover_all_samples(sample_X, sample_y):
    model = XGBoostPredictor(n_estimators=5)
    result = cross_val_predict_purged(model, sample_X, sample_y)
    assert len(result.oos_predictions) == len(sample_X)

def test_oos_accuracy_in_01(sample_X, sample_y):
    model = XGBoostPredictor(n_estimators=5)
    result = cross_val_predict_purged(model, sample_X, sample_y)
    assert 0.0 <= result.oos_accuracy <= 1.0

def test_feature_importance_aggregated(sample_X, sample_y):
    model = XGBoostPredictor(n_estimators=5)
    result = cross_val_predict_purged(model, sample_X, sample_y)
    assert isinstance(result.feature_importance, pd.Series)
    assert len(result.feature_importance) == sample_X.shape[1]
```

**Step 2:** `pytest tests/test_alpha_ml/test_cross_validate.py -q` → FAIL

**Step 3: Implement**

```python
@dataclass
class CVResult:
    oos_predictions: pd.Series
    oos_probabilities: pd.DataFrame  # shape (n, 2)
    fold_accuracies: list[float]
    oos_accuracy: float  # mean of fold_accuracies
    feature_importance: pd.Series  # mean across folds

def cross_val_predict_purged(model, X, y, n_splits=5, purge_window=5, embargo=2):
    pkf = PurgedKFold(n_splits=n_splits, purge_window=purge_window, embargo=embargo)
    # For each fold: fit on train, predict on test
    # Accumulate predictions, compute feature importance mean
    ...
```

**Step 4:** `pytest tests/test_alpha_ml/test_cross_validate.py -v` → ALL PASS

**Step 5: Commit**
```
git commit -m "feat(alpha-ml): purged cross-validation — CVResult with OOS predictions + feature importance"
```

---

### Task 5: ML Signal Generation Pipeline

**Why:** End-to-end: OHLCV → feature store → model → signals (as float Series) → run_backtest.

**Files:**
- Create: `src/alpha_ml/pipeline.py`
- Test: `tests/test_alpha_ml/test_ml_pipeline.py`

**Step 1: Write failing tests**

```python
from alpha_ml.pipeline import MLSignalPipeline

def test_returns_backtest_result(sample_ohlcv):
    pipeline = MLSignalPipeline(model=XGBoostPredictor(n_estimators=10))
    result = pipeline.run(sample_ohlcv)
    assert isinstance(result, BacktestResult)

def test_signals_generated_not_all_zero(sample_ohlcv):
    pipeline = MLSignalPipeline(model=XGBoostPredictor(n_estimators=10))
    signals = pipeline.generate_signals(sample_ohlcv)
    assert (signals != 0).any()

def test_signals_respect_no_lookahead(sample_ohlcv):
    """Train on first 60%, predict on last 40% only."""
    pipeline = MLSignalPipeline(model=XGBoostPredictor(n_estimators=10), train_ratio=0.6)
    signals = pipeline.generate_signals(sample_ohlcv)
    # First 60% should have 0 signal (train period, not traded)
    train_end = int(len(sample_ohlcv) * 0.6)
    assert (signals.iloc[:train_end] == 0).all()
```

**Step 2:** `pytest tests/test_alpha_ml/test_ml_pipeline.py -q` → FAIL

**Step 3: Implement `MLSignalPipeline`**
- `generate_signals()`: train on first `train_ratio` of data, predict on remainder
- Convert `predict_proba()[:, 1] > threshold` → +1 long / < (1-threshold) → -1 short / else 0
- Default threshold 0.55 (prediction confidence gate)
- Return signals that feed into `run_backtest()`

**Step 4:** `pytest tests/test_alpha_ml/test_ml_pipeline.py -v` → ALL PASS

**Step 5: Commit**
```
git commit -m "feat(alpha-ml): MLSignalPipeline — OHLCV → features → XGBoost → signals → backtest"
```

---

### Task 6: Full Suite + Smoke Test Update

**Files:**
- Modify: `scripts/run_backtest.py` — add ML pipeline section
- Run: full pytest suite
- Run: ruff clean
- Push: `git push origin main`

**Step 1: Update smoke test**

Add to `scripts/run_backtest.py`:
```python
from alpha_ml.pipeline import MLSignalPipeline
from alpha_ml.models.xgboost_model import XGBoostPredictor

ml_pipeline = MLSignalPipeline(model=XGBoostPredictor(n_estimators=100), train_ratio=0.6)
ml_result = ml_pipeline.run(df)
results.append(("XGBoost ML Signal", ml_result))
```

**Step 2: Verification checklist**

```powershell
# 1. All tests (no integration)
pytest -m "not integration" -q

# 2. Ruff clean
ruff check src/ tests/ scripts/

# 3. Smoke test
python scripts/run_backtest.py

# 4. Coverage
pytest --cov=src --cov-report=term-missing -m "not integration" -q
```

All 4 must pass before claiming Phase 2 done.

**Step 3: Commit + push**
```
git commit -m "feat(alpha-ml): Phase 2 complete — XGBoost ML signal pipeline with purged k-fold CV"
git push origin main
npx gitnexus analyze
```

---

## Verification Checklist (Phase 2 Done)

- [ ] `pytest -m "not integration" -q` → all GREEN (target: 120+ tests)
- [ ] `ruff check src/ tests/ scripts/` → `All checks passed!`
- [ ] `python scripts/run_backtest.py` → prints table including XGBoost row
- [ ] `pytest --cov=src --cov-report=term-missing -m "not integration" -q` → coverage report
- [ ] `npx gitnexus analyze` → updated node/edge count

## Risk Flags

- XGBoost install may need MSVC on Windows (`pip install xgboost` usually pre-built)
- `purge_window` must be ≥ prediction horizon to prevent label leakage
- Probability threshold of 0.55 is a hyperparameter — do not tune on test data
