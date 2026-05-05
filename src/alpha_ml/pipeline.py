"""End-to-end ML signal generation pipeline.

Pipeline flow:
  1. Build feature matrix from full OHLCV history
  2. Split into train (IS) and predict (OOS) periods
  3. Fit model on IS feature matrix
  4. Generate probability predictions for OOS period
  5. Convert probabilities → signals: +1 (long), -1 (short), 0 (neutral)
  6. Run vectorized backtest on OOS signals

No data from the OOS period is used to train the model.
Train period contributes zero signals (no trading in IS period).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.backtest.engine import run_backtest
from alpha_engine.backtest.types import BacktestResult
from alpha_ml.features.feature_store import FeatureStore
from alpha_ml.models.base import ModelTrainer


class MLSignalPipeline:
    """Supervised ML signal pipeline: OHLCV → features → model → signals → BacktestResult.

    Parameters
    ----------
    model : ModelTrainer
        Any concrete ModelTrainer (e.g. XGBoostPredictor).
    train_ratio : float
        Fraction of data used for training. Default 0.6 (60% IS, 40% OOS).
    horizon : int
        Forward return horizon used to construct target labels. Default 5.
    long_threshold : float
        Probability threshold above which signal = +1 (long). Default 0.55.
    short_threshold : float
        Probability threshold below which signal = -1 (short). Default 0.45.
    winsorize_pct : float
        Winsorization percentile for FeatureStore. Default 0.01.
    """

    def __init__(
        self,
        model: ModelTrainer,
        train_ratio: float = 0.6,
        horizon: int = 5,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        winsorize_pct: float = 0.01,
    ) -> None:
        self.model = model
        self.train_ratio = train_ratio
        self.horizon = horizon
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self._feature_store = FeatureStore(winsorize_pct=winsorize_pct)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Full OHLCV history.

        Returns
        -------
        pd.Series
            Float signals in {-1.0, 0.0, +1.0}, indexed like df.
            Train period (first train_ratio fraction) → 0.
        """
        # Build complete feature matrix + labels
        X, y = self._feature_store.build(df, horizon=self.horizon, target_type="direction")

        # Determine train/predict split using the cleaned index
        n = len(X)
        train_end = int(n * self.train_ratio)

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_oos = X.iloc[train_end:]

        # Fit on IS data only
        import copy
        fold_model = copy.deepcopy(self.model)
        fold_model.fit(X_train, y_train)

        # Predict probabilities on OOS
        proba_oos = fold_model.predict_proba(X_oos)  # shape (n_oos, 2)
        prob_up = proba_oos[:, 1]

        # Convert to signals
        oos_signals = np.zeros(len(X_oos))
        oos_signals[prob_up >= self.long_threshold] = 1.0
        oos_signals[prob_up <= self.short_threshold] = -1.0

        oos_signal_series = pd.Series(oos_signals, index=X_oos.index)

        # Build full signal series (train period = 0)
        full_signals = pd.Series(0.0, index=df.index)
        full_signals.update(oos_signal_series)

        return full_signals

    def run(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        commission_bps: int = 10,
        slippage_bps: int = 5,
    ) -> BacktestResult:
        """Generate ML signals and run full backtest.

        Parameters
        ----------
        df : pd.DataFrame
            Full OHLCV history.
        initial_capital : float
            Starting capital.
        commission_bps : int
            One-way commission in basis points.
        slippage_bps : int
            One-way slippage in basis points.

        Returns
        -------
        BacktestResult
        """
        signals = self.generate_signals(df)
        return run_backtest(signals, df, initial_capital, commission_bps, slippage_bps)
