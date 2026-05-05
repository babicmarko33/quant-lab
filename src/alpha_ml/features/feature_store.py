"""ML-ready feature store.

Wraps the quantcore feature pipeline and adds ML-specific transforms:
  1. Z-score normalization (zero mean, unit variance per feature)
  2. Winsorization (clip extreme outliers before z-scoring)
  3. NaN imputation (drop rows with insufficient history)
  4. Target label construction (direction or raw return)

All transforms are computed IN-SAMPLE only — the caller is responsible for
fitting on train data and transforming test data separately to prevent leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

from quantcore.features.pipeline import add_target, build_features


class FeatureStore:
    """Build ML-ready (X, y) pairs from raw OHLCV data.

    Parameters
    ----------
    winsorize_pct : float
        Fraction to clip from each tail (e.g. 0.01 = 1% each side).
        Set to 0.0 to disable winsorization.
    """

    def __init__(self, winsorize_pct: float = 0.01) -> None:
        self.winsorize_pct = winsorize_pct

    def build(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        target_type: str = "direction",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build feature matrix X and target y.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data.
        horizon : int
            Forward return horizon in bars for target construction.
        target_type : str
            'direction' → binary {0, 1}
            'return'    → raw forward log-return

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            (X, y) with aligned index, no NaN values.
        """
        # Build feature matrix using existing pipeline
        features_df = build_features(df)

        # Add target — pass close series (not full df)
        features_with_target = add_target(features_df, df["close"], horizon=horizon, target_type=target_type)

        # Separate X and y
        y_raw = features_with_target["target"].copy()
        X_raw = features_with_target.drop(columns=["target"])

        # Drop rows where y is NaN (end of series — no forward return available)
        valid_mask = y_raw.notna()
        y_raw = y_raw[valid_mask]
        X_raw = X_raw[valid_mask]

        # Drop rows where any X feature is NaN (warmup period)
        X_clean_mask = ~X_raw.isna().any(axis=1)
        X_raw = X_raw[X_clean_mask]
        y_raw = y_raw[X_clean_mask]

        # Winsorize each feature column
        X_winsorized = X_raw.copy()
        if self.winsorize_pct > 0:
            for col in X_winsorized.columns:
                X_winsorized[col] = np.array(
                    winsorize(X_raw[col].values, limits=[self.winsorize_pct, self.winsorize_pct])
                )

        # Z-score normalize
        means = X_winsorized.mean()
        stds = X_winsorized.std(ddof=1)
        # Avoid division by zero for constant columns
        stds = stds.replace(0.0, 1.0)
        X_normalized = (X_winsorized - means) / stds

        # Cast target to correct dtype
        if target_type == "direction":
            y_out = y_raw.astype(int)
        else:
            y_out = y_raw.astype(float)

        return X_normalized, y_out
