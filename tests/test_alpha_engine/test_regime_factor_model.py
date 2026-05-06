"""Tests for RegimeFactorModel — per-regime Fama-French OLS attribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.factor.regime_factor_model import RegimeFactorModel, RegimeFactorResult


@pytest.fixture
def monthly_data():
    """Synthetic 60-month returns + 3-factor data with 2 regime labels."""
    rng = np.random.default_rng(7)
    n = 60
    idx = pd.date_range("2019-01-01", periods=n, freq="ME")

    factors = pd.DataFrame({
        "Mkt_RF": rng.normal(0.005, 0.04, n),
        "SMB":    rng.normal(0.001, 0.02, n),
        "HML":    rng.normal(0.001, 0.02, n),
        "RF":     np.full(n, 0.0003),
    }, index=idx)

    # Returns: regime 0 = low-beta, regime 1 = high-beta
    regimes = np.array([0 if i % 3 != 2 else 1 for i in range(n)])
    betas = np.where(regimes == 0, 0.5, 1.5)
    returns = pd.Series(
        factors["RF"].values + betas * factors["Mkt_RF"].values + rng.normal(0, 0.01, n),
        index=idx,
    )

    return returns, factors, regimes


class TestRegimeFactorModel:
    def test_instantiation(self):
        """RegimeFactorModel can be created."""
        model = RegimeFactorModel()
        assert model is not None

    def test_fit_returns_dict(self, monthly_data):
        """fit() returns a dict keyed by regime label."""
        returns, factors, regimes = monthly_data
        model = RegimeFactorModel()
        result = model.fit(returns, factors, regimes)
        assert isinstance(result, dict)

    def test_keys_are_regime_labels(self, monthly_data):
        """Result keys are the unique regime integers."""
        returns, factors, regimes = monthly_data
        model = RegimeFactorModel()
        result = model.fit(returns, factors, regimes)
        assert set(result.keys()) == {0, 1}

    def test_values_are_regime_factor_results(self, monthly_data):
        """Each value is a RegimeFactorResult dataclass."""
        returns, factors, regimes = monthly_data
        model = RegimeFactorModel()
        result = model.fit(returns, factors, regimes)
        for v in result.values():
            assert isinstance(v, RegimeFactorResult)

    def test_regime_factor_result_has_fields(self, monthly_data):
        """RegimeFactorResult has alpha, betas, r_squared, n_obs fields."""
        returns, factors, regimes = monthly_data
        model = RegimeFactorModel()
        result = model.fit(returns, factors, regimes)
        for rfr in result.values():
            assert hasattr(rfr, "alpha")
            assert hasattr(rfr, "betas")
            assert hasattr(rfr, "r_squared")
            assert hasattr(rfr, "n_obs")

    def test_n_obs_sums_to_total(self, monthly_data):
        """Sum of n_obs across regimes equals total observations."""
        returns, factors, regimes = monthly_data
        model = RegimeFactorModel()
        result = model.fit(returns, factors, regimes)
        total = sum(v.n_obs for v in result.values())
        assert total == len(returns)

    def test_r_squared_in_range(self, monthly_data):
        """R² is in [0, 1] for all regimes."""
        returns, factors, regimes = monthly_data
        model = RegimeFactorModel()
        result = model.fit(returns, factors, regimes)
        for rfr in result.values():
            assert 0.0 <= rfr.r_squared <= 1.0

    def test_market_betas_differ_by_regime(self, monthly_data):
        """Low-vol regime has materially lower market beta than high-vol."""
        returns, factors, regimes = monthly_data
        model = RegimeFactorModel()
        result = model.fit(returns, factors, regimes)
        beta_0 = result[0].betas["Mkt_RF"]
        beta_1 = result[1].betas["Mkt_RF"]
        # Regime 1 constructed with 3x the beta of regime 0
        assert beta_1 > beta_0

    def test_betas_dict_has_factor_names(self, monthly_data):
        """betas dict contains Mkt_RF, SMB, HML keys."""
        returns, factors, regimes = monthly_data
        model = RegimeFactorModel()
        result = model.fit(returns, factors, regimes)
        for rfr in result.values():
            assert "Mkt_RF" in rfr.betas
            assert "SMB" in rfr.betas
            assert "HML" in rfr.betas

    def test_summary_dataframe(self, monthly_data):
        """summary() returns a DataFrame with one row per regime."""
        returns, factors, regimes = monthly_data
        model = RegimeFactorModel()
        result = model.fit(returns, factors, regimes)
        df = RegimeFactorModel.summary(result)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "alpha" in df.columns
        assert "r_squared" in df.columns
        assert "n_obs" in df.columns
