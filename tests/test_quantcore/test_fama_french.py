"""Tests for Fama-French factor fetcher and factor model."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alpha_engine.factor.factor_model import FactorModel, FactorResult
from quantcore.data.fama_french import FamaFrenchFetcher

# ---------------------------------------------------------------------------
# FamaFrenchFetcher
# ---------------------------------------------------------------------------

# Minimal CSV as returned by Kenneth French's data library (tab-separated, with header rows)
_FF3_CSV = """\
Fama/French 3 Factors
    -- date --,Mkt-RF,SMB,HML,RF
    202401,1.23,0.45,-0.12,0.42
    202402,-0.56,0.22,0.33,0.41
    202403,2.10,-0.31,0.15,0.43
"""


def test_fetcher_instantiation():
    """FamaFrenchFetcher can be instantiated."""
    fetcher = FamaFrenchFetcher()
    assert fetcher is not None


def test_fetcher_returns_dataframe():
    """fetch_3_factor() returns a non-empty DataFrame."""
    fetcher = FamaFrenchFetcher()
    mock_resp = MagicMock()
    mock_resp.text = _FF3_CSV
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        df = fetcher.fetch_3_factor()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_fetcher_3factor_columns():
    """fetch_3_factor() DataFrame has Mkt_RF, SMB, HML, RF columns."""
    fetcher = FamaFrenchFetcher()
    mock_resp = MagicMock()
    mock_resp.text = _FF3_CSV
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        df = fetcher.fetch_3_factor()

    assert "Mkt_RF" in df.columns
    assert "SMB" in df.columns
    assert "HML" in df.columns
    assert "RF" in df.columns


def test_fetcher_returns_decimal_returns():
    """Factor columns are in decimal form (÷ 100)."""
    fetcher = FamaFrenchFetcher()
    mock_resp = MagicMock()
    mock_resp.text = _FF3_CSV
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        df = fetcher.fetch_3_factor()

    # 1.23% input → 0.0123 in output
    assert df["Mkt_RF"].abs().max() < 0.5


def test_fetcher_index_is_datetime():
    """fetch_3_factor() index is DatetimeIndex."""
    fetcher = FamaFrenchFetcher()
    mock_resp = MagicMock()
    mock_resp.text = _FF3_CSV
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        df = fetcher.fetch_3_factor()

    assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# FactorModel
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_factors_and_returns():
    """Synthetic 3-factor + returns for model tests."""
    rng = np.random.default_rng(42)
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="ME")
    factors = pd.DataFrame({
        "Mkt_RF": rng.normal(0.005, 0.04, n),
        "SMB": rng.normal(0.001, 0.02, n),
        "HML": rng.normal(0.001, 0.02, n),
        "RF": rng.normal(0.0003, 0.0001, n),
    }, index=idx)
    # Create a return series with known beta~1 on Mkt_RF
    returns = factors["RF"] + 1.0 * factors["Mkt_RF"] + rng.normal(0, 0.01, n)
    returns.index = idx
    return factors, returns


def test_factor_model_instantiation():
    """FactorModel can be instantiated."""
    model = FactorModel()
    assert model is not None


def test_factor_model_fit_returns_result(sample_factors_and_returns):
    """fit() returns a FactorResult."""
    factors, returns = sample_factors_and_returns
    model = FactorModel()
    result = model.fit(returns, factors)
    assert isinstance(result, FactorResult)


def test_factor_result_has_alphas_betas(sample_factors_and_returns):
    """FactorResult has alpha and betas dict."""
    factors, returns = sample_factors_and_returns
    model = FactorModel()
    result = model.fit(returns, factors)
    assert hasattr(result, "alpha")
    assert hasattr(result, "betas")
    assert "Mkt_RF" in result.betas


def test_factor_result_market_beta_near_one(sample_factors_and_returns):
    """Market beta is near 1.0 for a synthetically constructed return."""
    factors, returns = sample_factors_and_returns
    model = FactorModel()
    result = model.fit(returns, factors)
    assert abs(result.betas["Mkt_RF"] - 1.0) < 0.3


def test_factor_result_r_squared_in_range(sample_factors_and_returns):
    """R² is between 0 and 1."""
    factors, returns = sample_factors_and_returns
    model = FactorModel()
    result = model.fit(returns, factors)
    assert 0.0 <= result.r_squared <= 1.0


def test_factor_result_residuals_shape(sample_factors_and_returns):
    """Residuals array has same length as input returns."""
    factors, returns = sample_factors_and_returns
    model = FactorModel()
    result = model.fit(returns, factors)
    assert len(result.residuals) == len(returns)
