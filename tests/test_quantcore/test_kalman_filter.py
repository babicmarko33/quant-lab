"""Tests for Kalman filter pairs trading module. Written before implementation (TDD RED)."""
import numpy as np
import pandas as pd
import pytest

from quantcore.signals.kalman_filter import KalmanPairFilter

N = 300


@pytest.fixture
def cointegrated_prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = pd.Series(np.cumsum(rng.standard_normal(N)) + 100.0, name="X")
    y = pd.Series(1.5 * x + rng.standard_normal(N) * 2.0, name="Y")
    return pd.DataFrame({"X": x, "Y": y})


class TestKalmanPairFilter:
    def test_instantiation(self):
        kf = KalmanPairFilter()
        assert isinstance(kf, KalmanPairFilter)

    def test_fit_returns_self(self, cointegrated_prices):
        kf = KalmanPairFilter()
        result = kf.fit(cointegrated_prices["X"], cointegrated_prices["Y"])
        assert result is kf

    def test_spread_shape(self, cointegrated_prices):
        kf = KalmanPairFilter()
        kf.fit(cointegrated_prices["X"], cointegrated_prices["Y"])
        spread = kf.spread_
        assert spread.shape == (N,)

    def test_hedge_ratio_shape(self, cointegrated_prices):
        kf = KalmanPairFilter()
        kf.fit(cointegrated_prices["X"], cointegrated_prices["Y"])
        hr = kf.hedge_ratio_
        assert hr.shape == (N,)

    def test_hedge_ratio_converges_near_true(self, cointegrated_prices):
        """After burn-in, hedge ratio should be close to 1.5."""
        kf = KalmanPairFilter()
        kf.fit(cointegrated_prices["X"], cointegrated_prices["Y"])
        # Use last 100 observations (post burn-in)
        mean_hr = float(kf.hedge_ratio_[-100:].mean())
        assert abs(mean_hr - 1.5) < 0.5, f"Expected ~1.5, got {mean_hr:.4f}"

    def test_zscore_shape(self, cointegrated_prices):
        kf = KalmanPairFilter()
        kf.fit(cointegrated_prices["X"], cointegrated_prices["Y"])
        z = kf.zscore_
        assert z.shape == (N,)

    def test_zscore_mean_near_zero(self, cointegrated_prices):
        """Z-score of spread should be approximately mean-zero after burn-in."""
        kf = KalmanPairFilter()
        kf.fit(cointegrated_prices["X"], cointegrated_prices["Y"])
        mean_z = float(kf.zscore_[-100:].mean())
        assert abs(mean_z) < 1.5, f"Z-score mean {mean_z:.4f} too far from zero"

    def test_raises_before_fit(self):
        kf = KalmanPairFilter()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = kf.spread_

    def test_raises_on_mismatched_lengths(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.standard_normal(100))
        y = pd.Series(rng.standard_normal(90))
        kf = KalmanPairFilter()
        with pytest.raises(ValueError, match="same length"):
            kf.fit(x, y)
