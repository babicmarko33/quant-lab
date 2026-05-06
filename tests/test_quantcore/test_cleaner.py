import numpy as np
import pandas as pd
import pytest

from quantcore.data.cleaner import (
    clean_ohlcv,
    detect_outliers,
    validate_ohlcv,
)


@pytest.fixture
def clean_df():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(10)],
            "high": [102.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [101.0 + i for i in range(10)],
            "volume": [1_000_000 + i * 100 for i in range(10)],
        },
        index=dates,
    )


class TestValidateOHLCV:
    def test_passes_clean_data(self, clean_df):
        validate_ohlcv(clean_df)  # should not raise

    def test_raises_on_missing_column(self, clean_df):
        with pytest.raises(ValueError, match="Missing columns"):
            validate_ohlcv(clean_df.drop(columns=["volume"]))

    def test_raises_on_zero_volume(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[3], "volume"] = 0
        with pytest.raises(ValueError, match="volume"):
            validate_ohlcv(df)

    def test_raises_on_negative_close(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[0], "close"] = -1.0
        with pytest.raises(ValueError, match="negative"):
            validate_ohlcv(df)

    def test_raises_on_high_below_low(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[2], "high"] = df.loc[df.index[2], "low"] - 1
        with pytest.raises(ValueError, match="high.*low"):
            validate_ohlcv(df)


class TestDetectOutliers:
    def test_no_outliers_in_clean_data(self, clean_df):
        mask = detect_outliers(clean_df["close"], z_threshold=3.0)
        assert not mask.any()

    def test_detects_spike(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[5], "close"] = 9999.0  # massive spike
        mask = detect_outliers(df["close"], z_threshold=3.0)
        assert mask.loc[df.index[5]]

    def test_returns_bool_series(self, clean_df):
        mask = detect_outliers(clean_df["close"])
        assert mask.dtype == bool
        assert len(mask) == len(clean_df)


class TestCleanOHLCV:
    def test_fills_nan_with_ffill(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[3], "close"] = np.nan
        result = clean_ohlcv(df)
        assert not result["close"].isna().any()

    def test_removes_outliers_when_requested(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[5], "close"] = 9999.0
        result = clean_ohlcv(df, remove_outliers=True, z_threshold=3.0)
        # Outlier replaced by NaN then forward-filled
        assert result.loc[df.index[5], "close"] != 9999.0

    def test_returns_dataframe(self, clean_df):
        result = clean_ohlcv(clean_df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_clean_data_unchanged(self, clean_df):
        result = clean_ohlcv(clean_df)
        pd.testing.assert_frame_equal(result, clean_df)

    def test_drops_rows_with_non_positive_volume(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[1], "volume"] = 0
        result = clean_ohlcv(df, drop_zero_volume=True)
        assert len(result) == len(clean_df) - 1

    def test_monotonic_date_index(self, clean_df):
        """Output index is always sorted ascending."""
        df = clean_df.iloc[::-1].copy()  # reverse order
        result = clean_ohlcv(df)
        assert result.index.is_monotonic_increasing
