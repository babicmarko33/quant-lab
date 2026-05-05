"""Tests for quantcore.data.fetcher module."""

import pandas as pd
import pytest

from quantcore.data.fetcher import OHLCV_COLUMNS, fetch_ohlcv


class TestFetchOHLCV:
    """Integration tests for data fetcher — requires internet connection."""

    @pytest.mark.integration
    def test_fetch_spy_daily(self) -> None:
        """Fetch SPY daily data and verify schema."""
        df = fetch_ohlcv("SPY", start="2024-01-01", end="2024-03-01", use_cache=False)

        # Verify all required columns present
        for col in OHLCV_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

        # Verify index
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "date"
        assert df.index.is_monotonic_increasing

        # Verify data quality
        assert len(df) > 30  # ~2 months of trading days
        assert (df["close"] > 0).all()
        assert (df["volume"] > 0).all()
        assert (df["high"] >= df["low"]).all()

    @pytest.mark.integration
    def test_fetch_multiple_tickers(self) -> None:
        """Fetch multiple tickers."""
        from quantcore.data.fetcher import fetch_multiple

        results = fetch_multiple(["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01")
        assert "AAPL" in results
        assert "MSFT" in results
        assert len(results["AAPL"]) > 15

    def test_invalid_ticker_raises(self) -> None:
        """Invalid ticker should raise RuntimeError after all providers fail."""
        with pytest.raises(RuntimeError, match="All providers failed"):
            fetch_ohlcv("ZZZZZZZZZ_INVALID", start="2024-01-01", end="2024-02-01", use_cache=False)
