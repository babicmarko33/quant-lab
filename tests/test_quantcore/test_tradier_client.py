"""Tests for Tradier options data client.

All HTTP calls are mocked — no real API key needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantcore.data.tradier_client import (
    OptionChain,
    OptionQuote,
    TradierClient,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHAIN_JSON = {
    "options": {
        "option": [
            {
                "symbol": "SPY241220C00500000",
                "description": "SPY Dec 20 2024 500.00 Call",
                "exch": "Z",
                "type": "call",
                "last": 12.5,
                "bid": 12.3,
                "ask": 12.7,
                "volume": 1234,
                "open_interest": 5678,
                "strike": 500.0,
                "expiration_date": "2024-12-20",
                "option_type": "call",
                "greeks": {
                    "delta": 0.55,
                    "gamma": 0.02,
                    "theta": -0.08,
                    "vega": 0.15,
                    "mid_iv": 0.18,
                },
            },
            {
                "symbol": "SPY241220P00500000",
                "description": "SPY Dec 20 2024 500.00 Put",
                "exch": "Z",
                "type": "put",
                "last": 10.2,
                "bid": 10.0,
                "ask": 10.4,
                "volume": 987,
                "open_interest": 3456,
                "strike": 500.0,
                "expiration_date": "2024-12-20",
                "option_type": "put",
                "greeks": {
                    "delta": -0.45,
                    "gamma": 0.02,
                    "theta": -0.07,
                    "vega": 0.15,
                    "mid_iv": 0.19,
                },
            },
        ]
    }
}

SAMPLE_EXPIRATIONS_JSON = {
    "expirations": {
        "date": ["2024-12-20", "2025-01-17", "2025-03-21"]
    }
}

SAMPLE_QUOTE_JSON = {
    "quotes": {
        "quote": {
            "symbol": "SPY",
            "last": 498.5,
            "bid": 498.4,
            "ask": 498.6,
        }
    }
}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_client_instantiation():
    """TradierClient can be created with an API token."""
    client = TradierClient(token="test-token-abc")
    assert client is not None


def test_client_default_sandbox():
    """Default base URL is the sandbox endpoint."""
    client = TradierClient(token="test-token")
    assert "sandbox" in client.base_url


def test_client_production_url():
    """Production URL is selectable."""
    client = TradierClient(token="test-token", sandbox=False)
    assert "api.tradier.com" in client.base_url
    assert "sandbox" not in client.base_url


# ---------------------------------------------------------------------------
# get_expirations
# ---------------------------------------------------------------------------

def test_get_expirations_returns_list():
    """get_expirations returns a list of date strings."""
    client = TradierClient(token="test-token")
    mock_resp = MagicMock()
    mock_resp.json.return_value = SAMPLE_EXPIRATIONS_JSON
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client._session, "get", return_value=mock_resp):
        expirations = client.get_expirations("SPY")

    assert isinstance(expirations, list)
    assert len(expirations) == 3
    assert expirations[0] == "2024-12-20"


# ---------------------------------------------------------------------------
# get_option_chain
# ---------------------------------------------------------------------------

def test_get_option_chain_returns_option_chain():
    """get_option_chain returns an OptionChain object."""
    client = TradierClient(token="test-token")
    mock_resp = MagicMock()
    mock_resp.json.return_value = SAMPLE_CHAIN_JSON
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client._session, "get", return_value=mock_resp):
        chain = client.get_option_chain("SPY", "2024-12-20")

    assert isinstance(chain, OptionChain)


def test_option_chain_has_calls_and_puts():
    """OptionChain splits options into calls and puts."""
    client = TradierClient(token="test-token")
    mock_resp = MagicMock()
    mock_resp.json.return_value = SAMPLE_CHAIN_JSON
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client._session, "get", return_value=mock_resp):
        chain = client.get_option_chain("SPY", "2024-12-20")

    assert len(chain.calls) == 1
    assert len(chain.puts) == 1


def test_option_quote_fields():
    """OptionQuote dataclass has required fields."""
    client = TradierClient(token="test-token")
    mock_resp = MagicMock()
    mock_resp.json.return_value = SAMPLE_CHAIN_JSON
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client._session, "get", return_value=mock_resp):
        chain = client.get_option_chain("SPY", "2024-12-20")

    call = chain.calls[0]
    assert isinstance(call, OptionQuote)
    assert call.strike == 500.0
    assert call.option_type == "call"
    assert call.mid_iv == pytest.approx(0.18)
    assert call.delta == pytest.approx(0.55)


def test_option_chain_to_dataframe():
    """OptionChain.to_dataframe returns a pd.DataFrame with strike column."""
    client = TradierClient(token="test-token")
    mock_resp = MagicMock()
    mock_resp.json.return_value = SAMPLE_CHAIN_JSON
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client._session, "get", return_value=mock_resp):
        chain = client.get_option_chain("SPY", "2024-12-20")

    df = chain.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "strike" in df.columns
    assert "mid_iv" in df.columns
    assert "option_type" in df.columns
    assert len(df) == 2


# ---------------------------------------------------------------------------
# get_quote
# ---------------------------------------------------------------------------

def test_get_quote_returns_float():
    """get_quote returns the last price as a float."""
    client = TradierClient(token="test-token")
    mock_resp = MagicMock()
    mock_resp.json.return_value = SAMPLE_QUOTE_JSON
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client._session, "get", return_value=mock_resp):
        price = client.get_quote("SPY")

    assert isinstance(price, float)
    assert price == pytest.approx(498.5)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_get_option_chain_raises_on_http_error():
    """get_option_chain propagates HTTP errors."""
    import requests

    client = TradierClient(token="test-token")
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")

    with patch.object(client._session, "get", return_value=mock_resp), pytest.raises(requests.HTTPError):
        client.get_option_chain("SPY", "2024-12-20")
