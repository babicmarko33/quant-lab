"""Tradier brokerage REST API client for options chain data.

Fetches live option chains, expirations, and quotes from Tradier's
sandbox (default) or production API.

Authentication requires a Tradier Bearer token set via:
  - environment variable ``TRADIER_TOKEN``
  - or passed directly to ``TradierClient(token=...)``
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
import requests


_SANDBOX_URL = "https://sandbox.tradier.com/v1"
_PRODUCTION_URL = "https://api.tradier.com/v1"


@dataclass
class OptionQuote:
    """Single option contract quote with greeks.

    Attributes
    ----------
    symbol:
        OCC option symbol.
    strike:
        Strike price in USD.
    expiration_date:
        Expiry as ``YYYY-MM-DD`` string.
    option_type:
        ``"call"`` or ``"put"``.
    bid:
        Best bid.
    ask:
        Best ask.
    last:
        Last traded price.
    volume:
        Today's traded volume.
    open_interest:
        Open interest.
    mid_iv:
        Mid implied volatility (decimal, e.g. 0.18 = 18%).
    delta:
        Black-Scholes delta.
    gamma:
        Black-Scholes gamma.
    theta:
        Black-Scholes theta (per day).
    vega:
        Black-Scholes vega.
    """

    symbol: str
    strike: float
    expiration_date: str
    option_type: str
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    mid_iv: float
    delta: float
    gamma: float
    theta: float
    vega: float

    @classmethod
    def from_dict(cls, data: dict) -> OptionQuote:
        """Parse a Tradier option JSON record."""
        greeks = data.get("greeks") or {}
        return cls(
            symbol=data.get("symbol", ""),
            strike=float(data.get("strike", 0.0)),
            expiration_date=data.get("expiration_date", ""),
            option_type=data.get("option_type", data.get("type", "")),
            bid=float(data.get("bid") or 0.0),
            ask=float(data.get("ask") or 0.0),
            last=float(data.get("last") or 0.0),
            volume=int(data.get("volume") or 0),
            open_interest=int(data.get("open_interest") or 0),
            mid_iv=float(greeks.get("mid_iv") or 0.0),
            delta=float(greeks.get("delta") or 0.0),
            gamma=float(greeks.get("gamma") or 0.0),
            theta=float(greeks.get("theta") or 0.0),
            vega=float(greeks.get("vega") or 0.0),
        )


@dataclass
class OptionChain:
    """Parsed option chain for a single expiration.

    Attributes
    ----------
    ticker:
        Underlying symbol.
    expiration_date:
        Expiry date string.
    calls:
        List of call ``OptionQuote`` objects.
    puts:
        List of put ``OptionQuote`` objects.
    """

    ticker: str
    expiration_date: str
    calls: list[OptionQuote] = field(default_factory=list)
    puts: list[OptionQuote] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all options (calls + puts) as a flat DataFrame.

        Columns include ``strike``, ``option_type``, ``bid``, ``ask``,
        ``last``, ``volume``, ``open_interest``, ``mid_iv``,
        ``delta``, ``gamma``, ``theta``, ``vega``.
        """
        all_quotes = self.calls + self.puts
        if not all_quotes:
            return pd.DataFrame()
        records = [
            {
                "symbol": q.symbol,
                "strike": q.strike,
                "expiration_date": q.expiration_date,
                "option_type": q.option_type,
                "bid": q.bid,
                "ask": q.ask,
                "last": q.last,
                "volume": q.volume,
                "open_interest": q.open_interest,
                "mid_iv": q.mid_iv,
                "delta": q.delta,
                "gamma": q.gamma,
                "theta": q.theta,
                "vega": q.vega,
            }
            for q in all_quotes
        ]
        return pd.DataFrame(records)


class TradierClient:
    """HTTP client for Tradier market data API.

    Parameters
    ----------
    token:
        Tradier Bearer token. Falls back to ``TRADIER_TOKEN`` env var.
    sandbox:
        Use sandbox endpoint (default). Set ``False`` for production.
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        token: str | None = None,
        sandbox: bool = True,
        timeout: int = 10,
    ) -> None:
        resolved = token or os.environ.get("TRADIER_TOKEN", "")
        self.base_url = _SANDBOX_URL if sandbox else _PRODUCTION_URL
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {resolved}",
            "Accept": "application/json",
        })

    def get_expirations(self, ticker: str) -> list[str]:
        """Return sorted list of available expiration date strings.

        Parameters
        ----------
        ticker:
            Underlying ticker symbol (e.g. ``"SPY"``).

        Returns
        -------
        list[str]
            Sorted list of ``YYYY-MM-DD`` expiration strings.
        """
        url = f"{self.base_url}/markets/options/expirations"
        resp = self._session.get(url, params={"symbol": ticker}, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        dates = data.get("expirations", {}).get("date", [])
        if isinstance(dates, str):
            dates = [dates]
        return sorted(dates)

    def get_option_chain(self, ticker: str, expiration: str) -> OptionChain:
        """Fetch the full option chain for a given expiration.

        Parameters
        ----------
        ticker:
            Underlying ticker.
        expiration:
            Expiry date string ``YYYY-MM-DD``.

        Returns
        -------
        OptionChain
            Parsed chain with calls and puts.
        """
        url = f"{self.base_url}/markets/options/chains"
        resp = self._session.get(
            url,
            params={"symbol": ticker, "expiration": expiration, "greeks": "true"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_options = data.get("options", {}).get("option", []) or []
        if isinstance(raw_options, dict):
            raw_options = [raw_options]

        quotes = [OptionQuote.from_dict(o) for o in raw_options]
        chain = OptionChain(
            ticker=ticker,
            expiration_date=expiration,
            calls=[q for q in quotes if q.option_type == "call"],
            puts=[q for q in quotes if q.option_type == "put"],
        )
        return chain

    def get_quote(self, ticker: str) -> float:
        """Return the last traded price for an equity ticker.

        Parameters
        ----------
        ticker:
            Equity ticker symbol.

        Returns
        -------
        float
            Last traded price.
        """
        url = f"{self.base_url}/markets/quotes"
        resp = self._session.get(url, params={"symbols": ticker}, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        quote = data.get("quotes", {}).get("quote", {})
        return float(quote.get("last", 0.0))
