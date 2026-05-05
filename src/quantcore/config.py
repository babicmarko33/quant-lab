"""Application configuration with environment variable loading and validation."""

from pathlib import Path

from pydantic_settings import BaseSettings


class MarketDataConfig(BaseSettings):
    """Market data provider configuration with fallback chain."""

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alphavantage_api_key: str = ""
    polygon_api_key: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


class BacktestConfig(BaseSettings):
    """Backtesting engine configuration."""

    commission_bps: int = 10  # basis points round-trip
    slippage_bps: int = 5
    default_benchmark: str = "SPY"
    initial_capital: float = 100_000.0

    model_config = {"env_prefix": "BACKTEST_", "env_file": ".env", "extra": "ignore"}


class AppConfig(BaseSettings):
    """Top-level application settings."""

    data_cache_dir: Path = Path("data/cache")
    log_level: str = "INFO"
    default_benchmark: str = "SPY"

    market_data: MarketDataConfig = MarketDataConfig()
    backtest: BacktestConfig = BacktestConfig()

    model_config = {"env_file": ".env", "extra": "ignore"}


# Singleton config instance — import this throughout the application
config = AppConfig()
