"""Strategy registry and public API for alpha_engine.strategies."""

from alpha_engine.strategies.base import Strategy
from alpha_engine.strategies.ema_crossover import EMACrossoverStrategy
from alpha_engine.strategies.mean_reversion import BollingerMeanReversionStrategy
from alpha_engine.strategies.momentum import MomentumStrategy
from alpha_engine.strategies.rsi_strategy import RSIMeanReversionStrategy
from alpha_engine.strategies.sma_crossover import SMACrossoverStrategy

__all__ = [
    "Strategy",
    "MomentumStrategy",
    "BollingerMeanReversionStrategy",
    "SMACrossoverStrategy",
    "EMACrossoverStrategy",
    "RSIMeanReversionStrategy",
]

REGISTRY: dict[str, type[Strategy]] = {
    "momentum": MomentumStrategy,
    "bollinger_mr": BollingerMeanReversionStrategy,
    "sma_crossover": SMACrossoverStrategy,
    "ema_crossover": EMACrossoverStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
}
