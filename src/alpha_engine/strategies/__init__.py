"""Strategy registry and public API for alpha_engine.strategies."""

from alpha_engine.strategies.base import Strategy
from alpha_engine.strategies.mean_reversion import BollingerMeanReversionStrategy
from alpha_engine.strategies.momentum import MomentumStrategy

__all__ = [
    "Strategy",
    "MomentumStrategy",
    "BollingerMeanReversionStrategy",
]

REGISTRY: dict[str, type[Strategy]] = {
    "momentum": MomentumStrategy,
    "bollinger_mr": BollingerMeanReversionStrategy,
}
