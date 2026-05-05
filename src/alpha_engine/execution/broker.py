"""Abstract Broker interface."""

from abc import ABC, abstractmethod

from alpha_engine.execution.order import Order


class Broker(ABC):
    """Abstract base class for all broker adapters."""

    @abstractmethod
    def submit(self, order: Order) -> Order:
        """Submit an order and return an updated copy with broker-assigned id/status."""
