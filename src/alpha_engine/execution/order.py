"""Order dataclass for the execution layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

VALID_SIDES = {"buy", "sell"}


@dataclass
class Order:
    """Represents a single market order.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. ``"SPY"``).
    side : str
        ``"buy"`` or ``"sell"``.
    qty : float
        Positive number of shares/units.
    order_id : str | None
        Unique order identifier. Auto-generated UUID if not provided.
    status : str
        Order status. Defaults to ``"pending"``.
    """

    symbol: str
    side: str
    qty: float
    order_id: str | None = field(default=None)
    status: str = "pending"

    def __post_init__(self) -> None:
        if self.side not in VALID_SIDES:
            raise ValueError(f"side must be one of {VALID_SIDES}, got '{self.side}'")
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")
        if self.order_id is None:
            object.__setattr__(self, "order_id", str(uuid.uuid4()))
