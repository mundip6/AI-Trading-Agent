from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str      # "buy" or "sell"
    qty: float     # base-asset quantity that actually changed hands, net of fees
    price: float   # average fill price
    fee: float     # fee in quote currency

    @property
    def value(self) -> float:
        return self.qty * self.price


class Broker(ABC):
    """Execution venue. Implementations: internal simulator, exchange demo account."""

    name: str

    @abstractmethod
    def price(self, symbol: str) -> float:
        """Latest traded price."""

    @abstractmethod
    def cash(self) -> float:
        """Free quote-currency balance."""

    def market_open(self, symbol: str) -> bool:
        """Whether orders for the symbol can fill right now. Crypto: always."""
        return True

    @abstractmethod
    def buy(self, symbol: str, qty: float) -> Fill:
        """Market buy."""

    @abstractmethod
    def sell(self, symbol: str, qty: float) -> Fill:
        """Market sell."""
