"""Internal simulator: real live prices, pretend money."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from trading_agent.broker.base import Broker, Fill

if TYPE_CHECKING:
    from trading_agent.journal import Journal

PriceFeed = Callable[[str], float]

_CASH_KEY = "paper_cash"


def ccxt_price_feed(exchange) -> PriceFeed:
    def feed(symbol: str) -> float:
        return float(exchange.fetch_ticker(symbol)["last"])

    return feed


class PaperBroker(Broker):
    name = "paper"

    def __init__(
        self,
        journal: Journal,
        price_feed: PriceFeed,
        initial_cash: float,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        market_open: Callable[[str], bool] | None = None,
    ):
        self._journal = journal
        self._feed = price_feed
        self._market_open = market_open
        self._fee_rate = fee_rate
        self._slippage = slippage
        if journal.get(_CASH_KEY) is None:
            journal.set(_CASH_KEY, initial_cash)

    def price(self, symbol: str) -> float:
        return self._feed(symbol)

    def cash(self) -> float:
        return float(self._journal.get(_CASH_KEY))

    def market_open(self, symbol: str) -> bool:
        return True if self._market_open is None else self._market_open(symbol)

    def buy(self, symbol: str, qty: float) -> Fill:
        price = self.price(symbol) * (1 + self._slippage)
        fee = qty * price * self._fee_rate
        cost = qty * price + fee
        if cost > self.cash():
            raise ValueError(f"Insufficient cash: need {cost:.2f}, have {self.cash():.2f}")
        self._journal.set(_CASH_KEY, self.cash() - cost)
        return Fill(symbol, "buy", qty, price, fee)

    def sell(self, symbol: str, qty: float) -> Fill:
        price = self.price(symbol) * (1 - self._slippage)
        fee = qty * price * self._fee_rate
        self._journal.set(_CASH_KEY, self.cash() + qty * price - fee)
        return Fill(symbol, "sell", qty, price, fee)
