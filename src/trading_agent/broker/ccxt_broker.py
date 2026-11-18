"""Exchange demo account (testnet) through ccxt.

Deliberately sandbox-only: there is no code path here that reaches a real
account. Live trading is a later phase with its own safeguards.
"""

from __future__ import annotations

import os

import ccxt

from trading_agent.broker.base import Broker, Fill


class CcxtTestnetBroker(Broker):
    def __init__(self, exchange_id: str, api_key: str, secret: str, quote: str = "USDT"):
        self.name = f"{exchange_id}_testnet"
        self._quote = quote
        self._ex = getattr(ccxt, exchange_id)(
            {"apiKey": api_key, "secret": secret, "enableRateLimit": True}
        )
        self._ex.set_sandbox_mode(True)  # must come before any request
        self._ex.load_markets()

    @classmethod
    def from_env(cls, exchange_id: str) -> "CcxtTestnetBroker":
        prefix = f"{exchange_id.upper()}_TESTNET"
        key = os.environ.get(f"{prefix}_API_KEY")
        secret = os.environ.get(f"{prefix}_API_SECRET")
        if not key or not secret:
            raise RuntimeError(
                f"Set {prefix}_API_KEY and {prefix}_API_SECRET to use the {exchange_id} testnet "
                "(on the app's API keys page, or in .env)"
            )
        return cls(exchange_id, key, secret)

    def price(self, symbol: str) -> float:
        return float(self._ex.fetch_ticker(symbol)["last"])

    def cash(self) -> float:
        return float(self._ex.fetch_balance()["free"].get(self._quote, 0.0))

    def buy(self, symbol: str, qty: float) -> Fill:
        return self._market(symbol, "buy", qty)

    def sell(self, symbol: str, qty: float) -> Fill:
        return self._market(symbol, "sell", qty)

    def _market(self, symbol: str, side: str, qty: float) -> Fill:
        amount = float(self._ex.amount_to_precision(symbol, qty))
        order = self._ex.create_order(symbol, "market", side, amount)
        if not order.get("filled") or not (order.get("average") or order.get("cost")):
            order = self._ex.fetch_order(order["id"], symbol)

        filled = float(order.get("filled") or 0.0)
        if filled <= 0:
            raise RuntimeError(f"{side} {symbol}: order {order.get('id')} was not filled")
        price = float(order.get("average") or order["cost"] / filled)

        base = self._ex.market(symbol)["base"]
        fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
        fee_quote = 0.0
        for fee in fees:
            cost = float(fee.get("cost") or 0.0)
            if fee.get("currency") == self._quote:
                fee_quote += cost
            elif fee.get("currency") == base:
                fee_quote += cost * price
                if side == "buy":
                    filled -= cost  # the exchange keeps part of what we bought
        return Fill(symbol, side, filled, price, fee_quote)
