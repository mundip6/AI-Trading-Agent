from __future__ import annotations

from typing import TYPE_CHECKING

from trading_agent.broker.base import Broker, Fill
from trading_agent.broker.paper import PaperBroker, ccxt_price_feed
from trading_agent.config import Config

if TYPE_CHECKING:
    from trading_agent.journal import Journal

TESTNET_SUFFIX = "_testnet"


def make_broker(cfg: Config, journal: Journal) -> Broker:
    kind = cfg.paper.broker
    if kind == "paper":
        from trading_agent.data.exchange import make_exchange

        exchange = make_exchange(cfg.data.exchange)
        return PaperBroker(
            journal,
            ccxt_price_feed(exchange),
            cfg.paper.initial_cash,
            cfg.paper.fee_rate,
            cfg.paper.slippage,
            market_open=getattr(exchange, "is_market_open", None),
        )
    if kind == "mt5":
        from trading_agent.broker.mt5_broker import MT5Broker

        return MT5Broker(cfg.mt5)
    if kind.endswith(TESTNET_SUFFIX):
        from trading_agent.broker.ccxt_broker import CcxtTestnetBroker

        return CcxtTestnetBroker.from_env(kind.removesuffix(TESTNET_SUFFIX))
    raise ValueError(f"Unknown broker '{kind}'. Use 'paper', 'mt5' or '<exchange>_testnet'")


__all__ = ["Broker", "Fill", "PaperBroker", "make_broker"]
