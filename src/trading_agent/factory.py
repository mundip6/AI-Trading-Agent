"""Builds the trading loop and its decider from a config. Shared by the CLI and the app."""

from __future__ import annotations

from trading_agent.broker import make_broker
from trading_agent.config import Config
from trading_agent.data import CandleStore, sync_candles
from trading_agent.data.exchange import is_continuous, make_exchange
from trading_agent.journal import Journal
from trading_agent.strategies import build_strategy
from trading_agent.trader import PaperTrader, StrategyDecider


def build_decider(cfg: Config, force_llm: bool = False):
    if force_llm or cfg.paper.decider == "llm":
        from trading_agent.agent import LLMDecider  # imported lazily: LangChain is slow to load

        context = {}
        if cfg.news.enabled:
            from trading_agent.news import NewsFeed

            context["news"] = NewsFeed(cfg.news).context
        if cfg.copytrade.enabled:
            from trading_agent.copytrade import CopyTrading

            context["followed_traders"] = CopyTrading(cfg.copytrade).context
        if cfg.paper.broker == "mt5":
            from trading_agent.agent.margin import MarginLLMDecider

            context.pop("followed_traders", None)   # that feed covers crypto only
            context["quote"] = _quote_context(cfg.data.exchange)
            context["session"] = _session_context
            return MarginLLMDecider(cfg.llm, cfg.paper.timeframe, limits=cfg.mt5, context=context,
                                    event_blackout_minutes=cfg.news.event_blackout_minutes if cfg.news.enabled else 0,
                                    asset_class="leveraged contracts (gold, forex, indices) at an MT5 broker")
        asset_class = "crypto" if is_continuous(cfg.data.exchange) else "stocks and ETFs"
        return LLMDecider(
            cfg.llm, cfg.paper.timeframe, context=context, asset_class=asset_class,
            earnings_blackout_days=cfg.news.earnings_blackout_days if cfg.news.enabled else 0,
        )
    if cfg.paper.broker == "mt5":
        raise ValueError("An MT5 account is traded by the LLM agent: set [paper].decider = \"llm\"")
    if cfg.paper.decider != "strategy":
        raise ValueError(f"[paper].decider must be 'strategy' or 'llm', not '{cfg.paper.decider}'")
    return StrategyDecider(build_strategy(cfg.paper.strategy, cfg.paper.params))


def _session_context(symbol: str) -> dict:
    import time

    from trading_agent.data.mt5 import hours_to_weekend_close

    return {"hours_until_weekend_close": round(hours_to_weekend_close(time.time()), 1)}


def _quote_context(exchange_id: str):
    exchange = None

    def quote(symbol: str) -> dict:
        nonlocal exchange
        exchange = exchange or make_exchange(exchange_id)
        ticker = exchange.fetch_ticker(symbol)
        return {"bid": ticker["bid"], "ask": ticker["ask"], "spread": round(ticker["ask"] - ticker["bid"], 6)}

    return quote


def build_trader(cfg: Config) -> PaperTrader:
    if (cfg.paper.broker == "mt5") != (cfg.data.exchange == "mt5"):
        raise ValueError('An MT5 account needs both [data].exchange = "mt5" and [paper].broker = "mt5"')
    journal = Journal(cfg.paper.db_path)
    store = CandleStore(cfg.data.db_path)
    exchange = make_exchange(cfg.data.exchange)
    if cfg.data.exchange == "yahoo":
        # One account holds one currency: a EUR-listed ETF next to a USD one would corrupt the P&L.
        currencies = {s: exchange.currency(s) for s in cfg.paper.symbols or cfg.data.symbols}
        if len(set(currencies.values())) > 1:
            raise ValueError(f"All symbols of one account must trade in the same currency, got {currencies}")

    def sync(symbol: str, timeframe: str) -> int:
        return sync_candles(store, cfg.data.exchange, symbol, timeframe, cfg.data.since, exchange)

    if cfg.paper.broker == "mt5":
        from trading_agent.margin import MarginTrader

        return MarginTrader(cfg, make_broker(cfg, journal), journal, store, build_decider(cfg), sync)
    return PaperTrader(cfg, make_broker(cfg, journal), journal, store, build_decider(cfg), sync)
