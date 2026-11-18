"""Demo-trading loop: watch stops, act on each newly closed candle."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd

from trading_agent.broker.base import Broker
from trading_agent.config import Config
from trading_agent.data.exchange import is_continuous
from trading_agent.data.store import CandleStore, timeframe_ms
from trading_agent.journal import Journal, Position
from trading_agent.risk import RiskManager
from trading_agent.strategies import Strategy
from trading_agent.watchlist import read_watchlist

LOOKBACK_BARS = 1500   # 250 days of 4h candles: enough for daily SMA200 context
SNAPSHOT_EVERY_MS = 15 * 60_000


@dataclass(frozen=True)
class Decision:
    action: str                        # "buy", "sell" or "hold"; leveraged accounts: "long", "short", "close", "hold"
    reason: str
    stop_distance: float | None = None
    take_profit_distance: float | None = None   # leveraged accounts only
    news: str = ""                              # the LLM agent's news briefing for the owner
    outlook: str = ""                           # its price view: "up", "down" or "unclear"


class Decider(Protocol):
    """Whatever chooses the action for a symbol: a rule strategy now, an LLM agent later."""

    name: str

    def decide(self, symbol: str, df: pd.DataFrame, position: Position | None) -> Decision: ...


class StrategyDecider:
    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        self.name = strategy.name

    def decide(self, symbol: str, df: pd.DataFrame, position: Position | None) -> Decision:
        has_position = position is not None
        last = self.strategy.generate(df).iloc[-1]
        if has_position and last["exit"]:
            return Decision("sell", f"{self.name}: exit signal")
        if not has_position and last["entry"] and not last["exit"]:
            return Decision("buy", f"{self.name}: entry signal", float(last["stop_distance"]))
        return Decision("hold", f"{self.name}: no signal")


class PaperTrader:
    def __init__(
        self,
        cfg: Config,
        broker: Broker,
        journal: Journal,
        store: CandleStore,
        decider: Decider,
        sync: Callable[[str, str], object],
    ):
        self.cfg = cfg
        self.broker = broker
        self.journal = journal
        self.store = store
        self.decider = decider
        self.sync = sync
        self.risk = RiskManager(cfg.risk)
        self.symbols = cfg.paper.symbols or cfg.data.symbols
        self.timeframe = cfg.paper.timeframe
        self._tf_ms = timeframe_ms(self.timeframe)
        # Crypto never closes: a candle more than one period late means the data feed is broken.
        # Stocks close for weekends and holidays, so allow for a long weekend.
        continuous = is_continuous(cfg.data.exchange)
        # Intraday candles from before a break (gold's weekend) say nothing about the price after
        # it: skip them and decide on the first candle that closes after the reopen.
        intraday = self._tf_ms < 86_400_000
        self._max_bar_age_ms = self._tf_ms if continuous or intraday else 5 * 86_400_000
        # Yahoo is a web download, not worth repeating all weekend; the MT5 terminal is local and instant.
        self._sync_retry_ms = 0 if continuous or cfg.data.exchange == "mt5" else 30 * 60_000
        self.lookback_bars = LOOKBACK_BARS
        self._next_sync: dict[str, int] = {}
        self._reported: dict[str, object] = {}

        bound = journal.get("broker")
        if bound is None:
            journal.set("broker", broker.name)
        elif bound != broker.name:
            raise RuntimeError(
                f"{cfg.paper.db_path} belongs to broker '{bound}', not '{broker.name}'. "
                "Use another [paper].db_path or run `trader paper reset --yes`."
            )

    # ------------------------------------------------------------------
    def equity(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or {}
        total = self.broker.cash()
        for symbol, pos in self.journal.positions().items():
            total += pos.qty * (prices.get(symbol) or self.broker.price(symbol))
        return total

    def tick(self, now_ms: int | None = None) -> list[str]:
        """One pass. Returns human-readable events."""
        now_ms = now_ms or int(time.time() * 1000)
        events: list[str] = []
        traded = False
        self._refresh_symbols(events)

        if self.journal.last_equity_time() is None:
            start = self.equity()
            self.journal.set("initial_equity", start)
            self.journal.record_equity(now_ms, start, self.broker.cash())

        # 1. Stop-losses, checked against the live price on every pass.
        for symbol, pos in self.journal.positions().items():
            if pos.stop is None or not self.broker.market_open(symbol):
                continue   # a closed market cannot fill a stop; it is checked again at the open
            price = self.broker.price(symbol)
            if price <= pos.stop:
                pnl = self._close(pos, now_ms, "stop")
                events.append(f"STOP  {symbol} price {price:.6g} <= stop {pos.stop:.6g}, P&L {pnl:+.2f}")
                traded = True

        # 2. One decision per symbol per newly closed candle. A symbol taken off the watchlist
        #    while a position is open stays managed until that position is closed.
        held = [s for s in self.journal.positions() if s not in self.symbols]
        for symbol in [*self.symbols, *held]:
            try:
                traded |= self._process_symbol(symbol, now_ms, events)
            except Exception as exc:  # one symbol failing must not stop the others
                events.append(f"ERROR {symbol}: {type(exc).__name__}: {exc}")

        # 3. Equity snapshot.
        last = self.journal.last_equity_time()
        if traded or last is None or now_ms - last >= SNAPSHOT_EVERY_MS:
            self.journal.record_equity(now_ms, self.equity(), self.broker.cash())
        return events

    # ------------------------------------------------------------------
    def _process_symbol(self, symbol: str, now_ms: int, events: list[str]) -> bool:
        bar = self._new_closed_bar(symbol, now_ms, events)
        if bar is None:
            return False
        last_ts, df = bar
        position = self.journal.positions().get(symbol)
        decision = self.decider.decide(symbol, df, position)
        action, reason, traded = self._act(symbol, decision, position, now_ms)

        recorded = reason + (f" | News: {decision.news} Outlook: {decision.outlook}." if decision.news else "")
        self.journal.record_decision(now_ms, symbol, self.timeframe, last_ts, self.decider.name, action, recorded)
        self.journal.set(f"last_bar:{symbol}:{self.timeframe}", last_ts)
        if action != "hold":
            events.append(f"{action.upper():<5} {symbol} {reason}")
        self._news_briefing(symbol, decision, action, now_ms, events)
        return traded

    def _news_briefing(self, symbol: str, decision: Decision, action: str, now_ms: int, events: list[str]) -> None:
        """The agent's news briefing, as a NEWS event: every few hours, and sooner when its view changes."""
        every = self.cfg.alerts.news_every_hours
        if not decision.news or every <= 0:
            return
        last = json.loads(self.journal.get(f"news_briefing:{symbol}") or "{}")
        hours = (now_ms - last["time"]) / 3_600_000 if last else float("inf")
        changed = last.get("outlook") != decision.outlook
        if hours < every and not (changed and hours >= self.cfg.alerts.news_min_hours):
            return
        self.journal.set(f"news_briefing:{symbol}", json.dumps({"time": now_ms, "outlook": decision.outlook}))
        view = {"up": "price likely UP", "down": "price likely DOWN"}.get(decision.outlook, "direction unclear")
        events.append(f"NEWS  {symbol}: {view}\n{decision.news}\nAgent's action now: {action}")

    def _new_closed_bar(self, symbol: str, now_ms: int, events: list[str]) -> tuple[int, pd.DataFrame] | None:
        """The candles up to a newly closed one that has not been decided on yet, else None."""
        exchange = self.cfg.data.exchange
        last_ts = self.store.last_ts(exchange, symbol, self.timeframe)
        newer_expected = last_ts is None or now_ms >= last_ts + 2 * self._tf_ms
        if newer_expected and now_ms >= self._next_sync.get(symbol, 0):
            # Set the retry time first so a failing or empty download is not repeated every pass
            # (a stock market has no new candle for a whole weekend).
            self._next_sync[symbol] = now_ms + self._sync_retry_ms
            self.sync(symbol, self.timeframe)
            newest = self.store.last_ts(exchange, symbol, self.timeframe)
            if newest != last_ts:
                self._next_sync.pop(symbol, None)
            last_ts = newest
        if last_ts is None:
            self._report_once(events, symbol, "none", f"SKIP  {symbol}: no candles")
            return None

        state_key = f"last_bar:{symbol}:{self.timeframe}"
        if self.journal.get(state_key) == str(last_ts):
            return None
        if now_ms >= last_ts + self._tf_ms + self._max_bar_age_ms:
            self._report_once(events, symbol, last_ts, f"SKIP  {symbol}: candle data is stale")
            return None
        if not self.broker.market_open(symbol):
            return None   # decide when the order can actually be filled: at the next open
        return last_ts, self.store.load(exchange, symbol, self.timeframe).tail(self.lookback_bars)

    def _act(self, symbol: str, decision: Decision, position: Position | None, now_ms: int) -> tuple[str, str, bool]:
        """Carry out a decision. Returns (what was done, the reason text, whether anything traded)."""
        action, reason, traded = decision.action, decision.reason, False
        if decision.action == "sell" and position:
            pnl = self._close(position, now_ms, "signal")
            reason += f"; closed with P&L {pnl:+.2f}"
            traded = True
        elif decision.action == "buy" and not position:
            action, detail, traded = self._open(symbol, decision, now_ms)
            reason += f"; {detail}"
        else:
            action = "hold"
        return action, reason, traded

    def _refresh_symbols(self, events: list[str]) -> None:
        """Pick up watchlist changes made in the app while the loop is running."""
        if not self.cfg.source_path:
            return
        chosen = read_watchlist(self.cfg.source_path)
        if chosen is None or tuple(chosen["symbols"]) == tuple(self.symbols):
            return
        added = [s for s in chosen["symbols"] if s not in self.symbols]
        removed = [s for s in self.symbols if s not in chosen["symbols"]]
        self.symbols = tuple(chosen["symbols"])
        events.append(f"WATCH now monitoring {len(self.symbols)} symbols"
                      + (f"; added {', '.join(added)}" if added else "")
                      + (f"; removed {', '.join(removed)}" if removed else ""))

    def _report_once(self, events: list[str], symbol: str, key: object, message: str) -> None:
        """Report a persistent condition once, not on every pass."""
        if self._reported.get(symbol) != key:
            self._reported[symbol] = key
            events.append(message)

    def _open(self, symbol: str, decision: Decision, now_ms: int) -> tuple[str, str, bool]:
        positions = self.journal.positions()
        prices = {s: self.broker.price(s) for s in positions}
        price = self.broker.price(symbol)
        equity = self.equity(prices)
        verdict = self.risk.size_entry(
            equity=equity,
            cash=self.broker.cash(),
            exposure=sum(p.qty * prices[s] for s, p in positions.items()),
            open_positions=len(positions),
            price=price,
            stop_distance=decision.stop_distance,
            day_start_equity=self.journal.day_start_equity(now_ms) or equity,
        )
        if not verdict.approved:
            return "rejected", f"risk: {verdict.reason}", False

        fill = self.broker.buy(symbol, verdict.qty)
        self.journal.record_fill(fill, now_ms)
        stop = fill.price - decision.stop_distance
        self.journal.open_position(
            Position(symbol, fill.qty, fill.price, now_ms, fill.value + fill.fee, stop, self.decider.name)
        )
        return "buy", f"bought {fill.qty:.6g} @ {fill.price:.6g}, stop {stop:.6g}", True

    def _close(self, position: Position, now_ms: int, reason: str) -> float:
        fill = self.broker.sell(position.symbol, position.qty)
        self.journal.record_fill(fill, now_ms)
        return self.journal.close_position(position, fill, now_ms, reason)
