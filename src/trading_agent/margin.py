"""Demo-trading loop for a leveraged MT5 account: long and short, stops held by the broker."""

from __future__ import annotations

import time

from trading_agent.data.mt5 import hours_to_weekend_close
from trading_agent.journal import Position
from trading_agent.trader import Decision, PaperTrader

SNAPSHOT_EVERY_MS = 5 * 60_000


class MarginTrader(PaperTrader):
    """Same candle rhythm as PaperTrader; the broker account is the source of truth.

    Stop-loss and take-profit live at the broker, so this loop does not watch prices:
    each pass it checks whether the broker closed a position, and decides on new candles.
    """

    def __init__(self, cfg, broker, journal, store, decider, sync):
        super().__init__(cfg, broker, journal, store, decider, sync)
        self.lookback_bars = cfg.mt5.lookback_bars
        self._open_state: dict[str, bool] = {}

    def equity(self, prices: dict[str, float] | None = None) -> float:
        return self.broker.equity()

    def tick(self, now_ms: int | None = None) -> list[str]:
        now_ms = now_ms or int(time.time() * 1000)
        events: list[str] = []
        self._refresh_symbols(events)
        account = self.broker.account()   # raises unless the terminal is on a demo account

        if self.journal.last_equity_time() is None:
            self.journal.set("initial_equity", account.equity)
            self.journal.record_equity(now_ms, account.equity, account.balance)

        # 1. Positions the broker closed (stop, target, or by hand in MetaTrader).
        traded = False
        for symbol, pos in self.journal.positions().items():
            try:
                traded |= self._reconcile(pos, now_ms, events)
            except Exception as exc:
                events.append(f"ERROR {symbol}: {type(exc).__name__}: {exc}")

        # 2. One decision per symbol per newly closed candle, only while its market is open.
        to_weekend = hours_to_weekend_close(now_ms / 1000)
        held = [s for s in self.journal.positions() if s not in self.symbols]
        for symbol in [*self.symbols, *held]:
            try:
                is_open = self.broker.market_open(symbol)
                was_open = self._open_state.get(symbol, True)     # an open market at start needs no notice
                self._open_state[symbol] = is_open
                if was_open != is_open:
                    events.append(f"MARKET {symbol} is open: the agent is deciding" if is_open else
                                  f"MARKET {symbol} is closed: the agent waits for it to reopen")
                if not is_open:
                    continue
                position = self.journal.positions().get(symbol)
                minutes = self.cfg.mt5.close_before_weekend_minutes
                if position is not None and minutes and 0 < to_weekend * 60 <= minutes:
                    traded |= self._close_for_weekend(position, now_ms, events)
                    continue
                traded |= self._process_symbol(symbol, now_ms, events)
            except Exception as exc:
                events.append(f"ERROR {symbol}: {type(exc).__name__}: {exc}")

        # 3. Equity snapshot, straight from the broker.
        last = self.journal.last_equity_time()
        if traded or last is None or now_ms - last >= SNAPSHOT_EVERY_MS:
            account = self.broker.account()
            self.journal.record_equity(now_ms, account.equity, account.balance)
        return events

    # ------------------------------------------------------------------
    def _reconcile(self, pos: Position, now_ms: int, events: list[str]) -> bool:
        if self.broker.position(pos.ticket) is not None:
            return False
        closed = self.broker.closed(pos.ticket)
        if closed is None:
            self._report_once(events, pos.symbol, ("gone", pos.ticket),
                              f"ERROR {pos.symbol}: position {pos.ticket} is gone but not yet in the account history")
            return False
        self._record_close(pos, closed.exit_price, closed.pnl, closed.exit_time_ms, closed.reason)
        kind = {"stop": "STOP ", "target": "TP   "}.get(closed.reason, "CLOSE")
        events.append(f"{kind} {pos.symbol} {pos.side} closed at {closed.exit_price:.6g} "
                      f"({closed.reason}), P&L {closed.pnl:+.2f}")
        return True

    def _close_for_weekend(self, pos: Position, now_ms: int, events: list[str]) -> bool:
        live = self.broker.position(pos.ticket)
        if live is None:
            return False        # the broker closed it first; the next pass records how
        self.broker.close(pos.ticket)
        closed = self.broker.closed(pos.ticket)
        price = closed.exit_price if closed else self.broker.price(pos.symbol)
        pnl = closed.pnl if closed else live.profit
        self._record_close(pos, price, pnl, now_ms, "weekend")
        events.append(f"CLOSE {pos.symbol} {pos.side} closed at {price:.6g} before the weekend, P&L {pnl:+.2f}")
        return True

    def _record_close(self, pos: Position, price: float, pnl: float, time_ms: int, reason: str) -> None:
        from trading_agent.broker.base import Fill

        self.journal.record_fill(Fill(pos.symbol, "sell" if pos.qty > 0 else "buy", abs(pos.qty), price, 0.0), time_ms)
        self.journal.close_margin_position(pos, price, pnl, time_ms, reason)

    def _act(self, symbol: str, decision: Decision, position: Position | None, now_ms: int) -> tuple[str, str, bool]:
        reason = decision.reason
        if position is not None:
            # While in a position the only action is to close it; a call for the opposite side also closes.
            opposite = "short" if position.side == "long" else "long"
            if decision.action not in ("close", opposite):
                return "hold", reason, False
            live = self.broker.position(position.ticket)
            if live is None:
                return "hold", reason + "; the broker had already closed the position", False
            self.broker.close(position.ticket)
            closed = self.broker.closed(position.ticket)
            price = closed.exit_price if closed else self.broker.price(symbol)
            pnl = closed.pnl if closed else live.profit
            self._record_close(position, price, pnl, now_ms, "signal")
            return "close", f"{reason}; closed {position.side} at {price:.6g} with P&L {pnl:+.2f}", True
        if decision.action in ("long", "short"):
            to_weekend = hours_to_weekend_close(now_ms / 1000)
            if to_weekend <= self.cfg.mt5.no_entry_hours_before_weekend:
                return "hold", f"{reason}; no entry: the market closes for the weekend in {to_weekend:.1f} h", False
            return self._open_margin(symbol, decision, now_ms)
        return "hold", reason, False

    def _open_margin(self, symbol: str, decision: Decision, now_ms: int) -> tuple[str, str, bool]:
        side, distance = decision.action, decision.stop_distance
        account = self.broker.account()
        spec = self.broker.spec(symbol)
        entry = spec.ask if side == "long" else spec.bid
        direction = 1 if side == "long" else -1
        valid = distance is not None and 0 < distance < entry
        stop = entry - direction * distance if valid else None
        mt5 = self.cfg.mt5
        verdict = self.risk.size_margin_entry(
            equity=account.equity,
            free_margin=account.free_margin,
            open_positions=len(self.journal.positions()),
            stop_distance=distance if valid else None,
            spread=spec.spread,
            loss_per_lot=self.broker.loss_per_lot(symbol, side, entry, stop) if valid else 0.0,
            margin_per_lot=self.broker.margin_per_lot(symbol, side, entry) if valid else 0.0,
            volume_min=spec.volume_min,
            volume_step=spec.volume_step,
            volume_max=spec.volume_max,
            max_lots=mt5.max_lots,
            max_margin_pct=mt5.max_margin_pct,
            max_spread_to_stop=mt5.max_spread_to_stop,
            min_stop_distance=spec.min_stop_distance,
            day_start_equity=self.journal.day_start_equity(now_ms) or account.equity,
        )
        if not verdict.approved:
            return "rejected", f"{decision.reason}; risk: {verdict.reason}", False

        target = entry + direction * decision.take_profit_distance if decision.take_profit_distance else None
        live = self.broker.open(symbol, side, verdict.qty, stop, target)
        units = direction * live.lots * spec.contract_size
        from trading_agent.broker.base import Fill

        self.journal.record_fill(
            Fill(symbol, "buy" if side == "long" else "sell", abs(units), live.entry_price, 0.0), now_ms)
        self.journal.open_position(Position(
            symbol, units, live.entry_price, now_ms, units * live.entry_price, live.stop or stop,
            self.decider.name, take_profit=live.take_profit or target, ticket=live.ticket, lots=live.lots,
        ))
        text = (f"{decision.reason}; {side} {live.lots:g} lots @ {live.entry_price:.6g}, "
                f"stop {live.stop or stop:.6g}" + (f", target {target:.6g}" if target else ""))
        return ("buy" if side == "long" else "sell"), text, True
