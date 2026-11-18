"""Demo account at an MT5 broker (e.g. PU Prime): leveraged, long and short.

Deliberately demo-only: every call that could trade first checks that the terminal
is logged in to a demo account and refuses otherwise. Stop-loss and take-profit are
placed at the broker, so they work even while the agent is not running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trading_agent.config import MT5Config
from trading_agent.data.mt5 import MT5Error, MT5Terminal, shared_terminal

_HINTS = {
    10004: "the price moved (requote)",
    10016: "the broker rejected the stop-loss or take-profit level",
    10018: "the market is closed",
    10019: "not enough free margin",
    10021: "no quotes at the moment",
    10027: "Algo Trading is switched off in MetaTrader 5: press the 'Algo Trading' button in its toolbar",
    10030: "the broker does not accept this order filling type",
}


@dataclass(frozen=True)
class AccountState:
    login: int
    server: str
    currency: str
    balance: float
    equity: float
    free_margin: float
    leverage: int


@dataclass(frozen=True)
class SymbolSpec:
    contract_size: float     # units per 1.0 lot (gold: 100 oz)
    volume_min: float
    volume_step: float
    volume_max: float
    digits: int
    min_stop_distance: float  # closest a stop may sit to the price
    bid: float
    ask: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class LivePosition:
    ticket: int
    symbol: str
    side: str                # "long" or "short"
    lots: float
    entry_price: float
    stop: float | None
    take_profit: float | None
    profit: float            # floating P&L in account currency, swap included


@dataclass(frozen=True)
class ClosedPosition:
    exit_price: float
    exit_time_ms: int
    pnl: float               # account currency: profit + swap + commission
    reason: str              # "stop", "target", "signal", "margin call" or "manual"


class MT5Broker:
    def __init__(self, cfg: MT5Config, terminal: MT5Terminal | None = None):
        self.cfg = cfg
        self.terminal = terminal or shared_terminal()
        self.name = f"mt5-demo:{self.account().login}"

    # --- account -----------------------------------------------------------
    def account(self) -> AccountState:
        """The account's state. Raises unless it is a demo account."""
        info = self.terminal.account()
        api = self.terminal.api
        if info.trade_mode != api.ACCOUNT_TRADE_MODE_DEMO:
            raise MT5Error(
                f"MetaTrader 5 is logged in to account {info.login} on {info.server}, which is NOT a demo "
                "account. This agent only trades demo accounts: log the terminal in to your demo account"
            )
        if self.cfg.login and info.login != self.cfg.login:
            raise MT5Error(f"MetaTrader 5 is logged in to account {info.login}, but [mt5].login is {self.cfg.login}")
        return AccountState(info.login, info.server, info.currency, float(info.balance), float(info.equity),
                            float(info.margin_free), int(info.leverage))

    def cash(self) -> float:
        return self.account().balance

    def equity(self) -> float:
        return self.account().equity

    # --- market ------------------------------------------------------------
    def spec(self, symbol: str) -> SymbolSpec:
        info = self.terminal.symbol(symbol)
        tick = self.terminal.tick(symbol)
        return SymbolSpec(
            contract_size=float(info.trade_contract_size), volume_min=float(info.volume_min),
            volume_step=float(info.volume_step), volume_max=float(info.volume_max), digits=int(info.digits),
            min_stop_distance=float(info.trade_stops_level) * float(info.point),
            bid=float(tick.bid), ask=float(tick.ask),
        )

    def price(self, symbol: str) -> float:
        tick = self.terminal.tick(symbol)
        return (tick.bid + tick.ask) / 2

    def market_open(self, symbol: str) -> bool:
        from trading_agent.data.mt5 import MT5Exchange

        info = self.terminal.symbol(symbol)
        if info.trade_mode != self.terminal.api.SYMBOL_TRADE_MODE_FULL:
            return False
        return MT5Exchange(self.terminal, self.terminal._clock).is_market_open(symbol)

    def loss_per_lot(self, symbol: str, side: str, entry: float, stop: float) -> float:
        """What one lot loses, in account currency, if the stop is hit."""
        api = self.terminal.api
        order = api.ORDER_TYPE_BUY if side == "long" else api.ORDER_TYPE_SELL
        return -float(self.terminal.call("order_calc_profit", order, symbol, 1.0, entry, stop))

    def margin_per_lot(self, symbol: str, side: str, price: float) -> float:
        api = self.terminal.api
        order = api.ORDER_TYPE_BUY if side == "long" else api.ORDER_TYPE_SELL
        return float(self.terminal.call("order_calc_margin", order, symbol, 1.0, price))

    # --- positions ---------------------------------------------------------
    def position(self, ticket: int) -> LivePosition | None:
        self.account()
        found = self.terminal.call("positions_get", ticket=ticket)
        return self._live(found[0]) if found else None

    def _live(self, p) -> LivePosition:
        side = "long" if p.type == self.terminal.api.POSITION_TYPE_BUY else "short"
        return LivePosition(p.ticket, p.symbol, side, float(p.volume), float(p.price_open),
                            float(p.sl) or None, float(p.tp) or None, float(p.profit) + float(p.swap))

    def open(self, symbol: str, side: str, lots: float, stop: float, take_profit: float | None) -> LivePosition:
        """Market order with its stop-loss and take-profit attached."""
        spec = self.spec(symbol)
        api = self.terminal.api
        request = {
            "action": api.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": api.ORDER_TYPE_BUY if side == "long" else api.ORDER_TYPE_SELL,
            "price": spec.ask if side == "long" else spec.bid,
            "sl": round(stop, spec.digits),
            "tp": round(take_profit, spec.digits) if take_profit else 0.0,
            "deviation": self.cfg.deviation_points,
            "magic": self.cfg.magic,
            "comment": "trading-agent",
            "type_time": api.ORDER_TIME_GTC,
        }
        result = self._send(symbol, request)
        for candidate in self.terminal.call("positions_get", symbol=symbol):
            if candidate.ticket == result.order or candidate.identifier == result.order:
                return self._live(candidate)
        mine = [p for p in self.terminal.call("positions_get", symbol=symbol) if p.magic == self.cfg.magic]
        if not mine:
            raise MT5Error(f"{symbol}: order {result.order} was accepted but no position appeared")
        return self._live(max(mine, key=lambda p: p.time_msc))

    def close(self, ticket: int) -> None:
        live = self.position(ticket)
        if live is None:
            return
        spec = self.spec(live.symbol)
        api = self.terminal.api
        self._send(live.symbol, {
            "action": api.TRADE_ACTION_DEAL,
            "symbol": live.symbol,
            "volume": live.lots,
            "type": api.ORDER_TYPE_SELL if live.side == "long" else api.ORDER_TYPE_BUY,
            "position": ticket,
            "price": spec.bid if live.side == "long" else spec.ask,
            "deviation": self.cfg.deviation_points,
            "magic": self.cfg.magic,
            "comment": "trading-agent close",
            "type_time": api.ORDER_TIME_GTC,
        })

    def closed(self, ticket: int) -> ClosedPosition | None:
        """How a position that no longer exists ended, from the account history."""
        api = self.terminal.api
        deals = self.terminal.call("history_deals_get", position=ticket)
        exits = [d for d in deals if d.entry != api.DEAL_ENTRY_IN]
        if not exits:
            return None
        volume = sum(d.volume for d in exits)
        price = sum(d.price * d.volume for d in exits) / volume if volume else exits[-1].price
        pnl = sum(d.profit + d.swap + d.commission + getattr(d, "fee", 0.0) for d in deals)
        reason = {api.DEAL_REASON_SL: "stop", api.DEAL_REASON_TP: "target", api.DEAL_REASON_SO: "margin call",
                  api.DEAL_REASON_EXPERT: "signal"}.get(exits[-1].reason, "manual")
        return ClosedPosition(float(price), self.terminal.to_utc_ms(exits[-1].time), float(pnl), reason)

    # --- orders ------------------------------------------------------------
    def _send(self, symbol: str, request: dict):
        self.account()   # demo check immediately before every order
        api = self.terminal.api
        allowed = int(self.terminal.symbol(symbol).filling_mode)
        fillings = [f for flag, f in ((2, api.ORDER_FILLING_IOC), (1, api.ORDER_FILLING_FOK)) if allowed & flag]
        fillings.append(api.ORDER_FILLING_RETURN)
        result = None
        for filling in fillings:
            result = self.terminal.call("order_send", {**request, "type_filling": filling})
            if result.retcode != api.TRADE_RETCODE_INVALID_FILL:
                break
        if result.retcode not in (api.TRADE_RETCODE_DONE, api.TRADE_RETCODE_PLACED):
            hint = _HINTS.get(result.retcode, result.comment)
            raise MT5Error(f"{symbol}: order rejected ({result.retcode}: {hint})")
        return result


def round_lots(lots: float, step: float) -> float:
    """Round down to the broker's volume step."""
    if step <= 0:
        return lots
    return round(math.floor(lots / step + 1e-9) * step, 8)
