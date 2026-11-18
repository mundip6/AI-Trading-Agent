"""MT5 account: data feed, demo-only broker, lot sizing, long/short loop. Uses a simulated terminal."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from trading_agent.agent.margin import MarginLLMDecider
from trading_agent.agent.snapshot import build_intraday_snapshot
from trading_agent.broker.mt5_broker import MT5Broker
from trading_agent.config import Config, LLMConfig, MT5Config, PaperConfig, RiskConfig
from trading_agent.data.mt5 import MT5Error, MT5Exchange, MT5Terminal, new_york_close_offset
from trading_agent.data.store import CandleStore, timeframe_ms, to_ms
from trading_agent.journal import Journal, Position
from trading_agent.margin import MarginTrader
from trading_agent.risk import RiskManager
from trading_agent.trader import Decision

GOLD = "XAUUSD.s"
TF = timeframe_ms("15m")
OFFSET = 2 * 3600                       # server clock: UTC+2
NOW = to_ms("2026-11-13 12:00:30")      # a Friday in winter, 30 s after a candle closed
LAST_BAR = to_ms("2026-11-13 11:45")


class FakeMT5:
    """The parts of the MetaTrader5 module the agent uses, with one gold symbol."""

    TIMEFRAME_M15, TIMEFRAME_D1 = 15, 16408
    TRADE_ACTION_DEAL, ORDER_TYPE_BUY, ORDER_TYPE_SELL, ORDER_TIME_GTC = 1, 0, 1, 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL = 0, 2
    TRADE_RETCODE_DONE, TRADE_RETCODE_PLACED, TRADE_RETCODE_INVALID_FILL = 10009, 10008, 10030
    DEAL_ENTRY_IN, DEAL_ENTRY_OUT = 0, 1
    DEAL_REASON_EXPERT, DEAL_REASON_SL, DEAL_REASON_TP, DEAL_REASON_SO = 3, 4, 5, 6
    POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
    SYMBOL_TRADE_MODE_FULL = 4

    def __init__(self):
        self.now = NOW / 1000
        self.trade_mode = self.ACCOUNT_TRADE_MODE_DEMO
        self.balance = 10_000.0
        self.bid, self.ask = 4000.0, 4000.3
        self.eurusd = 1.25                    # account in EUR, gold quoted in USD
        self.algo_trading = True
        self.accepted_filling = self.ORDER_FILLING_IOC
        self.positions, self.deals, self.requests = {}, [], []
        self._next = 1000
        self.error = (1, "Success")

    # connection
    def initialize(self, **kwargs):
        return True

    def last_error(self):
        return self.error

    def terminal_info(self):
        return SimpleNamespace(connected=True, trade_allowed=self.algo_trading, name="Fake", company="Fake")

    def account_info(self):
        floating = sum(self._profit(p) for p in self.positions.values())
        return SimpleNamespace(login=700, server="Fake-Demo", trade_mode=self.trade_mode, currency="EUR",
                               balance=self.balance, equity=self.balance + floating,
                               margin_free=self.balance + floating, leverage=500, trade_expert=True)

    # market
    def symbol_info(self, name):
        if name != GOLD:
            return None
        return SimpleNamespace(name=name, visible=True, digits=2, point=0.01, volume_min=0.01, volume_step=0.01,
                               volume_max=100.0, trade_contract_size=100.0, trade_stops_level=20,
                               filling_mode=3, trade_mode=self.SYMBOL_TRADE_MODE_FULL,
                               time=int(self.now) + OFFSET, bid=self.bid, ask=self.ask)

    def symbol_select(self, name, enable):
        return True

    def symbol_info_tick(self, name):
        return SimpleNamespace(time=int(self.now) + OFFSET, bid=self.bid, ask=self.ask)

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        last_open = int(self.now) // 900 * 900      # the candle still forming
        times = [last_open - i * 900 + OFFSET for i in range(count)][::-1]
        rates = np.zeros(count, dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                                       ("close", "f8"), ("tick_volume", "i8")])
        rates["time"], rates["open"], rates["high"] = times, 4000, 4002
        rates["low"], rates["close"], rates["tick_volume"] = 3998, 4000, 100
        return rates

    def order_calc_profit(self, order_type, symbol, lots, open_price, close_price):
        direction = 1 if order_type == self.ORDER_TYPE_BUY else -1
        return direction * (close_price - open_price) * 100 * lots / self.eurusd

    def order_calc_margin(self, order_type, symbol, lots, price):
        return price * 100 * lots / 500 / self.eurusd

    # trading
    def _profit(self, p):
        price = self.bid if p.type == self.POSITION_TYPE_BUY else self.ask
        return self.order_calc_profit(p.type, p.symbol, p.volume, p.price_open, price)

    def positions_get(self, ticket=None, symbol=None):
        found = [p for p in self.positions.values()
                 if (ticket is None or p.ticket == ticket) and (symbol is None or p.symbol == symbol)]
        for p in found:
            p.profit = self._profit(p)
        return tuple(found)

    def order_send(self, request):
        self.requests.append(request)
        if not self.algo_trading:
            return SimpleNamespace(retcode=10027, comment="AutoTrading disabled by client", order=0)
        if request["type_filling"] != self.accepted_filling:
            return SimpleNamespace(retcode=self.TRADE_RETCODE_INVALID_FILL, comment="Unsupported filling mode", order=0)
        if "position" in request:
            self.close_at_broker(request["position"], request["price"], self.DEAL_REASON_EXPERT)
            return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, comment="", order=request["position"])
        self._next += 1
        self.positions[self._next] = SimpleNamespace(
            ticket=self._next, identifier=self._next, symbol=request["symbol"], type=request["type"],
            volume=request["volume"], price_open=request["price"], sl=request["sl"], tp=request["tp"],
            profit=0.0, swap=0.0, magic=request["magic"], time_msc=int(self.now * 1000))
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, comment="", order=self._next)

    def close_at_broker(self, ticket, price, reason):
        """What the broker's server does when a stop or target is hit."""
        p = self.positions.pop(ticket)
        profit = self.order_calc_profit(p.type, p.symbol, p.volume, p.price_open, price)
        self.balance += profit
        self.deals.append(SimpleNamespace(position_id=ticket, entry=self.DEAL_ENTRY_OUT, volume=p.volume,
                                          price=price, profit=profit, swap=0.0, commission=0.0, fee=0.0,
                                          reason=reason, time=int(self.now) + OFFSET))

    def history_deals_get(self, position=None):
        return tuple(d for d in self.deals if d.position_id == position)


class Scripted:
    name = "scripted"

    def __init__(self):
        self.next = None
        self.calls = 0

    def decide(self, symbol, df, position):
        self.calls += 1
        decision, self.next = self.next, None
        return decision or Decision("hold", "nothing to do")


@pytest.fixture
def fake():
    return FakeMT5()


@pytest.fixture
def terminal(fake):
    return MT5Terminal(mt5=fake, clock=lambda: fake.now, env={})


@pytest.fixture
def env(tmp_path, fake, terminal):
    cfg = Config(
        paper=PaperConfig(broker="mt5", db_path=str(tmp_path / "gold.sqlite"), symbols=(GOLD,), timeframe="15m",
                          decider="llm"),
        risk=RiskConfig(risk_per_trade=0.005, max_open_positions=2, max_daily_loss=0.02),
    )
    cfg = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, exchange="mt5", db_path=str(tmp_path / "m.sqlite"), symbols=(GOLD,)))
    store = CandleStore(cfg.data.db_path)
    store.upsert("mt5", GOLD, "15m", [[LAST_BAR - i * TF, 4000, 4002, 3998, 4000, 100] for i in range(60)])
    journal = Journal(cfg.paper.db_path)
    decider = Scripted()
    trader = MarginTrader(cfg, MT5Broker(cfg.mt5, terminal), journal, store, decider, sync=lambda s, tf: 0)
    return trader, decider, fake, journal


def long(stop=8.0, target=16.0):
    return Decision("long", "trend up", stop, target)


def short(stop=8.0, target=16.0):
    return Decision("short", "trend down", stop, target)


# --- data feed -----------------------------------------------------------
def test_server_clock_follows_new_york_close():
    assert new_york_close_offset(to_ms("2026-11-13 12:00") / 1000) == 2 * 3600    # winter
    assert new_york_close_offset(to_ms("2026-07-01 12:00") / 1000) == 3 * 3600    # US summer time


def test_candles_are_converted_to_utc_and_closed(fake, terminal):
    from trading_agent.data.exchange import fetch_ohlcv

    exchange = MT5Exchange(terminal, clock=lambda: fake.now)
    rows = exchange.fetch_ohlcv(GOLD, "15m", since=NOW - 10 * TF)
    assert rows[-1][0] == to_ms("2026-11-13 12:00")       # the forming candle, stamped in UTC
    assert all(b[0] - a[0] == TF for a, b in zip(rows, rows[1:]))
    assert terminal.measured_utc_offset([GOLD]) == OFFSET
    assert exchange.is_market_open(GOLD)
    fake.now += 3600                                       # quotes stop: the market has closed
    exchange_later = MT5Exchange(terminal, clock=lambda: fake.now)
    fake.symbol_info_tick = lambda name: SimpleNamespace(time=int(NOW / 1000) + OFFSET, bid=4000.0, ask=4000.3)
    assert not exchange_later.is_market_open(GOLD)
    with pytest.raises(ValueError, match="not supported"):
        exchange.fetch_ohlcv(GOLD, "3m")
    assert fetch_ohlcv.__module__                          # the shared downloader drops the forming candle


def test_unknown_symbol_is_explained(terminal):
    with pytest.raises(MT5Error, match="Market Watch"):
        terminal.symbol("XAUUSD")


# --- broker --------------------------------------------------------------
def test_broker_refuses_a_real_account(fake, terminal):
    broker = MT5Broker(MT5Config(), terminal)
    fake.trade_mode = fake.ACCOUNT_TRADE_MODE_REAL
    with pytest.raises(MT5Error, match="NOT a demo"):
        broker.open(GOLD, "long", 0.01, 3990.0, 4020.0)
    assert fake.requests == []                             # nothing reached the terminal
    with pytest.raises(MT5Error, match="NOT a demo"):
        MT5Broker(MT5Config(), terminal)


def test_broker_refuses_another_login(terminal):
    with pytest.raises(MT5Error, match="login"):
        MT5Broker(MT5Config(login=123), terminal)


def test_order_retries_other_filling_types(fake, terminal):
    fake.accepted_filling = fake.ORDER_FILLING_FOK
    live = MT5Broker(MT5Config(), terminal).open(GOLD, "short", 0.05, 4010.0, 3980.0)
    assert [r["type_filling"] for r in fake.requests] == [fake.ORDER_FILLING_IOC, fake.ORDER_FILLING_FOK]
    assert (live.side, live.lots, live.entry_price, live.stop, live.take_profit) == ("short", 0.05, 4000.0, 4010.0, 3980.0)


def test_algo_trading_off_is_explained(fake, terminal):
    fake.algo_trading = False
    with pytest.raises(MT5Error, match="Algo Trading"):
        MT5Broker(MT5Config(), terminal).open(GOLD, "long", 0.01, 3990.0, None)


# --- sizing --------------------------------------------------------------
def size(**overrides):
    args = dict(equity=10_000, free_margin=10_000, open_positions=0, stop_distance=8.0, spread=0.3,
                loss_per_lot=640.0, margin_per_lot=640.0, volume_min=0.01, volume_step=0.01, volume_max=100,
                max_lots=1.0, max_margin_pct=0.5, max_spread_to_stop=0.15, min_stop_distance=0.2,
                day_start_equity=10_000)
    return RiskManager(RiskConfig(risk_per_trade=0.005, max_open_positions=2, max_daily_loss=0.02)).size_margin_entry(
        **{**args, **overrides})


def test_lots_follow_the_risk_budget():
    assert size().qty == pytest.approx(0.07)                       # 50 risk / 640 per lot, rounded down
    assert size(loss_per_lot=40.0).qty == pytest.approx(1.0)       # capped by max_lots
    assert size(loss_per_lot=40.0, free_margin=640.0).qty == pytest.approx(0.5)   # half the free margin


@pytest.mark.parametrize("overrides, text", [
    (dict(equity=1_000, day_start_equity=1_000), "below the broker minimum"),
    (dict(spread=2.0), "spread"),
    (dict(stop_distance=0.4), "closer than the broker allows"),
    (dict(stop_distance=None), "no valid stop"),
    (dict(open_positions=2), "already holding"),
    (dict(equity=9_700), "daily loss"),
])
def test_margin_entry_rejections(overrides, text):
    verdict = size(**overrides)
    assert not verdict.approved and text in verdict.reason


# --- trading loop --------------------------------------------------------
def test_long_is_opened_with_stop_and_target_at_the_broker(env):
    trader, decider, fake, journal = env
    decider.next = long()
    events = trader.tick(NOW)
    request = fake.requests[-1]
    # 0.5% of 10,000 EUR = 50 EUR; 8 USD stop on 100 oz = 800 USD = 640 EUR a lot -> 0.07 lots
    assert request["volume"] == pytest.approx(0.07)
    assert (request["type"], request["price"]) == (fake.ORDER_TYPE_BUY, 4000.3)
    assert (request["sl"], request["tp"]) == (pytest.approx(3992.3), pytest.approx(4016.3))
    position = journal.positions()[GOLD]
    assert (position.side, position.lots, position.qty) == ("long", 0.07, pytest.approx(7.0))
    assert position.take_profit == pytest.approx(4016.3) and position.ticket in fake.positions
    assert events and events[0].startswith("BUY")
    assert journal.decisions_frame()["action"].tolist() == ["buy"]
    assert trader.tick(NOW + 10_000) == [] and decider.calls == 1      # one decision per candle


def test_short_stopped_out_by_the_broker(env):
    trader, decider, fake, journal = env
    decider.next = short()
    assert trader.tick(NOW)[0].startswith("SELL")
    position = journal.positions()[GOLD]
    assert position.side == "short" and position.qty == pytest.approx(-7.0)
    assert fake.requests[-1]["sl"] == pytest.approx(4008.0)             # above the entry at the bid

    fake.bid, fake.ask = 4008.0, 4008.3
    fake.close_at_broker(position.ticket, 4008.0, fake.DEAL_REASON_SL)   # the server hits the stop
    events = trader.tick(NOW + 20_000)
    assert events[0].startswith("STOP") and "-44.80" in events[0]       # 8 USD x 7 oz / 1.25
    assert journal.positions() == {}
    trade = journal.trades_frame().iloc[0]
    assert trade["exit_reason"] == "stop" and trade["pnl"] == pytest.approx(-44.8)
    assert trade["qty"] < 0 and trade["pnl_pct"] == pytest.approx(-0.002)
    assert journal.fills_frame()["side"].tolist() == ["sell", "buy"]
    assert journal.equity_series().iloc[-1] == pytest.approx(10_000 - 44.8)


def test_take_profit_and_manual_close_are_reported(env):
    trader, decider, fake, journal = env
    decider.next = long()
    trader.tick(NOW)
    ticket = journal.positions()[GOLD].ticket
    fake.close_at_broker(ticket, 4016.3, fake.DEAL_REASON_TP)
    assert trader.tick(NOW + 20_000)[0].startswith("TP")
    assert journal.trades_frame().iloc[0]["exit_reason"] == "target"


def test_agent_closes_on_signal_and_ignores_same_side(env):
    trader, decider, fake, journal = env
    decider.next = long()
    trader.tick(NOW)
    fake.now += 900
    fake.bid, fake.ask = 4005.0, 4005.3

    decider.next = long()                                   # already long: nothing to do
    store_next = LAST_BAR + TF
    trader.store.upsert("mt5", GOLD, "15m", [[store_next, 4000, 4006, 3999, 4005, 100]])
    assert trader.tick(NOW + TF) == [] and GOLD in journal.positions()

    fake.now += 900
    trader.store.upsert("mt5", GOLD, "15m", [[store_next + TF, 4005, 4006, 4004, 4005, 100]])
    decider.next = Decision("close", "momentum gone")
    events = trader.tick(NOW + 2 * TF)
    assert events[0].startswith("CLOSE") and journal.positions() == {} and fake.positions == {}
    trade = journal.trades_frame().iloc[0]
    assert trade["exit_reason"] == "signal" and trade["pnl"] == pytest.approx((4005.0 - 4000.3) * 7 / 1.25)


def test_rejected_entry_and_broker_errors_do_not_open_anything(env):
    trader, decider, fake, journal = env
    fake.ask = 4003.0                                       # a 3.0 spread against an 8.0 stop
    decider.next = long()
    assert "spread" in trader.tick(NOW)[0] and fake.requests == []
    assert journal.decisions_frame()["action"].tolist() == ["rejected"]

    fake.ask = 4000.3
    fake.algo_trading = False
    fake.now += 900
    trader.store.upsert("mt5", GOLD, "15m", [[LAST_BAR + TF, 4000, 4002, 3998, 4000, 100]])
    decider.next = long()
    events = trader.tick(NOW + TF)
    assert events[0].startswith("ERROR") and "Algo Trading" in events[0]
    assert journal.positions() == {}


def test_loop_stops_if_the_terminal_switches_to_a_real_account(env):
    trader, decider, fake, journal = env
    fake.trade_mode = fake.ACCOUNT_TRADE_MODE_REAL
    decider.next = long()
    with pytest.raises(MT5Error, match="NOT a demo"):
        trader.tick(NOW)
    assert fake.requests == [] and decider.calls == 0


# --- agent ---------------------------------------------------------------
def make_df(n=3000) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    close = 4000 * np.exp(rng.normal(0, 0.001, n).cumsum())
    open_ = np.concatenate([[4000], close[:-1]])
    index = pd.date_range("2026-09-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 1, "low": np.minimum(open_, close) - 1,
                         "close": close, "volume": rng.uniform(50, 150, n)}, index=index)


DF = make_df()


def agent(**fields) -> MarginLLMDecider:
    base = {"reasoning": "Higher timeframes agree.", "action": "hold", "confidence": 0.8,
            "stop_atr_multiple": 2.0, "take_profit_r": 2.0}
    model = GenericFakeChatModel(messages=iter([AIMessage(content=json.dumps({**base, **fields}))]))
    return MarginLLMDecider(LLMConfig(structured_output="prompt", min_confidence=0.6), "15m", model=model)


def held(side: str) -> Position:
    qty = 7.0 if side == "long" else -7.0
    entry = float(DF["close"].iloc[-5])
    return Position(GOLD, qty, entry, int(DF.index[-5].timestamp() * 1000), qty * entry,
                    entry - 8 if side == "long" else entry + 8, "test", take_profit=entry + 16, ticket=1, lots=0.07)


def test_intraday_snapshot_shows_higher_timeframes_and_short_position():
    snapshot, atr = build_intraday_snapshot(GOLD, "15m", DF, held("short"))
    assert "NaN" not in json.dumps(snapshot) and atr > 0
    assert set(snapshot["higher_timeframes"]) == {"1h", "4h", "1d"}
    assert snapshot["higher_timeframes"]["1h"]["price_vs_ema200_pct"] is not None
    position = snapshot["open_position"]
    move = DF["close"].iloc[-1] / DF["close"].iloc[-5] - 1
    assert position["side"] == "short" and position["bars_held"] == 4
    assert position["unrealized_pct"] == pytest.approx(-move * 100, abs=0.001)   # a short gains when price falls
    assert "daily_context" not in snapshot


def test_agent_short_gets_clamped_stop_and_target():
    decision = agent(action="short", stop_atr_multiple=9, take_profit_r=0.2).decide(GOLD, DF, None)
    _, atr = build_intraday_snapshot(GOLD, "15m", DF, None)
    assert decision.action == "short"
    assert decision.stop_distance == pytest.approx(4 * atr)            # 9 clamped to 4 ATR
    assert decision.take_profit_distance == pytest.approx(4 * atr)     # 0.2R clamped to 1R


def test_agent_entry_below_confidence_is_a_hold():
    decision = agent(action="long", confidence=0.4).decide(GOLD, DF, None)
    assert decision.action == "hold" and "below confidence" in decision.reason


@pytest.mark.parametrize("side, said, expected", [
    ("long", "close", "close"), ("long", "short", "close"), ("long", "long", "hold"),
    ("short", "long", "close"), ("short", "hold", "hold"),
])
def test_agent_actions_while_in_a_position(side, said, expected):
    assert agent(action=said).decide(GOLD, DF, held(side)).action == expected


def test_agent_close_without_a_position_is_a_hold():
    assert agent(action="close").decide(GOLD, DF, None).action == "hold"


def test_journal_keeps_broker_details_of_a_position(tmp_path):
    journal = Journal(tmp_path / "j.sqlite")
    journal.open_position(held("short"))
    position = Journal(tmp_path / "j.sqlite").positions()[GOLD]
    assert (position.ticket, position.lots, position.side) == (1, 0.07, "short")
    journal.close_margin_position(position, position.entry_price - 8, 44.8, 1, "target")
    assert journal.positions() == {} and journal.get(f"position_extra:{GOLD}") is None


# --- weekend and release rules --------------------------------------------
def test_hours_to_weekend_close():
    from trading_agent.data.mt5 import hours_to_weekend_close

    assert hours_to_weekend_close(to_ms("2026-11-13 12:00") / 1000) == pytest.approx(10)     # Friday, closes 22:00 UTC
    assert hours_to_weekend_close(to_ms("2026-11-14 10:00") / 1000) == 0                     # Saturday
    assert hours_to_weekend_close(to_ms("2026-11-15 22:00") / 1000) == 0                     # Sunday before the open
    assert hours_to_weekend_close(to_ms("2026-11-15 23:30") / 1000) == pytest.approx(118.5)  # reopened
    assert hours_to_weekend_close(to_ms("2026-07-03 12:00") / 1000) == pytest.approx(9)      # summer: closes 21:00 UTC


def at(trader, fake, when: str) -> int:
    """Move the simulated clock to just after `when`, with a freshly closed candle in the store."""
    now = to_ms(when) + 30_000
    fake.now = now / 1000
    trader.store.upsert("mt5", GOLD, "15m", [[to_ms(when) - TF, 4000, 4002, 3998, 4000, 100]])
    return now


def test_no_entry_shortly_before_the_weekend(env):
    trader, decider, fake, journal = env
    decider.next = long()
    assert trader.tick(at(trader, fake, "2026-11-13 20:30")) == [] and fake.requests == []
    decision = journal.decisions_frame().iloc[-1]
    assert decision["action"] == "hold" and "closes for the weekend in 1.5 h" in decision["reason"]


def test_open_position_is_closed_before_the_weekend(env):
    trader, decider, fake, journal = env
    decider.next = long()
    trader.tick(at(trader, fake, "2026-11-13 15:00"))
    assert GOLD in journal.positions()
    assert trader.tick(at(trader, fake, "2026-11-13 21:00")) == []          # an hour before: still held
    events = trader.tick(at(trader, fake, "2026-11-13 21:45"))
    assert "before the weekend" in events[0] and journal.positions() == {} and fake.positions == {}
    assert journal.trades_frame().iloc[0]["exit_reason"] == "weekend"


def test_agent_is_not_asked_while_the_market_is_closed(env):
    trader, decider, fake, journal = env
    trader.tick(NOW)
    calls = decider.calls
    fake.symbol_info_tick = lambda name: SimpleNamespace(time=int(NOW / 1000) + OFFSET, bid=4000.0, ask=4000.3)
    saturday = at(trader, fake, "2026-11-14 10:00")                          # quotes stopped on Friday
    assert trader.tick(saturday) == [f"MARKET {GOLD} is closed: the agent waits for it to reopen"]
    assert trader.tick(saturday + 60_000) == [] and decider.calls == calls
    del fake.symbol_info_tick                                                # quotes flow again
    events = trader.tick(at(trader, fake, "2026-11-16 00:15"))
    assert events == [f"MARKET {GOLD} is open: the agent is deciding"] and decider.calls == calls + 1


def test_release_blackout_blocks_an_entry():
    release = {"in_hours": 0.2, "currency": "USD", "impact": "high", "event": "CPI m/m", "forecast": "", "previous": ""}
    base = {"reasoning": "Trend up.", "action": "long", "confidence": 0.9, "stop_atr_multiple": 2.0, "take_profit_r": 2.0}

    def decide(blackout, events):
        model = GenericFakeChatModel(messages=iter([AIMessage(content=json.dumps(base))]))
        decider = MarginLLMDecider(LLMConfig(structured_output="prompt"), "15m", model=model,
                                   event_blackout_minutes=blackout,
                                   context={"news": lambda s: {"headlines": [], "economic_calendar": events}})
        return decider.decide(GOLD, DF, None)

    blocked = decide(15, [release])
    assert blocked.action == "hold" and "CPI m/m in 12 min" in blocked.reason
    assert decide(15, [{**release, "in_hours": 2.0}]).action == "long"       # far enough away
    assert decide(15, [{**release, "impact": "medium"}]).action == "long"    # only high impact blocks
    assert decide(0, [release]).action == "long"                             # rule switched off


def test_candle_from_before_the_weekend_is_not_decided_on_at_the_reopen(env):
    trader, decider, fake, journal = env
    trader.tick(at(trader, fake, "2026-11-13 21:30"))
    calls = decider.calls
    trader.store.upsert("mt5", GOLD, "15m", [[to_ms("2026-11-13 21:45"), 4000, 4002, 3998, 4000, 100]])
    fake.copy_rates_from_pos = lambda *args: None                            # the terminal has no newer candle yet
    fake.now = (to_ms("2026-11-15 23:00") + 20_000) / 1000                   # Sunday, seconds after the reopen
    events = trader.tick(int(fake.now * 1000))
    assert decider.calls == calls and any("stale" in e for e in events)
    assert len(trader.tick(at(trader, fake, "2026-11-15 23:15"))) == 0 and decider.calls == calls + 1
