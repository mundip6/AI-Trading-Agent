import dataclasses
import time

import pandas as pd
import pytest

from trading_agent.backtest.metrics import _curve_stats
from trading_agent.broker import PaperBroker
from trading_agent.config import Config, DataConfig, PaperConfig
from trading_agent.data.exchange import is_continuous, sync_candles
from trading_agent.data.store import CandleStore, to_ms
from trading_agent.data.yahoo import YahooExchange
from trading_agent.journal import Journal
from trading_agent.trader import Decision, PaperTrader

DAY = 86_400_000


# --- Yahoo adapter ------------------------------------------------------
class FakeTicker:
    def __init__(self, daily, minutes):
        self._daily, self._minutes = daily, minutes
        self.fast_info = {"currency": "USD", "last_price": 99.0}

    def history(self, start=None, interval="1d", auto_adjust=True, period=None):
        if interval == "1m":
            return self._minutes
        return self._daily[self._daily.index >= pd.Timestamp(start, tz="America/New_York")]


class FakeYF:
    def __init__(self, daily, minutes):
        self.daily, self.minutes, self.calls = daily, minutes, 0

    def Ticker(self, symbol):
        self.calls += 1
        return FakeTicker(self.daily, self.minutes)


def daily_frame(dates, closes):
    index = pd.DatetimeIndex(dates, tz="America/New_York")
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000.0}, index=index)


def minute_frame(last_trade: pd.Timestamp, price=101.0):
    return pd.DataFrame({"Close": [price]}, index=pd.DatetimeIndex([last_trade]))


def test_daily_candles_are_stamped_with_the_trading_date_in_utc():
    yf = FakeYF(daily_frame(["2024-01-04", "2024-01-05", "2024-01-08"], [10.0, 11.0, 12.0]), minute_frame(pd.Timestamp.now(tz="UTC")))
    rows = YahooExchange(yf).fetch_ohlcv("SPY", "1d", since=to_ms("2024-01-05"))
    assert [r[0] for r in rows] == [to_ms("2024-01-05"), to_ms("2024-01-08")]   # weekend gap kept, no shift
    assert rows[0][4] == 11.0
    with pytest.raises(ValueError):
        YahooExchange(yf).fetch_ohlcv("SPY", "4h")


def test_sync_refetches_history_and_drops_the_unfinished_day(tmp_path):
    today = pd.Timestamp.now(tz="UTC").normalize()
    dates = [(today - pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in (3, 2, 1, 0)]
    yf = FakeYF(daily_frame(dates, [10.0, 11.0, 12.0, 13.0]), minute_frame(pd.Timestamp.now(tz="UTC")))
    store = CandleStore(tmp_path / "m.sqlite")
    exchange = YahooExchange(yf)

    assert sync_candles(store, "yahoo", "SPY", "1d", dates[0], exchange) == 3    # today's bar is still forming
    yf.daily = daily_frame(dates, [5.0, 5.5, 6.0, 6.5])                          # a 2:1 split re-adjusts history
    assert sync_candles(store, "yahoo", "SPY", "1d", dates[0], exchange) == 0    # nothing new...
    assert store.load("yahoo", "SPY", "1d")["close"].tolist() == [5.0, 5.5, 6.0]  # ...but the old bars are corrected


def test_quote_cache_and_market_open():
    now = pd.Timestamp.now(tz="UTC")
    yf = FakeYF(daily_frame([], []), minute_frame(now - pd.Timedelta(minutes=5)))
    exchange = YahooExchange(yf)
    assert exchange.fetch_ticker("SPY")["last"] == 101.0
    assert exchange.is_market_open("SPY")
    assert yf.calls == 1                                   # second read came from the cache

    closed = YahooExchange(FakeYF(daily_frame([], []), minute_frame(now - pd.Timedelta(hours=9))))
    assert not closed.is_market_open("SPY")
    empty = YahooExchange(FakeYF(daily_frame([], []), pd.DataFrame({"Close": []})))
    assert empty.fetch_ticker("SPY")["last"] == 99.0 and not empty.is_market_open("SPY")
    assert is_continuous("binance") and not is_continuous("yahoo")


# --- trading loop with a market that closes ---------------------------------
class Scripted:
    name = "scripted"

    def __init__(self):
        self.next, self.calls = {}, 0

    def decide(self, symbol, df, position):
        self.calls += 1
        return self.next.pop(symbol, Decision("hold", "nothing"))


@pytest.fixture
def stock_env(tmp_path):
    cfg = Config(
        data=DataConfig(exchange="yahoo", db_path=str(tmp_path / "m.sqlite"), symbols=("SPY",), timeframes=("1d",)),
        paper=PaperConfig(db_path=str(tmp_path / "p.sqlite"), timeframe="1d", fee_rate=0, slippage=0),
    )
    store = CandleStore(cfg.data.db_path)
    friday = to_ms("2024-01-05")
    store.upsert("yahoo", "SPY", "1d", [[friday - (59 - i) * DAY, 100, 101, 99, 100, 1] for i in range(60)])
    journal = Journal(cfg.paper.db_path)
    state = {"open": False, "price": 100.0, "syncs": 0}
    broker = PaperBroker(journal, lambda s: state["price"], 10_000, 0, 0, market_open=lambda s: state["open"])
    decider = Scripted()

    def sync(symbol, timeframe):
        state["syncs"] += 1

    return PaperTrader(cfg, broker, journal, store, decider, sync), decider, state, journal, friday


def test_decisions_wait_for_the_market_to_open(stock_env):
    trader, decider, state, journal, friday = stock_env
    decider.next["SPY"] = Decision("buy", "entry", stop_distance=5)

    sunday = friday + 2 * DAY + 12 * 3_600_000
    assert trader.tick(sunday) == []                       # weekend: not stale, not traded, not even asked
    assert decider.calls == 0 and journal.positions() == {}

    state["open"] = True
    monday_open = friday + 3 * DAY + 14 * 3_600_000 + 35 * 60_000
    events = trader.tick(monday_open)
    assert any(e.startswith("BUY") for e in events)
    assert journal.positions()["SPY"].stop == pytest.approx(95)


def test_stops_are_not_filled_while_closed(stock_env):
    trader, decider, state, journal, friday = stock_env
    state["open"] = True
    decider.next["SPY"] = Decision("buy", "entry", stop_distance=5)
    trader.tick(friday + DAY + 15 * 3_600_000)

    state.update(open=False, price=90.0)                   # a stale quote below the stop, market closed
    assert trader.tick(friday + DAY + 23 * 3_600_000) == []
    assert "SPY" in journal.positions()
    state["open"] = True                                   # the next session opens below the stop
    assert any(e.startswith("STOP") for e in trader.tick(friday + 3 * DAY + 15 * 3_600_000))


def test_weekend_downloads_are_throttled(stock_env):
    trader, _, state, _, friday = stock_env
    sunday = friday + 2 * DAY + 3_600_000
    for minute in range(0, 60, 5):                         # an hour of 5-minute passes
        trader.tick(sunday + minute * 60_000)
    assert state["syncs"] == 2                             # one attempt per half hour, not twelve


def test_data_older_than_a_long_weekend_is_stale(stock_env):
    trader, decider, state, journal, friday = stock_env
    state["open"] = True
    decider.next["SPY"] = Decision("buy", "entry", stop_distance=5)
    events = trader.tick(friday + 8 * DAY)
    assert any("stale" in e for e in events) and journal.positions() == {}
    assert trader.tick(friday + 8 * DAY + 60_000) == []    # reported once, not every pass


# --- metrics ------------------------------------------------------------
def test_sharpe_annualisation_counts_real_trading_days():
    days = pd.bdate_range("2023-01-02", periods=504, tz="UTC")          # two years of weekdays
    returns = pd.Series([0.01, -0.005] * 252, index=days)
    equity = 10_000 * (1 + returns).cumprod()
    stats = _curve_stats(equity, 10_000)
    realised = equity.pct_change().dropna()
    per_year = len(realised) / ((days[-1] - days[0]) / pd.Timedelta(days=365))
    assert 250 < per_year < 262
    assert stats["sharpe"] == pytest.approx(realised.mean() / realised.std() * per_year ** 0.5)
