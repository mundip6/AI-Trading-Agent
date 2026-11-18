import dataclasses

import pandas as pd
import pytest

from trading_agent.broker import PaperBroker
from trading_agent.config import load_config
from trading_agent.data.store import CandleStore, timeframe_ms, to_ms
from trading_agent.journal import Journal
from trading_agent.markets import crypto_overview, stock_overview
from trading_agent.trader import Decision, PaperTrader
from trading_agent.watchlist import read_watchlist, watchlist_path, write_watchlist

CONFIG = """
[data]
exchange = "binance"
db_path = "{root}/m.sqlite"
symbols = ["BTC/USDT", "ETH/USDT"]

[paper]
db_path = "{root}/p.sqlite"
fee_rate = 0.0
slippage = 0.0
"""


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG.format(root=tmp_path.as_posix()), encoding="utf-8")
    return path


# --- watchlist ----------------------------------------------------------
def test_watchlist_overrides_config_symbols(config_path):
    assert load_config(config_path).data.symbols == ("BTC/USDT", "ETH/USDT")
    assert read_watchlist(config_path) is None

    write_watchlist(config_path, ["SOL/USDT", "BTC/USDT", "SOL/USDT"])
    cfg = load_config(config_path)
    assert cfg.data.symbols == cfg.paper.symbols == ("SOL/USDT", "BTC/USDT")     # order kept, duplicates dropped
    assert cfg.source_path == str(config_path)

    write_watchlist(config_path, [])
    assert load_config(config_path).paper.symbols == ()                          # an empty watchlist means nothing


def test_extra_tickers_survive_a_symbol_save_and_bad_files_are_ignored(config_path):
    write_watchlist(config_path, ["SPY"], extra=["VUAA.DE"])
    write_watchlist(config_path, ["SPY", "QQQ"])
    assert read_watchlist(config_path) == {"symbols": ["SPY", "QQQ"], "extra": ["VUAA.DE"]}

    watchlist_path(config_path).write_text("not json", encoding="utf-8")
    assert read_watchlist(config_path) is None
    assert load_config(config_path).data.symbols == ("BTC/USDT", "ETH/USDT")


# --- running loop follows the watchlist ---------------------------------
class Scripted:
    name = "scripted"

    def __init__(self):
        self.next, self.asked = {}, []

    def decide(self, symbol, df, position):
        self.asked.append(symbol)
        return self.next.pop(symbol, Decision("hold", "nothing"))


def test_loop_picks_up_changes_and_keeps_managing_removed_positions(config_path):
    cfg = load_config(config_path)
    tf = timeframe_ms("4h")
    t0 = to_ms("2024-01-01")
    store = CandleStore(cfg.data.db_path)
    for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        store.upsert("binance", symbol, "4h", [[t0 + i * tf, 100, 101, 99, 100, 1] for i in range(50)])
    journal = Journal(cfg.paper.db_path)
    decider = Scripted()
    trader = PaperTrader(cfg, PaperBroker(journal, lambda s: 100.0, 10_000, 0, 0), journal, store, decider,
                         sync=lambda s, t: None)
    now = t0 + 50 * tf + 60_000

    decider.next["BTC/USDT"] = Decision("buy", "entry", stop_distance=5)
    trader.tick(now)
    assert sorted(decider.asked) == ["BTC/USDT", "ETH/USDT"] and "BTC/USDT" in journal.positions()

    write_watchlist(config_path, ["SOL/USDT"])              # BTC (held) and ETH come off, SOL goes on
    for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        store.upsert("binance", symbol, "4h", [[t0 + 50 * tf, 100, 101, 99, 100, 1]])
    decider.asked.clear()
    decider.next["BTC/USDT"] = Decision("sell", "exit")
    events = trader.tick(now + tf)

    assert any(e.startswith("WATCH") and "added SOL/USDT" in e for e in events)
    assert sorted(decider.asked) == ["BTC/USDT", "SOL/USDT"]          # ETH dropped, held BTC still managed
    assert journal.positions() == {}

    store.upsert("binance", "BTC/USDT", "4h", [[t0 + 51 * tf, 100, 101, 99, 100, 1]])
    decider.asked.clear()
    trader.tick(now + 2 * tf)
    assert "BTC/USDT" not in decider.asked                             # closed and unmonitored: left alone


# --- overviews ----------------------------------------------------------
class FakeExchange:
    def load_markets(self):
        market = lambda quote, spot=True, active=True: {"spot": spot, "active": active, "quote": quote}
        return {"BTC/USDT": market("USDT"), "USDC/USDT": market("USDT"), "NEW/USDT": market("USDT"),
                "ETH/BTC": market("BTC"), "OLD/USDT": market("USDT", active=False),
                "BTC/USDT:USDT": market("USDT", spot=False)}

    def fetch_tickers(self):
        return {"BTC/USDT": {"last": 110.0, "percentage": 2.5, "quoteVolume": 5e8, "high": 112, "low": 104},
                "USDC/USDT": {"last": 1.0002, "percentage": 0.01, "quoteVolume": 9e8, "high": 1.0005, "low": 0.9998},
                "NEW/USDT": {"last": 2.0, "percentage": -4.0, "quoteVolume": 1e6, "high": 2.2, "low": 1.9}}

    def fetch_ohlcv(self, symbol, timeframe, limit):
        if symbol == "NEW/USDT":
            return [[0, 0, 0, 0, 1.0, 0]] * 3            # listed three days ago
        if symbol == "USDC/USDT":
            raise OSError("down")
        return [[0, 0, 0, 0, 50.0 + i * 2, 0] for i in range(31)]   # 50 ... 110


def test_crypto_overview():
    table = crypto_overview(FakeExchange(), workers=2).set_index("symbol")
    assert sorted(table.index) == ["BTC/USDT", "NEW/USDT", "USDC/USDT"]
    btc = table.loc["BTC/USDT"]
    assert btc["change_1d"] == 2.5
    assert btc["change_7d"] == pytest.approx((110 / 96 - 1) * 100)      # the close seven days back
    assert btc["change_30d"] == pytest.approx(120.0)
    assert btc["kind"] == "Crypto" and table.loc["USDC/USDT", "kind"] == "Stablecoin"
    assert pd.isna(table.loc["NEW/USDT", "change_7d"]) and pd.isna(table.loc["USDC/USDT", "change_30d"])


class FakeYF:
    def download(self, tickers, **kwargs):
        index = pd.bdate_range("2026-08-03", periods=30)
        frames = {
            "SPY": pd.DataFrame({"Close": [100.0 + i for i in range(30)], "Volume": 1_000.0}, index=index),
            "NEWCO": pd.DataFrame({"Close": [10.0, 11.0] + [float("nan")] * 28, "Volume": 5.0}, index=index),
            "GONE": pd.DataFrame({"Close": [float("nan")] * 30, "Volume": 0.0}, index=index),
        }
        return pd.concat({t: frames[t] for t in tickers if t in frames}, axis=1)


def test_stock_overview():
    table = stock_overview(["SPY", "NEWCO", "GONE", "MISSING", "SPY"], yfinance=FakeYF()).set_index("symbol")
    assert list(table.index) == ["SPY", "NEWCO"]
    spy = table.loc["SPY"]
    assert spy["price"] == 129.0
    assert spy["change_1d"] == pytest.approx((129 / 128 - 1) * 100)
    assert spy["change_7d"] == pytest.approx((129 / 124 - 1) * 100)     # five trading days
    assert spy["change_30d"] == pytest.approx((129 / 108 - 1) * 100)    # twenty-one trading days
    assert spy["volume"] == pytest.approx(129_000.0)
    assert pd.isna(table.loc["NEWCO", "change_7d"])
    assert stock_overview([], yfinance=FakeYF()).empty
