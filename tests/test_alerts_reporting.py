import pandas as pd
import pytest

from trading_agent.alerts import Alerter
from trading_agent.config import AlertsConfig
from trading_agent.data.store import CandleStore, to_ms
from trading_agent.journal import DAY_MS, Journal, Position
from trading_agent.reporting import account_summary, benchmark_curve, indexed


# --- alerts -------------------------------------------------------------
def make_alerter(**cfg):
    sent = []
    return Alerter(AlertsConfig(telegram=True, **cfg), sent.append, label="demo"), sent


def test_only_enabled_event_kinds_are_sent():
    alerter, sent = make_alerter()
    count = alerter.handle_events(["BUY   BTC/USDT bought", "SKIP  ETH/USDT: stale", "STOP  SOL/USDT hit"])
    assert count == 2
    assert sent == ["[demo] BUY   BTC/USDT bought", "[demo] STOP  SOL/USDT hit"]


def test_repeated_errors_are_throttled():
    alerter, sent = make_alerter(error_cooldown_minutes=60)
    alerter.handle_events(["ERROR boom"], now=0)
    alerter.handle_events(["ERROR boom"], now=600)        # 10 min later: suppressed
    alerter.handle_events(["ERROR other"], now=600)       # a different error: sent
    alerter.handle_events(["ERROR boom"], now=3700)       # after the cooldown: sent again
    assert len(sent) == 3


def test_failed_delivery_does_not_raise_and_disabled_is_silent(capsys):
    def broken(text):
        raise OSError("network down")

    assert Alerter(AlertsConfig(telegram=True), broken).send("x") is False
    assert "not delivered" in capsys.readouterr().out
    off = Alerter(AlertsConfig(), None)
    assert not off.enabled and off.handle_events(["BUY x"]) == 0


def test_daily_summary_once_per_day(tmp_path):
    alerter, sent = make_alerter()
    journal = Journal(tmp_path / "p.sqlite")
    journal.set("initial_equity", 10_000)
    day1 = to_ms("2024-01-01 09:00")
    journal.record_equity(day1, 10_000, 10_000)
    journal.open_position(Position("BTC/USDT", 1, 100, day1, 100, 95, "test"))

    assert alerter.maybe_daily_summary(journal, 10_000, day1) is False     # first run
    assert alerter.maybe_daily_summary(journal, 10_100, day1 + 3_600_000) is False
    assert alerter.maybe_daily_summary(journal, 10_200, day1 + DAY_MS) is True
    assert alerter.maybe_daily_summary(journal, 10_200, day1 + DAY_MS + 60_000) is False
    assert len(sent) == 1
    assert "+2.00% on the day" in sent[0] and "BTC/USDT" in sent[0]


# --- reporting ----------------------------------------------------------
def test_account_summary():
    times = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    equity = pd.Series([10_000.0, 11_000, 9_900, 10_500], index=times)
    trades = pd.DataFrame({"pnl": [300.0, -100.0, 300.0]})
    summary = account_summary(equity, trades, initial=10_000)
    assert summary["total_return"] == pytest.approx(0.05)
    assert summary["max_drawdown"] == pytest.approx(-0.1)
    assert summary["win_rate"] == pytest.approx(2 / 3)
    assert summary["profit_factor"] == pytest.approx(6)
    assert account_summary(pd.Series(dtype=float), trades.iloc[:0])["trades"] == 0


def test_benchmark_uses_closes_known_at_the_time(tmp_path):
    store = CandleStore(tmp_path / "m.sqlite")
    t0, hour = to_ms("2024-01-01"), 3_600_000
    #                       open-time, o, h, l, close
    store.upsert("x", "A/USDT", "1h", [[t0 + i * hour, 0, 0, 0, c, 1] for i, c in enumerate([100, 110, 120, 130])])
    store.upsert("x", "B/USDT", "1h", [[t0 + i * hour, 0, 0, 0, c, 1] for i, c in enumerate([50, 50, 40, 40])])
    # 01:30 -> the 00:00 candle (close 100 / 50) is the latest closed one; 03:30 -> the 02:00 candle.
    times = pd.DatetimeIndex(["2024-01-01 01:30", "2024-01-01 03:30"], tz="UTC")
    curve = benchmark_curve(store, "x", ("A/USDT", "B/USDT"), "1h", times)
    assert curve.iloc[0] == pytest.approx(100)
    assert curve.iloc[1] == pytest.approx((120 + 80) / 2)
    assert indexed(pd.Series([50.0, 75.0])).tolist() == [100.0, 150.0]
    assert benchmark_curve(store, "x", ("NONE/USDT",), "1h", times).empty
