import dataclasses

import pytest

from trading_agent.broker import PaperBroker
from trading_agent.config import Config, PaperConfig, RiskConfig
from trading_agent.data.store import CandleStore, timeframe_ms, to_ms
from trading_agent.journal import Journal
from trading_agent.risk import RiskManager
from trading_agent.trader import Decision, PaperTrader

TF = timeframe_ms("4h")
T0 = to_ms("2024-01-01")
N_BARS = 50
LAST_BAR = T0 + (N_BARS - 1) * TF
NOW = LAST_BAR + TF + 60_000          # one minute after the last candle closed
SYMBOLS = ("BTC/USDT", "ETH/USDT")


class ScriptedDecider:
    name = "scripted"

    def __init__(self):
        self.next: dict[str, Decision] = {}
        self.calls = 0

    def decide(self, symbol, df, has_position):
        self.calls += 1
        return self.next.pop(symbol, Decision("hold", "nothing to do"))


@pytest.fixture
def env(tmp_path):
    cfg = Config(
        paper=PaperConfig(db_path=str(tmp_path / "paper.sqlite"), symbols=SYMBOLS, fee_rate=0, slippage=0),
        risk=RiskConfig(),
    )
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, db_path=str(tmp_path / "m.sqlite")))
    store = CandleStore(cfg.data.db_path)
    for symbol in SYMBOLS:
        store.upsert(cfg.data.exchange, symbol, "4h", [[T0 + i * TF, 100, 101, 99, 100, 1] for i in range(N_BARS)])
    journal = Journal(cfg.paper.db_path)
    prices = {"BTC/USDT": 100.0, "ETH/USDT": 100.0}
    broker = PaperBroker(journal, prices.__getitem__, 10_000, fee_rate=0, slippage=0)
    decider = ScriptedDecider()
    synced = []
    trader = PaperTrader(cfg, broker, journal, store, decider, sync=lambda s, tf: synced.append(s))
    return trader, decider, prices, journal, store, synced


# --- risk ---------------------------------------------------------------
def size(**overrides):
    args = dict(equity=10_000, cash=10_000, exposure=0, open_positions=0, price=100,
                stop_distance=5, day_start_equity=10_000)
    return RiskManager(RiskConfig()).size_entry(**{**args, **overrides})


def test_risk_sizes_to_stop_distance():
    verdict = size()
    assert verdict.approved and verdict.qty == pytest.approx(20)   # 100 risk / 5 per unit


def test_risk_caps_position_and_exposure():
    assert size(stop_distance=1).qty == pytest.approx(30)           # 30% of equity
    assert size(stop_distance=1, exposure=8_000).qty == pytest.approx(10)  # 90% total cap


@pytest.mark.parametrize(
    "overrides, text",
    [
        ({"equity": 9_600}, "daily loss"),
        ({"open_positions": 3}, "positions"),
        ({"stop_distance": None}, "stop-loss"),
        ({"stop_distance": float("nan")}, "stop-loss"),
        ({"exposure": 9_000}, "below the minimum"),
        ({"cash": 5}, "below the minimum"),
    ],
)
def test_risk_rejections(overrides, text):
    verdict = size(**overrides)
    assert not verdict.approved and text in verdict.reason


# --- broker -------------------------------------------------------------
def test_paper_broker_fees_slippage_and_persistence(tmp_path):
    journal = Journal(tmp_path / "p.sqlite")
    broker = PaperBroker(journal, lambda s: 100.0, 1_000, fee_rate=0.001, slippage=0.01)
    fill = broker.buy("BTC/USDT", 2)
    assert fill.price == pytest.approx(101) and fill.fee == pytest.approx(0.202)
    assert broker.cash() == pytest.approx(1_000 - 202 - 0.202)
    with pytest.raises(ValueError):
        broker.buy("BTC/USDT", 100)

    reopened = PaperBroker(Journal(tmp_path / "p.sqlite"), lambda s: 100.0, 1_000)
    assert reopened.cash() == pytest.approx(broker.cash())   # initial cash is not re-applied


# --- trading loop -------------------------------------------------------
def test_buy_then_signal_exit(env):
    trader, decider, prices, journal, store, _ = env
    decider.next["BTC/USDT"] = Decision("buy", "test entry", stop_distance=5)
    events = trader.tick(NOW)

    pos = journal.positions()["BTC/USDT"]
    assert pos.qty == pytest.approx(20) and pos.stop == pytest.approx(95)
    assert any(e.startswith("BUY") for e in events)
    assert trader.equity() == pytest.approx(10_000)

    # Same candle again: nothing is re-decided.
    calls = decider.calls
    assert trader.tick(NOW + 60_000) == []
    assert decider.calls == calls

    # A new candle closes with an exit signal at a higher price.
    store.upsert(trader.cfg.data.exchange, "BTC/USDT", "4h", [[LAST_BAR + TF, 100, 111, 99, 110, 1]])
    prices["BTC/USDT"] = 110.0
    decider.next["BTC/USDT"] = Decision("sell", "test exit")
    trader.tick(NOW + TF)

    assert journal.positions() == {}
    trades = journal.trades_frame()
    assert trades["pnl"].iloc[0] == pytest.approx(200) and trades["exit_reason"].iloc[0] == "signal"
    assert trader.broker.cash() == pytest.approx(10_200)


def test_stop_loss_closes_position(env):
    trader, decider, prices, journal, *_ = env
    decider.next["BTC/USDT"] = Decision("buy", "test entry", stop_distance=5)
    trader.tick(NOW)

    prices["BTC/USDT"] = 94.0
    events = trader.tick(NOW + 60_000)
    assert any(e.startswith("STOP") for e in events)
    assert journal.positions() == {}
    assert journal.trades_frame()["exit_reason"].iloc[0] == "stop"
    assert trader.broker.cash() == pytest.approx(10_000 - 20 * 6)


def test_risk_rejection_is_journaled_not_traded(env):
    trader, decider, _, journal, *_ = env
    decider.next["BTC/USDT"] = Decision("buy", "entry without stop")
    trader.tick(NOW)
    assert journal.positions() == {}
    decisions = journal.decisions_frame()
    row = decisions[decisions["symbol"] == "BTC/USDT"].iloc[0]
    assert row["action"] == "rejected" and "stop-loss" in row["reason"]


def test_daily_loss_limit_blocks_new_entries(env):
    trader, decider, _, journal, *_ = env
    journal.record_equity(NOW - 86_400_000, 10_400, 10_400)   # yesterday's close: 4% higher
    decider.next["BTC/USDT"] = Decision("buy", "entry", stop_distance=5)
    trader.tick(NOW)
    assert journal.positions() == {}
    assert "daily loss" in journal.decisions_frame().query("symbol == 'BTC/USDT'")["reason"].iloc[0]


def test_stale_candles_are_not_traded(env):
    trader, decider, _, journal, _, synced = env
    decider.next["BTC/USDT"] = Decision("buy", "entry", stop_distance=5)
    events = trader.tick(NOW + 5 * TF)   # sync is a no-op here, so data stays old
    assert "BTC/USDT" in synced
    assert journal.positions() == {}
    assert any("stale" in e for e in events)


def test_journal_is_bound_to_one_broker(env):
    trader, decider, *_ = env

    class Other(PaperBroker):
        name = "other"

    other = Other(trader.journal, lambda s: 1.0, 1)
    with pytest.raises(RuntimeError):
        PaperTrader(trader.cfg, other, trader.journal, trader.store, decider, sync=lambda s, tf: None)


# --- news briefing ---------------------------------------------------------
class BriefingTrader:
    """Just the briefing rule of the trading loop, with its own journal."""

    def __init__(self, tmp_path, every=4.0, minimum=1.0):
        from trading_agent.config import AlertsConfig

        self.cfg = Config(alerts=AlertsConfig(news_every_hours=every, news_min_hours=minimum))
        self.journal = Journal(tmp_path / "b.sqlite")

    def brief(self, hours: float, outlook: str, news: str = "Fed chair speaks today; traders expect a rate hike."):
        events: list[str] = []
        decision = Decision("hold", "no edge", news=news, outlook=outlook)
        PaperTrader._news_briefing(self, "XAUUSD.s", decision, "hold", NOW + int(hours * 3_600_000), events)
        return events


def test_news_briefing_cadence(tmp_path):
    trader = BriefingTrader(tmp_path)
    first = trader.brief(0, "down")
    assert first == ["NEWS  XAUUSD.s: price likely DOWN\nFed chair speaks today; traders expect a rate hike.\n"
                     "Agent's action now: hold"]
    assert trader.brief(0.25, "down") == []                 # same view: wait for the interval
    assert trader.brief(0.5, "up") == []                    # view changed, but less than an hour since the last
    assert "likely UP" in trader.brief(1.25, "up")[0]       # changed view, an hour later: sent early
    assert trader.brief(3.0, "up") == []
    assert "likely UP" in trader.brief(5.25, "up")[0]       # four hours after the last one
    assert "direction unclear" in trader.brief(9.5, "unclear")[0]


def test_news_briefing_off_or_empty(tmp_path):
    assert BriefingTrader(tmp_path, every=0).brief(0, "down") == []
    assert BriefingTrader(tmp_path).brief(0, "down", news="") == []


def test_decision_log_keeps_the_briefing(env):
    trader, decider, prices, journal, store, synced = env
    decider.next["BTC/USDT"] = Decision("hold", "no edge", news="Ceasefire talks resumed.", outlook="down")
    events = trader.tick(NOW)
    assert any(e.startswith("NEWS  BTC/USDT: price likely DOWN") for e in events)
    reasons = journal.decisions_frame()["reason"].tolist()
    assert "no edge | News: Ceasefire talks resumed. Outlook: down." in reasons
