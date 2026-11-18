import json

import pytest
from langchain_core.messages import AIMessage

from trading_agent.config import CopyTradeConfig, LLMConfig
from trading_agent.copytrade import CopyStore, CopyTrading
from trading_agent.copytrade.ranking import TraderStats, add_history_metrics, prefilter, rank

DAY = 86_400_000
HOUR = 3_600_000
CFG = CopyTradeConfig(followed=2, candidates=3, min_account_value=1_000, max_monthly_turnover=50,
                      min_history_days=90, max_drawdown=0.5, refresh_minutes=30, max_age_minutes=240)


def perf(pnl, roi, vlm):
    return {"pnl": str(pnl), "roi": str(roi), "vlm": str(vlm)}


def row(address, value, month_pnl, month_vlm, all_pnl, month_roi=0.1):
    return {"ethAddress": address, "accountValue": str(value), "displayName": None,
            "windowPerformances": [["day", perf(0, 0, 0)], ["week", perf(0, 0, 0)],
                                   ["month", perf(month_pnl, month_roi, month_vlm)],
                                   ["allTime", perf(all_pnl, 0.5, month_vlm * 10)]]}


def portfolio(daily_pnl_changes, alltime_pnl, capital=10_000.0, days=200):
    month, total = [], 0.0
    for i, change in enumerate(daily_pnl_changes):
        total += change
        month.append([i * DAY, str(total)])
    alltime = [[i * DAY * days // (len(alltime_pnl) - 1), str(v)] for i, v in enumerate(alltime_pnl)]
    values = [[t, str(capital)] for t, _ in alltime]
    return {"perpMonth": {"pnlHistory": month}, "perpAllTime": {"pnlHistory": alltime, "accountValueHistory": values}}


# --- ranking ------------------------------------------------------------
def test_prefilter_screens_and_orders_by_alltime_profit():
    rows = [
        row("small", 500, 10, 100, 100),                 # account too small
        row("loser", 5_000, 10, 100, -50),               # unprofitable overall
        row("cold", 5_000, -10, 100, 500),               # losing month
        row("maker", 5_000, 10, 5_000_000, 900),         # turnover far too high
        row("a", 5_000, 10, 1_000, 100),
        row("b", 5_000, 10, 1_000, 400),
        row("c", 5_000, 10, 1_000, 300),
        row("d", 5_000, 10, 1_000, 200),
    ]
    assert [t.address for t in prefilter(rows, CFG)] == ["b", "c", "d"]   # capped at `candidates`


def test_history_metrics():
    trader = TraderStats("x", "", 10_000, 0.1, 3_000)
    steady = [0] + [100, 100, -50, 100] * 4          # 12 up days, 4 down days
    add_history_metrics(trader, portfolio(steady, [0, 2_000, 1_000, 3_000]))
    assert trader.history_days == pytest.approx(200)
    assert trader.consistency == pytest.approx(0.75)
    assert trader.max_drawdown == pytest.approx(0.1)       # gave back 1,000 on 10,000 capital
    assert trader.profit_to_pain == pytest.approx(3.0)

    short = add_history_metrics(TraderStats("y", "", 1, 0, 0), portfolio([0, 5, 5], [0, 10]))
    assert short.consistency == 0.0                        # too few days to judge


def test_rank_applies_hard_limits_and_prefers_steady_traders():
    def trader(address, **kw):
        base = dict(name="", account_value=10_000, month_roi=0.1, alltime_pnl=1_000, consistency=0.6,
                    max_drawdown=0.2, profit_to_pain=3.0, history_days=200)
        return TraderStats(address, **{**base, **kw})

    ranked = rank([
        trader("steady", consistency=0.8, max_drawdown=0.05, profit_to_pain=10),
        trader("ok"),
        trader("wild", month_roi=0.9, consistency=0.4, max_drawdown=0.45, profit_to_pain=1.1),
        trader("new", history_days=20, month_roi=5.0),
        trader("blown", max_drawdown=0.8),
    ], CFG)
    assert [t.address for t in ranked] == ["steady", "ok"]
    assert ranked[0].score > ranked[1].score
    assert rank([], CFG) == []


# --- service ------------------------------------------------------------
class FakeClient:
    def __init__(self):
        self.accounts = {}
        self.calls = 0

    def positions(self, address):
        self.calls += 1
        return self.accounts[address]


def pos(coin, size, value):
    return {"coin": coin, "size": size, "entry_price": 1.0, "value": value, "leverage": 3.0}


@pytest.fixture
def copy(tmp_path):
    store = CopyStore(tmp_path / "c.sqlite")
    store.replace_traders([TraderStats(a, "", 10_000, 0.1, 1_000, 0.7, 0.1, 5.0, 200, 0.9) for a in "abcd"], 0)
    return CopyTrading(CFG, store, FakeClient())


def test_store_roundtrips_traders(copy):
    traders = copy.store.traders()
    assert [t.address for t in traders] == list("abcd")
    assert traders[0].consistency == 0.7 and traders[0].score == 0.9
    assert copy.store.ranked_at() == 0


def test_consensus_counts_exposure_and_changes(copy):
    t0 = 100 * DAY
    copy.client.accounts = {
        "a": (10_000, [pos("BTC", 1, 5_000)]),
        "b": (10_000, [pos("BTC", -1, 2_000)]),
        "c": (10_000, []),
        "d": (10_000, [pos("ETH", 1, 1_000)]),
    }
    assert copy.update(t0) == 4
    first = copy.consensus("BTC/USDT", t0 + 5 * 60_000)
    assert (first["long"], first["short"], first["no_position"]) == (1, 1, 2)
    assert first["avg_exposure_pct_of_account"] == pytest.approx(7.5)    # (50 - 20 + 0 + 0) / 4
    assert first["data_age_minutes"] == 5 and first["changes_last_24h"] is None

    copy.client.accounts = {
        "a": (10_000, []),                              # closed long
        "b": (10_000, [pos("BTC", 1, 2_000)]),          # flipped to long
        "c": (10_000, [pos("BTC", 2, 4_000)]),          # opened long
        "d": (10_000, [pos("BTC", -1, 1_000)]),         # opened short
    }
    copy.update(t0 + DAY)
    changes = copy.consensus("BTC/USDT", t0 + DAY)["changes_last_24h"]
    assert changes == {"hours_compared": 24.0, "opened_long": 1, "opened_short": 1, "closed_long": 1,
                       "closed_short": 0, "flipped_to_long": 1, "flipped_to_short": 0}
    assert copy.consensus("DOGE/USDT", t0 + DAY)["long"] == 0


def test_context_refreshes_when_stale_and_expires(copy):
    t0 = 100 * DAY
    copy.client.accounts = {a: (10_000, [pos("BTC", 1, 1_000)]) for a in "abcd"}
    assert copy.context("BTC/USDT", t0)["long"] == 4          # no snapshot yet -> reads positions
    assert copy.client.calls == 4
    copy.context("BTC/USDT", t0 + 10 * 60_000)                # fresh enough -> no new read
    assert copy.client.calls == 4
    copy.context("BTC/USDT", t0 + HOUR)                       # stale -> refreshed
    assert copy.client.calls == 8

    copy.client.accounts = {}                                 # the feed goes down
    assert copy.context("BTC/USDT", t0 + 2 * HOUR)["data_age_minutes"] == 60
    assert copy.context("BTC/USDT", t0 + 6 * HOUR) is None    # too old to trust


def test_context_is_none_without_followed_traders(tmp_path):
    assert CopyTrading(CFG, CopyStore(tmp_path / "e.sqlite"), FakeClient()).context("BTC/USDT") is None


# --- into the agent -----------------------------------------------------
def test_llm_snapshot_carries_followed_traders():
    from test_agent import DF, reply
    from trading_agent.agent import LLMDecider

    seen = {}

    class Model:
        def invoke(self, messages):
            seen["human"] = messages[1][1]
            return AIMessage(content=reply())

    def sent_snapshot():
        return json.loads(seen["human"].split("\n", 1)[1])

    cfg = LLMConfig(structured_output="prompt")
    decider = LLMDecider(cfg, "4h", model=Model(), context={"followed_traders": lambda s: {"long": 3, "short": 0}})
    decider.decide("BTC/USDT", DF, None)
    assert sent_snapshot()["followed_traders"] == {"long": 3, "short": 0}

    def broken(symbol):
        raise OSError("feed down")

    decider = LLMDecider(cfg, "4h", model=Model(), context={"followed_traders": broken})
    assert decider.decide("BTC/USDT", DF, None).action == "hold"
    assert sent_snapshot()["followed_traders"] is None
