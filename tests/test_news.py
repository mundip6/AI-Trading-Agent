import json
from datetime import date, datetime, timezone

from langchain_core.messages import AIMessage

from test_agent import DF, reply
from trading_agent.agent import LLMDecider
from trading_agent.config import LLMConfig, NewsConfig
from trading_agent.news import NewsFeed, yahoo_symbol

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc).timestamp()
CFG = NewsConfig(enabled=True, max_headlines=2, max_age_hours=72, cache_minutes=30)


def item(title, hours_ago, summary="", source="Reuters", nested=True):
    stamp = datetime.fromtimestamp(NOW - hours_ago * 3600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content = {"title": title, "summary": summary, "pubDate": stamp, "provider": {"displayName": source}}
    return {"id": "x", "content": content} if nested else content


class FakeYF:
    def __init__(self, news=(), calendar=None, fail=()):
        self.news, self.calendar, self.fail, self.symbols = list(news), calendar or {}, fail, []

    def Ticker(self, symbol):
        self.symbols.append(symbol)
        outer = self

        class Ticker:
            @property
            def news(self):
                if "news" in outer.fail:
                    raise OSError("down")
                return outer.news

            @property
            def calendar(self):
                if "calendar" in outer.fail:
                    raise OSError("down")
                return outer.calendar

        return Ticker()


def feed(yf, cfg=CFG, clock=lambda: NOW):
    return NewsFeed(cfg, yfinance=yf, clock=clock)


def test_symbol_mapping():
    assert yahoo_symbol("BTC/USDT") == "BTC-USD"
    assert yahoo_symbol("VUAA.DE") == "VUAA.DE"


def test_headlines_are_filtered_sorted_capped_and_cleaned():
    yf = FakeYF(news=[
        item("Old story", 100),
        item("Second", 5, summary="<p>Shares   fell\n\n<a href='x'>sharply</a></p>" + "x" * 400),
        item("  First\u0007 ", 1, nested=False),
        item("Third", 30),
        item("", 2),
        item("From the future", -3),
    ])
    result = feed(yf).context("BTC/USDT")
    assert yf.symbols == ["BTC-USD"]
    assert [h["title"] for h in result["headlines"]] == ["First", "Second"]     # newest first, capped at 2
    second = result["headlines"][1]
    assert second["hours_ago"] == 5.0 and second["source"] == "Reuters"
    assert second["summary"].startswith("Shares fell sharply") and len(second["summary"]) <= 280
    assert "<" not in second["summary"]
    assert result["next_earnings_in_days"] is None


def test_days_to_earnings():
    assert feed(FakeYF(calendar={"Earnings Date": [date(2026, 11, 17)]})).context("NVDA")["next_earnings_in_days"] == 5
    assert feed(FakeYF(calendar={"Earnings Date": [date(2026, 11, 12)]})).context("NVDA")["next_earnings_in_days"] == 0
    past = {"Earnings Date": [date(2026, 8, 1), date(2026, 11, 20)]}
    assert feed(FakeYF(calendar=past)).context("NVDA")["next_earnings_in_days"] == 8
    assert feed(FakeYF(calendar={})).context("SPY")["next_earnings_in_days"] is None


def test_partial_and_total_failure():
    partial = feed(FakeYF(news=[item("A", 1)], fail=("calendar",))).context("NVDA")
    assert len(partial["headlines"]) == 1 and partial["next_earnings_in_days"] is None
    assert feed(FakeYF(fail=("news", "calendar"))).context("NVDA") is None


def test_results_are_cached():
    clock = {"now": NOW}
    yf = FakeYF(news=[item("A", 1)])
    news = feed(yf, clock=lambda: clock["now"])
    news.context("NVDA"); news.context("NVDA")
    assert len(yf.symbols) == 1
    clock["now"] += 31 * 60
    news.context("NVDA")
    assert len(yf.symbols) == 2


# --- into the agent -----------------------------------------------------
class Model:
    def __init__(self, text):
        self.text, self.sent = text, None

    def invoke(self, messages):
        self.sent = json.loads(messages[1][1].split("\n", 1)[1])
        return AIMessage(content=self.text)


def decider(model, news, blackout=2):
    return LLMDecider(LLMConfig(structured_output="prompt"), "1d", model=model,
                      context={"news": lambda s: news}, asset_class="stocks and ETFs",
                      earnings_blackout_days=blackout)


def test_news_reaches_the_model_and_snapshot_method_matches():
    model = Model(reply())
    news = {"headlines": [{"hours_ago": 1.0, "source": "R", "title": "T", "summary": ""}], "next_earnings_in_days": 9}
    agent = decider(model, news)
    agent.decide("NVDA", DF, None)
    assert model.sent["news"] == news and model.sent["asset_class"] == "stocks and ETFs"
    assert agent.snapshot("NVDA", DF, None)[0]["news"] == news


def test_earnings_blackout_blocks_buys_only():
    buy = reply(action="buy", confidence=0.9)
    soon = {"headlines": [], "next_earnings_in_days": 2}
    blocked = decider(Model(buy), soon).decide("NVDA", DF, None)
    assert blocked.action == "hold" and "earnings in 2 days" in blocked.reason

    assert decider(Model(buy), {"headlines": [], "next_earnings_in_days": 3}).decide("NVDA", DF, None).action == "buy"
    assert decider(Model(buy), soon, blackout=0).decide("NVDA", DF, None).action == "buy"
    assert decider(Model(buy), None).decide("NVDA", DF, None).action == "buy"
    assert decider(Model(buy), {"headlines": [], "next_earnings_in_days": None}).decide("SPY", DF, None).action == "buy"


# --- wider news: topic searches, company news, economic calendar ----------
import dataclasses  # noqa: E402

RSS = """<rss><channel>
<item><title>Iran claims drone attack on US base - Reuters</title><source>Reuters</source>
<pubDate>Sun, 15 Nov 2026 10:00:00 GMT</pubDate></item>
<item><title>Iran claims drone attack on US base - Reuters</title><source>Reuters</source>
<pubDate>Sun, 15 Nov 2026 10:05:00 GMT</pubDate></item>
<item><title>Old story - AP</title><source>AP</source><pubDate>Mon, 02 Nov 2026 10:00:00 GMT</pubDate></item>
<item><title>Ceasefire talks resume &lt;b&gt;today&lt;/b&gt; - BBC</title><source>BBC</source>
<pubDate>Sun, 15 Nov 2026 08:00:00 GMT</pubDate></item>
<item><title>Third fresh one - AP</title><source>AP</source><pubDate>Sun, 15 Nov 2026 07:00:00 GMT</pubDate></item>
</channel></rss>"""
WIDE_NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc).timestamp()
EVENTS = [
    {"title": "CPI m/m", "country": "USD", "date": "2026-11-15T08:30:00-05:00", "impact": "High",
     "forecast": "0.3%", "previous": "0.2%"},
    {"title": "German ZEW", "country": "EUR", "date": "2026-11-15T09:00:00-05:00", "impact": "High"},
    {"title": "Crude Oil Inventories", "country": "USD", "date": "2026-11-15T10:30:00-05:00", "impact": "Low"},
    {"title": "FOMC Minutes", "country": "USD", "date": "2026-11-18T14:00:00-05:00", "impact": "High"},
    {"title": "Retail Sales m/m", "country": "USD", "date": "2026-11-15T05:30:00-05:00", "impact": "Medium"},
]


class Web:
    def __init__(self, fail=False):
        self.urls, self.fail = [], fail

    def __call__(self, url):
        self.urls.append(url)
        if self.fail:
            raise OSError("offline")
        return json.dumps(EVENTS).encode() if "calendar" in url else RSS.encode()


class NamedYF(FakeYF):
    """Yahoo stand-in whose tickers also carry company details."""

    def __init__(self, info):
        super().__init__()
        self.info = info

    def Ticker(self, symbol):
        ticker = super().Ticker(symbol)
        ticker.info = self.info
        return ticker


def wide_feed(tmp_path, web, yf=None, **overrides):
    cfg = dataclasses.replace(CFG, max_age_hours=48, **overrides)
    return NewsFeed(cfg, yfinance=yf or FakeYF(), clock=lambda: WIDE_NOW, fetch=web, cache_dir=tmp_path)


def test_topic_search_gives_recent_distinct_headlines(tmp_path):
    web = Web()
    feed_ = wide_feed(tmp_path, web, topics=("war OR ceasefire",), topic_headlines=2)
    found = feed_.context("XAUUSD.s")["topics"]["war OR ceasefire"]
    assert found == [
        {"hours_ago": 2.0, "source": "Reuters", "title": "Iran claims drone attack on US base"},
        {"hours_ago": 4.0, "source": "BBC", "title": "Ceasefire talks resume today"},
    ]
    assert "war+OR+ceasefire+when%3A2d" in web.urls[0]
    feed_.context("XAUUSD.s")
    assert len(web.urls) == 1                                  # cached


def test_company_news_is_searched_by_name(tmp_path):
    from trading_agent.news import search_query

    assert search_query("NVIDIA Corporation") == '"NVIDIA"'
    assert search_query("Advanced Micro Devices, Inc.") == '"Advanced Micro Devices"'
    web = Web()
    yf = NamedYF({"quoteType": "EQUITY", "shortName": "NVIDIA Corporation"})
    news = wide_feed(tmp_path, web, yf=yf, company_news=True).context("NVDA")
    assert list(news["topics"]) == ["about NVIDIA Corporation"] and "%22NVIDIA%22" in web.urls[0]
    etf = NamedYF({"quoteType": "ETF", "shortName": "SPDR S&P 500"})
    assert "topics" not in (wide_feed(tmp_path, Web(), yf=etf, company_news=True).context("SPY") or {})


def test_economic_calendar_window_and_shared_cache(tmp_path):
    web = Web()
    events = wide_feed(tmp_path, web, calendar=("USD",)).context("XAUUSD.s")["economic_calendar"]
    # another currency, low impact and far-off releases are dropped
    assert [(e["event"], e["in_hours"], e["impact"]) for e in events] == [
        ("Retail Sales m/m", -1.5, "medium"), ("CPI m/m", 1.5, "high")]
    assert events[1]["forecast"] == "0.3%"
    second = Web()
    assert wide_feed(tmp_path, second, calendar=("USD",)).context("XAUUSD.s")["economic_calendar"] == events
    assert second.urls == []                                   # another account reuses the saved download


def test_wider_news_failures_leave_the_basics_intact(tmp_path):
    yf = FakeYF(news=[item("A", 1)])
    cfg = dataclasses.replace(CFG, topics=("war",), calendar=("USD",), company_news=True)
    news = NewsFeed(cfg, yfinance=yf, clock=lambda: NOW, fetch=Web(fail=True), cache_dir=tmp_path).context("NVDA")
    assert [h["title"] for h in news["headlines"]] == ["A"]
    assert "topics" not in news and "economic_calendar" not in news
