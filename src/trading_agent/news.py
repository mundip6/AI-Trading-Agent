"""What the agent reads besides prices: headlines, wider news topics, the economic calendar.

- Headlines and the next earnings date for the symbol itself: Yahoo Finance.
- Wider topics (wars, central banks, a company by name): Google News search.
- Scheduled economic releases: the Forex Factory weekly calendar.

Headlines are third-party text. They are cleaned, shortened and passed to the
LLM as data inside the snapshot; the agent's prompt tells it never to treat
them as instructions, and its output stays limited to buy / sell / hold.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from trading_agent.config import NewsConfig

SUMMARY_CHARS = 280
TITLE_CHARS = 200
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CALENDAR_CACHE_HOURS = 4          # the file changes rarely and the site limits how often it may be read
CALENDAR_HOURS_BACK = 3           # releases this recent are still shown, as just out
SEARCH_URL = "https://news.google.com/rss/search?"
_COMPANY_SUFFIX = re.compile(
    r"[\s,]+(inc|incorporated|corp|corporation|company|co|ltd|limited|plc|holdings|group|n\.v|s\.a|se|ag)\.?$", re.I)
_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


_BROKER_SYMBOLS = {"XAUUSD": "GC=F", "XAGUSD": "SI=F", "XTIUSD": "CL=F", "USOIL": "CL=F", "XBRUSD": "BZ=F",
                   "UKOIL": "BZ=F", "US500": "^GSPC", "SP500": "^GSPC", "NAS100": "^NDX", "US30": "^DJI",
                   "GER40": "^GDAXI", "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD"}
_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}


def yahoo_symbol(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC-USD'; broker names like 'XAUUSD.s' -> 'GC=F'; stock tickers are unchanged."""
    if "/" in symbol:
        return symbol.split("/")[0] + "-USD"
    base = symbol.split(".")[0].upper()
    if base in _BROKER_SYMBOLS:
        return _BROKER_SYMBOLS[base]
    if len(base) == 6 and base[:3] in _CURRENCIES and base[3:] in _CURRENCIES:
        return base + "=X"
    return symbol


def _clean(text, limit: int) -> str:
    text = _SPACE.sub(" ", _TAGS.sub(" ", str(text or ""))).strip()
    text = "".join(ch for ch in text if ch.isprintable())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (news-reader)"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read()


def search_query(company_name: str) -> str:
    """'NVIDIA Corporation' -> '"NVIDIA"': the name as people write it in headlines."""
    name = company_name.strip()
    for _ in range(3):
        name = _COMPANY_SUFFIX.sub("", name).strip(" ,")
    return f'"{name}"'


class NewsFeed:
    def __init__(self, cfg: NewsConfig, yfinance=None, clock=time.time, fetch=http_get,
                 cache_dir: str | Path = "data/cache"):
        if yfinance is None:
            try:
                import yfinance
            except ImportError as exc:
                raise RuntimeError('News needs extra packages: pip install -e ".[stocks]"') from exc
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        self.cfg = cfg
        self._yf = yfinance
        self._clock = clock
        self._fetch = fetch
        self._cache_dir = Path(cache_dir)
        self._cache: dict[str, tuple[float, dict | None]] = {}
        self._searches: dict[str, tuple[float, list[dict]]] = {}
        self._names: dict[str, str | None] = {}
        self._events: tuple[float, list[dict]] | None = None

    def context(self, symbol: str) -> dict | None:
        """{'headlines': [...], 'next_earnings_in_days': int | None}, or None if nothing could be read."""
        now = self._clock()
        cached = self._cache.get(symbol)
        if cached and now - cached[0] < self.cfg.cache_minutes * 60:
            return cached[1]

        ticker = self._yf.Ticker(yahoo_symbol(symbol))
        headlines = earnings = None
        try:
            headlines = self._headlines(ticker.news or [], now)
        except Exception:
            pass
        try:
            earnings = self._days_to_earnings(ticker.calendar or {}, now)
            calendar_ok = True
        except Exception:
            calendar_ok = False

        result = None
        if headlines is not None or calendar_ok:
            result = {"headlines": headlines or [], "next_earnings_in_days": earnings}

        # Wider news. Each part is optional: a failed download leaves its field out.
        extra = {}
        topics = {topic: self._search(topic, now) for topic in self.cfg.topics}
        if self.cfg.company_news:
            name = self._company_name(symbol, ticker)
            if name:
                topics[f"about {name}"] = self._search(search_query(name), now)
        topics = {topic: found for topic, found in topics.items() if found}
        if topics:
            extra["topics"] = topics
        if self.cfg.calendar:
            events = self._calendar(now)
            if events is not None:
                extra["economic_calendar"] = events
        if extra:
            result = {**(result or {"headlines": [], "next_earnings_in_days": None}), **extra}
        self._cache[symbol] = (now, result)
        return result

    # --- wider news --------------------------------------------------------
    def _company_name(self, symbol: str, ticker) -> str | None:
        if symbol not in self._names:
            name = None
            try:
                info = ticker.info or {}
                if info.get("quoteType") == "EQUITY":
                    name = info.get("shortName") or info.get("longName")
            except Exception:
                pass
            self._names[symbol] = name
        return self._names[symbol]

    def _search(self, query: str, now: float) -> list[dict]:
        """Recent headlines matching a search, most relevant first. [] when the search fails."""
        cached = self._searches.get(query)
        if cached and now - cached[0] < self.cfg.cache_minutes * 60:
            return cached[1]
        days = max(1, math.ceil(self.cfg.max_age_hours / 24))
        url = SEARCH_URL + urllib.parse.urlencode(
            {"q": f"{query} when:{days}d", "hl": "en-US", "gl": "US", "ceid": "US:en"})
        found, seen = [], set()
        try:
            for item in ElementTree.fromstring(self._fetch(url)).iterfind("./channel/item"):
                source = _clean(item.findtext("source"), 60)
                title = _clean(item.findtext("title"), TITLE_CHARS)
                if source and title.endswith(f" - {source}"):
                    title = title[: -len(source) - 3]
                hours = (now - parsedate_to_datetime(item.findtext("pubDate")).timestamp()) / 3600
                if not title or title.lower() in seen or hours < 0 or hours > self.cfg.max_age_hours:
                    continue
                seen.add(title.lower())
                found.append({"hours_ago": round(hours, 1), "source": source, "title": title})
                if len(found) >= self.cfg.topic_headlines:
                    break
        except Exception:
            found = cached[1] if cached else []
        self._searches[query] = (now, found)
        return found

    def _calendar(self, now: float) -> list[dict] | None:
        """Medium- and high-impact releases for the configured currencies, soonest first."""
        if self._events is None or now - self._events[0] > CALENDAR_CACHE_HOURS * 3600:
            self._events = (now, self._load_calendar(now))
        raw = self._events[1]
        if raw is None:
            return None
        events = []
        for event in raw:
            if event.get("country") not in self.cfg.calendar or event.get("impact") not in ("High", "Medium"):
                continue
            try:
                hours = (datetime.fromisoformat(event["date"]).timestamp() - now) / 3600
            except (KeyError, ValueError):
                continue
            if -CALENDAR_HOURS_BACK <= hours <= self.cfg.calendar_hours_ahead:
                events.append({
                    "in_hours": round(hours, 2), "currency": event["country"], "impact": event["impact"].lower(),
                    "event": _clean(event.get("title"), 80), "forecast": _clean(event.get("forecast"), 20),
                    "previous": _clean(event.get("previous"), 20),
                })
        return sorted(events, key=lambda e: e["in_hours"])

    def _load_calendar(self, now: float) -> list | None:
        """This week's calendar. Kept on disk so several accounts share one download."""
        path = self._cache_dir / "economic_calendar.json"
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = None
        if saved and now - saved["read_at"] < CALENDAR_CACHE_HOURS * 3600:
            return saved["events"]
        try:
            events = json.loads(self._fetch(CALENDAR_URL))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"read_at": now, "events": events}), encoding="utf-8")
            return events
        except Exception:
            return saved["events"] if saved else None

    def _headlines(self, items: list, now: float) -> list[dict]:
        out = []
        for item in items:
            content = item.get("content") or item
            published = content.get("pubDate") or content.get("displayTime")
            title = _clean(content.get("title"), TITLE_CHARS)
            if not published or not title:
                continue
            stamp = datetime.fromisoformat(str(published).replace("Z", "+00:00")).timestamp()
            hours = (now - stamp) / 3600
            if hours < 0 or hours > self.cfg.max_age_hours:
                continue
            out.append({
                "hours_ago": round(hours, 1),
                "source": _clean((content.get("provider") or {}).get("displayName"), 60),
                "title": title,
                "summary": _clean(content.get("summary"), SUMMARY_CHARS),
            })
        out.sort(key=lambda h: h["hours_ago"])
        return out[: self.cfg.max_headlines]

    @staticmethod
    def _days_to_earnings(calendar: dict, now: float) -> int | None:
        dates = calendar.get("Earnings Date") or []
        if isinstance(dates, (date, datetime)):
            dates = [dates]
        today = datetime.fromtimestamp(now, timezone.utc).date()
        ahead = []
        for d in dates:
            d = d.date() if isinstance(d, datetime) else d
            if isinstance(d, date) and d >= today:
                ahead.append((d - today).days)
        return min(ahead) if ahead else None
