from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    exchange: str = "binance"
    db_path: str = "data/market.sqlite"
    symbols: tuple[str, ...] = ("BTC/USDT",)
    timeframes: tuple[str, ...] = ("4h", "1d")
    since: str = "2021-01-01"


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    risk_per_trade: float = 0.01
    max_position_pct: float = 1.0


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.30
    max_open_positions: int = 3
    max_total_exposure: float = 0.90
    max_daily_loss: float = 0.03
    min_order_value: float = 10.0


@dataclass(frozen=True)
class PaperConfig:
    broker: str = "paper"
    db_path: str = "data/paper.sqlite"
    initial_cash: float = 10_000.0
    decider: str = "strategy"       # "strategy" = rule strategy below, "llm" = the [llm] agent
    strategy: str = "ema_cross"
    params: dict = field(default_factory=dict)
    timeframe: str = "4h"
    symbols: tuple[str, ...] = ()   # empty = use [data].symbols
    poll_seconds: int = 60
    fee_rate: float = 0.001
    slippage: float = 0.0005


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "anthropic"            # any LangChain provider: openai, google_genai, ollama, ...
    model: str = "claude-opus-5-5"
    structured_output: str = "json_schema"  # LangChain method name, or "prompt" to parse plain JSON
    min_confidence: float = 0.6
    options: dict = field(default_factory=dict)   # passed straight to the chat model


@dataclass(frozen=True)
class AlertsConfig:
    telegram: bool = False
    events: tuple[str, ...] = ("BUY", "SELL", "STOP", "REJECTED", "ERROR")
    error_cooldown_minutes: int = 60
    daily_summary: bool = True
    # The LLM agent's news briefing (a NEWS event): what it read and where it thinks price goes.
    news_every_hours: float = 4.0      # sent this often per symbol; 0 = never
    news_min_hours: float = 1.0        # sooner than that only when its view changes, and not more often than this


@dataclass(frozen=True)
class CopyTradeConfig:
    enabled: bool = False                 # feed followed traders' positioning to the LLM agent
    db_path: str = "data/copytrade.sqlite"
    followed: int = 10                    # how many traders to follow
    candidates: int = 60                  # how many to examine in detail when ranking
    min_account_value: float = 100_000.0
    max_monthly_turnover: float = 50.0    # monthly volume / account value; above = market maker
    min_history_days: int = 90
    max_drawdown: float = 0.5
    refresh_minutes: int = 30             # re-read positions when older than this
    max_age_minutes: int = 240            # ignore the signal when older than this


@dataclass(frozen=True)
class NewsConfig:
    enabled: bool = False              # give the LLM agent recent headlines and the next earnings date
    max_headlines: int = 8
    max_age_hours: int = 72
    cache_minutes: int = 30
    earnings_blackout_days: int = 2    # no new entry this close to earnings; 0 = no rule
    # Wider news, searched on Google News. Each topic is a search query, e.g. "gold price".
    topics: tuple[str, ...] = ()
    topic_headlines: int = 4           # headlines kept per topic
    company_news: bool = False         # stocks: also search news about the company by its name
    # Scheduled economic releases (inflation, jobs, central banks) for these currencies, e.g. ["USD"].
    calendar: tuple[str, ...] = ()
    calendar_hours_ahead: int = 24
    event_blackout_minutes: int = 0    # leveraged accounts: no new entry this close to a high-impact release


@dataclass(frozen=True)
class MT5Config:
    """Trading through a MetaTrader 5 demo account ([paper].broker = "mt5")."""
    login: int = 0                     # if set, refuse to trade on any other account number
    magic: int = 461113                # tag on the agent's orders, to tell them from manual ones
    deviation_points: int = 30         # largest accepted slippage on a market order
    max_lots: float = 1.0              # hard cap on one position
    max_margin_pct: float = 0.5        # a new position may use at most this share of the free margin
    max_spread_to_stop: float = 0.15   # skip an entry when the spread exceeds this share of the stop distance
    min_stop_atr: float = 1.0          # the agent's stop, in ATR multiples, is kept inside this range
    max_stop_atr: float = 4.0
    min_take_profit_r: float = 1.0     # the agent's take-profit, in multiples of the stop distance
    max_take_profit_r: float = 4.0
    lookback_bars: int = 6000          # candles the agent's indicators are computed on
    no_entry_hours_before_weekend: float = 2.0    # no new position this close to the weekend close
    close_before_weekend_minutes: int = 30        # close open positions this long before it; 0 = keep them


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    copytrade: CopyTradeConfig = field(default_factory=CopyTradeConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    source_path: str = ""              # the file this config was loaded from


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines into the environment; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    load_dotenv(path.parent / ".env")
    if not path.exists():
        return Config()
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    data = dict(raw.get("data", {}))
    for key in ("symbols", "timeframes"):
        if key in data:
            data[key] = tuple(data[key])
    paper = dict(raw.get("paper", {}))
    if "symbols" in paper:
        paper["symbols"] = tuple(paper["symbols"])
    alerts = dict(raw.get("alerts", {}))
    if "events" in alerts:
        alerts["events"] = tuple(alerts["events"])

    # A watchlist chosen in the app replaces the symbols written in the file.
    from trading_agent.watchlist import read_watchlist

    chosen = read_watchlist(path)
    if chosen is not None:
        data["symbols"] = tuple(chosen["symbols"])
        paper["symbols"] = tuple(chosen["symbols"])
    return Config(
        source_path=str(path),
        alerts=AlertsConfig(**alerts),
        copytrade=CopyTradeConfig(**raw.get("copytrade", {})),
        news=NewsConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.get("news", {}).items()}),
        mt5=MT5Config(**raw.get("mt5", {})),
        data=DataConfig(**data),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        paper=PaperConfig(**paper),
        llm=LLMConfig(**raw.get("llm", {})),
    )
