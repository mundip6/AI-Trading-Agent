"""Choose which traders are worth following.

The leaderboard rewards whoever made the most money recently, which mostly finds
luck and leverage. Ranking here favours traders who have been profitable for a
long time, steadily, without deep drawdowns, and who trade slowly enough that
their positions mean something to a swing trader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from trading_agent.config import CopyTradeConfig

DAY_MS = 86_400_000
PROFIT_TO_PAIN_CAP = 50.0


@dataclass
class TraderStats:
    address: str
    name: str
    account_value: float
    month_roi: float
    alltime_pnl: float
    consistency: float = 0.0       # share of profitable days in the last month
    max_drawdown: float = 1.0      # largest P&L give-back as a fraction of peak capital
    profit_to_pain: float = 0.0    # all-time profit divided by the largest give-back
    history_days: float = 0.0
    score: float = 0.0


def _windows(row: dict) -> dict[str, dict[str, float]]:
    return {name: {k: float(v) for k, v in perf.items()} for name, perf in row["windowPerformances"]}


def prefilter(rows: list[dict], cfg: CopyTradeConfig) -> list[TraderStats]:
    """Cheap screen on leaderboard data alone; returns the candidates worth a closer look."""
    out = []
    for row in rows:
        value = float(row["accountValue"])
        perf = _windows(row)
        month, alltime = perf.get("month", {}), perf.get("allTime", {})
        if value < cfg.min_account_value:
            continue
        if alltime.get("pnl", 0) <= 0 or month.get("pnl", 0) <= 0 or month.get("vlm", 0) <= 0:
            continue
        if month["vlm"] / value > cfg.max_monthly_turnover:   # market makers and high-frequency accounts
            continue
        out.append(TraderStats(
            address=row["ethAddress"], name=row.get("displayName") or "", account_value=value,
            month_roi=month.get("roi", 0.0), alltime_pnl=alltime["pnl"],
        ))
    out.sort(key=lambda t: t.alltime_pnl, reverse=True)
    return out[: cfg.candidates]


def _series(history: list) -> pd.Series:
    return pd.Series([float(v) for _, v in history], index=[int(t) for t, _ in history], dtype=float)


def add_history_metrics(trader: TraderStats, portfolio: dict[str, dict]) -> TraderStats:
    """Fill in the metrics that need the trader's P&L history."""
    alltime = portfolio.get("perpAllTime") or portfolio.get("allTime") or {}
    pnl = _series(alltime.get("pnlHistory", []))
    value = _series(alltime.get("accountValueHistory", []))
    if len(pnl) >= 2:
        trader.history_days = (pnl.index[-1] - pnl.index[0]) / DAY_MS
        give_back = float((pnl.cummax() - pnl).max())
        capital = float(value.max()) if len(value) else 0.0
        trader.max_drawdown = give_back / capital if capital > 0 else 1.0
        final = float(pnl.iloc[-1])
        trader.profit_to_pain = min(final / give_back, PROFIT_TO_PAIN_CAP) if give_back > 0 else PROFIT_TO_PAIN_CAP
        if final <= 0:
            trader.profit_to_pain = 0.0

    month = portfolio.get("perpMonth") or portfolio.get("month") or {}
    daily = _series(month.get("pnlHistory", [])).diff().dropna()
    daily = daily[daily != 0]
    trader.consistency = float((daily > 0).mean()) if len(daily) >= 10 else 0.0
    return trader


def rank(traders: list[TraderStats], cfg: CopyTradeConfig) -> list[TraderStats]:
    """Drop traders that fail the hard limits, score the rest, best first.

    The score is the average percentile rank across four measures, so no single
    one (and no outlier) dominates.
    """
    kept = [
        t for t in traders
        if t.history_days >= cfg.min_history_days and t.max_drawdown <= cfg.max_drawdown and t.consistency > 0
    ]
    if not kept:
        return []
    frame = pd.DataFrame([asdict(t) for t in kept])
    ranks = pd.concat(
        [
            frame["month_roi"].rank(pct=True),
            frame["consistency"].rank(pct=True),
            frame["profit_to_pain"].rank(pct=True),
            (-frame["max_drawdown"]).rank(pct=True),
        ],
        axis=1,
    ).mean(axis=1)
    for trader, score in zip(kept, ranks):
        trader.score = round(float(score), 4)
    kept.sort(key=lambda t: t.score, reverse=True)
    return kept[: cfg.followed]
