"""Copy-trading as a signal: what are the traders we follow doing in this market?"""

from __future__ import annotations

import time
from typing import Callable

from trading_agent.config import CopyTradeConfig
from trading_agent.copytrade.hyperliquid import HyperliquidClient
from trading_agent.copytrade.ranking import TraderStats, add_history_metrics, prefilter, rank
from trading_agent.copytrade.store import CopyStore

DAY_MS = 86_400_000


def coin_of(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC' (Hyperliquid's name for the perpetual)."""
    return symbol.split("/")[0]


def _now_ms() -> int:
    return int(time.time() * 1000)


class CopyTrading:
    def __init__(self, cfg: CopyTradeConfig, store: CopyStore | None = None, client: HyperliquidClient | None = None):
        self.cfg = cfg
        self.store = store or CopyStore(cfg.db_path)
        self.client = client or HyperliquidClient()

    # --- maintenance -------------------------------------------------------
    def rerank(self, progress: Callable[[str], None] = lambda _: None, now_ms: int | None = None) -> list[TraderStats]:
        """Rebuild the list of followed traders from the public leaderboard."""
        progress("Downloading the leaderboard...")
        candidates = prefilter(self.client.leaderboard(), self.cfg)
        progress(f"{len(candidates)} candidates pass the first screen; fetching their histories...")
        for i, trader in enumerate(candidates, 1):
            try:
                add_history_metrics(trader, self.client.portfolio(trader.address))
            except Exception as exc:   # one unreachable account should not sink the run
                progress(f"  {trader.address[:10]}: skipped ({type(exc).__name__})")
            if i % 10 == 0:
                progress(f"  {i}/{len(candidates)}")
        followed = rank(candidates, self.cfg)
        self.store.replace_traders(followed, now_ms or _now_ms())
        return followed

    def update(self, now_ms: int | None = None) -> int:
        """Snapshot the open positions of every followed trader. Returns how many were read."""
        accounts = {}
        for trader in self.store.traders():
            try:
                accounts[trader.address] = self.client.positions(trader.address)
            except Exception:
                continue
        if accounts:
            self.store.save_snapshot(now_ms or _now_ms(), accounts)
        return len(accounts)

    # --- the signal --------------------------------------------------------
    def consensus(self, symbol: str, now_ms: int | None = None) -> dict | None:
        """Positioning of the followed traders in one market, from the latest snapshot."""
        now_ms = now_ms or _now_ms()
        latest = self.store.latest_snapshot_time()
        if latest is None:
            return None
        coin = coin_of(symbol)
        current = self.store.exposures(latest, coin)
        if current.empty:
            return None

        out = {
            "traders_followed": len(current),
            "long": int((current > 0).sum()),
            "short": int((current < 0).sum()),
            "no_position": int((current == 0).sum()),
            "avg_exposure_pct_of_account": round(float(current.mean()) * 100, 1),
            "data_age_minutes": int((now_ms - latest) / 60_000),
            "changes_last_24h": None,
        }

        previous_time = self.store.latest_snapshot_time(latest - DAY_MS) or self.store.first_snapshot_time()
        if previous_time is not None and previous_time < latest:
            previous = self.store.exposures(previous_time, coin)
            both = current.index.intersection(previous.index)
            now_side = current[both].apply(_side)
            then_side = previous[both].apply(_side)
            changes = {
                "hours_compared": round((latest - previous_time) / 3_600_000, 1),
                "opened_long": int(((then_side == 0) & (now_side > 0)).sum()),
                "opened_short": int(((then_side == 0) & (now_side < 0)).sum()),
                "closed_long": int(((then_side > 0) & (now_side == 0)).sum()),
                "closed_short": int(((then_side < 0) & (now_side == 0)).sum()),
                "flipped_to_long": int(((then_side < 0) & (now_side > 0)).sum()),
                "flipped_to_short": int(((then_side > 0) & (now_side < 0)).sum()),
            }
            out["changes_last_24h"] = changes
        return out

    def context(self, symbol: str, now_ms: int | None = None) -> dict | None:
        """Consensus for the LLM snapshot, refreshing the positions first if they are stale.

        Returns None (and the agent simply decides without it) when nobody is
        followed yet or the data cannot be refreshed.
        """
        now_ms = now_ms or _now_ms()
        if not self.store.traders():
            return None
        latest = self.store.latest_snapshot_time()
        if latest is None or now_ms - latest > self.cfg.refresh_minutes * 60_000:
            try:
                self.update(now_ms)
            except Exception:
                pass
        result = self.consensus(symbol, now_ms)
        if result is None or result["data_age_minutes"] > self.cfg.max_age_minutes:
            return None
        return result


def _side(exposure: float) -> int:
    return (exposure > 0) - (exposure < 0)
