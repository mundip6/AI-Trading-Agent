"""Read-only client for Hyperliquid's public data. No account or API key involved."""

from __future__ import annotations

import time

import requests

INFO_URL = "https://api.hyperliquid.xyz/info"
LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"


class HyperliquidClient:
    def __init__(self, session: requests.Session | None = None, pause: float = 1.0):
        self._session = session or requests.Session()
        self._pause = pause          # seconds between info calls, to stay inside the rate limit
        self._last_call = 0.0

    def _info(self, payload: dict):
        wait = self._last_call + self._pause - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        response = self._session.post(INFO_URL, json=payload, timeout=30)
        self._last_call = time.monotonic()
        response.raise_for_status()
        return response.json()

    def leaderboard(self) -> list[dict]:
        """Every ranked account (tens of thousands of rows, a large download)."""
        response = self._session.get(LEADERBOARD_URL, timeout=180)
        response.raise_for_status()
        return response.json()["leaderboardRows"]

    def portfolio(self, address: str) -> dict[str, dict]:
        """P&L and account-value history per window: day, week, month, allTime, perp*."""
        return dict(self._info({"type": "portfolio", "user": address}))

    def positions(self, address: str) -> tuple[float, list[dict]]:
        """(account value, open perpetual positions). `size` is signed: negative = short."""
        state = self._info({"type": "clearinghouseState", "user": address})
        positions = []
        for item in state.get("assetPositions", []):
            pos = item["position"]
            size = float(pos["szi"])
            if size == 0:
                continue
            positions.append({
                "coin": pos["coin"],
                "size": size,
                "entry_price": float(pos.get("entryPx") or 0),
                "value": float(pos.get("positionValue") or 0),
                "leverage": float((pos.get("leverage") or {}).get("value") or 0),
            })
        return float(state["marginSummary"]["accountValue"]), positions
