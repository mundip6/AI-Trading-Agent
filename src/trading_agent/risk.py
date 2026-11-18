"""Hard risk rules. Nothing upstream (strategy or LLM) can override these."""

from __future__ import annotations

import math
from dataclasses import dataclass

from trading_agent.config import RiskConfig


@dataclass(frozen=True)
class Verdict:
    approved: bool
    qty: float
    reason: str


def _reject(reason: str) -> Verdict:
    return Verdict(False, 0.0, reason)


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def size_entry(
        self,
        *,
        equity: float,
        cash: float,
        exposure: float,          # current value of all open positions
        open_positions: int,
        price: float,
        stop_distance: float | None,
        day_start_equity: float,
    ) -> Verdict:
        """Decide whether a new long may be opened, and how large."""
        cfg = self.cfg

        if day_start_equity > 0 and equity / day_start_equity - 1 <= -cfg.max_daily_loss:
            return _reject(
                f"daily loss limit reached ({equity / day_start_equity - 1:+.1%}); no new entries today"
            )
        if open_positions >= cfg.max_open_positions:
            return _reject(f"already holding {open_positions} positions (max {cfg.max_open_positions})")
        if stop_distance is None or not math.isfinite(stop_distance) or stop_distance <= 0:
            return _reject("no valid stop-loss distance")
        if stop_distance >= price:
            return _reject("stop-loss distance is larger than the price")

        value = min(
            equity * cfg.risk_per_trade / stop_distance * price,   # lose risk_per_trade if stopped
            equity * cfg.max_position_pct,                          # single-position cap
            equity * cfg.max_total_exposure - exposure,             # portfolio cap
            cash * 0.995,                                           # leave room for fee and slippage
        )
        if value < cfg.min_order_value:
            return _reject(f"order value {max(value, 0):.2f} is below the minimum {cfg.min_order_value:.2f}")
        return Verdict(True, value / price, "ok")


    def size_margin_entry(
        self,
        *,
        equity: float,
        free_margin: float,
        open_positions: int,
        stop_distance: float | None,
        spread: float,
        loss_per_lot: float,          # account currency lost by 1.0 lot if the stop is hit
        margin_per_lot: float,
        volume_min: float,
        volume_step: float,
        volume_max: float,
        max_lots: float,
        max_margin_pct: float,
        max_spread_to_stop: float,
        min_stop_distance: float,
        day_start_equity: float,
    ) -> Verdict:
        """Leveraged account: may a position (long or short) be opened, and with how many lots."""
        cfg = self.cfg

        if day_start_equity > 0 and equity / day_start_equity - 1 <= -cfg.max_daily_loss:
            return _reject(
                f"daily loss limit reached ({equity / day_start_equity - 1:+.1%}); no new entries today"
            )
        if open_positions >= cfg.max_open_positions:
            return _reject(f"already holding {open_positions} positions (max {cfg.max_open_positions})")
        if stop_distance is None or not math.isfinite(stop_distance) or stop_distance <= 0:
            return _reject("no valid stop-loss distance")
        if stop_distance < min_stop_distance + spread:
            return _reject(f"stop distance {stop_distance:.6g} is closer than the broker allows")
        if spread > stop_distance * max_spread_to_stop:
            return _reject(f"spread {spread:.6g} is too wide for a {stop_distance:.6g} stop")
        if not (math.isfinite(loss_per_lot) and loss_per_lot > 0 and margin_per_lot > 0):
            return _reject("the broker returned no valid contract values")

        lots = min(
            equity * cfg.risk_per_trade / loss_per_lot,        # lose risk_per_trade if stopped
            free_margin * max_margin_pct / margin_per_lot,     # keep margin in reserve
            max_lots,
            volume_max,
        )
        lots = round(math.floor(lots / volume_step + 1e-9) * volume_step, 8) if volume_step > 0 else lots
        if lots < volume_min:
            return _reject(
                f"the risk budget allows {max(lots, 0):.2f} lots, below the broker minimum {volume_min:.2f}"
            )
        return Verdict(True, lots, "ok")
