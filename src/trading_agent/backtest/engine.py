"""Bar-by-bar, long-only, single-symbol backtester.

Timing rules (these are what keep the results honest):
  * A signal computed on bar i's close is filled at bar i+1's open.
  * Every fill pays the fee and is moved against us by the slippage.
  * A stop-loss is checked against each bar's low; if the bar opens below the
    stop (a gap), the fill is the open, not the stop price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_agent.config import BacktestConfig


@dataclass
class BacktestResult:
    equity: pd.Series        # account value at each bar's close
    in_position: pd.Series   # bool, whether a position was held at the close
    trades: pd.DataFrame     # one row per closed trade


TRADE_COLUMNS = [
    "entry_time", "exit_time", "entry_price", "exit_price",
    "qty", "pnl", "pnl_pct", "bars_held", "exit_reason",
]


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    if not df.index.equals(signals.index):
        raise ValueError("signals must be aligned with the candles")

    opens = df["open"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    entries = signals["entry"].to_numpy(bool)
    exits = signals["exit"].to_numpy(bool)
    stop_dists = signals["stop_distance"].to_numpy(float)
    times = df.index
    n = len(df)

    cash = cfg.initial_cash
    qty = 0.0
    stop = np.nan
    entry_cost = 0.0
    entry_price = 0.0
    entry_bar = -1

    pending: str | None = None
    pending_stop_dist = np.nan

    equity = np.empty(n)
    held = np.zeros(n, dtype=bool)
    trades: list[dict] = []

    def close_position(i: int, raw_price: float, reason: str) -> None:
        nonlocal cash, qty, stop
        price = raw_price * (1 - cfg.slippage)
        proceeds = qty * price * (1 - cfg.fee_rate)
        cash += proceeds
        trades.append(
            {
                "entry_time": times[entry_bar],
                "exit_time": times[i],
                "entry_price": entry_price,
                "exit_price": price,
                "qty": qty,
                "pnl": proceeds - entry_cost,
                "pnl_pct": proceeds / entry_cost - 1,
                "bars_held": i - entry_bar,
                "exit_reason": reason,
            }
        )
        qty = 0.0
        stop = np.nan

    for i in range(n):
        # 1. Fill the order decided on the previous bar's close.
        if pending == "sell" and qty > 0:
            close_position(i, opens[i], "signal")
        elif pending == "buy" and qty == 0:
            price = opens[i] * (1 + cfg.slippage)
            unit_cost = price * (1 + cfg.fee_rate)
            has_stop = np.isfinite(pending_stop_dist) and pending_stop_dist > 0
            max_qty = cash * cfg.max_position_pct / unit_cost
            risk_qty = cash * cfg.risk_per_trade / pending_stop_dist if has_stop else np.inf
            size = min(max_qty, risk_qty)
            if size > 0:
                qty = size
                entry_cost = size * unit_cost
                cash -= entry_cost
                entry_price = price
                entry_bar = i
                stop = price - pending_stop_dist if has_stop else np.nan
        pending = None

        # 2. Stop-loss.
        if qty > 0 and np.isfinite(stop) and lows[i] <= stop:
            close_position(i, min(opens[i], stop), "stop")

        # 3. Mark to market.
        equity[i] = cash + qty * closes[i]
        held[i] = qty > 0

        # 4. Decide on this bar's close; fills on the next bar.
        if qty > 0 and exits[i]:
            pending = "sell"
        elif qty == 0 and entries[i] and not exits[i]:
            pending = "buy"
            pending_stop_dist = stop_dists[i]

    # Close whatever is still open so the last trade is counted.
    if qty > 0 and n > 0:
        close_position(n - 1, closes[-1], "end")
        equity[-1] = cash

    return BacktestResult(
        equity=pd.Series(equity, index=times, name="equity"),
        in_position=pd.Series(held, index=times, name="in_position"),
        trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
    )
