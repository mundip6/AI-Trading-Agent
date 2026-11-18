"""Account statistics shared by the dashboard and the alerts."""

from __future__ import annotations

import pandas as pd

from trading_agent.backtest.metrics import max_drawdown
from trading_agent.data.store import CandleStore, timeframe_ms


def account_summary(equity: pd.Series, trades: pd.DataFrame, initial: float | None = None) -> dict:
    if equity.empty:
        return {"equity": initial or 0.0, "initial": initial or 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                "trades": 0, "win_rate": 0.0, "realised_pnl": 0.0, "profit_factor": 0.0}
    initial = initial or float(equity.iloc[0])
    wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = -trades.loc[trades["pnl"] <= 0, "pnl"].sum()
    return {
        "equity": float(equity.iloc[-1]),
        "initial": initial,
        "total_return": float(equity.iloc[-1] / initial - 1),
        "max_drawdown": max_drawdown(equity),
        "trades": len(trades),
        "win_rate": float((trades["pnl"] > 0).mean()) if len(trades) else 0.0,
        "realised_pnl": float(trades["pnl"].sum()),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0,
    }


def indexed(series: pd.Series) -> pd.Series:
    """Rebase so the first value is 100."""
    return series / series.iloc[0] * 100 if len(series) else series


def benchmark_curve(
    store: CandleStore, exchange: str, symbols: tuple[str, ...], timeframe: str, times: pd.DatetimeIndex
) -> pd.Series:
    """Equal-weight buy-and-hold of `symbols`, sampled at `times`, first value = 100.

    Uses each candle's close at the moment it became known (its close time).
    """
    if len(times) == 0:
        return pd.Series(dtype=float)
    tf = pd.Timedelta(milliseconds=timeframe_ms(timeframe))
    curves = []
    for symbol in symbols:
        close = store.load(exchange, symbol, timeframe, start=times[0] - 3 * tf)["close"]
        if close.empty:
            continue
        close.index = close.index + tf
        sampled = close.reindex(close.index.union(times)).ffill().reindex(times).bfill()
        curves.append(indexed(sampled))
    if not curves:
        return pd.Series(dtype=float)
    return pd.concat(curves, axis=1).mean(axis=1).rename("benchmark")
