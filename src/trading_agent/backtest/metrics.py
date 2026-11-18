from __future__ import annotations

import numpy as np
import pandas as pd

from trading_agent.backtest.engine import BacktestResult
from trading_agent.config import BacktestConfig

YEAR = pd.Timedelta(days=365)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough loss, as a negative fraction."""
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1).min())


def _curve_stats(equity: pd.Series, initial: float) -> dict[str, float]:
    if len(equity) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    total = equity.iloc[-1] / initial - 1
    years = (equity.index[-1] - equity.index[0]) / YEAR
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")

    returns = equity.pct_change().dropna()
    # Counted, not assumed: about 2190 for 4h crypto candles, about 252 for daily stock candles.
    bars_per_year = len(returns) / years if years > 0 else 0.0
    std = returns.std()
    sharpe = returns.mean() / std * np.sqrt(bars_per_year) if std > 0 else 0.0

    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "max_drawdown": max_drawdown(equity),
        "sharpe": float(sharpe),
    }


def buy_and_hold(df: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    """Equity curve of buying at the first open and never selling (one fee paid)."""
    qty = cfg.initial_cash / (df["open"].iloc[0] * (1 + cfg.slippage) * (1 + cfg.fee_rate))
    return (df["close"] * qty).rename("buy_hold")


def compute_metrics(result: BacktestResult, df: pd.DataFrame, cfg: BacktestConfig) -> dict[str, float]:
    stats = _curve_stats(result.equity, cfg.initial_cash)
    hold = _curve_stats(buy_and_hold(df, cfg), cfg.initial_cash)

    trades = result.trades
    wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = -trades.loc[trades["pnl"] <= 0, "pnl"].sum()

    return {
        **stats,
        "trades": len(trades),
        "win_rate": float((trades["pnl"] > 0).mean()) if len(trades) else 0.0,
        "profit_factor": float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0,
        "avg_trade": float(trades["pnl_pct"].mean()) if len(trades) else 0.0,
        "exposure": float(result.in_position.mean()) if len(result.in_position) else 0.0,
        "hold_return": hold["total_return"],
        "hold_max_drawdown": hold["max_drawdown"],
        "hold_sharpe": hold["sharpe"],
    }
