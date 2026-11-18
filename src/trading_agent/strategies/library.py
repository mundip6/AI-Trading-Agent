"""Baseline swing strategies. They are benchmarks to beat, not proven edges."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_agent import indicators as ind
from trading_agent.strategies.base import Strategy, crossed_above, crossed_below, signals_frame


@dataclass
class EmaCross(Strategy):
    name = "ema_cross"
    description = "Trend following: long while the fast EMA is above the slow EMA"

    fast: int = 20
    slow: int = 50
    atr_period: int = 14
    atr_stop: float = 2.5

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = ind.ema(df["close"], self.fast)
        slow = ind.ema(df["close"], self.slow)
        return signals_frame(
            df.index,
            entry=crossed_above(fast, slow),
            exit=crossed_below(fast, slow),
            stop_distance=ind.atr(df, self.atr_period) * self.atr_stop,
        )


@dataclass
class DonchianBreakout(Strategy):
    name = "donchian_breakout"
    description = "Breakout: buy a new N-bar high, sell a new M-bar low"

    entry_period: int = 20
    exit_period: int = 10
    atr_period: int = 14
    atr_stop: float = 2.0

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # shift(1): compare the close against the channel of the *previous* bars
        upper = df["high"].rolling(self.entry_period).max().shift(1)
        lower = df["low"].rolling(self.exit_period).min().shift(1)
        return signals_frame(
            df.index,
            entry=df["close"] > upper,
            exit=df["close"] < lower,
            stop_distance=ind.atr(df, self.atr_period) * self.atr_stop,
        )


@dataclass
class RsiDip(Strategy):
    name = "rsi_dip"
    description = "Mean reversion: buy oversold dips inside a long-term uptrend"

    rsi_period: int = 14
    oversold: float = 30.0
    exit_level: float = 55.0
    trend_period: int = 200
    atr_period: int = 14
    atr_stop: float = 3.0

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = ind.rsi(df["close"], self.rsi_period)
        trend = ind.sma(df["close"], self.trend_period)
        return signals_frame(
            df.index,
            entry=(rsi < self.oversold) & (df["close"] > trend),
            exit=rsi > self.exit_level,
            stop_distance=ind.atr(df, self.atr_period) * self.atr_stop,
        )
