"""Turn candles into the compact, numeric market picture the LLM reasons over.

The model never does arithmetic on raw charts: everything here is computed in
code, from closed candles only.
"""

from __future__ import annotations

import math

import pandas as pd

from trading_agent import indicators as ind
from trading_agent.data.store import timeframe_ms
from trading_agent.journal import Position
from trading_agent.strategies import REGISTRY

RECENT_BARS = 12


def _num(value) -> float | None:
    """Round to 5 significant digits; NaN/inf -> None."""
    if value is None or not math.isfinite(value):
        return None
    return float(f"{float(value):.5g}")


def _pct(value, base) -> float | None:
    if value is None or base is None or not math.isfinite(value) or not math.isfinite(base) or base == 0:
        return None
    return round((value / base - 1) * 100, 2)


def _last(series: pd.Series, back: int = 0) -> float:
    return float(series.iloc[-1 - back]) if len(series) > back else float("nan")


def _daily_context(df: pd.DataFrame) -> dict:
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    close = daily["close"]
    price = _last(close)
    ema50, sma200 = _last(ind.ema(close, 50)), _last(ind.sma(close, 200))
    return {
        "days_of_history": len(daily),
        "price_vs_ema50_pct": _pct(price, ema50),
        "price_vs_sma200_pct": _pct(price, sma200),
        "rsi14": _num(_last(ind.rsi(close, 14))),
        "return_7d_pct": _pct(price, _last(close, 7)),
        "return_30d_pct": _pct(price, _last(close, 30)),
        "return_90d_pct": _pct(price, _last(close, 90)),
    }


def build_snapshot(
    symbol: str, timeframe: str, df: pd.DataFrame, position: Position | None
) -> tuple[dict, float]:
    """Returns (snapshot, ATR). ATR is NaN when there is too little history."""
    close = df["close"]
    price = _last(close)
    ema20, ema50, ema200 = (_last(ind.ema(close, n)) for n in (20, 50, 200))
    macd = ind.macd(close)
    atr = _last(ind.atr(df, 14))
    bands = ind.bollinger(close).iloc[-1]
    band_width = bands["upper"] - bands["lower"]
    high20 = float(df["high"].iloc[-21:-1].max()) if len(df) > 20 else float("nan")
    low20 = float(df["low"].iloc[-21:-1].min()) if len(df) > 20 else float("nan")
    avg_volume = _last(df["volume"].rolling(20).mean())

    baseline = {}
    for name, cls in REGISTRY.items():
        last = cls().generate(df).iloc[-1]
        baseline[name] = "exit" if last["exit"] else "entry" if last["entry"] else "none"

    recent = df.tail(RECENT_BARS)
    snapshot = {
        "symbol": symbol,
        "timeframe": timeframe,
        "last_closed_candle_utc": df.index[-1].strftime("%Y-%m-%d %H:%M"),
        "price": _num(price),
        "trend": {
            "price_vs_ema20_pct": _pct(price, ema20),
            "price_vs_ema50_pct": _pct(price, ema50),
            "price_vs_ema200_pct": _pct(price, ema200),
            "ema20_vs_ema50_pct": _pct(ema20, ema50),
            "ema50_vs_ema200_pct": _pct(ema50, ema200),
        },
        "momentum": {
            "rsi14": _num(_last(ind.rsi(close, 14))),
            "macd_hist_last_4": [_num(v) for v in macd["hist"].tail(4)],
            "return_1_bar_pct": _pct(price, _last(close, 1)),
            "return_6_bars_pct": _pct(price, _last(close, 6)),
            "return_30_bars_pct": _pct(price, _last(close, 30)),
        },
        "volatility": {
            "atr14": _num(atr),
            "atr14_pct_of_price": _num(atr / price * 100),
            "bollinger_percent_b": _num((price - bands["lower"]) / band_width) if band_width else None,
        },
        "range": {
            "prior_20_bar_high": _num(high20),
            "prior_20_bar_low": _num(low20),
            "price_vs_20_bar_high_pct": _pct(price, high20),
            "price_vs_20_bar_low_pct": _pct(price, low20),
        },
        "volume_vs_20_bar_avg": _num(_last(df["volume"]) / avg_volume) if avg_volume else None,
        "daily_context": _daily_context(df),
        "rule_strategy_signals_this_bar": baseline,
        "recent_candles": [
            {
                "t": t.strftime("%m-%d %H:%M"),
                "o": _num(r.open), "h": _num(r.high), "l": _num(r.low), "c": _num(r.close),
                "v": _num(r.volume),
            }
            for t, r in zip(recent.index, recent.itertuples())
        ],
        "open_position": None,
    }

    if position is not None:
        last_ms = int(df.index[-1].timestamp() * 1000)
        snapshot["open_position"] = {
            "entry_price": _num(position.entry_price),
            "unrealized_pct": _pct(price, position.entry_price),
            "stop_loss": _num(position.stop),
            "stop_distance_pct": _pct(position.stop, price) if position.stop else None,
            "bars_held": max(0, (last_ms - position.entry_time) // timeframe_ms(timeframe)),
        }
    return snapshot, atr


def _higher_timeframe(df: pd.DataFrame, rule: str) -> dict:
    bars = df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    bars = bars.iloc[:-1] if len(bars) > 1 else bars      # the last bucket is usually still forming
    close = bars["close"]
    price = _last(df["close"])
    ema20, ema50, ema200 = (_last(ind.ema(close, n)) if len(close) >= n else float("nan") for n in (20, 50, 200))
    return {
        "closed_bars": len(bars),
        "price_vs_ema20_pct": _pct(price, ema20),
        "price_vs_ema50_pct": _pct(price, ema50),
        "price_vs_ema200_pct": _pct(price, ema200),
        "ema20_vs_ema50_pct": _pct(ema20, ema50),
        "rsi14": _num(_last(ind.rsi(close, 14))),
        "atr14": _num(_last(ind.atr(bars, 14))),
        "return_last_bar_pct": _pct(_last(close), _last(close, 1)),
        "return_5_bars_pct": _pct(_last(close), _last(close, 5)),
    }


def _today(df: pd.DataFrame) -> dict:
    """Where price sits in the current and previous UTC day's range."""
    days = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    today, previous = days.iloc[-1], days.iloc[-2] if len(days) > 1 else None
    price = _last(df["close"])
    return {
        "open": _num(today["open"]), "high": _num(today["high"]), "low": _num(today["low"]),
        "change_since_open_pct": _pct(price, today["open"]),
        "previous_day_high": _num(previous["high"]) if previous is not None else None,
        "previous_day_low": _num(previous["low"]) if previous is not None else None,
        "previous_day_close": _num(previous["close"]) if previous is not None else None,
    }


def build_intraday_snapshot(
    symbol: str, timeframe: str, df: pd.DataFrame, position: Position | None
) -> tuple[dict, float]:
    """Snapshot for short-term, long-and-short trading: adds higher timeframes and the day's range."""
    snapshot, atr = build_snapshot(symbol, timeframe, df, None)
    del snapshot["daily_context"]
    snapshot.pop("open_position")
    snapshot["weekday_utc"] = df.index[-1].strftime("%A")
    snapshot["higher_timeframes"] = {"1h": _higher_timeframe(df, "1h"), "4h": _higher_timeframe(df, "4h"),
                                     "1d": _higher_timeframe(df, "1D")}
    snapshot["today_utc"] = _today(df)
    snapshot["open_position"] = None
    if position is not None:
        price = snapshot["price"]
        direction = -1 if position.qty < 0 else 1
        last_ms = int(df.index[-1].timestamp() * 1000)
        risk = abs(position.entry_price - position.stop) if position.stop else None
        snapshot["open_position"] = {
            "side": position.side,
            "entry_price": _num(position.entry_price),
            "unrealized_pct": round(direction * (price / position.entry_price - 1) * 100, 3),
            "unrealized_in_r": round(direction * (price - position.entry_price) / risk, 2) if risk else None,
            "stop_loss": _num(position.stop),
            "take_profit": _num(position.take_profit),
            "bars_held": max(0, (last_ms - position.entry_time) // timeframe_ms(timeframe)),
        }
    return snapshot, atr
