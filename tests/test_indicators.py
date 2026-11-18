import numpy as np
import pandas as pd
import pytest

from trading_agent import indicators as ind


def candles(close: list[float]) -> pd.DataFrame:
    close = pd.Series(close, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0}
    )


def test_sma_values_and_warmup():
    out = ind.sma(pd.Series([1.0, 2, 3, 4, 5]), 3)
    assert out.isna().tolist() == [True, True, False, False, False]
    assert out.iloc[-1] == pytest.approx(4.0)


def test_ema_of_constant_is_constant():
    out = ind.ema(pd.Series([7.0] * 30), 10)
    assert out.iloc[:9].isna().all()
    assert out.iloc[9:].eq(7.0).all()


def test_rsi_bounds_and_extremes():
    rng = np.random.default_rng(0)
    close = pd.Series(100 + rng.normal(0, 1, 300).cumsum())
    out = ind.rsi(close, 14).dropna()
    assert out.between(0, 100).all()
    assert ind.rsi(pd.Series(np.arange(1.0, 40)), 14).iloc[-1] == pytest.approx(100)
    assert ind.rsi(pd.Series(np.arange(40.0, 1, -1)), 14).iloc[-1] == pytest.approx(0)


def test_atr_constant_range():
    df = candles([100.0] * 40)
    assert ind.atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_indicators_are_causal():
    """Changing future bars must not change past indicator values."""
    rng = np.random.default_rng(1)
    df = candles(list(100 + rng.normal(0, 1, 200).cumsum()))
    cut = df.iloc[:150]
    for full, part in [
        (ind.rsi(df["close"]), ind.rsi(cut["close"])),
        (ind.atr(df), ind.atr(cut)),
        (ind.macd(df["close"])["hist"], ind.macd(cut["close"])["hist"]),
        (ind.bollinger(df["close"])["upper"], ind.bollinger(cut["close"])["upper"]),
    ]:
        pd.testing.assert_series_equal(full.iloc[:150], part)
