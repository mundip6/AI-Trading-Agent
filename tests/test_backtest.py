import numpy as np
import pandas as pd
import pytest

from trading_agent.backtest import compute_metrics, run_backtest
from trading_agent.backtest.metrics import max_drawdown
from trading_agent.config import BacktestConfig
from trading_agent.strategies import REGISTRY, build_strategy

FREE = BacktestConfig(initial_cash=10_000, fee_rate=0, slippage=0, risk_per_trade=0.01, max_position_pct=1.0)


def make_df(opens, lows=None, closes=None) -> pd.DataFrame:
    opens = np.asarray(opens, dtype=float)
    closes = opens if closes is None else np.asarray(closes, dtype=float)
    lows = np.minimum(opens, closes) if lows is None else np.asarray(lows, dtype=float)
    index = pd.date_range("2024-01-01", periods=len(opens), freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": np.maximum(opens, closes), "low": lows, "close": closes, "volume": 1.0},
        index=index,
    )


def make_signals(df, entry=(), exit=(), stop_distance=np.nan) -> pd.DataFrame:
    sig = pd.DataFrame({"entry": False, "exit": False, "stop_distance": stop_distance}, index=df.index)
    sig.iloc[list(entry), 0] = True
    sig.iloc[list(exit), 1] = True
    return sig


def test_orders_fill_on_next_bar_open():
    df = make_df([100, 101, 102, 110, 111, 112, 120, 121])
    result = run_backtest(df, make_signals(df, entry=[2], exit=[5]), FREE)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == 110          # open of bar 3, not close of bar 2
    assert trade["exit_price"] == 120           # open of bar 6
    assert trade["exit_reason"] == "signal"
    assert result.equity.iloc[-1] == pytest.approx(10_000 * 120 / 110)


def test_stop_loss_sizes_position_to_risk():
    #            bar: 0    1    2    3    4
    df = make_df([100, 100, 100, 100, 100], lows=[100, 100, 100, 94, 100])
    result = run_backtest(df, make_signals(df, entry=[0], stop_distance=5.0), FREE)

    trade = result.trades.iloc[0]
    assert trade["qty"] == pytest.approx(20)    # 1% of 10k = 100 risk / 5 per unit
    assert trade["exit_price"] == 95
    assert trade["exit_reason"] == "stop"
    assert result.equity.iloc[-1] == pytest.approx(9_900)


def test_stop_gap_fills_at_open():
    df = make_df([100, 100, 90, 90], lows=[100, 100, 88, 90])
    result = run_backtest(df, make_signals(df, entry=[0], stop_distance=5.0), FREE)
    assert result.trades.iloc[0]["exit_price"] == 90


def test_fees_and_slippage_cost_money():
    df = make_df([100] * 6)
    cfg = BacktestConfig(initial_cash=10_000, fee_rate=0.001, slippage=0.0005)
    result = run_backtest(df, make_signals(df, entry=[0], exit=[2]), cfg)
    expected = 10_000 * (1 - 0.0005) * (1 - 0.001) / ((1 + 0.0005) * (1 + 0.001))
    assert result.equity.iloc[-1] == pytest.approx(expected)
    assert result.trades.iloc[0]["pnl"] < 0


def test_open_position_is_closed_at_the_end():
    df = make_df([100, 100, 100, 150])
    result = run_backtest(df, make_signals(df, entry=[0]), FREE)
    assert result.trades.iloc[0]["exit_reason"] == "end"
    assert result.equity.iloc[-1] == pytest.approx(15_000)


def test_max_drawdown():
    assert max_drawdown(pd.Series([100.0, 120, 60, 90])) == pytest.approx(-0.5)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_strategies_run_and_do_not_look_ahead(name):
    rng = np.random.default_rng(42)
    closes = 100 * np.exp(rng.normal(0.0005, 0.02, 600).cumsum())
    opens = np.concatenate([[100], closes[:-1]])
    df = make_df(opens, closes=closes)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.005
    df["low"] = df[["open", "close"]].min(axis=1) * 0.995

    strategy = build_strategy(name)
    full = strategy.generate(df)
    part = strategy.generate(df.iloc[:400])
    pd.testing.assert_frame_equal(full.iloc[:400], part)

    result = run_backtest(df, full, FREE)
    metrics = compute_metrics(result, df, FREE)
    assert result.equity.gt(0).all()
    assert -1 <= metrics["max_drawdown"] <= 0


def test_build_strategy_casts_params():
    strategy = build_strategy("ema_cross", {"fast": "10", "atr_stop": "1.5"})
    assert strategy.fast == 10 and strategy.atr_stop == 1.5
    with pytest.raises(ValueError):
        build_strategy("ema_cross", {"nope": "1"})
