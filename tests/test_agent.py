import json

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from trading_agent.agent import LLMDecider, TradeDecision, build_snapshot
from trading_agent.config import LLMConfig
from trading_agent.journal import Position


def make_df(n=1500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 * np.exp(rng.normal(0.0003, 0.01, n).cumsum())
    open_ = np.concatenate([[100], close[:-1]])
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) * 1.003, "low": np.minimum(open_, close) * 0.997,
         "close": close, "volume": rng.uniform(50, 150, n)},
        index=index,
    )


DF = make_df()
PROMPT_CFG = LLMConfig(structured_output="prompt", min_confidence=0.6)


def reply(**fields) -> str:
    base = {"reasoning": "Trend and momentum agree.", "action": "hold", "confidence": 0.5, "stop_atr_multiple": 2.5}
    return json.dumps({**base, **fields})


def prompt_decider(*replies: str) -> LLMDecider:
    model = GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))
    return LLMDecider(PROMPT_CFG, "4h", model=model)


def position() -> Position:
    entry_ms = int(DF.index[-11].timestamp() * 1000)
    return Position("BTC/USDT", 1.0, float(DF["close"].iloc[-11]), entry_ms, 100.0, 90.0, "test")


# --- snapshot -----------------------------------------------------------
def test_snapshot_is_json_and_complete():
    snapshot, atr = build_snapshot("BTC/USDT", "4h", DF, None)
    text = json.dumps(snapshot)                      # serialisable, no NaN
    assert "NaN" not in text
    assert atr > 0
    assert snapshot["open_position"] is None
    assert snapshot["daily_context"]["price_vs_sma200_pct"] is not None
    assert len(snapshot["recent_candles"]) == 12
    assert set(snapshot["rule_strategy_signals_this_bar"]) == {"ema_cross", "donchian_breakout", "rsi_dip"}


def test_snapshot_with_position_and_short_history():
    snapshot, _ = build_snapshot("BTC/USDT", "4h", DF, position())
    assert snapshot["open_position"]["bars_held"] == 10

    short, atr = build_snapshot("BTC/USDT", "4h", DF.head(10), None)
    assert np.isnan(atr)
    assert short["trend"]["price_vs_ema200_pct"] is None
    json.dumps(short)


def test_snapshot_does_not_use_future_bars():
    full, _ = build_snapshot("BTC/USDT", "4h", DF.iloc[:1000], None)
    again, _ = build_snapshot("BTC/USDT", "4h", DF.iloc[:1000].copy(), None)
    assert full == again
    assert full["last_closed_candle_utc"] == DF.index[999].strftime("%Y-%m-%d %H:%M")


# --- decisions ----------------------------------------------------------
def test_buy_sets_stop_from_atr_multiple():
    _, atr = build_snapshot("BTC/USDT", "4h", DF, None)
    decision = prompt_decider(reply(action="buy", confidence=0.8, stop_atr_multiple=2.0)).decide("BTC/USDT", DF, None)
    assert decision.action == "buy"
    assert decision.stop_distance == pytest.approx(2.0 * atr)
    assert "confidence 0.80" in decision.reason


def test_stop_multiple_is_clamped():
    _, atr = build_snapshot("BTC/USDT", "4h", DF, None)
    decision = prompt_decider(reply(action="buy", confidence=0.9, stop_atr_multiple=0.1)).decide("BTC/USDT", DF, None)
    assert decision.stop_distance == pytest.approx(1.0 * atr)


def test_low_confidence_buy_becomes_hold():
    decision = prompt_decider(reply(action="buy", confidence=0.4)).decide("BTC/USDT", DF, None)
    assert decision.action == "hold" and "threshold" in decision.reason


def test_impossible_actions_become_hold():
    assert prompt_decider(reply(action="sell", confidence=0.9)).decide("BTC/USDT", DF, None).action == "hold"
    assert prompt_decider(reply(action="buy", confidence=0.9)).decide("BTC/USDT", DF, position()).action == "hold"
    assert prompt_decider(reply(action="sell", confidence=0.9)).decide("BTC/USDT", DF, position()).action == "sell"


def test_prompt_mode_tolerates_text_around_json_and_block_content():
    wrapped = "Here is my decision:\n```json\n" + reply(action="hold") + "\n```"
    assert prompt_decider(wrapped).decide("BTC/USDT", DF, None).action == "hold"

    blocks = AIMessage(content=[{"type": "thinking", "thinking": "..."}, {"type": "text", "text": reply()}])
    decider = LLMDecider(PROMPT_CFG, "4h", model=GenericFakeChatModel(messages=iter([blocks])))
    assert decider.decide("BTC/USDT", DF, None).action == "hold"


def test_bad_reply_raises():
    with pytest.raises(ValueError):
        prompt_decider("I cannot decide.").decide("BTC/USDT", DF, None)
    with pytest.raises(ValueError):
        prompt_decider(reply(action="short")).decide("BTC/USDT", DF, None)


def test_structured_mode_uses_the_configured_method():
    calls = {}

    class Structured:
        def invoke(self, messages):
            calls["messages"] = messages
            return TradeDecision(reasoning="ok", action="buy", confidence=0.9, stop_atr_multiple=3.0)

    class Model:
        def with_structured_output(self, schema, method):
            calls["schema"], calls["method"] = schema, method
            return Structured()

    decider = LLMDecider(LLMConfig(structured_output="json_schema"), "4h", model=Model())
    assert decider.decide("ETH/USDT", DF, None).action == "buy"
    assert calls["schema"] is TradeDecision and calls["method"] == "json_schema"
    system, human = calls["messages"]
    assert system[0] == "system" and "JSON object" not in system[1]
    assert '"symbol": "ETH/USDT"' in human[1]
