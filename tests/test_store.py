from trading_agent.data.store import CandleStore, to_ms


def test_roundtrip_upsert_and_range(tmp_path):
    store = CandleStore(tmp_path / "m.sqlite")
    t0 = to_ms("2024-01-01")
    day = 86_400_000
    rows = [[t0 + i * day, 1 + i, 2 + i, 0.5 + i, 1.5 + i, 10] for i in range(5)]

    assert store.upsert("x", "BTC/USDT", "1d", rows) == 5
    store.upsert("x", "BTC/USDT", "1d", [[t0, 9, 9, 9, 9, 9]])  # overwrite, not duplicate

    df = store.load("x", "BTC/USDT", "1d")
    assert len(df) == 5
    assert df["close"].iloc[0] == 9
    assert str(df.index.tz) == "UTC"
    assert store.last_ts("x", "BTC/USDT", "1d") == t0 + 4 * day
    assert store.last_ts("x", "ETH/USDT", "1d") is None

    part = store.load("x", "BTC/USDT", "1d", start="2024-01-02", end="2024-01-04")
    assert len(part) == 2
    store.close()
