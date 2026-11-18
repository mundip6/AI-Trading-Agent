"""Persistent record of the demo account: positions, fills, trades, decisions, equity."""

from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading_agent.broker.base import Fill

DAY_MS = 86_400_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY, qty REAL NOT NULL, entry_price REAL NOT NULL,
    entry_time INTEGER NOT NULL, entry_cost REAL NOT NULL, stop REAL, source TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time INTEGER NOT NULL, symbol TEXT NOT NULL,
    side TEXT NOT NULL, qty REAL NOT NULL, price REAL NOT NULL, fee REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, source TEXT,
    entry_time INTEGER NOT NULL, exit_time INTEGER NOT NULL,
    entry_price REAL NOT NULL, exit_price REAL NOT NULL, qty REAL NOT NULL,
    pnl REAL NOT NULL, pnl_pct REAL NOT NULL, exit_reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time INTEGER NOT NULL, symbol TEXT NOT NULL,
    timeframe TEXT, bar_time INTEGER, source TEXT, action TEXT NOT NULL, reason TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    time INTEGER PRIMARY KEY, equity REAL NOT NULL, cash REAL NOT NULL
);
"""

_TABLES = ["kv", "positions", "fills", "trades", "decisions", "equity"]


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: int          # ms since epoch
    entry_cost: float        # quote currency spent, fee included
    stop: float | None
    source: str              # which strategy / agent opened it
    # Leveraged (MT5) accounts only. There `qty` is signed: negative = short.
    take_profit: float | None = None
    ticket: int | None = None
    lots: float | None = None

    @property
    def side(self) -> str:
        return "short" if self.qty < 0 else "long"


class Journal:
    def __init__(self, db_path: str | Path):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def reset(self) -> None:
        for table in _TABLES:
            self._conn.execute(f"DELETE FROM {table}")
        self._conn.commit()

    # --- key/value state -------------------------------------------------
    def get(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set(self, key: str, value) -> None:
        self._conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, str(value)))
        self._conn.commit()

    # --- positions -------------------------------------------------------
    def positions(self) -> dict[str, Position]:
        rows = self._conn.execute(
            "SELECT symbol, qty, entry_price, entry_time, entry_cost, stop, source FROM positions"
        ).fetchall()
        positions = {row[0]: Position(*row) for row in rows}
        for symbol, position in positions.items():
            extra = json.loads(self.get(f"position_extra:{symbol}") or "{}")
            position.take_profit, position.ticket, position.lots = (
                extra.get("take_profit"), extra.get("ticket"), extra.get("lots"))
        return positions

    def open_position(self, position: Position) -> None:
        p = position
        self._conn.execute(
            "INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (p.symbol, p.qty, p.entry_price, p.entry_time, p.entry_cost, p.stop, p.source),
        )
        self._conn.execute("DELETE FROM kv WHERE key=?", (f"position_extra:{p.symbol}",))
        if p.ticket is not None:
            self._conn.execute(
                "INSERT INTO kv VALUES (?, ?)",
                (f"position_extra:{p.symbol}",
                 json.dumps({"take_profit": p.take_profit, "ticket": p.ticket, "lots": p.lots})),
            )
        self._conn.commit()

    def close_margin_position(
        self, position: Position, exit_price: float, pnl: float, time_ms: int, reason: str
    ) -> None:
        """Leveraged position closed at the broker: the broker's own P&L is recorded.

        `qty` stays signed (negative = short); `pnl_pct` is the price move in the trade's favour."""
        direction = -1 if position.qty < 0 else 1
        self._conn.execute("DELETE FROM positions WHERE symbol=?", (position.symbol,))
        self._conn.execute("DELETE FROM kv WHERE key=?", (f"position_extra:{position.symbol}",))
        self._conn.execute(
            "INSERT INTO trades (symbol, source, entry_time, exit_time, entry_price, exit_price,"
            " qty, pnl, pnl_pct, exit_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (position.symbol, position.source, position.entry_time, time_ms, position.entry_price,
             exit_price, position.qty, pnl, direction * (exit_price / position.entry_price - 1), reason),
        )
        self._conn.commit()

    def close_position(self, position: Position, fill: Fill, time_ms: int, reason: str) -> float:
        """Remove the position, record the closed trade, return its P&L."""
        proceeds = fill.value - fill.fee
        pnl = proceeds - position.entry_cost
        self._conn.execute("DELETE FROM positions WHERE symbol=?", (position.symbol,))
        self._conn.execute(
            "INSERT INTO trades (symbol, source, entry_time, exit_time, entry_price, exit_price,"
            " qty, pnl, pnl_pct, exit_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (position.symbol, position.source, position.entry_time, time_ms, position.entry_price,
             fill.price, fill.qty, pnl, pnl / position.entry_cost, reason),
        )
        self._conn.commit()
        return pnl

    # --- logs ------------------------------------------------------------
    def record_fill(self, fill: Fill, time_ms: int) -> None:
        self._conn.execute(
            "INSERT INTO fills (time, symbol, side, qty, price, fee) VALUES (?, ?, ?, ?, ?, ?)",
            (time_ms, fill.symbol, fill.side, fill.qty, fill.price, fill.fee),
        )
        self._conn.commit()

    def record_decision(
        self, time_ms: int, symbol: str, timeframe: str, bar_time: int,
        source: str, action: str, reason: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO decisions (time, symbol, timeframe, bar_time, source, action, reason)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time_ms, symbol, timeframe, bar_time, source, action, reason),
        )
        self._conn.commit()

    def record_equity(self, time_ms: int, equity: float, cash: float) -> None:
        self._conn.execute("INSERT OR REPLACE INTO equity VALUES (?, ?, ?)", (time_ms, equity, cash))
        self._conn.commit()

    def last_equity_time(self) -> int | None:
        return self._conn.execute("SELECT MAX(time) FROM equity").fetchone()[0]

    def day_start_equity(self, now_ms: int) -> float | None:
        """Equity the account had when the current UTC day began."""
        day_start = now_ms - now_ms % DAY_MS
        row = self._conn.execute(
            "SELECT equity FROM equity WHERE time < ? ORDER BY time DESC LIMIT 1", (day_start,)
        ).fetchone() or self._conn.execute(
            "SELECT equity FROM equity WHERE time >= ? ORDER BY time LIMIT 1", (day_start,)
        ).fetchone()
        return row[0] if row else None

    # --- reporting -------------------------------------------------------
    def _frame(self, query: str, time_cols: list[str], params: tuple = (), as_text: bool = True) -> pd.DataFrame:
        df = pd.read_sql_query(query, self._conn, params=params)
        for col in time_cols:
            df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
            if as_text:
                df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M")
        return df

    def trades_frame(self, as_text: bool = True) -> pd.DataFrame:
        return self._frame(
            "SELECT symbol, source, entry_time, exit_time, entry_price, exit_price, qty, pnl,"
            " pnl_pct, exit_reason FROM trades ORDER BY id",
            ["entry_time", "exit_time"],
            as_text=as_text,
        )

    def decisions_frame(self, limit: int = 30, as_text: bool = True) -> pd.DataFrame:
        df = self._frame(
            "SELECT time, symbol, timeframe, bar_time, source, action, reason FROM decisions"
            " ORDER BY id DESC LIMIT ?",
            ["time", "bar_time"],
            (limit,),
            as_text=as_text,
        )
        return df.iloc[::-1]

    def fills_frame(self) -> pd.DataFrame:
        return self._frame(
            "SELECT time, symbol, side, qty, price, fee FROM fills ORDER BY id", ["time"], as_text=False
        )

    def equity_series(self) -> pd.Series:
        """Account value over time, indexed by UTC timestamp."""
        df = self._frame("SELECT time, equity FROM equity ORDER BY time", ["time"], as_text=False)
        return df.set_index("time")["equity"]
