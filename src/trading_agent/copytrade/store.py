"""SQLite storage for followed traders and snapshots of their positions."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd

from trading_agent.copytrade.ranking import TraderStats

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traders (
    address TEXT PRIMARY KEY, name TEXT, account_value REAL, month_roi REAL, alltime_pnl REAL,
    consistency REAL, max_drawdown REAL, profit_to_pain REAL, history_days REAL, score REAL,
    ranked_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshot_traders (
    time INTEGER NOT NULL, address TEXT NOT NULL, account_value REAL NOT NULL,
    PRIMARY KEY (time, address)
);
CREATE TABLE IF NOT EXISTS positions (
    time INTEGER NOT NULL, address TEXT NOT NULL, coin TEXT NOT NULL, size REAL NOT NULL,
    entry_price REAL, value REAL NOT NULL, leverage REAL,
    PRIMARY KEY (time, address, coin)
);
"""

_TRADER_COLUMNS = [f.name for f in fields(TraderStats)]


class CopyStore:
    def __init__(self, db_path: str | Path):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # --- followed traders ------------------------------------------------
    def replace_traders(self, traders: list[TraderStats], ranked_at: int) -> None:
        self._conn.execute("DELETE FROM traders")
        marks = ", ".join("?" * (len(_TRADER_COLUMNS) + 1))
        self._conn.executemany(
            f"INSERT INTO traders ({', '.join(_TRADER_COLUMNS)}, ranked_at) VALUES ({marks})",
            [(*asdict(t).values(), ranked_at) for t in traders],
        )
        self._conn.commit()

    def traders(self) -> list[TraderStats]:
        rows = self._conn.execute(
            f"SELECT {', '.join(_TRADER_COLUMNS)} FROM traders ORDER BY score DESC"
        ).fetchall()
        return [TraderStats(*row) for row in rows]

    def ranked_at(self) -> int | None:
        return self._conn.execute("SELECT MAX(ranked_at) FROM traders").fetchone()[0]

    # --- position snapshots ----------------------------------------------
    def save_snapshot(self, time_ms: int, accounts: dict[str, tuple[float, list[dict]]]) -> None:
        """`accounts` maps address -> (account value, positions). Flat traders are recorded too."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO snapshot_traders VALUES (?, ?, ?)",
            [(time_ms, address, value) for address, (value, _) in accounts.items()],
        )
        self._conn.executemany(
            "INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (time_ms, address, p["coin"], p["size"], p["entry_price"], p["value"], p["leverage"])
                for address, (_, positions) in accounts.items()
                for p in positions
            ],
        )
        self._conn.commit()

    def latest_snapshot_time(self, at_or_before: int | None = None) -> int | None:
        if at_or_before is None:
            return self._conn.execute("SELECT MAX(time) FROM snapshot_traders").fetchone()[0]
        return self._conn.execute(
            "SELECT MAX(time) FROM snapshot_traders WHERE time <= ?", (at_or_before,)
        ).fetchone()[0]

    def first_snapshot_time(self) -> int | None:
        return self._conn.execute("SELECT MIN(time) FROM snapshot_traders").fetchone()[0]

    def exposures(self, time_ms: int, coin: str) -> pd.Series:
        """Signed exposure to `coin` per trader in one snapshot, as a fraction of their account.

        Positive = long, negative = short, 0 = no position. Indexed by address.
        """
        rows = self._conn.execute(
            "SELECT s.address, s.account_value, p.size, p.value FROM snapshot_traders s "
            "LEFT JOIN positions p ON p.time = s.time AND p.address = s.address AND p.coin = ? "
            "WHERE s.time = ?",
            (coin, time_ms),
        ).fetchall()
        data = {}
        for address, account_value, size, value in rows:
            if size is None or account_value <= 0:
                data[address] = 0.0
            else:
                data[address] = (1 if size > 0 else -1) * abs(value) / account_value
        return pd.Series(data, dtype=float)
