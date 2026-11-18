"""Which symbols an account monitors and trades, chosen on the app's Markets page.

Stored next to the account's data as JSON. When the file exists it replaces the
`symbols` lists of the account's config, and a running loop re-reads it on
every pass, so changes apply without a restart.
"""

from __future__ import annotations

import json
from pathlib import Path


def watchlist_path(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    return config_path.parent / "data" / "watchlists" / f"{config_path.stem}.json"


def read_watchlist(config_path: str | Path) -> dict | None:
    """{'symbols': [...], 'extra': [...]} or None when the account has no watchlist file."""
    path = watchlist_path(config_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("symbols"), list):
        return None
    return {"symbols": [str(s) for s in data["symbols"]], "extra": [str(s) for s in data.get("extra", [])]}


def write_watchlist(config_path: str | Path, symbols: list[str], extra: list[str] | None = None) -> Path:
    """`extra` holds tickers the user added to the browsable list (stocks and ETFs)."""
    path = watchlist_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = read_watchlist(config_path) or {}
    data = {
        "symbols": list(dict.fromkeys(symbols)),
        "extra": list(dict.fromkeys(previous.get("extra", []) if extra is None else extra)),
    }
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    temp.replace(path)
    return path
