"""Performance overview of everything an account could trade.

Crypto: every active USDT spot pair on the exchange. Stocks and ETFs: a starter
list plus any tickers the user added. Used by the app's Markets page.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

COLUMNS = ["symbol", "price", "change_1d", "change_7d", "change_30d", "volume", "kind"]

DEFAULT_STOCKS = [
    # Broad and regional ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VEA", "VWO", "EFA", "EEM",
    # Bonds, gold, commodities
    "TLT", "IEF", "HYG", "LQD", "GLD", "SLV", "USO",
    # US sectors
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC", "SMH", "ARKK",
    # Large companies
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "JPM", "V", "MA",
    "UNH", "XOM", "JNJ", "WMT", "PG", "HD", "COST", "NFLX", "AMD", "CRM", "ORCL", "ADBE", "KO",
    "PEP", "DIS", "BAC", "INTC", "PLTR",
]


def _change(now: float, then: float | None) -> float | None:
    return (now / then - 1) * 100 if then else None


def _is_stable(price: float, high: float | None, low: float | None) -> bool:
    """Pegged to one dollar: priced near 1 and barely moving."""
    if not 0.97 <= price <= 1.03 or not high or not low:
        return False
    return (high - low) / price < 0.015


def crypto_overview(exchange, quote: str = "USDT", workers: int = 10) -> pd.DataFrame:
    """One row per active spot pair quoted in `quote`. `exchange` is a ccxt exchange."""
    markets = exchange.load_markets()
    symbols = [s for s, m in markets.items() if m.get("spot") and m.get("active") and m.get("quote") == quote]
    tickers = exchange.fetch_tickers()

    def month(symbol: str) -> list[float]:
        try:
            return [row[4] for row in exchange.fetch_ohlcv(symbol, "1d", limit=31)]
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        closes = dict(zip(symbols, pool.map(month, symbols)))

    rows = []
    for symbol in symbols:
        ticker = tickers.get(symbol) or {}
        price = ticker.get("last")
        if not price:
            continue
        history = closes[symbol]   # daily closes, oldest first, the last one is today so far
        rows.append({
            "symbol": symbol,
            "price": float(price),
            "change_1d": ticker.get("percentage"),
            "change_7d": _change(price, history[-8]) if len(history) >= 8 else None,
            "change_30d": _change(price, history[0]) if len(history) >= 31 else None,
            "volume": float(ticker.get("quoteVolume") or 0.0),
            "kind": "Stablecoin" if _is_stable(price, ticker.get("high"), ticker.get("low")) else "Crypto",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def mt5_overview(terminal, watched: list[str] | None = None) -> pd.DataFrame:
    """One row per tradable symbol of the MT5 broker. `watched` symbols also get 7- and 30-day changes."""
    api = terminal.api
    rows = []
    for info in terminal.call("symbols_get"):
        if info.trade_mode != api.SYMBOL_TRADE_MODE_FULL:
            continue
        price = info.bid or info.last      # zero until the symbol is in Market Watch
        rows.append({
            "symbol": info.name, "price": float(price) if price else None,
            "change_1d": float(info.price_change) if price else None,
            "change_7d": None, "change_30d": None, "volume": 1.0 if price else 0.0,   # quoted ones sort first
            "kind": info.path.replace("/", "\\").split("\\")[0] or "Other",
        })
    table = pd.DataFrame(rows, columns=COLUMNS)
    for symbol in watched or []:
        match = table["symbol"] == symbol
        if not match.any():
            continue
        try:
            terminal.symbol(symbol)
            closes = [float(r["close"]) for r in terminal.call("copy_rates_from_pos", symbol, api.TIMEFRAME_D1, 0, 23)]
        except Exception:
            continue
        price = float(table.loc[match, "price"].iloc[0])
        if len(closes) >= 6:
            table.loc[match, "change_7d"] = _change(price, closes[-6])
        if len(closes) >= 22:
            table.loc[match, "change_30d"] = _change(price, closes[-22])
    return table


def stock_overview(tickers: list[str], yfinance=None) -> pd.DataFrame:
    """One row per ticker Yahoo has data for. Changes are over 1, 5 and 21 trading days."""
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(columns=COLUMNS)
    if yfinance is None:
        import yfinance
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    data = yfinance.download(tickers, period="3mo", interval="1d", auto_adjust=True,
                             group_by="ticker", progress=False, threads=True)
    rows = []
    for ticker in tickers:
        try:
            frame = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
            frame = frame.dropna(subset=["Close"])
        except KeyError:
            continue
        if len(frame) < 2:
            continue
        close = frame["Close"]
        price = float(close.iloc[-1])
        rows.append({
            "symbol": ticker,
            "price": price,
            "change_1d": _change(price, float(close.iloc[-2])),
            "change_7d": _change(price, float(close.iloc[-6])) if len(close) >= 6 else None,
            "change_30d": _change(price, float(close.iloc[-22])) if len(close) >= 22 else None,
            "volume": float(frame["Volume"].iloc[-1]) * price,
            "kind": "Stock or ETF",
        })
    return pd.DataFrame(rows, columns=COLUMNS)
