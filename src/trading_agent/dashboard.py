"""The app: accounts, performance, ask the agent, followed traders, settings.

Opened by the `trading-agent` launcher or `trader dashboard`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_agent.config import load_config
from trading_agent.credentials import KEY_FOR_PROVIDER
from trading_agent.data.exchange import is_continuous, make_exchange
from trading_agent.data.store import CandleStore
from trading_agent.journal import Journal
from trading_agent.reporting import account_summary, benchmark_curve, indexed
from trading_agent.runner import (
    PROJECT_ROOT, Account, discover_accounts, is_running, running_loops, start_loop, stop_loop, tail,
)

# Chart colours: validated categorical slots 1-3 and chrome, per theme.
PALETTE = {
    "light": {"account": "#2a78d6", "benchmark": "#eb6834", "buy": "#1baf7a", "sell": "#eb6834",
              "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "surface": "#ffffff"},
    "dark": {"account": "#3987e5", "benchmark": "#d95926", "buy": "#199e70", "sell": "#d95926",
             "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "surface": "#1a1a19"},
}


def theme() -> dict:
    try:
        kind = st.context.theme.type or "light"
    except Exception:
        kind = "light"
    return PALETTE.get(kind, PALETTE["light"])


@st.cache_data(ttl=30, show_spinner=False)
def live_price(exchange_id: str, symbol: str) -> float | None:
    try:
        return float(make_exchange(exchange_id).fetch_ticker(symbol)["last"])
    except Exception:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def mt5_floating(tickets: tuple[int, ...]) -> dict[int, float]:
    """Floating P&L of MT5 positions in the account currency, read from the terminal."""
    try:
        from trading_agent.data.mt5 import shared_terminal

        terminal = shared_terminal()
        result = {}
        for ticket in tickets:
            found = terminal.call("positions_get", ticket=ticket)
            if found:
                result[ticket] = float(found[0].profit) + float(found[0].swap)
        return result
    except Exception:
        return {}


def style(fig: go.Figure, colors: dict, height: int, y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=56, r=110, t=10, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=colors["ink2"], size=13),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=colors["surface"], font_color=colors["ink"], bordercolor=colors["axis"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font_color=colors["ink2"]),
    )
    axis = dict(gridcolor=colors["grid"], linecolor=colors["axis"], zeroline=False,
                tickfont_color=colors["muted"])
    fig.update_xaxes(**axis, showgrid=False, showspikes=True, spikecolor=colors["axis"],
                     spikethickness=1, spikedash="solid", spikemode="across")
    fig.update_yaxes(**axis, title_text=y_title, title_font_color=colors["muted"])
    return fig


def end_label(fig: go.Figure, series: pd.Series, text: str, colors: dict, yshift: int = 0) -> None:
    fig.add_annotation(x=series.index[-1], y=series.iloc[-1], text=text, showarrow=False,
                       xanchor="left", xshift=8, yshift=yshift, font=dict(color=colors["ink"], size=12))


def equity_chart(account: pd.Series, benchmark: pd.Series, colors: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=account.index, y=account, name="Account", mode="lines",
                    line=dict(color=colors["account"], width=2), hovertemplate="%{y:.2f}")
    # Nudge the two end labels apart when the lines finish close together.
    spread = max(account.max(), benchmark.max() if len(benchmark) else 0) - min(
        account.min(), benchmark.min() if len(benchmark) else account.min())
    close = len(benchmark) and abs(account.iloc[-1] - benchmark.iloc[-1]) < 0.08 * (spread or 1)
    above = not len(benchmark) or account.iloc[-1] >= benchmark.iloc[-1]
    nudge = 9 if close else 0
    end_label(fig, account, f"Account {account.iloc[-1]:.1f}", colors, nudge if above else -nudge)
    if len(benchmark):
        fig.add_scatter(x=benchmark.index, y=benchmark, name="Buy and hold", mode="lines",
                        line=dict(color=colors["benchmark"], width=2), hovertemplate="%{y:.2f}")
        end_label(fig, benchmark, f"Buy and hold {benchmark.iloc[-1]:.1f}", colors, -nudge if above else nudge)
    fig.update_xaxes(range=[account.index[0], account.index[-1]])   # end labels sit in the right margin
    return style(fig, colors, 340)


def price_chart(candles: pd.DataFrame, fills: pd.DataFrame, stop: float | None, colors: dict,
                target: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=candles.index, y=candles["close"], name="Close", mode="lines",
                    line=dict(color=colors["account"], width=2), hovertemplate="%{y:,.6g}")
    ring = dict(color=colors["surface"], width=2)
    for side, label, symbol, color in (("buy", "Buy", "triangle-up", colors["buy"]),
                                       ("sell", "Sell", "triangle-down", colors["sell"])):
        rows = fills[fills["side"] == side]
        if len(rows):
            fig.add_scatter(x=rows["time"], y=rows["price"], name=label, mode="markers",
                            marker=dict(symbol=symbol, size=13, color=color, line=ring),
                            hovertemplate=label + " %{y:,.6g}")
    if stop:
        fig.add_hline(y=stop, line=dict(color=colors["muted"], width=1, dash="dash"),
                      annotation_text=f"Stop {stop:,.6g}", annotation_position="right",
                      annotation_font_color=colors["ink2"])
    if target:
        fig.add_hline(y=target, line=dict(color=colors["muted"], width=1, dash="dot"),
                      annotation_text=f"Target {target:,.6g}", annotation_position="right",
                      annotation_font_color=colors["ink2"])
    return style(fig, colors, 360)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    account = current_account()
    cfg = account.cfg
    colors = theme()
    if not Path(cfg.paper.db_path).exists():
        st.title(account.name)
        st.info("This account has not been started yet. Start it on the Accounts page.")
        return
    journal = Journal(cfg.paper.db_path)
    store = CandleStore(cfg.data.db_path)
    symbols = cfg.paper.symbols or cfg.data.symbols
    timeframe = cfg.paper.timeframe

    equity = journal.equity_series()
    trades = journal.trades_frame(as_text=False)
    fills = journal.fills_frame()
    decisions = journal.decisions_frame(limit=500, as_text=False)
    positions = journal.positions()
    if equity.empty:
        st.title(account.name)
        st.info("This account has no history yet. Start it on the Accounts page.")
        return

    # Live valuation of open positions; fall back to the last recorded equity.
    margin = cfg.paper.broker == "mt5"
    floating = mt5_floating(tuple(p.ticket for p in positions.values())) if margin else {}
    rows, invested, live_ok = [], 0.0, True
    for symbol, pos in positions.items():
        price = live_price(cfg.data.exchange, symbol)
        live_ok &= price is not None
        price = price or pos.entry_price
        if margin:
            rows.append({
                "Symbol": symbol, "Side": pos.side.capitalize(), "Lots": pos.lots, "Entry": pos.entry_price,
                "Price": price, "Stop": pos.stop, "Target": pos.take_profit, "P&L": floating.get(pos.ticket),
                "Opened": pd.Timestamp(pos.entry_time, unit="ms", tz="UTC"), "By": pos.source,
            })
            continue
        value = pos.qty * price
        invested += value
        rows.append({
            "Symbol": symbol, "Quantity": pos.qty, "Entry": pos.entry_price, "Price": price,
            "Stop": pos.stop, "Value": value, "P&L": value - pos.entry_cost,
            "P&L %": (value / pos.entry_cost - 1) * 100,
            "Opened": pd.Timestamp(pos.entry_time, unit="ms", tz="UTC"), "By": pos.source,
        })

    summary = account_summary(equity, trades, float(journal.get("initial_equity") or equity.iloc[0]))
    benchmark = benchmark_curve(store, cfg.data.exchange, symbols, timeframe, equity.index)
    source = decisions["source"].iloc[-1] if len(decisions) else "n/a"
    last = equity.index[-1]

    st.title(account.name)
    st.caption(
        f"Broker: {journal.get('broker')} · Decided by: {source} · {timeframe} candles on "
        f"{', '.join(symbols)} · Last recorded {last:%Y-%m-%d %H:%M} UTC"
    )
    if not is_running(account):
        st.warning("This account's trading loop is not running, so nothing new is traded. "
                   + ("Open positions keep their stop-loss and take-profit at the broker. " if margin
                      else "Stop-losses are not being watched. ")
                   + "Start it on the Accounts page.", icon=":material/warning:")
    if not live_ok:
        st.caption("Live prices are unavailable; open positions are shown at their entry price.")

    bench_return = benchmark.iloc[-1] / 100 - 1 if len(benchmark) else None
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Equity", f"{summary['equity']:,.2f}", f"{summary['total_return']:+.2%} since start")
    k2.metric("Buy and hold, same period", "n/a" if bench_return is None else f"{bench_return:+.2%}",
              None if bench_return is None else
              f"account {'ahead' if summary['total_return'] >= bench_return else 'behind'} by "
              f"{abs(summary['total_return'] - bench_return) * 100:.2f} pts",
              delta_color="off", delta_arrow="off")
    k3.metric("Max drawdown", f"{summary['max_drawdown']:.2%}")
    k4.metric("Closed trades", summary["trades"],
              f"{summary['win_rate']:.0%} winners" if summary["trades"] else None, delta_color="off", delta_arrow="off")
    if margin:
        k5.metric("Open positions", len(positions),
                  f"{sum(floating.values()):+,.2f} floating" if floating else None, delta_color="off", delta_arrow="off")
    else:
        k5.metric("Invested", f"{invested / summary['equity']:.0%}" if summary["equity"] else "0%",
                  f"{len(positions)} open", delta_color="off", delta_arrow="off")

    st.subheader("Account against buy and hold")
    st.caption("Both start at 100. Buy and hold is an equal split across the traded symbols, never sold."
               + (" The account is in its own currency, as the broker reports it." if margin else ""))
    st.plotly_chart(equity_chart(indexed(equity), benchmark, colors), theme=None, config={"displayModeBar": False})

    st.subheader("Open positions")
    if rows:
        st.dataframe(
            pd.DataFrame(rows), hide_index=True,
            column_config={
                "Value": st.column_config.NumberColumn(format="%.2f"),
                "P&L": st.column_config.NumberColumn(format="%+.2f"),
                "P&L %": st.column_config.NumberColumn(format="%+.2f%%"),
                "Opened": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            },
        )
    else:
        st.caption("None.")

    st.subheader("Market")
    left, right = st.columns([2, 1])
    charted = list(dict.fromkeys([*symbols, *positions]))
    if not charted:
        st.caption("Nothing is monitored. Choose symbols on the Markets page.")
        return
    if len(charted) > 6:
        symbol = left.selectbox("Symbol", charted)
    else:
        symbol = left.segmented_control("Symbol", charted, default=charted[0]) or charted[0]
    from trading_agent.data.store import timeframe_ms

    periods = [2, 7, 30] if timeframe_ms(timeframe) <= 3_600_000 else [30, 90, 180]   # intraday: fewer days
    days = right.segmented_control("Period", periods, default=periods[1], format_func=lambda d: f"{d} days",
                                   key=f"period-{account.key}") or periods[1]
    candles = store.load(cfg.data.exchange, symbol, timeframe,
                         start=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days))
    if candles.empty:
        st.caption("No candles stored for this symbol.")
    else:
        visible = fills[(fills["symbol"] == symbol) & (fills["time"] >= candles.index[0])]
        position = positions.get(symbol)
        st.plotly_chart(price_chart(candles, visible, position.stop if position else None, colors,
                                    position.take_profit if position else None),
                        theme=None, config={"displayModeBar": False})

    if is_continuous(cfg.data.exchange) and Path(cfg.copytrade.db_path).exists():   # the feed is crypto-only
        from trading_agent.copytrade import CopyTrading

        copy = CopyTrading(cfg.copytrade)
        followed = copy.store.traders()
        positioning = [(s, copy.consensus(s)) for s in symbols]
        positioning = [(s, c) for s, c in positioning if c]
        if followed and positioning:
            st.subheader("Followed traders")
            age = positioning[0][1]["data_age_minutes"]
            st.caption(
                f"How the {len(followed)} followed Hyperliquid traders are positioned, read {age} minutes ago. "
                "Average exposure is their position as a share of their account; negative means short. "
                + ("The LLM agent sees this." if cfg.copytrade.enabled else "Not shown to the agent in this config.")
            )
            st.dataframe(
                pd.DataFrame([
                    {"Symbol": s, "Long": c["long"], "Short": c["short"], "No position": c["no_position"],
                     "Average exposure %": c["avg_exposure_pct_of_account"]}
                    for s, c in positioning
                ]),
                hide_index=True,
                column_config={"Average exposure %": st.column_config.NumberColumn(format="%+.1f%%")},
            )

    tab_decisions, tab_trades = st.tabs(["Decisions", "Closed trades"])
    with tab_decisions:
        only_actions = st.toggle("Hide holds", value=False)
        shown = decisions.iloc[::-1]
        if only_actions:
            shown = shown[shown["action"] != "hold"]
        st.dataframe(
            shown[["time", "symbol", "action", "reason", "source"]].rename(columns=str.capitalize),
            hide_index=True,
            column_config={
                "Time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "Reason": st.column_config.TextColumn(width="large"),
            },
        )
    with tab_trades:
        if len(trades):
            table = trades.iloc[::-1].assign(pnl_pct=lambda d: d["pnl_pct"] * 100)
            if margin:   # quantity is signed there: negative = short
                table.insert(1, "Side", table["qty"].map(lambda q: "Short" if q < 0 else "Long"))
                table["qty"] = table["qty"].abs()
            st.dataframe(
                table.rename(columns={
                    "symbol": "Symbol", "source": "By", "entry_time": "Opened", "exit_time": "Closed",
                    "entry_price": "Entry", "exit_price": "Exit", "qty": "Quantity", "pnl": "P&L",
                    "pnl_pct": "Price move %" if margin else "P&L %", "exit_reason": "Closed by",
                }),
                hide_index=True,
                column_config={
                    "Opened": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    "Closed": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    "P&L": st.column_config.NumberColumn(format="%+.2f"),
                    "P&L %": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Price move %": st.column_config.NumberColumn(
                        format="%+.2f%%", help="Price change in the trade's favour. P&L is what the broker booked."),
                },
            )
        else:
            st.caption("No closed trades yet.")


# ---------------------------------------------------------------------------
# Accounts: start and stop the trading loops
# ---------------------------------------------------------------------------
def account_brief(cfg) -> dict | None:
    """Headline numbers of an account, or None if it has never been started."""
    if not Path(cfg.paper.db_path).exists():
        return None
    journal = Journal(cfg.paper.db_path)
    equity = journal.equity_series()
    if equity.empty:
        return None
    decisions = journal.decisions_frame(limit=1, as_text=False)
    initial = float(journal.get("initial_equity") or equity.iloc[0])
    return {
        "equity": float(equity.iloc[-1]),
        "return": float(equity.iloc[-1] / initial - 1),
        "positions": len(journal.positions()),
        "trades": len(journal.trades_frame()),
        "last_decision": decisions["time"].iloc[-1] if len(decisions) else None,
    }


@st.fragment(run_every="15s")
def accounts_panel() -> None:
    accounts = discover_accounts()
    loops = running_loops()
    valid = [a for a in accounts if a.cfg]
    running = [a for a in valid if is_running(a, loops)]

    left, right, _ = st.columns([1, 1, 4])
    if left.button("Start all", icon=":material/play_arrow:", disabled=len(running) == len(valid), width="stretch"):
        for account in valid:
            start_loop(account)
        st.rerun(scope="fragment")
    if right.button("Stop all", icon=":material/stop:", disabled=not running, width="stretch"):
        for account in running:
            stop_loop(account)
        st.rerun(scope="fragment")

    for account in accounts:
        with st.container(border=True):
            if account.cfg is None:
                st.markdown(f"**{account.path.name}** :red-badge[:material/error: Invalid]")
                st.caption(f"This config cannot be read: {account.error}. Fix it on the Settings page.")
                continue
            cfg = account.cfg
            active = is_running(account, loops)
            brief = account_brief(cfg)
            badge = (":green-badge[:material/play_arrow: Running]" if active
                     else ":gray-badge[:material/stop: Stopped]")
            head, action = st.columns([5, 1], vertical_alignment="center")
            head.markdown(f"**{account.name}** {badge}")
            head.caption(f"{account.path.name} · {cfg.paper.timeframe} candles on "
                         f"{', '.join(cfg.paper.symbols or cfg.data.symbols)}")
            if active:
                if action.button("Stop", key=f"stop-{account.key}", icon=":material/stop:", width="stretch"):
                    stop_loop(account)
                    st.rerun(scope="fragment")
            elif action.button("Start", key=f"start-{account.key}", icon=":material/play_arrow:",
                               type="primary", width="stretch"):
                start_loop(account)
                st.rerun(scope="fragment")

            if brief is None:
                st.caption("Not started yet: it trades the balance of the demo account in MetaTrader 5."
                           if cfg.paper.broker == "mt5" else
                           "Not started yet: it begins with the starting cash set in its config.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Equity", f"{brief['equity']:,.2f}", f"{brief['return']:+.2%} since start")
                c2.metric("Open positions", brief["positions"])
                c3.metric("Closed trades", brief["trades"])
                last = brief["last_decision"]
                c4.metric("Last decision (UTC)", f"{last:%b %d %H:%M}" if last is not None else "None yet")
            if brief and brief["positions"] and not active:
                if cfg.paper.broker == "mt5":
                    st.caption("Stopped with open positions: their stop-loss and take-profit stay active "
                               "at the broker, but the agent is not deciding.")
                else:
                    st.warning("This account holds positions but its loop is stopped, so their stop-losses "
                               "are not being watched.", icon=":material/warning:")
            with st.expander("Log"):
                text = tail(account.log_path)
                if text:
                    st.code(text, language=None)
                elif active:
                    st.caption("This loop was started from a terminal; its output is in that terminal.")
                else:
                    st.caption("No log yet.")


def page_accounts() -> None:
    st.title("Accounts")
    st.caption("Each account is a demo-trading loop with its own settings and pretend money. "
               "A MetaTrader 5 account trades the demo account its terminal is logged in to. "
               "Loops keep running when this window is closed.")
    load_config(PROJECT_ROOT / "config.toml")       # loads .env into this process if not yet loaded
    if not any(os.environ.get(k) for k in KEY_FOR_PROVIDER.values()):
        st.info("No AI key yet: the LLM accounts and Ask the agent need one. Add yours on the API keys page.",
                icon=":material/key:")
    accounts_panel()


# ---------------------------------------------------------------------------
# Markets: browse everything, choose what the agent monitors and trades
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def crypto_markets(exchange_id: str) -> tuple[pd.DataFrame, str]:
    from trading_agent.markets import crypto_overview

    exchange = make_exchange(exchange_id)
    exchange.enableRateLimit = False   # a burst of small requests, well inside the exchange's limits
    return crypto_overview(exchange), f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}"


@st.cache_data(ttl=900, show_spinner=False)
def stock_markets(tickers: tuple[str, ...]) -> tuple[pd.DataFrame, str]:
    from trading_agent.markets import stock_overview

    return stock_overview(list(tickers)), f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}"


@st.cache_data(ttl=300, show_spinner=False)
def mt5_markets(watched: tuple[str, ...]) -> tuple[pd.DataFrame, str]:
    from trading_agent.data.mt5 import shared_terminal
    from trading_agent.markets import mt5_overview

    return mt5_overview(shared_terminal(), list(watched)), f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}"


def mixed_currencies(exchange_id: str, symbols: list[str]) -> dict[str, str]:
    """For stocks and ETFs: the symbols' currencies if they are not all the same, else {}."""
    exchange = make_exchange(exchange_id)
    currencies = {s: exchange.currency(s) for s in symbols}
    return currencies if len(set(currencies.values())) > 1 else {}


def page_markets() -> None:
    from trading_agent.data.store import timeframe_ms
    from trading_agent.markets import DEFAULT_STOCKS
    from trading_agent.watchlist import read_watchlist, write_watchlist

    account = current_account()
    cfg = account.cfg
    stocks = cfg.data.exchange == "yahoo"
    broker_symbols = cfg.data.exchange == "mt5"
    saved = list(cfg.paper.symbols or cfg.data.symbols)
    extra = (read_watchlist(account.path) or {}).get("extra", [])

    st.title("Markets")
    st.caption(f"{account.name}. Tick what this account should monitor and trade, then save. "
               "A running loop picks the change up within a minute.")

    if stocks:
        with st.form("add-ticker", clear_on_submit=True, border=False):
            left, right = st.columns([3, 1], vertical_alignment="bottom")
            ticker = left.text_input("Add a ticker to the list", placeholder="Any Yahoo Finance ticker, e.g. VUAA.DE")
            if right.form_submit_button("Add", icon=":material/add:", width="stretch") and ticker.strip():
                ticker = ticker.strip().upper()
                if stock_markets((ticker,))[0].empty:
                    st.error(f"Yahoo Finance has no data for {ticker}.")
                else:
                    write_watchlist(account.path, saved, [*extra, ticker])
                    st.rerun()
        universe = tuple(dict.fromkeys([*DEFAULT_STOCKS, *extra, *saved]))
        with st.spinner("Loading prices..."):
            table, stamp = stock_markets(universe)
    elif broker_symbols:
        try:
            table, stamp = mt5_markets(tuple(saved))
        except Exception as exc:
            st.error(f"{exc}")
            return
    else:
        with st.spinner("Loading every market on the exchange (about half a minute the first time)..."):
            table, stamp = crypto_markets(cfg.data.exchange)
    if table.empty:
        st.error("No market data could be loaded. Check the connection and press Refresh.")
        return

    pending_key = f"pending-{account.key}"
    if pending_key not in st.session_state:
        st.session_state[pending_key] = set(saved)
    pending: set[str] = st.session_state[pending_key]

    def reset_selection() -> None:
        """Forget unsaved ticks and rebuild the table from what is saved."""
        del st.session_state[pending_key]
        st.session_state[f"reset-{account.key}"] = st.session_state.get(f"reset-{account.key}", 0) + 1

    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    search = c1.text_input("Search", placeholder="Symbol", key=f"search-{account.key}").strip().upper()
    only_chosen = c2.toggle("Monitored only", key=f"only-{account.key}")
    hide_stable = False if stocks or broker_symbols else c3.toggle("Hide stablecoins", value=True, key=f"stable-{account.key}")

    # The table widget remembers ticks by row position. So the rows it is given must not change
    # while it is being edited: they are built from `base`, a copy of the selection taken when
    # the table was last (re)created, never from the live selection. Otherwise a tick would
    # re-sort the rows and land on whichever symbol moved into that position.
    signature = (f"{stamp}|{len(table)}|{search}|{only_chosen}|{hide_stable}|"
                 f"{st.session_state.get(f'reset-{account.key}', 0)}")
    base_key = f"base-{account.key}"
    if st.session_state.get(base_key, ("", None))[0] != signature:
        st.session_state[base_key] = (signature, frozenset(pending))
    base: frozenset[str] = st.session_state[base_key][1]

    view = table
    if search:
        view = view[view["symbol"].str.upper().str.contains(search, regex=False)]
    if only_chosen:
        view = view[view["symbol"].isin(base)]
    if hide_stable:
        view = view[(view["kind"] != "Stablecoin") | view["symbol"].isin(base)]
    view = view.assign(monitor=view["symbol"].isin(base))
    view = view.sort_values(["monitor", "volume", "symbol"], ascending=[False, False, True]).reset_index(drop=True)

    st.caption(f"{len(view)} of {len(table)} markets shown. Prices read {stamp} UTC. "
               "Click a column header to sort. "
               + ("Changes are over 1, 5 and 21 trading days." if stocks
                  else "Every symbol your broker offers; 7- and 30-day changes are shown for monitored ones."
                  if broker_symbols else "Volume is the last 24 hours, in USDT."))
    columns = ["monitor", "symbol", "price", "change_1d", "change_7d", "change_30d", "volume", "kind"]
    if broker_symbols:
        columns.remove("volume")
    edited = st.data_editor(
        view[columns],
        hide_index=True,
        height=560,
        disabled=["symbol", "price", "change_1d", "change_7d", "change_30d", "volume", "kind"],
        column_config={
            "monitor": st.column_config.CheckboxColumn("Monitor", help="Monitored and traded by this account"),
            "symbol": "Symbol",
            "price": st.column_config.NumberColumn("Price", format="%.6g"),
            "change_1d": st.column_config.NumberColumn("1 day", format="%+.2f%%"),
            "change_7d": st.column_config.NumberColumn("7 days", format="%+.2f%%"),
            "change_30d": st.column_config.NumberColumn("30 days", format="%+.2f%%"),
            "volume": st.column_config.NumberColumn("Volume", format="compact"),
            "kind": "Type",
        },
        key=f"markets-{account.key}-{signature}",   # a fresh table whenever its rows are rebuilt
    )
    shown = set(view["symbol"])
    ticked = set(edited.loc[edited["monitor"], "symbol"])
    pending -= shown - ticked
    pending |= ticked

    added = sorted(pending - set(saved))
    removed = sorted(set(saved) - pending)
    ordered = [s for s in saved if s in pending] + added

    per_day = 1 if stocks else 86_400_000 // timeframe_ms(cfg.paper.timeframe)
    if broker_symbols:
        per_day = per_day * 23 // 24    # these markets pause for about an hour a day
    note = f"**{len(pending)} monitored.**"
    if cfg.paper.decider == "llm":
        note += (f" The LLM agent is asked once per symbol per closed candle: about "
                 f"{len(pending) * per_day} model calls a day.")
    st.markdown(note)
    if added or removed:
        st.caption(("Adding: " + ", ".join(added) + ". " if added else "")
                   + ("Removing: " + ", ".join(removed) + "." if removed else ""))

    left, right, _ = st.columns([1, 1, 4])
    if left.button("Save selection", type="primary", icon=":material/save:", disabled=not (added or removed),
                   width="stretch"):
        clash = mixed_currencies(cfg.data.exchange, ordered) if stocks and ordered else {}
        if clash:
            st.error(f"Not saved: one account can hold only one currency, and these differ: {clash}")
        else:
            write_watchlist(account.path, ordered)
            reset_selection()
            st.toast(f"Saved: {len(ordered)} symbols monitored.")
            st.rerun()
    if right.button("Discard changes", disabled=not (added or removed), width="stretch"):
        reset_selection()
        st.rerun()
    if removed:
        st.caption("An open position in a removed symbol stays managed until it is closed; no new one is opened.")


# ---------------------------------------------------------------------------
# Ask the agent
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def llm_decider(config_path: str, modified: float):
    from trading_agent.factory import build_decider

    return build_decider(load_config(config_path), force_llm=True)


def ask_agent(account: Account, symbols: list[str]) -> list[dict]:
    from trading_agent.data import sync_candles
    from trading_agent.trader import LOOKBACK_BARS

    cfg = account.cfg
    decider = llm_decider(str(account.path), account.path.stat().st_mtime)
    store = CandleStore(cfg.data.db_path)
    exchange = make_exchange(cfg.data.exchange)
    db = Path(cfg.paper.db_path)
    positions = Journal(db).positions() if db.exists() else {}
    results = []
    for symbol in symbols:
        sync_candles(store, cfg.data.exchange, symbol, cfg.paper.timeframe, cfg.data.since, exchange)
        lookback = cfg.mt5.lookback_bars if cfg.paper.broker == "mt5" else LOOKBACK_BARS
        df = store.load(cfg.data.exchange, symbol, cfg.paper.timeframe).tail(lookback)
        if len(df) < 2:
            results.append({"symbol": symbol, "error": "No candles for this symbol."})
            continue
        position = positions.get(symbol)
        snapshot, _ = decider.snapshot(symbol, df, position)
        decision = decider.decide(symbol, df, position)
        results.append({"symbol": symbol, "action": decision.action, "reason": decision.reason,
                        "stop_distance": decision.stop_distance, "price": float(df["close"].iloc[-1]),
                        "target_distance": decision.take_profit_distance,
                        "news": decision.news, "outlook": decision.outlook,
                        "snapshot": snapshot, "asked": pd.Timestamp.now(tz="UTC")})
    return results


def page_ask() -> None:
    account = current_account()
    cfg = account.cfg
    st.title("Ask the agent")
    st.caption(f"{account.name}. The agent ({cfg.llm.model}) looks at the latest closed candle and says "
               "what it would do. No order is placed.")

    needed = KEY_FOR_PROVIDER.get(cfg.llm.provider)
    if needed and not os.environ.get(needed):
        st.warning(f"{needed} is not set. Add it on the API keys page.", icon=":material/key:")
        return

    symbols = list(cfg.paper.symbols or cfg.data.symbols)
    chosen = st.pills("Symbols", symbols, selection_mode="multi", default=symbols[:1], key=f"ask-{account.key}")
    st.caption("One model call per symbol.")
    if st.button("Ask", type="primary", icon=":material/psychology:", disabled=not chosen):
        with st.spinner("Reading the market and asking the model..."):
            try:
                st.session_state[f"answers-{account.key}"] = ask_agent(account, chosen)
            except Exception as exc:
                st.error(f"The agent could not answer: {type(exc).__name__}: {exc}")

    badges = {"buy": ":green-badge[Buy]", "sell": ":orange-badge[Sell]", "hold": ":gray-badge[Hold]",
              "long": ":green-badge[Long]", "short": ":orange-badge[Short]", "close": ":blue-badge[Close]"}
    for answer in st.session_state.get(f"answers-{account.key}", []):
        with st.container(border=True):
            if "error" in answer:
                st.markdown(f"**{answer['symbol']}**")
                st.caption(answer["error"])
                continue
            st.markdown(f"**{answer['symbol']}** {badges[answer['action']]}")
            st.write(answer["reason"])
            if answer["stop_distance"]:
                down = answer["action"] != "short"
                stop = answer["price"] - answer["stop_distance"] * (1 if down else -1)
                text = (f"Stop-loss would sit at about {stop:,.6g} "
                        f"({answer['stop_distance'] / answer['price']:.2%} {'below' if down else 'above'} the last close).")
                if answer.get("target_distance"):
                    target = answer["price"] + answer["target_distance"] * (1 if down else -1)
                    text += f" Take-profit at about {target:,.6g}."
                st.caption(text)
            if answer.get("news"):
                view = {"up": "price likely up", "down": "price likely down"}.get(answer["outlook"], "direction unclear")
                st.markdown(f"**News, as the agent reads it** ({view})")
                st.write(answer["news"])
            st.caption(f"Asked {answer['asked']:%Y-%m-%d %H:%M} UTC")
            with st.expander("What the agent was shown"):
                st.json(answer["snapshot"], expanded=1)


# ---------------------------------------------------------------------------
# Followed traders (copy-trading)
# ---------------------------------------------------------------------------
def page_traders() -> None:
    account = current_account()
    cfg = account.cfg
    st.title("Followed traders")
    if not is_continuous(cfg.data.exchange):
        st.info("Copy-trading data comes from Hyperliquid and covers crypto only. Pick a crypto account in the sidebar.")
        return
    from trading_agent.copytrade import CopyTrading

    copy = CopyTrading(cfg.copytrade)
    st.caption("Consistently profitable Hyperliquid traders, ranked from the public leaderboard. "
               + ("The LLM agent of this account sees their positioning." if cfg.copytrade.enabled
                  else "This account's agent does not use it (copytrade is off in its config)."))

    left, right, _ = st.columns([1, 1, 3])
    if left.button("Update positions", icon=":material/refresh:", width="stretch"):
        with st.spinner("Reading positions..."):
            copy.update()
    if right.button("Rebuild the list", icon=":material/leaderboard:", width="stretch",
                    help="Downloads the leaderboard and re-ranks. Takes a minute or two."):
        with st.status("Ranking traders...", expanded=True) as status:
            followed = copy.rerank(progress=st.write)
            copy.update()
            status.update(label=f"Following {len(followed)} traders", state="complete", expanded=False)

    traders = copy.store.traders()
    if not traders:
        st.info("Nobody is followed yet. Use Rebuild the list.")
        return

    symbols = cfg.paper.symbols or cfg.data.symbols
    positioning = [(s, copy.consensus(s)) for s in symbols]
    positioning = [(s, c) for s, c in positioning if c]
    st.subheader("How they are positioned")
    if positioning:
        st.caption(f"Read {positioning[0][1]['data_age_minutes']} minutes ago. Average exposure is the position "
                   "as a share of the trader's account; negative means short.")
        rows = []
        for symbol, c in positioning:
            changes = c["changes_last_24h"] or {}
            moved = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in changes.items() if k != "hours_compared" and v)
            rows.append({"Symbol": symbol, "Long": c["long"], "Short": c["short"], "No position": c["no_position"],
                         "Average exposure %": c["avg_exposure_pct_of_account"],
                         "Changes, last day": moved or ("none" if changes else "not enough history yet")})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"Average exposure %": st.column_config.NumberColumn(format="%+.1f%%")})
    else:
        st.caption("No position snapshot yet. Use Update positions.")

    ranked = pd.Timestamp(copy.store.ranked_at(), unit="ms", tz="UTC")
    st.subheader("Who is followed")
    st.caption(f"Ranked {ranked:%Y-%m-%d %H:%M} UTC.")
    st.dataframe(
        pd.DataFrame([
            {"Trader": t.name or t.address[:10] + "…", "Score": t.score, "Account value": t.account_value,
             "Month return %": t.month_roi * 100, "Profitable days %": t.consistency * 100,
             "Max drawdown %": t.max_drawdown * 100, "Profit to pain": t.profit_to_pain,
             "History, days": t.history_days}
            for t in traders
        ]),
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.2f"),
            "Account value": st.column_config.NumberColumn(format="dollar"),
            "Month return %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Profitable days %": st.column_config.NumberColumn(format="%.0f%%"),
            "Max drawdown %": st.column_config.NumberColumn(format="%.1f%%"),
            "Profit to pain": st.column_config.NumberColumn(format="%.1f"),
            "History, days": st.column_config.NumberColumn(format="%.0f"),
        },
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def save_config(path: Path, text: str) -> str | None:
    """Validate and save. Returns an error message, or None on success."""
    candidate = path.with_name(path.name + ".tmp")
    candidate.write_text(text, encoding="utf-8")
    try:
        load_config(candidate)
    except Exception as exc:
        candidate.unlink(missing_ok=True)
        return f"{type(exc).__name__}: {exc}"
    path.with_name(path.name + ".bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    candidate.replace(path)
    return None


def page_settings() -> None:
    st.title("Settings")
    accounts = discover_accounts()
    account = st.selectbox("Config file", accounts, format_func=lambda a: f"{a.path.name} — {a.name}")
    st.caption("Symbols, strategy or LLM model, risk limits, alerts, news and copy-trading are all set here. "
               "A running loop picks up changes when it is restarted.")
    if account.cfg is None:
        st.error(f"This file cannot be read: {account.error}")
    current = account.path.read_text(encoding="utf-8")
    text = st.text_area("Contents", current, height=520, key=f"cfg-{account.key}-{account.path.stat().st_mtime}",
                        label_visibility="collapsed")
    if st.button("Save", type="primary", icon=":material/save:", disabled=text == current):
        error = save_config(account.path, text)
        if error:
            st.error(f"Not saved, the file would be invalid: {error}")
        else:
            st.cache_resource.clear()
            st.toast(f"Saved {account.path.name}. The previous version is in {account.path.name}.bak.")
            st.rerun()

    st.caption("API keys for the AI model, Telegram, the Binance testnet and MetaTrader 5 are entered on the "
               "API keys page.")


# ---------------------------------------------------------------------------
# API keys: each user's own credentials, saved in .env on this computer
# ---------------------------------------------------------------------------
ENV_PATH = PROJECT_ROOT / ".env"


def _show(result: tuple[bool, str]) -> None:
    ok, text = result
    (st.success if ok else st.error)(text)


def page_keys() -> None:
    from trading_agent import credentials as cr

    st.title("API keys")
    st.caption("Your own keys and logins, saved in the .env file in the project folder on this computer. "
               "That file is never uploaded (git ignores it). Secrets are not shown again after saving, only "
               "their last four characters. Running loops use the new values after Stop and Start.")
    saved = cr.read_env(ENV_PATH)
    version = st.session_state.setdefault("keys-version", 0)

    def value(name: str) -> str:
        return saved.get(name) or os.environ.get(name, "")

    for group, (title, intro) in cr.GROUPS.items():
        items = [c for c in cr.CREDENTIALS if c.group == group]
        with st.container(border=True):
            st.subheader(title)
            st.caption(intro)
            with st.form(f"keys-{group}", border=False):
                entered = {}
                for c in items:
                    current = saved.get(c.name, "")
                    status = cr.describe(c, current)
                    if not current and os.environ.get(c.name):
                        status = "set in the system environment"
                    if c.secret:
                        entered[c.name] = st.text_input(
                            f"{c.label} ({status})", type="password", help=c.help or None,
                            placeholder="Paste to set or replace", key=f"in-{c.name}-{version}")
                    else:
                        entered[c.name] = st.text_input(c.label, value=current, help=c.help or None,
                                                        key=f"in-{c.name}-{version}")
                removable = [c for c in items if c.secret and saved.get(c.name)]
                remove = st.multiselect("Remove saved", removable, format_func=lambda c: c.label,
                                        key=f"rm-{group}-{version}") if removable else []
                if st.form_submit_button("Save", type="primary", icon=":material/save:"):
                    updates = {}
                    for c in items:
                        new = entered[c.name].strip()
                        if c.secret and new:
                            updates[c.name] = new
                        elif not c.secret and new != saved.get(c.name, ""):
                            updates[c.name] = new or None
                    updates.update({c.name: None for c in remove})
                    login = updates.get("MT5_LOGIN", saved.get("MT5_LOGIN")) or ""
                    if login and not login.isdigit():
                        st.error("The account number must be digits only.")
                    elif updates:
                        cr.save_env(ENV_PATH, updates)
                        st.session_state["keys-version"] = version + 1
                        st.cache_resource.clear()
                        st.toast(f"{title}: saved. Stop and Start running accounts to use it.", icon=":material/key:")
                        st.rerun()
                    else:
                        st.toast("Nothing to save.")

            if group == "ai":
                set_keys = [c for c in items if value(c.name)]
                cols = st.columns(max(len(set_keys), 1))
                for col, c in zip(cols, set_keys):
                    if col.button(f"Test {c.label.replace(' API key', '')} key", key=f"test-{c.name}"):
                        with st.spinner("Checking..."):
                            _show(cr.check_ai(c.name, value(c.name)))
            elif group == "telegram":
                token, chat = value("TELEGRAM_BOT_TOKEN"), value("TELEGRAM_CHAT_ID")
                left, right = st.columns(2)
                if left.button("Find my chat ID", icon=":material/search:", disabled=not token):
                    from trading_agent.alerts import telegram_chats

                    try:
                        st.session_state["tg-chats"] = telegram_chats(token)
                    except Exception as exc:
                        st.error(f"Telegram did not answer: {exc}")
                if right.button("Send a test message", icon=":material/send:", disabled=not (token and chat)):
                    _show(cr.check_telegram(token, chat))
                chats = st.session_state.get("tg-chats")
                if chats == []:
                    st.info("No messages yet. Open your bot in Telegram, send it any message, then press Find my chat ID again.")
                elif chats:
                    pick = st.selectbox("Chats that messaged the bot", chats,
                                        format_func=lambda c: f"{c['name']} ({c['type']}, {c['id']})")
                    if st.button("Use this chat", type="primary"):
                        cr.save_env(ENV_PATH, {"TELEGRAM_CHAT_ID": str(pick["id"])})
                        st.session_state.pop("tg-chats")
                        st.session_state["keys-version"] = version + 1
                        st.rerun()
            elif group == "binance":
                key, secret = value("BINANCE_TESTNET_API_KEY"), value("BINANCE_TESTNET_API_SECRET")
                if st.button("Test connection", key="test-binance", disabled=not (key and secret)):
                    with st.spinner("Asking the testnet..."):
                        _show(cr.check_binance(key, secret))
            elif group == "mt5":
                import importlib.util

                if importlib.util.find_spec("MetaTrader5") is None:
                    st.caption('MetaTrader 5 support is not installed: `pip install -e ".[mt5]"` (Windows only).')
                elif st.button("Test connection", key="test-mt5"):
                    with st.spinner("Talking to the terminal..."):
                        _show(cr.check_mt5())
                st.caption("Saving an account number makes the terminal log in to that account.")


# ---------------------------------------------------------------------------
def current_account() -> Account:
    return st.session_state["current_account"]


def main() -> None:
    st.set_page_config(page_title="Trading agent", page_icon=":material/monitoring:", layout="wide")
    os.chdir(PROJECT_ROOT)
    accounts = [a for a in discover_accounts() if a.cfg]
    if not accounts:
        st.error(f"No valid config*.toml file found in {PROJECT_ROOT}.")
        return

    pages = st.navigation([
        st.Page(page_accounts, title="Accounts", icon=":material/dns:", default=True),
        st.Page(page_markets, title="Markets", icon=":material/candlestick_chart:", url_path="markets"),
        st.Page(page_dashboard, title="Performance", icon=":material/monitoring:", url_path="performance"),
        st.Page(page_ask, title="Ask the agent", icon=":material/psychology:", url_path="ask"),
        st.Page(page_traders, title="Followed traders", icon=":material/groups:", url_path="traders"),
        st.Page(page_settings, title="Settings", icon=":material/settings:", url_path="settings"),
        st.Page(page_keys, title="API keys", icon=":material/key:", url_path="keys"),
    ])
    with st.sidebar:
        by_key = {a.key: a for a in accounts}
        if "account_key" not in st.session_state and st.query_params.get("account") in by_key:
            st.session_state["account_key"] = st.query_params["account"]     # e.g. ?account=config.gold
        key = st.selectbox("Account", list(by_key), format_func=lambda k: by_key[k].name, key="account_key",
                           help="Used by the Performance, Ask the agent and Followed traders pages.")
        st.session_state["current_account"] = by_key[key]
        if st.button("Refresh", icon=":material/refresh:"):
            st.cache_data.clear()
            st.rerun()
    pages.run()


main()
