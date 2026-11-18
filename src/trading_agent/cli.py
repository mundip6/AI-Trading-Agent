from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trading_agent.alerts import Alerter
from trading_agent.backtest import compute_metrics, run_backtest
from trading_agent.broker import make_broker
from trading_agent.config import Config, load_config
from trading_agent.data import CandleStore, sync_candles
from trading_agent.data.exchange import make_exchange, market_label
from trading_agent.journal import Journal
from trading_agent.strategies import REGISTRY, build_strategy
from trading_agent.factory import build_decider as _build_decider
from trading_agent.factory import build_trader as _build_trader
from trading_agent.trader import LOOKBACK_BARS

PCT_COLUMNS = [
    "total_return", "cagr", "max_drawdown", "win_rate", "avg_trade",
    "exposure", "hold_return", "hold_max_drawdown",
]


def cmd_fetch(args: argparse.Namespace, cfg: Config) -> int:
    symbols = args.symbol or cfg.data.symbols
    timeframes = args.timeframe or cfg.data.timeframes
    since = args.since or cfg.data.since
    store = CandleStore(cfg.data.db_path)
    exchange = make_exchange(cfg.data.exchange)
    failed = 0
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                added = sync_candles(store, cfg.data.exchange, symbol, timeframe, since, exchange)
            except Exception as exc:  # network / exchange errors: report and keep going
                failed += 1
                print(f"{symbol} {timeframe}: FAILED - {type(exc).__name__}: {exc}")
                continue
            total = len(store.load(cfg.data.exchange, symbol, timeframe))
            print(f"{symbol} {timeframe}: +{added} candles ({total} stored)")
    store.close()
    return 1 if failed else 0


def cmd_strategies(args: argparse.Namespace, cfg: Config) -> int:
    for name, cls in REGISTRY.items():
        defaults = ", ".join(f"{k}={v}" for k, v in vars(cls()).items())
        print(f"{name}\n  {cls.description}\n  params: {defaults}")
    return 0


def cmd_backtest(args: argparse.Namespace, cfg: Config) -> int:
    params = dict(p.split("=", 1) for p in args.param)
    names = list(REGISTRY) if args.strategy == "all" else [args.strategy]
    if params and len(names) > 1:
        print("--param needs a single --strategy", file=sys.stderr)
        return 2
    symbols = args.symbol or cfg.data.symbols
    timeframes = args.timeframe or cfg.data.timeframes

    store = CandleStore(cfg.data.db_path)
    rows = []
    last_trades = None
    for symbol in symbols:
        for timeframe in timeframes:
            df = store.load(cfg.data.exchange, symbol, timeframe, args.start, args.end)
            if len(df) < 2:
                print(f"No candles for {symbol} {timeframe} - run `trader fetch` first", file=sys.stderr)
                continue
            for name in names:
                strategy = build_strategy(name, params)
                result = run_backtest(df, strategy.generate(df), cfg.backtest)
                rows.append(
                    {"symbol": symbol, "tf": timeframe, "strategy": name,
                     **compute_metrics(result, df, cfg.backtest)}
                )
                last_trades = result.trades
    store.close()
    if not rows:
        return 1

    table = pd.DataFrame(rows)
    period = f"{args.start or 'start'} -> {args.end or 'now'}"
    print(f"Backtest {period} | fee {cfg.backtest.fee_rate:.2%} | slippage {cfg.backtest.slippage:.2%}\n")
    formatters = {c: "{:+.1%}".format for c in PCT_COLUMNS}
    formatters |= {c: "{:.2f}".format for c in ("sharpe", "profit_factor", "hold_sharpe")}
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.to_string(index=False, formatters=formatters))

    if args.trades and len(rows) == 1:
        print("\nTrades:")
        print(last_trades.to_string(index=False))
    return 0


def cmd_analyze(args: argparse.Namespace, cfg: Config) -> int:
    """Ask the LLM agent about the latest closed candle. Places no orders."""
    from trading_agent.agent import build_snapshot

    timeframe = cfg.paper.timeframe
    symbols = args.symbol or cfg.paper.symbols or cfg.data.symbols
    store = CandleStore(cfg.data.db_path)
    exchange = make_exchange(cfg.data.exchange)
    positions = Journal(cfg.paper.db_path).positions()
    decider = None if args.snapshot_only else _build_decider(cfg, force_llm=True)

    for symbol in symbols:
        sync_candles(store, cfg.data.exchange, symbol, timeframe, cfg.data.since, exchange)
        lookback = cfg.mt5.lookback_bars if cfg.paper.broker == "mt5" else LOOKBACK_BARS
        df = store.load(cfg.data.exchange, symbol, timeframe).tail(lookback)
        if len(df) < 2:
            print(f"{symbol}: no candles")
            continue
        position = positions.get(symbol)
        print(f"=== {symbol} {timeframe}, candle closed {df.index[-1] + pd.Timedelta(timeframe):%Y-%m-%d %H:%M} UTC ===")
        if decider is not None and args.show_snapshot:
            print(json.dumps(decider.snapshot(symbol, df, position)[0], indent=1))   # with news etc.
        elif args.snapshot_only:
            print(json.dumps(build_snapshot(symbol, timeframe, df, position)[0], indent=1))
        if decider is None:
            continue
        decision = decider.decide(symbol, df, position)
        stop = f", stop distance {decision.stop_distance:.6g}" if decision.stop_distance else ""
        if decision.take_profit_distance:
            stop += f", target distance {decision.take_profit_distance:.6g}"
        print(f"{decision.action.upper()}{stop}\n{decision.reason}")
        if decision.news:
            print(f"News: {decision.news}\nOutlook: {decision.outlook}")
        print()
    store.close()
    return 0


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def cmd_paper_run(args: argparse.Namespace, cfg: Config) -> int:
    trader = _build_trader(cfg)
    market = {"Stocks and ETFs": "Stocks", "MetaTrader 5": "MT5 demo"}.get(
        market_label(cfg.data.exchange), "Crypto")
    try:
        alerter = Alerter.from_config(cfg.alerts, label=f"{market} · {trader.decider.name}")
    except RuntimeError as exc:      # Telegram switched on but no keys yet: trade without alerts
        print(f"[{_stamp()}] {exc}")
        alerter = Alerter(cfg.alerts, None)
    symbols = ", ".join(trader.symbols)
    print(f"[{_stamp()}] Demo trading on '{trader.broker.name}': {trader.decider.name} "
          f"{trader.timeframe} on {symbols}. Telegram alerts {'on' if alerter.enabled else 'off'}. "
          "Ctrl+C to stop.")
    try:
        while True:
            try:
                events = trader.tick()
            except Exception as exc:  # network hiccups etc.: report, retry on the next pass
                events = [f"ERROR {type(exc).__name__}: {exc}"]
            for event in events:
                print(f"[{_stamp()}] {event}", flush=True)
            alerter.handle_events(events)
            if alerter.enabled and cfg.alerts.daily_summary:
                try:
                    alerter.maybe_daily_summary(trader.journal, trader.equity(), int(time.time() * 1000))
                except Exception as exc:
                    print(f"[{_stamp()}] Daily summary failed: {type(exc).__name__}: {exc}", flush=True)
            if args.once:
                print(f"[{_stamp()}] equity {trader.equity():,.2f}")
                return 0
            time.sleep(cfg.paper.poll_seconds)
    except KeyboardInterrupt:
        kept = ("their stop-loss and take-profit stay active at the broker" if cfg.paper.broker == "mt5"
                else "stops are NOT watched while stopped")
        print(f"\n[{_stamp()}] Stopped. Open positions are kept; {kept}.")
        return 0


def cmd_paper_status(args: argparse.Namespace, cfg: Config) -> int:
    journal = Journal(cfg.paper.db_path)
    if journal.get("broker") is None:
        print("No demo account yet - run `trader paper run` first")
        return 0
    broker = make_broker(cfg, journal)
    positions = journal.positions()
    if cfg.paper.broker == "mt5":
        return _mt5_status(broker, journal, positions)
    rows, exposure = [], 0.0
    for symbol, pos in positions.items():
        price = broker.price(symbol)
        value = pos.qty * price
        exposure += value
        rows.append({
            "symbol": symbol, "qty": pos.qty, "entry": pos.entry_price, "price": price,
            "stop": pos.stop, "value": value, "pnl": value - pos.entry_cost,
            "pnl_pct": f"{value / pos.entry_cost - 1:+.1%}", "source": pos.source,
        })
    cash = broker.cash()
    equity = cash + exposure
    initial = float(journal.get("initial_equity", equity))
    trades = journal.trades_frame()

    print(f"Broker   {journal.get('broker')}")
    print(f"Equity   {equity:,.2f}  ({equity / initial - 1:+.2%} since start at {initial:,.2f})")
    print(f"Cash     {cash:,.2f}")
    print(f"Invested {exposure:,.2f}  ({exposure / equity:.0%} of equity)")
    if len(trades):
        wins = (trades["pnl"] > 0).sum()
        print(f"Closed   {len(trades)} trades, {wins} winners, realised P&L {trades['pnl'].sum():+,.2f}")
    else:
        print("Closed   0 trades")
    print("\nOpen positions:")
    print(pd.DataFrame(rows).to_string(index=False) if rows else "  none")
    return 0


def _mt5_status(broker, journal: Journal, positions: dict) -> int:
    account = broker.account()
    initial = float(journal.get("initial_equity", account.equity))
    trades = journal.trades_frame()
    rows = []
    for symbol, pos in positions.items():
        live = broker.position(pos.ticket)
        rows.append({"symbol": symbol, "side": pos.side, "lots": pos.lots, "entry": pos.entry_price,
                     "price": broker.price(symbol), "stop": pos.stop, "target": pos.take_profit,
                     "pnl": live.profit if live else float("nan"), "source": pos.source})
    print(f"Broker   {journal.get('broker')} on {account.server} (demo), leverage 1:{account.leverage}")
    print(f"Equity   {account.equity:,.2f} {account.currency}  "
          f"({account.equity / initial - 1:+.2%} since start at {initial:,.2f})")
    print(f"Balance  {account.balance:,.2f}   free margin {account.free_margin:,.2f}")
    wins = (trades["pnl"] > 0).sum() if len(trades) else 0
    realised = trades["pnl"].sum() if len(trades) else 0.0
    print(f"Closed   {len(trades)} trades, {wins} winners, realised P&L {realised:+,.2f}")
    print("\nOpen positions:")
    print(pd.DataFrame(rows).to_string(index=False) if rows else "  none")
    return 0


def cmd_mt5_check(args: argparse.Namespace, cfg: Config) -> int:
    """Everything the account needs from the terminal, checked one by one. Places no orders."""
    from trading_agent.data.mt5 import MT5Error, MT5Exchange, shared_terminal

    terminal = shared_terminal()
    symbols = args.symbol or cfg.paper.symbols or cfg.data.symbols
    problems = 0

    def report(ok: bool, text: str) -> None:
        nonlocal problems
        problems += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {text}")

    try:
        terminal.connect()
        info = terminal.call("terminal_info")
        account = terminal.account()
    except MT5Error as exc:
        report(False, str(exc))
        return 1
    api = terminal.api
    report(True, f"Terminal: {info.name} ({info.company})")
    report(bool(info.connected), "Terminal is connected to the broker server")
    demo = account.trade_mode == api.ACCOUNT_TRADE_MODE_DEMO
    report(demo, f"Account {account.login} on {account.server} is a demo account" if demo else
           f"Account {account.login} on {account.server} is NOT a demo account: the agent will refuse to trade it")
    if demo:
        print(f"     balance {account.balance:,.2f} {account.currency}, leverage 1:{account.leverage}")
    report(bool(info.trade_allowed), "Algo Trading is switched on in the terminal" if info.trade_allowed else
           "Algo Trading is switched off: press the Algo Trading button in the MetaTrader toolbar")
    report(bool(account.trade_expert), "The account allows automated trading")

    exchange = MT5Exchange(terminal)
    for symbol in symbols:
        try:
            spec = terminal.symbol(symbol)
            tick = terminal.tick(symbol)
            since = int(time.time() * 1000) - 20 * exchange.parse_timeframe(cfg.paper.timeframe) * 1000
            candles = exchange.fetch_ohlcv(symbol, cfg.paper.timeframe, since=since)
        except (MT5Error, ValueError) as exc:
            report(False, f"{symbol}: {exc}")
            continue
        digits = spec.digits
        report(True, f"{symbol}: bid {tick.bid:.{digits}f} ask {tick.ask:.{digits}f}, "
               f"spread {tick.ask - tick.bid:.{digits}f}, 1 lot = {spec.trade_contract_size:g} units, "
               f"lots {spec.volume_min:g} to {spec.volume_max:g} in steps of {spec.volume_step:g}")
        report(spec.trade_mode == api.SYMBOL_TRADE_MODE_FULL, f"{symbol}: trading is enabled for this symbol")
        is_open = exchange.is_market_open(symbol)
        print(f"     market is {'open' if is_open else 'closed right now'}; "
              f"{len(candles)} recent {cfg.paper.timeframe} candles read")

    try:
        measured = terminal.measured_utc_offset(list(symbols))
    except MT5Error:
        measured = None
    used = terminal.utc_offset()
    if measured is None:
        print(f"     Server clock: assuming UTC{used / 3600:+g}h (no live quote to verify it; "
              "check again when the market is open)")
    else:
        report(measured == used, f"Server clock is UTC{measured / 3600:+g}h" if measured == used else
               f"Server clock is UTC{measured / 3600:+g}h but UTC{used / 3600:+g}h is assumed: "
               f"add MT5_SERVER_UTC_OFFSET_HOURS={measured / 3600:g} to .env")
    print("\nReady for demo trading." if not problems else f"\n{problems} problem(s) to fix before starting.")
    return 1 if problems else 0


def cmd_paper_trades(args: argparse.Namespace, cfg: Config) -> int:
    trades = Journal(cfg.paper.db_path).trades_frame()
    print(trades.to_string(index=False) if len(trades) else "No closed trades yet")
    return 0


def cmd_paper_decisions(args: argparse.Namespace, cfg: Config) -> int:
    decisions = Journal(cfg.paper.db_path).decisions_frame(args.limit)
    with pd.option_context("display.width", 250, "display.max_colwidth", 600):
        print(decisions.to_string(index=False) if len(decisions) else "No decisions yet")
    return 0


def cmd_paper_reset(args: argparse.Namespace, cfg: Config) -> int:
    if not args.yes:
        print(f"This deletes the whole demo account history in {cfg.paper.db_path}. Re-run with --yes.")
        return 2
    Journal(cfg.paper.db_path).reset()
    print("Demo account reset")
    return 0


def _print_followed(copy, cfg: Config) -> None:
    traders = copy.store.traders()
    if not traders:
        print("No traders followed yet - run `trader copy rank` first")
        return
    table = pd.DataFrame([
        {"trader": t.name or t.address[:10] + "...", "score": t.score, "account": t.account_value,
         "month_roi": t.month_roi, "profitable_days": t.consistency, "max_drawdown": t.max_drawdown,
         "profit_to_pain": t.profit_to_pain, "history_days": t.history_days}
        for t in traders
    ])
    formatters = {"account": "{:,.0f}".format, "month_roi": "{:+.1%}".format, "profitable_days": "{:.0%}".format,
                  "max_drawdown": "{:.1%}".format, "profit_to_pain": "{:.1f}".format,
                  "history_days": "{:.0f}".format, "score": "{:.2f}".format}
    ranked = datetime.fromtimestamp(copy.store.ranked_at() / 1000, timezone.utc)
    print(f"Followed traders (ranked {ranked:%Y-%m-%d %H:%M} UTC):")
    with pd.option_context("display.width", 250):
        print(table.to_string(index=False, formatters=formatters))

    rows = []
    for symbol in cfg.paper.symbols or cfg.data.symbols:
        c = copy.consensus(symbol)
        if c:
            changes = c["changes_last_24h"] or {}
            moved = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in changes.items() if k != "hours_compared" and v)
            rows.append({"symbol": symbol, "long": c["long"], "short": c["short"], "flat": c["no_position"],
                         "avg_exposure": f"{c['avg_exposure_pct_of_account']:+.1f}%",
                         "age_min": c["data_age_minutes"],
                         "changes": moved or ("none" if changes else "n/a (first snapshot)")})
    if rows:
        print("\nTheir positioning:")
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("\nNo position snapshot yet - run `trader copy update`")


def cmd_copy(args: argparse.Namespace, cfg: Config) -> int:
    from trading_agent.copytrade import CopyTrading

    copy = CopyTrading(cfg.copytrade)
    if args.copy_command == "rank":
        followed = copy.rerank(progress=lambda line: print(line, flush=True))
        print(f"Following {len(followed)} traders; reading their positions...")
        copy.update()
    elif args.copy_command == "update":
        print(f"Read positions of {copy.update()} traders")
    _print_followed(copy, cfg)
    return 0


def cmd_alerts_chat_id(args: argparse.Namespace, cfg: Config) -> int:
    import os
    import urllib.error

    from trading_agent.alerts import telegram_chats

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Add TELEGRAM_BOT_TOKEN on the app's API keys page (or in .env) first", file=sys.stderr)
        return 1
    try:
        chats = telegram_chats(token)
    except urllib.error.HTTPError as exc:
        print("Telegram rejected the token" if exc.code in (401, 404) else f"Telegram error {exc.code}",
              file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Could not reach Telegram: {getattr(exc, 'reason', exc)}", file=sys.stderr)
        return 1
    if not chats:
        print("No messages found. Open your bot in Telegram, press Start (or send it any message), then run this again.")
        return 1
    for chat in chats:
        print(f"{chat['name'] or '(no name)'} ({chat['type']}): TELEGRAM_CHAT_ID={chat['id']}")
    return 0


def cmd_alerts_test(args: argparse.Namespace, cfg: Config) -> int:
    from trading_agent.alerts import telegram_sender_from_env

    try:
        sender = telegram_sender_from_env()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    alerter = Alerter(cfg.alerts, sender, label="test")
    ok = alerter.send("Trading agent: alerts are working.")
    print("Test message sent" if ok else "Test message was NOT delivered")
    return 0 if ok else 1


def cmd_dashboard(args: argparse.Namespace, cfg: Config) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print('The dashboard needs extra packages: pip install -e ".[dashboard]"', file=sys.stderr)
        return 1
    from trading_agent.app import server_command

    command = server_command(args.port, headless=False)
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 0


def cmd_app(args: argparse.Namespace, cfg: Config) -> int:
    from trading_agent import app

    if args.shortcut or args.desktop_shortcut:
        folder = Path.home() / "Desktop" if args.desktop_shortcut else app.PROJECT_ROOT
        print(f"Created {app.create_shortcut(folder)}")
        return 0
    return app.main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trader", description="Swing-trading research agent")
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_market_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--symbol", action="append", help="e.g. BTC/USDT (repeatable; default: config)")
        p.add_argument("--timeframe", action="append", help="e.g. 4h (repeatable; default: config)")

    fetch = sub.add_parser("fetch", help="download candles that are not stored yet")
    add_market_args(fetch)
    fetch.add_argument("--since", help="first date for a new download, e.g. 2021-01-01")
    fetch.set_defaults(func=cmd_fetch)

    strategies = sub.add_parser("strategies", help="list strategies and their parameters")
    strategies.set_defaults(func=cmd_strategies)

    backtest = sub.add_parser("backtest", help="run strategies on stored candles")
    add_market_args(backtest)
    backtest.add_argument("--strategy", default="all", choices=[*REGISTRY, "all"])
    backtest.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    backtest.add_argument("--start", help="first date (inclusive)")
    backtest.add_argument("--end", help="last date (exclusive)")
    backtest.add_argument("--trades", action="store_true", help="print the trade list (single run only)")
    backtest.set_defaults(func=cmd_backtest)

    analyze = sub.add_parser("analyze", help="ask the LLM agent what it would do now (no orders)")
    analyze.add_argument("--symbol", action="append", help="default: the demo-trading symbols")
    analyze.add_argument("--show-snapshot", action="store_true", help="also print the data sent to the model")
    analyze.add_argument("--snapshot-only", action="store_true", help="print the data without calling the model")
    analyze.set_defaults(func=cmd_analyze)

    app = sub.add_parser("app", help="open the desktop app")
    app.add_argument("--shortcut", action="store_true", help="create a shortcut in the project folder")
    app.add_argument("--desktop-shortcut", action="store_true", help="create a shortcut on the Desktop")
    app.set_defaults(func=cmd_app)

    dashboard = sub.add_parser("dashboard", help="run the app server in this terminal")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.set_defaults(func=cmd_dashboard)

    copy = sub.add_parser("copy", help="copy-trading: follow top traders").add_subparsers(dest="copy_command", required=True)
    copy.add_parser("rank", help="rebuild the list of followed traders (takes a minute or two)").set_defaults(func=cmd_copy)
    copy.add_parser("update", help="re-read the followed traders' positions").set_defaults(func=cmd_copy)
    copy.add_parser("show", help="followed traders and their positioning").set_defaults(func=cmd_copy)

    alerts = sub.add_parser("alerts", help="alert tools").add_subparsers(dest="alerts_command", required=True)
    alerts.add_parser("test", help="send a test Telegram message").set_defaults(func=cmd_alerts_test)
    alerts.add_parser("chat-id", help="show the chat ID of whoever messaged the bot").set_defaults(func=cmd_alerts_chat_id)

    mt5 = sub.add_parser("mt5", help="MetaTrader 5 tools").add_subparsers(dest="mt5_command", required=True)
    check = mt5.add_parser("check", help="check the terminal, the demo account and the symbols (no orders)")
    check.add_argument("--symbol", action="append", help="default: the symbols of the account")
    check.set_defaults(func=cmd_mt5_check)

    paper = sub.add_parser("paper", help="demo trading").add_subparsers(dest="paper_command", required=True)
    run = paper.add_parser("run", help="start the demo-trading loop")
    run.add_argument("--once", action="store_true", help="do a single pass and exit")
    run.set_defaults(func=cmd_paper_run)
    paper.add_parser("status", help="equity, cash and open positions").set_defaults(func=cmd_paper_status)
    paper.add_parser("trades", help="closed trades").set_defaults(func=cmd_paper_trades)
    decisions = paper.add_parser("decisions", help="recent decisions and their reasons")
    decisions.add_argument("--limit", type=int, default=30)
    decisions.set_defaults(func=cmd_paper_decisions)
    reset = paper.add_parser("reset", help="wipe the demo account")
    reset.add_argument("--yes", action="store_true")
    reset.set_defaults(func=cmd_paper_reset)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args, load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
