# AI Trading Agent

An AI agent that trades **demo accounts** and shows you why it does what it does.

- **Crypto** on a built-in simulator with live prices, or on the Binance testnet.
- **US stocks and ETFs** on the simulator, with company news and earnings dates.
- **Gold, forex and indices** short-term, long and short, on a MetaTrader 5 demo account.

An LLM (Claude by default; OpenAI, Gemini, Ollama and others through LangChain) reads the chart
on several timeframes, the news, the economic calendar and, for crypto, how top traders are
positioned. It decides buy, sell or hold with a stop-loss; hard risk rules size every trade and
can veto it. A desktop app shows each account against buy-and-hold, every decision with its
reasoning and the agent's news briefings, and Telegram sends you the trades and briefings.

> **Demo and paper trading only.** No code path places an order on a real-money account: the
> MetaTrader 5 broker refuses anything but a demo account, and the Binance broker only talks
> to the testnet. This is a research project, not financial advice.

## Quick start

You need Python 3.11 or newer. The desktop shortcut and MetaTrader 5 are Windows only; on
macOS or Linux run the same app with `trader dashboard`.

```powershell
git clone <this repository's URL> trading-agent
cd trading-agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,dashboard,stocks]"   # add ,mt5 for MetaTrader 5
.\.venv\Scripts\trader.exe app --shortcut      # creates the "Trading Agent" shortcut
```

Open **Trading Agent**, go to the **API keys** page and enter your own credentials. Each has
a free **Test** button.

| What | Needed for | Where to get it |
|---|---|---|
| AI key (Anthropic, OpenAI or Google) | The LLM agent accounts and Ask the agent | console.anthropic.com, platform.openai.com, aistudio.google.com |
| Telegram bot token and chat ID | Alerts on your phone (optional) | @BotFather in Telegram; then press **Find my chat ID** |
| Binance testnet key and secret | `broker = "binance_testnet"` only | testnet.binance.vision |
| MetaTrader 5 login, password, server | Optional: without them the terminal's current login is used | Your broker's demo account |

The keys are saved in a `.env` file in the project folder, on your computer only; git ignores
it, so they are never uploaded. You can also copy `.env.example` to `.env` and edit it by hand.
The rule-strategy account (`config.toml`) needs no key at all.

Then press **Start** on an account on the Accounts page.

## The app

Double-click **Trading Agent** in the project folder (a shortcut to `.venv\Scripts\trading-agent.exe`).
It opens a window with these pages:

| Page | What it does |
|---|---|
| Accounts | Start and stop each account's trading loop, see equity, positions and the loop's log |
| Markets | Every coin on the exchange (or a list of stocks and ETFs) with its 1, 7 and 30 day change; tick what the account monitors and trades |
| Performance | Account against buy-and-hold, open positions, market charts, decisions, closed trades |
| Ask the agent | Ask the LLM what it would do now for chosen symbols; no order is placed |
| Followed traders | Copy-trading: who is followed, how they are positioned, update and re-rank |
| Settings | Edit any config file (validated before saving) |
| API keys | Your AI, Telegram, Binance testnet and MetaTrader 5 credentials, with a test for each |

- An account is one `config*.toml` file. Add a file to add an account.
- The selection made on the Markets page is saved per account in `data/watchlists/<config name>.json`
  and replaces the `symbols` lists in that config. A running loop picks it up within a minute;
  a position in a symbol you remove stays managed until it closes. Delete the file to go back
  to the config's own list.
- For stocks and ETFs the Markets page starts from about 60 well-known tickers; add any Yahoo
  Finance ticker with the Add box. An LLM account makes one model call per monitored symbol
  per closed candle, and the page shows that number before you save.
- Trading loops and the app server keep running after the window is closed. Stop loops on the
  Accounts page; the server only listens on this machine (`127.0.0.1:8501`).
- `trader app --desktop-shortcut` puts the shortcut on the Desktop. `trader dashboard` runs the
  same app in a terminal.
- The exe is a launcher for this project's Python environment, not a standalone program: it
  stops working if the project folder or `.venv` is moved or deleted (reinstall to fix).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,dashboard,stocks]"
.\.venv\Scripts\trader.exe app --shortcut      # creates the "Trading Agent" shortcut
```

## Usage

```powershell
.\.venv\Scripts\trader.exe fetch                 # download new candles for everything in config.toml
.\.venv\Scripts\trader.exe strategies            # list strategies and parameters
.\.venv\Scripts\trader.exe backtest              # every strategy x symbol x timeframe
.\.venv\Scripts\trader.exe backtest --strategy ema_cross --symbol BTC/USDT --timeframe 4h --trades
.\.venv\Scripts\trader.exe backtest --strategy ema_cross --param fast=10 --param slow=30 --start 2024-01-01
.\.venv\Scripts\python.exe -m pytest
```

Symbols, timeframes, exchange, fees and risk settings live in `config.toml`.

## Demo trading

```powershell
.\.venv\Scripts\trader.exe paper run             # loop: watch stops, act on each closed candle
.\.venv\Scripts\trader.exe paper run --once      # single pass
.\.venv\Scripts\trader.exe paper status          # equity, cash, open positions
.\.venv\Scripts\trader.exe paper trades          # closed trades
.\.venv\Scripts\trader.exe paper decisions       # every decision and its reason
.\.venv\Scripts\trader.exe paper reset --yes     # wipe the demo account
```

- `[paper].broker = "paper"` is the internal simulator: live prices, pretend money.
- `[paper].broker = "binance_testnet"` trades on a Binance demo account. Create keys at
  https://testnet.binance.vision and enter them on the **API keys** page
  (`BINANCE_TESTNET_API_KEY` and `BINANCE_TESTNET_API_SECRET` in `.env`). Use a separate `[paper].db_path` for it.
- Entries pass through the `[risk]` rules; rejected entries are journaled with the reason.
- Stops are checked against the live price every `poll_seconds`, and only while the loop runs.

## LLM agent

```powershell
.\.venv\Scripts\trader.exe analyze                          # what would the agent do now? (no orders)
.\.venv\Scripts\trader.exe analyze --symbol BTC/USDT --show-snapshot
.\.venv\Scripts\trader.exe analyze --snapshot-only          # the data only, no model call
```

- Enter the API key on the app's **API keys** page (or put it in `.env`, e.g. `ANTHROPIC_API_KEY=...`).
- Set `[paper].decider = "llm"` to let the agent drive demo trading. Use a separate
  `[paper].db_path` (or a second config file via `--config`) to keep its track record
  apart from the rule strategy's.
- The provider and model are set in `[llm]`; the agent runs on LangChain, so OpenAI, Gemini,
  Ollama and others work after installing the matching extra (see the table in `config.toml`).
- The model sees only numbers computed from closed candles, returns
  buy / sell / hold with a confidence, a stop distance and its reasoning, and the `[risk]`
  rules still decide whether and how much to buy.

## Dashboard and alerts

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard]"   # once
.\.venv\Scripts\trader.exe dashboard                          # opens http://localhost:8501
.\.venv\Scripts\trader.exe alerts test                        # send a test Telegram message
```

- The dashboard shows the demo account against buy-and-hold, open positions, each market with
  the agent's buys and sells, every decision with its reason, and closed trades. It reads the
  journal, so it works whether or not the trading loop is running, and warns when it is not.
- With several journal files in `data/` (for example one per decider), pick the account in the sidebar.
- Telegram alerts: create a bot with @BotFather, paste its token on the **API keys** page, send
  the bot a message and press **Find my chat ID**; set `[alerts].telegram = true`. The trading loop then sends buys, sells, stops,
  rejected entries, errors (throttled) and a daily summary.

## Copy-trading

```powershell
.\.venv\Scripts\trader.exe copy rank      # build / rebuild the followed list (a minute or two; run weekly or so)
.\.venv\Scripts\trader.exe copy update    # re-read the followed traders' positions
.\.venv\Scripts\trader.exe copy show      # who is followed and how they are positioned
```

- Source: Hyperliquid, where every account's positions and P&L history are public. No account or key needed.
- `copy rank` screens the leaderboard (account size, profitable overall and this month, not a
  market maker), then scores the candidates on monthly return, share of profitable days,
  profit relative to the worst give-back, and drawdown. Limits are in `[copytrade]`.
- With `[copytrade].enabled = true`, the LLM agent's snapshot includes how the followed traders
  are positioned in the coin (long / short / flat, average exposure, changes over the last day).
  It is one more input for the agent; nothing is copied automatically.
- Positions are re-read automatically when older than `refresh_minutes`; the followed list is
  only rebuilt when you run `copy rank`.

Two configs are set up: `config.toml` (rule strategy, `data/paper.sqlite`) and `config.llm.toml`
(LLM agent with copy-trading input, `data/paper_llm.sqlite`). Run the second with
`trader --config config.llm.toml paper run`.

## Stocks and ETFs

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[stocks]"                    # once
.\.venv\Scripts\trader.exe --config config.stocks.toml fetch
.\.venv\Scripts\trader.exe --config config.stocks.toml backtest
.\.venv\Scripts\trader.exe --config config.stocks.toml analyze
.\.venv\Scripts\trader.exe --config config.stocks.toml paper run
```

- Data: daily candles from Yahoo Finance (`[data].exchange = "yahoo"`), adjusted for splits and
  dividends. Any Yahoo ticker works, including EU listings such as `VUAA.DE`.
- One account, one currency: all symbols in a config must trade in the same currency. Use a
  separate config (and `[paper].db_path`) for EUR-listed ETFs.
- Market hours: a decision on a day's candle is taken, and filled, once the market is open
  again. Stops are only checked while the market is open, so an overnight gap can fill below the stop.
- The simulator assumes fractional shares and the fees in `[paper]`.
- The copy-trading feed is crypto-only and is off in this config.

## News and earnings

With `[news].enabled = true` the LLM agent's snapshot includes the latest headlines for the
symbol (age, source, title, short summary) and, for companies, the days until the next
earnings report. Source: Yahoo Finance, for stocks, ETFs and crypto alike.

- `earnings_blackout_days` is a hard rule: a buy this close to earnings is turned into a hold,
  whatever the model says.
- Headlines are third-party text. They are cleaned and shortened, the prompt tells the model to
  treat them as data and never as instructions, and the model can still only answer buy, sell or hold.
- `trader analyze --show-snapshot` prints everything the model is shown, news included.

### News briefing

With every decision the LLM agent also writes a short briefing: the news that matters for
that market and where it expects the price to go (up, down or unclear). It is saved with the
decision (Performance page, Decisions tab), shown on the Ask page, and sent as a `NEWS`
event to the log and Telegram every `[alerts].news_every_hours` hours per symbol, or sooner
when its view changes (at most every `news_min_hours`). Remove `"NEWS"` from
`[alerts].events` to keep it out of Telegram.

### Wider news and the economic calendar

Set in each account's `[news]` section:

- `topics`: Google News searches whose latest headlines the agent reads on every decision,
  e.g. `"war OR ceasefire OR sanctions"` for gold or `"stock market OR Wall Street"` for stocks.
- `company_news = true` (stocks): also searches news about each company by its name.
- `calendar = ["USD"]`: scheduled economic releases (inflation, jobs, central banks) for
  those currencies in the next `calendar_hours_ahead` hours, from the Forex Factory calendar.
- `event_blackout_minutes` (MT5 accounts): no new entry this close to a high-impact release.

Every source is optional input: if one cannot be read, the agent decides without it.

## Gold and forex on a MetaTrader 5 demo account

`config.gold.toml` trades `XAUUSD.s` short-term (15-minute candles), long and short, on the
demo account your MetaTrader 5 terminal is logged in to (tested with PU Prime).

```
pip install -e ".[mt5]"
trader --config config.gold.toml mt5 check     # terminal, demo account, symbol, clock; no orders
trader --config config.gold.toml analyze       # what the agent would do now; no orders
trader --config config.gold.toml paper run     # or press Start in the app
```

What you need in MetaTrader 5: the terminal running on this PC, logged in to a **demo**
account, with the **Algo Trading** button switched on. `mt5 check` tells you what is missing.

How this account differs from the crypto and stock accounts:

- **Demo only.** Before every order the agent checks that the terminal is on a demo account
  and refuses otherwise. Set `[mt5].login` to also pin it to one account number.
- **The broker holds the money.** Balance, equity and profit come from MetaTrader 5, in the
  account's currency. You see every trade in the terminal as well.
- **Long and short**, one position per symbol. The agent answers long, short, close or hold.
- **Stop-loss and take-profit sit at the broker**, so they work while the agent is stopped.
  The agent picks the stop (in ATR multiples) and the target (in multiples of the stop).
- **Lot size comes from the risk rule:** a stopped trade loses `[risk].risk_per_trade` of the
  account (0.5%). Entries are refused when the spread is wide against the stop, after the
  daily loss limit, or above `[mt5].max_lots`.
- Trades you open by hand in MetaTrader are left alone. If you close the agent's position by
  hand, the agent records it as closed.
- About 92 model calls per symbol per trading day at 15 minutes (roughly $2.30 a day with
  Claude Opus 5.5). Change `[paper].timeframe` to `1h` or `5m` to trade slower or faster.

Optional settings in `.env`: `MT5_TERMINAL_PATH`, `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`
(let the agent log the terminal in itself) and `MT5_SERVER_UTC_OFFSET_HOURS` (only if
`mt5 check` reports a different server clock).

Market hours: the agent is only asked while the market is open (no model calls at the weekend
or in the daily break; the log says when it pauses and resumes). It opens nothing in the last
`[mt5].no_entry_hours_before_weekend` hours before the weekend close and closes open positions
`[mt5].close_before_weekend_minutes` before it, so nothing is carried over the weekend gap.

## Layout

| Path | Purpose |
|---|---|
| `src/trading_agent/data/` | Candle download (ccxt for crypto, Yahoo Finance for stocks and ETFs) and SQLite cache |
| `src/trading_agent/indicators.py` | SMA, EMA, RSI, MACD, ATR, Bollinger, Donchian |
| `src/trading_agent/strategies/` | Strategy interface and baseline strategies |
| `src/trading_agent/backtest/` | Backtest engine and metrics |
| `src/trading_agent/broker/` | Broker interface, simulator, exchange testnet |
| `src/trading_agent/risk.py` | Hard risk rules and position sizing |
| `src/trading_agent/journal.py` | Positions, fills, trades, decisions, equity (SQLite) |
| `src/trading_agent/trader.py` | Demo-trading loop and the `Decider` interface |
| `src/trading_agent/agent/` | LLM decision agent (LangChain) and the market snapshot it reads |
| `src/trading_agent/copytrade/` | Hyperliquid client, trader ranking, position snapshots, consensus signal |
| `src/trading_agent/data/mt5.py` | MetaTrader 5 terminal as a data feed |
| `src/trading_agent/broker/mt5_broker.py` | Demo-only MT5 broker: orders with stop-loss and take-profit |
| `src/trading_agent/margin.py` | Long/short trading loop for the MT5 account |
| `src/trading_agent/agent/margin.py` | The agent's prompt and rules for short-term long/short trading |
| `src/trading_agent/news.py` | Headlines and next earnings date for the agent |
| `src/trading_agent/dashboard.py` | The app's pages (Streamlit) |
| `src/trading_agent/app.py` | Desktop launcher (`trading-agent.exe`) and shortcut creation |
| `src/trading_agent/runner.py` | Accounts and their background trading loops |
| `src/trading_agent/markets.py` | Performance overview of every tradable market |
| `src/trading_agent/watchlist.py` | The per-account list of monitored symbols |
| `src/trading_agent/factory.py` | Builds the trading loop and decider from a config |
| `src/trading_agent/alerts.py` | Telegram alerts |
| `src/trading_agent/credentials.py` | The user's API keys in `.env`, and their connection tests |
| `src/trading_agent/reporting.py` | Account statistics and the buy-and-hold benchmark |
| `src/trading_agent/cli.py` | `trader` command |

## Backtest rules

- Signals are computed on a bar's close and filled at the next bar's open.
- Every fill pays the fee and slippage from `config.toml`.
- Position size is set so that hitting the stop loses `risk_per_trade` of equity.
- Long-only, one symbol at a time.

## Roadmap

1. Data, indicators, backtester (done)
2. Paper trading: simulator and demo broker, risk rules, trade journal (done)
3. LLM decision agent, built on LangChain so the model provider is swappable (done)
4. Dashboard and alerts (done)
5. Copy-trading module (done)
6. Live trading

## License

MIT: see [LICENSE](LICENSE). Use at your own risk; this software trades demo accounts and gives no financial advice.
