"""The user's own credentials: API keys for the AI provider, Telegram, the Binance testnet and MetaTrader 5.

They live in the `.env` file next to the config files (never committed: see .gitignore) and are
entered on the app's API keys page or by editing the file. Running loops read them at start.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Credential:
    name: str           # environment variable
    label: str
    group: str
    secret: bool = True
    help: str = ""


GROUPS = {
    "ai": ("AI model", "The LLM agent uses the provider set in `[llm].provider` of the account's config. "
                       "One key is enough."),
    "telegram": ("Telegram alerts", "Create a bot with @BotFather in Telegram, paste its token, send the bot "
                                    "any message, then press Find my chat ID."),
    "binance": ("Binance testnet", "Demo keys from testnet.binance.vision (spot testnet). Real-money keys "
                                   "are never used: the agent only talks to the testnet."),
    "mt5": ("MetaTrader 5", "All optional. Without them the agent uses whatever account the MetaTrader 5 "
                            "terminal on this PC is logged in to. Only demo accounts are traded."),
}

CREDENTIALS = [
    Credential("ANTHROPIC_API_KEY", "Anthropic API key", "ai", help="console.anthropic.com → API keys"),
    Credential("OPENAI_API_KEY", "OpenAI API key", "ai", help="platform.openai.com → API keys"),
    Credential("GOOGLE_API_KEY", "Google Gemini API key", "ai", help="aistudio.google.com → Get API key"),
    Credential("TELEGRAM_BOT_TOKEN", "Bot token", "telegram", help="From @BotFather, like 123456:ABC-DEF..."),
    Credential("TELEGRAM_CHAT_ID", "Chat ID", "telegram", secret=False,
               help="Where alerts go. Use Find my chat ID below to fill it in."),
    Credential("BINANCE_TESTNET_API_KEY", "API key", "binance"),
    Credential("BINANCE_TESTNET_API_SECRET", "API secret", "binance"),
    Credential("MT5_LOGIN", "Account number", "mt5", secret=False),
    Credential("MT5_PASSWORD", "Password", "mt5"),
    Credential("MT5_SERVER", "Server", "mt5", secret=False, help="As shown in the terminal, e.g. PUPrime-Demo"),
    Credential("MT5_TERMINAL_PATH", "Terminal path", "mt5", secret=False,
               help="Only if MetaTrader 5 is installed in an unusual folder: the full path of terminal64.exe"),
]

BY_NAME = {c.name: c for c in CREDENTIALS}
KEY_FOR_PROVIDER = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google_genai": "GOOGLE_API_KEY"}


# --- the .env file ----------------------------------------------------------
def _key_of(line: str) -> str | None:
    """The variable a line sets, also for a commented-out example line like '# KEY=' (None otherwise)."""
    text = line.strip().lstrip("#").strip()
    if "=" not in text:
        return None
    key = text.split("=", 1)[0].strip()
    return key if key.replace("_", "").isalnum() and key.upper() == key else None


def read_env(path: Path) -> dict[str, str]:
    """The values set in the file (commented lines are ignored)."""
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#"):
                continue
            key = _key_of(line)
            if key:
                values[key] = line.split("=", 1)[1].strip().strip("\"'")
    return values


def _quote(value: str) -> str:
    return f'"{value}"' if any(ch in value for ch in " #'\"") else value


def save_env(path: Path, updates: dict[str, str | None], environ=os.environ) -> None:
    """Set (or, with None or "", remove) variables in the file, keeping every other line and comment.

    The current process sees the change too, so loops started from the app afterwards get the new values.
    """
    for key, value in updates.items():
        if value and ("\n" in value or "\r" in value):
            raise ValueError(f"{key} cannot contain a line break")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    values = {k: (v.strip() if v else "") for k, v in updates.items()}
    out, written = [], set()
    for line in lines:
        key = _key_of(line)
        if key not in values:
            out.append(line)
            continue
        if line.strip().startswith("#"):
            out.append(line)                    # keep an example line; the value goes right after it
        if values[key] and key not in written:
            out.append(f"{key}={_quote(values[key])}")
            written.add(key)
        # an old value line is dropped: replaced above, or removed
    out.extend(f"{key}={_quote(value)}" for key, value in values.items() if value and key not in written)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(path)
    for key, value in values.items():
        if value:
            environ[key] = value
        else:
            environ.pop(key, None)


def describe(credential: Credential, value: str | None) -> str:
    """How a saved value is shown: secrets only by their last four characters."""
    if not value:
        return "not set"
    if not credential.secret:
        return value
    return f"set, ends in …{value[-4:]}" if len(value) >= 12 else "set"


# --- connection checks (free: none of them places an order or spends model tokens) ---------------
Check = tuple[bool, str]


def _get(url: str, headers: dict, timeout: float = 15) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "trading-agent", **headers})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def check_ai(name: str, key: str, get: Callable = _get) -> Check:
    """List the provider's models with the key: proves the key works without a paid call."""
    endpoints = {
        "ANTHROPIC_API_KEY": ("https://api.anthropic.com/v1/models?limit=1",
                              {"x-api-key": key, "anthropic-version": "2023-06-01"}),
        "OPENAI_API_KEY": ("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"}),
        "GOOGLE_API_KEY": ("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
                           {"x-goog-api-key": key}),
    }
    url, headers = endpoints[name]
    try:
        get(url, headers)
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 401, 403):
            return False, "The provider rejected this key."
        return False, f"The provider answered with error {exc.code}."
    except OSError as exc:
        return False, f"Could not reach the provider: {getattr(exc, 'reason', exc)}"
    return True, "The key works."


def check_telegram(token: str, chat_id: str, send: Callable | None = None) -> Check:
    from trading_agent.alerts import telegram_sender

    try:
        (send or telegram_sender(token, chat_id))("Trading agent: test message. Alerts will arrive here.")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 404):
            return False, "Telegram rejected the bot token."
        if exc.code == 400:
            return False, "Telegram does not know this chat ID, or you have not messaged the bot yet."
        return False, f"Telegram error {exc.code}."
    except OSError as exc:
        return False, f"Could not reach Telegram: {getattr(exc, 'reason', exc)}"
    return True, "Test message sent: check Telegram."


def check_binance(key: str, secret: str) -> Check:
    import ccxt

    try:
        exchange = ccxt.binance({"apiKey": key, "secret": secret, "enableRateLimit": True})
        exchange.set_sandbox_mode(True)
        balance = exchange.fetch_balance()
    except ccxt.AuthenticationError:
        return False, "The testnet rejected these keys (are they from testnet.binance.vision?)."
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    usdt = balance.get("total", {}).get("USDT")
    return True, "The keys work." + (f" Testnet balance: {usdt:,.2f} USDT." if usdt is not None else "")


def check_mt5(environ=os.environ) -> Check:
    from trading_agent.data.mt5 import MT5Error, MT5Terminal

    terminal = MT5Terminal(env=environ)
    try:
        account = terminal.account()
    except (MT5Error, ValueError) as exc:
        return False, str(exc)
    demo = account.trade_mode == terminal.api.ACCOUNT_TRADE_MODE_DEMO
    text = f"Connected to account {account.login} on {account.server}, {account.balance:,.2f} {account.currency}."
    return demo, text + (" It is a demo account." if demo else " It is NOT a demo account: the agent will refuse to trade it.")
