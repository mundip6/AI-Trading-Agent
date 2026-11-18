"""Alerts for trading events. A failed alert never interrupts trading."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Callable

from trading_agent.config import AlertsConfig
from trading_agent.journal import DAY_MS, Journal

Sender = Callable[[str], None]


def telegram_sender(token: str, chat_id: str) -> Sender:
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def send(text: str) -> None:
        body = json.dumps({"chat_id": chat_id, "text": text[:4000]}).encode()
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()

    return send


def telegram_chats(token: str) -> list[dict]:
    """Chats that have recently messaged the bot: [{'id', 'name', 'type'}], newest last."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    with urllib.request.urlopen(url, timeout=15) as response:
        updates = json.loads(response.read())["result"]
    chats = {}
    for update in updates:
        message = update.get("message") or update.get("channel_post") or update.get("my_chat_member") or {}
        chat = message.get("chat")
        if chat:
            name = chat.get("title") or " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")]))
            chats[chat["id"]] = {"id": chat["id"], "name": name or chat.get("username", ""), "type": chat.get("type", "")}
    return list(chats.values())


def telegram_sender_from_env() -> Sender:
    token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Telegram alerts are on but TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are not set: add them on the app's API keys page (or in .env), or set telegram = false")
    return telegram_sender(token, chat_id)


class Alerter:
    def __init__(self, cfg: AlertsConfig, sender: Sender | None, label: str = ""):
        self.cfg = cfg
        self._sender = sender
        self._prefix = f"[{label}] " if label else ""
        self._last_sent: dict[str, float] = {}

    @classmethod
    def from_config(cls, cfg: AlertsConfig, label: str = "") -> "Alerter":
        return cls(cfg, telegram_sender_from_env() if cfg.telegram else None, label)

    @property
    def enabled(self) -> bool:
        return self._sender is not None

    def send(self, text: str) -> bool:
        if self._sender is None:
            return False
        try:
            self._sender(self._prefix + text)
            return True
        except Exception as exc:
            # str(exc) only: a urllib error repr can include the URL, which contains the bot token
            print(f"Alert not delivered: {type(exc).__name__}: {getattr(exc, 'reason', '') or getattr(exc, 'code', '')}")
            return False

    def handle_events(self, events: list[str], now: float | None = None) -> int:
        """Send the events whose kind is enabled. Repeats of the same error are throttled."""
        now = time.time() if now is None else now
        sent = 0
        for event in events:
            kind = event.split(" ", 1)[0]
            if kind not in self.cfg.events:
                continue
            if kind == "ERROR":
                last = self._last_sent.get(event)
                if last is not None and now - last < self.cfg.error_cooldown_minutes * 60:
                    continue
                self._last_sent[event] = now
            sent += self.send(event)
        return sent

    def maybe_daily_summary(self, journal: Journal, equity: float, now_ms: int) -> bool:
        """Once per UTC day: yesterday's result and what is open."""
        if not (self.enabled and self.cfg.daily_summary):
            return False
        day = str(now_ms // DAY_MS)
        last = journal.get("last_summary_day")
        if last == day:
            return False
        journal.set("last_summary_day", day)
        if last is None:
            return False   # first run: nothing to summarise yet

        start = journal.day_start_equity(now_ms - DAY_MS) or equity
        initial = float(journal.get("initial_equity", equity))
        positions = journal.positions()
        held = ", ".join(positions) if positions else "none"
        return self.send(
            f"Daily summary\nEquity {equity:,.2f} ({equity / start - 1:+.2%} on the day, "
            f"{equity / initial - 1:+.2%} since start)\nOpen positions: {held}"
        )
