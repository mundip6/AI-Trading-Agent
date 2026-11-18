"""Accounts (one per config file) and their background trading loops.

A loop is an ordinary `trader --config X paper run` process. Loops are found by
scanning running processes, so one started from a terminal is recognised just
like one started from the app, and nothing breaks if the app is closed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil

from trading_agent.config import Config, load_config
from trading_agent.data.exchange import market_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # editable install: <root>/src/trading_agent/
DEFAULT_CONFIG = "config.toml"


@dataclass
class Account:
    path: Path
    cfg: Config | None
    error: str = ""

    @property
    def key(self) -> str:
        return self.path.stem

    @property
    def name(self) -> str:
        if self.cfg is None:
            return f"{self.path.name} (invalid)"
        market = market_label(self.cfg.data.exchange)
        if self.cfg.data.exchange == "mt5":
            market = f"{market} demo ({', '.join(self.cfg.paper.symbols or self.cfg.data.symbols)})"
        if self.cfg.paper.decider == "llm":
            decider = f"LLM agent ({self.cfg.llm.model})"
        else:
            decider = f"{self.cfg.paper.strategy} strategy"
        return f"{market} · {decider}"

    @property
    def log_path(self) -> Path:
        return self.path.parent / "data" / "logs" / f"{self.key}.log"


def discover_accounts(root: Path = PROJECT_ROOT) -> list[Account]:
    accounts = []
    for path in sorted(root.glob("config*.toml"), key=lambda p: p.stem):
        try:
            accounts.append(Account(path, load_config(path)))
        except Exception as exc:
            accounts.append(Account(path, None, f"{type(exc).__name__}: {exc}"))
    return accounts


def _loop_config(process: psutil.Process, root: Path) -> Path | None:
    """The config file a process is trading with, or None if it is not a trading loop."""
    try:
        args = process.cmdline()
    except (psutil.Error, OSError):
        return None
    if not any("trading_agent" in a or Path(a).stem.lower() == "trader" for a in args[:4]):
        return None
    if not any(a == "paper" and args[i + 1 : i + 2] == ["run"] for i, a in enumerate(args)):
        return None
    if "--once" in args:
        return None
    config = Path(args[args.index("--config") + 1]) if "--config" in args[:-1] else Path(DEFAULT_CONFIG)
    if not config.is_absolute():
        try:
            config = Path(process.cwd()) / config
        except (psutil.Error, OSError):
            config = root / config
    return config.resolve()


def running_loops(root: Path = PROJECT_ROOT) -> dict[Path, list[psutil.Process]]:
    """Trading-loop processes grouped by the config file they use."""
    loops: dict[Path, list[psutil.Process]] = {}
    for process in psutil.process_iter():
        config = _loop_config(process, root)
        if config is not None:
            loops.setdefault(config, []).append(process)
    return loops


def is_running(account: Account, loops: dict[Path, list[psutil.Process]] | None = None) -> bool:
    loops = running_loops(account.path.parent) if loops is None else loops
    return account.path.resolve() in loops


def _python() -> str:
    """The console interpreter of this environment (the app itself may run under pythonw)."""
    exe = Path(sys.executable)
    candidate = exe.with_name("python.exe" if os.name == "nt" else "python")
    return str(candidate if candidate.exists() else exe)


def detached_popen(command: list[str], cwd: Path, log_path: Path) -> subprocess.Popen:
    """Start a background process with no window that outlives the caller; output goes to a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with log_path.open("ab") as log:
        return subprocess.Popen(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, creationflags=flags, env=env)


def start_loop(account: Account) -> bool:
    """Start the account's trading loop unless one is already running."""
    if is_running(account):
        return False
    command = [_python(), "-u", "-m", "trading_agent", "--config", str(account.path), "paper", "run"]
    detached_popen(command, account.path.parent, account.log_path)
    return True


def stop_loop(account: Account) -> int:
    """Stop every loop process of the account. Returns how many were stopped."""
    processes = running_loops(account.path.parent).get(account.path.resolve(), [])
    for process in processes:
        try:
            process.terminate()
        except psutil.Error:
            pass
    psutil.wait_procs(processes, timeout=5)
    return len(processes)


def tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - 16_384))
        text = fh.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])
