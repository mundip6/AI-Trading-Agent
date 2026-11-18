"""Desktop launcher: starts the local app server if needed and opens it in its own window.

Installed as the windowless `trading-agent.exe` (see [project.gui-scripts]).
The server and the trading loops keep running after the window is closed.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from trading_agent.runner import PROJECT_ROOT, _python, detached_popen

PORT = 8501
APP_SCRIPT = Path(__file__).with_name("dashboard.py")


def server_command(port: int = PORT, headless: bool = True) -> list[str]:
    return [
        _python(), "-m", "streamlit", "run", str(APP_SCRIPT),
        "--server.port", str(port),
        "--server.address", "127.0.0.1",          # this machine only: the app can start and stop trading
        "--server.headless", "true" if headless else "false",
        "--browser.gatherUsageStats", "false",
        "--client.toolbarMode", "minimal",
    ]


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def ensure_server(port: int = PORT, timeout: float = 45.0) -> bool:
    if port_open(port):
        return True
    detached_popen(server_command(port), PROJECT_ROOT, PROJECT_ROOT / "data" / "logs" / "app.log")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_open(port):
            return True
        time.sleep(0.5)
    return False


def _browser() -> str | None:
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LocalAppData", "")) / "Google/Chrome/Application/chrome.exe",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return shutil.which("msedge") or shutil.which("chrome")


def open_window(port: int = PORT) -> None:
    url = f"http://localhost:{port}"
    browser = _browser()
    if browser:   # --app gives a plain window without tabs or an address bar
        subprocess.Popen([browser, f"--app={url}", "--window-size=1440,960"])
    else:
        webbrowser.open(url)


def create_shortcut(folder: Path) -> Path:
    """Create a 'Trading Agent' shortcut to the launcher in `folder` (Windows only)."""
    if os.name != "nt":
        raise RuntimeError("Shortcuts are only supported on Windows")
    launcher = Path(sys.executable).with_name("trading-agent.exe")
    if not launcher.exists():
        raise RuntimeError(f"{launcher} not found; run: pip install -e \".[dashboard]\"")
    link = folder / "Trading Agent.lnk"
    icon = PROJECT_ROOT / "assets" / "trading-agent.ico"
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:TA_LINK); "
        "$s.TargetPath = $env:TA_TARGET; $s.WorkingDirectory = $env:TA_CWD; "
        "$s.Description = 'Trading agent'; "
        "if (Test-Path $env:TA_ICON) { $s.IconLocation = $env:TA_ICON }; $s.Save()"
    )
    env = {**os.environ, "TA_LINK": str(link), "TA_TARGET": str(launcher),
           "TA_CWD": str(PROJECT_ROOT), "TA_ICON": str(icon)}
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, env=env)
    return link


def _error(message: str) -> None:
    if os.name == "nt":
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "Trading agent", 0x10)
    else:
        print(message, file=sys.stderr)


def main() -> int:
    os.chdir(PROJECT_ROOT)
    try:
        if not ensure_server():
            _error(f"The app did not start. See {PROJECT_ROOT / 'data' / 'logs' / 'app.log'}")
            return 1
        open_window()
        return 0
    except Exception as exc:
        _error(f"Could not start the app: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
