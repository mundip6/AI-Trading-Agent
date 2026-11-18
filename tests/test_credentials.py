import urllib.error

from trading_agent import credentials as cr

EXAMPLE = """ANTHROPIC_API_KEY=sk-ant-old-value-1234
# OPENAI_API_KEY=
# TELEGRAM_BOT_TOKEN=
# MetaTrader 5 (all optional)
# MT5_TERMINAL_PATH=C:/Program Files/MetaTrader 5/terminal64.exe
OTHER=keep me
"""


def test_save_replaces_adds_and_removes_keeping_comments(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(EXAMPLE, encoding="utf-8")
    environ = {"ANTHROPIC_API_KEY": "sk-ant-old-value-1234"}
    cr.save_env(env_file, {"ANTHROPIC_API_KEY": "sk-ant-new-value-9876", "TELEGRAM_BOT_TOKEN": "123:abc",
                           "TELEGRAM_CHAT_ID": "42", "MT5_SERVER": "Broker Demo"}, environ)
    text = env_file.read_text(encoding="utf-8")
    assert "sk-ant-old" not in text and "ANTHROPIC_API_KEY=sk-ant-new-value-9876" in text
    assert "# TELEGRAM_BOT_TOKEN=\nTELEGRAM_BOT_TOKEN=123:abc" in text       # next to its example line
    assert "# MetaTrader 5 (all optional)" in text and "OTHER=keep me" in text
    assert cr.read_env(env_file) == {"ANTHROPIC_API_KEY": "sk-ant-new-value-9876", "TELEGRAM_BOT_TOKEN": "123:abc",
                                     "OTHER": "keep me", "TELEGRAM_CHAT_ID": "42", "MT5_SERVER": "Broker Demo"}
    assert environ["TELEGRAM_CHAT_ID"] == "42" and environ["MT5_SERVER"] == "Broker Demo"

    cr.save_env(env_file, {"ANTHROPIC_API_KEY": None, "TELEGRAM_CHAT_ID": ""}, environ)
    values = cr.read_env(env_file)
    assert "ANTHROPIC_API_KEY" not in values and "TELEGRAM_CHAT_ID" not in values
    assert "ANTHROPIC_API_KEY" not in environ and "TELEGRAM_CHAT_ID" not in environ


def test_save_creates_the_file_and_rejects_line_breaks(tmp_path):
    env_file = tmp_path / ".env"
    cr.save_env(env_file, {"GOOGLE_API_KEY": "  key-with-spaces-around  "}, {})
    assert cr.read_env(env_file) == {"GOOGLE_API_KEY": "key-with-spaces-around"}
    try:
        cr.save_env(env_file, {"GOOGLE_API_KEY": "a\nEVIL=1"}, {})
        raise AssertionError("expected a ValueError")
    except ValueError:
        pass
    assert "EVIL" not in env_file.read_text(encoding="utf-8")


def test_secrets_are_shown_only_by_their_end():
    key = cr.BY_NAME["ANTHROPIC_API_KEY"]
    assert cr.describe(key, "sk-ant-api03-abcdefgh-WXYZ") == "set, ends in …WXYZ"
    assert cr.describe(key, "short") == "set"
    assert cr.describe(key, "") == "not set"
    assert cr.describe(cr.BY_NAME["MT5_SERVER"], "PUPrime-Demo") == "PUPrime-Demo"


def test_ai_key_check():
    seen = []
    assert cr.check_ai("ANTHROPIC_API_KEY", "k", get=lambda url, headers: seen.append((url, headers))) == (True, "The key works.")
    assert seen[0][1]["x-api-key"] == "k"

    def rejected(url, headers):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

    ok, text = cr.check_ai("OPENAI_API_KEY", "bad", get=rejected)
    assert not ok and "rejected" in text


def test_telegram_check():
    sent = []
    assert cr.check_telegram("t", "1", send=sent.append)[0] and "test message" in sent[0]

    def unknown_chat(text):
        raise urllib.error.HTTPError("u", 400, "Bad Request", {}, None)

    ok, text = cr.check_telegram("t", "1", send=unknown_chat)
    assert not ok and "chat ID" in text
