"""Tests for trading/alerts.py — no real network or SMTP."""
from __future__ import annotations

import json

import trading.alerts as alerts


def _clear_channels(monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ALERT_EMAIL_TO", raising=False)


def test_no_channel_returns_false_and_does_not_raise(monkeypatch):
    _clear_channels(monkeypatch)
    assert alerts.send_alert("hello") is False


def test_webhook_called_when_configured(monkeypatch):
    _clear_channels(monkeypatch)
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.test/hook")
    seen = {}
    monkeypatch.setattr(
        alerts, "_send_webhook",
        lambda url, message: seen.update(url=url, message=message),
    )
    assert alerts.send_alert("boom") is True
    assert seen == {"url": "https://example.test/hook", "message": "boom"}


def test_email_called_when_configured(monkeypatch):
    _clear_channels(monkeypatch)
    monkeypatch.setenv("ALERT_EMAIL_TO", "me@example.test")
    sent = []
    monkeypatch.setattr(alerts, "_send_email", lambda message: sent.append(message))
    assert alerts.send_alert("boom") is True
    assert sent == ["boom"]


def test_failing_channel_is_caught_and_returns_false(monkeypatch):
    _clear_channels(monkeypatch)
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.test/hook")

    def boom(url, message):
        raise RuntimeError("network down")

    monkeypatch.setattr(alerts, "_send_webhook", boom)
    # Must not raise; no channel succeeded -> False.
    assert alerts.send_alert("x") is False


def test_both_channels_attempted(monkeypatch):
    _clear_channels(monkeypatch)
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.test/hook")
    monkeypatch.setenv("ALERT_EMAIL_TO", "me@example.test")
    calls = []
    monkeypatch.setattr(alerts, "_send_webhook", lambda url, message: calls.append("web"))
    monkeypatch.setattr(alerts, "_send_email", lambda message: calls.append("email"))
    assert alerts.send_alert("x") is True
    assert calls == ["web", "email"]


def test_send_webhook_posts_json_payload(monkeypatch):
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["content_type"] = req.headers.get("Content-type")
        return _Resp()

    monkeypatch.setattr(alerts.urllib.request, "urlopen", fake_urlopen)
    alerts._send_webhook("https://example.test/hook", "hi there")
    assert captured["url"] == "https://example.test/hook"
    assert json.loads(captured["data"]) == {"text": "hi there"}
    assert captured["content_type"] == "application/json"


def test_main_joins_argv_into_message(monkeypatch):
    sent = []
    monkeypatch.setattr(alerts, "send_alert", lambda msg: sent.append(msg) or True)
    monkeypatch.setattr(alerts.sys, "argv", ["prog", "weights", "job", "failed"])
    assert alerts.main() == 0
    assert sent == ["weights job failed"]
