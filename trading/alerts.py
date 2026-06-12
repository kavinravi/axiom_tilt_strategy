"""Failure alerting for unattended jobs.

Sends a short message to whatever channel is configured via env vars:
- ALERT_WEBHOOK_URL          ntfy URLs get a plain-text POST (body = message,
                             Title/Priority headers); others get Slack-style
                             JSON {"text": message}.
- ALERT_EMAIL_TO + ALERT_SMTP_*   send an email.

Both are optional. If neither is set, the message is only logged — send_alert never
raises, so a missing/broken alert channel can't itself crash a job. CLI:

    python -m trading.alerts "weights job failed"

Wired from systemd via `OnFailure=axiom-alert@%n.service`.
"""
from __future__ import annotations

import json
import logging
import smtplib
import sys
import urllib.parse
import urllib.request
from email.message import EmailMessage

from src.utils.env import get_env

logger = logging.getLogger(__name__)


def _send_webhook(url: str, message: str) -> None:
    if "ntfy" in urllib.parse.urlsplit(url).netloc:
        # ntfy: the raw body IS the notification text; metadata rides in headers.
        data = message.encode()
        headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Title": "axiom-tilt alert",
            "Priority": "high",
            "Tags": "rotating_light",
        }
    else:
        # Slack-style JSON for everything else.
        data = json.dumps({"text": message}).encode()
        headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp.read()


def _send_email(message: str) -> None:
    host = get_env("ALERT_SMTP_HOST", required=True)
    port = int(get_env("ALERT_SMTP_PORT", default="587"))
    user = get_env("ALERT_SMTP_USER", required=True)
    password = get_env("ALERT_SMTP_PASSWORD", required=True)
    to_addr = get_env("ALERT_EMAIL_TO", required=True)

    msg = EmailMessage()
    msg["Subject"] = "[axiom-tilt] job alert"
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(message)

    with smtplib.SMTP(host, port, timeout=20) as s:
        s.starttls()
        s.login(user, password)
        s.send_message(msg)


def send_alert(message: str) -> bool:
    """Send `message` to every configured channel. Returns True if any succeeded.

    Never raises — alerting must not be able to crash the caller.
    """
    sent = False
    webhook = get_env("ALERT_WEBHOOK_URL")
    if webhook:
        try:
            _send_webhook(webhook, message)
            sent = True
        except Exception as exc:  # noqa: BLE001
            logger.error("alert webhook failed: %s", exc)
    if get_env("ALERT_EMAIL_TO"):
        try:
            _send_email(message)
            sent = True
        except Exception as exc:  # noqa: BLE001
            logger.error("alert email failed: %s", exc)
    if not sent:
        logger.warning("ALERT (no channel delivered): %s", message)
    return sent


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    message = " ".join(sys.argv[1:]) or "axiom-tilt: unspecified job failure"
    send_alert(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
