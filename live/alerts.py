"""Alert delivery for the manual (Fidelity) workflow.

Every alert goes to the console and to a dated file under live/runtime/alerts/.
If ALERT_WEBHOOK is set, it also POSTs {"text": ...} — the shape Slack, Discord,
and Telegram-bot incoming webhooks accept. Delivery is best-effort: a webhook
failure never breaks the run.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent / "runtime"
ALERTS_DIR = RUNTIME / "alerts"


def _post_webhook(text: str) -> str | None:
    url = os.environ.get("ALERT_WEBHOOK", "").strip()
    if not url:
        return None
    try:
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return f"webhook {resp.status}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return f"webhook failed: {e}"


def deliver(subject: str, body: str = "", *, to_console: bool = True) -> dict:
    """Send an alert to console + dated file + optional webhook."""
    text = subject if not body else f"{subject}\n{body}"
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    logfile = ALERTS_DIR / f"{stamp.date().isoformat()}.txt"
    with logfile.open("a") as f:
        f.write(f"\n[{stamp.isoformat()}] {text}\n")
    if to_console:
        print(text)
    hook = _post_webhook(text)
    return {"file": str(logfile), "webhook": hook}
