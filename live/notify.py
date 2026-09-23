"""Slack approval channel for the taxable executor (reaction polling).

Outbound-only, raw REST (no SDK, no Socket Mode, no inbound port):
  1. `execute stage` posts the staged plan to a private channel.
  2. You react ✅ to approve or ❌ to reject from the Slack phone app.
  3. `execute await` polls reactions.get until a decision, the MOC cutoff, or a timeout.

Safety: only a reaction from SLACK_APPROVER_USER_ID on the EXACT staged message
counts (a reaction on an old plan is ignored), and the actual submit still re-runs
the plausibility guard + MOC-window check. No reaction by the cutoff = nothing fires.

Slack app setup (free workspace is fine):
  api.slack.com/apps -> Create New App -> OAuth & Permissions -> Bot Token Scopes:
    chat:write, reactions:read
  Install to Workspace -> copy the Bot User OAuth Token (xoxb-...).
  In Slack: create a private channel, /invite @<your-bot>, then copy the channel ID
  (channel details -> bottom) and your own member ID (your profile -> ... -> Copy member ID).
  .env:  SLACK_BOT_TOKEN=xoxb-...  SLACK_CHANNEL_ID=C...  SLACK_APPROVER_USER_ID=U...

Check:  .venv/bin/python -m live.notify            # auth test
        .venv/bin/python -m live.notify --post     # also posts a test message
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLACK_API = "https://slack.com/api"
APPROVE_EMOJI = "white_check_mark"  # ✅
REJECT_EMOJI = "x"                  # ❌


class SlackError(RuntimeError):
    """Missing config or a non-ok Slack API response."""


def configured() -> bool:
    return bool(os.environ.get("SLACK_BOT_TOKEN", "").strip()
                and os.environ.get("SLACK_CHANNEL_ID", "").strip())


class Slack:
    def __init__(self, token: str | None = None, channel: str | None = None,
                 approver: str | None = None, timeout: int = 15):
        self.token = (token or os.environ.get("SLACK_BOT_TOKEN", "")).strip()
        self.channel = (channel or os.environ.get("SLACK_CHANNEL_ID", "")).strip()
        self.approver = (approver or os.environ.get("SLACK_APPROVER_USER_ID", "")).strip()
        if not self.token or not self.channel:
            raise SlackError("Set SLACK_BOT_TOKEN and SLACK_CHANNEL_ID in .env")
        self.timeout = timeout

    def _call(self, method: str, **params) -> dict:
        import requests
        r = requests.post(f"{SLACK_API}/{method}", data=params,
                          headers={"Authorization": f"Bearer {self.token}"}, timeout=self.timeout)
        try:
            data = r.json()
        except ValueError:
            raise SlackError(f"{method}: non-JSON response {r.status_code}")
        if not data.get("ok"):
            raise SlackError(f"{method}: {data.get('error', 'unknown error')}")
        return data

    def auth_test(self) -> dict:
        d = self._call("auth.test")
        return {"team": d.get("team"), "bot_user": d.get("user"), "user_id": d.get("user_id")}

    def post(self, text: str, thread_ts: str | None = None) -> str:
        """Post mrkdwn text (optionally as a thread reply); returns the message ts."""
        params = {"channel": self.channel, "text": text, "mrkdwn": "true"}
        if thread_ts:
            params["thread_ts"] = thread_ts
        return self._call("chat.postMessage", **params)["ts"]

    def reactions(self, ts: str) -> dict[str, list[str]]:
        """{emoji_name: [user_ids]} on one message."""
        d = self._call("reactions.get", channel=self.channel, timestamp=ts, full="true")
        out: dict[str, list[str]] = {}
        for r in (d.get("message", {}).get("reactions") or []):
            out[r["name"]] = list(r.get("users", []))
        return out

    def decision(self, ts: str) -> str | None:
        """'approve' | 'reject' | None — only the configured approver's reactions count.
        A ❌ wins over a ✅ if both are present (fail safe)."""
        if not self.approver:
            raise SlackError("Set SLACK_APPROVER_USER_ID so only YOUR reaction can approve")
        rx = self.reactions(ts)
        if self.approver in rx.get(REJECT_EMOJI, []):
            return "reject"
        if self.approver in rx.get(APPROVE_EMOJI, []):
            return "approve"
        return None


# ── plan rendering ───────────────────────────────────────────────────────────

def plan_id(payload: dict) -> str:
    return f"{payload.get('signal_day')}·{str(payload.get('staged_at', ''))[11:16]}Z"


def render_plan(payload: dict, mode: str) -> str:
    sc = payload.get("scores") or {}
    notes = payload.get("notes") or {}
    cap = notes.get("capital") or 0
    lines = [f"*V19d · Alpaca taxable · {payload.get('signal_day')}*  ({mode.upper()})",
             f"Signal  QQQ {sc.get('QQQ')}/3 · IVV {sc.get('IVV')}/3 · IAU {sc.get('IAU')}/3"
             f"   capital ${cap:,.0f}"]
    orders = payload.get("orders") or []
    if not orders:
        lines.append("_No orders — hold. Nothing to approve._")
    for o in orders:
        tag = "⚠️ *CB* " if o.get("urgent") else "• "
        lines.append(f"{tag}{o['side'].upper()} {o['qty']} {o['symbol']} MOC"
                     f"  (~${o['qty'] * o.get('ref_price', 0):,.0f}) — {o.get('reason', '')}")
    for b in (notes.get("blocked") or []):
        lines.append(f"🚫 BLOCKED  {b}")
    for w in (notes.get("warnings") or []):
        lines.append(f"⚠️ WASH  {w}")
    if orders:
        lines.append(f"React ✅ to approve · ❌ to reject — *closes at 3:50 ET*.   plan `{plan_id(payload)}`")
    return "\n".join(lines)


def post_plan(payload: dict, mode: str) -> str:
    return Slack().post(render_plan(payload, mode))


def _self_check(post: bool, whoami: bool) -> None:
    import time
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    try:
        s = Slack()
        a = s.auth_test()
    except SlackError as e:
        print(f"  ✗ {e}")
        return
    print(f"  Slack team={a['team']}  bot={a['bot_user']}  channel={s.channel}"
          f"  approver={s.approver or '(unset — run with --whoami to find it)'}")
    if whoami:
        # Find YOUR member ID without any extra scope: react to a message and read
        # the reactor list back via reactions.get (the same call `await` uses).
        ts = s.post("👋 Identify yourself: react ✅ to this message and I'll print your member ID.")
        print(f"  posted; react ✅ to it in Slack (waiting up to 120s)…")
        for _ in range(24):
            time.sleep(5)
            users = s.reactions(ts).get(APPROVE_EMOJI, [])
            if users:
                for u in users:
                    print(f"  → member ID: {u}")
                print(f"  paste into .env:  SLACK_APPROVER_USER_ID={users[0]}")
                break
        else:
            print("  no ✅ seen in 120s — react to the message and rerun.")
    elif post:
        ts = s.post("✓ V19d executor connected — react ✅ here to test approval polling.")
        print(f"  posted test message ts={ts}")
        print(f"  react ✅ to it, then: python -c \"from live.notify import Slack; print(Slack().decision('{ts}'))\"")
    print("  ✓ Slack OK")


if __name__ == "__main__":
    _self_check(post="--post" in sys.argv, whoami="--whoami" in sys.argv)
