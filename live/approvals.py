"""Human approval queue. Buys and rebalances wait here; CB sells never do."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from live.broker import get_broker
from live.ledger import log_fill
from live.sizing import shares_for, sleeve_notional
from live.state import load_state, save_state

RUNTIME = Path(__file__).resolve().parent / "runtime"
PATH = RUNTIME / "approvals.json"

AUTO = {"CB_SELL"}
NEEDS_YOU = {"BUY", "REBALANCE"}
INFORM = {"BLOCKED_WASH"}


def load_queue() -> dict:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if not PATH.exists():
        return {"pending": [], "closed": []}
    return json.loads(PATH.read_text())


def save_queue(q: dict) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(q, indent=2, default=str))


def ticket_id(sleeve: str, action: str) -> str:
    return f"{sleeve}-{action.lower()}"


def upsert_pending(action: dict, capital: float, last_px: float | None) -> dict:
    """One open ticket per sleeve+action. Replaces if ticker/mode changed."""
    q = load_queue()
    tid = ticket_id(action["sleeve"], action["action"])
    mode = action.get("mode")
    ticker = action.get("ticker")
    notional = sleeve_notional(capital, action["sleeve"], mode) if mode else 0.0
    qty = shares_for(notional, last_px or 0.0) if ticker else 0.0
    ticket = {
        "id": tid,
        "created": datetime.now(timezone.utc).isoformat(),
        "sleeve": action["sleeve"],
        "action": action["action"],
        "ticker": ticker,
        "mode": mode,
        "levered": bool(action.get("levered")),
        "qty": qty,
        "notional": round(notional, 2),
        "px": last_px,
        "reason": action.get("reason", ""),
        "status": "pending",
    }
    q["pending"] = [t for t in q["pending"] if t["id"] != tid]
    q["pending"].append(ticket)
    save_queue(q)
    return ticket


def drop_stale(valid_ids: set[str]) -> None:
    """Close pending tickets whose signal is gone (mode flipped, already filled)."""
    q = load_queue()
    keep, closed = [], q.get("closed", [])
    for t in q.get("pending", []):
        if t["id"] in valid_ids:
            keep.append(t)
        else:
            t = dict(t)
            t["status"] = "superseded"
            t["closed"] = datetime.now(timezone.utc).isoformat()
            closed.append(t)
    q["pending"] = keep
    q["closed"] = closed[-200:]
    save_queue(q)


def pending() -> list[dict]:
    return load_queue().get("pending", [])


def _close(tid: str, status: str) -> dict | None:
    q = load_queue()
    found = None
    keep = []
    for t in q.get("pending", []):
        if t["id"] == tid:
            found = dict(t)
            found["status"] = status
            found["closed"] = datetime.now(timezone.utc).isoformat()
        else:
            keep.append(t)
    if found is None:
        return None
    q["pending"] = keep
    q.setdefault("closed", []).append(found)
    q["closed"] = q["closed"][-200:]
    save_queue(q)
    return found


def reject(tid: str) -> dict:
    t = _close(tid, "rejected")
    if t is None:
        raise SystemExit(f"no pending ticket {tid}")
    st = load_state()
    sl = st.sleeves.get(t["sleeve"])
    if sl:
        sl["status"] = "FLAT"
        sl["rejected_mode"] = t.get("mode")
        save_state(st)
    return t


def approve(tid: str) -> dict:
    q = load_queue()
    ticket = next((t for t in q.get("pending", []) if t["id"] == tid), None)
    if ticket is None:
        raise SystemExit(f"no pending ticket {tid}")

    quotes = {}
    if ticket.get("ticker") and ticket.get("px"):
        quotes[ticket["ticker"]] = float(ticket["px"])
    broker = get_broker(quotes)
    ticker = ticket.get("ticker")
    qty = float(ticket.get("qty") or 0)
    if not ticker or qty <= 0:
        raise SystemExit(f"{tid}: nothing to buy (qty={qty})")

    qte = broker.get_quote(ticker)
    order = broker.place_equity_order(ticker, "buy", qty, "limit", qte.last if qte.last else None)
    if not order.ok:
        raise SystemExit(order.message)

    log_fill(
        sleeve=ticket["sleeve"],
        action=ticket["action"],
        ticker=ticker,
        qty=qty,
        intended_px=qte.last,
        fill_px=order.fill_px or qte.last,
        dry_run=order.dry_run,
        note=order.message,
    )
    st = load_state()
    sl = st.sleeves[ticket["sleeve"]]
    sl["status"] = "HOLD"
    sl["mode"] = ticket.get("mode")
    sl["levered"] = bool(ticket.get("levered"))
    sl["ticker"] = ticker
    sl["last_fill_px"] = order.fill_px
    sl["rejected_mode"] = None
    save_state(st)
    closed = _close(tid, "approved")
    return {"ticket": closed, "order": order.__dict__}


def brief_ticket(t: dict) -> str:
    px = t.get("px")
    px_s = f" @ {px:.2f}" if px else ""
    return (
        f"{t['id']:>16}  {t['action']:<9}  {t.get('ticker') or '—':<4}  "
        f"{t.get('qty') or 0:>8.4f} sh  ${t.get('notional') or 0:,.0f}{px_s}  "
        f"{t.get('reason', '')}"
    )


def main():
    ap = argparse.ArgumentParser(description="V19d approval queue")
    ap.add_argument("cmd", nargs="?", default="list", choices=["list", "approve", "reject"])
    ap.add_argument("ids", nargs="*")
    args = ap.parse_args()
    if args.cmd == "list":
        rows = pending()
        if not rows:
            print("No pending approvals.")
            return
        print("Pending — reply approve <id> …  or reject <id>")
        for t in rows:
            print(" ", brief_ticket(t))
        return
    if not args.ids:
        raise SystemExit(f"{args.cmd} needs ticket ids (e.g. pod1-buy)")
    for tid in args.ids:
        if args.cmd == "approve":
            print(json.dumps(approve(tid), indent=2, default=str))
        else:
            print(json.dumps(reject(tid), indent=2, default=str))


if __name__ == "__main__":
    main()
