"""V19d taxable-sleeve automated executor (Alpaca, all-MOC, human-in-the-loop).

This is the AUTOMATED counterpart to the manual `live/fidelity.py` planner, for the
taxable sleeve only. It reuses fidelity's planning + FIFO-lot + wash-sale machinery
(no logic fork) and adds an Alpaca execution lifecycle:

  sync    pull positions + cash from Alpaca (the broker is the source of truth)
  stage   ~3:45 ET: build the V19d plan, size to whole shares, guard it, alert you
  submit  after you approve: place the staged orders as MOC (tif=cls) before 3:50 ET
  settle  after 4:00 ET: read the auction fills, update lots/realized, refresh tracker
  status  show staged/submitted orders and their fill state

Every order is market-on-close, so fills happen in the official closing auction.
Paper by default; live requires ALPACA_PAPER=0 in .env AND `submit --live`.

  .venv/bin/python -m live.execute sync
  .venv/bin/python -m live.execute stage
  .venv/bin/python -m live.execute submit --yes
  .venv/bin/python -m live.execute settle
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from live import fidelity as F
from live.alerts import deliver
from live.broker import AlpacaBroker, AlpacaError
from live.signals import compute_smas, evaluate_day
from live.tax_lots import add_lot

ACCOUNT = "taxable"
MOC_CUTOFF_MIN = 10          # minutes before close that Alpaca stops accepting CLS (15:50)
MAX_DAILY_MOVE = 0.25        # plausibility: reject if intraday vs last close deviates > ±25%
STALE_DAYS = 4               # plausibility: reject if the signal's last close is older than this


def _pending_path():
    return F.RUNTIME / f"{ACCOUNT}_pending_orders.json"


def _broker(allow_live: bool = False) -> AlpacaBroker:
    return AlpacaBroker(allow_live=allow_live)


# ── state sync (Alpaca = truth) ──────────────────────────────────────────────

def sync_from_alpaca(broker: AlpacaBroker) -> dict:
    """Overwrite tracked taxable holdings + cash from Alpaca positions."""
    F.set_account(ACCOUNT)
    st = F.load_state()
    acct = broker.get_account()
    positions = broker.get_positions()
    today = datetime.now(timezone.utc).date().isoformat()

    for s in F.SLEEVES:
        st["sleeves"][s].update({"ticker": None, "shares": 0.0, "mode": "cash",
                                 "levered": False, "lots": []})
    held = []
    for sym, p in positions.items():
        s = F.TKR_TO_SLEEVE.get(sym)
        if not s:
            continue  # a symbol V19d doesn't manage (shouldn't happen in a dedicated account)
        st["sleeves"][s].update({
            "ticker": sym, "shares": round(p.shares, 6), "mode": F.MODE_FOR[sym],
            "levered": sym in ("QLD", "SSO"),
            "lots": [{"shares": round(p.shares, 6), "cost": p.avg_cost, "date": today}]})
        held.append(f"{sym} {p.shares:g}")
    st["cash"] = round(acct["cash"], 2)
    F.save_state(st)
    return {"cash": st["cash"], "held": held, "equity": acct["equity"]}


# ── plausibility guard ───────────────────────────────────────────────────────

def plausibility_guard(broker: AlpacaBroker, plan_prices: dict, day: pd.Timestamp) -> list[str]:
    """Block reasons for NOT auto-submitting. Empty list = clear to trade.

    Cheap backstop against bad data firing a leveraged order: stale signal data,
    or an intraday quote wildly off the signal's last close (a split/feed glitch).
    """
    reasons = []
    acct = broker.get_account()
    if acct.get("trading_blocked"):
        reasons.append("Alpaca account trading_blocked=True")
    age = (datetime.now(timezone.utc).date() - day.date()).days if hasattr(day, "date") else None
    if age is not None and age > STALE_DAYS:
        reasons.append(f"signal data is stale — last close {day.date()} ({age}d old)")
    for tkr, eod_px in plan_prices.items():
        if not eod_px or tkr not in F.TKR_TO_SLEEVE:
            continue
        try:
            q = broker.get_quote(tkr).last
        except AlpacaError:
            continue
        if q and abs(q / eod_px - 1) > MAX_DAILY_MOVE:
            reasons.append(f"{tkr}: intraday ${q:,.2f} vs last close ${eod_px:,.2f} "
                           f"({q/eod_px-1:+.0%}) exceeds ±{MAX_DAILY_MOVE:.0%} — possible bad data")
    return reasons


# ── MOC window ───────────────────────────────────────────────────────────────

def moc_window(broker: AlpacaBroker) -> tuple[bool, str]:
    c = broker.get_clock()
    if not c["is_open"]:
        return False, "market is closed today"
    close = datetime.fromisoformat(c["next_close"])
    now = datetime.fromisoformat(c["timestamp"])
    cutoff = close - timedelta(minutes=MOC_CUTOFF_MIN)
    if now > cutoff:
        return False, f"past the {cutoff:%H:%M} MOC cutoff (close {close:%H:%M})"
    return True, f"open until {cutoff:%H:%M} MOC cutoff"


# ── stage ────────────────────────────────────────────────────────────────────

def _build_plan_now(force: bool):
    """Reuse fidelity's planner against the current signal + synced state."""
    prices = F.load_prices(False)
    day = prices.dropna(how="all").index.max()
    smas = compute_smas(prices[["QQQ", "IVV", "IAU"]].dropna(how="all"))
    ev = evaluate_day(day, prices, smas)
    px = F.last_prices(prices)
    st = F.load_state()
    ms = F.is_month_start(day, prices)
    already = st.get("last_rebalance_month") == ev["day"][:7]
    trades, notes = F.build_plan(st, ev, px, day, month_start=ms and not already, force=force)
    return st, ev, px, day, trades, notes


def cmd_stage(args) -> None:
    F.set_account(ACCOUNT)
    st, ev, px, day, trades, notes = _build_plan_now(args.force)

    # translate sized trades into whole-share MOC orders
    orders = []
    for t in trades:
        whole = int(round(t.shares))
        if whole < 1:
            continue
        orders.append({"sleeve": t.sleeve, "side": t.side.lower(), "symbol": t.ticker,
                       "qty": whole, "reason": t.reason, "urgent": t.urgent,
                       "ref_price": t.price})
    payload = {"account": ACCOUNT, "staged_at": datetime.now(timezone.utc).isoformat(),
               "signal_day": ev["day"], "scores": ev["scores"],
               "orders": orders, "prices": px,
               "notes": {k: notes.get(k) for k in ("capital", "month_start", "rebalance",
                                                   "warnings", "blocked")}}
    _pending_path().write_text(json.dumps(payload, indent=2, default=str))

    lines = [f"V19d · Alpaca taxable · {ev['day']}  (staged)",
             f"Signal QQQ {ev['scores']['QQQ']}/3  IVV {ev['scores']['IVV']}/3  IAU {ev['scores']['IAU']}/3"]
    if not orders:
        lines.append("No orders to place — hold.")
    else:
        for o in orders:
            tag = "⚠️ CB " if o["urgent"] else "     "
            lines.append(f"{tag}{o['side'].upper()} {o['qty']} {o['symbol']} MOC  (~${o['qty']*o['ref_price']:,.0f})  — {o['reason']}")
        lines.append(f"APPROVE by 3:50 ET →  .venv/bin/python -m live.execute submit --yes")
    for b in (notes.get("blocked") or []):
        lines.append(f"BLOCKED  {b}")
    for w in (notes.get("warnings") or []):
        lines.append(f"⚠️  WASH SALE  {w}")
    deliver("\n".join(lines))


# ── submit ───────────────────────────────────────────────────────────────────

def cmd_submit(args) -> None:
    F.set_account(ACCOUNT)
    pend = _pending_path()
    if not pend.exists():
        raise SystemExit("Nothing staged. Run: python -m live.execute stage")
    payload = json.loads(pend.read_text())
    orders = payload.get("orders", [])
    if not orders:
        print("Staged plan has no orders — nothing to submit.")
        return
    if payload.get("submitted"):
        print("This staged plan was already submitted. Run `settle` after the close.")
        return
    if not args.yes:
        raise SystemExit("Refusing to submit without approval. Re-run with --yes to place these MOC orders.")

    broker = _broker(allow_live=args.live)
    # guards
    guard = plausibility_guard(broker, payload.get("prices", {}),
                               pd.Timestamp(payload["signal_day"]))
    if guard and not args.override_guard:
        deliver("V19d · Alpaca taxable · SUBMIT BLOCKED by guard:\n  - " + "\n  - ".join(guard)
                + "\n(no orders sent). Re-run with --override-guard only if you've verified the data.")
        raise SystemExit(1)
    ok, why = moc_window(broker)
    if not ok and not broker.paper and not args.force_window:
        raise SystemExit(f"Outside MOC window ({why}); not submitting live. Use --force-window to override.")

    results = []
    for o in orders:
        try:
            r = broker.place_moc(o["symbol"], o["side"], o["qty"],
                                 client_order_id=f"v19d-{payload['signal_day']}-{o['sleeve']}-{o['side']}")
            results.append({**o, "order_id": r.broker_id, "status": r.message, "ok": True})
        except AlpacaError as e:
            results.append({**o, "order_id": None, "status": str(e), "ok": False})

    payload["submitted"] = datetime.now(timezone.utc).isoformat()
    payload["mode"] = "paper" if broker.paper else "live"
    payload["results"] = results
    pend.write_text(json.dumps(payload, indent=2, default=str))

    good = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    lines = [f"V19d · Alpaca taxable · SUBMITTED ({payload['mode']})  {why}"]
    for r in good:
        lines.append(f"  ✓ {r['side'].upper()} {r['qty']} {r['symbol']} MOC → {r['status']} ({r['order_id']})")
    for r in bad:
        lines.append(f"  ✗ {r['side'].upper()} {r['qty']} {r['symbol']} FAILED → {r['status']}  (T+1 fallback)")
    lines.append("After 4:00 ET run:  .venv/bin/python -m live.execute settle")
    deliver("\n".join(lines))


# ── settle ───────────────────────────────────────────────────────────────────

def cmd_settle(args) -> None:
    F.set_account(ACCOUNT)
    pend = _pending_path()
    if not pend.exists() or not json.loads(pend.read_text()).get("submitted"):
        raise SystemExit("No submitted plan to settle.")
    payload = json.loads(pend.read_text())
    broker = _broker(allow_live=(payload.get("mode") == "live"))
    st = F.load_state()
    today = datetime.now(timezone.utc).date().isoformat()
    st.setdefault("wash_until", {})

    settled, unfilled = [], []
    buys_today = []
    for r in payload.get("results", []):
        if not r.get("ok") or not r.get("order_id"):
            unfilled.append({**r, "why": "was not submitted"})
            continue
        o = broker.get_order(r["order_id"])
        status = o.get("status")
        filled_qty = float(o.get("filled_qty") or 0)
        fill_px = float(o.get("filled_avg_price") or 0)
        if status != "filled" or filled_qty < 1 or fill_px <= 0:
            unfilled.append({**r, "why": f"status={status} filled_qty={filled_qty}"})
            continue
        sl = st["sleeves"][r["sleeve"]]
        sl.setdefault("lots", [])
        if r["side"] == "sell" and sl.get("ticker") == r["symbol"]:
            from live.tax_lots import sell_fifo
            events = sell_fifo(sl["lots"], filled_qty, fill_px, today)
            for e in events:
                e["ticker"] = r["symbol"]
            append_realized(events)
            net = sum(e["gain"] for e in events)
            if F.PROFILE["wash_sale"] and net < 0:
                from live.tax_lots import as_date
                st["wash_until"][r["symbol"]] = (as_date(today) + timedelta(days=F.WASH_DAYS)).isoformat()
            sl["shares"] = round(sum(l["shares"] for l in sl["lots"]), 6)
            if sl["shares"] <= 1e-6:
                sl.update({"shares": 0.0, "ticker": None, "mode": "cash", "levered": False, "lots": []})
            st["cash"] = round(float(st.get("cash", 0.0)) + filled_qty * fill_px, 2)
        elif r["side"] == "buy":
            if sl.get("ticker") not in (None, r["symbol"]):
                sl["lots"] = []
            sl["ticker"] = r["symbol"]
            sl["mode"] = F.MODE_FOR[r["symbol"]]
            sl["levered"] = r["symbol"] in ("QLD", "SSO")
            add_lot(sl["lots"], filled_qty, fill_px, today)
            sl["shares"] = round(sum(l["shares"] for l in sl["lots"]), 6)
            st["cash"] = round(float(st.get("cash", 0.0)) - filled_qty * fill_px, 2)
            buys_today.append(r["symbol"])
        settled.append({**r, "fill_px": fill_px, "filled_qty": filled_qty})

    for t in set(buys_today):
        F.disallow_taxable_losses(t, today)
    F.save_state(st)
    payload["settled"] = datetime.now(timezone.utc).isoformat()
    pend.write_text(json.dumps(payload, indent=2, default=str))

    # rebuild the tracker view
    try:
        F.cmd_refresh(argparse.Namespace(account=ACCOUNT))
    except Exception:
        pass

    lines = [f"V19d · Alpaca taxable · SETTLED  {today}"]
    for r in settled:
        lines.append(f"  ✓ {r['side'].upper()} {r['filled_qty']:g} {r['symbol']} @ ${r['fill_px']:,.2f} (auction)")
    for r in unfilled:
        lines.append(f"  … {r['side'].upper()} {r['qty']} {r['symbol']} NOT filled ({r['why']}) — T+1 fallback")
    deliver("\n".join(lines))


# helper: append_realized bound to the taxable ledger (fidelity uses module globals)
def append_realized(events):
    F.append_realized(events)


# ── sync / status commands ───────────────────────────────────────────────────

def cmd_sync(args) -> None:
    F.set_account(ACCOUNT)
    info = sync_from_alpaca(_broker())
    print(f"Synced taxable from Alpaca: cash ${info['cash']:,.2f}  equity ${info['equity']:,.2f}")
    print("  held: " + (", ".join(info["held"]) or "(none)"))


def cmd_status(args) -> None:
    F.set_account(ACCOUNT)
    pend = _pending_path()
    if not pend.exists():
        print("No staged plan.")
        return
    p = json.loads(pend.read_text())
    print(f"Staged {p.get('staged_at')}  signal {p.get('signal_day')}  "
          f"submitted={bool(p.get('submitted'))}  settled={bool(p.get('settled'))}")
    for o in p.get("results") or p.get("orders", []):
        oid = o.get("order_id", "")
        print(f"  {o['side'].upper()} {o['qty']} {o['symbol']} MOC  {o.get('status','staged')}  {oid}")


def main() -> None:
    ap = argparse.ArgumentParser(description="V19d taxable automated executor (Alpaca, all-MOC)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync", help="pull positions + cash from Alpaca").set_defaults(func=cmd_sync)
    st = sub.add_parser("stage", help="build the plan and stage MOC orders + alert")
    st.add_argument("--force", action="store_true", help="force a full rebalance")
    st.set_defaults(func=cmd_stage)
    su = sub.add_parser("submit", help="place staged orders as MOC (needs --yes)")
    su.add_argument("--yes", action="store_true", help="approval: actually submit")
    su.add_argument("--live", action="store_true", help="allow live (requires ALPACA_PAPER=0)")
    su.add_argument("--override-guard", action="store_true", help="submit despite a guard block")
    su.add_argument("--force-window", action="store_true", help="submit outside the MOC window (paper testing)")
    su.set_defaults(func=cmd_submit)
    sub.add_parser("settle", help="read auction fills, update lots/realized").set_defaults(func=cmd_settle)
    sub.add_parser("status", help="show staged/submitted orders").set_defaults(func=cmd_status)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
