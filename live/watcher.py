"""Daily V19d agent loop.

Circuit-breaker sells execute immediately (pre-authorized).
Buys and rebalances are queued for human approval — nothing else is asked.

  .venv/bin/python -m live.watcher --live-quotes
  .venv/bin/python -m live.approvals list
  .venv/bin/python -m live.approvals approve pod1-buy pod2-buy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import pandas as pd

from live.approvals import AUTO, INFORM, NEEDS_YOU, brief_ticket, drop_stale, upsert_pending
from live.broker import get_broker
from live.ledger import log_fill, log_signal
from live.signals import compute_smas, evaluate_day
from live.state import arm_wash_sale, load_state, save_state, transition_on_signal


def load_prices(use_live: bool) -> pd.DataFrame:
    """Signal prices (adjusted closes) from the configured source (PRICE_SOURCE, default
    Tiingo). `use_live=True` forces a fresh EOD pull (bypasses the cache-freshness check)
    rather than overlaying an intraday print — the signal stays purely close-based; the
    intraday 3:45 cutoff quote is Alpaca's job in live/execute.py."""
    from live.prices import load_signal_prices
    return load_signal_prices(refresh=use_live)


def _px(prices: pd.DataFrame, ticker: str | None) -> float | None:
    if not ticker or ticker not in prices.columns:
        return None
    s = prices[ticker].dropna()
    return float(s.iloc[-1]) if len(s) else None


def format_brief(report: dict) -> str:
    sc = report["scores"]
    md = report["modes"]
    lines = [
        f"V19d {report['day']}  capital ${report['capital']:,.0f}  "
        f"{'DRY-RUN' if report['dry_run'] else 'LIVE'}",
        f"Scores  QQQ {sc['QQQ']}/3  IVV {sc['IVV']}/3  IAU {sc['IAU']}/3",
        f"Target  pod1 {md['pod1']}  pod2 {md['pod2']}  gold {md['gold']}",
    ]
    auto = report["auto"]
    if auto:
        lines.append("AUTO (executed, no ask):")
        for row in auto:
            o = row.get("order") or {}
            act = row["action"]
            lines.append(
                f"  {act['action']} {act.get('ticker') or '—'}  "
                f"{o.get('message', '')}"
            )
    else:
        lines.append("AUTO: none")
    pending = report["needs_approval"]
    if pending:
        lines.append("NEEDS YOUR APPROVAL:")
        for t in pending:
            lines.append("  " + brief_ticket(t))
        ids = " ".join(t["id"] for t in pending)
        lines.append(f"Reply: approve {ids}   |   reject <id>")
    else:
        lines.append("NEEDS YOUR APPROVAL: none")
    for b in report.get("blocked") or []:
        lines.append(f"BLOCKED  {b['sleeve']}  {b.get('reason', '')}")
    return "\n".join(lines)


def run(use_live_quotes: bool = False, execute_auto: bool = True) -> dict:
    prices = load_prices(use_live_quotes)
    day = prices.dropna(how="all").index.max()
    smas = compute_smas(prices[["QQQ", "IVV", "IAU"]].dropna(how="all"))
    ev = evaluate_day(day, prices, smas)

    st = load_state()
    actions = transition_on_signal(st, ev)
    log_signal(ev, actions)

    quotes = {
        t: float(prices[t].dropna().iloc[-1])
        for t in ("QQQ", "IVV", "IAU", "QLD", "SSO")
        if t in prices.columns and prices[t].notna().any()
    }
    broker = get_broker(quotes)

    auto_out = []
    pending = []
    blocked = []
    pending_ids: set[str] = set()

    for act in actions:
        kind = act["action"]
        if kind in AUTO and execute_auto and act.get("ticker"):
            tkr = act["ticker"]
            q = broker.get_quote(tkr)
            qty = 1.0  # flatten whatever we hold; MCP will size from position later
            order = broker.place_equity_order(tkr, "sell", qty, "market", None)
            fill = log_fill(
                sleeve=act["sleeve"],
                action="CB_SELL",
                ticker=tkr,
                qty=qty,
                intended_px=q.last,
                fill_px=order.fill_px or q.last,
                dry_run=order.dry_run,
                note=order.message,
            )
            sl = st.sleeves[act["sleeve"]]
            last = sl.get("last_fill_px")
            loss = bool(last is not None and order.fill_px is not None and order.fill_px < float(last))
            if st.tax_mode.startswith("TAXABLE") and (loss or last is None):
                arm_wash_sale(sl, loss=True)
            sl["status"] = "FLAT"
            sl["mode"] = "cash"
            sl["levered"] = False
            sl["ticker"] = None
            sl["last_fill_px"] = order.fill_px
            auto_out.append({"action": act, "order": order.__dict__, "fill": fill})
        elif kind in NEEDS_YOU:
            px = _px(prices, act.get("ticker"))
            ticket = upsert_pending(act, st.capital, px)
            pending_ids.add(ticket["id"])
            pending.append(ticket)
        elif kind in INFORM:
            blocked.append(act)

    drop_stale(pending_ids)
    save_state(st)

    report = {
        "day": ev["day"],
        "scores": ev["scores"],
        "modes": {
            "pod1": ev["p1_mode"],
            "pod2": ev["p2_mode"],
            "gold": ev["gold_mode"],
        },
        "breaches": ev["breaches"],
        "capital": st.capital,
        "auto": auto_out,
        "needs_approval": pending,
        "blocked": blocked,
        "dry_run": st.dry_run,
        "stage": st.stage,
    }
    report["brief"] = format_brief(report)
    return report


def main():
    ap = argparse.ArgumentParser(description="V19d agentic watcher — auto CB, approve buys")
    ap.add_argument("--live-quotes", action="store_true")
    ap.add_argument("--no-exec", action="store_true", help="detect only; skip auto CB fills")
    ap.add_argument("--json", action="store_true", help="print full JSON instead of the brief")
    args = ap.parse_args()
    report = run(use_live_quotes=args.live_quotes, execute_auto=not args.no_exec)
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "brief"}, indent=2, default=str))
        print(report["brief"], file=sys.stderr)
    else:
        print(report["brief"])


if __name__ == "__main__":
    main()
