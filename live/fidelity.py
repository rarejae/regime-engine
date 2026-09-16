"""V19d manual planner + tracker — multi-account (Fidelity HSA / Robinhood taxable).

Nothing here places orders. Each run turns the V19d signal into a **trade plan you
execute by hand**, delivers it as an alert, and waits for you to `confirm`. Every
command takes `--account {hsa,taxable}` (default hsa); each account keeps its own
state files and tax treatment — HSA is tax-free (no wash-sale), taxable is not.

The repo tracks the portfolio itself: given a data source and the shares it last
recorded, it knows current weights and what to trade to hit target — a Fidelity
feed is only needed to reconcile (`positions`), not to run.

HSA = tax-free wrapper, so there is no wash-sale clock and no tax drag. V19d runs
at its full pre-tax profile here.

Workflow
--------
  # once, to fund the account
  .venv/bin/python -m live.fidelity add-cash 5000

  # daily: get the plan (CB sells any day; full rebalance only on month start)
  .venv/bin/python -m live.fidelity plan --live-quotes

  # after you place the trades at Fidelity
  .venv/bin/python -m live.fidelity confirm

  # anytime
  .venv/bin/python -m live.fidelity status
  .venv/bin/python -m live.fidelity positions ~/Downloads/Portfolio_Positions.csv
  .venv/bin/python -m live.fidelity withdraw 2000   # emergency: cash first, then pro-rata sells
  .venv/bin/python -m live.fidelity refresh         # rebuild tracker JSON (account vs benchmarks)

Data split: account holdings/cash come from the Fidelity CSV (`positions`) or from
`confirm`; market/benchmark prices come from yfinance (the signals' source). The
`tracker/` Node app is a read-only view over the JSON `refresh` writes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import pandas as pd

from live.alerts import deliver
from live.ledger import log_fill
from live.signals import evaluate_day, compute_smas, ticker_for_mode
from live.sizing import WEIGHTS, sleeve_notional
from live.tax_lots import add_lot, sell_fifo, estimate_tax, as_date
from live.watcher import load_prices

# Taxable-account estimator rates (edit to your bracket). Estimator, not a 1099.
TAX_ORDINARY, TAX_LTCG, TAX_STATE = 0.32, 0.15, 0.05
WASH_DAYS = 31  # re-entry block / disallowance window

RUNTIME = ROOT / "live" / "runtime"

# One manual-tracking system, multiple accounts. Each account has its own state
# files and tax treatment. HSA = tax-free (no wash-sale); taxable = wash-sale applies.
PROFILES = {
    "hsa": {"label": "Fidelity HSA", "broker": "Fidelity", "tax_mode": "HSA", "wash_sale": False},
    "taxable": {"label": "Alpaca taxable", "broker": "Alpaca", "tax_mode": "TAXABLE", "wash_sale": True},
}
ACCOUNT = "hsa"
PROFILE = PROFILES["hsa"]
STATE_PATH = PLAN_PATH = CONTRIB_PATH = TRACKER_PATH = SNAP_PATH = REALIZED_PATH = CHECKPOINT_PATH = None


def set_account(acct: str) -> None:
    """Point all state paths + the active profile at one account."""
    global ACCOUNT, PROFILE, STATE_PATH, PLAN_PATH, CONTRIB_PATH, TRACKER_PATH, SNAP_PATH, REALIZED_PATH, CHECKPOINT_PATH
    if acct not in PROFILES:
        raise SystemExit(f"unknown account '{acct}' — choose from {list(PROFILES)}")
    ACCOUNT = acct
    PROFILE = PROFILES[acct]
    STATE_PATH = RUNTIME / f"{acct}_state.json"
    PLAN_PATH = RUNTIME / f"{acct}_plan.json"
    CONTRIB_PATH = RUNTIME / f"{acct}_contributions.jsonl"
    TRACKER_PATH = RUNTIME / f"{acct}_tracker.json"
    SNAP_PATH = RUNTIME / f"{acct}_snapshots.jsonl"
    REALIZED_PATH = RUNTIME / f"{acct}_realized.jsonl"
    CHECKPOINT_PATH = RUNTIME / f"{acct}_checkpoints.jsonl"


set_account("hsa")


def log_flow(amount: float, kind: str) -> None:
    """Record a dated cash flow so the tracker can compare against benchmarks."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    with CONTRIB_PATH.open("a") as f:
        f.write(json.dumps({"ts": now.isoformat(), "date": now.date().isoformat(),
                            "amount": round(float(amount), 2), "kind": kind}) + "\n")


def read_flows() -> list[dict]:
    if not CONTRIB_PATH.exists():
        return []
    out = []
    for line in CONTRIB_PATH.read_text().strip().splitlines():
        if line:
            r = json.loads(line)
            out.append({"date": r["date"], "amount": float(r["amount"])})
    return out


# ── Realized-gain ledger + wash-sale (taxable) ───────────────────────────────

def _realized_path(acct: str | None) -> Path:
    return (RUNTIME / f"{acct}_realized.jsonl") if acct else REALIZED_PATH


def read_realized(acct: str | None = None) -> list[dict]:
    p = _realized_path(acct)
    if not p or not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().strip().splitlines() if l]


def write_realized(events: list[dict], acct: str | None = None) -> None:
    p = _realized_path(acct)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(e) + "\n" for e in events))


def append_realized(events: list[dict]) -> None:
    if not events:
        return
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with REALIZED_PATH.open("a") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def hsa_holdings() -> set:
    """Tickers currently held in the HSA — for cross-account wash-sale checks."""
    p = RUNTIME / "hsa_state.json"
    if not p.exists():
        return set()
    try:
        st = json.loads(p.read_text())
        return {sl.get("ticker") for sl in st.get("sleeves", {}).values() if sl.get("ticker")}
    except Exception:
        return set()


def disallow_taxable_losses(ticker: str, buy_date) -> None:
    """A buy of `ticker` in ANY account disallows taxable losses on it sold within
    the prior WASH_DAYS. Rewrites the taxable realized ledger (idempotent)."""
    if not ticker:
        return
    events = read_realized("taxable")
    if not events:
        return
    bd = as_date(buy_date)
    changed = False
    for e in events:
        if e.get("ticker") == ticker and e["gain"] < 0 and not e.get("wash_disallowed"):
            if 0 <= (bd - as_date(e["sell_date"])).days <= WASH_DAYS:
                e["wash_disallowed"] = True
                changed = True
    if changed:
        write_realized(events, "taxable")

SLEEVES = ("pod1", "pod2", "gold")
SLEEVE_ASSET = {"pod1": "QQQ", "pod2": "IVV", "gold": "IAU"}
DRIFT_THRESHOLD = 0.05  # portfolio weight drift that triggers a monthly rebalance


# ── State ──────────────────────────────────────────────────────────────────────

def _empty_state() -> dict:
    return {
        "account": ACCOUNT,
        "label": PROFILE["label"],
        "broker": PROFILE["broker"],
        "tax_mode": PROFILE["tax_mode"],
        "updated": datetime.now(timezone.utc).isoformat(),
        "cash": 0.0,
        "sleeves": {s: {"mode": "cash", "ticker": None, "levered": False, "shares": 0.0}
                    for s in SLEEVES},
        "last_rebalance_month": None,
    }


def load_state() -> dict:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        st = _empty_state()
        save_state(st)
        return st
    st = json.loads(STATE_PATH.read_text())
    for s in SLEEVES:
        st.setdefault("sleeves", {}).setdefault(
            s, {"mode": "cash", "ticker": None, "levered": False, "shares": 0.0})
    st.setdefault("cash", 0.0)
    st.setdefault("last_rebalance_month", None)
    return st


def save_state(st: dict) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    st["updated"] = datetime.now(timezone.utc).isoformat()
    STATE_PATH.write_text(json.dumps(st, indent=2, default=str))


# ── Valuation ──────────────────────────────────────────────────────────────────

def last_prices(prices: pd.DataFrame) -> dict[str, float]:
    out = {}
    for t in ("QQQ", "IVV", "IAU", "QLD", "SSO"):
        if t in prices.columns:
            s = prices[t].dropna()
            if len(s):
                out[t] = float(s.iloc[-1])
    return out


def sleeve_value(sl: dict, px: dict[str, float]) -> float:
    tkr = sl.get("ticker")
    if not tkr or tkr not in px:
        return 0.0
    return float(sl.get("shares", 0.0)) * px[tkr]


def _avg_cost(sl: dict) -> float:
    lots = sl.get("lots") or []
    sh = sum(l["shares"] for l in lots)
    return (sum(l["shares"] * l["cost"] for l in lots) / sh) if sh > 0 else 0.0


def total_capital(st: dict, px: dict[str, float]) -> float:
    return float(st.get("cash", 0.0)) + sum(sleeve_value(st["sleeves"][s], px) for s in SLEEVES)


# ── Planning ───────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    sleeve: str
    side: str          # BUY | SELL
    ticker: str
    shares: float
    notional: float
    price: float
    reason: str
    urgent: bool       # CB sells are urgent (place same day)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def is_month_start(day: pd.Timestamp, prices: pd.DataFrame) -> bool:
    idx = prices.dropna(how="all").index
    same_month = idx[(idx.year == day.year) & (idx.month == day.month)]
    return len(same_month) > 0 and day == same_month.min()


def target_mode(sleeve: str, ev: dict) -> tuple[str, bool]:
    """V19d target (mode, levered) for a sleeve from today's signal."""
    if sleeve == "pod1":
        return ev["p1_mode"], ev["p1_lev"]
    if sleeve == "pod2":
        return ev["p2_mode"], ev["p2_lev"]
    return ev["gold_mode"], False


def build_plan(st: dict, ev: dict, px: dict[str, float], day: pd.Timestamp,
               month_start: bool, force: bool = False) -> tuple[list[Trade], dict]:
    """Diff current holdings -> V19d target. CB any day; buys/rebal on month start."""
    cap = total_capital(st, px)
    trades: list[Trade] = []
    notes = {"capital": cap, "month_start": month_start, "day": ev["day"]}

    notes["warnings"] = []
    notes["blocked"] = []
    hsa_held = hsa_holdings() if PROFILE["wash_sale"] else set()

    # 1) Circuit breaker — any day. A held risk sleeve whose asset breached -> cash.
    cb_fired = set()
    for s in SLEEVES:
        sl = st["sleeves"][s]
        held = sl.get("ticker") and sl.get("mode") not in (None, "cash")
        if held and ev["breaches"][SLEEVE_ASSET[s]]:
            tkr = sl["ticker"]
            val = sleeve_value(sl, px)
            trades.append(Trade(s, "SELL", tkr, float(sl.get("shares", 0.0)),
                                val, px.get(tkr, 0.0),
                                f"CIRCUIT BREAKER — {SLEEVE_ASSET[s]} below all 3 SMAs → cash",
                                urgent=True))
            cb_fired.add(s)
            # cross-account wash-sale warning: taxable loss while the HSA holds T
            if PROFILE["wash_sale"] and _avg_cost(sl) and px.get(tkr, 0) < _avg_cost(sl) and tkr in hsa_held:
                notes["warnings"].append(
                    f"{tkr}: this is a LOSS sale, and your HSA holds {tkr}. The HSA's "
                    f"position (or its re-entry) will likely DISALLOW this taxable loss "
                    f"(cross-account wash sale). The estimator won't credit it.")

    if not (month_start or force):
        return trades, notes  # off month-start: only CB acts

    # 2) Monthly rebalance to target modes + 45/45/10 weights.
    # Recompute capital as if CB cash already raised (targets use full capital).
    tgt_notional = {}
    tgt_ticker = {}
    tgt_mode = {}
    tgt_lev = {}
    for s in SLEEVES:
        mode, lev = target_mode(s, ev)
        # a sleeve under CB this run cannot be re-entered until next month start anyway,
        # but V19d re-entry is monthly, so honor the signal target here.
        tgt_mode[s] = mode
        tgt_lev[s] = lev
        tgt_ticker[s] = ticker_for_mode(mode)
        tgt_notional[s] = sleeve_notional(cap, s, mode)

    # drift check across sleeves (portfolio-level, like the backtest)
    cur_w = {s: (sleeve_value(st["sleeves"][s], px) / cap if cap > 0 else 0.0) for s in SLEEVES}
    max_drift = max(abs(cur_w[s] - WEIGHTS[s]) for s in SLEEVES)
    mode_changed = any(st["sleeves"][s].get("mode") != tgt_mode[s]
                       or st["sleeves"][s].get("ticker") != tgt_ticker[s] for s in SLEEVES)
    rebalance = force or mode_changed or max_drift > DRIFT_THRESHOLD
    notes["max_drift"] = max_drift
    notes["rebalance"] = rebalance
    if not rebalance:
        return trades, notes

    for s in SLEEVES:
        if s in cb_fired:
            continue  # already selling to cash this run; don't also re-buy
        sl = st["sleeves"][s]
        cur_tkr = sl.get("ticker")
        cur_sh = float(sl.get("shares", 0.0))
        want_tkr = tgt_ticker[s]
        want_notional = tgt_notional[s]
        # exit current if ticker changes or going to cash
        if cur_tkr and cur_tkr != want_tkr and cur_sh > 0:
            trades.append(Trade(s, "SELL", cur_tkr, cur_sh, sleeve_value(sl, px),
                                px.get(cur_tkr, 0.0),
                                f"rebalance out of {cur_tkr}", urgent=False))
        # wash-sale re-entry block (taxable): don't re-buy a ticker we took a loss on
        _today = datetime.now(timezone.utc).date()
        wb = st.get("wash_until", {}).get(want_tkr)
        if PROFILE["wash_sale"] and want_tkr and wb and as_date(wb) > _today:
            notes["blocked"].append(
                f"{want_tkr}: re-entry blocked until {wb} — loss taken within {WASH_DAYS}d "
                f"(wash sale). Sleeve stays cash.")
            continue
        # enter/resize target
        if want_tkr and want_notional > 0 and want_tkr in px:
            want_sh = round(want_notional / px[want_tkr], 4)
            have_sh = cur_sh if cur_tkr == want_tkr else 0.0
            delta = round(want_sh - have_sh, 4)
            if abs(delta * px[want_tkr]) >= 0.01 * cap or (have_sh == 0 and want_sh > 0):
                side = "BUY" if delta > 0 else "SELL"
                trades.append(Trade(s, side, want_tkr, abs(delta),
                                    abs(delta) * px[want_tkr], px[want_tkr],
                                    f"target {tgt_mode[s]} @ {WEIGHTS[s]:.0%}"
                                    + (" (2×)" if tgt_lev[s] else ""), urgent=False))
    return trades, notes


def apply_plan(st: dict, plan: dict, px: dict[str, float]) -> None:
    """Execute the saved rebalance plan: update FIFO lots, record realized gains,
    arm wash-sale clocks (taxable), disallow washed losses, then reconcile cash."""
    today = datetime.now(timezone.utc).date().isoformat()
    st.setdefault("wash_until", {})
    for s in SLEEVES:
        st["sleeves"].setdefault(s, {"mode": "cash", "ticker": None, "levered": False, "shares": 0.0, "lots": []})
        st["sleeves"][s].setdefault("lots", [])
    buys_today = []
    for tr in plan["trades"]:
        s = tr["sleeve"]
        sl = st["sleeves"][s]
        if tr["side"] == "SELL":
            if sl.get("ticker") == tr["ticker"]:
                events = sell_fifo(sl["lots"], tr["shares"], tr["price"], today)
                for e in events:
                    e["ticker"] = tr["ticker"]
                append_realized(events)
                net_gain = sum(e["gain"] for e in events)
                if PROFILE["wash_sale"] and net_gain < 0:
                    st["wash_until"][tr["ticker"]] = (as_date(today) + timedelta(days=WASH_DAYS)).isoformat()
                sl["shares"] = round(sum(l["shares"] for l in sl["lots"]), 6)
                if sl["shares"] <= 1e-6:
                    sl.update({"shares": 0.0, "ticker": None, "mode": "cash", "levered": False, "lots": []})
        else:  # BUY
            if sl.get("ticker") not in (None, tr["ticker"]):
                sl["lots"] = []  # ticker switched; prior lots were sold above
            sl["ticker"] = tr["ticker"]
            add_lot(sl["lots"], tr["shares"], tr["price"], today)
            sl["shares"] = round(sum(l["shares"] for l in sl["lots"]), 6)
            buys_today.append(tr["ticker"])
        log_fill(sleeve=s, action=("CB_SELL" if tr["urgent"] else tr["side"]),
                 ticker=tr["ticker"], qty=tr["shares"], intended_px=tr["price"],
                 fill_px=tr["price"], dry_run=False, note=f"{ACCOUNT} manual: " + tr["reason"])
    # a buy of T in this account disallows taxable losses on T within the window
    for t in set(buys_today):
        disallow_taxable_losses(t, today)
    # set modes/levered from the plan's targets, then reconcile cash to keep total constant
    for s, tgt in plan.get("targets", {}).items():
        sl = st["sleeves"][s]
        if sl.get("shares", 0.0) > 0:
            sl["mode"] = tgt["mode"]
            sl["levered"] = tgt["levered"]
            sl["ticker"] = tgt["ticker"]
        else:
            sl.update({"mode": "cash", "levered": False, "ticker": None})
    cap = plan["notes"]["capital"]
    st["cash"] = round(cap - sum(sleeve_value(st["sleeves"][s], px) for s in SLEEVES), 2)
    st["last_rebalance_month"] = plan["notes"]["day"][:7] if plan["notes"].get("rebalance") else st.get("last_rebalance_month")


# ── Rendering ──────────────────────────────────────────────────────────────────

def render(st: dict, ev: dict, px: dict[str, float], trades: list[Trade], notes: dict) -> str:
    sc = ev["scores"]
    cap = notes["capital"]
    lines = [
        f"V19d · {PROFILE['label']} · {ev['day']}   account ${cap:,.0f}",
        f"Signal  QQQ {sc['QQQ']}/3   IVV {sc['IVV']}/3   IAU {sc['IAU']}/3"
        + ("   [MONTH-START]" if notes["month_start"] else ""),
    ]
    held = [f"{s}:{st['sleeves'][s].get('ticker') or 'cash'}"
            f"({sleeve_value(st['sleeves'][s], px)/cap:.0%})" if cap else s
            for s in SLEEVES]
    lines.append("Holding " + "  ".join(held))
    warnings = notes.get("warnings") or []
    blocked = notes.get("blocked") or []
    urgent = [t for t in trades if t.urgent]
    normal = [t for t in trades if not t.urgent]
    if urgent:
        lines.append("⚠️  PLACE TODAY (circuit breaker):")
        for t in urgent:
            lines.append(f"   {t.side} {t.shares:g} {t.ticker}  (~${t.notional:,.0f})  — {t.reason}")
    if normal:
        lines.append(f"PLACE AT {PROFILE['broker'].upper()} (monthly rebalance):")
        for t in normal:
            lines.append(f"   {t.side} {t.shares:g} {t.ticker} @~${t.price:,.2f}  (~${t.notional:,.0f})  — {t.reason}")
    for b in blocked:
        lines.append(f"BLOCKED  {b}")
    for w in warnings:
        lines.append(f"⚠️  WASH SALE  {w}")
    if not trades and not blocked and not warnings:
        lines.append("ACTION: none — hold. (No CB; "
                     + ("no rebalance needed." if notes["month_start"] else "not month-start.") + ")")
    elif trades:
        lines.append(f"After placing: .venv/bin/python -m live.fidelity confirm --account {ACCOUNT}")
    return "\n".join(lines)


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_plan(args) -> None:
    prices = load_prices(args.live_quotes)
    day = prices.dropna(how="all").index.max()
    smas = compute_smas(prices[["QQQ", "IVV", "IAU"]].dropna(how="all"))
    ev = evaluate_day(day, prices, smas)
    px = last_prices(prices)
    st = load_state()
    ms = is_month_start(day, prices)
    already = st.get("last_rebalance_month") == ev["day"][:7]
    trades, notes = build_plan(st, ev, px, day, month_start=ms and not already, force=args.force)
    text = render(st, ev, px, trades, notes)
    # persist plan for confirm
    targets = {s: {"mode": target_mode(s, ev)[0], "levered": target_mode(s, ev)[1],
                   "ticker": ticker_for_mode(target_mode(s, ev)[0])} for s in SLEEVES}
    PLAN_PATH.write_text(json.dumps(
        {"type": "rebalance", "day": ev["day"], "scores": ev["scores"],
         "trades": [t.as_dict() for t in trades],
         "targets": targets, "notes": notes, "prices": px}, indent=2, default=str))
    deliver(text)


def build_withdrawal(st: dict, px: dict[str, float], amount: float) -> tuple[list[Trade], dict]:
    """Raise `amount` in cash: draw from cash first, then sell sleeves pro-rata."""
    cap = total_capital(st, px)
    if amount > cap + 1e-6:
        raise SystemExit(f"Cannot withdraw ${amount:,.2f} — account is only ${cap:,.2f}.")
    from_cash = min(float(st.get("cash", 0.0)), amount)
    remainder = round(amount - from_cash, 2)
    trades: list[Trade] = []
    if remainder > 0.01:
        held = {s: sleeve_value(st["sleeves"][s], px) for s in SLEEVES}
        tot = sum(held.values())
        for s in SLEEVES:
            if held[s] <= 0 or tot <= 0:
                continue
            raise_s = remainder * held[s] / tot
            tkr = st["sleeves"][s]["ticker"]
            p = px.get(tkr, 0.0)
            if p <= 0:
                continue
            sh = round(raise_s / p, 4)
            trades.append(Trade(s, "SELL", tkr, sh, sh * p, p,
                                f"withdrawal: raise ${raise_s:,.0f} from {tkr}", urgent=False))
    notes = {"capital": cap, "withdraw_amount": amount, "from_cash": from_cash,
             "day": datetime.now(timezone.utc).date().isoformat()}
    return trades, notes


def cmd_withdraw(args) -> None:
    prices = load_prices(False)
    px = last_prices(prices)
    st = load_state()
    trades, notes = build_withdrawal(st, px, args.amount)
    cap = notes["capital"]
    lines = [f"V19d · Fidelity HSA · WITHDRAWAL ${args.amount:,.0f}   (account ${cap:,.0f} → ${cap-args.amount:,.0f})"]
    if notes["from_cash"] > 0:
        lines.append(f"   ${notes['from_cash']:,.0f} from cash on hand")
    if trades:
        lines.append("SELL AT FIDELITY, then withdraw the cash:")
        for t in trades:
            lines.append(f"   SELL {t.shares:g} {t.ticker} @~${t.price:,.2f}  (~${t.notional:,.0f})")
    else:
        lines.append("   Fully covered by cash — no sells needed.")
    lines.append("Reminder: non-medical HSA withdrawals before 65 owe income tax + 20% penalty; "
                 "qualified medical is tax-free.")
    lines.append("After selling + withdrawing: .venv/bin/python -m live.fidelity confirm")
    PLAN_PATH.write_text(json.dumps(
        {"type": "withdraw", "day": notes["day"], "amount": args.amount,
         "trades": [t.as_dict() for t in trades], "notes": notes, "prices": px},
        indent=2, default=str))
    deliver("\n".join(lines))


def cmd_confirm(args) -> None:
    if not PLAN_PATH.exists():
        raise SystemExit("No saved plan. Run `plan` or `withdraw` first.")
    plan = json.loads(PLAN_PATH.read_text())
    if plan.get("confirmed"):
        print("This plan is already confirmed.")
        return
    st = load_state()
    px = plan.get("prices", {})
    if plan.get("type") == "withdraw":
        # apply sells (proceeds to cash + realized gains), then remove withdrawn cash
        today = datetime.now(timezone.utc).date().isoformat()
        st.setdefault("wash_until", {})
        for tr in plan["trades"]:
            sl = st["sleeves"][tr["sleeve"]]
            sl.setdefault("lots", [])
            if sl.get("ticker") == tr["ticker"]:
                events = sell_fifo(sl["lots"], tr["shares"], tr["price"], today)
                for e in events:
                    e["ticker"] = tr["ticker"]
                append_realized(events)
                net_gain = sum(e["gain"] for e in events)
                if PROFILE["wash_sale"] and net_gain < 0:
                    st["wash_until"][tr["ticker"]] = (as_date(today) + timedelta(days=WASH_DAYS)).isoformat()
                st["cash"] = round(float(st.get("cash", 0.0)) + tr["shares"] * tr["price"], 2)
                sl["shares"] = round(sum(l["shares"] for l in sl["lots"]), 6)
                if sl["shares"] <= 1e-6:
                    sl.update({"shares": 0.0, "ticker": None, "mode": "cash", "levered": False, "lots": []})
        st["cash"] = round(float(st.get("cash", 0.0)) - plan["amount"], 2)
        save_state(st)
        log_flow(-plan["amount"], "withdrawal")
        print(f"Withdrew ${plan['amount']:,.2f}. New state:")
        cmd_status(args, reuse=st)
    else:
        if not plan["trades"]:
            print("Plan had no trades — nothing to confirm.")
            return
        apply_plan(st, plan, px)
        save_state(st)
        print(f"Confirmed {len(plan['trades'])} trade(s). New state:")
        cmd_status(args, reuse=st)
    PLAN_PATH.write_text(json.dumps({**plan, "confirmed": datetime.now(timezone.utc).isoformat()},
                                    indent=2, default=str))


def cmd_add_cash(args) -> None:
    st = load_state()
    st["cash"] = round(float(st.get("cash", 0.0)) + args.amount, 2)
    save_state(st)
    log_flow(args.amount, "contribution")
    print(f"cash += ${args.amount:,.2f} → ${st['cash']:,.2f}. "
          f"Deploys on next month-start rebalance.")


def cmd_set_cash(args) -> None:
    st = load_state()
    st["cash"] = round(args.amount, 2)
    save_state(st)
    print(f"cash set to ${st['cash']:,.2f}")


def cmd_status(args, reuse: dict | None = None) -> None:
    st = reuse or load_state()
    prices = load_prices(False)
    px = last_prices(prices)
    cap = total_capital(st, px)
    print(f"{PROFILE['label']} · account={ACCOUNT} · tax_mode {PROFILE['tax_mode']}")
    print(f"  total ${cap:,.2f}  (cash ${st.get('cash', 0.0):,.2f})")
    for s in SLEEVES:
        sl = st["sleeves"][s]
        val = sleeve_value(sl, px)
        w = val / cap if cap else 0.0
        print(f"  {s:<5} {sl.get('ticker') or 'cash':<5} {sl.get('shares', 0.0):>10.4f} sh"
              f"  ${val:>10,.2f}  {w:>5.1%}  (target {WEIGHTS[s]:.0%})")


TKR_TO_SLEEVE = {"QLD": "pod1", "QQQ": "pod1", "SSO": "pod2", "IVV": "pod2", "IAU": "gold"}
MODE_FOR = {"QLD": "qld", "QQQ": "qqq", "SSO": "sso", "IVV": "ivv", "IAU": "iau"}


def _num(x):
    try:
        return float(str(x).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def parse_broker_csv(csv_path: str) -> tuple[dict, float]:
    """Parse a Fidelity/Robinhood positions CSV -> ({ticker:{shares,cost,value}}, cash).

    Liberal column matching. Cash rows (CASH/FDIC/SPAXX/money market) sum from the
    value column.
    """
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    def col(*subs):
        for c in df.columns:
            cl = c.lower()
            if any(s in cl for s in subs):
                return c
        return None

    sym_col = col("symbol", "ticker")
    qty_col = col("quantity", "shares", "qty")
    percost_col = col("average cost basis", "cost basis per share", "average cost", "avg cost")
    totcost_col = col("cost basis total", "total cost", "cost basis")
    val_col = col("current value", "market value", "value")
    if not sym_col or not qty_col:
        raise SystemExit(f"CSV missing Symbol/Quantity columns; found {list(df.columns)}")

    holdings, cash = {}, 0.0
    for _, r in df.iterrows():
        sym = str(r[sym_col]).strip().upper()
        qty = _num(r[qty_col])
        if sym in TKR_TO_SLEEVE and qty:
            cost = _num(r[percost_col]) if percost_col else None
            if cost is None and totcost_col:
                tot = _num(r[totcost_col])
                cost = (tot / qty) if (tot and qty) else None
            holdings[sym] = {"shares": round(qty, 6), "cost": cost,
                             "value": _num(r[val_col]) if val_col else None}
        elif "CASH" in sym or "FDIC" in sym or sym.startswith("SPAXX") or "MONEY MARKET" in sym:
            v = _num(r[val_col]) if val_col else None
            if v:
                cash += v
    return holdings, round(cash, 2)


def _append_checkpoint(row: dict) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def cmd_positions(args) -> None:
    """Overwrite tracked holdings from a broker CSV (simple import)."""
    holdings, cash = parse_broker_csv(args.csv)
    st = load_state()
    today = datetime.now(timezone.utc).date().isoformat()
    for s in SLEEVES:
        st["sleeves"][s].update({"ticker": None, "shares": 0.0, "lots": []})
    for sym, h in holdings.items():
        s = TKR_TO_SLEEVE[sym]
        st["sleeves"][s].update({
            "ticker": sym, "shares": h["shares"], "mode": MODE_FOR[sym],
            "levered": sym in ("QLD", "SSO"),
            "lots": [{"shares": h["shares"], "cost": h["cost"] or 0.0, "date": today}]})
    if cash:
        st["cash"] = cash
    save_state(st)
    print(f"Imported {args.csv}: {[(k, v['shares']) for k, v in holdings.items()]}  cash ${st['cash']:,.2f}")
    cmd_status(args, reuse=st)


def reconcile_csv(csv_path: str) -> dict:
    """Diff tracked state vs a broker CSV, apply broker-as-truth, record a checkpoint."""
    holdings, cash = parse_broker_csv(csv_path)
    st = load_state()
    st.setdefault("wash_until", {})
    today = datetime.now(timezone.utc).date().isoformat()
    broker_by_sleeve: dict = {}
    for sym, h in holdings.items():
        broker_by_sleeve.setdefault(TKR_TO_SLEEVE[sym], {})[sym] = h

    diffs = []
    for s in SLEEVES:
        sl = st["sleeves"][s]
        cur_tkr = sl.get("ticker")
        cur_sh = round(float(sl.get("shares", 0.0)), 4)
        bh = broker_by_sleeve.get(s, {})
        btkr = next(iter(bh), None)
        bsh = round(bh[btkr]["shares"], 4) if btkr else 0.0
        bcost = bh[btkr]["cost"] if btkr else None
        matched = (cur_tkr == btkr) and abs(cur_sh - bsh) < 1e-4
        diffs.append({"sleeve": s, "system_ticker": cur_tkr, "system_shares": cur_sh,
                      "broker_ticker": btkr, "broker_shares": bsh,
                      "delta": round(bsh - cur_sh, 4), "status": "match" if matched else "drift"})
        if matched:
            if bcost and sl.get("lots"):
                for lot in sl["lots"]:
                    lot["cost"] = bcost
            continue
        if btkr:
            keep_date = (sl["lots"][0]["date"] if sl.get("lots") and cur_tkr == btkr and sl["lots"] else today)
            cost = bcost if bcost else (_avg_cost(sl) or 0.0)
            sl.update({"ticker": btkr, "shares": bsh, "mode": MODE_FOR[btkr],
                       "levered": btkr in ("QLD", "SSO"),
                       "lots": [{"shares": bsh, "cost": cost, "date": keep_date}]})
        else:
            sl.update({"ticker": None, "shares": 0.0, "mode": "cash", "levered": False, "lots": []})

    cash_before = round(float(st.get("cash", 0.0)), 2)
    if cash:
        st["cash"] = cash
    save_state(st)
    _append_checkpoint({"ts": datetime.now(timezone.utc).isoformat(),
                        "source": os.path.basename(csv_path),
                        "holdings": holdings, "cash": cash, "diffs": diffs})
    return {"account": ACCOUNT, "source": os.path.basename(csv_path),
            "cash_before": cash_before, "cash_after": round(float(st.get("cash", 0.0)), 2),
            "diffs": diffs,
            "n_match": sum(1 for d in diffs if d["status"] == "match"),
            "n_drift": sum(1 for d in diffs if d["status"] == "drift")}


def cmd_reconcile(args) -> None:
    result = reconcile_csv(args.csv)
    if getattr(args, "json", False):
        print(json.dumps(result))
        return
    print(f"Reconciled {ACCOUNT} against {result['source']}: "
          f"{result['n_match']} match, {result['n_drift']} drift")
    for d in result["diffs"]:
        tag = "OK   " if d["status"] == "match" else "DRIFT"
        print(f"  [{tag}] {d['sleeve']:<5} system {d['system_ticker'] or 'cash'} {d['system_shares']:g}"
              f"  →  broker {d['broker_ticker'] or 'cash'} {d['broker_shares']:g}  (Δ{d['delta']:+g})")
    print(f"  cash ${result['cash_before']:,.2f} → ${result['cash_after']:,.2f}")


# ── Tracker payload (for the local Node dashboard) ───────────────────────────
# Market data = yfinance (the same source the signals use). Account data = the
# tracked state (seeded from the Fidelity CSV via `positions`). No third feed.

def _yf_series(tickers: list[str], start: str) -> dict:
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            d = yf.download(t, start=start, progress=False, auto_adjust=True)
        except Exception:
            d = None
        if d is None or getattr(d, "empty", True):
            out[t] = None
            continue
        p = d["Close"]
        if hasattr(p, "columns"):
            p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        out[t] = p.dropna()
    return out


def _asof(series, date) -> float | None:
    if series is None or len(series) == 0:
        return None
    s = series.loc[: pd.Timestamp(date)]
    return float(s.iloc[-1]) if len(s) else None


def _now(series) -> float | None:
    return float(series.iloc[-1]) if series is not None and len(series) else None


def _flow_value(flows: list[dict], series) -> float | None:
    """Value of the same dated cash flows if invested in one buy-and-hold asset."""
    if series is None or len(series) == 0:
        return None
    now = _now(series)
    units = 0.0
    for f in flows:
        p = _asof(series, f["date"])
        if p:
            units += f["amount"] / p
    return round(units * now, 2)


def _blend_value(flows, spy, tlt) -> float | None:
    if spy is None or tlt is None or not len(spy) or not len(tlt):
        return None
    sn, tn = _now(spy), _now(tlt)
    su = tu = 0.0
    for f in flows:
        sp, tp = _asof(spy, f["date"]), _asof(tlt, f["date"])
        if sp:
            su += 0.6 * f["amount"] / sp
        if tp:
            tu += 0.4 * f["amount"] / tp
    return round(su * sn + tu * tn, 2)


def _read_snaps() -> list[dict]:
    if not SNAP_PATH.exists():
        return []
    return [json.loads(l) for l in SNAP_PATH.read_text().strip().splitlines() if l]


def _append_snap(row: dict) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    snaps = _read_snaps()
    if snaps and snaps[-1]["date"] == row["date"]:
        snaps[-1] = row
        SNAP_PATH.write_text("\n".join(json.dumps(r) for r in snaps) + "\n")
    else:
        with SNAP_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")


def build_tracker_payload() -> dict:
    st = load_state()
    flows = read_flows()
    contributed = round(sum(f["amount"] for f in flows), 2)
    first = flows[0]["date"] if flows else datetime.now(timezone.utc).date().isoformat()

    acct = sorted({sl.get("ticker") for sl in st["sleeves"].values() if sl.get("ticker")})
    series = _yf_series(sorted(set(acct) | {"SPY", "QQQ", "TLT"}), first)
    cur = {t: _now(series.get(t)) for t in series}

    sleeves = {}
    holdings_value = 0.0
    for s in SLEEVES:
        sl = st["sleeves"][s]
        t = sl.get("ticker")
        p = cur.get(t)
        val = round(float(sl.get("shares", 0.0)) * p, 2) if (t and p) else 0.0
        holdings_value += val
        sleeves[s] = {"ticker": t, "shares": float(sl.get("shares", 0.0)), "price": p,
                      "value": val, "mode": sl.get("mode"), "target": WEIGHTS[s]}
    cash = float(st.get("cash", 0.0))
    v19d = round(holdings_value + cash, 2)

    benches = {"spy": _flow_value(flows, series.get("SPY")),
               "qqq": _flow_value(flows, series.get("QQQ")),
               "blend": _blend_value(flows, series.get("SPY"), series.get("TLT"))}
    plan = json.loads(PLAN_PATH.read_text()) if PLAN_PATH.exists() else {}

    _append_snap({"date": datetime.now(timezone.utc).date().isoformat(),
                  "contributed": contributed, "v19d": v19d, **benches})

    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "account": ACCOUNT, "label": PROFILE["label"], "broker": PROFILE["broker"],
        "tax_mode": PROFILE["tax_mode"], "wash_sale": PROFILE["wash_sale"],
        "v19d": v19d, "cash": cash, "holdingsValue": round(holdings_value, 2),
        "contributed": contributed, "net_gain": round(v19d - contributed, 2),
        "sleeves": sleeves, "benches": benches,
        "vs": {k: (round(v19d - v, 2) if v is not None else None) for k, v in benches.items()},
        "signal": {"day": plan.get("day"), "scores": plan.get("scores")},
        "seeded": bool(flows), "firstDate": first, "history": _read_snaps(),
        "tax_estimate": (estimate_tax(read_realized(), datetime.now(timezone.utc).year,
                                      TAX_ORDINARY, TAX_LTCG, TAX_STATE)
                         if PROFILE["wash_sale"] else None),
        "warnings": plan.get("notes", {}).get("warnings", []),
        "blocked": plan.get("notes", {}).get("blocked", []),
    }


def cmd_refresh(args) -> None:
    payload = build_tracker_payload()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(json.dumps(payload, indent=2, default=str))
    b = payload["benches"]
    print(f"Wrote {TRACKER_PATH.name}. V19d ${payload['v19d']:,.0f}  |  "
          f"SPY ${b['spy'] or 0:,.0f}  QQQ ${b['qqq'] or 0:,.0f}  60/40 ${b['blend'] or 0:,.0f}  "
          f"(same ${payload['contributed']:,.0f} contributed)")


def main() -> None:
    ap = argparse.ArgumentParser(description="V19d manual planner — multi-account (HSA / taxable)")
    # shared: every subcommand accepts --account
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--account", default="hsa", choices=list(PROFILES),
                        help="which account (default: hsa)")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("plan", parents=[parent], help="compute today's trade plan + alert")
    p.add_argument("--live-quotes", action="store_true")
    p.add_argument("--force", action="store_true", help="force a full rebalance")
    p.set_defaults(func=cmd_plan)

    c = sub.add_parser("confirm", parents=[parent], help="mark the saved plan executed")
    c.set_defaults(func=cmd_confirm)

    a = sub.add_parser("add-cash", parents=[parent], help="record a contribution")
    a.add_argument("amount", type=float)
    a.set_defaults(func=cmd_add_cash)

    w = sub.add_parser("withdraw", parents=[parent], help="raise cash for a withdrawal (cash first, then pro-rata sells)")
    w.add_argument("amount", type=float)
    w.set_defaults(func=cmd_withdraw)

    sc = sub.add_parser("set-cash", parents=[parent], help="set cash balance")
    sc.add_argument("amount", type=float)
    sc.set_defaults(func=cmd_set_cash)

    s = sub.add_parser("status", parents=[parent], help="show holdings")
    s.set_defaults(func=cmd_status)

    po = sub.add_parser("positions", parents=[parent], help="overwrite holdings from a broker positions CSV")
    po.add_argument("csv")
    po.set_defaults(func=cmd_positions)

    rc = sub.add_parser("reconcile", parents=[parent], help="diff state vs a broker CSV checkpoint (broker = truth)")
    rc.add_argument("csv")
    rc.add_argument("--json", action="store_true", help="emit JSON (for the tracker)")
    rc.set_defaults(func=cmd_reconcile)

    rf = sub.add_parser("refresh", parents=[parent], help="write the tracker JSON (account vs benchmarks) for the Node app")
    rf.set_defaults(func=cmd_refresh)

    args = ap.parse_args()
    if not getattr(args, "cmd", None):
        args = ap.parse_args(["plan", "--account", "hsa"])
    set_account(args.account)
    args.func(args)


if __name__ == "__main__":
    main()
