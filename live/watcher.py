"""Daily V19d watcher — scores, CB actions, dry-run fills.

Usage:
  .venv/bin/python -m live.watcher
  .venv/bin/python -m live.watcher --live-quotes   # yfinance last
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

from live.broker import get_broker
from live.ledger import log_fill, log_signal
from live.signals import compute_smas, evaluate_day, ticker_for_mode
from live.state import arm_wash_sale, load_state, save_state, transition_on_signal


def load_prices(use_live: bool) -> pd.DataFrame:
    """Prefer experiment yf cache; optionally refresh last prints via yfinance."""
    cache = ROOT / "experiments" / "v19d_marketstack_verification" / "yf_cache.parquet"
    if not cache.exists():
        raise SystemExit(
            f"Missing {cache}. Run: .venv/bin/python experiments/v19d_marketstack_verification/backtest.py"
        )
    raw = pd.read_parquet(cache)
    # Map to strategy names
    colmap = {"QQQ": "QQQ", "SPY": "IVV", "GLD": "IAU", "QLD": "QLD", "SSO": "SSO", "TLT": "VGLT"}
    prices = pd.DataFrame({dst: raw[src] for src, dst in colmap.items() if src in raw.columns})
    prices.index = pd.to_datetime(prices.index)
    if use_live:
        import yfinance as yf

        for sym, col in [("QQQ", "QQQ"), ("SPY", "IVV"), ("GLD", "IAU"), ("QLD", "QLD"), ("SSO", "SSO")]:
            d = yf.download(sym, period="5d", progress=False, auto_adjust=True)
            if d is not None and not d.empty:
                p = d["Close"]
                if hasattr(p, "columns"):
                    p = p.iloc[:, 0]
                p.index = pd.to_datetime(p.index).tz_localize(None)
                # update/append last
                for dt, val in p.items():
                    prices.loc[dt, col] = float(val)
        prices = prices.sort_index()
    return prices


def run(use_live_quotes: bool = False, execute_cb: bool = True) -> dict:
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

    results = []
    for act in actions:
        if act["action"] == "CB_SELL" and execute_cb and act.get("ticker"):
            tkr = act["ticker"]
            q = broker.get_quote(tkr)
            # Stage policy: CB pre-authorized; qty placeholder 1 share in dry-run
            qty = 1.0
            limit = q.last * 0.9985 if q.last else None  # -15 bps
            order = broker.place_equity_order(tkr, "sell", qty, "limit", limit)
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
            # Mark flat + wash if loss vs last_fill
            sl = st.sleeves[act["sleeve"]]
            last = sl.get("last_fill_px")
            loss = bool(last is not None and order.fill_px is not None and order.fill_px < float(last))
            if st.tax_mode.startswith("TAXABLE") and loss:
                arm_wash_sale(sl, loss=True)
            elif st.tax_mode.startswith("TAXABLE") and last is None:
                # No cost basis yet — arm conservatively on first CB exit
                arm_wash_sale(sl, loss=True)
            sl["status"] = "FLAT"
            sl["mode"] = "cash"
            sl["levered"] = False
            sl["ticker"] = None
            sl["last_fill_px"] = order.fill_px
            results.append({"action": act, "order": order.__dict__, "fill": fill})
        else:
            results.append({"action": act, "order": None})

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
        "actions": actions,
        "executions": results,
        "dry_run": st.dry_run,
        "stage": st.stage,
    }
    return report


def main():
    ap = argparse.ArgumentParser(description="V19d live watcher")
    ap.add_argument("--live-quotes", action="store_true")
    ap.add_argument("--no-exec", action="store_true", help="detect only, no dry-run CB fills")
    args = ap.parse_args()
    report = run(use_live_quotes=args.live_quotes, execute_cb=not args.no_exec)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
