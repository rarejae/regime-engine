"""V19d: V19 with Circuit Breaker on Gold Sleeve.

Adds 3/3 SMA breach CB to IAU: if IAU closes below all 3 daily SMAs,
exit to cash, re-entry at monthly rebalance only.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from experiments.v11_beta_scaled.backtest import (
    SMA_PERIODS, SSO_EXP, QLD_EXP,
    load_data, asset_score, check_breach, lev_ret,
    run_baseline, run_v9,
    cagr, max_dd, sharpe_r, sortino_r, calmar_r, dca_terminal, metrics_row,
)
from experiments.v19_cb_cash_exit.backtest import run_v19


def run_v19d(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
             start_date, capture_events=False):
    """V19 + gold CB."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb1 = 0; cb2 = 0; cb_g = 0
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10

    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_mode = "cash"; gold_delev = False
    scores = {}
    gold_cb_events = []

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False; gold_delev = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = {a: asset_score(sd, a, dpdf, daily_smas) for a in ["QQQ", "IVV", "IAU"]}
            sc_q = scores["QQQ"]; sc_i = scores["IVV"]

            if sc_q >= 3:
                if sc_i <= 1: p1_mode = "qqq"; p1_lev = False
                else: p1_mode = "qld"; p1_lev = True
            elif sc_q == 2: p1_mode = "qqq_partial"; p1_lev = False
            else: p1_mode = "cash"; p1_lev = False

            if sc_i >= 3: p2_mode = "sso"; p2_lev = True
            elif sc_i == 2: p2_mode = "ivv_partial"; p2_lev = False
            else: p2_mode = "cash"; p2_lev = False

            gold_mode = "iau" if scores["IAU"] >= 3 else "cash"

            if day != trading_days[0]:
                total = nav1 + nav2 + nav_g
                if total > 0:
                    drift = max(abs(nav1/total - 0.45), abs(nav2/total - 0.45), abs(nav_g/total - 0.10))
                    if drift > 0.05:
                        nav1 = total * 0.45; nav2 = total * 0.45; nav_g = total * 0.10

        # Equity CBs → cash
        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; cb1 += 1; p1_mode = "cash"
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; cb2 += 1; p2_mode = "cash"

        # Gold CB → cash (V19d addition)
        if gold_mode == "iau" and not gold_delev:
            if check_breach(day, "IAU", dpdf, daily_smas):
                gold_mode = "cash"; gold_delev = True; cb_g += 1
                if capture_events:
                    gold_cb_events.append({"date": day})

        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
        iau_u = float(dr.get("IAU", 0.0)) if pd.notna(dr.get("IAU", np.nan)) else 0.0

        if p1_mode == "qld":
            r1 = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start) if p1_lev else qqq_u
        elif p1_mode == "qqq": r1 = qqq_u
        elif p1_mode == "qqq_partial": r1 = 0.70 * qqq_u + 0.30 * rfr
        else: r1 = rfr

        if p2_mode == "sso":
            r2 = lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start) if p2_lev else ivv_u
        elif p2_mode == "ivv": r2 = ivv_u
        elif p2_mode == "ivv_partial": r2 = 0.70 * ivv_u + 0.30 * rfr
        else: r2 = rfr

        rg = iau_u if gold_mode == "iau" else rfr

        prev_total = nav1 + nav2 + nav_g
        nav1 *= (1 + r1); nav2 *= (1 + r2); nav_g *= (1 + rg)
        new_total = nav1 + nav2 + nav_g
        port[day] = new_total / prev_total - 1 if prev_total > 0 else 0

    return pd.Series(port).sort_index(), cb1 + cb2, cb_g, gold_cb_events


def main():
    print("=" * 140)
    print("  V19d: V19 WITH GOLD CIRCUIT BREAKER — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    print("  Running V19d...")
    v19d_full, v19d_eq_cb, v19d_g_cb, g_events = run_v19d(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
        "2002-01-01", capture_events=True)

    print("  Running V19 (control)...")
    v19_full, _, _, _ = run_v19(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")

    print("  Running Baseline...")
    bl_full, bl_cb = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                   dbmf_ret, dbmf_inception, "2002-01-01")
    qqq_full = daily_ret["QQQ"].loc["2002-01-01":"2026-03-31"].dropna()

    # ── TABLE 1 ──
    print(f"\n{'=' * 140}")
    print("  TABLE 1: CORE METRICS (2002-2026)")
    print(f"{'=' * 140}")
    print(f"\n  {'Strategy':<24} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9} {'EqCB':>5} {'AuCB':>5}")
    print(f"  {'-' * 24} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 5} {'-' * 5}")
    for nm, s, ecb, gcb in [("V19d (gold CB)", v19d_full, v19d_eq_cb, v19d_g_cb),
                              ("V19 (no gold CB)", v19_full, 27, 0)]:
        c = cagr(s); v = s.std()*np.sqrt(252); sh = sharpe_r(s); so = sortino_r(s)
        dd = max_dd(s); cl = calmar_r(s); t = (1+s).cumprod().iloc[-1]
        sm = s.resample("MS").apply(lambda x: (1+x).prod()-1); dca = dca_terminal(sm)
        print(f"  {nm:<24} {c:>6.2%} {v:>6.2%} {sh:>7.3f} {so:>8.3f} {dd:>6.1%} {cl:>7.2f} ${t:>8.2f} ${dca/1e6:>7.2f}M {ecb:>5} {gcb:>5}")
    print(metrics_row("Baseline (Sweep-40)   ", bl_full, bl_cb))

    # Gold CB events
    print(f"\n  Gold CB events: {v19d_g_cb}")
    for e in g_events:
        print(f"    {e['date'].strftime('%Y-%m-%d')}")

    # ── TABLE 2: Crisis drawdowns ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: CRISIS DRAWDOWNS")
    print(f"{'=' * 140}")
    crises = [("Dot-com","2002-01-01","2003-03-31"),("GFC","2007-11-01","2009-03-31"),
              ("COVID","2020-02-01","2020-04-30"),("2022","2022-01-01","2022-12-31")]
    names = ["V19d", "V19", "BL"]
    all_s = [v19d_full, v19_full, bl_full]
    print(f"\n  {'Crisis':<18}" + "".join(f"{nm:>10}" for nm in names))
    for label, cs, ce in crises:
        cells = [max_dd(s[(s.index >= cs) & (s.index <= ce)]) if len(s[(s.index >= cs) & (s.index <= ce)]) > 5 else 0 for s in all_s]
        print(f"  {label:<18}" + "".join(f"{c:>10.1%}" for c in cells))

    # ── Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<24}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    for nm, fn in [("V19d (gold CB)", "v19d"), ("V19 (no gold CB)", "v19")]:
        row = f"  {nm:<24}"
        for sd in start_dates:
            if fn == "v19d":
                s, _, _, _ = run_v19d(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            else:
                s, _, _, _ = run_v19(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            row += f"{cagr(s):>10.2%}"
        print(row)

    # ── Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: PASS / FAIL")
    print(f"{'=' * 140}")
    v19d_c = cagr(v19d_full); v19d_sh = sharpe_r(v19d_full); v19d_dd = max_dd(v19d_full)
    v19_c = cagr(v19_full); v19_sh = sharpe_r(v19_full); v19_dd = max_dd(v19_full)

    print(f"\n  V19d: CAGR {v19d_c:.2%}, Sharpe {v19d_sh:.3f}, MaxDD {v19d_dd:.1%}")
    print(f"  V19:  CAGR {v19_c:.2%}, Sharpe {v19_sh:.3f}, MaxDD {v19_dd:.1%}")

    sh_ok = v19d_sh >= v19_sh; dd_ok = v19d_dd >= v19_dd
    cagr_ok = v19d_c >= v19_c - 0.002
    wash = abs(v19d_sh - v19_sh) < 0.005 and abs(v19d_dd - v19_dd) < 0.01

    print(f"\n  Sharpe: {v19d_sh:.3f} vs {v19_sh:.3f} → {'✓' if sh_ok else '✗'}")
    print(f"  MaxDD:  {v19d_dd:.1%} vs {v19_dd:.1%} → {'✓' if dd_ok else '✗'}")
    print(f"  CAGR:   {v19d_c:.2%} vs {v19_c:.2%} (within 0.2pp) → {'✓' if cagr_ok else '✗'}")

    if wash:
        print(f"\n  → WASH — adopt V19d for design consistency")
    elif sh_ok and dd_ok and cagr_ok:
        print(f"\n  → V19d PASSES — gold CB improves or maintains")
    else:
        print(f"\n  → V19d {'improves' if sh_ok else 'worsens'} Sharpe, "
              f"{'improves' if dd_ok else 'worsens'} DD — keep V19")

    print()


if __name__ == "__main__":
    main()
