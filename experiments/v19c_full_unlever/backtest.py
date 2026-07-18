"""V19c: V19 with 100% unlevered at score 2/3 (instead of 70/30)."""

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


def run_v19c(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, start_date):
    """V19 with score 2/3 → 100% underlying (not 70/30)."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb1 = 0; cb2 = 0
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10
    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_mode = "cash"; scores = {}
    score2_months_q = 0; score2_months_i = 0

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = {a: asset_score(sd, a, dpdf, daily_smas) for a in ["QQQ", "IVV", "IAU"]}
            sc_q = scores["QQQ"]; sc_i = scores["IVV"]

            # Pod 1 — score 2 → 100% QQQ (V19c change)
            if sc_q >= 3:
                if sc_i <= 1: p1_mode = "qqq"; p1_lev = False
                else: p1_mode = "qld"; p1_lev = True
            elif sc_q == 2:
                p1_mode = "qqq"; p1_lev = False  # V19c: 100% QQQ (was 70/30)
                score2_months_q += 1
            else: p1_mode = "cash"; p1_lev = False

            # Pod 2 — score 2 → 100% IVV (V19c change)
            if sc_i >= 3: p2_mode = "sso"; p2_lev = True
            elif sc_i == 2:
                p2_mode = "ivv"; p2_lev = False  # V19c: 100% IVV (was 70/30)
                score2_months_i += 1
            else: p2_mode = "cash"; p2_lev = False

            gold_mode = "iau" if scores["IAU"] >= 3 else "cash"

            if day != trading_days[0]:
                total = nav1 + nav2 + nav_g
                if total > 0:
                    drift = max(abs(nav1/total - 0.45), abs(nav2/total - 0.45), abs(nav_g/total - 0.10))
                    if drift > 0.05:
                        nav1 = total * 0.45; nav2 = total * 0.45; nav_g = total * 0.10

        # CB → cash
        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; cb1 += 1; p1_mode = "cash"
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; cb2 += 1; p2_mode = "cash"

        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
        iau_u = float(dr.get("IAU", 0.0)) if pd.notna(dr.get("IAU", np.nan)) else 0.0

        if p1_mode == "qld":
            r1 = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start) if p1_lev else qqq_u
        elif p1_mode == "qqq": r1 = qqq_u
        else: r1 = rfr

        if p2_mode == "sso":
            r2 = lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start) if p2_lev else ivv_u
        elif p2_mode == "ivv": r2 = ivv_u
        else: r2 = rfr

        rg = iau_u if gold_mode == "iau" else rfr

        prev_total = nav1 + nav2 + nav_g
        nav1 *= (1 + r1); nav2 *= (1 + r2); nav_g *= (1 + rg)
        new_total = nav1 + nav2 + nav_g
        port[day] = new_total / prev_total - 1 if prev_total > 0 else 0

    return pd.Series(port).sort_index(), cb1 + cb2, score2_months_q, score2_months_i


def main():
    print("=" * 140)
    print("  V19c: 100% UNLEVERED AT SCORE 2/3 — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    print("  Running V19c...")
    v19c_full, v19c_cb, s2q, s2i = run_v19c(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")

    print("  Running V19 (control)...")
    v19_full, _, _, _ = run_v19(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")

    print("  Running V9...")
    v9_full, v9_cb = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running Baseline...")
    bl_full, bl_cb = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                   dbmf_ret, dbmf_inception, "2002-01-01")
    qqq_full = daily_ret["QQQ"].loc["2002-01-01":"2026-03-31"].dropna()

    # ── TABLE 1 ──
    print(f"\n{'=' * 140}")
    print("  TABLE 1: CORE METRICS (2002-2026)")
    print(f"{'=' * 140}")
    print(f"\n  {'Strategy':<24} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9} {'CB':>4}")
    print(f"  {'-' * 24} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 4}")
    print(metrics_row("V19c 100% at sc2      ", v19c_full, v19c_cb))
    print(metrics_row("V19 70/30 at sc2      ", v19_full, 27))
    print(metrics_row("V9 QLD+IVVguard       ", v9_full, v9_cb))
    print(metrics_row("Baseline (Sweep-40)   ", bl_full, bl_cb))
    print(metrics_row("QQQ B&H               ", qqq_full))

    print(f"\n  Score 2/3 months: QQQ {s2q}, IVV {s2i}")

    # Score 2 return analysis
    v19c_monthly = v19c_full.resample("MS").apply(lambda x: (1+x).prod()-1)
    v19_monthly = v19_full.resample("MS").apply(lambda x: (1+x).prod()-1)

    # ── TABLE 2: Crisis drawdowns ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: CRISIS DRAWDOWNS")
    print(f"{'=' * 140}")
    crises = [("Dot-com","2002-01-01","2003-03-31"),("GFC","2007-11-01","2009-03-31"),
              ("COVID","2020-02-01","2020-04-30"),("2022","2022-01-01","2022-12-31")]
    names = ["V19c", "V19", "V9", "BL"]
    all_s = [v19c_full, v19_full, v9_full, bl_full]
    print(f"\n  {'Crisis':<18}" + "".join(f"{nm:>10}" for nm in names))
    print(f"  {'-' * 18}" + "".join(f" {'-' * 9}" for _ in names))
    for label, cs, ce in crises:
        cells = [max_dd(s[(s.index >= cs) & (s.index <= ce)]) if len(s[(s.index >= cs) & (s.index <= ce)]) > 5 else 0 for s in all_s]
        print(f"  {label:<18}" + "".join(f"{c:>10.1%}" for c in cells))

    # ── TABLE 3: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<24}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 24}" + "".join(f" {'-' * 9}" for _ in start_dates))
    for nm, fn in [("V19c 100% at sc2", "v19c"), ("V19 70/30 at sc2", "v19"), ("V9", "v9")]:
        row = f"  {nm:<24}"
        for sd in start_dates:
            if fn == "v19c":
                s, _, _, _ = run_v19c(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v19":
                s, _, _, _ = run_v19(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            else:
                s, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            row += f"{cagr(s):>10.2%}"
        print(row)

    # ── TABLE 4: DCA ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: DCA TERMINAL BY YEAR-END (2013 start)")
    print(f"{'=' * 140}")
    show = [("V19c", v19c_full), ("V19", v19_full), ("V9", v9_full)]
    print(f"\n  {'Year':<6}" + "".join(f"{nm:>13}" for nm, _ in show) + f"{'QQQ':>13}")
    for yr in [2015, 2018, 2020, 2022, 2025, 2026]:
        end = f"{yr}-12-31"
        row = f"  {yr:<6}"
        for nm, s in show:
            sp = s[(s.index >= "2013-01-01") & (s.index <= end)]
            sm = sp.resample("MS").apply(lambda x: (1+x).prod()-1)
            row += f"${dca_terminal(sm)/1e3:>11.0f}K"
        qs = qqq_full[(qqq_full.index >= "2013-01-01") & (qqq_full.index <= end)]
        qm = qs.resample("MS").apply(lambda x: (1+x).prod()-1)
        row += f"${dca_terminal(qm)/1e3:>11.0f}K"
        print(row)

    # ── Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: PASS / FAIL")
    print(f"{'=' * 140}")
    v19c_c = cagr(v19c_full); v19c_sh = sharpe_r(v19c_full); v19c_dd = max_dd(v19c_full)
    v19_c = cagr(v19_full); v19_sh = sharpe_r(v19_full); v19_dd = max_dd(v19_full)

    print(f"\n  V19c: CAGR {v19c_c:.2%}, Sharpe {v19c_sh:.3f}, MaxDD {v19c_dd:.1%}")
    print(f"  V19:  CAGR {v19_c:.2%}, Sharpe {v19_sh:.3f}, MaxDD {v19_dd:.1%}")

    sh_ok = v19c_sh >= v19_sh; dd_ok = v19c_dd >= v19_dd; cagr_ok = v19c_c >= v19_c
    v19c_dominates = sh_ok and dd_ok and cagr_ok
    v19_dominates = (v19_sh >= v19c_sh) and (v19_dd >= v19c_dd) and (v19_c >= v19c_c)
    wash = abs(v19c_sh - v19_sh) < 0.005 and abs(v19c_dd - v19_dd) < 0.01

    print(f"\n  Sharpe: {v19c_sh:.3f} vs {v19_sh:.3f} → {'✓' if sh_ok else '✗'}")
    print(f"  MaxDD:  {v19c_dd:.1%} vs {v19_dd:.1%} → {'✓' if dd_ok else '✗'}")
    print(f"  CAGR:   {v19c_c:.2%} vs {v19_c:.2%} → {'✓' if cagr_ok else '✗'}")

    if v19c_dominates:
        print(f"\n  → V19c DOMINATES V19 — adopt 100% unlevered at score 2/3")
    elif v19_dominates:
        print(f"\n  → V19 dominates V19c — keep 70/30")
    elif wash:
        print(f"\n  → WASH — take V19c (simpler: 100% vs 70/30)")
    else:
        print(f"\n  → TRADEOFF — V19c wins on {'CAGR' if cagr_ok else ''} {'Sharpe' if sh_ok else ''} {'DD' if dd_ok else ''}, "
              f"V19 wins on {'CAGR' if not cagr_ok else ''} {'Sharpe' if not sh_ok else ''} {'DD' if not dd_ok else ''}")

    print()


if __name__ == "__main__":
    main()
