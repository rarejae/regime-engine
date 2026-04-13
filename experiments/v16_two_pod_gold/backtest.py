"""V16: Two-Pod + Gold — 45/45/10.

Pod 1 (45%): V9 QLD unchanged.
Pod 2 (45%): IVV/SSO V9-logic, no guard.
Gold (10%): IAU Faber-gated, 1× only.
Monthly rebalance to 45/45/10 if any component drifts >5%.

Variants: IAU threshold ≥ 2 (A) and ≥ 3 (B).
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
from experiments.v12_independent_2x.backtest import run_v12
from experiments.v15_two_pod.backtest import run_v15


def run_v16(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            start_date, iau_threshold=2, capture_diag=False):
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}
    cb1 = 0; cb2 = 0; rebal_events = 0

    # Component NAVs: target 45/45/10
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10

    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_mode = "cash"
    scores = {}
    diag = {"monthly": []}

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = {a: asset_score(sd, a, dpdf, daily_smas) for a in ["QQQ", "IVV", "IAU"]}

            # Pod 1: V9
            sc_q = scores["QQQ"]; sc_i = scores["IVV"]
            if sc_q >= 3:
                if sc_i <= 1: p1_mode = "qqq"; p1_lev = False
                else: p1_mode = "qld"; p1_lev = True
            elif sc_q == 2: p1_mode = "qqq_partial"; p1_lev = False
            else: p1_mode = "cash"; p1_lev = False

            # Pod 2: IVV/SSO
            if sc_i >= 3: p2_mode = "sso"; p2_lev = True
            elif sc_i == 2: p2_mode = "ivv_partial"; p2_lev = False
            else: p2_mode = "cash"; p2_lev = False

            # Gold
            gold_mode = "iau" if scores["IAU"] >= iau_threshold else "cash"

            # Rebalance to 45/45/10
            if day != trading_days[0]:
                total = nav1 + nav2 + nav_g
                w1 = nav1 / total; w2 = nav2 / total; wg = nav_g / total
                if abs(w1 - 0.45) > 0.05 or abs(w2 - 0.45) > 0.05 or abs(wg - 0.10) > 0.05:
                    nav1 = total * 0.45; nav2 = total * 0.45; nav_g = total * 0.10
                    rebal_events += 1

            if capture_diag:
                total = nav1 + nav2 + nav_g
                diag["monthly"].append({
                    "month": day, "qqq_sc": sc_q, "ivv_sc": sc_i, "iau_sc": scores["IAU"],
                    "p1_mode": p1_mode, "p2_mode": p2_mode, "gold_mode": gold_mode,
                })

        # CB Pod 1
        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; cb1 += 1; p1_mode = "qqq"
        # CB Pod 2
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; cb2 += 1; p2_mode = "ivv"

        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
        iau_u = float(dr.get("IAU", 0.0)) if pd.notna(dr.get("IAU", np.nan)) else 0.0

        # Pod 1 return
        if p1_mode == "qld":
            r1 = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start) if p1_lev else qqq_u
        elif p1_mode == "qqq": r1 = qqq_u
        elif p1_mode == "qqq_partial": r1 = 0.70 * qqq_u + 0.30 * rfr
        else: r1 = rfr

        # Pod 2 return
        if p2_mode == "sso":
            r2 = lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start) if p2_lev else ivv_u
        elif p2_mode == "ivv": r2 = ivv_u
        elif p2_mode == "ivv_partial": r2 = 0.70 * ivv_u + 0.30 * rfr
        else: r2 = rfr

        # Gold return
        rg = iau_u if gold_mode == "iau" else rfr

        # Update NAVs
        prev_total = nav1 + nav2 + nav_g
        nav1 *= (1 + r1); nav2 *= (1 + r2); nav_g *= (1 + rg)
        new_total = nav1 + nav2 + nav_g
        port[day] = new_total / prev_total - 1

    return pd.Series(port).sort_index(), cb1, cb2, rebal_events, diag


def main():
    print("=" * 140)
    print("  V16 TWO-POD + GOLD (45/45/10) — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    variants = {}
    for name, thresh in [("V16-A (IAU≥2)", 2), ("V16-B (IAU≥3)", 3)]:
        print(f"  Running {name}...")
        s, c1, c2, rb, diag = run_v16(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                       "2002-01-01", iau_threshold=thresh, capture_diag=True)
        variants[name] = (s, c1, c2, rb, diag)

    print("  Running V15...")
    v15_full, v15c1, v15c2, v15rb, _, _, _ = run_v15(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01", rebalance=True)
    print("  Running V9...")
    v9_full, v9_cb = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running V12...")
    v12_full, v12_cb, _ = run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running Baseline...")
    bl_full, bl_cb = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                   dbmf_ret, dbmf_inception, "2002-01-01")
    qqq_full = daily_ret["QQQ"].loc["2002-01-01":"2026-03-31"].dropna()

    # ── TABLE 1 ──
    print(f"\n{'=' * 140}")
    print("  TABLE 1: CORE METRICS (2002-2026)")
    print(f"{'=' * 140}")
    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9} {'CB':>4}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 4}")
    for nm in ["V16-A (IAU≥2)", "V16-B (IAU≥3)"]:
        s, c1, c2, rb, _ = variants[nm]
        print(metrics_row(nm, s, c1 + c2))
    print(metrics_row("V15 Two-Pod", v15_full, v15c1 + v15c2))
    print(metrics_row("V9 QLD+IVVguard", v9_full, v9_cb))
    print(metrics_row("V12 Independent 2×", v12_full, v12_cb))
    print(metrics_row("Baseline (Sweep-40)", bl_full, bl_cb))
    print(metrics_row("QQQ B&H", qqq_full))

    for nm in ["V16-A (IAU≥2)", "V16-B (IAU≥3)"]:
        _, c1, c2, rb, _ = variants[nm]
        print(f"  {nm}: Pod1 CB={c1}, Pod2 CB={c2}, Rebalances={rb}")

    # ── TABLE 2: Gold utilization ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: GOLD UTILIZATION")
    print(f"{'=' * 140}")
    for nm in ["V16-A (IAU≥2)", "V16-B (IAU≥3)"]:
        diag = variants[nm][4]
        months = diag["monthly"]
        total_m = len(months)
        iau_active = sum(1 for m in months if m["gold_mode"] == "iau")
        eq_off = [m for m in months if m["p1_mode"] == "cash" and m["p2_mode"] == "cash"]
        iau_active_off = sum(1 for m in eq_off if m["gold_mode"] == "iau")
        print(f"\n  {nm}:")
        print(f"    IAU active: {iau_active}/{total_m} ({iau_active/total_m:.0%})")
        print(f"    IAU in cash: {total_m - iau_active}/{total_m} ({(total_m-iau_active)/total_m:.0%})")
        print(f"    Equity-off months: {len(eq_off)}")
        print(f"    IAU active during equity-off: {iau_active_off}/{len(eq_off)} ({iau_active_off/len(eq_off):.0%})" if eq_off else "")

    # ── TABLE 3: Gold during crises ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: GOLD DURING CRISES (V16-A)")
    print(f"{'=' * 140}")
    diag_a = variants["V16-A (IAU≥2)"][4]
    crises = [
        ("Dot-com 02-03", "2002-01-01", "2003-03-31"),
        ("GFC 07-09",     "2007-11-01", "2009-03-31"),
        ("COVID 2020",    "2020-02-01", "2020-04-30"),
        ("2022 bear",     "2022-01-01", "2022-12-31"),
    ]
    iau_daily = daily_ret["IAU"].loc["2002-01-01":"2026-03-31"].dropna() if "IAU" in daily_ret.columns else pd.Series()
    for label, cs, ce in crises:
        crisis_months = [m for m in diag_a["monthly"]
                         if m["month"] >= pd.Timestamp(cs) and m["month"] <= pd.Timestamp(ce)]
        iau_on = sum(1 for m in crisis_months if m["gold_mode"] == "iau")
        if len(iau_daily) > 0:
            iau_sp = iau_daily[(iau_daily.index >= pd.Timestamp(cs)) & (iau_daily.index <= pd.Timestamp(ce))]
            iau_cum = (1 + iau_sp).prod() - 1 if len(iau_sp) > 5 else 0
        else:
            iau_cum = 0
        print(f"  {label}: IAU active {iau_on}/{len(crisis_months)} months, IAU cumul return: {iau_cum:+.1%}")

    # ── TABLE 4: 2022 detail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: 2022 MONTH-BY-MONTH (V16-A)")
    print(f"{'=' * 140}")
    v16a_monthly = variants["V16-A (IAU≥2)"][0].resample("MS").apply(lambda x: (1+x).prod()-1)
    v15_monthly = v15_full.resample("MS").apply(lambda x: (1+x).prod()-1)
    v9_monthly = v9_full.resample("MS").apply(lambda x: (1+x).prod()-1)
    print(f"\n  {'Month':<10}{'QQQ':>5}{'IVV':>5}{'IAU':>5}{'Pod1':>8}{'Pod2':>8}{'Gold':>6}"
          f"{'V16A':>9}{'V15':>9}{'V9':>9}")
    print(f"  {'-'*10}{'-'*4:>5}{'-'*4:>5}{'-'*4:>5}{'-'*7:>8}{'-'*7:>8}{'-'*5:>6}"
          f"{'-'*8:>9}{'-'*8:>9}{'-'*8:>9}")
    for m in diag_a["monthly"]:
        if m["month"].year == 2022:
            ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
            r16 = float(v16a_monthly.get(ts, np.nan))
            r15 = float(v15_monthly.get(ts, np.nan))
            r9 = float(v9_monthly.get(ts, np.nan))
            print(f"  {m['month'].strftime('%Y-%m'):<10}{m['qqq_sc']:>5}{m['ivv_sc']:>5}{m['iau_sc']:>5}"
                  f"{m['p1_mode']:>8}{m['p2_mode']:>8}{m['gold_mode']:>6}"
                  f"{r16:>9.2%}{r15:>9.2%}{r9:>9.2%}")

    # ── TABLE 5: Crisis drawdowns ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: CRISIS DRAWDOWNS")
    print(f"{'=' * 140}")
    names = ["V16-A", "V16-B", "V15", "V9", "BL"]
    series_list = [variants["V16-A (IAU≥2)"][0], variants["V16-B (IAU≥3)"][0],
                   v15_full, v9_full, bl_full]
    print(f"\n  {'Crisis':<18}" + "".join(f"{nm:>10}" for nm in names))
    print(f"  {'-' * 18}" + "".join(f" {'-' * 9}" for _ in names))
    for label, cs, ce in crises:
        cells = []
        for s in series_list:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            cells.append(max_dd(sp) if len(sp) > 5 else 0)
        print(f"  {label:<18}" + "".join(f"{c:>10.1%}" for c in cells))

    # ── TABLE 6: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 6: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<22}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 22}" + "".join(f" {'-' * 9}" for _ in start_dates))
    for nm, thresh in [("V16-A (IAU≥2)", 2), ("V16-B (IAU≥3)", 3)]:
        row = f"  {nm:<22}"
        for sd in start_dates:
            s, _, _, _, _ = run_v16(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                     sd, iau_threshold=thresh)
            row += f"{cagr(s):>10.2%}"
        print(row)
    for nm, fn in [("V15 Two-Pod", "v15"), ("V9", "v9"), ("Baseline", "bl")]:
        row = f"  {nm:<22}"
        for sd in start_dates:
            if fn == "v15":
                s, _, _, _, _, _, _ = run_v15(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v9":
                s, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            else:
                s, _ = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                    dbmf_ret, dbmf_inception, sd)
            row += f"{cagr(s):>10.2%}"
        print(row)
    row = f"  {'QQQ B&H':<22}"
    for sd in start_dates:
        qs = qqq_full[qqq_full.index >= pd.Timestamp(sd)]
        row += f"{cagr(qs):>10.2%}"
    print(row)

    # ── TABLE 7: DCA dollar gap ──
    print(f"\n{'=' * 140}")
    print("  TABLE 7: DCA TERMINAL BY YEAR-END (2013 start)")
    print(f"{'=' * 140}")
    show = [("V16-A", variants["V16-A (IAU≥2)"][0]), ("V16-B", variants["V16-B (IAU≥3)"][0]),
            ("V15", v15_full), ("V9", v9_full)]
    print(f"\n  {'Year':<6}" + "".join(f"{nm:>13}" for nm, _ in show) + f"{'QQQ':>13}{'V16A-QQQ':>13}")
    for yr in range(2013, 2027):
        end = f"{yr}-12-31"
        vals = {}
        for nm, s in show:
            sp = s[(s.index >= "2013-01-01") & (s.index <= end)]
            sm = sp.resample("MS").apply(lambda x: (1+x).prod()-1)
            vals[nm] = dca_terminal(sm)
        qs = qqq_full[(qqq_full.index >= "2013-01-01") & (qqq_full.index <= end)]
        qm = qs.resample("MS").apply(lambda x: (1+x).prod()-1)
        vals["QQQ"] = dca_terminal(qm)
        row = f"  {yr:<6}"
        for nm, _ in show:
            row += f"${vals[nm]/1e3:>11.0f}K"
        row += f"${vals['QQQ']/1e3:>11.0f}K ${(vals['V16-A']-vals['QQQ'])/1e3:>11.0f}K"
        print(row)

    # ── TABLE 8: Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 8: PASS / FAIL")
    print(f"{'=' * 140}")
    v15_c = cagr(v15_full); v15_sh = sharpe_r(v15_full); v15_dd = max_dd(v15_full)

    for nm in ["V16-A (IAU≥2)", "V16-B (IAU≥3)"]:
        s = variants[nm][0]
        vc = cagr(s); vsh = sharpe_r(s); vdd = max_dd(s)
        sh_ok = vsh >= v15_sh
        dd_ok = vdd > v15_dd  # shallower = more positive
        cagr_ok = vc >= v15_c - 0.01
        passed = sh_ok and dd_ok and cagr_ok

        print(f"\n  {nm}:")
        print(f"    CAGR:   {vc:.2%} vs V15 {v15_c:.2%} (within 1pp) → {'✓' if cagr_ok else '✗'}")
        print(f"    Sharpe: {vsh:.3f} vs V15 {v15_sh:.3f} → {'✓' if sh_ok else '✗'}")
        print(f"    MaxDD:  {vdd:.1%} vs V15 {v15_dd:.1%} → {'✓' if dd_ok else '✗'}")
        print(f"    → {'PASS' if passed else 'FAIL'}")

    print()


if __name__ == "__main__":
    main()
