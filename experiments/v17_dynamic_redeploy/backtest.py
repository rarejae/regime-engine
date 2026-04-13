"""V17: Two-Pod + Gold with Dynamic Redeployment.

When gold's Faber gate is off, redeploy 10% to active equity pods
instead of holding T-bills. When gold is on, identical to V16-B.
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
from experiments.v15_two_pod.backtest import run_v15
from experiments.v16_two_pod_gold.backtest import run_v16


def run_v17(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            start_date, capture_diag=False):
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}
    cb1 = 0; cb2 = 0; rebal_events = 0

    # Component NAVs
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10

    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_on = False
    scores = {}

    # Effective weights after redeployment
    w1 = 0.45; w2 = 0.45; wg = 0.10

    diag = {"monthly": [], "redeploy_months": []}

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
            gold_on = scores["IAU"] >= 3

            # Determine effective weights
            p1_active = p1_mode != "cash"
            p2_active = p2_mode != "cash"

            if gold_on:
                w1 = 0.45; w2 = 0.45; wg = 0.10
            else:
                # Redeploy gold's 10%
                if p1_active and p2_active:
                    w1 = 0.50; w2 = 0.50; wg = 0.0
                elif p1_active and not p2_active:
                    w1 = 0.55; w2 = 0.45; wg = 0.0  # pod2 in cash but nav tracks rfr
                elif p2_active and not p1_active:
                    w1 = 0.45; w2 = 0.55; wg = 0.0  # pod1 in cash but nav tracks rfr
                else:
                    w1 = 0.45; w2 = 0.45; wg = 0.10  # all cash — keep weights, all earn rfr

            # Rebalance NAVs to target weights
            # w1/w2/wg represent the target split of active capital
            # When a component is in cash, its nav still earns rfr
            if day != trading_days[0]:
                total = nav1 + nav2 + nav_g
                if total > 0:
                    cur_w1 = nav1 / total; cur_w2 = nav2 / total; cur_wg = nav_g / total
                    drift = max(abs(cur_w1 - w1), abs(cur_w2 - w2), abs(cur_wg - wg))
                    if drift > 0.05:
                        nav1 = total * w1; nav2 = total * w2; nav_g = total * wg
                        rebal_events += 1

            if capture_diag:
                redeploy = not gold_on and (p1_active or p2_active)
                diag["monthly"].append({
                    "month": day, "qqq_sc": sc_q, "ivv_sc": sc_i, "iau_sc": scores["IAU"],
                    "p1_mode": p1_mode, "p2_mode": p2_mode, "gold_on": gold_on,
                    "w1": w1, "w2": w2, "wg": wg, "redeploy": redeploy,
                })
                if redeploy:
                    diag["redeploy_months"].append(day)

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
        rg = iau_u if gold_on else rfr

        # Update NAVs
        prev_total = nav1 + nav2 + nav_g
        nav1 *= (1 + r1); nav2 *= (1 + r2); nav_g *= (1 + rg)
        new_total = nav1 + nav2 + nav_g
        port[day] = new_total / prev_total - 1

    return pd.Series(port).sort_index(), cb1, cb2, rebal_events, diag


def main():
    print("=" * 140)
    print("  V17 TWO-POD + GOLD WITH DYNAMIC REDEPLOYMENT — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    print("  Running V17...")
    v17_full, v17c1, v17c2, v17rb, v17_diag = run_v17(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
        "2002-01-01", capture_diag=True)

    print("  Running V16-B...")
    v16_full, v16c1, v16c2, v16rb, _ = run_v16(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
        "2002-01-01", iau_threshold=3)

    print("  Running V15...")
    v15_full, _, _, _, _, _, _ = run_v15(
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
    print(metrics_row("V17 Dyn Redeploy      ", v17_full, v17c1 + v17c2))
    print(metrics_row("V16-B (45/45/10)      ", v16_full, v16c1 + v16c2))
    print(metrics_row("V15 Two-Pod           ", v15_full, 27))
    print(metrics_row("V9 QLD+IVVguard       ", v9_full, v9_cb))
    print(metrics_row("Baseline (Sweep-40)   ", bl_full, bl_cb))
    print(metrics_row("QQQ B&H               ", qqq_full))

    print(f"\n  V17 CB: Pod1={v17c1}, Pod2={v17c2}. Rebalances={v17rb}")

    # ── TABLE 2: Redeployment diagnostics ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: REDEPLOYMENT DIAGNOSTICS")
    print(f"{'=' * 140}")
    months_data = v17_diag["monthly"]
    total_m = len(months_data)
    redeploy_months = [m for m in months_data if m["redeploy"]]
    non_redeploy = [m for m in months_data if not m["redeploy"]]

    print(f"\n  Total months: {total_m}")
    print(f"  Redeployment active (gold off + equity on): {len(redeploy_months)} ({len(redeploy_months)/total_m:.0%})")
    print(f"  Gold on (no redeployment): {sum(1 for m in months_data if m['gold_on'])}")
    print(f"  All cash (no redeployment): {sum(1 for m in months_data if not m['redeploy'] and not m['gold_on'])}")

    # Mean return in redeployment months: V17 vs V16
    v17_monthly = v17_full.resample("MS").apply(lambda x: (1+x).prod()-1)
    v16_monthly = v16_full.resample("MS").apply(lambda x: (1+x).prod()-1)
    v15_monthly = v15_full.resample("MS").apply(lambda x: (1+x).prod()-1)

    rd_v17 = []; rd_v16 = []
    for m in redeploy_months:
        ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
        r17 = v17_monthly.get(ts, np.nan); r16 = v16_monthly.get(ts, np.nan)
        if not pd.isna(r17): rd_v17.append(r17)
        if not pd.isna(r16): rd_v16.append(r16)

    if rd_v17 and rd_v16:
        print(f"\n  During redeployment months:")
        print(f"    V17 mean: {np.mean(rd_v17):+.2%}")
        print(f"    V16 mean: {np.mean(rd_v16):+.2%}")
        print(f"    Delta:    {np.mean(rd_v17)-np.mean(rd_v16):+.2%}")
        pos_v17 = sum(1 for r in rd_v17 if r > 0)
        print(f"    V17 positive: {pos_v17}/{len(rd_v17)} ({pos_v17/len(rd_v17):.0%})")

    # ── TABLE 3: State occupancy ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: STATE OCCUPANCY")
    print(f"{'=' * 140}")
    state_counts = {}
    for m in months_data:
        p1a = m["p1_mode"] != "cash"
        p2a = m["p2_mode"] != "cash"
        go = m["gold_on"]
        if p1a and p2a and go: lab = "Both eq + gold"
        elif p1a and p2a and not go: lab = "Both eq, gold→eq"
        elif p1a and not p2a and go: lab = "Pod1 + gold"
        elif p1a and not p2a and not go: lab = "Pod1 only, gold→Pod1"
        elif not p1a and p2a and go: lab = "Pod2 + gold"
        elif not p1a and p2a and not go: lab = "Pod2 only, gold→Pod2"
        elif not p1a and not p2a and go: lab = "Gold only"
        else: lab = "All cash"
        state_counts[lab] = state_counts.get(lab, 0) + 1
        # stash for reporting
        m["_state"] = lab

    print(f"\n  {'State':<28}{'Months':>8}{'Pct':>8}{'Weights':>18}")
    print(f"  {'-'*28}{'-'*7:>8}{'-'*7:>8}{'-'*17:>18}")
    weight_map = {
        "Both eq + gold": "45/45/10",
        "Both eq, gold→eq": "50/50/0",
        "Pod1 + gold": "45/0/10",
        "Pod1 only, gold→Pod1": "55/0/0",
        "Pod2 + gold": "0/45/10",
        "Pod2 only, gold→Pod2": "0/55/0",
        "Gold only": "0/0/10",
        "All cash": "0/0/0",
    }
    for lab in ["Both eq + gold", "Both eq, gold→eq", "Pod1 + gold", "Pod1 only, gold→Pod1",
                "Pod2 + gold", "Pod2 only, gold→Pod2", "Gold only", "All cash"]:
        n = state_counts.get(lab, 0)
        if n > 0:
            print(f"  {lab:<28}{n:>8}{n/total_m:>8.1%}{weight_map.get(lab,''):>18}")

    # ── TABLE 4: 2022 detail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: 2022 MONTH-BY-MONTH")
    print(f"{'=' * 140}")
    v9_monthly = v9_full.resample("MS").apply(lambda x: (1+x).prod()-1)
    print(f"\n  {'Month':<10}{'QQQ':>5}{'IVV':>5}{'IAU':>5}{'Wts':>12}{'V17':>9}{'V16':>9}{'V9':>9}")
    print(f"  {'-'*10}{'-'*4:>5}{'-'*4:>5}{'-'*4:>5}{'-'*11:>12}{'-'*8:>9}{'-'*8:>9}{'-'*8:>9}")
    for m in months_data:
        if m["month"].year == 2022:
            ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
            r17 = float(v17_monthly.get(ts, np.nan))
            r16 = float(v16_monthly.get(ts, np.nan))
            r9 = float(v9_monthly.get(ts, np.nan))
            wts = f"{int(m['w1']*100)}/{int(m['w2']*100)}/{int(m['wg']*100)}"
            print(f"  {m['month'].strftime('%Y-%m'):<10}{m['qqq_sc']:>5}{m['ivv_sc']:>5}{m['iau_sc']:>5}"
                  f"{wts:>12}{r17:>9.2%}{r16:>9.2%}{r9:>9.2%}")

    # ── TABLE 5: Crisis drawdowns ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: CRISIS DRAWDOWNS")
    print(f"{'=' * 140}")
    crises = [
        ("Dot-com 02-03", "2002-01-01", "2003-03-31"),
        ("GFC 07-09",     "2007-11-01", "2009-03-31"),
        ("COVID 2020",    "2020-02-01", "2020-04-30"),
        ("2022 bear",     "2022-01-01", "2022-12-31"),
    ]
    names = ["V17", "V16-B", "V15", "V9", "BL"]
    series_list = [v17_full, v16_full, v15_full, v9_full, bl_full]
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
    print(f"\n  {'Strategy':<24}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 24}" + "".join(f" {'-' * 9}" for _ in start_dates))
    for nm, fn in [("V17 Dyn Redeploy", "v17"), ("V16-B", "v16"), ("V15", "v15"),
                    ("V9", "v9"), ("Baseline", "bl")]:
        row = f"  {nm:<24}"
        for sd in start_dates:
            if fn == "v17":
                s, _, _, _, _ = run_v17(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v16":
                s, _, _, _, _ = run_v16(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd, 3)
            elif fn == "v15":
                s, _, _, _, _, _, _ = run_v15(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v9":
                s, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            else:
                s, _ = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                    dbmf_ret, dbmf_inception, sd)
            row += f"{cagr(s):>10.2%}"
        print(row)
    row = f"  {'QQQ B&H':<24}"
    for sd in start_dates:
        qs = qqq_full[qqq_full.index >= pd.Timestamp(sd)]
        row += f"{cagr(qs):>10.2%}"
    print(row)

    # ── TABLE 7: DCA ──
    print(f"\n{'=' * 140}")
    print("  TABLE 7: DCA TERMINAL BY YEAR-END (2013 start)")
    print(f"{'=' * 140}")
    show = [("V17", v17_full), ("V16-B", v16_full), ("V15", v15_full), ("V9", v9_full)]
    print(f"\n  {'Year':<6}" + "".join(f"{nm:>13}" for nm, _ in show) + f"{'QQQ':>13}{'V17-QQQ':>13}")
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
        row += f"${vals['QQQ']/1e3:>11.0f}K ${(vals['V17']-vals['QQQ'])/1e3:>11.0f}K"
        print(row)

    # ── TABLE 8: Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 8: PASS / FAIL")
    print(f"{'=' * 140}")
    v17_c = cagr(v17_full); v17_sh = sharpe_r(v17_full); v17_dd = max_dd(v17_full)
    v16_c = cagr(v16_full); v16_sh = sharpe_r(v16_full); v16_dd = max_dd(v16_full)

    cagr_ok = v17_c >= v16_c
    sh_ok = v17_sh >= v16_sh - 0.02
    dd_ok = v17_dd >= v16_dd - 0.03  # within 3pp
    passed = cagr_ok and sh_ok and dd_ok

    print(f"\n  V17: CAGR {v17_c:.2%}, Sharpe {v17_sh:.3f}, MaxDD {v17_dd:.1%}")
    print(f"  V16: CAGR {v16_c:.2%}, Sharpe {v16_sh:.3f}, MaxDD {v16_dd:.1%}")
    print(f"\n  vs V16-B:")
    print(f"    CAGR ≥ V16:     {v17_c:.2%} vs {v16_c:.2%} → {'✓' if cagr_ok else '✗'}")
    print(f"    Sharpe ≥ V16-0.02: {v17_sh:.3f} vs {v16_sh-0.02:.3f} → {'✓' if sh_ok else '✗'}")
    print(f"    MaxDD ≤ V16+3pp: {v17_dd:.1%} vs {v16_dd-0.03:.1%} → {'✓' if dd_ok else '✗'}")
    print(f"    → {'PASS' if passed else 'FAIL'}")

    # Is redeployment net positive?
    if rd_v17 and rd_v16:
        delta = np.mean(rd_v17) - np.mean(rd_v16)
        print(f"\n  Redeployment delta: {delta:+.2%}/mo ({'positive — redeployment adds value' if delta > 0 else 'negative — gold cash is load-bearing'})")

    print()


if __name__ == "__main__":
    main()
