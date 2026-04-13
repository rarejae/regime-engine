"""V15: Two-Pod Architecture — V9 QLD (50%) + IVV/SSO (50%).

Pod 1: V9 unchanged (QLD, IVV guard, CB).
Pod 2: IVV/SSO with V9 logic, no cross-asset guard.
Each pod gets 50%. Monthly rebalance to 50/50 if drift > 5%.

Tests both with and without pod rebalancing.
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


def run_v15(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            start_date, rebalance=True, capture_diag=False):
    """Two-pod: Pod1=V9 QLD, Pod2=IVV/SSO V9-logic."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; pod1_port = {}; pod2_port = {}
    cb1 = 0; cb2 = 0; rebal_events = 0

    # Pod NAVs (start at 1.0 each, total 2.0)
    nav1 = 1.0; nav2 = 1.0

    # Pod 1 state (V9)
    p1_mode = "cash"; p1_lev = False; p1_delev = False
    # Pod 2 state (IVV/SSO)
    p2_mode = "cash"; p2_lev = False; p2_delev = False

    scores = {"QQQ": 0, "IVV": 0}
    diag = {"monthly": [], "diverge": []}

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = {"QQQ": asset_score(sd, "QQQ", dpdf, daily_smas),
                      "IVV": asset_score(sd, "IVV", dpdf, daily_smas)}

            # Pod 1: V9 logic
            sc_q = scores["QQQ"]; sc_i = scores["IVV"]
            if sc_q >= 3:
                if sc_i <= 1:
                    p1_mode = "qqq"; p1_lev = False
                else:
                    p1_mode = "qld"; p1_lev = True
            elif sc_q == 2:
                p1_mode = "qqq_partial"; p1_lev = False
            else:
                p1_mode = "cash"; p1_lev = False

            # Pod 2: IVV/SSO, no guard
            if sc_i >= 3:
                p2_mode = "sso"; p2_lev = True
            elif sc_i == 2:
                p2_mode = "ivv_partial"; p2_lev = False
            else:
                p2_mode = "cash"; p2_lev = False

            # Pod rebalancing (if enabled)
            if rebalance and day != trading_days[0]:
                total = nav1 + nav2
                w1 = nav1 / total
                if w1 > 0.55 or w1 < 0.45:
                    nav1 = total * 0.50
                    nav2 = total * 0.50
                    rebal_events += 1

            if capture_diag:
                total = nav1 + nav2
                diag["monthly"].append({
                    "month": day, "qqq_sc": sc_q, "ivv_sc": sc_i,
                    "p1_mode": p1_mode, "p2_mode": p2_mode,
                    "w1": nav1 / total, "w2": nav2 / total,
                })

        # Pod 1 CB
        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; cb1 += 1
                p1_mode = "qqq"

        # Pod 2 CB
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; cb2 += 1
                p2_mode = "ivv"

        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0

        # Pod 1 return
        if p1_mode == "qld":
            r1 = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start) if p1_lev else qqq_u
        elif p1_mode == "qqq":
            r1 = qqq_u
        elif p1_mode == "qqq_partial":
            r1 = 0.70 * qqq_u + 0.30 * rfr
        else:
            r1 = rfr

        # Pod 2 return
        if p2_mode == "sso":
            r2 = lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start) if p2_lev else ivv_u
        elif p2_mode == "ivv":
            r2 = ivv_u
        elif p2_mode == "ivv_partial":
            r2 = 0.70 * ivv_u + 0.30 * rfr
        else:
            r2 = rfr

        # Update NAVs
        nav1 *= (1 + r1); nav2 *= (1 + r2)
        total = nav1 + nav2
        port[day] = (nav1 + nav2) / ((nav1 / (1 + r1)) + (nav2 / (1 + r2))) - 1  # weighted return
        pod1_port[day] = r1; pod2_port[day] = r2

    return (pd.Series(port).sort_index(), cb1, cb2, rebal_events,
            pd.Series(pod1_port).sort_index(), pd.Series(pod2_port).sort_index(), diag)


def main():
    print("=" * 140)
    print("  V15 TWO-POD: V9 QLD + IVV/SSO — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    print("  Running V15 (with rebalancing)...")
    v15_full, v15_cb1, v15_cb2, v15_rebal, pod1_s, pod2_s, v15_diag = run_v15(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
        "2002-01-01", rebalance=True, capture_diag=True)

    print("  Running V15 (no rebalancing)...")
    v15nr_full, _, _, _, _, _, _ = run_v15(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
        "2002-01-01", rebalance=False)

    print("  Running V9...")
    v9_full, v9_cb = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running V12...")
    v12_full, v12_cb, _ = run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running Baseline...")
    bl_full, bl_cb = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                   dbmf_ret, dbmf_inception, "2002-01-01")
    qqq_full = daily_ret["QQQ"].loc["2002-01-01":"2026-03-31"].dropna()
    ivv_full = daily_ret["IVV"].loc["2002-01-01":"2026-03-31"].dropna()

    # ── TABLE 1 ──
    print(f"\n{'=' * 140}")
    print("  TABLE 1: CORE METRICS (2002-2026)")
    print(f"{'=' * 140}")
    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9} {'CB':>4}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 4}")
    print(metrics_row("V15 Two-Pod (rebal)", v15_full, v15_cb1 + v15_cb2))
    print(metrics_row("V15 (no rebal)", v15nr_full, v15_cb1 + v15_cb2))
    print(metrics_row("V9 QLD+IVVguard", v9_full, v9_cb))
    print(metrics_row("V12 Independent 2×", v12_full, v12_cb))
    print(metrics_row("Baseline (Sweep-40)", bl_full, bl_cb))
    print(metrics_row("QQQ B&H", qqq_full))
    print(metrics_row("IVV B&H", ivv_full))

    print(f"\n  Pod CB events: Pod1(QQQ)={v15_cb1}, Pod2(IVV)={v15_cb2}")
    print(f"  Rebalance events (drift >5%): {v15_rebal}")

    # ── TABLE 2: Pod standalone ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: POD STANDALONE METRICS")
    print(f"{'=' * 140}")
    print(f"\n  {'Pod':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Term$1':>9}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 9}")
    for nm, s in [("Pod 1 (V9 QLD)", pod1_s), ("Pod 2 (IVV/SSO)", pod2_s)]:
        c = cagr(s); v = s.std() * np.sqrt(252); sh = sharpe_r(s); so = sortino_r(s)
        dd = max_dd(s); t = (1 + s).cumprod().iloc[-1]
        print(f"  {nm:<22} {c:>6.2%} {v:>6.2%} {sh:>7.3f} {so:>8.3f} {dd:>6.1%} ${t:>8.2f}")
    # V9 validation
    c9 = cagr(v9_full); s9 = sharpe_r(v9_full)
    cp1 = cagr(pod1_s); sp1 = sharpe_r(pod1_s)
    print(f"\n  Pod 1 vs V9 validation: CAGR {cp1:.2%} vs {c9:.2%}, Sharpe {sp1:.3f} vs {s9:.3f}")
    print(f"  → {'MATCH ✓' if abs(cp1 - c9) < 0.005 else 'MISMATCH ✗'}")

    # ── TABLE 3: Signal divergence ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: SIGNAL DIVERGENCE BETWEEN PODS")
    print(f"{'=' * 140}")

    v15_monthly = v15_full.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    v9_monthly = v9_full.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    v12_monthly = v12_full.resample("MS").apply(lambda x: (1 + x).prod() - 1)

    state_rows = []
    for m in v15_diag["monthly"]:
        ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
        p1_on = m["p1_mode"] in ("qld", "qqq", "qqq_partial")
        p2_on = m["p2_mode"] in ("sso", "ivv", "ivv_partial")
        p1_lev = m["p1_mode"] == "qld"
        p2_lev = m["p2_mode"] == "sso"

        if p1_lev and p2_lev: lab = "Both leveraged"
        elif p1_on and p2_on: lab = "Both on (mixed lev)"
        elif p1_on and not p2_on: lab = "Pod1 on, Pod2 cash"
        elif not p1_on and p2_on: lab = "Pod1 cash, Pod2 on"
        else: lab = "Both cash"

        v15r = float(v15_monthly.get(ts, np.nan))
        v9r = float(v9_monthly.get(ts, np.nan))
        v12r = float(v12_monthly.get(ts, np.nan))
        state_rows.append({"month": ts, "state": lab, "v15": v15r, "v9": v9r, "v12": v12r})

    sdf = pd.DataFrame(state_rows)
    total_m = len(sdf)
    print(f"\n  {'State':<25}{'Count':>8}{'Pct':>8}{'V15 mean':>12}{'V9 mean':>12}{'V12 mean':>12}")
    print(f"  {'-' * 25}{'-' * 7:>8}{'-' * 7:>8}{'-' * 11:>12}{'-' * 11:>12}{'-' * 11:>12}")
    for lab in ["Both leveraged", "Both on (mixed lev)", "Pod1 on, Pod2 cash",
                "Pod1 cash, Pod2 on", "Both cash"]:
        sub = sdf[sdf["state"] == lab]
        if len(sub) > 0:
            n = len(sub)
            print(f"  {lab:<25}{n:>8}{n / total_m:>8.1%}"
                  f"{sub['v15'].mean():>12.2%}{sub['v9'].mean():>12.2%}{sub['v12'].mean():>12.2%}")

    # ── TABLE 4: Rebalancing impact ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: REBALANCING IMPACT")
    print(f"{'=' * 140}")
    c_r = cagr(v15_full); sh_r = sharpe_r(v15_full); dd_r = max_dd(v15_full); t_r = (1+v15_full).cumprod().iloc[-1]
    c_n = cagr(v15nr_full); sh_n = sharpe_r(v15nr_full); dd_n = max_dd(v15nr_full); t_n = (1+v15nr_full).cumprod().iloc[-1]
    print(f"\n  {'Variant':<22}{'CAGR':>10}{'Sharpe':>10}{'MaxDD':>10}{'Term$1':>12}")
    print(f"  {'-'*22}{'-'*9:>10}{'-'*9:>10}{'-'*9:>10}{'-'*11:>12}")
    print(f"  {'With rebal':<22}{c_r:>10.2%}{sh_r:>10.3f}{dd_r:>10.1%}${t_r:>11.2f}")
    print(f"  {'No rebal':<22}{c_n:>10.2%}{sh_n:>10.3f}{dd_n:>10.1%}${t_n:>11.2f}")
    print(f"  {'Delta (rebal-norebal)':<22}{c_r-c_n:>+10.2%}{sh_r-sh_n:>+10.3f}"
          f"{dd_r-dd_n:>+10.1%}${t_r-t_n:>+11.2f}")
    print(f"  Rebalance events: {v15_rebal} over 24 years ({v15_rebal/24:.1f}/yr)")

    # ── TABLE 5: 2022 detail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: 2022 MONTH-BY-MONTH")
    print(f"{'=' * 140}")
    bl_monthly = bl_full.resample("MS").apply(lambda x: (1+x).prod()-1)
    print(f"\n  {'Month':<10}{'QQQ':>5}{'IVV':>5}{'Pod1':>8}{'Pod2':>8}{'V15 ret':>10}{'V9 ret':>10}{'V12 ret':>10}")
    print(f"  {'-'*10}{'-'*4:>5}{'-'*4:>5}{'-'*7:>8}{'-'*7:>8}{'-'*9:>10}{'-'*9:>10}{'-'*9:>10}")
    for m in v15_diag["monthly"]:
        if m["month"].year == 2022:
            ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
            r15 = float(v15_monthly.get(ts, np.nan))
            r9 = float(v9_monthly.get(ts, np.nan))
            r12 = float(v12_monthly.get(ts, np.nan))
            print(f"  {m['month'].strftime('%Y-%m'):<10}{m['qqq_sc']:>5}{m['ivv_sc']:>5}"
                  f"{m['p1_mode']:>8}{m['p2_mode']:>8}{r15:>10.2%}{r9:>10.2%}{r12:>10.2%}")

    # ── TABLE 6: Crisis drawdowns ──
    print(f"\n{'=' * 140}")
    print("  TABLE 6: CRISIS DRAWDOWNS")
    print(f"{'=' * 140}")
    crises = [
        ("Dot-com 02-03", "2002-01-01", "2003-03-31"),
        ("GFC 07-09",     "2007-11-01", "2009-03-31"),
        ("COVID 2020",    "2020-02-01", "2020-04-30"),
        ("2022 bear",     "2022-01-01", "2022-12-31"),
    ]
    names = ["V15", "V9", "V12", "BL", "QQQ"]
    series_list = [v15_full, v9_full, v12_full, bl_full, qqq_full]
    print(f"\n  {'Crisis':<18}" + "".join(f"{nm:>10}" for nm in names))
    print(f"  {'-' * 18}" + "".join(f" {'-' * 9}" for _ in names))
    for label, cs, ce in crises:
        cells = []
        for s in series_list:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            cells.append(max_dd(sp) if len(sp) > 5 else 0)
        print(f"  {label:<18}" + "".join(f"{c:>10.1%}" for c in cells))

    # ── TABLE 7: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 7: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<22}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 22}" + "".join(f" {'-' * 9}" for _ in start_dates))
    for nm, fn in [("V15 Two-Pod", "v15"), ("V9", "v9"), ("V12", "v12"), ("Baseline", "bl")]:
        row = f"  {nm:<22}"
        for sd in start_dates:
            if fn == "v15":
                s, _, _, _, _, _, _ = run_v15(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v9":
                s, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v12":
                s, _, _ = run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
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

    # ── TABLE 8: DCA dollar gap ──
    print(f"\n{'=' * 140}")
    print("  TABLE 8: DCA TERMINAL BY YEAR-END (2013 start)")
    print(f"{'=' * 140}")
    show_names = ["V15", "V9", "V12", "BL"]
    show_series = [v15_full, v9_full, v12_full, bl_full]
    print(f"\n  {'Year':<6}" + "".join(f"{nm:>14}" for nm in show_names) + f"{'QQQ':>13}{'V15-QQQ':>13}")
    for yr in range(2013, 2027):
        end = f"{yr}-12-31"
        vals = {}
        for nm, s in zip(show_names, show_series):
            sp = s[(s.index >= "2013-01-01") & (s.index <= end)]
            sm = sp.resample("MS").apply(lambda x: (1 + x).prod() - 1)
            vals[nm] = dca_terminal(sm)
        qs = qqq_full[(qqq_full.index >= "2013-01-01") & (qqq_full.index <= end)]
        qm = qs.resample("MS").apply(lambda x: (1 + x).prod() - 1)
        vals["QQQ"] = dca_terminal(qm)
        row = f"  {yr:<6}"
        for nm in show_names:
            row += f"${vals[nm]/1e3:>12.0f}K"
        row += f"${vals['QQQ']/1e3:>11.0f}K ${(vals['V15']-vals['QQQ'])/1e3:>11.0f}K"
        print(row)

    # ── TABLE 9: Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 9: PASS / FAIL")
    print(f"{'=' * 140}")
    v15_c = cagr(v15_full); v15_sh = sharpe_r(v15_full); v15_dd = max_dd(v15_full)
    v9_c = cagr(v9_full); v9_sh = sharpe_r(v9_full); v9_dd = max_dd(v9_full)
    v12_c = cagr(v12_full); v12_sh = sharpe_r(v12_full); v12_dd = max_dd(v12_full)

    print(f"\n  vs V9:")
    print(f"    Sharpe:   {v15_sh:.3f} vs {v9_sh:.3f} → {'✓' if v15_sh > v9_sh else '✗'}")
    print(f"    MaxDD:    {v15_dd:.1%} vs {v9_dd:.1%} → {'✓' if v15_dd > v9_dd else '✗'}")
    print(f"    CAGR:     {v15_c:.2%} vs {v9_c:.2%} (within 2pp) → {'✓' if v15_c >= v9_c - 0.02 else '✗'}")
    pass_v9 = (v15_sh > v9_sh) and (v15_dd > v9_dd) and (v15_c >= v9_c - 0.02)
    print(f"    → {'PASS' if pass_v9 else 'FAIL'}")

    print(f"\n  vs V12:")
    print(f"    Sharpe:   {v15_sh:.3f} vs {v12_sh:.3f} (within 0.01) → {'✓' if abs(v15_sh - v12_sh) < 0.01 else '✗'}")
    print(f"    MaxDD:    {v15_dd:.1%} vs {v12_dd:.1%} (within 2pp) → {'✓' if abs(v15_dd - v12_dd) < 0.02 else '✗'}")
    print(f"    CAGR:     {v15_c:.2%} vs {v12_c:.2%} (within 1pp) → {'✓' if abs(v15_c - v12_c) < 0.01 else '✗'}")
    v15_approx_v12 = (abs(v15_sh - v12_sh) < 0.01) and (abs(v15_dd - v12_dd) < 0.02)
    print(f"    → {'V15 ≈ V12 (equivalent)' if v15_approx_v12 else 'V15 ≠ V12 (different)'}")

    bonus = v15_sh > 0.85 and v15_c > 0.17
    if bonus:
        print(f"\n  BONUS: Sharpe {v15_sh:.3f} > 0.85 AND CAGR {v15_c:.2%} > 17% → NEW FRONTIER POINT")

    print()


if __name__ == "__main__":
    main()
