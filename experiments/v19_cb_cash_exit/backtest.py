"""V19: V16-B with CB → Full Cash Exit.

One change from V16-B: when per-asset CB fires, exit to CASH (not unlevered equity).
Re-entry at next monthly rebalance only. Everything else identical.
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
from experiments.v16_two_pod_gold.backtest import run_v16


def run_v19(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            start_date, capture_events=False):
    """V16-B with CB → cash exit."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb1 = 0; cb2 = 0
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10

    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_mode = "cash"
    scores = {}
    cb_events = []

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False
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

        # Per-asset CB → EXIT TO CASH (V19 change)
        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; cb1 += 1
                p1_mode = "cash"  # V19: cash, not "qqq"
                if capture_events:
                    cb_events.append({"date": day, "pod": 1, "prior": "qld"})
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; cb2 += 1
                p2_mode = "cash"  # V19: cash, not "ivv"
                if capture_events:
                    cb_events.append({"date": day, "pod": 2, "prior": "sso"})

        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
        iau_u = float(dr.get("IAU", 0.0)) if pd.notna(dr.get("IAU", np.nan)) else 0.0

        if p1_mode == "qld":
            r1 = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start) if p1_lev else qqq_u
        elif p1_mode == "qqq": r1 = qqq_u
        elif p1_mode == "qqq_partial": r1 = 0.70 * qqq_u + 0.30 * rfr
        else: r1 = rfr  # cash

        if p2_mode == "sso":
            r2 = lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start) if p2_lev else ivv_u
        elif p2_mode == "ivv": r2 = ivv_u
        elif p2_mode == "ivv_partial": r2 = 0.70 * ivv_u + 0.30 * rfr
        else: r2 = rfr  # cash

        rg = iau_u if gold_mode == "iau" else rfr

        prev_total = nav1 + nav2 + nav_g
        nav1 *= (1 + r1); nav2 *= (1 + r2); nav_g *= (1 + rg)
        new_total = nav1 + nav2 + nav_g
        port[day] = new_total / prev_total - 1 if prev_total > 0 else 0

    return pd.Series(port).sort_index(), cb1, cb2, cb_events


def run_v16_control(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                    start_date, capture_events=False):
    """V16-B control with CB event logging for comparison."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb1 = 0; cb2 = 0
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10

    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_mode = "cash"
    scores = {}
    cb_events = []

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False
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

        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; cb1 += 1
                p1_mode = "qqq"  # V16-B: unlevered equity
                if capture_events:
                    cb_events.append({"date": day, "pod": 1, "prior": "qld"})
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; cb2 += 1
                p2_mode = "ivv"  # V16-B: unlevered equity
                if capture_events:
                    cb_events.append({"date": day, "pod": 2, "prior": "sso"})

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

    return pd.Series(port).sort_index(), cb1, cb2, cb_events


def main():
    print("=" * 140)
    print("  V19 CB → CASH EXIT — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    print("  Running V19 (CB→cash)...")
    v19_full, v19c1, v19c2, v19_events = run_v19(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
        "2002-01-01", capture_events=True)

    print("  Running V16-B control (CB→equity)...")
    v16_full, v16c1, v16c2, v16_events = run_v16_control(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
        "2002-01-01", capture_events=True)

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
    print(metrics_row("V19 CB→Cash           ", v19_full, v19c1 + v19c2))
    print(metrics_row("V16-B CB→Equity       ", v16_full, v16c1 + v16c2))
    print(metrics_row("V9 QLD+IVVguard       ", v9_full, v9_cb))
    print(metrics_row("Baseline (Sweep-40)   ", bl_full, bl_cb))
    print(metrics_row("QQQ B&H               ", qqq_full))

    # ── TABLE 2: CB event-by-event comparison ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: CB EVENT-BY-EVENT COMPARISON (post-CB return: equity vs cash)")
    print(f"{'=' * 140}")

    # For each CB event, compute the return from CB date to next month-end
    # for both V19 (cash) and V16-B (equity)
    v19_monthly = v19_full.resample("MS").apply(lambda x: (1+x).prod()-1)
    v16_monthly = v16_full.resample("MS").apply(lambda x: (1+x).prod()-1)

    print(f"\n  {'CB Date':<12}{'Pod':>5}{'V16 post-CB':>14}{'V19 post-CB':>14}{'Delta':>10}{'Winner':>10}")
    print(f"  {'-'*12}{'-'*4:>5}{'-'*13:>14}{'-'*13:>14}{'-'*9:>10}{'-'*9:>10}")

    equity_wins = 0; cash_wins = 0
    equity_total = 0.0; cash_total = 0.0
    trading_days = daily_ret.loc["2002-01-01":"2026-03-31"].dropna(how="all").index

    # Use V16-B events as reference (same dates as V19)
    for e in v16_events:
        cb_date = e["date"]
        pod = e["pod"]
        # Find next month-start after cb_date
        next_months = [d for d in trading_days if d > cb_date and
                       d.month != cb_date.month or d.year != cb_date.year]
        # Get return from CB date to end of current month
        month_end_days = [d for d in trading_days if d.month == cb_date.month and
                          d.year == cb_date.year and d >= cb_date]

        if len(month_end_days) > 0:
            v16_post = (1 + v16_full.reindex(month_end_days).fillna(0)).prod() - 1
            v19_post = (1 + v19_full.reindex(month_end_days).fillna(0)).prod() - 1
            delta = v19_post - v16_post
            winner = "CASH" if v19_post > v16_post else "EQUITY"
            if winner == "CASH": cash_wins += 1
            else: equity_wins += 1
            equity_total += v16_post; cash_total += v19_post
            print(f"  {cb_date.strftime('%Y-%m-%d'):<12}{pod:>5}{v16_post:>13.2%}{v19_post:>13.2%}"
                  f"{delta:>+9.2%}{winner:>10}")

    total_events = equity_wins + cash_wins
    print(f"\n  Summary: Equity wins {equity_wins}/{total_events}, Cash wins {cash_wins}/{total_events}")
    print(f"  Cumulative: equity post-CB return {equity_total:+.2%}, cash post-CB return {cash_total:+.2%}")
    print(f"  Mean per event: equity {equity_total/total_events:+.2%}, cash {cash_total/total_events:+.2%}")

    # ── TABLE 3: Crisis drawdowns ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: CRISIS DRAWDOWNS")
    print(f"{'=' * 140}")
    crises = [
        ("Dot-com 02-03", "2002-01-01", "2003-03-31"),
        ("GFC 07-09",     "2007-11-01", "2009-03-31"),
        ("COVID 2020",    "2020-02-01", "2020-04-30"),
        ("2022 bear",     "2022-01-01", "2022-12-31"),
    ]
    names = ["V19", "V16-B", "V9", "BL"]
    all_s = [v19_full, v16_full, v9_full, bl_full]
    print(f"\n  {'Crisis':<18}" + "".join(f"{nm:>10}" for nm in names))
    print(f"  {'-' * 18}" + "".join(f" {'-' * 9}" for _ in names))
    for label, cs, ce in crises:
        cells = []
        for s in all_s:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            cells.append(max_dd(sp) if len(sp) > 5 else 0)
        print(f"  {label:<18}" + "".join(f"{c:>10.1%}" for c in cells))

    # ── TABLE 4: COVID day-by-day ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: COVID FEB-MAR 2020 — DAY-BY-DAY")
    print(f"{'=' * 140}")
    covid_range = v16_full.loc["2020-02-20":"2020-03-31"].index
    v16_cum = (1 + v16_full.reindex(covid_range).fillna(0)).cumprod()
    v19_cum = (1 + v19_full.reindex(covid_range).fillna(0)).cumprod()
    v16_dd = (v16_cum - v16_cum.expanding().max()) / v16_cum.expanding().max()
    v19_dd = (v19_cum - v19_cum.expanding().max()) / v19_cum.expanding().max()

    print(f"\n  {'Date':<12}{'V16-B cum':>12}{'V16-B DD':>11}{'V19 cum':>12}{'V19 DD':>11}{'Delta DD':>11}")
    print(f"  {'-'*12}{'-'*11:>12}{'-'*10:>11}{'-'*11:>12}{'-'*10:>11}{'-'*10:>11}")
    for d in covid_range:
        if d in v16_cum.index and d in v19_cum.index:
            print(f"  {d.strftime('%Y-%m-%d'):<12}{v16_cum[d]:>12.4f}{v16_dd[d]:>10.2%}"
                  f"{v19_cum[d]:>12.4f}{v19_dd[d]:>10.2%}{v19_dd[d]-v16_dd[d]:>+10.2%}")

    # ── TABLE 5: Recovery speed ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: RECOVERY-PERIOD RETURNS")
    print(f"{'=' * 140}")
    recoveries = [
        ("GFC trough → 1yr",    "2009-03-09", "2010-03-09"),
        ("COVID trough → 6mo",  "2020-03-23", "2020-09-23"),
        ("2022 trough → 6mo",   "2022-10-12", "2023-04-12"),
    ]
    print(f"\n  {'Window':<22}{'V19':>10}{'V16-B':>10}{'V9':>10}")
    print(f"  {'-'*22}{'-'*9:>10}{'-'*9:>10}{'-'*9:>10}")
    for label, cs, ce in recoveries:
        cells = []
        for s in [v19_full, v16_full, v9_full]:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            cells.append((1+sp).prod()-1 if len(sp) > 5 else 0)
        print(f"  {label:<22}" + "".join(f"{c:>10.2%}" for c in cells))

    # ── TABLE 6: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 6: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<24}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 24}" + "".join(f" {'-' * 9}" for _ in start_dates))
    for nm, fn in [("V19 CB→Cash", "v19"), ("V16-B CB→Equity", "v16"), ("V9", "v9"), ("Baseline", "bl")]:
        row = f"  {nm:<24}"
        for sd in start_dates:
            if fn == "v19":
                s, _, _, _ = run_v19(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v16":
                s, _, _, _ = run_v16_control(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
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
    show = [("V19", v19_full), ("V16-B", v16_full), ("V9", v9_full)]
    print(f"\n  {'Year':<6}" + "".join(f"{nm:>13}" for nm, _ in show) + f"{'QQQ':>13}{'V19-QQQ':>13}")
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
        row += f"${vals['QQQ']/1e3:>11.0f}K ${(vals['V19']-vals['QQQ'])/1e3:>11.0f}K"
        print(row)

    # ── TABLE 8: Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 8: PASS / FAIL")
    print(f"{'=' * 140}")
    v19_c = cagr(v19_full); v19_sh = sharpe_r(v19_full); v19_dd = max_dd(v19_full)
    v16_c = cagr(v16_full); v16_sh = sharpe_r(v16_full); v16_dd = max_dd(v16_full)

    dd_ok = v19_dd > v16_dd + 0.05  # 5pp improvement
    sh_ok = v19_sh >= v16_sh
    cagr_ok = v19_c >= v16_c - 0.015

    print(f"\n  V19: CAGR {v19_c:.2%}, Sharpe {v19_sh:.3f}, MaxDD {v19_dd:.1%}")
    print(f"  V16: CAGR {v16_c:.2%}, Sharpe {v16_sh:.3f}, MaxDD {v16_dd:.1%}")
    print(f"\n  MaxDD < -22.0% (5pp improvement): {v19_dd:.1%} → {'✓' if dd_ok else '✗'}")
    print(f"  Sharpe ≥ V16-B (0.846):             {v19_sh:.3f} → {'✓' if sh_ok else '✗'}")
    print(f"  CAGR ≥ 15.56% (within 1.5pp):       {v19_c:.2%} → {'✓' if cagr_ok else '✗'}")
    print(f"  → {'PASS' if dd_ok and sh_ok and cagr_ok else 'FAIL'}")

    # Post-CB window verdict
    if total_events > 0:
        print(f"\n  Post-CB window verdict:")
        print(f"    Equity wins {equity_wins}/{total_events} ({equity_wins/total_events:.0%})")
        print(f"    Cash wins {cash_wins}/{total_events} ({cash_wins/total_events:.0%})")
        print(f"    → {'CB→equity is correct' if equity_wins > cash_wins else 'CB→cash is correct' if cash_wins > equity_wins else 'Coin flip'}")

    print()


if __name__ == "__main__":
    main()
