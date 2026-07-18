"""V18b: Intra-Month Portfolio Circuit Breaker (-20%).

V16-B base with overlay: if portfolio drops X% within a single month
(from month-open value), strip leverage (SSO→IVV, QLD→QQQ) for the
rest of the month. Monthly reset prevents V18's whipsaw problem.
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


def run_v16_imc(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                start_date, imc_threshold=-0.20, capture_diag=False):
    """V16-B + intra-month circuit breaker."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb1 = 0; cb2 = 0
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10
    month_open_value = nav1 + nav2 + nav_g
    imc_active = False

    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_mode = "cash"
    scores = {}

    imc_events = []
    diag_months = []

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False; imc_active = False
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

            # Rebalance
            if day != trading_days[0]:
                total = nav1 + nav2 + nav_g
                if total > 0:
                    drift = max(abs(nav1/total - 0.45), abs(nav2/total - 0.45), abs(nav_g/total - 0.10))
                    if drift > 0.05:
                        nav1 = total * 0.45; nav2 = total * 0.45; nav_g = total * 0.10

            month_open_value = nav1 + nav2 + nav_g

            if capture_diag:
                diag_months.append({"month": day, "qqq_sc": sc_q, "ivv_sc": sc_i,
                                    "iau_sc": scores["IAU"],
                                    "p1_mode": p1_mode, "p2_mode": p2_mode})

        # Per-asset CB (fires before IMC check)
        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; cb1 += 1; p1_mode = "qqq"
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; cb2 += 1; p2_mode = "ivv"

        # Compute return BEFORE IMC check (today's return uses current holding)
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

        # IMC check (after today's return computed, at close)
        if not imc_active and month_open_value > 0:
            intra_month_ret = (new_total - month_open_value) / month_open_value
            if intra_month_ret <= imc_threshold:
                # Strip leverage for rest of month — takes effect NEXT day
                if p1_lev: p1_lev = False; p1_mode = "qqq"
                if p2_lev: p2_lev = False; p2_mode = "ivv"
                imc_active = True

                # Count trading days into month
                month_start_idx = None
                for i, d in enumerate(trading_days):
                    if d.month == day.month and d.year == day.year:
                        if month_start_idx is None: month_start_idx = i
                days_into = trading_days.get_loc(day) - month_start_idx + 1

                imc_events.append({
                    "date": day, "intra_ret": intra_month_ret,
                    "days_into_month": days_into,
                    "month_open": month_open_value, "value_at_trigger": new_total,
                })

    # Post-process: compute month-end value for each IMC event
    s = pd.Series(port).sort_index()
    for e in imc_events:
        m = e["date"].month; y = e["date"].year
        month_days = s[(s.index.month == m) & (s.index.year == y)]
        if len(month_days) > 0:
            month_cum = (1 + month_days).prod()
            e["full_month_ret"] = month_cum - 1

    return s, cb1, cb2, imc_events, diag_months


def run_v16_nooverlay(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, start_date):
    """V16-B control — same as run_v16 with threshold=3."""
    s, c1, c2, rb, _ = run_v16(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                start_date, iau_threshold=3)
    return s, c1 + c2


def main():
    print("=" * 140)
    print("  V18b INTRA-MONTH PORTFOLIO CIRCUIT BREAKER — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    # Run all thresholds
    variants = {}
    for thresh in [-0.15, -0.18, -0.20, -0.25]:
        name = f"IMC-{abs(int(thresh*100))}"
        print(f"  Running {name}...")
        s, c1, c2, events, diag = run_v16_imc(
            daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            "2002-01-01", imc_threshold=thresh, capture_diag=True)
        variants[name] = (s, c1 + c2, events, diag)

    print("  Running V16-B control...")
    v16_full, v16_cb = run_v16_nooverlay(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")

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
    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9} {'IMC':>5}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 5}")
    for nm in ["IMC-15", "IMC-18", "IMC-20", "IMC-25"]:
        s, cb, events, _ = variants[nm]
        c = cagr(s); v = s.std()*np.sqrt(252); sh = sharpe_r(s); so = sortino_r(s)
        dd = max_dd(s); cl = calmar_r(s); t = (1+s).cumprod().iloc[-1]
        sm = s.resample("MS").apply(lambda x: (1+x).prod()-1)
        dca = dca_terminal(sm)
        print(f"  {nm:<22} {c:>6.2%} {v:>6.2%} {sh:>7.3f} {so:>8.3f} {dd:>6.1%} {cl:>7.2f} ${t:>8.2f} ${dca/1e6:>7.2f}M {len(events):>5}")
    print(metrics_row("V16-B (no IMC)        ", v16_full, v16_cb))
    print(metrics_row("V9 QLD+IVVguard       ", v9_full, v9_cb))
    print(metrics_row("Baseline (Sweep-40)   ", bl_full, bl_cb))

    # ── TABLE 2: IMC event log ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: IMC EVENT LOG (all thresholds)")
    print(f"{'=' * 140}")
    for nm in ["IMC-15", "IMC-18", "IMC-20", "IMC-25"]:
        events = variants[nm][2]
        print(f"\n  {nm} ({len(events)} events):")
        if events:
            print(f"  {'Date':<12}{'Ret@trigger':>13}{'Days in':>8}{'Month ret':>11}")
            print(f"  {'-'*12}{'-'*12:>13}{'-'*7:>8}{'-'*10:>11}")
            for e in events:
                mr = e.get("full_month_ret", np.nan)
                print(f"  {e['date'].strftime('%Y-%m-%d'):<12}{e['intra_ret']:>12.2%}{e['days_into_month']:>8}"
                      f"{mr:>10.2%}" if not np.isnan(mr) else
                      f"  {e['date'].strftime('%Y-%m-%d'):<12}{e['intra_ret']:>12.2%}{e['days_into_month']:>8}{'N/A':>11}")

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
    names = ["IMC-15", "IMC-18", "IMC-20", "IMC-25", "V16-B", "BL"]
    all_series = [variants[n][0] for n in ["IMC-15","IMC-18","IMC-20","IMC-25"]] + [v16_full, bl_full]
    print(f"\n  {'Crisis':<18}" + "".join(f"{nm:>10}" for nm in names))
    print(f"  {'-' * 18}" + "".join(f" {'-' * 9}" for _ in names))
    for label, cs, ce in crises:
        cells = []
        for s in all_series:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            cells.append(max_dd(sp) if len(sp) > 5 else 0)
        print(f"  {label:<18}" + "".join(f"{c:>10.1%}" for c in cells))

    # ── TABLE 4: COVID day-by-day (for IMC-20) ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: COVID MARCH 2020 — DAY-BY-DAY (V16-B vs IMC-20)")
    print(f"{'=' * 140}")
    covid_days = daily_ret.loc["2020-02-15":"2020-04-15"].index
    v16_daily = v16_full.reindex(covid_days).dropna()
    imc20_daily = variants["IMC-20"][0].reindex(covid_days).dropna()
    v16_cum = (1 + v16_daily).cumprod()
    imc20_cum = (1 + imc20_daily).cumprod()
    v16_dd_series = (v16_cum - v16_cum.expanding().max()) / v16_cum.expanding().max()
    imc20_dd_series = (imc20_cum - imc20_cum.expanding().max()) / imc20_cum.expanding().max()

    # Show key days
    print(f"\n  {'Date':<12}{'V16-B daily':>13}{'V16-B DD':>11}{'IMC-20 daily':>14}{'IMC-20 DD':>12}")
    print(f"  {'-'*12}{'-'*12:>13}{'-'*10:>11}{'-'*13:>14}{'-'*11:>12}")
    key_dates = sorted(set(
        list(v16_dd_series.nsmallest(10).index) +
        [d for d in covid_days if d.month == 3 and d.year == 2020 and d in v16_daily.index]
    ))
    for d in key_dates:
        if d in v16_daily.index and d in imc20_daily.index:
            print(f"  {d.strftime('%Y-%m-%d'):<12}{v16_daily[d]:>12.2%}{v16_dd_series[d]:>10.2%}"
                  f"{imc20_daily[d]:>13.2%}{imc20_dd_series[d]:>11.2%}")

    # ── TABLE 5: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<22}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 22}" + "".join(f" {'-' * 9}" for _ in start_dates))
    for nm in ["IMC-15", "IMC-18", "IMC-20", "IMC-25"]:
        row = f"  {nm:<22}"
        thresh = -int(nm.split("-")[1]) / 100
        for sd in start_dates:
            s, _, _, _, _ = run_v16_imc(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                         sd, imc_threshold=thresh)
            row += f"{cagr(s):>10.2%}"
        print(row)
    for nm, fn in [("V16-B", "v16"), ("V9", "v9"), ("Baseline", "bl")]:
        row = f"  {nm:<22}"
        for sd in start_dates:
            if fn == "v16":
                s, _ = run_v16_nooverlay(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
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

    # ── TABLE 6: DCA ──
    print(f"\n{'=' * 140}")
    print("  TABLE 6: DCA TERMINAL BY YEAR-END (2013 start)")
    print(f"{'=' * 140}")
    # Pick best IMC + controls
    best_nm = min(["IMC-15","IMC-18","IMC-20","IMC-25"],
                  key=lambda n: abs(sharpe_r(variants[n][0]) - 0.846))
    show = [(best_nm, variants[best_nm][0]), ("V16-B", v16_full), ("V9", v9_full)]
    print(f"\n  {'Year':<6}" + "".join(f"{nm:>13}" for nm, _ in show) + f"{'QQQ':>13}{best_nm+'-QQQ':>14}")
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
        row += f"${vals['QQQ']/1e3:>11.0f}K ${(vals[best_nm]-vals['QQQ'])/1e3:>12.0f}K"
        print(row)

    # ── TABLE 7: Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 7: PASS / FAIL")
    print(f"{'=' * 140}")
    v16_c = cagr(v16_full); v16_sh = sharpe_r(v16_full); v16_dd = max_dd(v16_full)
    print(f"\n  V16-B reference: CAGR {v16_c:.2%}, Sharpe {v16_sh:.3f}, MaxDD {v16_dd:.1%}")

    for nm in ["IMC-15", "IMC-18", "IMC-20", "IMC-25"]:
        s, cb, events, _ = variants[nm]
        vc = cagr(s); vsh = sharpe_r(s); vdd = max_dd(s)
        dd_ok = vdd > v16_dd + 0.03  # 3pp improvement (less negative)
        sh_ok = vsh >= 0.840
        cagr_ok = vc >= 0.16
        freq_ok = len(events) <= 10
        passed = dd_ok and sh_ok and cagr_ok and freq_ok

        print(f"\n  {nm} ({len(events)} events):")
        print(f"    MaxDD:  {vdd:.1%} vs V16-B {v16_dd:.1%} (need < {v16_dd+0.03:.1%}) → {'✓' if dd_ok else '✗'}")
        print(f"    Sharpe: {vsh:.3f} (need ≥ 0.840) → {'✓' if sh_ok else '✗'}")
        print(f"    CAGR:   {vc:.2%} (need ≥ 16.0%) → {'✓' if cagr_ok else '✗'}")
        print(f"    Events: {len(events)} (need ≤ 10) → {'✓' if freq_ok else '✗'}")
        print(f"    → {'PASS' if passed else 'FAIL'}")

    print()


if __name__ == "__main__":
    main()
