"""V12: Independent Faber-Gated 2× on IVV + QQQ.

Two 50/50 sleeves, fully independent Faber gating and leverage.
No defensive assets, no coupling, no beta formula — just two binary switches + cash.

Pass criteria:
  vs Baseline: higher CAGR, higher terminal
  vs V9:        lower max DD, higher Sharpe
  vs V11:       comparable or better CAGR + Sharpe (auto-wins on simplicity)
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# Reuse all V11 helpers for data loading, scoring, CB, benchmark runners.
from experiments.v11_beta_scaled.backtest import (
    SMA_PERIODS, SSO_EXP, QLD_EXP,
    load_data, asset_score, check_breach, lev_ret,
    run_baseline, run_v9, run_v11,
    cagr, max_dd, sharpe_r, sortino_r, calmar_r, dca_terminal, metrics_row,
)


# ── V12 runner ──────────────────────────────────────────────────────────────

def run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            start_date, capture_diag=False):
    """Two 50/50 sleeves. Independent Faber gating + per-asset leverage."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb_count = 0
    diag = {"monthly": []}

    ivv_w = qqq_w = cash_w = 0.0
    ivv_lev = qqq_lev = False
    delev_ivv = delev_qqq = False
    scores = {"IVV": 0, "QQQ": 0}
    month_start_date = None
    month_accum = 1.0

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            # Save prior month diagnostics
            if capture_diag and month_start_date is not None:
                diag["monthly"].append({
                    "month": month_start_date,
                    "ivv_score": scores["IVV"], "qqq_score": scores["QQQ"],
                    "ivv_w": ivv_w, "qqq_w": qqq_w, "cash_w": cash_w,
                    "ivv_lev_start": ivv_lev or delev_ivv,
                    "qqq_lev_start": qqq_lev or delev_qqq,
                    "ret": month_accum - 1.0,
                })

            delev_ivv = delev_qqq = False
            month_accum = 1.0
            month_start_date = day
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = {"IVV": asset_score(sd, "IVV", dpdf, daily_smas),
                      "QQQ": asset_score(sd, "QQQ", dpdf, daily_smas)}

            # IVV sleeve
            sc_i = scores["IVV"]
            if sc_i == 3:
                ivv_w = 0.50; ivv_lev = True
            elif sc_i == 2:
                ivv_w = 0.50 * 0.70; ivv_lev = False  # 35%
            else:
                ivv_w = 0.0; ivv_lev = False

            # QQQ sleeve
            sc_q = scores["QQQ"]
            if sc_q == 3:
                qqq_w = 0.50; qqq_lev = True
            elif sc_q == 2:
                qqq_w = 0.50 * 0.70; qqq_lev = False
            else:
                qqq_w = 0.0; qqq_lev = False

            cash_w = 1.0 - ivv_w - qqq_w

        # Per-asset CB
        if ivv_lev and not delev_ivv:
            if check_breach(day, "IVV", dpdf, daily_smas):
                ivv_lev = False; delev_ivv = True; cb_count += 1
        if qqq_lev and not delev_qqq:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                qqq_lev = False; delev_qqq = True; cb_count += 1

        rfr = float(rfr_daily.get(day, 0.0))
        ret = 0.0

        if ivv_w > 0:
            ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
            if ivv_lev:
                ret += ivv_w * lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start)
            else:
                ret += ivv_w * ivv_u
        if qqq_w > 0:
            qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
            if qqq_lev:
                ret += qqq_w * lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start)
            else:
                ret += qqq_w * qqq_u
        if cash_w > 0:
            ret += cash_w * rfr

        port[day] = ret
        month_accum *= (1 + ret)

    return pd.Series(port).sort_index(), cb_count, diag


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 140)
    print("  V12 INDEPENDENT 2× ON IVV + QQQ — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    print("  Running V12...")
    v12_full, v12_cb, v12_diag = run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                          "2002-01-01", capture_diag=True)
    print("  Running Baseline (Faber-Sweep-40 v5)...")
    bl_full, bl_cb = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                   dbmf_ret, dbmf_inception, "2002-01-01")
    print("  Running V9 (QLD+IVVguard)...")
    v9_full, v9_cb = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running V11 (Beta-Scaled)...")
    v11_full, v11_cb, _ = run_v11(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                   dbmf_ret, dbmf_inception, "2002-01-01")

    qqq_full = daily_ret["QQQ"].loc["2002-01-01":"2026-03-31"].dropna()
    ivv_full = daily_ret["IVV"].loc["2002-01-01":"2026-03-31"].dropna()

    # ── TABLE 1: Core metrics ──
    print(f"\n{'=' * 140}")
    print("  TABLE 1: CORE METRICS (2002-2026)")
    print(f"{'=' * 140}")
    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9} {'CB':>4}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 4}")
    print(metrics_row("V12 Independent 2×", v12_full, v12_cb))
    print(metrics_row("V11 Beta-Scaled", v11_full, v11_cb))
    print(metrics_row("V9 QLD+IVVguard", v9_full, v9_cb))
    print(metrics_row("Baseline (Sweep-40)", bl_full, bl_cb))
    print(metrics_row("QQQ B&H", qqq_full))
    print(metrics_row("IVV B&H", ivv_full))

    # ── TABLE 2: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<22}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 22}" + "".join(f" {'-' * 9}" for _ in start_dates))

    cagr_map = {}  # name -> {start_date: (series, cagr)}
    for name, fn in [
        ("V12 Independent 2×", "v12"),
        ("V11 Beta-Scaled",    "v11"),
        ("V9 QLD+IVVguard",    "v9"),
        ("Baseline",           "bl"),
    ]:
        row = f"  {name:<22}"; cagr_map[name] = {}
        for sd in start_dates:
            if fn == "v12":
                s, _, _ = run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v11":
                s, _, _ = run_v11(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                  dbmf_ret, dbmf_inception, sd)
            elif fn == "v9":
                s, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            else:
                s, _ = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                    dbmf_ret, dbmf_inception, sd)
            c = cagr(s); cagr_map[name][sd] = (s, c)
            row += f"{c:>10.2%}"
        print(row)
    row = f"  {'QQQ B&H':<22}"
    for sd in start_dates:
        qs = qqq_full[qqq_full.index >= pd.Timestamp(sd)]
        row += f"{cagr(qs):>10.2%}"
    print(row)
    qqq_2013_cagr = cagr(qqq_full[qqq_full.index >= pd.Timestamp("2013-01-01")])

    # ── TABLE 3: Signal divergence analysis ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: SIGNAL DIVERGENCE — STATE FREQUENCY + MEAN MONTHLY RETURN")
    print(f"{'=' * 140}")

    # Build a joint DataFrame: monthly V12 returns with state labels + baseline monthly
    v12_monthly = v12_full.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    bl_monthly = bl_full.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    v9_monthly = v9_full.resample("MS").apply(lambda x: (1 + x).prod() - 1)

    # v12_diag["monthly"] records state at start of each month
    state_rows = []
    for m in v12_diag["monthly"]:
        ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
        sc_i = m["ivv_score"]; sc_q = m["qqq_score"]
        if sc_i == 3 and sc_q == 3: lab = "Both 3/3 (both lev)"
        elif sc_i == 3 and sc_q < 3: lab = "IVV 3, QQQ<3"
        elif sc_q == 3 and sc_i < 3: lab = "QQQ 3, IVV<3"
        else: lab = "Neither 3/3"
        state_rows.append({"month": ts, "state": lab, "v12": m["ret"],
                           "bl": float(bl_monthly.get(ts, np.nan)),
                           "v9": float(v9_monthly.get(ts, np.nan))})
    sdf = pd.DataFrame(state_rows)
    total_m = len(sdf)
    print(f"\n  Total months: {total_m}")
    print(f"\n  {'State':<25}{'Count':>8}{'Pct':>8}{'V12 mean':>12}{'BL mean':>12}{'V9 mean':>12}{'V12-BL':>12}")
    print(f"  {'-' * 25}{'-' * 7:>8}{'-' * 7:>8}{'-' * 11:>12}{'-' * 11:>12}{'-' * 11:>12}{'-' * 11:>12}")
    for lab in ["Both 3/3 (both lev)", "IVV 3, QQQ<3", "QQQ 3, IVV<3", "Neither 3/3"]:
        sub = sdf[sdf["state"] == lab]
        if len(sub) > 0:
            n = len(sub)
            v12m = sub["v12"].mean(); blm = sub["bl"].mean(); v9m = sub["v9"].mean()
            print(f"  {lab:<25}{n:>8}{n / total_m:>8.1%}"
                  f"{v12m:>12.2%}{blm:>12.2%}{v9m:>12.2%}{(v12m - blm):>12.2%}")

    # ── TABLE 4: Effective equity exposure histogram ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: V12 EFFECTIVE EQUITY EXPOSURE BY MONTH")
    print(f"{'=' * 140}")
    # Compute eff equity per month from scores
    def eff_equity(sc_i, sc_q):
        e = 0.0
        if sc_i == 3: e += 1.00  # 50%*2
        elif sc_i == 2: e += 0.35
        if sc_q == 3: e += 1.00
        elif sc_q == 2: e += 0.35
        return e
    exposures = []
    for m in v12_diag["monthly"]:
        exposures.append(eff_equity(m["ivv_score"], m["qqq_score"]))
    buckets = {"200%": 0, "135%": 0, "100%": 0, "70%": 0, "35%": 0, "0%": 0}
    for e in exposures:
        if e > 1.9: buckets["200%"] += 1
        elif e > 1.3: buckets["135%"] += 1
        elif e > 0.9: buckets["100%"] += 1
        elif e > 0.65: buckets["70%"] += 1
        elif e > 0.3: buckets["35%"] += 1
        else: buckets["0%"] += 1
    print(f"\n  {'Eff Equity':<15}{'Months':>10}{'Pct':>10}")
    for k, n in buckets.items():
        print(f"  {k:<15}{n:>10}{n / len(exposures):>10.1%}")

    # ── TABLE 5: 2022 detail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: 2022 MONTH-BY-MONTH")
    print(f"{'=' * 140}")
    print(f"\n  {'Month':<10}{'IVV':>5}{'QQQ':>5}{'IVVlev':>8}{'QQQlev':>8}{'V12 ret':>10}{'BL ret':>10}{'V9 ret':>10}")
    print(f"  {'-'*10}{'-'*4:>5}{'-'*4:>5}{'-'*7:>8}{'-'*7:>8}{'-'*9:>10}{'-'*9:>10}{'-'*9:>10}")
    for m in v12_diag["monthly"]:
        if m["month"].year == 2022:
            ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
            blr = float(bl_monthly.get(ts, np.nan)); v9r = float(v9_monthly.get(ts, np.nan))
            print(f"  {m['month'].strftime('%Y-%m'):<10}{m['ivv_score']:>5}{m['qqq_score']:>5}"
                  f"{('Y' if m['ivv_lev_start'] else 'N'):>8}{('Y' if m['qqq_lev_start'] else 'N'):>8}"
                  f"{m['ret']:>10.2%}{blr:>10.2%}{v9r:>10.2%}")

    # ── TABLE 6: DCA dollar gap year-by-year ──
    print(f"\n{'=' * 140}")
    print("  TABLE 6: DCA TERMINAL BY YEAR-END (2013 start, $21K + $700/mo)")
    print(f"{'=' * 140}")
    print(f"\n  {'Year':<6}{'V12':>13}{'V11':>13}{'V9':>13}{'BL':>13}{'QQQ':>13}{'V12-QQQ':>13}")
    print(f"  {'-' * 6}{'-' * 12:>13}{'-' * 12:>13}{'-' * 12:>13}{'-' * 12:>13}{'-' * 12:>13}{'-' * 12:>13}")
    for yr in range(2013, 2027):
        end = f"{yr}-12-31"
        vals = {}
        for nm, s in [("V12", v12_full), ("V11", v11_full), ("V9", v9_full), ("BL", bl_full)]:
            sp = s[(s.index >= "2013-01-01") & (s.index <= end)]
            sm = sp.resample("MS").apply(lambda x: (1 + x).prod() - 1)
            vals[nm] = dca_terminal(sm)
        qs = qqq_full[(qqq_full.index >= "2013-01-01") & (qqq_full.index <= end)]
        qm = qs.resample("MS").apply(lambda x: (1 + x).prod() - 1)
        vals["QQQ"] = dca_terminal(qm)
        print(f"  {yr:<6}${vals['V12']/1e3:>11.0f}K ${vals['V11']/1e3:>11.0f}K "
              f"${vals['V9']/1e3:>11.0f}K ${vals['BL']/1e3:>11.0f}K ${vals['QQQ']/1e3:>11.0f}K "
              f"${(vals['V12']-vals['QQQ'])/1e3:>11.0f}K")

    # ── TABLE 7: Crisis drawdowns ──
    print(f"\n{'=' * 140}")
    print("  TABLE 7: CRISIS DRAWDOWNS")
    print(f"{'=' * 140}")
    crises = [
        ("Dot-com 02-03",    "2002-01-01", "2003-03-31"),
        ("GFC 07-09",        "2007-11-01", "2009-03-31"),
        ("COVID 2020",       "2020-02-01", "2020-04-30"),
        ("2022 bear",        "2022-01-01", "2022-12-31"),
    ]
    print(f"\n  {'Crisis':<18}{'V12':>10}{'V11':>10}{'V9':>10}{'Baseline':>12}{'QQQ B&H':>12}")
    print(f"  {'-' * 18}{'-' * 9:>10}{'-' * 9:>10}{'-' * 9:>10}{'-' * 11:>12}{'-' * 11:>12}")
    for label, cs, ce in crises:
        row = f"  {label:<18}"
        for s in [v12_full, v11_full, v9_full, bl_full, qqq_full]:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            dd = max_dd(sp) if len(sp) > 5 else 0
            row += f"{dd:>10.1%}" if label != "Baseline" else f"{dd:>12.1%}"
            # unify width
        # rebuild properly
        cells = []
        for s in [v12_full, v11_full, v9_full, bl_full, qqq_full]:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            dd = max_dd(sp) if len(sp) > 5 else 0
            cells.append(dd)
        print(f"  {label:<18}{cells[0]:>10.1%}{cells[1]:>10.1%}{cells[2]:>10.1%}"
              f"{cells[3]:>12.1%}{cells[4]:>12.1%}")

    # ── TABLE 8: Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 8: PASS / FAIL")
    print(f"{'=' * 140}")

    v12_c = cagr(v12_full); v12_sh = sharpe_r(v12_full); v12_dd = max_dd(v12_full)
    v12_t = (1 + v12_full).cumprod().iloc[-1]
    bl_c = cagr(bl_full); bl_t = (1 + bl_full).cumprod().iloc[-1]
    v9_sh = sharpe_r(v9_full); v9_dd = max_dd(v9_full)
    v11_c = cagr(v11_full); v11_sh = sharpe_r(v11_full)

    print(f"\n  vs Baseline:")
    print(f"    CAGR:       V12 {v12_c:.2%} vs BL {bl_c:.2%} → {'PASS' if v12_c > bl_c else 'FAIL'}")
    print(f"    Terminal$1: V12 ${v12_t:.2f} vs BL ${bl_t:.2f} → {'PASS' if v12_t > bl_t else 'FAIL'}")

    print(f"\n  vs V9:")
    print(f"    Max DD: V12 {v12_dd:.1%} vs V9 {v9_dd:.1%} → {'PASS' if v12_dd > v9_dd else 'FAIL'}")
    print(f"    Sharpe: V12 {v12_sh:.3f} vs V9 {v9_sh:.3f} → {'PASS' if v12_sh > v9_sh else 'FAIL'}")

    print(f"\n  vs V11:")
    cagr_ok = v12_c >= v11_c - 0.01  # within 1pp
    sh_ok = v12_sh >= v11_sh
    print(f"    CAGR:   V12 {v12_c:.2%} vs V11 {v11_c:.2%} → {'PASS' if cagr_ok else 'FAIL'} (within 1pp)")
    print(f"    Sharpe: V12 {v12_sh:.3f} vs V11 {v11_sh:.3f} → {'PASS' if sh_ok else 'FAIL'}")
    print(f"    Simplicity: automatic PASS (2 switches vs 16-row table)")

    pareto_bl = (v12_c > bl_c) and (v12_t > bl_t)
    pareto_v9 = (v12_dd > v9_dd) and (v12_sh > v9_sh)
    pareto_v11 = cagr_ok and sh_ok
    print(f"\n  OVERALL: vs BL {'PASS' if pareto_bl else 'FAIL'} | "
          f"vs V9 {'PASS' if pareto_v9 else 'FAIL'} | vs V11 {'PASS' if pareto_v11 else 'FAIL'}")
    if pareto_bl and pareto_v9 and pareto_v11:
        print("  → V12 PASSES ALL CRITERIA")
    else:
        print("  → V12 does NOT pass all criteria. See honest tradeoff below.")

    print()


if __name__ == "__main__":
    main()
