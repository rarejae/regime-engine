"""V14: V9 + Defensive Rotation During Cash Periods.

V9's offense is byte-for-byte identical. The ONLY change: when V9 holds cash,
V14 rotates to Faber-gated defensive assets (IVV, VGLT, IAU, DBC).

Variants:
  A: IVV in defensive pool when IVV_score >= 2
  B: IVV in defensive pool when IVV_score >= 3
  C: No IVV in defensive pool (VGLT/IAU/DBC only)
  D: V9 control (100% cash during off-signal)
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


def run_v14(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            start_date, ivv_threshold=2, include_ivv=True, capture_diag=False):
    """V14: V9 offense + defensive rotation during off-signal.

    ivv_threshold: 2 (variant A) or 3 (variant B)
    include_ivv: True for A/B, False for C
    """
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb_count = 0
    diag = {"monthly": [], "off_signal_months": []}

    # State
    mode = "cash"  # "offense_qld", "offense_qqq", "defense", "cash"
    holding_detail = {}  # asset -> weight
    lev = False; delevered = False
    cur_scores = {}

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            delevered = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day

            # Compute all scores
            cur_scores = {}
            for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
                cur_scores[a] = asset_score(sd, a, dpdf, daily_smas)

            sc_q = cur_scores["QQQ"]; sc_i = cur_scores["IVV"]

            # V9 offense logic (identical to V9)
            if sc_q >= 3:
                if sc_i <= 1:
                    # IVV guard fires — QQQ at 1×
                    mode = "offense_qqq"; holding_detail = {"QQQ": 1.0}; lev = False
                else:
                    # Full conviction
                    mode = "offense_qld"; holding_detail = {"QLD": 1.0}; lev = True
            elif sc_q == 2:
                if sc_i >= 2:
                    # V9's partial — 70% QQQ + 30% cash (keep 30% as cash, per spec Option 1)
                    mode = "offense_qqq"; holding_detail = {"QQQ": 0.70, "cash": 0.30}; lev = False
                else:
                    # QQQ 2 but IVV weak — defense
                    mode = "defense"; lev = False
                    holding_detail = _build_defensive(cur_scores, ivv_threshold, include_ivv)
            else:
                # QQQ 0-1 → defense
                mode = "defense"; lev = False
                holding_detail = _build_defensive(cur_scores, ivv_threshold, include_ivv)

            if capture_diag:
                diag["monthly"].append({
                    "month": day, "qqq_sc": sc_q, "ivv_sc": sc_i,
                    "vglt_sc": cur_scores["VGLT"], "iau_sc": cur_scores["IAU"],
                    "dbc_sc": cur_scores["DBC"],
                    "mode": mode, "holding": dict(holding_detail),
                })

        # Daily CB — only when holding QLD (leveraged)
        if lev and not delevered:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                # V9 CB: QLD → QQQ at next open (keep 1× exposure)
                lev = False; delevered = True; cb_count += 1
                holding_detail = {"QQQ": 1.0}
                mode = "offense_qqq"

        # Compute return
        rfr = float(rfr_daily.get(day, 0.0))
        ret = 0.0
        for a, w in holding_detail.items():
            if w <= 0.0001: continue
            if a == "cash":
                ret += w * rfr
            elif a == "QLD":
                qu = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
                if lev:
                    ret += w * lev_ret(qu, rfr, QLD_EXP, day, actual_lev, "QLD", both_start)
                else:
                    ret += w * qu  # post-CB, holding QQQ at 1×
            elif a == "QQQ":
                qu = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
                ret += w * qu
            else:
                # Defensive asset (IVV, VGLT, IAU, DBC)
                r = float(dr.get(a, 0.0)) if pd.notna(dr.get(a, np.nan)) else 0.0
                ret += w * r

        port[day] = ret

    return pd.Series(port).sort_index(), cb_count, diag


def _build_defensive(scores, ivv_threshold, include_ivv):
    """Build equal-weight defensive pool from Faber-gated assets."""
    candidates = []
    if include_ivv and scores.get("IVV", 0) >= ivv_threshold:
        candidates.append("IVV")
    if scores.get("VGLT", 0) >= 2: candidates.append("VGLT")
    if scores.get("IAU", 0) >= 2:  candidates.append("IAU")
    if scores.get("DBC", 0) >= 2:  candidates.append("DBC")

    if len(candidates) == 0:
        return {"cash": 1.0}
    w = 1.0 / len(candidates)
    return {a: w for a in candidates}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 140)
    print("  V14 DEFENSIVE ROTATION DURING V9 CASH PERIODS — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    # Run all variants
    variants = {}
    for name, kwargs in [
        ("V14-A (IVV≥2)", {"ivv_threshold": 2, "include_ivv": True}),
        ("V14-B (IVV≥3)", {"ivv_threshold": 3, "include_ivv": True}),
        ("V14-C (no IVV)", {"ivv_threshold": 2, "include_ivv": False}),
    ]:
        print(f"  Running {name}...")
        s, cb, diag = run_v14(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                              "2002-01-01", capture_diag=True, **kwargs)
        variants[name] = (s, cb, diag)

    print("  Running V9 (control)...")
    v9_full, v9_cb = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    variants["V9 (control)"] = (v9_full, v9_cb, None)

    print("  Running V12...")
    v12_full, v12_cb, _ = run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running Baseline...")
    bl_full, bl_cb = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
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
    for nm in ["V14-A (IVV≥2)", "V14-B (IVV≥3)", "V14-C (no IVV)", "V9 (control)"]:
        s, cb, _ = variants[nm]
        print(metrics_row(nm, s, cb))
    print(metrics_row("V12 Independent 2×", v12_full, v12_cb))
    print(metrics_row("Baseline (Sweep-40)", bl_full, bl_cb))
    print(metrics_row("QQQ B&H", qqq_full))

    # ── TABLE 2: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<22}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 22}" + "".join(f" {'-' * 9}" for _ in start_dates))

    cagr_map = {}
    for nm, kwargs in [
        ("V14-A (IVV≥2)", {"ivv_threshold": 2, "include_ivv": True}),
        ("V14-B (IVV≥3)", {"ivv_threshold": 3, "include_ivv": True}),
        ("V14-C (no IVV)", {"ivv_threshold": 2, "include_ivv": False}),
    ]:
        row = f"  {nm:<22}"; cagr_map[nm] = {}
        for sd in start_dates:
            s, _, _ = run_v14(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                              sd, **kwargs)
            c = cagr(s); cagr_map[nm][sd] = c
            row += f"{c:>10.2%}"
        print(row)
    for nm, fn in [("V9 (control)", "v9"), ("V12 Indep 2×", "v12"), ("Baseline", "bl")]:
        row = f"  {nm:<22}"; cagr_map[nm] = {}
        for sd in start_dates:
            if fn == "v9":
                s, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v12":
                s, _, _ = run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            else:
                s, _ = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                    dbmf_ret, dbmf_inception, sd)
            c = cagr(s); cagr_map[nm][sd] = c
            row += f"{c:>10.2%}"
        print(row)
    row = f"  {'QQQ B&H':<22}"
    for sd in start_dates:
        qs = qqq_full[qqq_full.index >= pd.Timestamp(sd)]
        row += f"{cagr(qs):>10.2%}"
    print(row)

    # ── TABLE 3: Defensive utilization diagnostics ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: DEFENSIVE UTILIZATION DURING V9 OFF-SIGNAL MONTHS")
    print(f"{'=' * 140}")

    # Use V14-A diagnostics (richest info)
    diag_a = variants["V14-A (IVV≥2)"][2]
    off_months = [m for m in diag_a["monthly"] if m["mode"] == "defense"]
    total_off = len(off_months)
    print(f"\n  Total off-signal months (V9 would hold cash): {total_off}")

    # Compute monthly returns for each strategy during off-signal months
    v14a_monthly = variants["V14-A (IVV≥2)"][0].resample("MS").apply(lambda x: (1+x).prod()-1)
    v14b_monthly = variants["V14-B (IVV≥3)"][0].resample("MS").apply(lambda x: (1+x).prod()-1)
    v14c_monthly = variants["V14-C (no IVV)"][0].resample("MS").apply(lambda x: (1+x).prod()-1)
    v9_monthly = v9_full.resample("MS").apply(lambda x: (1+x).prod()-1)

    # Mean return during off-signal months
    off_rets = {"V14-A": [], "V14-B": [], "V14-C": [], "V9": []}
    for m in off_months:
        ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
        for nm, series in [("V14-A", v14a_monthly), ("V14-B", v14b_monthly),
                           ("V14-C", v14c_monthly), ("V9", v9_monthly)]:
            r = series.get(ts, np.nan)
            if not pd.isna(r): off_rets[nm].append(r)

    print(f"\n  Mean monthly return during off-signal months:")
    for nm in ["V14-A", "V14-B", "V14-C", "V9"]:
        if off_rets[nm]:
            m = np.mean(off_rets[nm])
            s = np.std(off_rets[nm])
            pos = sum(1 for r in off_rets[nm] if r > 0)
            print(f"    {nm:>8}: {m:>+6.2%} (vol {s:.2%}, {pos}/{len(off_rets[nm])} positive)")

    # Per-asset contribution during off-signal months
    print(f"\n  Defensive asset activity during off-signal months:")
    asset_counts = {"IVV": 0, "VGLT": 0, "IAU": 0, "DBC": 0, "cash_only": 0}
    for m in off_months:
        held = [a for a in m["holding"] if a != "cash"]
        if not held:
            asset_counts["cash_only"] += 1
        for a in held:
            if a in asset_counts: asset_counts[a] += 1
    for a in ["IVV", "VGLT", "IAU", "DBC", "cash_only"]:
        print(f"    {a:>10}: {asset_counts[a]:>4}/{total_off} months ({asset_counts[a]/total_off:.0%})")

    # How many defensives active per off-signal month
    n_active_counts = {}
    for m in off_months:
        n = len([a for a in m["holding"] if a != "cash"])
        n_active_counts[n] = n_active_counts.get(n, 0) + 1
    print(f"\n  Number of active defensives per off-signal month:")
    for n in sorted(n_active_counts.keys()):
        print(f"    {n} assets: {n_active_counts[n]} months")

    # ── TABLE 4: 2022 detail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: 2022 MONTH-BY-MONTH (V14-A)")
    print(f"{'=' * 140}")
    bl_monthly = bl_full.resample("MS").apply(lambda x: (1+x).prod()-1)
    print(f"\n  {'Month':<10}{'QQQ':>5}{'IVV':>5}{'Mode':>10}{'Defensives':>25}{'V14-A':>9}{'V9':>9}{'BL':>9}")
    print(f"  {'-'*10}{'-'*4:>5}{'-'*4:>5}{'-'*9:>10}{'-'*24:>25}{'-'*8:>9}{'-'*8:>9}{'-'*8:>9}")
    for m in diag_a["monthly"]:
        if m["month"].year == 2022:
            ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
            r14a = v14a_monthly.get(ts, np.nan)
            r9 = v9_monthly.get(ts, np.nan)
            rbl = bl_monthly.get(ts, np.nan)
            defs = "+".join(a for a in m["holding"] if a != "cash") or "cash"
            print(f"  {m['month'].strftime('%Y-%m'):<10}{m['qqq_sc']:>5}{m['ivv_sc']:>5}"
                  f"{m['mode']:>10}{defs:>25}{r14a:>9.2%}{r9:>9.2%}{rbl:>9.2%}")

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
    header_names = ["V14-A", "V14-B", "V14-C", "V9", "V12", "BL", "QQQ"]
    all_series = [variants["V14-A (IVV≥2)"][0], variants["V14-B (IVV≥3)"][0],
                  variants["V14-C (no IVV)"][0], v9_full, v12_full, bl_full, qqq_full]
    print(f"\n  {'Crisis':<18}" + "".join(f"{nm:>10}" for nm in header_names))
    print(f"  {'-' * 18}" + "".join(f" {'-' * 9}" for _ in header_names))
    for label, cs, ce in crises:
        cells = []
        for s in all_series:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            cells.append(max_dd(sp) if len(sp) > 5 else 0)
        print(f"  {label:<18}" + "".join(f"{c:>10.1%}" for c in cells))

    # ── TABLE 6: DCA dollar gap ──
    print(f"\n{'=' * 140}")
    print("  TABLE 6: DCA TERMINAL BY YEAR-END (2013 start)")
    print(f"{'=' * 140}")
    show = ["V14-A (IVV≥2)", "V14-B (IVV≥3)", "V14-C (no IVV)", "V9 (control)"]
    print(f"\n  {'Year':<6}" + "".join(f"{nm:>16}" for nm in show) + f"{'QQQ':>13}{'Best-QQQ':>13}")
    for yr in range(2013, 2027):
        end = f"{yr}-12-31"
        vals = {}
        for nm in show:
            s = variants[nm][0]
            sp = s[(s.index >= "2013-01-01") & (s.index <= end)]
            sm = sp.resample("MS").apply(lambda x: (1+x).prod()-1)
            vals[nm] = dca_terminal(sm)
        qs = qqq_full[(qqq_full.index >= "2013-01-01") & (qqq_full.index <= end)]
        qm = qs.resample("MS").apply(lambda x: (1+x).prod()-1)
        qd = dca_terminal(qm)
        best = max(vals.values())
        best_nm = [k for k, v in vals.items() if v == best][0]
        row = f"  {yr:<6}"
        for nm in show:
            row += f"${vals[nm]/1e3:>14.0f}K"
        row += f"${qd/1e3:>11.0f}K ${(best-qd)/1e3:>11.0f}K"
        print(row)

    # ── TABLE 7: Recovery speed ──
    print(f"\n{'=' * 140}")
    print("  TABLE 7: RECOVERY-PERIOD RETURNS")
    print(f"{'=' * 140}")
    recoveries = [
        ("GFC trough → 1yr",    "2009-03-09", "2010-03-09"),
        ("COVID trough → 6mo",  "2020-03-23", "2020-09-23"),
        ("2022 trough → 6mo",   "2022-10-12", "2023-04-12"),
    ]
    rec_names = ["V14-A", "V14-B", "V14-C", "V9", "QQQ"]
    rec_series = [variants["V14-A (IVV≥2)"][0], variants["V14-B (IVV≥3)"][0],
                  variants["V14-C (no IVV)"][0], v9_full, qqq_full]
    print(f"\n  {'Window':<22}" + "".join(f"{nm:>10}" for nm in rec_names))
    print(f"  {'-'*22}" + "".join(f" {'-'*9}" for _ in rec_names))
    for label, cs, ce in recoveries:
        cells = []
        for s in rec_series:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            cells.append((1+sp).prod()-1 if len(sp) > 5 else 0)
        print(f"  {label:<22}" + "".join(f"{c:>10.2%}" for c in cells))

    # ── TABLE 8: Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 8: PASS / FAIL")
    print(f"{'=' * 140}")
    v9_c = cagr(v9_full); v9_sh = sharpe_r(v9_full); v9_dd = max_dd(v9_full)

    for nm in ["V14-A (IVV≥2)", "V14-B (IVV≥3)", "V14-C (no IVV)"]:
        s = variants[nm][0]
        vc = cagr(s); vsh = sharpe_r(s); vdd = max_dd(s)

        sh_ok = vsh >= v9_sh
        dd_ok = vdd >= v9_dd  # >= because DD is negative; shallower = larger number
        cagr_ok = vc >= v9_c - 0.01
        bonus = vsh > 0.80 and vc > 0.18

        passed = sh_ok and dd_ok and cagr_ok

        print(f"\n  {nm}:")
        print(f"    CAGR:   {vc:.2%} vs V9 {v9_c:.2%} → {'✓' if cagr_ok else '✗'} (within 1pp)")
        print(f"    Sharpe: {vsh:.3f} vs V9 {v9_sh:.3f} → {'✓' if sh_ok else '✗'}")
        print(f"    MaxDD:  {vdd:.1%} vs V9 {v9_dd:.1%} → {'✓' if dd_ok else '✗'}")
        print(f"    → {'PASS' if passed else 'FAIL'}")
        if bonus:
            print(f"    BONUS: Sharpe > 0.80 AND CAGR > 18% → FRONTIER LEADER")

    # Fail conditions
    if off_rets["V14-A"] and off_rets["V9"]:
        def_mean = np.mean(off_rets["V14-A"]); cash_mean = np.mean(off_rets["V9"])
        print(f"\n  Defensive pool vs cash during off-signal:")
        print(f"    V14-A mean: {def_mean:+.2%}, V9 cash mean: {cash_mean:+.2%}")
        print(f"    → {'Defensives beat cash ✓' if def_mean > cash_mean else 'Cash beats defensives ✗ — premise rejected'}")

    print()


if __name__ == "__main__":
    main()
