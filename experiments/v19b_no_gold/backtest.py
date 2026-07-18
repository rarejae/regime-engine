"""V19b: V19 Without Gold — 50/50 QLD/SSO, CB → Cash.

Isolates whether gold is necessary in V19's CB→cash architecture.
Essentially V15 (50/50 two-pod) with CB→cash instead of CB→equity.
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
from experiments.v19_cb_cash_exit.backtest import run_v19


def run_v19b(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, start_date):
    """50/50 QLD/SSO, CB → cash, no gold."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb1 = 0; cb2 = 0
    nav1 = 0.50; nav2 = 0.50

    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    scores = {}

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = {"QQQ": asset_score(sd, "QQQ", dpdf, daily_smas),
                      "IVV": asset_score(sd, "IVV", dpdf, daily_smas)}

            sc_q = scores["QQQ"]; sc_i = scores["IVV"]
            if sc_q >= 3:
                if sc_i <= 1: p1_mode = "qqq"; p1_lev = False
                else: p1_mode = "qld"; p1_lev = True
            elif sc_q == 2: p1_mode = "qqq_partial"; p1_lev = False
            else: p1_mode = "cash"; p1_lev = False

            if sc_i >= 3: p2_mode = "sso"; p2_lev = True
            elif sc_i == 2: p2_mode = "ivv_partial"; p2_lev = False
            else: p2_mode = "cash"; p2_lev = False

            if day != trading_days[0]:
                total = nav1 + nav2
                if total > 0:
                    w1 = nav1 / total
                    if abs(w1 - 0.50) > 0.05:
                        nav1 = total * 0.50; nav2 = total * 0.50

        # CB → CASH (V19 behavior)
        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; cb1 += 1; p1_mode = "cash"
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; cb2 += 1; p2_mode = "cash"

        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0

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

        prev_total = nav1 + nav2
        nav1 *= (1 + r1); nav2 *= (1 + r2)
        new_total = nav1 + nav2
        port[day] = new_total / prev_total - 1 if prev_total > 0 else 0

    return pd.Series(port).sort_index(), cb1 + cb2


def main():
    print("=" * 140)
    print("  V19b: V19 WITHOUT GOLD (50/50 CB→CASH) — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    print("  Running V19b (50/50, no gold, CB→cash)...")
    v19b_full, v19b_cb = run_v19b(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")

    print("  Running V19 (45/45/10, gold, CB→cash)...")
    v19_full, v19c1, v19c2, _ = run_v19(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")

    print("  Running V15 (50/50, CB→equity)...")
    v15_full, _, _, _, _, _, _ = run_v15(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")

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
    print(f"\n  {'Strategy':<26} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9} {'CB':>4}")
    print(f"  {'-' * 26} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 4}")
    print(metrics_row("V19b 50/50 CB→Cash (noAU)", v19b_full, v19b_cb))
    print(metrics_row("V19 45/45/10 CB→Cash     ", v19_full, v19c1 + v19c2))
    print(metrics_row("V15 50/50 CB→Equity      ", v15_full, 27))
    print(metrics_row("V9 QLD+IVVguard          ", v9_full, v9_cb))
    print(metrics_row("Baseline (Sweep-40)      ", bl_full, bl_cb))
    print(metrics_row("QQQ B&H                  ", qqq_full))

    # ── TABLE 2: Crisis drawdowns ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: CRISIS DRAWDOWNS")
    print(f"{'=' * 140}")
    crises = [
        ("Dot-com 02-03", "2002-01-01", "2003-03-31"),
        ("GFC 07-09",     "2007-11-01", "2009-03-31"),
        ("COVID 2020",    "2020-02-01", "2020-04-30"),
        ("2022 bear",     "2022-01-01", "2022-12-31"),
    ]
    names = ["V19b", "V19", "V15", "V9", "BL"]
    all_s = [v19b_full, v19_full, v15_full, v9_full, bl_full]
    print(f"\n  {'Crisis':<18}" + "".join(f"{nm:>10}" for nm in names))
    print(f"  {'-' * 18}" + "".join(f" {'-' * 9}" for _ in names))
    for label, cs, ce in crises:
        cells = []
        for s in all_s:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            cells.append(max_dd(sp) if len(sp) > 5 else 0)
        print(f"  {label:<18}" + "".join(f"{c:>10.1%}" for c in cells))

    # ── TABLE 3: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<26}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 26}" + "".join(f" {'-' * 9}" for _ in start_dates))
    for nm, fn in [("V19b 50/50 CB→Cash", "v19b"), ("V19 45/45/10 CB→Cash", "v19"),
                    ("V15 50/50 CB→Equity", "v15"), ("V9", "v9"), ("Baseline", "bl")]:
        row = f"  {nm:<26}"
        for sd in start_dates:
            if fn == "v19b":
                s, _ = run_v19b(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v19":
                s, _, _, _ = run_v19(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v15":
                s, _, _, _, _, _, _ = run_v15(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v9":
                s, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            else:
                s, _ = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                    dbmf_ret, dbmf_inception, sd)
            row += f"{cagr(s):>10.2%}"
        print(row)
    row = f"  {'QQQ B&H':<26}"
    for sd in start_dates:
        qs = qqq_full[qqq_full.index >= pd.Timestamp(sd)]
        row += f"{cagr(qs):>10.2%}"
    print(row)

    # ── TABLE 4: DCA ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: DCA TERMINAL BY YEAR-END (2013 start)")
    print(f"{'=' * 140}")
    show = [("V19b", v19b_full), ("V19", v19_full), ("V15", v15_full), ("V9", v9_full)]
    print(f"\n  {'Year':<6}" + "".join(f"{nm:>13}" for nm, _ in show) + f"{'QQQ':>13}")
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
        row += f"${vals['QQQ']/1e3:>11.0f}K"
        print(row)

    # ── TABLE 5: Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: PASS / FAIL — DOES GOLD EARN ITS 10%?")
    print(f"{'=' * 140}")
    v19b_c = cagr(v19b_full); v19b_sh = sharpe_r(v19b_full); v19b_dd = max_dd(v19b_full)
    v19_c = cagr(v19_full); v19_sh = sharpe_r(v19_full); v19_dd = max_dd(v19_full)
    v15_c = cagr(v15_full); v15_sh = sharpe_r(v15_full); v15_dd = max_dd(v15_full)

    print(f"\n  V19b (no gold): CAGR {v19b_c:.2%}, Sharpe {v19b_sh:.3f}, MaxDD {v19b_dd:.1%}")
    print(f"  V19  (gold):    CAGR {v19_c:.2%}, Sharpe {v19_sh:.3f}, MaxDD {v19_dd:.1%}")
    print(f"  V15  (no gold, CB→eq): CAGR {v15_c:.2%}, Sharpe {v15_sh:.3f}, MaxDD {v15_dd:.1%}")

    gold_unnecessary = (v19b_sh >= v19_sh) and (v19b_dd >= v19_dd) and (v19b_c >= v19_c)
    print(f"\n  V19b dominates V19 (gold unnecessary)?")
    print(f"    Sharpe: {v19b_sh:.3f} vs {v19_sh:.3f} → {'✓' if v19b_sh >= v19_sh else '✗'}")
    print(f"    MaxDD:  {v19b_dd:.1%} vs {v19_dd:.1%} → {'✓' if v19b_dd >= v19_dd else '✗'}")
    print(f"    CAGR:   {v19b_c:.2%} vs {v19_c:.2%} → {'✓' if v19b_c >= v19_c else '✗'}")
    print(f"    → {'Gold is UNNECESSARY — drop it' if gold_unnecessary else 'Gold EARNS its 10% — keep it'}")

    # Also check: V19b vs V15 (isolates CB→cash improvement without gold)
    print(f"\n  CB→cash improvement without gold (V19b vs V15):")
    print(f"    Sharpe: {v19b_sh:.3f} vs {v15_sh:.3f} ({v19b_sh - v15_sh:+.3f})")
    print(f"    MaxDD:  {v19b_dd:.1%} vs {v15_dd:.1%} ({v19b_dd - v15_dd:+.1%})")
    print(f"    CAGR:   {v19b_c:.2%} vs {v15_c:.2%} ({v19b_c - v15_c:+.2%})")

    print()


if __name__ == "__main__":
    main()
