"""V19d-QQQ: 60/30/10 QQQ Tilt Test."""

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


def run_v19d_weighted(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                      start_date, w1=0.45, w2=0.45, wg=0.10):
    """V19d with configurable pod weights."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}
    nav1 = w1; nav2 = w2; nav_g = wg
    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_mode = "cash"; gold_delev = False
    scores = {}

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
                    drift = max(abs(nav1/total - w1), abs(nav2/total - w2), abs(nav_g/total - wg))
                    if drift > 0.05:
                        nav1 = total * w1; nav2 = total * w2; nav_g = total * wg

        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; p1_mode = "cash"
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; p2_mode = "cash"
        if gold_mode == "iau" and not gold_delev:
            if check_breach(day, "IAU", dpdf, daily_smas):
                gold_mode = "cash"; gold_delev = True

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

    return pd.Series(port).sort_index()


def main():
    print("=" * 140)
    print("  V19d-QQQ: 60/30/10 QQQ TILT TEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    print("  Running V19d-60 (60/30/10)...")
    v60 = run_v19d_weighted(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                             "2002-01-01", w1=0.60, w2=0.30, wg=0.10)
    print("  Running V19d (45/45/10 control)...")
    v45 = run_v19d_weighted(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                             "2002-01-01", w1=0.45, w2=0.45, wg=0.10)
    print("  Running V9...")
    v9, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running Baseline...")
    bl, _ = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                          dbmf_ret, dbmf_inception, "2002-01-01")
    qqq = daily_ret["QQQ"].loc["2002-01-01":"2026-03-31"].dropna()
    ivv = daily_ret["IVV"].loc["2002-01-01":"2026-03-31"].dropna()

    # ── TABLE 1 ──
    print(f"\n{'=' * 140}")
    print("  TABLE 1: CORE METRICS (2002-2026)")
    print(f"{'=' * 140}")
    print(f"\n  {'Strategy':<24} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9}")
    print(f"  {'-' * 24} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9}")
    for nm, s in [("V19d-60 (60/30/10)", v60), ("V19d (45/45/10)", v45),
                   ("V9 QLD+IVVguard", v9), ("Baseline", bl), ("QQQ B&H", qqq), ("IVV B&H", ivv)]:
        c = cagr(s); v = s.std()*np.sqrt(252); sh = sharpe_r(s); so = sortino_r(s)
        dd = max_dd(s); cl = calmar_r(s); t = (1+s).cumprod().iloc[-1]
        sm = s.resample("MS").apply(lambda x: (1+x).prod()-1); dca = dca_terminal(sm)
        print(f"  {nm:<24} {c:>6.2%} {v:>6.2%} {sh:>7.3f} {so:>8.3f} {dd:>6.1%} {cl:>7.2f} ${t:>8.2f} ${dca/1e6:>7.2f}M")

    # ── TABLE 2: Crisis drawdowns ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: CRISIS DRAWDOWNS")
    print(f"{'=' * 140}")
    crises = [("Dot-com","2002-01-01","2003-03-31"),("GFC","2007-11-01","2009-03-31"),
              ("COVID","2020-02-01","2020-04-30"),("2022","2022-01-01","2022-12-31")]
    names = ["V19d-60", "V19d", "V9", "BL"]
    all_s = [v60, v45, v9, bl]
    print(f"\n  {'Crisis':<18}" + "".join(f"{nm:>10}" for nm in names))
    for label, cs, ce in crises:
        cells = [max_dd(s[(s.index >= cs) & (s.index <= ce)]) for s in all_s]
        print(f"  {label:<18}" + "".join(f"{c:>10.1%}" for c in cells))

    # ── TABLE 3: Key divergence years ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: ANNUAL RETURNS (QQQ vs IVV divergence years)")
    print(f"{'=' * 140}")
    print(f"\n  {'Year':<6}{'V19d-60':>9}{'V19d':>9}{'QQQ':>9}{'IVV':>9}{'QQQ-IVV':>9}")
    for yr in [2005, 2007, 2013, 2017, 2020, 2022, 2023]:
        row = f"  {yr:<6}"
        for s in [v60, v45, qqq, ivv]:
            sp = s[s.index.year == yr]
            row += f"{((1+sp).prod()-1):>9.2%}"
        sq = qqq[qqq.index.year == yr]; si = ivv[ivv.index.year == yr]
        row += f"{((1+sq).prod()-1) - ((1+si).prod()-1):>+9.2%}"
        print(row)

    # ── TABLE 4: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<24}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    for nm, w in [("V19d-60 (60/30/10)", (0.60, 0.30, 0.10)), ("V19d (45/45/10)", (0.45, 0.45, 0.10))]:
        row = f"  {nm:<24}"
        for sd in start_dates:
            s = run_v19d_weighted(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                  sd, w1=w[0], w2=w[1], wg=w[2])
            row += f"{cagr(s):>10.2%}"
        print(row)
    row = f"  {'V9':<24}"
    for sd in start_dates:
        s, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
        row += f"{cagr(s):>10.2%}"
    print(row)

    # ── TABLE 5: Verdict ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: VERDICT")
    print(f"{'=' * 140}")
    c60 = cagr(v60); sh60 = sharpe_r(v60); dd60 = max_dd(v60); t60 = (1+v60).cumprod().iloc[-1]
    c45 = cagr(v45); sh45 = sharpe_r(v45); dd45 = max_dd(v45); t45 = (1+v45).cumprod().iloc[-1]

    print(f"\n  V19d-60: CAGR {c60:.2%}, Sharpe {sh60:.3f}, MaxDD {dd60:.1%}, Terminal ${t60:.2f}")
    print(f"  V19d:    CAGR {c45:.2%}, Sharpe {sh45:.3f}, MaxDD {dd45:.1%}, Terminal ${t45:.2f}")
    print(f"\n  Delta: CAGR {c60-c45:+.2%}, Sharpe {sh60-sh45:+.3f}, MaxDD {dd60-dd45:+.1%}, Terminal ${t60-t45:+.2f}")

    pref = (c60 > c45) and (sh60 >= sh45 - 0.02) and (dd60 >= dd45 - 0.03)
    wash = abs(c60 - c45) < 0.005

    if wash:
        print(f"\n  → WASH (CAGR diff < 0.5pp). Keep 45/45/10.")
    elif pref:
        print(f"\n  → V19d-60 PREFERRED (more CAGR, acceptable Sharpe/DD cost)")
    else:
        reasons = []
        if sh60 < sh45 - 0.02: reasons.append(f"Sharpe drops too much ({sh60:.3f} < {sh45-0.02:.3f})")
        if dd60 < dd45 - 0.03: reasons.append(f"MaxDD too deep ({dd60:.1%} < {dd45-0.03:.1%})")
        print(f"\n  → V19d-60 NOT PREFERRED: {'; '.join(reasons)}")

    print()


if __name__ == "__main__":
    main()
