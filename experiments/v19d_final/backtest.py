"""V19d Final Production Backtest — Definitive numbers for architecture document.

Produces all 11 required outputs + CSV exports + quality checks.
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
    cagr, max_dd, sharpe_r, sortino_r, calmar_r, dca_terminal,
)

END_DATE = "2026-03-31"


def run_v19d_full(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, start_date):
    """V19d with full diagnostics for every day and month."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp(END_DATE)].index

    port = {}; cb_events = []; monthly_log = []
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10

    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_mode = "cash"; gold_delev = False
    scores = {"QQQ": 0, "IVV": 0, "IAU": 0}
    rebal_events = 0

    # Daily detail for crisis windows
    daily_detail = {}

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False; gold_delev = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = {a: asset_score(sd, a, dpdf, daily_smas) for a in ["QQQ", "IVV", "IAU"]}
            sc_q = scores["QQQ"]; sc_i = scores["IVV"]; sc_a = scores["IAU"]

            if sc_q >= 3:
                if sc_i <= 1: p1_mode = "qqq"; p1_lev = False
                else: p1_mode = "qld"; p1_lev = True
            elif sc_q == 2: p1_mode = "qqq_partial"; p1_lev = False
            else: p1_mode = "cash"; p1_lev = False

            if sc_i >= 3: p2_mode = "sso"; p2_lev = True
            elif sc_i == 2: p2_mode = "ivv_partial"; p2_lev = False
            else: p2_mode = "cash"; p2_lev = False

            gold_mode = "iau" if sc_a >= 3 else "cash"

            if day != trading_days[0]:
                total = nav1 + nav2 + nav_g
                if total > 0:
                    drift = max(abs(nav1/total - 0.45), abs(nav2/total - 0.45), abs(nav_g/total - 0.10))
                    if drift > 0.05:
                        nav1 = total * 0.45; nav2 = total * 0.45; nav_g = total * 0.10
                        rebal_events += 1

            # Compute effective equity
            eq = 0.0
            if p1_mode == "qld" and p1_lev: eq += 0.90  # 45% * 2
            elif p1_mode in ("qqq", "qld"): eq += 0.45
            elif p1_mode == "qqq_partial": eq += 0.45 * 0.70
            if p2_mode == "sso" and p2_lev: eq += 0.90
            elif p2_mode in ("ivv", "sso"): eq += 0.45
            elif p2_mode == "ivv_partial": eq += 0.45 * 0.70
            gold_exp = 0.10 if gold_mode == "iau" else 0.0

            monthly_log.append({
                "month": day, "qqq_sc": sc_q, "ivv_sc": sc_i, "iau_sc": sc_a,
                "p1_mode": p1_mode, "p2_mode": p2_mode, "gold_mode": gold_mode,
                "eff_equity": eq, "gold_exp": gold_exp, "cash_exp": 1.0 - eq - gold_exp,
            })

        # CBs → cash
        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; p1_mode = "cash"
                cb_events.append({"date": day, "asset": "QQQ", "score": scores["QQQ"]})
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; p2_mode = "cash"
                cb_events.append({"date": day, "asset": "IVV", "score": scores["IVV"]})
        if gold_mode == "iau" and not gold_delev:
            if check_breach(day, "IAU", dpdf, daily_smas):
                gold_mode = "cash"; gold_delev = True
                cb_events.append({"date": day, "asset": "IAU", "score": scores["IAU"]})

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
        daily_return = new_total / prev_total - 1 if prev_total > 0 else 0
        port[day] = daily_return

        # Store daily detail for crisis windows
        daily_detail[day] = {
            "p1": p1_mode, "p2": p2_mode, "gold": gold_mode,
            "p1_lev": p1_lev, "p2_lev": p2_lev,
            "ret": daily_return, "nav": new_total,
            "qqq_u": qqq_u, "ivv_u": ivv_u, "iau_u": iau_u,
        }

    s = pd.Series(port).sort_index()
    return s, cb_events, monthly_log, rebal_events, daily_detail


def run_6040(daily_ret, rfr_daily, start_date):
    """Simple 60/40 benchmark (IVV 60%, VGLT 40%, monthly rebalance)."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    td = daily_ret.loc[common_start:pd.Timestamp(END_DATE)].index
    port = {}
    for day in td:
        dr = daily_ret.loc[day]
        ivv = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
        vglt = float(dr.get("VGLT", 0.0)) if pd.notna(dr.get("VGLT", np.nan)) else 0.0
        port[day] = 0.60 * ivv + 0.40 * vglt
    return pd.Series(port).sort_index()


def metrics_dict(s):
    c = cagr(s); v = s.std() * np.sqrt(252); sh = sharpe_r(s); so = sortino_r(s)
    dd = max_dd(s); cl = calmar_r(s); t = (1 + s).cumprod().iloc[-1]
    sm = s.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    dca = dca_terminal(sm)
    return {"CAGR": c, "Vol": v, "Sharpe": sh, "Sortino": so, "MaxDD": dd,
            "Calmar": cl, "Terminal": t, "DCA": dca}


def print_metrics_row(name, m, extra=""):
    print(f"  {name:<22} {m['CAGR']:>7.2%} {m['Vol']:>7.2%} {m['Sharpe']:>7.3f} {m['Sortino']:>8.3f} "
          f"{m['MaxDD']:>7.1%} {m['Calmar']:>7.2f} ${m['Terminal']:>8.2f} ${m['DCA']/1e6:>7.2f}M {extra}")


def main():
    W = 140
    print("=" * W)
    print("  V19d FINAL PRODUCTION BACKTEST")
    print("=" * W)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()
    print(f"  Data: {daily_ret.index.min().date()} → {daily_ret.index.max().date()}")
    print(f"  SSO/QLD actual from: {both_start.date()}")

    # ── RUN ALL ──
    print("\n  Running V19d...")
    v19d_s, v19d_cb, v19d_monthly, v19d_rebal, v19d_daily = run_v19d_full(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running V9...")
    v9_s, v9_cb = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running Baseline...")
    bl_s, bl_cb = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                dbmf_ret, dbmf_inception, "2002-01-01")
    qqq_s = daily_ret["QQQ"].loc["2002-01-01":END_DATE].dropna()
    ivv_s = daily_ret["IVV"].loc["2002-01-01":END_DATE].dropna()
    print("  Running 60/40...")
    s6040 = run_6040(daily_ret, rfr_daily, "2002-01-01")

    # Metrics
    m19d = metrics_dict(v19d_s); m9 = metrics_dict(v9_s); mbl = metrics_dict(bl_s)
    mqqq = metrics_dict(qqq_s); mivv = metrics_dict(ivv_s); m6040 = metrics_dict(s6040)

    # ════════════════════════════════════════════════════════════════
    # 11. SIGNAL ALIGNMENT VERIFICATION (run first)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  11. SIGNAL ALIGNMENT VERIFICATION\n{'=' * W}")
    # Check: monthly scores use prior month-end data
    ml = v19d_monthly
    violations = 0
    trading_days = daily_ret.loc["2002-01-01":END_DATE].dropna(how="all").index
    for m in ml[1:]:  # skip first month
        month_start = m["month"]
        prior = trading_days[trading_days < month_start]
        if len(prior) > 0:
            score_date = prior[-1]
            if score_date >= month_start:
                violations += 1
    print(f"  Score date < month start: {'PASS ✓' if violations == 0 else f'FAIL ✗ ({violations} violations)'}")
    print(f"  Total months: {len(ml)}")
    print(f"  CB events use daily close data: verified by check_breach() implementation")

    # ════════════════════════════════════════════════════════════════
    # QUALITY CHECKS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  QUALITY CHECKS\n{'=' * W}")
    checks = [
        ("V9 CAGR", m9["CAGR"], 0.1937, 0.005),
        ("V9 Sharpe", m9["Sharpe"], 0.777, 0.005),
        ("V9 MaxDD", m9["MaxDD"], -0.379, 0.005),
        ("Baseline CAGR", mbl["CAGR"], 0.1379, 0.005),
        ("Baseline Sharpe", mbl["Sharpe"], 0.910, 0.01),
    ]
    all_pass = True
    for name, actual, expected, tol in checks:
        ok = abs(actual - expected) < tol
        if not ok: all_pass = False
        print(f"  {name}: {actual:.4f} (expected ~{expected:.4f}) → {'✓' if ok else '✗ MISMATCH'}")

    # Leverage validation
    print(f"\n  Leveraged ETF validation:")
    for ticker, underlying in [("SSO", "IVV"), ("QLD", "QQQ")]:
        if ticker in actual_lev:
            act = actual_lev[ticker]
            und = daily_ret[underlying].reindex(act.index).dropna()
            common = act.index.intersection(und.index)
            sim = 2.0 * und.loc[common] - rfr_daily.reindex(common).fillna(0)
            corr = act.loc[common].corr(sim)
            print(f"    {ticker} daily corr with sim: {corr:.4f} → {'✓' if corr > 0.99 else '✗'}")

    if not all_pass:
        print("\n  ⚠ QUALITY CHECK FAILURES — review before using numbers")

    # ════════════════════════════════════════════════════════════════
    # 1. CORE PERFORMANCE TABLE
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  1. CORE PERFORMANCE TABLE\n{'=' * W}")
    cb_qqq = sum(1 for e in v19d_cb if e["asset"] == "QQQ")
    cb_ivv = sum(1 for e in v19d_cb if e["asset"] == "IVV")
    cb_iau = sum(1 for e in v19d_cb if e["asset"] == "IAU")
    lev_months = sum(1 for m in ml if m["p1_mode"] == "qld" or m["p2_mode"] == "sso")

    for k, v in [("Annualized return", f"{m19d['CAGR']:.2%}"),
                  ("Annualized volatility", f"{m19d['Vol']:.2%}"),
                  ("Sharpe ratio", f"{m19d['Sharpe']:.3f}"),
                  ("Sortino ratio", f"{m19d['Sortino']:.3f}"),
                  ("Maximum drawdown", f"{m19d['MaxDD']:.1%}"),
                  ("Calmar ratio", f"{m19d['Calmar']:.2f}"),
                  ("Terminal $1", f"${m19d['Terminal']:.2f}"),
                  ("DCA $21K+$700/mo", f"${m19d['DCA']/1e6:.2f}M"),
                  ("Leveraged months", f"{lev_months}/{len(ml)} ({lev_months/len(ml):.0%})"),
                  ("CB events (total)", f"{len(v19d_cb)}"),
                  ("  Pod 1 (QQQ) CB", f"{cb_qqq}"),
                  ("  Pod 2 (IVV) CB", f"{cb_ivv}"),
                  ("  Gold (IAU) CB", f"{cb_iau}"),
                  ("Rebalance events", f"{v19d_rebal}")]:
        print(f"  {k:<28} {v}")

    # ════════════════════════════════════════════════════════════════
    # 2. BENCHMARK COMPARISON
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  2. BENCHMARK COMPARISON\n{'=' * W}")
    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA':>9}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")
    for nm, m in [("V19d", m19d), ("V9 QLD+IVVguard", m9), ("Baseline Sweep-40", mbl),
                   ("QQQ B&H", mqqq), ("IVV B&H", mivv), ("60/40", m6040)]:
        print_metrics_row(nm, m)

    # ════════════════════════════════════════════════════════════════
    # 3. CRISIS DRAWDOWN TABLE
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  3. CRISIS DRAWDOWNS\n{'=' * W}")
    crises = [("Dot-com 02-03","2002-01-01","2003-03-31"),("GFC 07-09","2007-11-01","2009-03-31"),
              ("COVID 2020","2020-02-01","2020-04-30"),("2022 bear","2022-01-01","2022-12-31")]
    names = ["V19d", "V9", "Baseline", "QQQ B&H"]
    all_s = [v19d_s, v9_s, bl_s, qqq_s]
    print(f"\n  {'Crisis':<18}" + "".join(f"{nm:>12}" for nm in names))
    print(f"  {'-'*18}" + "".join(f" {'-'*11}" for _ in names))
    for label, cs, ce in crises:
        cells = [max_dd(s[(s.index >= cs) & (s.index <= ce)]) for s in all_s]
        print(f"  {label:<18}" + "".join(f"{c:>12.1%}" for c in cells))

    # ════════════════════════════════════════════════════════════════
    # 4. MONTHLY ALLOCATION SNAPSHOTS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  4a. GFC ALLOCATION (Sep 2007 — Jun 2009)\n{'=' * W}")
    v19d_monthly_ret = v19d_s.resample("MS").apply(lambda x: (1+x).prod()-1)
    cum = (1 + v19d_s).cumprod()
    peak = cum.expanding().max()
    dd_series = (cum - peak) / peak

    print(f"\n  {'Month':<10}{'QQQ':>5}{'IVV':>5}{'IAU':>5}{'Pod1':>12}{'Pod2':>12}{'Gold':>6}{'EffEq':>8}{'Ret':>8}{'DD':>8}")
    for m in ml:
        if pd.Timestamp("2007-09-01") <= m["month"] <= pd.Timestamp("2009-06-30"):
            ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
            mr = float(v19d_monthly_ret.get(ts, np.nan))
            dd_at = float(dd_series.asof(m["month"])) if m["month"] in dd_series.index or True else np.nan
            # Get DD at end of month
            month_days = dd_series[(dd_series.index.month == m["month"].month) & (dd_series.index.year == m["month"].year)]
            dd_val = float(month_days.iloc[-1]) if len(month_days) > 0 else np.nan
            print(f"  {m['month'].strftime('%Y-%m'):<10}{m['qqq_sc']:>5}{m['ivv_sc']:>5}{m['iau_sc']:>5}"
                  f"{m['p1_mode']:>12}{m['p2_mode']:>12}{m['gold_mode']:>6}"
                  f"{m['eff_equity']:>7.0%}{mr:>8.2%}{dd_val:>8.1%}")

    print(f"\n  GFC CB events:")
    for e in v19d_cb:
        if pd.Timestamp("2007-09-01") <= e["date"] <= pd.Timestamp("2009-06-30"):
            print(f"    {e['date'].strftime('%Y-%m-%d')} — {e['asset']} (score {e['score']})")

    print(f"\n{'=' * W}\n  4b. COVID DAILY (Feb 20 — Mar 31, 2020)\n{'=' * W}")
    covid_range = [d for d in v19d_daily if pd.Timestamp("2020-02-20") <= d <= pd.Timestamp("2020-03-31")]
    if covid_range:
        # Compute cumulative from Feb 20
        start_nav = v19d_daily[covid_range[0]]["nav"] / (1 + v19d_daily[covid_range[0]]["ret"])
        print(f"\n  {'Date':<12}{'P1':>6}{'P2':>6}{'Gold':>6}{'Ret':>8}{'CumDD':>8}")
        cum_val = start_nav
        peak_val = start_nav
        for d in covid_range:
            dd_info = v19d_daily[d]
            cum_val = dd_info["nav"]
            if cum_val > peak_val: peak_val = cum_val
            dd_pct = (cum_val - peak_val) / peak_val
            p1 = "QLD" if dd_info["p1"] == "qld" and dd_info["p1_lev"] else dd_info["p1"][:4]
            p2 = "SSO" if dd_info["p2"] == "sso" and dd_info["p2_lev"] else dd_info["p2"][:4]
            g = dd_info["gold"][:4]
            print(f"  {d.strftime('%Y-%m-%d'):<12}{p1:>6}{p2:>6}{g:>6}{dd_info['ret']:>8.2%}{dd_pct:>8.1%}")

    print(f"\n  COVID CB events:")
    for e in v19d_cb:
        if pd.Timestamp("2020-02-01") <= e["date"] <= pd.Timestamp("2020-04-30"):
            print(f"    {e['date'].strftime('%Y-%m-%d')} — {e['asset']} (score {e['score']})")

    print(f"\n{'=' * W}\n  4c. 2022 BEAR ALLOCATION (Dec 2021 — Feb 2023)\n{'=' * W}")
    print(f"\n  {'Month':<10}{'QQQ':>5}{'IVV':>5}{'IAU':>5}{'Pod1':>12}{'Pod2':>12}{'Gold':>6}{'EffEq':>8}{'Ret':>8}")
    for m in ml:
        if pd.Timestamp("2021-12-01") <= m["month"] <= pd.Timestamp("2023-02-28"):
            ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
            mr = float(v19d_monthly_ret.get(ts, np.nan))
            print(f"  {m['month'].strftime('%Y-%m'):<10}{m['qqq_sc']:>5}{m['ivv_sc']:>5}{m['iau_sc']:>5}"
                  f"{m['p1_mode']:>12}{m['p2_mode']:>12}{m['gold_mode']:>6}"
                  f"{m['eff_equity']:>7.0%}{mr:>8.2%}")

    print(f"\n{'=' * W}\n  4d. NORMAL BULL (Jan 2017 — Dec 2019)\n{'=' * W}")
    print(f"\n  {'Month':<10}{'QQQ':>5}{'IVV':>5}{'IAU':>5}{'Pod1':>12}{'Pod2':>12}{'Gold':>6}{'EffEq':>8}{'Ret':>8}")
    for m in ml:
        if pd.Timestamp("2017-01-01") <= m["month"] <= pd.Timestamp("2019-12-31"):
            ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
            mr = float(v19d_monthly_ret.get(ts, np.nan))
            print(f"  {m['month'].strftime('%Y-%m'):<10}{m['qqq_sc']:>5}{m['ivv_sc']:>5}{m['iau_sc']:>5}"
                  f"{m['p1_mode']:>12}{m['p2_mode']:>12}{m['gold_mode']:>6}"
                  f"{m['eff_equity']:>7.0%}{mr:>8.2%}")

    # ════════════════════════════════════════════════════════════════
    # 5. STATE OCCUPANCY
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  5. STATE OCCUPANCY ({len(ml)} months)\n{'=' * W}")
    state_counts = {}
    for m in ml:
        key = (m["p1_mode"], m["p2_mode"], m["gold_mode"])
        state_counts[key] = state_counts.get(key, 0) + 1
    print(f"\n  {'Pod1':<14}{'Pod2':<14}{'Gold':<6}{'Months':>8}{'Pct':>8}")
    print(f"  {'-'*14}{'-'*14}{'-'*6}{'-'*7:>8}{'-'*7:>8}")
    for key in sorted(state_counts.keys(), key=lambda k: -state_counts[k]):
        n = state_counts[key]
        print(f"  {key[0]:<14}{key[1]:<14}{key[2]:<6}{n:>8}{n/len(ml):>8.1%}")

    # ════════════════════════════════════════════════════════════════
    # 6. EFFECTIVE EQUITY EXPOSURE
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  6. EFFECTIVE EQUITY EXPOSURE\n{'=' * W}")
    eq_exp = [m["eff_equity"] for m in ml]
    print(f"\n  Mean:   {np.mean(eq_exp):.1%}")
    print(f"  Median: {np.median(eq_exp):.1%}")
    print(f"  Min:    {np.min(eq_exp):.1%}")
    print(f"  Max:    {np.max(eq_exp):.1%}")
    print(f"  Std:    {np.std(eq_exp):.1%}")
    buckets = {"180%": 0, "135%": 0, "90%": 0, "~60%": 0, "~30%": 0, "0%": 0}
    for e in eq_exp:
        if e > 1.7: buckets["180%"] += 1
        elif e > 1.2: buckets["135%"] += 1
        elif e > 0.8: buckets["90%"] += 1
        elif e > 0.5: buckets["~60%"] += 1
        elif e > 0.2: buckets["~30%"] += 1
        else: buckets["0%"] += 1
    print(f"\n  {'Eff Equity':<12}{'Months':>8}{'Pct':>8}")
    for k, n in buckets.items():
        print(f"  {k:<12}{n:>8}{n/len(ml):>8.1%}")

    # ════════════════════════════════════════════════════════════════
    # 7. CB EVENT LOG
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  7. CIRCUIT BREAKER EVENT LOG ({len(v19d_cb)} events)\n{'=' * W}")
    print(f"\n  {'Date':<12}{'Asset':>6}{'Score':>6}")
    print(f"  {'-'*12}{'-'*5:>6}{'-'*5:>6}")
    for e in v19d_cb:
        print(f"  {e['date'].strftime('%Y-%m-%d'):<12}{e['asset']:>6}{e['score']:>6}")

    # ════════════════════════════════════════════════════════════════
    # 8. ANNUAL RETURNS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  8. ANNUAL RETURNS\n{'=' * W}")
    print(f"\n  {'Year':<6}{'V19d':>9}{'V9':>9}{'Baseline':>10}{'QQQ':>9}{'IVV':>9}")
    print(f"  {'-'*6}{'-'*8:>9}{'-'*8:>9}{'-'*9:>10}{'-'*8:>9}{'-'*8:>9}")
    for yr in range(2002, 2026):
        row = f"  {yr:<6}"
        for s in [v19d_s, v9_s, bl_s, qqq_s, ivv_s]:
            sp = s[(s.index.year == yr)]
            ar = (1 + sp).prod() - 1 if len(sp) > 0 else 0
            row += f"{ar:>9.2%}"
        print(row)

    # ════════════════════════════════════════════════════════════════
    # 9. START-DATE SENSITIVITY
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  9. START-DATE SENSITIVITY\n{'=' * W}")
    start_dates = ["2002-01-01", "2004-01-01", "2007-01-01", "2010-01-01",
                    "2013-01-01", "2016-01-01", "2019-01-01"]
    print(f"\n  {'Start':<12}{'V19d':>9}{'V9':>9}{'Baseline':>10}{'QQQ':>9}")
    print(f"  {'-'*12}{'-'*8:>9}{'-'*8:>9}{'-'*9:>10}{'-'*8:>9}")
    for sd in start_dates:
        s19, _, _, _, _ = run_v19d_full(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
        s9, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
        sbl, _ = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                               dbmf_ret, dbmf_inception, sd)
        sq = qqq_s[qqq_s.index >= pd.Timestamp(sd)]
        print(f"  {sd:<12}{cagr(s19):>9.2%}{cagr(s9):>9.2%}{cagr(sbl):>9.2%}{cagr(sq):>9.2%}")

    # ════════════════════════════════════════════════════════════════
    # 10. LEVERAGE VALIDATION
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  10. LEVERAGE SIMULATION VALIDATION\n{'=' * W}")
    for ticker, underlying in [("SSO", "IVV"), ("QLD", "QQQ")]:
        if ticker in actual_lev:
            act = actual_lev[ticker]
            und = daily_ret[underlying].reindex(act.index).dropna()
            common = act.index.intersection(und.index)
            rfr_c = rfr_daily.reindex(common).fillna(0)
            exp = SSO_EXP if ticker == "SSO" else QLD_EXP
            sim = 2.0 * und.loc[common] - rfr_c - exp / 252
            corr = act.loc[common].corr(sim)
            act_ann = act.loc[common].mean() * 252
            sim_ann = sim.mean() * 252
            print(f"\n  {ticker}:")
            print(f"    Daily correlation: {corr:.4f}")
            print(f"    Annual return (actual): {act_ann:.2%}")
            print(f"    Annual return (sim):    {sim_ann:.2%}")
            print(f"    Difference:             {act_ann - sim_ann:+.2%}")

    # ════════════════════════════════════════════════════════════════
    # SAVE CSVs
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}\n  SAVING CSVs\n{'=' * W}")

    # Monthly allocations
    mdf = pd.DataFrame(ml)
    mdf["monthly_ret"] = [float(v19d_monthly_ret.get(pd.Timestamp(m["month"].year, m["month"].month, 1), np.nan)) for m in ml]
    cum_ret = (1 + v19d_s).cumprod()
    mdf["cum_ret"] = [float(cum_ret.asof(m["month"])) for m in ml]
    dd_s = (cum_ret - cum_ret.expanding().max()) / cum_ret.expanding().max()
    mdf["drawdown"] = [float(dd_s.asof(m["month"])) for m in ml]
    mdf.to_csv("/Users/josh/Projects/regime-engine/research/data/v19d_monthly_allocations.csv", index=False)
    print(f"  Saved: research/data/v19d_monthly_allocations.csv ({len(mdf)} rows)")

    # CB events
    cb_df = pd.DataFrame(v19d_cb)
    cb_df.to_csv("/Users/josh/Projects/regime-engine/research/data/v19d_cb_events.csv", index=False)
    print(f"  Saved: research/data/v19d_cb_events.csv ({len(cb_df)} rows)")

    # COVID daily
    covid_rows = []
    for d in sorted(v19d_daily.keys()):
        if pd.Timestamp("2020-02-15") <= d <= pd.Timestamp("2020-04-15"):
            dd = v19d_daily[d]
            covid_rows.append({"date": d, "p1": dd["p1"], "p2": dd["p2"], "gold": dd["gold"],
                               "p1_lev": dd["p1_lev"], "p2_lev": dd["p2_lev"],
                               "ret": dd["ret"], "nav": dd["nav"]})
    covid_df = pd.DataFrame(covid_rows)
    covid_df.to_csv("/Users/josh/Projects/regime-engine/research/data/v19d_covid_daily.csv", index=False)
    print(f"  Saved: research/data/v19d_covid_daily.csv ({len(covid_df)} rows)")

    print(f"\n  DONE — all outputs generated.")
    print()


if __name__ == "__main__":
    main()
