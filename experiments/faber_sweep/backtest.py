"""Faber-Sweep: flat substitution sweep to characterize return/Sharpe/drawdown tradeoff.

Single condition. Single leverage level. No tiers, no medians, no transitions.

Each month-end:
1. Run Faber filter as normal
2. If BOTH IVV and QQQ at 3/3 → replace SUB% of each with SSO/QLD
3. If not → full 1x

Daily granularity. Monthly rebalance only. No weekly circuit breaker.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import load_daily_etf_returns, load_monthly_prices, load_monthly_asset_returns
from taa.faber import compute_trend_scores, apply_faber_filter

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE.keys())
EQ_WEIGHT = 0.70  # IVV + QQQ when both at 3/3

SSO_EXPENSE = 0.0089
QLD_EXPENSE = 0.0095

SUB_LEVELS = [0.00, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65]
PRINCIPLED_TARGET = 0.40


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
    monthly_prices = load_monthly_prices()
    trend_df = compute_trend_scores(monthly_prices)

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    return daily_ret, trend_df, rfr_daily


# ── Leveraged return ─────────────────────────────────────────────────────────

def leveraged_daily_return(iw, qw, ir, qr, rfr, sub, base_ret):
    if sub <= 0:
        return iw * ir + qw * qr + base_ret
    sso = 2.0 * ir - rfr - SSO_EXPENSE / 252
    qld = 2.0 * qr - rfr - QLD_EXPENSE / 252
    return iw * ((1 - sub) * ir + sub * sso) + qw * ((1 - sub) * qr + sub * qld) + base_ret


# ── Theoretical (no-drag) leveraged return for ETF drag estimation ───────────

def theoretical_leveraged_return(iw, qw, ir, qr, sub, base_ret):
    """Return as if leverage were free (no rfr cost, no expense)."""
    if sub <= 0:
        return iw * ir + qw * qr + base_ret
    lev_ir = 2.0 * ir  # theoretical 2x, no cost
    lev_qr = 2.0 * qr
    return iw * ((1 - sub) * ir + sub * lev_ir) + qw * ((1 - sub) * qr + sub * lev_qr) + base_ret


# ── Backtest ─────────────────────────────────────────────────────────────────

def run_backtest(daily_ret, trend_df, rfr_daily):
    print(f"\n{'='*110}")
    print(f"  BACKTEST")
    print(f"{'='*110}")

    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index

    print(f"  {len(trading_days)} trading days ({common_start.date()} to {trading_days.max().date()})")
    print(f"  Signal alignment: month T Faber scores applied to month T+1 daily returns")
    print(f"  Leveraged ETF sim: 2x * daily_ret - rfr/252 - expense/252")

    # Strategy labels
    strat_labels = {}
    for sub in SUB_LEVELS:
        if sub == 0:
            strat_labels[sub] = "Faber-1x"
        else:
            strat_labels[sub] = f"Faber-Sweep-{int(sub*100)}"

    all_strats = list(SUB_LEVELS)
    results = {sub: {} for sub in all_strats}
    results_theo = {sub: {} for sub in all_strats}  # theoretical (no drag)
    bench_ivv = {}
    bench_6040 = {}

    cur_trends = {a: 3 for a in ASSETS if a != "cash"}
    w_faber = dict(BASELINE)
    faber_conv = False
    months_levered = 0
    total_months = 0

    for day in trading_days:
        if day not in daily_ret.index:
            continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3:
            continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            total_months += 1

            # Update Faber signals (PIT safe — trend_df already shifted by 1)
            ts_c = trend_df.index[trend_df.index <= day]
            if len(ts_c) > 0:
                for a in ASSETS:
                    if a == "cash":
                        continue
                    if a in trend_df.columns:
                        v = trend_df.loc[ts_c[-1], a]
                        cur_trends[a] = int(v) if pd.notna(v) else 0

            # Faber-only weights: freed capital to cash
            w1, pool = apply_faber_filter(cur_trends, BASELINE)
            w_faber = dict(w1)
            w_faber["cash"] = w_faber.get("cash", 0) + pool

            # Assert PIT
            assert ts_c[-1] <= day, f"Look-ahead: trend date {ts_c[-1]} > {day}"

            # Leverage condition
            faber_conv = (cur_trends.get("IVV", 0) >= 3 and
                          cur_trends.get("QQQ", 0) >= 3)
            if faber_conv:
                months_levered += 1

        iw = w_faber.get("IVV", 0)
        qw = w_faber.get("QQQ", 0)
        ir = actual.get("IVV", 0)
        qr = actual.get("QQQ", 0)
        base_ret = sum(w_faber.get(a, 0) * actual.get(a, 0)
                       for a in avail if a not in ["IVV", "QQQ"])

        for sub in all_strats:
            active_sub = sub if faber_conv else 0.0
            results[sub][day] = leveraged_daily_return(
                iw, qw, ir, qr, rfr, active_sub, base_ret)
            results_theo[sub][day] = theoretical_leveraged_return(
                iw, qw, ir, qr, active_sub, base_ret)

        bench_ivv[day] = actual.get("IVV", 0)
        vglt_ret = actual.get("VGLT", 0)
        bench_6040[day] = 0.60 * actual.get("IVV", 0) + 0.40 * vglt_ret

    ret_series = {sub: pd.Series(d).sort_index() for sub, d in results.items()}
    ret_theo = {sub: pd.Series(d).sort_index() for sub, d in results_theo.items()}
    ivv_s = pd.Series(bench_ivv).sort_index()
    b60_s = pd.Series(bench_6040).sort_index()

    pct_levered = months_levered / total_months if total_months > 0 else 0
    print(f"  Months with leverage condition met: {months_levered}/{total_months} ({pct_levered:.0%})")

    return ret_series, ret_theo, ivv_s, b60_s, strat_labels, pct_levered


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(s):
    ar = s.mean() * 252
    av = s.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]
    ds = neg.std() * np.sqrt(252) if len(neg) > 10 else av
    sortino = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    calmar = ar / abs(dd) if dd != 0 else 0
    final = cum.iloc[-1]
    return {"ar": ar, "av": av, "sh": sh, "sortino": sortino,
            "dd": dd, "calmar": calmar, "final": final}


# ── Report ───────────────────────────────────────────────────────────────────

def print_report(ret_series, ret_theo, ivv_s, b60_s, strat_labels, pct_levered):

    perfs = {sub: compute_metrics(ret_series[sub]) for sub in SUB_LEVELS}
    p_ivv = compute_metrics(ivv_s)
    p_60 = compute_metrics(b60_s)

    # ── Full performance table ───────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  FULL PERFORMANCE TABLE")
    print(f"{'='*140}")
    header = (f"  {'Strategy':<18} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
              f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'vs 60/40':>10} {'vs IVV B&H':>11} {'Mo-Levered':>11}")
    print(header)
    print(f"  {'-'*18} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*10} {'-'*11} {'-'*11}")

    for sub in SUB_LEVELS:
        p = perfs[sub]
        label = strat_labels[sub]
        vs60 = p["final"] - p_60["final"]
        vsivv = p["final"] - p_ivv["final"]
        ml = f"{pct_levered:.0%}" if sub > 0 else "0%"
        marker = "  <-" if abs(sub - PRINCIPLED_TARGET) < 0.001 else ""
        print(f"  {label:<18} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f} {vs60:>+9.2f} {vsivv:>+10.2f} {ml:>11}{marker}")

    # IVV B&H and 60/40
    print(f"  {'IVV B&H':<18} {p_ivv['ar']:>7.1%} {p_ivv['av']:>6.1%} {p_ivv['sh']:>8.3f} {p_ivv['sortino']:>8.3f} "
          f"{p_ivv['dd']:>7.1%} {p_ivv['calmar']:>8.2f} ${p_ivv['final']:>12.2f} "
          f"{p_ivv['final']-p_60['final']:>+9.2f} {'baseline':>11} {'—':>11}")
    print(f"  {'60/40':<18} {p_60['ar']:>7.1%} {p_60['av']:>6.1%} {p_60['sh']:>8.3f} {p_60['sortino']:>8.3f} "
          f"{p_60['dd']:>7.1%} {p_60['calmar']:>8.2f} ${p_60['final']:>12.2f} "
          f"{'baseline':>10} {p_60['final']-p_ivv['final']:>+10.2f} {'—':>11}")

    # ── Crisis analysis ──────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*140}")

    all_series = [(sub, strat_labels[sub], ret_series[sub]) for sub in SUB_LEVELS]
    all_series += [(-1, "IVV B&H", ivv_s), (-2, "60/40", b60_s)]

    for cname, cs, ce in [("GFC (2008-2009)", "2008-09-01", "2009-03-31"),
                           ("COVID (Feb-Mar 2020)", "2020-02-19", "2020-03-23"),
                           ("2022 Bear", "2022-01-03", "2022-10-31")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<18} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*18} {'-'*10} {'-'*10}")
        for _, label, sr in all_series:
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {label:<18} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Tradeoff curve ───────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  TRADEOFF CURVE")
    print(f"{'='*140}")

    p_1x = perfs[0.0]

    # Compute ETF drag
    # Drag = (theoretical no-cost leveraged return) - (actual leveraged return)
    # Annualized
    drags = {}
    for sub in SUB_LEVELS:
        if sub == 0:
            drags[sub] = 0.0
        else:
            actual_ann = perfs[sub]["ar"]
            theo_s = ret_theo[sub]
            theo_ann = theo_s.mean() * 252
            drags[sub] = theo_ann - actual_ann

    print(f"\n  {'SUB%':>6} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Terminal':>10} "
          f"{'Sharpe Cost':>12} {'Return Gain':>12} {'ETF Drag':>12}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

    for sub in SUB_LEVELS:
        p = perfs[sub]
        sh_cost = p["sh"] - p_1x["sh"]
        ret_gain = p["ar"] - p_1x["ar"]
        drag = drags[sub]
        marker = "  <- principled target" if abs(sub - PRINCIPLED_TARGET) < 0.001 else ""
        print(f"  {sub:>5.0%} {p['ar']:>7.1%} {p['sh']:>8.3f} {p['dd']:>7.1%} ${p['final']:>9.2f} "
              f"{sh_cost:>+11.3f} {ret_gain:>+11.1%} {'~' + f'{drag:.2%}':>11}{marker}")

    # ── Principled target confirmation ───────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  PRINCIPLED TARGET CONFIRMATION")
    print(f"{'='*140}")

    p40 = perfs[PRINCIPLED_TARGET]
    eff_eq = EQ_WEIGHT * (1 + PRINCIPLED_TARGET)
    sh_cost = p40["sh"] - p_1x["sh"]
    ret_gain = p40["ar"] - p_1x["ar"]

    # Position on Sharpe curve: is 40% above, at, or below the Sharpe peak?
    sharpes = [(sub, perfs[sub]["sh"]) for sub in SUB_LEVELS]
    peak_sub, peak_sh = max(sharpes, key=lambda x: x[1])
    if abs(PRINCIPLED_TARGET - peak_sub) < 0.001:
        position = "AT peak"
    elif PRINCIPLED_TARGET < peak_sub:
        position = "below peak (conservative side)"
    else:
        position = "above peak (more aggressive than optimal)"

    print(f"\n  Principled target (40% sub, ~{eff_eq:.0%} effective equity):")
    print(f"    Return:              {p40['ar']:.1%}")
    print(f"    Sharpe:              {p40['sh']:.3f}")
    print(f"    MaxDD:               {p40['dd']:.1%}")
    print(f"    Terminal:            ${p40['final']:.2f} (vs IVV B&H ${p_ivv['final']:.2f}, vs 60/40 ${p_60['final']:.2f})")
    print(f"    Sharpe cost vs 1x:   {sh_cost:+.3f}")
    print(f"    Return gain vs 1x:   {ret_gain:+.1%}")
    print(f"    Position on curve:   {position} (peak at {peak_sub:.0%} sub)")

    # ── Calendar year returns ────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*140}")

    show_subs = [0.0, 0.25, 0.40, 0.50, 0.65]
    header = f"  {'Year':>6}"
    for sub in show_subs:
        header += f" {strat_labels[sub]:>15}"
    header += f" {'IVV B&H':>10}"
    print(header)

    for yr in range(2002, 2027):
        row = f"  {yr:>6}"
        for sub in show_subs:
            sr = ret_series[sub]
            y = sr[sr.index.year == yr]
            if len(y) > 20:
                row += f" {(1+y).prod()-1:>+14.1%}"
            else:
                row += f" {'--':>15}"
        y_ivv = ivv_s[ivv_s.index.year == yr]
        if len(y_ivv) > 20:
            row += f" {(1+y_ivv).prod()-1:>+9.1%}"
        else:
            row += f" {'--':>10}"
        print(row)

    print()
    return perfs, p_ivv, p_60


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 110)
    print("  FABER-SWEEP: FLAT SUBSTITUTION TRADEOFF CHARACTERIZATION")
    print("=" * 110)
    print(f"\n  Rule: when BOTH IVV and QQQ at 3/3 Faber → replace SUB% with SSO/QLD. Otherwise 1x.")
    print(f"  No tiers, no medians, no weekly checks. Monthly rebalance only.")
    print(f"  SSO expense: {SSO_EXPENSE}, QLD expense: {QLD_EXPENSE}")
    print(f"  Principled target: {PRINCIPLED_TARGET:.0%} sub → ~{EQ_WEIGHT*(1+PRINCIPLED_TARGET):.0%} effective equity")

    print(f"\n  Loading data...")
    daily_ret, trend_df, rfr_daily = load_data()

    ret_series, ret_theo, ivv_s, b60_s, strat_labels, pct_levered = \
        run_backtest(daily_ret, trend_df, rfr_daily)

    perfs, p_ivv, p_60 = print_report(
        ret_series, ret_theo, ivv_s, b60_s, strat_labels, pct_levered)
