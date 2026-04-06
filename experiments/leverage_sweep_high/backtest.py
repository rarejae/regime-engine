"""Comprehensive leverage sweep on Faber-Sweep-40-Daily-Daily.

2x ETF sweep: 40% to 100% substitution (SSO/QLD)
3x ETF strategies: 25%, 40%, 50% substitution (SPXL/TQQQ)
All with daily circuit breaker, daily SMAs (126/200/252), monthly rebalance + monthly re-entry.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import load_daily_etf_returns, load_monthly_prices
from taa.faber import compute_trend_scores, apply_faber_filter

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE.keys())
DAILY_SMA_PERIODS = [126, 200, 252]

# Expense ratios
SSO_EXP = 0.0089; QLD_EXP = 0.0095  # 2x
SPXL_EXP = 0.0091; TQQQ_EXP = 0.0086  # 3x


def load_data():
    import yfinance as yf
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

    ticker_map = {"IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT", "IAU": "GLD", "DBC": "DBC"}
    dp = {}
    for our, ticker in ticker_map.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            dp[our] = p
    dpdf = pd.DataFrame(dp).sort_index()

    daily_smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in DAILY_SMA_PERIODS}
    return daily_ret, trend_df, rfr_daily, dpdf, daily_smas


def sma_scores(day, dpdf, smas):
    scores = {}
    for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
        if a not in dpdf.columns: scores[a] = 0; continue
        p = dpdf.loc[:day, a]
        if len(p) == 0 or pd.isna(p.iloc[-1]): scores[a] = 0; continue
        price = p.iloc[-1]; sc = 0
        for per in DAILY_SMA_PERIODS:
            s = smas[per].loc[:day, a]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price > s.iloc[-1]: sc += 1
        scores[a] = sc
    return scores


def check_breach(day, dpdf, smas):
    for etf in ["IVV", "QQQ"]:
        if etf not in dpdf.columns: continue
        p = dpdf.loc[:day, etf]
        if len(p) == 0: continue
        price = p.iloc[-1]; breaches = 0
        for per in DAILY_SMA_PERIODS:
            s = smas[per].loc[:day, etf]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price < s.iloc[-1]: breaches += 1
        if breaches >= 3: return True, f"{etf} below all 3 SMAs ({price:.1f})"
    return False, ""


def lev_return_generic(iw, qw, ir, qr, rfr, sub, multiplier, ivv_exp, qqq_exp, base_ret):
    """Compute leveraged return for any multiplier (2x or 3x)."""
    if sub <= 0:
        return iw * ir + qw * qr + base_ret
    lev_ir = multiplier * ir - (multiplier - 1) * rfr - ivv_exp / 252
    lev_qr = multiplier * qr - (multiplier - 1) * rfr - qqq_exp / 252
    return iw * ((1 - sub) * ir + sub * lev_ir) + qw * ((1 - sub) * qr + sub * lev_qr) + base_ret


def theo_return(iw, qw, ir, qr, sub, multiplier, base_ret):
    """Theoretical return without drag (for ETF drag estimation)."""
    if sub <= 0:
        return iw * ir + qw * qr + base_ret
    lev_ir = multiplier * ir
    lev_qr = multiplier * qr
    return iw * ((1 - sub) * ir + sub * lev_ir) + qw * ((1 - sub) * qr + sub * lev_qr) + base_ret


# ── Strategy definitions ─────────────────────────────────────────────────────

STRATEGIES = [
    # (name, sub%, multiplier, ivv_exp, qqq_exp)
    ("S40-40%",   0.40, 2, SSO_EXP, QLD_EXP),
    ("S40-50%",   0.50, 2, SSO_EXP, QLD_EXP),
    ("S40-60%",   0.60, 2, SSO_EXP, QLD_EXP),
    ("S40-70%",   0.70, 2, SSO_EXP, QLD_EXP),
    ("S40-80%",   0.80, 2, SSO_EXP, QLD_EXP),
    ("S40-100%",  1.00, 2, SSO_EXP, QLD_EXP),
    ("S40-3x-25%", 0.25, 3, SPXL_EXP, TQQQ_EXP),
    ("S40-3x-40%", 0.40, 3, SPXL_EXP, TQQQ_EXP),
    ("S40-3x-50%", 0.50, 3, SPXL_EXP, TQQQ_EXP),
]


def run_backtest(daily_ret, trend_df, rfr_daily, dpdf, smas):
    print(f"\n{'='*130}")
    print(f"  BACKTEST")
    print(f"{'='*130}")

    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index
    print(f"  {len(trading_days)} trading days ({common_start.date()} to {trading_days.max().date()})")

    # State — shared Faber logic and circuit breaker (identical for all strategies)
    cur_trends = {a: 3 for a in ASSETS if a != "cash"}
    w_faber = dict(BASELINE)
    faber_conv = False
    leverage_active = False
    delevered_this_month = False
    breaker_events = []

    results = {s[0]: {} for s in STRATEGIES}
    results_theo = {s[0]: {} for s in STRATEGIES}
    results["IVV B&H"] = {}

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            delevered_this_month = False
            prior_days = trading_days[trading_days < day]
            score_day = prior_days[-1] if len(prior_days) > 0 else day
            cur_trends = sma_scores(score_day, dpdf, smas)

            w1, pool = apply_faber_filter(cur_trends, BASELINE)
            w_faber = dict(w1)
            w_faber["cash"] = w_faber.get("cash", 0) + pool
            faber_conv = (cur_trends.get("IVV", 0) >= 3 and cur_trends.get("QQQ", 0) >= 3)
            leverage_active = faber_conv

        # Daily circuit breaker
        if leverage_active and not delevered_this_month:
            breach, detail = check_breach(day, dpdf, smas)
            if breach:
                breaker_events.append({"date": day, "detail": detail})
                leverage_active = False
                delevered_this_month = True

        iw = w_faber.get("IVV", 0)
        qw = w_faber.get("QQQ", 0)
        ir = actual.get("IVV", 0)
        qr = actual.get("QQQ", 0)
        base_ret = sum(w_faber.get(a, 0) * actual.get(a, 0)
                       for a in avail if a not in ["IVV", "QQQ"])

        for name, sub_pct, mult, ivv_exp, qqq_exp in STRATEGIES:
            active_sub = sub_pct if leverage_active else 0.0
            results[name][day] = lev_return_generic(
                iw, qw, ir, qr, rfr, active_sub, mult, ivv_exp, qqq_exp, base_ret)
            results_theo[name][day] = theo_return(
                iw, qw, ir, qr, active_sub, mult, base_ret)

        results["IVV B&H"][day] = actual.get("IVV", 0)

    ret = {s: pd.Series(d).sort_index() for s, d in results.items()}
    ret_theo = {s: pd.Series(d).sort_index() for s, d in results_theo.items()}
    return ret, ret_theo, breaker_events


def metrics(s):
    s = s.dropna()
    if len(s) < 50: return None
    ar = s.mean() * 252; av = s.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]; ds = neg.std() * np.sqrt(252) if len(neg) > 10 else av
    so = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    cal = ar / abs(dd) if dd != 0 else 0
    return {"ar": ar, "av": av, "sh": sh, "sortino": so, "dd": dd, "calmar": cal, "final": cum.iloc[-1]}


def report(ret, ret_theo, breaker_events):
    all_names = [s[0] for s in STRATEGIES] + ["IVV B&H"]
    perfs = {n: metrics(ret[n]) for n in all_names}
    p_base = perfs.get("S40-40%", {})

    # Compute ETF drag
    drags = {}
    for name, sub_pct, mult, ivv_exp, qqq_exp in STRATEGIES:
        actual_ar = perfs[name]["ar"] if perfs[name] else 0
        theo_ar = ret_theo[name].mean() * 252 if name in ret_theo else 0
        drags[name] = theo_ar - actual_ar

    # ── Performance table ────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  FULL PERFORMANCE TABLE")
    print(f"{'='*140}")
    print(f"  {'Strategy':<16} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'vs 40%':>10} {'ETF Drag':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*10} {'-'*10}")

    for n in [s[0] for s in STRATEGIES] + ["IVV B&H"]:
        p = perfs.get(n)
        if not p: continue
        vs40 = p["final"] - p_base.get("final", 0) if n != "IVV B&H" else 0
        vs_str = f"{vs40:>+9.2f}" if n != "IVV B&H" else f"{'—':>10}"
        drag = drags.get(n, 0)
        drag_str = f"~{drag:.2%}" if n != "IVV B&H" else f"{'—':>10}"
        print(f"  {n:<16} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f} {vs_str} {drag_str:>10}")

    # ── Crisis analysis ──────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*140}")
    for cname, cs, ce in [("GFC (2008-09)", "2008-09-01", "2009-03-31"),
                           ("COVID (Feb-Mar 2020)", "2020-02-19", "2020-03-23"),
                           ("2022 Bear", "2022-01-03", "2022-10-31")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<16} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*16} {'-'*10} {'-'*10}")
        for n in [s[0] for s in STRATEGIES] + ["IVV B&H"]:
            sr = ret[n]; c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {n:<16} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Tradeoff summary ─────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  LEVERAGE TRADEOFF SUMMARY")
    print(f"{'='*140}")
    print(f"\n  {'SUB%':>8} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Terminal':>10} "
          f"{'Sharpe vs40':>12} {'Term vs40':>12} {'ETF Drag':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for name, sub_pct, mult, _, _ in STRATEGIES:
        p = perfs.get(name)
        if not p: continue
        sh_d = p["sh"] - p_base.get("sh", 0)
        t_d = p["final"] - p_base.get("final", 0)
        drag = drags.get(name, 0)
        label = f"{int(sub_pct*100)}% {mult}x"
        print(f"  {label:>8} {p['ar']:>7.1%} {p['sh']:>8.3f} {p['dd']:>7.1%} ${p['final']:>9.2f} "
              f"{sh_d:>+11.3f} ${t_d:>+11.2f} ~{drag:>.2%}")

    # ── Age-65 projection ────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  AGE-65 PROJECTION ($21,000 starting balance, 40-year horizon)")
    print(f"{'='*140}")

    starting = 21000
    horizon = 40

    print(f"\n  {'Strategy':<16} {'Ann Return':>11} {'$21K at 65':>13} {'vs S40-40%':>12}")
    print(f"  {'-'*16} {'-'*11} {'-'*13} {'-'*12}")

    base_proj = starting * (1 + p_base["ar"]) ** horizon if p_base else 0
    for name, sub_pct, mult, _, _ in STRATEGIES:
        p = perfs.get(name)
        if not p: continue
        proj = starting * (1 + p["ar"]) ** horizon
        vs = proj - base_proj
        print(f"  {name:<16} {p['ar']:>10.1%} ${proj:>12,.0f} ${vs:>+11,.0f}")

    p_ivv = perfs.get("IVV B&H")
    if p_ivv:
        proj = starting * (1 + p_ivv["ar"]) ** horizon
        print(f"  {'IVV B&H':<16} {p_ivv['ar']:>10.1%} ${proj:>12,.0f}")

    # ── Circuit breaker at high leverage ─────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  CIRCUIT BREAKER BEHAVIOR AT HIGH LEVERAGE")
    print(f"{'='*140}")

    print(f"\n  Circuit breaker events: {len(breaker_events)} over 24 years ({len(breaker_events)/24:.1f}/year)")
    print(f"  (Same events for all strategies — breaker is independent of leverage level)")

    # COVID max DD comparison
    print(f"\n  COVID MAX DD BY LEVERAGE LEVEL:")
    print(f"  {'Strategy':<16} {'COVID MaxDD':>12}")
    for name, _, _, _, _ in STRATEGIES:
        sr = ret[name]
        c = sr[(sr.index >= "2020-02-19") & (sr.index <= "2020-03-23")]
        if len(c) > 0:
            cum = (1 + c).cumprod()
            mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
            print(f"  {name:<16} {mdd:>11.1%}")

    # 2022 DD comparison
    print(f"\n  2022 MAX DD BY LEVERAGE LEVEL:")
    print(f"  {'Strategy':<16} {'2022 MaxDD':>12}")
    for name, _, _, _, _ in STRATEGIES:
        sr = ret[name]
        c = sr[(sr.index >= "2022-01-03") & (sr.index <= "2022-10-31")]
        if len(c) > 0:
            cum = (1 + c).cumprod()
            mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
            print(f"  {name:<16} {mdd:>11.1%}")

    # ── Key questions ────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  KEY QUESTIONS")
    print(f"{'='*140}")

    # Q1: Does daily breaker allow higher leverage to beat monthly sweep?
    # Q2: Terminal wealth peak for 2x
    two_x = [(n, p["final"]) for n, sub, mult, _, _ in STRATEGIES
             for p in [perfs.get(n)] if p and mult == 2]
    if two_x:
        peak_2x = max(two_x, key=lambda x: x[1])
        print(f"\n  Q2. 2x terminal wealth peak: {peak_2x[0]} at ${peak_2x[1]:.2f}")

    # Q3: 3x vs maxed 2x
    three_x = [(n, p["final"]) for n, sub, mult, _, _ in STRATEGIES
               for p in [perfs.get(n)] if p and mult == 3]
    p_100 = perfs.get("S40-100%")
    if three_x and p_100:
        best_3x = max(three_x, key=lambda x: x[1])
        print(f"  Q3. Best 3x: {best_3x[0]} at ${best_3x[1]:.2f} vs 2x-100% at ${p_100['final']:.2f}")
        if best_3x[1] > p_100["final"]:
            print(f"      → 3x BEATS maxed 2x by ${best_3x[1]-p_100['final']:.2f}")
        else:
            print(f"      → 3x LOSES to maxed 2x — volatility decay overwhelms")

    # Q4: Max sustainable leverage
    for name, sub, mult, _, _ in STRATEGIES:
        p = perfs.get(name)
        if p and p["final"] > p_base.get("final", 0):
            continue
        else:
            # First strategy that underperforms baseline terminal
            if p:
                print(f"  Q4. Terminal wealth drops below 40% baseline at: {name} (${p['final']:.2f} vs ${p_base['final']:.2f})")
            break

    # Q5: Recommendation for 25-year-old
    # Max DD of -30% on $21K = $6,300 loss
    dd_limit = -0.30
    candidates = [(n, perfs[n]) for n, _, _, _, _ in STRATEGIES
                  if perfs.get(n) and perfs[n]["dd"] >= dd_limit]
    if candidates:
        best = max(candidates, key=lambda x: x[1]["final"])
        print(f"\n  Q5. For 25-year-old (max DD <= {dd_limit:.0%}, $21K portfolio):")
        print(f"      Best: {best[0]} — {best[1]['ar']:.1%} return, {best[1]['dd']:.1%} DD, "
              f"${starting * (1 + best[1]['ar'])**horizon:,.0f} at age 65")
        # Also show best without DD constraint
        overall_best = max([(n, perfs[n]) for n, _, _, _, _ in STRATEGIES if perfs.get(n)],
                           key=lambda x: x[1]["final"])
        if overall_best[0] != best[0]:
            print(f"      Unconstrained best: {overall_best[0]} — {overall_best[1]['ar']:.1%} return, "
                  f"{overall_best[1]['dd']:.1%} DD, ${starting * (1 + overall_best[1]['ar'])**horizon:,.0f} at 65")

    print()
    return perfs


if __name__ == "__main__":
    print("=" * 130)
    print("  COMPREHENSIVE LEVERAGE SWEEP: Faber-Sweep-40-Daily-Daily")
    print("=" * 130)
    print(f"  2x ETFs: SSO ({SSO_EXP}), QLD ({QLD_EXP})")
    print(f"  3x ETFs: SPXL ({SPXL_EXP}), TQQQ ({TQQQ_EXP})")
    print(f"  Daily SMAs: {DAILY_SMA_PERIODS}, daily circuit breaker, monthly rebalance + re-entry")

    print(f"\n  Loading data...")
    daily_ret, trend_df, rfr_daily, dpdf, smas = load_data()

    ret, ret_theo, breaker_events = run_backtest(daily_ret, trend_df, rfr_daily, dpdf, smas)
    perfs = report(ret, ret_theo, breaker_events)
