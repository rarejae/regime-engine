"""Leverage tier experiment: 1.5x intermediate tier for mixed Faber signals.

Tests whether adding a 1.5x leverage tier (when one equity ETF is at 3/3
and the other at 2/3) improves on the current binary 2x system.

Production baseline (PROD): 2x when both IVV+QQQ at 3/3, 1x otherwise.
Hybrid data: actual SSO/QLD from inception (Jun 2006), synthetic before.
Daily circuit breaker, daily SMAs (126/200/252), monthly re-entry.
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
SSO_EXP = 0.0089; QLD_EXP = 0.0095
DAILY_SMA_PERIODS = [126, 200, 252]


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

    # Actual leveraged ETF data
    actual_lev = {}
    for ticker in ["SSO", "QLD"]:
        d = yf.download(ticker, start="2006-01-01", progress=False, auto_adjust=True)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            actual_lev[ticker] = p.pct_change().dropna()

    both_start = pd.Timestamp("2099-01-01")
    for t in ["SSO", "QLD"]:
        if t in actual_lev:
            both_start = max(both_start, actual_lev[t].index.min()) if both_start < pd.Timestamp("2098-01-01") else actual_lev[t].index.min()
    # Fix: get the later of the two starts
    if "SSO" in actual_lev and "QLD" in actual_lev:
        both_start = max(actual_lev["SSO"].index.min(), actual_lev["QLD"].index.min())

    return daily_ret, trend_df, rfr_daily, dpdf, daily_smas, actual_lev, both_start


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
        if breaches >= 3: return True
    return False


def classify_state(ivv_score, qqq_score):
    """Classify combined IVV/QQQ signal state."""
    if ivv_score >= 3 and qqq_score >= 3: return "A"  # Full conviction
    if ivv_score >= 3 and qqq_score == 2: return "B"  # IVV strong, QQQ moderate
    if ivv_score == 2 and qqq_score >= 3: return "C"  # QQQ strong, IVV moderate
    if ivv_score == 2 and qqq_score == 2: return "D"  # Both moderate
    return "E"  # No leverage


def get_sub_for_state(state, config):
    """Return substitution percentage for given state and strategy config."""
    return config.get(state, 0.0)


def lev_return_hybrid(iw, qw, ir, qr, rfr, sub, base_ret, day, actual_lev, both_start):
    """Compute leveraged return using hybrid data."""
    if sub <= 0:
        return iw * ir + qw * qr + base_ret

    if day >= both_start and "SSO" in actual_lev and "QLD" in actual_lev:
        sso_r = actual_lev["SSO"].get(day, np.nan)
        qld_r = actual_lev["QLD"].get(day, np.nan)
        if np.isnan(sso_r): sso_r = 2.0 * ir - rfr - SSO_EXP / 252
        if np.isnan(qld_r): qld_r = 2.0 * qr - rfr - QLD_EXP / 252
        sso_r = float(sso_r); qld_r = float(qld_r)
    else:
        sso_r = 2.0 * ir - rfr - SSO_EXP / 252
        qld_r = 2.0 * qr - rfr - QLD_EXP / 252

    return iw * ((1 - sub) * ir + sub * sso_r) + qw * ((1 - sub) * qr + sub * qld_r) + base_ret


# ── Strategy configs: state → substitution pct ───────────────────────────────

STRATEGY_CONFIGS = {
    "PROD":           {"A": 1.00, "B": 0.00, "C": 0.00, "D": 0.00, "E": 0.00},
    "T1-BC-100":      {"A": 1.00, "B": 0.50, "C": 0.50, "D": 0.00, "E": 0.00},
    "T1-BC-100-D25":  {"A": 1.00, "B": 0.50, "C": 0.50, "D": 0.25, "E": 0.00},
    "T1-BC-100-D50":  {"A": 1.00, "B": 0.50, "C": 0.50, "D": 0.50, "E": 0.00},
    "T2-ALL-2x":      {"A": 1.00, "B": 1.00, "C": 1.00, "D": 0.00, "E": 0.00},
    "T2-ALL-2x-D50":  {"A": 1.00, "B": 1.00, "C": 1.00, "D": 0.50, "E": 0.00},
}


def run_backtest(daily_ret, trend_df, rfr_daily, dpdf, daily_smas, actual_lev, both_start):
    print(f"\n{'='*130}")
    print(f"  BACKTEST")
    print(f"{'='*130}")

    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index
    print(f"  {len(trading_days)} trading days ({common_start.date()} to {trading_days.max().date()})")
    print(f"  Hybrid data: actual SSO/QLD from {both_start.date()}, synthetic before")

    results = {s: {} for s in STRATEGY_CONFIGS}
    results["IVV B&H"] = {}

    # Shared state
    cur_trends = {a: 3 for a in ASSETS if a != "cash"}
    w_faber = dict(BASELINE)
    current_state = "E"

    # Per-strategy: leverage_active, delevered
    strat_state = {}
    for s in STRATEGY_CONFIGS:
        strat_state[s] = {"leverage_active": False, "delevered": False, "active_sub": 0.0,
                          "breaker_events": []}

    # State tracking for frequency analysis
    state_log = []
    fwd_returns = []  # (state, fwd_1m_return_IVV)

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
            # Score using prior day's daily SMAs (PIT)
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            cur_trends = sma_scores(sd, dpdf, daily_smas)

            # Faber allocation
            w1, pool = apply_faber_filter(cur_trends, BASELINE)
            w_faber = dict(w1)
            w_faber["cash"] = w_faber.get("cash", 0) + pool

            # Classify state
            ivv_s = cur_trends.get("IVV", 0)
            qqq_s = cur_trends.get("QQQ", 0)
            current_state = classify_state(ivv_s, qqq_s)
            state_log.append({"date": day, "state": current_state,
                              "ivv_score": ivv_s, "qqq_score": qqq_s})

            # Update each strategy
            for s_name, config in STRATEGY_CONFIGS.items():
                ss = strat_state[s_name]
                ss["delevered"] = False
                target_sub = get_sub_for_state(current_state, config)
                ss["active_sub"] = target_sub
                ss["leverage_active"] = target_sub > 0

        # Daily circuit breaker — applies to ALL strategies with leverage
        for s_name in STRATEGY_CONFIGS:
            ss = strat_state[s_name]
            if ss["leverage_active"] and not ss["delevered"]:
                if check_breach(day, dpdf, daily_smas):
                    ss["breaker_events"].append(day)
                    ss["leverage_active"] = False
                    ss["active_sub"] = 0.0
                    ss["delevered"] = True

        # Compute returns
        iw = w_faber.get("IVV", 0)
        qw = w_faber.get("QQQ", 0)
        ir = actual.get("IVV", 0)
        qr = actual.get("QQQ", 0)
        base_ret = sum(w_faber.get(a, 0) * actual.get(a, 0)
                       for a in avail if a not in ["IVV", "QQQ"])

        for s_name in STRATEGY_CONFIGS:
            ss = strat_state[s_name]
            sub = ss["active_sub"]
            results[s_name][day] = lev_return_hybrid(
                iw, qw, ir, qr, rfr, sub, base_ret, day, actual_lev, both_start)

        results["IVV B&H"][day] = actual.get("IVV", 0)

    ret = {s: pd.Series(d).sort_index() for s, d in results.items()}
    state_df = pd.DataFrame(state_log).set_index("date")

    # Forward 1-month return by state
    ivv_monthly = daily_ret["IVV"].resample("MS").apply(lambda x: (1 + x).prod() - 1)
    for _, row in state_df.iterrows():
        dt = row.name
        fwd_dt = dt + pd.DateOffset(months=1)
        closest = ivv_monthly.index[ivv_monthly.index >= fwd_dt]
        if len(closest) > 0:
            fwd_ret = ivv_monthly.get(closest[0], np.nan)
            fwd_returns.append({"state": row["state"], "fwd_ret": fwd_ret})

    return ret, state_df, pd.DataFrame(fwd_returns), strat_state


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


def report(ret, state_df, fwd_df, strat_state):
    strat_order = list(STRATEGY_CONFIGS.keys()) + ["IVV B&H"]
    perfs = {s: metrics(ret[s]) for s in strat_order}
    p_prod = perfs.get("PROD", {})

    # ── State frequency ──────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  STATE FREQUENCY ANALYSIS")
    print(f"{'='*130}")

    total = len(state_df)
    counts = state_df["state"].value_counts().sort_index()
    state_labels = {"A": "3/3 + 3/3 (full conviction)",
                     "B": "3/3 + 2/3 (IVV strong, QQQ mod)",
                     "C": "2/3 + 3/3 (QQQ strong, IVV mod)",
                     "D": "2/3 + 2/3 (both moderate)",
                     "E": "<2/3 either (no leverage)"}

    print(f"\n  State frequency (2002-2026, {total} months):\n")
    for state in ["A", "B", "C", "D", "E"]:
        n = int(counts.get(state, 0))
        pct = n / total * 100
        print(f"    State {state} ({state_labels[state]}):  {n:>4} months ({pct:.0f}%)")

    bc = int(counts.get("B", 0)) + int(counts.get("C", 0))
    prod_lev = int(counts.get("A", 0))
    t1_lev = prod_lev + bc
    print(f"\n    B+C combined:              {bc:>4} months ({bc/total*100:.0f}%) — mixed conviction")
    print(f"    Total leveraged (PROD):    {prod_lev:>4} months ({prod_lev/total*100:.0f}%)")
    print(f"    Total leveraged (T1-BC):   {t1_lev:>4} months ({t1_lev/total*100:.0f}%)")

    # Forward returns by state
    print(f"\n  Average forward 1-month IVV return by state:")
    for state in ["A", "B", "C", "D", "E"]:
        sub = fwd_df[fwd_df["state"] == state]["fwd_ret"].dropna()
        if len(sub) > 0:
            print(f"    State {state}: {sub.mean()*100:>+.2f}% avg  ({len(sub)} obs, "
                  f"positive {(sub > 0).mean()*100:.0f}%)")

    # ── Performance table ────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  FULL PERFORMANCE TABLE")
    print(f"{'='*130}")

    print(f"\n  {'Strategy':<18} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'vs PROD':>10} {'Lev Mo%':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*10} {'-'*8}")

    for s in strat_order:
        p = perfs.get(s)
        if not p: continue
        vs = p["final"] - p_prod.get("final", 0) if s != "IVV B&H" else 0
        vs_str = f"{vs:>+9.2f}" if s != "IVV B&H" else f"{'—':>10}"

        # Compute % of months leveraged
        config = STRATEGY_CONFIGS.get(s, {})
        lev_months = sum(1 for _, row in state_df.iterrows()
                         if config.get(row["state"], 0) > 0)
        lev_pct = f"{lev_months/total*100:.0f}%" if s in STRATEGY_CONFIGS else "—"

        print(f"  {s:<18} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f} {vs_str} {lev_pct:>8}")

    # ── Crisis analysis ──────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*130}")

    for cname, cs, ce in [("GFC (2008-09)", "2008-09-01", "2009-03-31"),
                           ("COVID (Feb-Mar 2020)", "2020-02-19", "2020-03-23"),
                           ("2022 Bear", "2022-01-03", "2022-10-31")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<18} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*18} {'-'*10} {'-'*10}")
        for s in strat_order:
            sr = ret[s]; c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {s:<18} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Age-65 projections ───────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  AGE-65 PROJECTION ($21,000, 40 years)")
    print(f"{'='*130}")

    print(f"\n  {'Strategy':<18} {'Ann Return':>11} {'$21K at 65':>13} {'vs PROD':>12}")
    print(f"  {'-'*18} {'-'*11} {'-'*13} {'-'*12}")

    prod_proj = 21000 * (1 + p_prod.get("ar", 0)) ** 40
    for s in list(STRATEGY_CONFIGS.keys()):
        p = perfs.get(s)
        if not p: continue
        proj = 21000 * (1 + p["ar"]) ** 40
        vs = proj - prod_proj
        print(f"  {s:<18} {p['ar']:>10.2%} ${proj:>12,.0f} ${vs:>+11,.0f}")

    # ── Circuit breaker comparison ───────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  CIRCUIT BREAKER EVENTS BY STRATEGY")
    print(f"{'='*130}")

    for s in STRATEGY_CONFIGS:
        n = len(strat_state[s]["breaker_events"])
        print(f"    {s:<18}: {n} events ({n/24:.1f}/year)")

    # ── Key questions ────────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  KEY QUESTIONS")
    print(f"{'='*130}")

    # Q1: B+C frequency and forward returns
    bc_fwd = fwd_df[fwd_df["state"].isin(["B", "C"])]["fwd_ret"].dropna()
    print(f"\n  Q1. B+C months: {bc} ({bc/total*100:.0f}%), avg forward return: {bc_fwd.mean()*100:+.2f}%")

    # Q2-Q6: Compare each strategy
    for s in ["T1-BC-100", "T1-BC-100-D25", "T1-BC-100-D50", "T2-ALL-2x", "T2-ALL-2x-D50"]:
        p = perfs.get(s)
        if not p: continue
        term_vs = p["final"] - p_prod.get("final", 0)
        sh_vs = p["sh"] - p_prod.get("sh", 0)
        dd_vs = (p["dd"] - p_prod.get("dd", 0)) * 100
        calmar_vs = p["calmar"] - p_prod.get("calmar", 0)
        print(f"\n  {s}:")
        print(f"    Sharpe: {p_prod['sh']:.3f} → {p['sh']:.3f} ({sh_vs:+.3f})")
        print(f"    Terminal: ${p_prod['final']:.2f} → ${p['final']:.2f} (${term_vs:+.2f})")
        print(f"    MaxDD: {p_prod['dd']:.1%} → {p['dd']:.1%} ({dd_vs:+.1f}%)")
        print(f"    Calmar: {p_prod['calmar']:.2f} → {p['calmar']:.2f} ({calmar_vs:+.2f})")

    # Final verdict
    best = max([(s, perfs[s]) for s in STRATEGY_CONFIGS if perfs.get(s)],
               key=lambda x: x[1]["final"])
    print(f"\n  Q6. Best terminal wealth: {best[0]} at ${best[1]['final']:.2f}")
    if best[0] != "PROD":
        term_gain = best[1]["final"] - p_prod["final"]
        sh_cost = best[1]["sh"] - p_prod["sh"]
        print(f"      vs PROD: +${term_gain:.2f} terminal, {sh_cost:+.3f} Sharpe")
        if term_gain > 0 and sh_cost >= 0:
            print(f"      → ADOPT: improves both Sharpe and terminal")
        elif term_gain > 0 and sh_cost > -0.01:
            print(f"      → CONSIDER: improves terminal, negligible Sharpe cost")
        elif term_gain > 0:
            print(f"      → TRADEOFF: more terminal but lower Sharpe")
        else:
            print(f"      → REJECT: no improvement over PROD")
    else:
        print(f"      PROD is already the best — no tier improves on binary 2x")

    print()
    return perfs


if __name__ == "__main__":
    print("=" * 130)
    print("  LEVERAGE TIERS: 1.5x INTERMEDIATE FOR MIXED FABER SIGNALS")
    print("=" * 130)

    print(f"\n  Loading data...")
    daily_ret, trend_df, rfr_daily, dpdf, daily_smas, actual_lev, both_start = load_data()
    print(f"  Actual SSO/QLD from: {both_start.date()}")

    ret, state_df, fwd_df, strat_state = run_backtest(
        daily_ret, trend_df, rfr_daily, dpdf, daily_smas, actual_lev, both_start)

    perfs = report(ret, state_df, fwd_df, strat_state)
