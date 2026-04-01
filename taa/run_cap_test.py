"""Test: removal of 40% single-asset concentration cap.

Runs two identical backtests — one with the production 40% cap,
one with cap effectively removed (set to 1.0) — and compares.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd
from collections import defaultdict

from taa.data import (load_daily_etf_returns, load_monthly_prices,
                       load_realized_vols, load_monthly_macro, load_monthly_asset_returns)
from taa.faber import compute_trend_scores, apply_faber_filter
from taa.harvey import compute_zscore_variables, find_similar_months, compute_expected_returns
from taa.allocation import BASELINE, ASSETS, EQUITY, normalize, MAX_SINGLE

TIER_SUBS = {0: 0.0, 1: 0.30, 2: 0.65}
SEED = {"IVV": 0.0028, "QQQ": 0.0080}


def direct_capital_with_cap(weights, pool, expected_returns, realized_vols, single_cap):
    """Harvey inverse-vol capital direction with configurable single-eligible cap."""
    detail = {"pool": pool, "scores": {}, "directed_to": {}}
    if pool <= 0.001:
        return weights, detail

    candidates = {}
    for a in ASSETS:
        if a == "cash" or weights.get(a, 0) <= 0:
            continue
        er = expected_returns.get(a, 0)
        vol = realized_vols.get(a, 0.15)
        if er > 0 and vol > 0.01:
            score = er / vol
            candidates[a] = score
            detail["scores"][a] = {"er": er, "vol": vol, "score": score}

    if not candidates:
        weights["cash"] = weights.get("cash", 0) + pool
        return weights, detail

    if len(candidates) == 1:
        a = list(candidates.keys())[0]
        alloc = min(pool, single_cap - weights.get(a, 0))
        alloc = max(alloc, 0)
        weights[a] = weights.get(a, 0) + alloc
        weights["cash"] = weights.get("cash", 0) + (pool - alloc)
        return weights, detail

    total_score = sum(candidates.values())
    for a, score in candidates.items():
        share = pool * (score / total_score)
        weights[a] = weights.get(a, 0) + share

    return weights, detail


def normalize_with_cap(weights, single_eligible_cap):
    """Normalize with configurable single-eligible-asset cap."""
    w = dict(weights)

    risky = [a for a in w if a != "cash" and w.get(a, 0) > 0.01]
    if len(risky) == 1 and w[risky[0]] > single_eligible_cap:
        excess = w[risky[0]] - single_eligible_cap
        w[risky[0]] = single_eligible_cap
        w["cash"] = w.get("cash", 0) + excess

    for _ in range(10):
        total = sum(max(v, 0) for v in w.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            w = {a: max(v, 0) / total for a, v in w.items()}

        changed = False
        for a in w:
            if w[a] > MAX_SINGLE:
                w["cash"] = w.get("cash", 0) + w[a] - MAX_SINGLE
                w[a] = MAX_SINGLE
                changed = True

        eq = sum(w.get(a, 0) for a in EQUITY)
        if eq > 0.85:
            r = 0.85 / eq
            for a in EQUITY:
                freed = w[a] - w[a] * r
                w[a] *= r
                w["cash"] = w.get("cash", 0) + freed
            changed = True

        if w.get("cash", 0) < 0.03:
            deficit = 0.03 - w["cash"]
            others = [a for a in w if a != "cash" and w[a] > 0]
            tot = sum(w[a] for a in others)
            if tot > deficit:
                for a in others:
                    w[a] -= deficit * (w[a] / tot)
            w["cash"] = 0.03
            changed = True

        if not changed:
            break

    total = sum(max(v, 0) for v in w.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        w = {a: max(v, 0) / total for a, v in w.items()}
    return w


def run_backtest(single_cap, daily_ret, monthly_prices, rvol_monthly,
                 asset_ret_fwd, rfr_daily, dpdf, z_clean, trend_df,
                 sma_6, sma_10, sma_12, pre_ers, trading_days):
    """Full backtest with configurable single-asset cap."""

    tier = 0; delevered = False
    hist = {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])}
    med = dict(SEED); events = []
    ct = {a: 3 for a in ASSETS if a != "cash"}
    he = {a: 0.0 for a in ASSETS}
    wp = dict(BASELINE)

    strat_returns = {}
    alloc_log = []  # track allocations for concentration analysis

    def count_sma_breaches(etf, day):
        if etf not in dpdf.columns: return 0
        prices_up_to = dpdf.loc[:day, etf]
        if len(prices_up_to) == 0: return 0
        price = prices_up_to.iloc[-1]
        ms = pd.Timestamp(f"{day.year}-{day.month:02d}-01")
        breaches = 0
        for sma_df in [sma_6, sma_10, sma_12]:
            sma_dates = sma_df.index[sma_df.index <= ms]
            if len(sma_dates) == 0: continue
            val = sma_df.loc[sma_dates[-1], etf]
            if pd.notna(val) and price < val:
                breaches += 1
        return breaches

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day)-1].month)
        is_friday = day.weekday() == 4

        if is_ms:
            ts_c = trend_df.index[trend_df.index <= day]
            if len(ts_c) > 0:
                for a in ASSETS:
                    if a == "cash": continue
                    if a in trend_df.columns:
                        v = trend_df.loc[ts_c[-1], a]
                        ct[a] = int(v) if pd.notna(v) else 0

            z_c = z_clean.index[z_clean.index < day]
            if len(z_c) > 0:
                try:
                    sim, _ = find_similar_months(z_clean, z_c[-1])
                    he = compute_expected_returns(sim, asset_ret_fwd, avail)
                except: pass

            rv_c = rvol_monthly.index[rvol_monthly.index <= day]; rvols = {}
            if len(rv_c) > 0:
                for a in ASSETS:
                    if a == "cash": continue
                    if a in rvol_monthly.columns:
                        v = rvol_monthly.loc[rv_c[-1], a]
                        rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

            w1, pool = apply_faber_filter(ct, BASELINE)
            w2, _ = direct_capital_with_cap(dict(w1), pool, he, rvols, single_cap)
            wp = normalize_with_cap(w2, single_cap)

            # Log allocation
            alloc_log.append({"date": day, **wp})

            f_c = ct.get("IVV", 0) >= 3 and ct.get("QQQ", 0) >= 3
            h_c = he.get("IVV", 0) > 0 and he.get("QQQ", 0) > 0
            conv = f_c and h_c

            delevered = False
            if conv:
                above = all(he.get(a, 0) > med.get(a, 0) for a in ["IVV", "QQQ"])
                tier = 2 if above else 1
                for a in ["IVV", "QQQ"]:
                    hist[a].append(he.get(a, 0))
                    med[a] = float(np.median(hist[a]))
            else:
                tier = 0

        if is_friday:
            ivv_b = count_sma_breaches("IVV", day)
            qqq_b = count_sma_breaches("QQQ", day)
            if tier > 0 and not delevered:
                if ivv_b >= 3 or qqq_b >= 3:
                    events.append({"date": day, "tier": tier})
                    tier = 0; delevered = True

        iw = wp.get("IVV", 0); qw = wp.get("QQQ", 0)
        ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
        sr = 2*ir - rfr - 0.0091/252
        ql = 2*qr - rfr - 0.0089/252
        base = sum(wp.get(a, 0)*actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])

        sub = TIER_SUBS.get(tier, 0)
        if sub > 0:
            strat_returns[day] = (iw*(1-sub)*ir + iw*sub*sr +
                                   qw*(1-sub)*qr + qw*sub*ql + base)
        else:
            strat_returns[day] = iw*ir + qw*qr + base

    return pd.Series(strat_returns).sort_index(), events, pd.DataFrame(alloc_log)


def metrics(s):
    ar = s.mean()*252; av = s.std()*np.sqrt(252)
    sh = ar/av if av > 0 else 0
    neg = s[s < 0]; ds = neg.std()*np.sqrt(252) if len(neg) > 10 else av
    so = ar/ds if ds > 0 else 0
    cum = (1+s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    cal = ar/abs(dd) if dd != 0 else 0
    final = cum.iloc[-1]
    return {"ann_ret": ar, "ann_vol": av, "sharpe": sh, "sortino": so,
            "max_dd": dd, "calmar": cal, "terminal": final}


def crisis_dd(s, start, end):
    c = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    if len(c) < 5:
        return float("nan"), float("nan")
    total = (1+c).prod() - 1
    cum = (1+c).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    return total, dd


# ── Load data ────────────────────────────────────────────────────
print("Loading data...")
daily_ret = load_daily_etf_returns()
daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
monthly_prices = load_monthly_prices()
rvol_monthly = load_realized_vols()
asset_ret = load_monthly_asset_returns()
asset_ret_fwd = asset_ret.shift(-1)
trend_df = compute_trend_scores(monthly_prices)
z_data = compute_zscore_variables(load_monthly_macro())
z_clean = z_data[[c for c in z_data.columns if c.endswith("_z")]].dropna()

from fredapi import Fred
rfr_daily = pd.Series(0.0, index=daily_ret.index)
key = os.environ.get("FRED_API_KEY")
if key:
    tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
    tb.index = pd.to_datetime(tb.index)
    rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

import yfinance as yf
daily_prices = {}
for our, ticker in [("IVV", "SPY"), ("QQQ", "QQQ")]:
    d = yf.download(ticker, start="1998-01-01", progress=False)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p, "columns"): p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        daily_prices[our] = p
dpdf = pd.DataFrame(daily_prices)

mc = dpdf.resample("MS").last()
sma_6 = mc.rolling(6, min_periods=6).mean()
sma_10 = mc.rolling(10, min_periods=10).mean()
sma_12 = mc.rolling(12, min_periods=12).mean()

bt_start = pd.Timestamp("2002-01-01")
pre_ers = {"IVV": [], "QQQ": []}
for z_dt in z_clean.index[z_clean.index < bt_start]:
    try:
        sim, _ = find_similar_months(z_clean, z_dt)
        er = compute_expected_returns(sim, asset_ret_fwd, ["IVV", "QQQ"])
        for a in ["IVV", "QQQ"]:
            if not np.isnan(er.get(a, np.nan)): pre_ers[a].append(er[a])
    except: pass

common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
trading_days = daily_ret.loc[common_start:].index

# ── Run both configs ─────────────────────────────────────────────
print("Running baseline (40% cap)...")
s_base, ev_base, alloc_base = run_backtest(
    0.40, daily_ret, monthly_prices, rvol_monthly, asset_ret_fwd,
    rfr_daily, dpdf, z_clean, trend_df, sma_6, sma_10, sma_12,
    pre_ers, trading_days)

print("Running uncapped (1.0)...")
s_uncap, ev_uncap, alloc_uncap = run_backtest(
    1.00, daily_ret, monthly_prices, rvol_monthly, asset_ret_fwd,
    rfr_daily, dpdf, z_clean, trend_df, sma_6, sma_10, sma_12,
    pre_ers, trading_days)

m_base = metrics(s_base)
m_uncap = metrics(s_uncap)

# ── Report ───────────────────────────────────────────────────────
print()
print("=" * 70)
print("  SINGLE-ASSET CAP REMOVAL TEST")
print("=" * 70)
print()
print("Config change: single-eligible-asset cap from 0.40 (40%) to 1.00 (uncapped)")
print(f"  Affects: direct_capital() single-candidate alloc limit")
print(f"  Affects: normalize() single-risky-asset ceiling")
print()

print("PERFORMANCE COMPARISON (2002-2026)")
print(f"  {'Metric':>14} {'Baseline':>12} {'Uncapped':>12} {'Delta':>10}")
print(f"  {'-'*50}")
for label, key, fmt in [
    ("Ann. Return", "ann_ret", "pct"),
    ("Volatility", "ann_vol", "pct"),
    ("Sharpe", "sharpe", "f2"),
    ("Sortino", "sortino", "f2"),
    ("Max Drawdown", "max_dd", "pct"),
    ("Calmar", "calmar", "f2"),
    ("Terminal $1", "terminal", "dollar"),
]:
    vb = m_base[key]; vu = m_uncap[key]
    if fmt == "pct":
        print(f"  {label:>14} {vb:>11.1%} {vu:>11.1%} {vu-vb:>+9.2%}")
    elif fmt == "f2":
        print(f"  {label:>14} {vb:>11.2f} {vu:>11.2f} {vu-vb:>+9.3f}")
    elif fmt == "dollar":
        print(f"  {label:>14} ${vb:>10.2f} ${vu:>10.2f} {vu-vb:>+9.2f}")

# Crisis periods
print()
print("CRISIS PERIODS")
print(f"  {'Period':>14} {'Baseline':>12} {'Uncapped':>12} {'Delta':>10}")
print(f"  {'-'*50}")
for name, cs, ce in [("GFC 2008-09", "2008-09-01", "2009-03-31"),
                      ("COVID 2020", "2020-02-19", "2020-03-23"),
                      ("2022 Bear", "2022-01-03", "2022-10-31")]:
    tb, db = crisis_dd(s_base, cs, ce)
    tu, du = crisis_dd(s_uncap, cs, ce)
    print(f"  {name:>14} {tb:>+11.1%} {tu:>+11.1%} {tu-tb:>+9.2%}")
    print(f"  {'(max DD)':>14} {db:>11.1%} {du:>11.1%} {du-db:>+9.2%}")

# Concentration analysis
print()
print("CONCENTRATION ANALYSIS")

non_cash = [a for a in ASSETS if a != "cash"]
n_months = len(alloc_uncap)

# Months where any asset exceeded 40%
exceeded_40 = 0
max_alloc = 0.0
max_alloc_asset = ""
max_alloc_date = None
asset_exceed_count = defaultdict(int)

for _, row in alloc_uncap.iterrows():
    for a in non_cash:
        w = row.get(a, 0)
        if w > 0.40:
            exceeded_40 += 1
            asset_exceed_count[a] += 1
            break  # count month once
    for a in non_cash:
        w = row.get(a, 0)
        if w > max_alloc:
            max_alloc = w
            max_alloc_asset = a
            max_alloc_date = row["date"]

# Max weight per month
max_weights_uncap = []
max_weights_base = []
for _, row in alloc_uncap.iterrows():
    max_weights_uncap.append(max(row.get(a, 0) for a in non_cash))
for _, row in alloc_base.iterrows():
    max_weights_base.append(max(row.get(a, 0) for a in non_cash))

print(f"  Months where any asset > 40% (uncapped): {exceeded_40} of {n_months} ({exceeded_40/n_months:.0%})")
print(f"  Max single-asset allocation: {max_alloc:.1%} ({max_alloc_asset}, {max_alloc_date.strftime('%Y-%m-%d') if max_alloc_date else 'N/A'})")
print(f"  Mean max-asset weight (uncapped): {np.mean(max_weights_uncap):.1%}")
print(f"  Mean max-asset weight (baseline): {np.mean(max_weights_base):.1%}")

print()
print("  Assets exceeding 40% (uncapped):")
for a, cnt in sorted(asset_exceed_count.items(), key=lambda x: -x[1]):
    print(f"    {a}: {cnt} months ({cnt/n_months:.0%})")

# Allocation distribution shift
print()
print("ALLOCATION DISTRIBUTION SHIFT")
print(f"  {'Asset':>8} {'Base Mean':>10} {'Uncap Mean':>11} {'Delta':>8}")
print(f"  {'-'*40}")
for a in ASSETS:
    mb = alloc_base[a].mean() if a in alloc_base.columns else 0
    mu = alloc_uncap[a].mean() if a in alloc_uncap.columns else 0
    print(f"  {a:>8} {mb:>9.1%} {mu:>10.1%} {mu-mb:>+7.2%}")

# De-lever events
print()
print(f"  De-lever events: baseline={len(ev_base)}, uncapped={len(ev_uncap)}")

# Recommendation
print()
print("RECOMMENDATION")
sharpe_diff = m_uncap["sharpe"] - m_base["sharpe"]
dd_diff = m_uncap["max_dd"] - m_base["max_dd"]

if abs(sharpe_diff) < 0.02 and abs(dd_diff) < 0.02:
    print("  Keep 40% cap — negligible performance difference, cap provides")
    print("  structural protection against extreme concentration.")
elif sharpe_diff > 0.03 and dd_diff >= -0.02:
    print("  Consider removing cap — meaningful Sharpe improvement with")
    print("  acceptable drawdown impact.")
elif sharpe_diff < -0.03:
    print("  Keep 40% cap — uncapped version underperforms.")
else:
    dd_worse = dd_diff < -0.02
    print(f"  Keep 40% cap — Sharpe delta ({sharpe_diff:+.3f}) does not justify")
    print(f"  {'increased drawdown risk' if dd_worse else 'removing structural protection'}.")

print()
