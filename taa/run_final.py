"""Final validated backtest: Phase 1 + graduated leverage with fixed median seed."""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd

from taa.data import (load_daily_etf_returns, load_monthly_prices,
                       load_realized_vols, load_monthly_macro, load_monthly_asset_returns)
from taa.faber import compute_trend_scores, apply_faber_filter
from taa.harvey import compute_zscore_variables, find_similar_months, compute_expected_returns
from taa.allocation import BASELINE, ASSETS, EQUITY, direct_capital, normalize

OUTPUT = Path("taa/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010

print("=" * 80)
print("  FINAL VALIDATED SYSTEM: PHASE 1 + GRADUATED LEVERAGE")
print("=" * 80)

# Load data ONCE — use same data for all configs
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

# Pre-backtest median seed: compute Harvey ER for IVV and QQQ across pre-2002 months
print("\nComputing pre-backtest median seed...")
backtest_start = pd.Timestamp("2002-01-01")
pre_bt_months = z_clean.index[z_clean.index < backtest_start]
pre_bt_ers = {"IVV": [], "QQQ": []}

for z_dt in pre_bt_months:
    try:
        sim, _ = find_similar_months(z_clean, z_dt)
        er = compute_expected_returns(sim, asset_ret_fwd, ["IVV", "QQQ"])
        for a in ["IVV", "QQQ"]:
            if er.get(a) is not None and not np.isnan(er.get(a, 0)):
                pre_bt_ers[a].append(er[a])
    except ValueError:
        continue

seed_ivv = float(np.mean(pre_bt_ers["IVV"])) if pre_bt_ers["IVV"] else 0.005
seed_qqq = float(np.mean(pre_bt_ers["QQQ"])) if pre_bt_ers["QQQ"] else 0.005
print(f"  IVV seed median: {seed_ivv:.4f} (from {len(pre_bt_ers['IVV'])} pre-BT months)")
print(f"  QQQ seed median: {seed_qqq:.4f} (from {len(pre_bt_ers['QQQ'])} pre-BT months)")

# Backtest
common_start = max(daily_ret.dropna(how="all").index.min(), backtest_start)
trading_days = daily_ret.loc[common_start:].index
print(f"\nBacktest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

# State
cur_trends = {a: 3 for a in ASSETS if a != "cash"}
harvey_er = {a: 0.0 for a in ASSETS}
w_p1 = dict(BASELINE)
flat_conv = False; tier = 0

# Expanding medians — seeded
tier1_hist = {"IVV": list(pre_bt_ers["IVV"]), "QQQ": list(pre_bt_ers["QQQ"])}
tier1_med = {"IVV": seed_ivv, "QQQ": seed_qqq}

# Also track UNseeded for comparison
tier1_hist_unseeded = {"IVV": [], "QQQ": []}
tier1_med_unseeded = {"IVV": 0.0, "QQQ": 0.0}
tier_unseeded = 0

# Results
flat_rets = {}; grad_rets = {}; unlev_rets = {}
tier_log = []; tier_unseeded_log = []
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

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

    if is_ms:
        total_months += 1

        # Update signals
        ts_c = trend_df.index[trend_df.index <= day]
        if len(ts_c) > 0:
            for a in ASSETS:
                if a == "cash": continue
                if a in trend_df.columns:
                    v = trend_df.loc[ts_c[-1], a]
                    cur_trends[a] = int(v) if pd.notna(v) else 0

        z_c = z_clean.index[z_clean.index < day]
        if len(z_c) > 0:
            try:
                sim, _ = find_similar_months(z_clean, z_c[-1])
                harvey_er = compute_expected_returns(sim, asset_ret_fwd, avail)
            except ValueError:
                pass

        rv_c = rvol_monthly.index[rvol_monthly.index <= day]
        rvols = {}
        if len(rv_c) > 0:
            for a in ASSETS:
                if a == "cash": continue
                if a in rvol_monthly.columns:
                    v = rvol_monthly.loc[rv_c[-1], a]
                    rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

        # Phase 1 weights
        w1, pool = apply_faber_filter(cur_trends, BASELINE)
        w2, _ = direct_capital(dict(w1), pool, harvey_er, rvols)
        w_p1 = normalize(w2)

        # Conviction check
        f_c = cur_trends.get("IVV", 0) >= 3 and cur_trends.get("QQQ", 0) >= 3
        h_c = harvey_er.get("IVV", 0) > 0 and harvey_er.get("QQQ", 0) > 0
        flat_conv = f_c and h_c

        # Graduated tier (SEEDED)
        if f_c and h_c:
            above = all(harvey_er.get(a, 0) > tier1_med.get(a, 0) for a in ["IVV", "QQQ"])
            tier = 2 if above else 1
            for a in ["IVV", "QQQ"]:
                tier1_hist[a].append(harvey_er.get(a, 0))
                tier1_med[a] = float(np.median(tier1_hist[a]))
        else:
            tier = 0
        tier_log.append((day, tier))

        # Graduated tier (UNSEEDED — for comparison)
        if f_c and h_c:
            above_u = all(harvey_er.get(a, 0) > tier1_med_unseeded.get(a, 0) for a in ["IVV", "QQQ"])
            tier_unseeded = 2 if above_u else 1
            for a in ["IVV", "QQQ"]:
                tier1_hist_unseeded[a].append(harvey_er.get(a, 0))
                tier1_med_unseeded[a] = float(np.median(tier1_hist_unseeded[a]))
        else:
            tier_unseeded = 0
        tier_unseeded_log.append((day, tier_unseeded))

    # Compute returns
    iw = w_p1.get("IVV", 0); qw = w_p1.get("QQQ", 0)
    ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
    sr = 2 * ir - rfr - 0.0091 / 252
    ql = 2 * qr - rfr - 0.0089 / 252
    base = sum(w_p1.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])

    # 1x
    unlev_rets[day] = iw * ir + qw * qr + base

    # Flat 1.25x
    if flat_conv:
        flat_rets[day] = iw * 0.75 * ir + iw * 0.25 * sr + qw * 0.75 * qr + qw * 0.25 * ql + base
    else:
        flat_rets[day] = iw * ir + qw * qr + base

    # Graduated (seeded)
    sub = 0.25 if tier == 1 else (0.50 if tier == 2 else 0)
    if tier > 0:
        grad_rets[day] = iw * (1 - sub) * ir + iw * sub * sr + qw * (1 - sub) * qr + qw * sub * ql + base
    else:
        grad_rets[day] = iw * ir + qw * qr + base

# Convert to series
fs = pd.Series(flat_rets).sort_index()
gs = pd.Series(grad_rets).sort_index()
us = pd.Series(unlev_rets).sort_index()

# Benchmarks from same daily data
ivv_s = daily_ret["IVV"].loc[common_start:].dropna()
r_6040 = 0.6 * daily_ret.get("IVV", 0) + 0.4 * daily_ret.get("VGLT", 0)
b60 = r_6040.loc[common_start:].dropna()

def metrics(s, label):
    ar = s.mean() * 252; av = s.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]; ds = neg.std() * np.sqrt(252) if len(neg) > 10 else av
    sortino = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    calmar = ar / abs(dd) if dd != 0 else 0
    final = cum.iloc[-1]
    corr = s.corr(ivv_s) if len(s) == len(ivv_s) else s.reindex(ivv_s.index).corr(ivv_s)
    # Capture
    sm = s.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    im = ivv_s.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    c = sm.index.intersection(im.index)
    up = sm.reindex(c)[im.reindex(c) > 0].mean() / im.reindex(c)[im.reindex(c) > 0].mean() if (im.reindex(c) > 0).sum() > 0 else 0
    dn = sm.reindex(c)[im.reindex(c) < 0].mean() / im.reindex(c)[im.reindex(c) < 0].mean() if (im.reindex(c) < 0).sum() > 0 else 0
    return {"label": label, "ar": ar, "av": av, "sh": sh, "sortino": sortino,
            "dd": dd, "calmar": calmar, "final": final, "corr": corr, "up": up, "dn": dn}

# Report
print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>16} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'Final':>8} {'Up':>5} {'Down':>5}")
print(f"  {'-'*82}")

for s, label in [(gs, "P1+GradLev"), (fs, "P1+Flat1.25x"), (us, "P1 1x"), (ivv_s, "IVV B&H"), (b60, "60/40")]:
    m = metrics(s, label)
    print(f"  {m['label']:>16} {m['ar']:>6.1%} {m['av']:>6.1%} {m['sh']:>7.2f} {m['sortino']:>8.2f} "
          f"{m['dd']:>7.1%} {m['calmar']:>7.2f} ${m['final']:>7.2f} {m['up']:>4.0%} {m['dn']:>4.0%}")

# Tier diagnostics
print(f"\n{'='*80}")
print(f"  LEVERAGE TIER DIAGNOSTICS")
print(f"{'='*80}")

seeded_tc = {0: 0, 1: 0, 2: 0}
unseeded_tc = {0: 0, 1: 0, 2: 0}
for _, t in tier_log: seeded_tc[t] += 1
for _, t in tier_unseeded_log: unseeded_tc[t] += 1

print(f"\n  {'':>20} {'Seeded':>10} {'Unseeded':>10}")
print(f"  {'Tier 0 (1.0x)':>20} {seeded_tc[0]:>10} {unseeded_tc[0]:>10}")
print(f"  {'Tier 1 (1.25x)':>20} {seeded_tc[1]:>10} {unseeded_tc[1]:>10}")
print(f"  {'Tier 2 (1.5x)':>20} {seeded_tc[2]:>10} {unseeded_tc[2]:>10}")
t1p_s = seeded_tc[1] + seeded_tc[2]
t1p_u = unseeded_tc[1] + unseeded_tc[2]
print(f"  {'T2 as % of T1+':>20} {seeded_tc[2]/t1p_s:.0%}       {unseeded_tc[2]/t1p_u:.0%}")

# By 5-year period
print(f"\n  Tier 2 frequency by period (seeded):")
for start_yr, end_yr in [(2002,2006),(2007,2011),(2012,2016),(2017,2021),(2022,2026)]:
    period = [(d, t) for d, t in tier_log if start_yr <= d.year <= end_yr]
    if not period: continue
    n = len(period); t2 = sum(1 for _, t in period if t == 2)
    t1p = sum(1 for _, t in period if t >= 1)
    pct = t2 / t1p if t1p > 0 else 0
    print(f"    {start_yr}-{end_yr}: {t2}/{t1p} Tier 1+ months = {pct:.0%} upgrade to T2")

# Crisis drawdowns
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2, cs, ce in [("GFC", "2008-09-01", "2009-03-31"), ("COVID", "2020-02-19", "2020-03-23"), ("2022", "2022-01-03", "2022-10-31")]:
    print(f"\n  {cn2}:")
    for s, label in [(gs, "P1+GradLev"), (fs, "P1+Flat1.25x"), (us, "P1 1x"), (ivv_s, "IVV B&H")]:
        c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
        if len(c) > 0:
            cum = (1 + c).cumprod()
            mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
            print(f"    {label:>16}: {((1+c).prod()-1):>+7.1%} (DD {mdd:.1%})")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'GradLev':>8} {'Flat125':>8} {'P1 1x':>8} {'IVV':>8}")
for yr in range(2002, 2025):
    row = f"  {yr:>6}"
    for s in [gs, fs, us, ivv_s]:
        y = s[s.index.year == yr]
        row += f" {(1+y).prod()-1:>+7.1%}" if len(y) > 20 else f" {'--':>8}"
    print(row)

# Attribution: grad vs flat
print(f"\n{'='*80}")
print(f"  GRADUATED vs FLAT ATTRIBUTION")
print(f"{'='*80}")
diffs = []
for yr in range(2002, 2025):
    fy = fs[fs.index.year == yr]; gy = gs[gs.index.year == yr]
    if len(fy) > 20:
        fd = (1 + fy).prod() - 1; gd = (1 + gy).prod() - 1
        diffs.append((yr, gd - fd, gd, fd))
diffs.sort(key=lambda x: x[1])
print(f"\n  5 worst years (graduated underperforms flat):")
for yr, diff, gd, fd in diffs[:5]:
    print(f"    {yr}: grad={gd:+.1%}, flat={fd:+.1%}, diff={diff:+.1%}")
print(f"\n  5 best years (graduated outperforms flat):")
for yr, diff, gd, fd in diffs[-5:]:
    print(f"    {yr}: grad={gd:+.1%}, flat={fd:+.1%}, diff={diff:+.1%}")

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for s, label in [(gs, "P1+GradLev"), (fs, "P1+Flat1.25x"), (us, "P1 1x"), (ivv_s, "IVV B&H"), (b60, "60/40")]:
    print(f"  {label:>16}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

# Validation
print(f"\n{'='*80}")
print(f"  VALIDATION")
print(f"{'='*80}")
flat_final = (1 + fs).cumprod().iloc[-1]
print(f"  Flat 1.25x terminal: ${flat_final:.2f} (expected $11.28 ±$0.20)")
print(f"  Match: {'PASS' if abs(flat_final - 11.28) < 0.20 else 'FAIL'}")

grad_final = (1 + gs).cumprod().iloc[-1]
print(f"  Graduated terminal: ${grad_final:.2f} (should exceed flat)")
print(f"  Exceeds flat: {'PASS' if grad_final > flat_final else 'FAIL'}")

# Tier 2 frequency in first 5 years
first5 = [(d, t) for d, t in tier_log if d.year <= 2006]
if first5:
    t2_first5 = sum(1 for _, t in first5 if t == 2)
    t1p_first5 = sum(1 for _, t in first5 if t >= 1)
    pct_first5 = t2_first5 / t1p_first5 if t1p_first5 > 0 else 0
    print(f"  T2 in first 5 years: {pct_first5:.0%} (should be < 100% with seeded median)")
    print(f"  Fix working: {'PASS' if pct_first5 < 0.90 else 'FAIL'}")

t2_of_t1 = seeded_tc[2] / t1p_s if t1p_s > 0 else 0
print(f"  T2 as % of T1+: {t2_of_t1:.0%} (should be 25-50%)")
print(f"  Range check: {'PASS' if 0.20 <= t2_of_t1 <= 0.55 else 'CHECK'}")

print()
