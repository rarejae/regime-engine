"""Phase 1 + recalibrated graduated leverage (1.3x / 1.65x tiers)."""

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

print("=" * 80)
print("  PHASE 1 + RECALIBRATED LEVERAGE (1.3x / 1.65x)")
print("=" * 80)

# Load data
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

# Pre-backtest seed
backtest_start = pd.Timestamp("2002-01-01")
pre_months = z_clean.index[z_clean.index < backtest_start]
pre_ers = {"IVV": [], "QQQ": []}
for z_dt in pre_months:
    try:
        sim, _ = find_similar_months(z_clean, z_dt)
        er = compute_expected_returns(sim, asset_ret_fwd, ["IVV", "QQQ"])
        for a in ["IVV", "QQQ"]:
            if er.get(a) is not None and not np.isnan(er.get(a, 0)):
                pre_ers[a].append(er[a])
    except ValueError:
        continue

SEED = {a: float(np.mean(pre_ers[a])) if pre_ers[a] else 0.005 for a in ["IVV", "QQQ"]}
print(f"Seeds: IVV={SEED['IVV']:.4f}, QQQ={SEED['QQQ']:.4f}")

common_start = max(daily_ret.dropna(how="all").index.min(), backtest_start)
trading_days = daily_ret.loc[common_start:].index
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

# Tier substitution fractions
TIERS = {
    "grad_130_165": {1: 0.30, 2: 0.65},
    "grad_125_150": {1: 0.25, 2: 0.50},
    "flat_125":     {1: 0.25, 2: 0.25},  # flat: both tiers use same sub
}

# State
ct = {a: 3 for a in ASSETS if a != "cash"}
he = {a: 0.0 for a in ASSETS}
wp = dict(BASELINE)
fc = False

# Per-config tier tracking
tier_hist = {}
tier_med = {}
for cfg in ["grad_130_165", "grad_125_150"]:
    tier_hist[cfg] = {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])}
    tier_med[cfg] = dict(SEED)

tier_log = {cfg: [] for cfg in TIERS}
strats = {cfg: {} for cfg in TIERS}
strats["p1_1x"] = {}
strats["bench_ivv"] = {}
strats["bench_6040"] = {}

# Effective equity exposure tracking
eff_eq = {cfg: [] for cfg in TIERS}

total_months = 0

for day in trading_days:
    if day not in daily_ret.index: continue
    dr = daily_ret.loc[day]
    avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3: continue
    actual = {a: float(dr[a]) for a in avail}
    rfr = float(rfr_daily.get(day, 0))

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

    if is_ms:
        total_months += 1
        # Signals
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
            except ValueError: pass
        rv_c = rvol_monthly.index[rvol_monthly.index <= day]
        rvols = {}
        if len(rv_c) > 0:
            for a in ASSETS:
                if a == "cash": continue
                if a in rvol_monthly.columns:
                    v = rvol_monthly.loc[rv_c[-1], a]
                    rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

        w1, pool = apply_faber_filter(ct, BASELINE)
        w2, _ = direct_capital(dict(w1), pool, he, rvols)
        wp = normalize(w2)

        f_c = ct.get("IVV", 0) >= 3 and ct.get("QQQ", 0) >= 3
        h_c = he.get("IVV", 0) > 0 and he.get("QQQ", 0) > 0
        fc = f_c and h_c

        # Determine tiers per config
        for cfg in ["grad_130_165", "grad_125_150"]:
            if fc:
                above = all(he.get(a, 0) > tier_med[cfg].get(a, 0) for a in ["IVV", "QQQ"])
                tier = 2 if above else 1
                for a in ["IVV", "QQQ"]:
                    tier_hist[cfg][a].append(he.get(a, 0))
                    tier_med[cfg][a] = float(np.median(tier_hist[cfg][a]))
            else:
                tier = 0
            tier_log[cfg].append((day, tier))

        # Flat: always tier 1 when conviction (same sub for T1 and T2)
        tier_log["flat_125"].append((day, 1 if fc else 0))

    # Ensure all tier_logs have an entry for current month
    for cfg in TIERS:
        if not tier_log[cfg] or tier_log[cfg][-1][0] != day:
            # Carry forward last known tier
            last_t = tier_log[cfg][-1][1] if tier_log[cfg] else 0
            if is_ms:
                pass  # already appended above
            elif not tier_log[cfg]:
                tier_log[cfg].append((day, 0))

        # Effective equity exposure
        iw = wp.get("IVV", 0); qw = wp.get("QQQ", 0)
        base_eq = iw + qw
        for cfg, subs in TIERS.items():
            t = tier_log[cfg][-1][1] if tier_log[cfg] else 0
            sub = subs.get(t, 0)
            # Effective: unlev portion × 1 + lev portion × 2
            eff = (iw * (1 - sub) * 1 + iw * sub * 2) + (qw * (1 - sub) * 1 + qw * sub * 2)
            eff_eq[cfg].append((day, eff, t))

    # Compute daily returns
    iw = wp.get("IVV", 0); qw = wp.get("QQQ", 0)
    ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
    sr = 2 * ir - rfr - 0.0091 / 252
    ql = 2 * qr - rfr - 0.0089 / 252
    base = sum(wp.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])

    # 1x
    strats["p1_1x"][day] = iw * ir + qw * qr + base
    strats["bench_ivv"][day] = actual.get("IVV", 0)
    strats["bench_6040"][day] = 0.6 * actual.get("IVV", 0) + 0.4 * actual.get("VGLT", 0)

    for cfg, subs in TIERS.items():
        t = tier_log[cfg][-1][1] if tier_log[cfg] else 0
        sub = subs.get(t, 0)
        if sub > 0:
            strats[cfg][day] = (iw * (1 - sub) * ir + iw * sub * sr +
                                qw * (1 - sub) * qr + qw * sub * ql + base)
        else:
            strats[cfg][day] = iw * ir + qw * qr + base

results = {c: pd.Series(d).sort_index() for c, d in strats.items()}
ivv_s = results["bench_ivv"]

labels = {"grad_130_165": "1.3x/1.65x", "grad_125_150": "1.25x/1.5x",
          "flat_125": "Flat 1.25x", "p1_1x": "P1 1x",
          "bench_ivv": "IVV B&H", "bench_6040": "60/40"}
order = ["grad_130_165", "grad_125_150", "flat_125", "p1_1x", "bench_6040", "bench_ivv"]

# Report
print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>12} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'Final':>8} {'Up':>5} {'Down':>5}")
print(f"  {'-'*82}")

for cn in order:
    s = results[cn]
    ar = s.mean()*252; av = s.std()*np.sqrt(252); sh = ar/av if av > 0 else 0
    neg = s[s<0]; ds = neg.std()*np.sqrt(252) if len(neg)>10 else av
    sortino = ar/ds if ds>0 else 0
    cum = (1+s).cumprod(); dd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
    calmar = ar/abs(dd) if dd!=0 else 0; final = cum.iloc[-1]
    sm = s.resample("MS").apply(lambda x:(1+x).prod()-1)
    im = ivv_s.resample("MS").apply(lambda x:(1+x).prod()-1)
    c = sm.index.intersection(im.index)
    up = sm.reindex(c)[im.reindex(c)>0].mean()/im.reindex(c)[im.reindex(c)>0].mean() if (im.reindex(c)>0).sum()>0 else 0
    dn = sm.reindex(c)[im.reindex(c)<0].mean()/im.reindex(c)[im.reindex(c)<0].mean() if (im.reindex(c)<0).sum()>0 else 0
    print(f"  {labels[cn]:>12} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {sortino:>8.2f} {dd:>7.1%} {calmar:>7.2f} ${final:>7.2f} {up:>4.0%} {dn:>4.0%}")

# Effective equity exposure
print(f"\n{'='*80}")
print(f"  EFFECTIVE EQUITY EXPOSURE")
print(f"{'='*80}")
for cfg in ["grad_130_165", "grad_125_150", "flat_125"]:
    ee = pd.DataFrame(eff_eq[cfg], columns=["date","eff","tier"]).set_index("date")
    for t in [0, 1, 2]:
        sub = ee[ee["tier"]==t]["eff"]
        if len(sub) > 0:
            print(f"  {labels[cfg]:>12} Tier {t}: avg eff equity = {sub.mean():.0%} (n={len(sub)})")

# Tier counts
print(f"\n{'='*80}")
print(f"  TIER DISTRIBUTION")
print(f"{'='*80}")
for cfg in ["grad_130_165", "grad_125_150", "flat_125"]:
    tc = {0:0,1:0,2:0}
    for _,t in tier_log[cfg]: tc[t]+=1
    t1p = tc[1]+tc[2]
    print(f"  {labels[cfg]:>12}: T0={tc[0]} T1={tc[1]} T2={tc[2]} (T2/{'{T1+}'}={tc[2]/t1p:.0%})" if t1p>0 else f"  {labels[cfg]:>12}: T0={tc[0]}")

# Tier 2 by period for new config
print(f"\n  1.3x/1.65x Tier 2 by period:")
for sy, ey in [(2002,2006),(2007,2011),(2012,2016),(2017,2021),(2022,2026)]:
    period = [(d,t) for d,t in tier_log["grad_130_165"] if sy<=d.year<=ey]
    if not period: continue
    t1p = sum(1 for _,t in period if t>=1); t2 = sum(1 for _,t in period if t==2)
    print(f"    {sy}-{ey}: {t2}/{t1p} = {t2/t1p:.0%}" if t1p>0 else f"    {sy}-{ey}: no T1+")

# Crisis
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2,cs,ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for cn in order:
        s = results[cn]
        c = s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c)>0:
            cum=(1+c).cumprod(); mdd=((cum-cum.expanding().max())/cum.expanding().max()).min()
            print(f"    {labels[cn]:>12}: {((1+c).prod()-1):>+7.1%} (DD {mdd:.1%})")

# Worst monthly return
print(f"\n{'='*80}")
print(f"  WORST CASE ANALYSIS")
print(f"{'='*80}")
for cfg in ["grad_130_165", "grad_125_150", "flat_125"]:
    s = results[cfg]
    monthly = s.resample("MS").apply(lambda x:(1+x).prod()-1)
    worst = monthly.min(); worst_dt = monthly.idxmin()
    print(f"  {labels[cfg]:>12}: worst month = {worst:+.1%} ({worst_dt.strftime('%Y-%m')})")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'1.3/1.65':>8} {'1.25/1.5':>8} {'Flat125':>8} {'P1 1x':>8} {'IVV':>8}")
for yr in range(2002,2025):
    row = f"  {yr:>6}"
    for cn in ["grad_130_165","grad_125_150","flat_125","p1_1x","bench_ivv"]:
        s = results[cn]; y = s[s.index.year==yr]
        row += f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
    print(row)

# Attribution: 1.3x/1.65x vs flat 1.25x
print(f"\n{'='*80}")
print(f"  1.3x/1.65x vs FLAT 1.25x ATTRIBUTION")
print(f"{'='*80}")
diffs = []
for yr in range(2002,2025):
    ny = results["grad_130_165"][results["grad_130_165"].index.year==yr]
    fy = results["flat_125"][results["flat_125"].index.year==yr]
    if len(ny)>20 and len(fy)>20:
        nd=(1+ny).prod()-1; fd=(1+fy).prod()-1
        diffs.append((yr,nd-fd,nd,fd))
diffs.sort(key=lambda x:x[1])
print("\n  5 worst:")
for yr,d,n,f in diffs[:5]: print(f"    {yr}: new={n:+.1%}, flat={f:+.1%}, diff={d:+.1%}")
print("\n  5 best:")
for yr,d,n,f in diffs[-5:]: print(f"    {yr}: new={n:+.1%}, flat={f:+.1%}, diff={d:+.1%}")

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in order:
    s = results[cn]
    print(f"  {labels[cn]:>12}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

# Validation
print(f"\n{'='*80}")
print(f"  VALIDATION")
print(f"{'='*80}")
f125 = (1+results["flat_125"]).cumprod().iloc[-1]
g125 = (1+results["grad_125_150"]).cumprod().iloc[-1]
g130 = (1+results["grad_130_165"]).cumprod().iloc[-1]
print(f"  Flat 1.25x: ${f125:.2f} (expected $11.28 ±$0.20) — {'PASS' if abs(f125-11.28)<0.20 else 'FAIL'}")
print(f"  Grad 1.25/1.5: ${g125:.2f} (expected $13.40 ±$0.20) — {'PASS' if abs(g125-13.40)<0.20 else 'FAIL'}")
print(f"  Grad 1.3/1.65: ${g130:.2f} (should exceed 1.25/1.5) — {'PASS' if g130>g125 else 'FAIL'}")

print()
