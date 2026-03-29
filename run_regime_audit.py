"""Part 1: Regime Classification Audit — pure macro analysis, no trading layer."""
from __future__ import annotations
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median, stdev

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv; load_dotenv()
from utils.logging import setup_logging; setup_logging("WARNING")
from loguru import logger; logger.remove(); logger.add(sys.stderr, level="ERROR")

import numpy as np
import pandas as pd
import duckdb

from backtest.macro_backfill import (
    load_cached_condition_vectors, compute_condition_vectors,
    RAW_INDICATORS_PATH, AXIS_MAP,
)
from engine.strategy_scorer import rank_strategies, STRATEGY_DISPLAY

# ── Load all data ────────────────────────────────────────────────────────────

cvs = load_cached_condition_vectors()
if not cvs:
    print("Computing CVs..."); compute_condition_vectors(); cvs = load_cached_condition_vectors()

raw = pd.read_parquet(str(RAW_INDICATORS_PATH))
raw.index = pd.to_datetime(raw.index)

# SPY close from options data
con = duckdb.connect()
spy_df = con.execute("""
    SELECT DISTINCT trade_date::DATE as dt, underlying_close as spy
    FROM 'data/processed/spy_options_*.parquet'
    WHERE underlying_close > 0
    ORDER BY dt
""").fetchdf()
spy_df = spy_df.set_index('dt').sort_index()
spy_df = spy_df[~spy_df.index.duplicated(keep='first')]

def spy_ret(d, days=30):
    """Forward return from date d over `days` calendar days."""
    t0 = pd.Timestamp(d)
    t1 = t0 + pd.Timedelta(days=days)
    s0 = spy_df.loc[:t0+pd.Timedelta(days=3)]
    s1 = spy_df.loc[:t1+pd.Timedelta(days=3)]
    if s0.empty or s1.empty: return None
    v0 = s0.iloc[-1]['spy']; v1 = s1.iloc[-1]['spy']
    if v0 == 0: return None
    return (v1 - v0) / v0 * 100

# ── Helper: find first trading day of month ──────────────────────────────────

def first_td(year, month):
    for d in range(1, 8):
        try:
            dt = date(year, month, d)
        except: continue
        if dt in cvs: return dt
    return None

direction_map = {
    "GOLDILOCKS": "bullish", "REFLATION": "bullish",
    "STAGFLATION": "bearish", "DEFLATION": "bearish",
    "MIXED": "neutral",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1A: Regime Timeline
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 110)
print("  SECTION 1A: REGIME TIMELINE (168 months, 2010-01 through 2023-12)")
print("=" * 110)
print(f"{'Date':>10} {'Growth':>7} {'Infl':>7} {'Liq':>7} {'Risk':>7} {'Label':>14} {'SPY_1m':>8} {'SPY_3m':>8}")
print("-" * 110)

monthly_data = []
for year in range(2010, 2024):
    for month in range(1, 13):
        dt = first_td(year, month)
        if dt is None: continue
        cv = cvs[dt]
        r1 = spy_ret(dt, 30)
        r3 = spy_ret(dt, 90)
        r1s = f"{r1:+.1f}%" if r1 is not None else "N/A"
        r3s = f"{r3:+.1f}%" if r3 is not None else "N/A"
        print(f"{str(dt):>10} {cv.growth:>+7.0f} {cv.inflation:>+7.0f} {cv.liquidity:>+7.0f} {cv.risk:>+7.0f} {cv.regime_label:>14} {r1s:>8} {r3s:>8}")
        monthly_data.append({"date": dt, "cv": cv, "r1": r1, "r3": r3})


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1B: Direction Accuracy
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("  SECTION 1B: REGIME DIRECTION ACCURACY")
print("=" * 110)

def dir_accuracy(data, horizon_key):
    by_regime = defaultdict(lambda: {"correct": 0, "incorrect": 0, "n": 0})
    total_c = total_i = 0
    for row in data:
        label = row["cv"].regime_label
        direction = direction_map[label]
        ret = row[horizon_key]
        if ret is None or direction == "neutral":
            by_regime[label]["n"] += 1
            continue
        by_regime[label]["n"] += 1
        if direction == "bullish":
            correct = ret > 0
        else:
            correct = ret < 0
        if correct:
            by_regime[label]["correct"] += 1
            total_c += 1
        else:
            by_regime[label]["incorrect"] += 1
            total_i += 1
    return by_regime, total_c, total_i

for hz, key in [("1-MONTH", "r1"), ("3-MONTH", "r3")]:
    br, tc, ti = dir_accuracy(monthly_data, key)
    total = tc + ti
    print(f"\n{hz} HORIZON (excluding MIXED):")
    print(f"  Overall: {tc}/{total} = {tc/total:.1%}" if total > 0 else "  No data")
    print(f"  {'Regime':>14} {'Correct':>8} {'Incorrect':>10} {'N':>5} {'Accuracy':>9}")
    for r in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION", "MIXED"]:
        d = br[r]
        c, i, n = d["correct"], d["incorrect"], d["n"]
        if r == "MIXED":
            print(f"  {r:>14} {'---':>8} {'---':>10} {n:>5} {'excluded':>9}")
        elif c + i > 0:
            print(f"  {r:>14} {c:>8} {i:>10} {n:>5} {c/(c+i):.0%}:>9")
        else:
            print(f"  {r:>14} {0:>8} {0:>10} {n:>5} {'N/A':>9}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1C: Known Macro Events
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("  SECTION 1C: REGIME ACCURACY AT 16 KNOWN MACRO EVENTS")
print("=" * 110)

KNOWN_EVENTS = [
    ("2010-06-01", "DEFLATION",    "European debt crisis, double-dip fears, falling CPI"),
    ("2011-08-01", "DEFLATION",    "US credit downgrade, debt ceiling crisis, VIX 48"),
    ("2013-06-03", "MIXED",        "Taper tantrum, rising yields but growth OK"),
    ("2015-08-24", "DEFLATION",    "China devaluation, flash crash, VIX 40"),
    ("2016-06-24", "MIXED",        "Brexit vote shock, immediate uncertainty"),
    ("2017-06-01", "GOLDILOCKS",   "Low vol, steady growth, low inflation"),
    ("2018-02-05", "MIXED",        "Volmageddon, VIX spike, strong economy"),
    ("2018-12-24", "DEFLATION",    "Christmas Eve crash, Fed overtightening"),
    ("2019-07-01", "GOLDILOCKS",   "Fed pivot to cuts, low inflation, moderate growth"),
    ("2020-03-16", "DEFLATION",    "COVID crash, VIX 82"),
    ("2020-11-02", "REFLATION",    "Vaccine news, massive stimulus, recovery"),
    ("2021-06-01", "REFLATION",    "Reopening boom, stimulus, inflation starting"),
    ("2022-06-15", "STAGFLATION",  "CPI 9.1%, Fed hiking, GDP negative Q1"),
    ("2022-10-03", "STAGFLATION",  "CPI above 8%, recession fears, housing collapse"),
    ("2023-01-03", "MIXED",        "Inflation falling, growth unclear, soft landing debate"),
    ("2023-07-03", "GOLDILOCKS",   "Inflation cooling, jobs strong, soft landing"),
]

correct = 0
mismatches = []
print(f"\n{'Date':>10} {'G':>6} {'I':>6} {'L':>6} {'R':>6} {'Engine':>14} {'Expected':>14} {'Result':>8}")
print("-" * 85)
for date_str, expected, desc in KNOWN_EVENTS:
    dt = date.fromisoformat(date_str)
    # Find nearest trading day
    cv = cvs.get(dt)
    if not cv:
        for off in range(-5, 6):
            try: alt = dt + timedelta(days=off)
            except: continue
            cv = cvs.get(alt)
            if cv: dt = alt; break
    if not cv:
        print(f"{date_str:>10} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'NO DATA':>14} {expected:>14} {'SKIP':>8}")
        continue
    match = cv.regime_label == expected
    if match: correct += 1
    else: mismatches.append((dt, cv, expected, desc))
    tag = "MATCH" if match else "MISS"
    print(f"{str(dt):>10} {cv.growth:>+6.0f} {cv.inflation:>+6.0f} {cv.liquidity:>+6.0f} {cv.risk:>+6.0f} {cv.regime_label:>14} {expected:>14} {tag:>8}")

print(f"\nScore: {correct}/16 correct")
if mismatches:
    print("\nMismatch details:")
    for dt, cv, expected, desc in mismatches:
        print(f"  {dt}: engine={cv.regime_label}, expected={expected}")
        print(f"    Context: {desc}")
        # Which axis is wrong?
        exp_dir = direction_map.get(expected, "neutral")
        eng_dir = direction_map.get(cv.regime_label, "neutral")
        if exp_dir != eng_dir:
            if expected in ("STAGFLATION", "DEFLATION") and cv.growth > -20:
                print(f"    Problem: Growth={cv.growth:+.0f} (should be < -20 for {expected})")
            if expected in ("GOLDILOCKS", "REFLATION") and cv.growth < 20:
                print(f"    Problem: Growth={cv.growth:+.0f} (should be > +20 for {expected})")
            if expected in ("STAGFLATION", "REFLATION") and cv.inflation < 20:
                print(f"    Problem: Inflation={cv.inflation:+.0f} (should be > +20 for {expected})")
            if expected in ("DEFLATION", "GOLDILOCKS") and cv.inflation > 20:
                print(f"    Problem: Inflation={cv.inflation:+.0f} (should be < +20 for {expected})")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1D: Axis Score Distributions
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("  SECTION 1D: AXIS SCORE DISTRIBUTIONS (all trading days 2010-2023)")
print("=" * 110)

# Collect all CV scores for trading days in 2010-2023
all_scores = {"growth": [], "inflation": [], "liquidity": [], "risk": []}
for dt, cv in sorted(cvs.items()):
    if isinstance(dt, date) and date(2010,1,1) <= dt <= date(2023,12,31):
        all_scores["growth"].append(cv.growth)
        all_scores["inflation"].append(cv.inflation)
        all_scores["liquidity"].append(cv.liquidity)
        all_scores["risk"].append(cv.risk)

print(f"\n{'Axis':>12} {'Mean':>7} {'Median':>7} {'StdDev':>7} {'Min':>7} {'Max':>7} {'<-50':>6} {'-50/-20':>7} {'-20/+20':>7} {'+20/+50':>7} {'>+50':>6}")
print("-" * 100)
for axis_name, vals in all_scores.items():
    n = len(vals)
    arr = np.array(vals)
    pct = lambda lo, hi: f"{np.sum((arr >= lo) & (arr < hi)) / n:.0%}"
    print(f"{axis_name:>12} {np.mean(arr):>+7.1f} {np.median(arr):>+7.1f} {np.std(arr):>7.1f} "
          f"{np.min(arr):>+7.0f} {np.max(arr):>+7.0f} "
          f"{pct(-101,-50):>6} {pct(-50,-20):>7} {pct(-20,20):>7} {pct(20,50):>7} {pct(50,101):>6}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1E: Regime Transition Quality
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("  SECTION 1E: REGIME TRANSITION QUALITY")
print("=" * 110)

# Build daily regime series
sorted_dates = sorted(dt for dt in cvs if isinstance(dt, date) and date(2010,1,1) <= dt <= date(2023,12,31))
transitions = []
for i in range(1, len(sorted_dates)):
    d_prev, d_curr = sorted_dates[i-1], sorted_dates[i]
    cv_prev, cv_curr = cvs[d_prev], cvs[d_curr]
    if cv_prev.regime_label != cv_curr.regime_label:
        transitions.append((d_curr, cv_prev.regime_label, cv_curr.regime_label))

# SPY returns around transitions
print(f"\nTotal transitions: {len(transitions)}")
print(f"\n{'Transition':>30} {'N':>3} {'SPY 20d before':>15} {'SPY 20d after':>15} {'Change':>8}")
print("-" * 80)
trans_stats = defaultdict(lambda: {"before": [], "after": []})
for dt, fr, to in transitions:
    r_before = spy_ret(dt - timedelta(days=25), 20)
    r_after = spy_ret(dt, 20)
    if r_before is not None and r_after is not None:
        key = f"{fr} → {to}"
        trans_stats[key]["before"].append(r_before)
        trans_stats[key]["after"].append(r_after)

for key in sorted(trans_stats, key=lambda k: abs(mean(trans_stats[k]["after"]) - mean(trans_stats[k]["before"])), reverse=True):
    d = trans_stats[key]
    n = len(d["before"])
    if n < 2: continue
    b = mean(d["before"]); a = mean(d["after"])
    print(f"{key:>30} {n:>3} {b:>+14.2f}% {a:>+14.2f}% {a-b:>+7.2f}%")

# Regime duration stats
print(f"\nRegime duration statistics:")
regime_runs = []
cur_start = sorted_dates[0]; cur_label = cvs[cur_start].regime_label
for dt in sorted_dates[1:]:
    if cvs[dt].regime_label != cur_label:
        dur = (dt - cur_start).days
        regime_runs.append((cur_label, dur))
        cur_label = cvs[dt].regime_label; cur_start = dt
regime_runs.append((cur_label, (sorted_dates[-1] - cur_start).days))

dur_by_regime = defaultdict(list)
for r, d in regime_runs:
    dur_by_regime[r].append(d)

print(f"\n{'Regime':>14} {'N':>4} {'Mean':>6} {'Median':>7} {'Min':>5} {'Max':>5}")
for r in ["GOLDILOCKS","REFLATION","STAGFLATION","DEFLATION","MIXED"]:
    durs = dur_by_regime.get(r, [])
    if not durs: continue
    print(f"{r:>14} {len(durs):>4} {mean(durs):>6.0f} {median(durs):>7.0f} {min(durs):>5} {max(durs):>5}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1F: Growth Axis Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("  SECTION 1F: GROWTH AXIS INDICATOR DEEP DIVE")
print("=" * 110)

# Compute z-scores
lookback = 252 * 5
rolling_mean = raw.rolling(window=lookback, min_periods=60).mean()
rolling_std = raw.rolling(window=lookback, min_periods=60).std()
z = (raw - rolling_mean) / rolling_std.replace(0, np.nan)
z = z.clip(-3, 3)

# SPY forward 1-month returns for correlation
spy_fwd_1m = {}
for dt in sorted_dates:
    r = spy_ret(dt, 30)
    if r is not None:
        spy_fwd_1m[pd.Timestamp(dt)] = r
spy_fwd_series = pd.Series(spy_fwd_1m)

growth_indicators = {ind: (pol, wt) for ind, (ax, pol, wt) in AXIS_MAP.items() if ax == "GROWTH"}
key_dates = [("2019-07-01", "goldilocks"), ("2020-03-16", "crash"), ("2022-06-15", "recession")]

print(f"\n{'Indicator':>15} {'Weight':>6}",end="")
for ds, _ in key_dates:
    print(f" {'Z('+ds+')':>16}",end="")
print(f" {'Corr_SPY_1m':>12} {'Note':>20}")
print("-" * 110)

for ind in sorted(growth_indicators.keys()):
    pol, wt = growth_indicators[ind]
    if ind not in z.columns:
        print(f"{ind:>15} {wt:>6.2f} {'N/A':>16} {'N/A':>16} {'N/A':>16} {'N/A':>12} NOT IN DATA")
        continue
    zvals = []
    for ds, _ in key_dates:
        ts = pd.Timestamp(ds)
        if ts in z.index:
            v = z.loc[ts, ind]
            zvals.append(f"{v:>+.3f}" if not np.isnan(v) else "NaN")
        else:
            zvals.append("N/A")

    # Correlation with forward SPY returns
    z_series = z[ind].reindex(spy_fwd_series.index)
    valid = pd.concat([z_series, spy_fwd_series], axis=1).dropna()
    if len(valid) > 50:
        corr = valid.iloc[:,0].corr(valid.iloc[:,1]) * pol  # multiply by polarity for interpretability
        corr_s = f"{corr:>+.3f}"
    else:
        corr_s = "N/A"

    # Flags
    notes = []
    ts_crash = pd.Timestamp("2020-03-16")
    if ts_crash in z.index:
        zc = z.loc[ts_crash, ind]
        if not np.isnan(zc) and zc * pol > 0.5:
            notes.append("WRONG@crash")
    ts_rec = pd.Timestamp("2022-06-15")
    if ts_rec in z.index:
        zr = z.loc[ts_rec, ind]
        if not np.isnan(zr) and zr * pol > 1.0:
            notes.append("WRONG@2022")

    note_s = ", ".join(notes) if notes else ""
    print(f"{ind:>15} {wt:>6.2f}", end="")
    for v in zvals:
        print(f" {v:>16}", end="")
    print(f" {corr_s:>12} {note_s:>20}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1G: Inflation Axis 2022 Verification
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("  SECTION 1G: INFLATION AXIS VERIFICATION FOR 2022")
print("=" * 110)

inflation_indicators = {ind: (pol, wt) for ind, (ax, pol, wt) in AXIS_MAP.items() if ax == "INFLATION"}

print(f"\n{'Month':>7} {'I_Score':>8}",end="")
for ind in sorted(inflation_indicators.keys()):
    print(f" {ind+'_raw':>12} {ind+'_z':>10}",end="")
print()
print("-" * 120)

for month in range(1, 13):
    dt = first_td(2022, month)
    if not dt: continue
    cv = cvs[dt]
    ts = pd.Timestamp(dt)
    print(f"2022-{month:02d} {cv.inflation:>+8.0f}",end="")
    for ind in sorted(inflation_indicators.keys()):
        if ind in raw.columns and ts in raw.index:
            rv = raw.loc[ts, ind]
            zv = z.loc[ts, ind] if ts in z.index else float('nan')
            print(f" {rv:>12.2f} {zv:>+10.3f}",end="")
        else:
            print(f" {'N/A':>12} {'N/A':>10}",end="")
    print()

# Check range
inf_scores_2022 = [cvs[first_td(2022,m)].inflation for m in range(1,13) if first_td(2022,m)]
print(f"\nInflation axis score range in 2022: {min(inf_scores_2022):+.0f} to {max(inf_scores_2022):+.0f}")
print(f"All months above +50: {'YES' if all(s > 50 for s in inf_scores_2022) else 'NO'}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("  PART 1: REGIME CLASSIFICATION AUDIT SUMMARY")
print("=" * 110)

# Direction accuracy
br1, tc1, ti1 = dir_accuracy(monthly_data, "r1")
br3, tc3, ti3 = dir_accuracy(monthly_data, "r3")
tot1 = tc1+ti1; tot3 = tc3+ti3

print(f"\nDIRECTION ACCURACY:")
print(f"  1-month horizon (excluding MIXED): {tc1}/{tot1} = {tc1/tot1:.1%}" if tot1 else "  N/A")
print(f"  3-month horizon (excluding MIXED): {tc3}/{tot3} = {tc3/tot3:.1%}" if tot3 else "  N/A")
print(f"\n  Per regime (1-month):")
for r in ["GOLDILOCKS","REFLATION","STAGFLATION","DEFLATION","MIXED"]:
    d = br1[r]
    c,i,n = d["correct"],d["incorrect"],d["n"]
    if r == "MIXED":
        print(f"    {r:>14}: {n} observations (excluded)")
    elif c+i > 0:
        print(f"    {r:>14}: {c/(c+i):.0%} correct ({c+i} observations, {c} correct, {i} incorrect)")
    else:
        print(f"    {r:>14}: N/A (0 observations)")
print(f"\n  Per regime (3-month):")
for r in ["GOLDILOCKS","REFLATION","STAGFLATION","DEFLATION","MIXED"]:
    d = br3[r]
    c,i,n = d["correct"],d["incorrect"],d["n"]
    if r == "MIXED":
        print(f"    {r:>14}: {n} observations (excluded)")
    elif c+i > 0:
        print(f"    {r:>14}: {c/(c+i):.0%} correct ({c+i} observations)")

print(f"\nKNOWN EVENT ACCURACY: {correct}/16 correct")
if mismatches:
    print("  Mismatches:")
    for dt, cv, expected, desc in mismatches:
        print(f"    {dt}: engine={cv.regime_label}, expected={expected} — {desc}")

print(f"\nAXIS DISTRIBUTIONS:")
for ax in ["growth","inflation","liquidity","risk"]:
    arr = np.array(all_scores[ax])
    dz = np.sum((arr >= -20) & (arr < 20)) / len(arr)
    print(f"  {ax.title():>12}: mean={np.mean(arr):+.1f}, std={np.std(arr):.1f}, dead zone [-20,+20]={dz:.0%}")

print(f"\n2022 INFLATION AXIS:")
print(f"  Score range: {min(inf_scores_2022):+.0f} to {max(inf_scores_2022):+.0f}")
print(f"  Correctly elevated (all >+50): {'YES' if all(s > 50 for s in inf_scores_2022) else 'NO'}")

# Overall verdict
acc1 = tc1/tot1 if tot1 else 0
print(f"\nOVERALL VERDICT:")
if acc1 >= 0.65:
    print(f"  Regime engine reliability: PASS")
    print(f"  Direction accuracy: {acc1:.0%} (above 60% trading threshold)")
    print(f"  Ready for options strategy layer: YES")
elif acc1 >= 0.55:
    print(f"  Regime engine reliability: PARTIAL")
    print(f"  Direction accuracy: {acc1:.0%} (marginal — near 60% threshold)")
    print(f"  Ready for options strategy layer: YES WITH CAVEATS")
else:
    print(f"  Regime engine reliability: FAIL")
    print(f"  Direction accuracy: {acc1:.0%} (below 60% trading threshold)")
    print(f"  Ready for options strategy layer: NO — fix regime engine first")

print()
