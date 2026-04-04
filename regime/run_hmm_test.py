"""Standalone evaluation of 2-state HMM trend detector."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np
import pandas as pd

from regime.hmm_trend import (
    fetch_daily_data, compute_features, compute_features_2f,
    fit_and_predict_rolling, apply_persistence_filter,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

OUTPUT = Path("experiments/signal_development/output")
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("  2-STATE HMM TREND DETECTOR — EVALUATION")
print("=" * 80)

# Fetch and compute
raw = fetch_daily_data()
features_3f = compute_features(raw)
features_2f = compute_features_2f(raw)

print(f"\nData: {raw.shape[0]} days, {raw.index.min().date()} to {raw.index.max().date()}")
print(f"3-feature set: {features_3f.dropna().shape[0]} clean days")
print(f"2-feature set: {features_2f.dropna().shape[0]} clean days")

# Fit both models
print("\nFitting 3-feature HMM (return + vol + credit)...")
pred_3f = fit_and_predict_rolling(features_3f)
print(f"  {len(pred_3f)} predictions")

print("Fitting 2-feature HMM (return + vol only)...")
pred_2f = fit_and_predict_rolling(features_2f)
print(f"  {len(pred_2f)} predictions")

# Save daily states
if len(pred_3f) > 0:
    pred_3f.to_parquet(OUTPUT / "hmm_daily_states.parquet", index=False)

# Merge with SPY for analysis
spy = raw["spy_close"]
spy_ret = spy.pct_change()

for label, pred in [("3-feature", pred_3f), ("2-feature", pred_2f)]:
    if len(pred) == 0:
        print(f"\n{label}: no predictions")
        continue

    pred = pred.set_index("date")
    pred["spy_ret"] = spy_ret.reindex(pred.index)
    pred["spy_close"] = spy.reindex(pred.index)

    # Apply persistence filter
    pred["state_filtered"] = apply_persistence_filter(pred["state"])

    # Zone classification
    pred["zone"] = "neutral"
    pred.loc[pred["bull_prob"] > 0.7, "zone"] = "bull"
    pred.loc[pred["bull_prob"] < 0.3, "zone"] = "bear"
    pred["zone_filtered"] = apply_persistence_filter(pred["zone"])

    print(f"\n{'='*80}")
    print(f"  {label.upper()} HMM RESULTS")
    print(f"{'='*80}")

    # State characterization
    print(f"\n  STATE CHARACTERIZATION:")
    for state in ["bull", "bear"]:
        mask = pred["state"] == state
        sd = pred[mask]
        pct = len(sd) / len(pred)
        ret_mean = sd["spy_ret"].mean() * 252
        ret_std = sd["spy_ret"].std() * np.sqrt(252)
        print(f"\n    {state.upper()} ({pct:.0%} of days):")
        print(f"      Ann return: {ret_mean:+.1%}, Ann vol: {ret_std:.1%}")

    # Forward return analysis
    print(f"\n  FORWARD RETURN ANALYSIS:")
    print(f"  {'State':>6} {'1d':>8} {'5d':>8} {'21d':>8} {'63d':>8}  {'DirAcc21d':>10}")
    for state in ["bull", "bear"]:
        mask = pred["state"] == state
        row = f"  {state:>6}"
        for horizon in [1, 5, 21, 63]:
            fwd = (spy / spy.shift(-horizon) - 1).reindex(pred.index)
            # Oops, that's backward. Fix:
            fwd = (spy.shift(-horizon) / spy - 1).reindex(pred.index)
            sub = fwd[mask].dropna()
            avg = sub.mean() * (252/horizon)  # annualize
            row += f" {avg:>+7.1%}"
        # Direction accuracy 21d
        fwd21 = (spy.shift(-21) / spy - 1).reindex(pred.index)
        sub21 = fwd21[mask].dropna()
        if state == "bull":
            da = (sub21 > 0).mean()
        else:
            da = (sub21 < 0).mean()
        row += f"  {da:>9.0%}"
        print(row)

    # Trading signal test
    print(f"\n  SIMPLE TRADING SIGNAL TEST:")
    # Long when bull_prob > 0.7 (filtered), cash when < 0.3, 50/50 between
    sig = pred["zone_filtered"].map({"bull": 1.0, "bear": 0.0, "neutral": 0.5})
    strat_ret = sig * pred["spy_ret"]
    bh_ret = pred["spy_ret"]

    # Filter to valid
    valid = pd.DataFrame({"strat": strat_ret, "bh": bh_ret}).dropna()
    if len(valid) > 252:
        s_ar = valid["strat"].mean() * 252
        s_av = valid["strat"].std() * np.sqrt(252)
        s_sh = s_ar / s_av if s_av > 0 else 0
        b_ar = valid["bh"].mean() * 252
        b_av = valid["bh"].std() * np.sqrt(252)
        b_sh = b_ar / b_av if b_av > 0 else 0

        s_cum = (1 + valid["strat"]).cumprod()
        s_dd = ((s_cum - s_cum.expanding().max()) / s_cum.expanding().max()).min()
        b_cum = (1 + valid["bh"]).cumprod()
        b_dd = ((b_cum - b_cum.expanding().max()) / b_cum.expanding().max()).min()

        print(f"    {'':>16} {'Signal':>10} {'Buy&Hold':>10}")
        print(f"    {'Ann return':>16} {s_ar:>9.1%} {b_ar:>9.1%}")
        print(f"    {'Ann vol':>16} {s_av:>9.1%} {b_av:>9.1%}")
        print(f"    {'Sharpe':>16} {s_sh:>10.2f} {b_sh:>10.2f}")
        print(f"    {'Max DD':>16} {s_dd:>9.1%} {b_dd:>9.1%}")
        print(f"    {'Final ($1)':>16} ${s_cum.iloc[-1]:>9.2f} ${b_cum.iloc[-1]:>9.2f}")

    # Transitions per year
    transitions = (pred["state_filtered"] != pred["state_filtered"].shift(1)).sum()
    n_years = len(pred) / 252
    print(f"\n    State transitions: {transitions} total, {transitions/n_years:.1f}/year")

    # Zone allocation
    for z in ["bull", "bear", "neutral"]:
        pct = (pred["zone_filtered"] == z).mean()
        print(f"    Zone {z}: {pct:.0%}")

# Known event validation
print(f"\n{'='*80}")
print("  KNOWN EVENT VALIDATION (3-feature HMM)")
print(f"{'='*80}")

if len(pred_3f) > 0:
    p3 = pred_3f.set_index("date")
    p3["zone_filtered"] = apply_persistence_filter(
        pd.Series("neutral", index=p3.index).where(
            (p3["bull_prob"] >= 0.3) & (p3["bull_prob"] <= 0.7),
            pd.Series("bull", index=p3.index).where(p3["bull_prob"] > 0.7, "bear")
        )
    )

    events = [
        ("GFC", "2007-06", "2009-06"),
        ("2013 Bull", "2013-01", "2013-12"),
        ("COVID", "2020-01", "2020-12"),
        ("2022 Bear", "2022-01", "2022-12"),
    ]

    for ename, es, ee in events:
        mask = (p3.index >= pd.Timestamp(es)) & (p3.index <= pd.Timestamp(ee))
        sub = p3[mask]
        if len(sub) == 0:
            print(f"\n  {ename}: no data")
            continue

        print(f"\n  {ename} ({es} to {ee}):")
        # Monthly average bull probability
        monthly = sub["bull_prob"].resample("MS").mean()
        for dt, bp in monthly.items():
            bar = "█" * int(bp * 30) + "░" * (30 - int(bp * 30))
            print(f"    {dt.strftime('%Y-%m')}: {bp:.2f} {bar}")

# Credit spread contribution
print(f"\n{'='*80}")
print("  CREDIT SPREAD CONTRIBUTION (3f vs 2f)")
print(f"{'='*80}")

if len(pred_3f) > 0 and len(pred_2f) > 0:
    p3 = pred_3f.set_index("date")
    p2 = pred_2f.set_index("date")

    # When did each first transition to bear during GFC?
    for label, p in [("3-feature", p3), ("2-feature", p2)]:
        gfc = p[(p.index >= "2007-06-01") & (p.index <= "2008-12-31")]
        bear_dates = gfc[gfc["state"] == "bear"].index
        if len(bear_dates) > 0:
            first_bear = bear_dates[0]
            print(f"  {label}: first bear state during GFC = {first_bear.date()}")
        else:
            print(f"  {label}: never entered bear state during GFC period")

    # Signal comparison
    for label, p in [("3-feature", p3), ("2-feature", p2)]:
        sr = spy_ret.reindex(p.index)
        sig = p["bull_prob"].apply(lambda x: 1.0 if x > 0.7 else (0.0 if x < 0.3 else 0.5))
        strat = sig * sr
        valid = strat.dropna()
        if len(valid) > 252:
            sh = valid.mean() * 252 / (valid.std() * np.sqrt(252))
            print(f"  {label} signal Sharpe: {sh:.2f}")

print()
