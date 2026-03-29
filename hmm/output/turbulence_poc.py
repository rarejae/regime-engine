"""Kritzman turbulence index + macro context regime signal — proof of concept.

Computes cross-asset Mahalanobis distance turbulence, combines with NFCI growth
and breakeven inflation context, classifies into 4 regimes, measures forward
SPY return separation. Also fits a 2-state HMM on smoothed turbulence.

All PIT-safe. Standalone — no imports from hmm/ module.

Usage: PYTHONPATH=. python hmm/output/turbulence_poc.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

PARQUET = Path("data/macro/raw_indicators.parquet")
HMM_PRED = Path("hmm/output/predictions.parquet")

# ─── Data loading ─────────────────────────────────────────────────────────────

raw = pd.read_parquet(PARQUET)
raw.index = pd.to_datetime(raw.index)
raw = raw.sort_index()

print("=" * 80)
print("  KRITZMAN TURBULENCE INDEX + MACRO CONTEXT — POC")
print("=" * 80)
print(f"Data: {raw.shape[0]} rows, {raw.index.min().date()} to {raw.index.max().date()}")

# ─── Step 1: Build turbulence index ──────────────────────────────────────────

# Asset basket — daily returns/changes
spy_ret = np.log(raw["SPY_CLOSE"] / raw["SPY_CLOSE"].shift(1))
ust_chg = raw["UST_10Y"].diff()
vix_chg = raw["VIX"].diff()
hy_chg = raw["HY_OAS"].diff()
copper_ret = np.log(raw["COPPER"] / raw["COPPER"].shift(1))
gold_ret = np.log(raw["GOLD"] / raw["GOLD"].shift(1))

basket = pd.DataFrame({
    "spy": spy_ret, "ust10y": ust_chg, "vix": vix_chg,
    "hy_oas": hy_chg, "copper": copper_ret, "gold": gold_ret,
}).dropna()

print(f"\nAsset basket: {basket.shape[0]} days, {basket.shape[1]} assets")

# Rolling Mahalanobis distance
lookback = 252
turbulence = pd.Series(np.nan, index=basket.index, dtype=float)

for i in range(lookback, len(basket)):
    window = basket.iloc[i - lookback:i]
    mu = window.mean().values
    cov = window.cov().values

    try:
        obs = basket.iloc[i].values
        d = mahalanobis(obs, mu, np.linalg.inv(cov))
        turbulence.iloc[i] = d
    except (np.linalg.LinAlgError, ValueError):
        continue

turbulence = turbulence.dropna()
print(f"Raw turbulence: {len(turbulence)} days")

# Smooth: sqrt + 21-day EMA (Kritzman method to reduce right-skew)
turb_smooth = np.sqrt(turbulence).ewm(span=21).mean()
turb_smooth = turb_smooth.shift(1)  # PIT safe

print(f"Smoothed turbulence: mean={turb_smooth.mean():.2f}, "
      f"std={turb_smooth.std():.2f}, "
      f"max={turb_smooth.max():.2f}")

# ─── Step 2: Build macro context ─────────────────────────────────────────────

def ewm_zscore(s, halflife=252, min_periods=63, clip=3.0):
    mu = s.ewm(halflife=halflife, min_periods=min_periods).mean()
    sd = s.ewm(halflife=halflife, min_periods=min_periods).std().replace(0, np.nan)
    return ((s - mu) / sd).clip(-clip, clip).shift(1)

nfci_z = ewm_zscore(raw["NFCI"])       # Higher = tighter = bearish for growth
breakeven_z = ewm_zscore(raw["BREAKEVEN_5Y"])

# ─── Step 3: Classify 2×2 regimes ────────────────────────────────────────────

# Align all series
df = pd.DataFrame({
    "turb": turb_smooth,
    "nfci_z": nfci_z,
    "breakeven_z": breakeven_z,
}, index=raw.index).dropna()

# SPY forward returns (from SPY_CLOSE)
spy = raw["SPY_CLOSE"]
df["spy_fwd_1m"] = (spy.shift(-21) / spy - 1).reindex(df.index)
df["spy_fwd_3m"] = (spy.shift(-63) / spy - 1).reindex(df.index)

# Turbulence: high vs low (trailing 252-day 80th percentile)
turb_80 = df["turb"].rolling(252, min_periods=126).quantile(0.80)
df["turb_high"] = df["turb"] > turb_80

# Growth direction: NFCI below median = loose = growth-favorable
nfci_med = df["nfci_z"].rolling(252, min_periods=126).median()
df["growth_favorable"] = df["nfci_z"] < nfci_med

# 2×2 regime
conditions = [
    (~df["turb_high"]) & (df["growth_favorable"]),   # calm + bullish
    (~df["turb_high"]) & (~df["growth_favorable"]),   # calm + bearish
    (df["turb_high"]) & (df["growth_favorable"]),     # turbulent + bullish
    (df["turb_high"]) & (~df["growth_favorable"]),    # turbulent + bearish
]
labels = ["calm_bull", "calm_bear", "turb_bull", "turb_bear"]
df["regime_2x2"] = np.select(conditions, labels, default="unknown")
df = df[df["regime_2x2"] != "unknown"]

print(f"\n2x2 classification: {len(df)} days")

# ─── Step 4: 2-state HMM on smoothed turbulence ─────────────────────────────

turb_arr = turb_smooth.dropna().values.reshape(-1, 1)
scaler = StandardScaler()
turb_scaled = scaler.fit_transform(turb_arr)

hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="full",
                              n_iter=100, random_state=42)
hmm_model.fit(turb_scaled)
hmm_states = hmm_model.predict(turb_scaled)

# Label: state with higher mean turbulence = "event"
means = hmm_model.means_.flatten()
event_state = int(np.argmax(means))
normal_state = 1 - event_state

hmm_labels = pd.Series("normal", index=turb_smooth.dropna().index)
hmm_labels[hmm_states == event_state] = "event"

df["regime_hmm2"] = hmm_labels.reindex(df.index)
df["regime_hmm2"] = df["regime_hmm2"].fillna("normal")

hmm_trans = hmm_model.transmat_
normal_self = hmm_trans[normal_state, normal_state]
event_self = hmm_trans[event_state, event_state]

print(f"\n2-state HMM on turbulence:")
print(f"  Normal state: mean={scaler.inverse_transform([[means[normal_state]]])[0][0]:.2f}, "
      f"self-trans={normal_self:.3f}, implied dur={1/(1-normal_self):.0f}d")
print(f"  Event state:  mean={scaler.inverse_transform([[means[event_state]]])[0][0]:.2f}, "
      f"self-trans={event_self:.3f}, implied dur={1/(1-event_self):.0f}d")

# ─── Step 5: Measure forward returns per regime ─────────────────────────────

unconditional = df["spy_fwd_1m"].mean()
print(f"\nUnconditional mean 1m return: {unconditional:.4f} ({unconditional:.2%})")


def regime_stats(df_in, regime_col, label):
    """Compute stats for each regime in a column."""
    print(f"\n{'─'*75}")
    print(f"  {label}")
    print(f"{'─'*75}")
    print(f"  {'Regime':>16} {'N':>6} {'%Time':>6} {'Avg1m':>8} {'Excess':>8} "
          f"{'Std1m':>7} {'DirAcc':>7} {'Sharpe':>7} {'AvgDur':>7}")
    print(f"  {'-'*73}")

    returns_by_state = []

    for regime in sorted(df_in[regime_col].unique()):
        mask = df_in[regime_col] == regime
        rd = df_in[mask]
        n = len(rd)
        pct = n / len(df_in)
        r1 = rd["spy_fwd_1m"].dropna()
        avg = r1.mean() if len(r1) else 0
        excess = avg - unconditional
        std = r1.std() if len(r1) > 1 else 0
        sharpe = avg / std * np.sqrt(12) if std > 0 else 0

        # Direction accuracy (excess)
        if excess > 0.002:
            dir_acc = (r1 > unconditional).mean()
            direction = "bullish"
        elif excess < -0.002:
            dir_acc = (r1 < unconditional).mean()
            direction = "bearish"
        else:
            dir_acc = 0.5
            direction = "neutral"

        # Duration
        runs = []
        cur = 0
        for v in df_in[regime_col]:
            if v == regime:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        avg_dur = np.mean(runs) if runs else 0

        returns_by_state.append(avg)

        print(f"  {regime:>16} {n:>6} {pct:>5.0%} {avg:>+7.2%} {excess:>+7.2%} "
              f"{std:>6.2%} {dir_acc:>6.0%} {sharpe:>+7.2f} {avg_dur:>6.0f}d")

    # Return separation
    if len(returns_by_state) >= 2:
        pooled_std = df_in["spy_fwd_1m"].std()
        sep = (max(returns_by_state) - min(returns_by_state)) / pooled_std if pooled_std > 0 else 0
        print(f"\n  Return separation: {sep:.3f} (range of means / pooled std)")
        print(f"  Range: {max(returns_by_state):.2%} - {min(returns_by_state):.2%} = {max(returns_by_state)-min(returns_by_state):.2%}")
    return returns_by_state


# 2×2 regimes
r_2x2 = regime_stats(df, "regime_2x2", "2×2 TURBULENCE + MACRO CONTEXT")

# 2-state HMM
r_hmm2 = regime_stats(df, "regime_hmm2", "2-STATE HMM ON TURBULENCE")

# ─── Step 6: Compare to 4-state HMM ─────────────────────────────────────────

if HMM_PRED.exists():
    hmm4 = pd.read_parquet(HMM_PRED)
    hmm4["trade_date"] = pd.to_datetime(hmm4["trade_date"])
    hmm4 = hmm4.set_index("trade_date")

    df["regime_hmm4"] = hmm4["most_likely_state"].reindex(df.index)
    df_hmm4 = df.dropna(subset=["regime_hmm4"])
    df_hmm4["regime_hmm4"] = df_hmm4["regime_hmm4"].astype(int).astype(str).apply(lambda x: f"state_{x}")

    r_hmm4 = regime_stats(df_hmm4, "regime_hmm4", "4-STATE HMM (CURRENT BEST)")
else:
    print("\n  4-state HMM predictions not found — skipping comparison")
    r_hmm4 = []

# ─── Step 7: Spike validation ────────────────────────────────────────────────

print(f"\n{'─'*75}")
print(f"  TURBULENCE SPIKE VALIDATION")
print(f"{'─'*75}")

key_dates = [
    ("2008-10-10", "GFC peak"),
    ("2011-08-08", "US downgrade"),
    ("2015-08-24", "China deval"),
    ("2020-03-16", "COVID crash"),
    ("2022-06-15", "Rate hike selloff"),
]

for ds, label in key_dates:
    dt = pd.Timestamp(ds)
    # Find nearest date
    for off in range(-3, 4):
        try:
            alt = dt + pd.Timedelta(days=off)
        except:
            continue
        if alt in turb_smooth.index:
            val = turb_smooth.loc[alt]
            regime = df.loc[alt, "regime_2x2"] if alt in df.index else "N/A"
            hmm_r = df.loc[alt, "regime_hmm2"] if alt in df.index else "N/A"
            p80 = turb_80.loc[alt] if alt in turb_80.index else 0
            print(f"  {ds} ({label:>20}): turb={val:.2f} (p80={p80:.2f}), "
                  f"2x2={regime}, hmm={hmm_r}")
            break

# ─── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")

seps = {
    "2x2 turb+macro": (max(r_2x2) - min(r_2x2)) / df["spy_fwd_1m"].std() if len(r_2x2) >= 2 else 0,
    "2-state HMM turb": (max(r_hmm2) - min(r_hmm2)) / df["spy_fwd_1m"].std() if len(r_hmm2) >= 2 else 0,
}
if r_hmm4:
    seps["4-state HMM"] = (max(r_hmm4) - min(r_hmm4)) / df["spy_fwd_1m"].std()

print(f"\n  Return separation comparison:")
for name, sep in sorted(seps.items(), key=lambda x: x[1], reverse=True):
    print(f"    {name:>20}: {sep:.3f}")

# Check: does turb_bear have negative excess?
for regime in ["turb_bear", "turb_bull", "calm_bear", "calm_bull"]:
    rd = df[df["regime_2x2"] == regime]["spy_fwd_1m"]
    avg = rd.mean()
    excess = avg - unconditional
    if regime == "turb_bear":
        print(f"\n  turb_bear excess return: {excess:+.2%} ({'PASS — negative' if excess < 0 else 'FAIL — positive'})")

# Drawdown detection
print(f"\n  Drawdown detection (turb_bear active during known drawdowns?):")
drawdowns = [
    ("2008-09-01", "2009-03-01", "GFC"),
    ("2011-07-01", "2011-10-01", "US downgrade"),
    ("2020-02-20", "2020-03-23", "COVID"),
    ("2022-01-01", "2022-10-01", "2022 bear"),
]
for start, end, name in drawdowns:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    period = df[(df.index >= s) & (df.index <= e)]
    if len(period) == 0:
        print(f"    {name}: no data")
        continue
    turb_bear_pct = (period["regime_2x2"] == "turb_bear").mean()
    turb_any_pct = period["turb_high"].mean() if "turb_high" in period.columns else 0
    print(f"    {name:>15} ({start} to {end}): turb_bear={turb_bear_pct:.0%}, any_turb={turb_any_pct:.0%}")

print()
