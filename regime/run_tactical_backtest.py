"""Daily backtest: baseline portfolio + rare tactical overlay."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np
import pandas as pd

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances
from regime.multi_asset import allocate_similarity_weighted
from regime.tactical_overlay import BASELINE, EQUITY, compute_confidence, apply_overlay
from regime.run_daily_backtest import fetch_daily_etf_returns

logging.basicConfig(level=logging.WARNING)
OUTPUT = Path("regime/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010
UNIVERSE = list(BASELINE.keys())

config = RegimeConfig()
raw_macro = fetch_monthly_history(config)
transformed = transform_variables(raw_macro, config)
z_data = get_valid_zscored(transformed, config)
z_data_lagged = z_data.shift(1).dropna()

asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
if "VXUS" in asset_ret.columns: asset_ret = asset_ret.drop(columns=["VXUS"])
# Drop VNQ — not in baseline
for col in list(asset_ret.columns):
    if col not in UNIVERSE: asset_ret = asset_ret.drop(columns=[col])
asset_ret_fwd = asset_ret.shift(-1)

daily_ret = fetch_daily_etf_returns()
# Keep only baseline assets
daily_cols = [c for c in daily_ret.columns if c in UNIVERSE]
daily_ret = daily_ret[daily_cols]

common_start = max(daily_ret.dropna(how="all").index.min(), pd.Timestamp("2002-01-01"))
trading_days = daily_ret.loc[common_start:].index

print("=" * 80)
print("  TACTICAL OVERLAY BACKTEST — BASELINE + RARE CONVICTION")
print("=" * 80)
print(f"Universe: {UNIVERSE}")
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

# Run
strats = {"T_tactical": {}, "B_baseline": {}, "H_harvey": {}, "bench_6040": {}, "bench_ivv": {}}
current_w = {c: None for c in strats}
total_tc = {c: 0.0 for c in strats}
n_trades = {c: 0 for c in strats}
harvey_w = {a: 0.0 for a in UNIVERSE}

trailing_min_dists = []
trailing_confidences = []
signal_log = []
current_tactical_action = "hold"

for day in trading_days:
    if day not in daily_ret.index: continue
    dr = daily_ret.loc[day]
    avail = [a for a in UNIVERSE if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3: continue
    actual = {a: float(dr[a]) for a in avail}

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

    tactical_w = None

    if is_ms:
        z_cands = z_data_lagged.index[z_data_lagged.index < day]
        if len(z_cands) > 0:
            z_dt = z_cands[-1]
            try:
                sim = compute_distances(z_data_lagged, z_dt, config)
                trailing_min_dists.append(sim.min_distance)

                # Harvey unconstrained
                sim_er = {}
                similar_fwd = {}
                for a in avail:
                    if a not in asset_ret_fwd.columns: sim_er[a] = 0.0; similar_fwd[a] = []; continue
                    rets = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates
                            if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                    sim_er[a] = np.mean(rets) if rets else 0.0
                    similar_fwd[a] = rets

                harvey_w = allocate_similarity_weighted(sim_er, max_single=0.50)

                # Tactical confidence
                conf_result = compute_confidence(
                    similar_fwd, sim.min_distance,
                    trailing_min_dists, trailing_confidences,
                )
                trailing_confidences.append(conf_result["confidence"])

                # Overlay decision
                tactical_w, action, desc = apply_overlay(
                    conf_result["confidence_pctl"],
                    conf_result["asset_metrics"],
                )

                eq_before = sum(BASELINE.get(a, 0) for a in EQUITY)
                eq_after = sum(tactical_w.get(a, 0) for a in EQUITY)

                if action != "hold":
                    signal_log.append({
                        "date": day,
                        "confidence": conf_result["confidence"],
                        "pctl": conf_result["confidence_pctl"],
                        "action": action,
                        "desc": desc,
                        "eq_before": eq_before,
                        "eq_after": eq_after,
                        "analog_quality": conf_result["analog_quality"],
                    })
                    current_tactical_action = action
                else:
                    if current_tactical_action != "hold":
                        signal_log.append({
                            "date": day, "confidence": conf_result["confidence"],
                            "pctl": conf_result["confidence_pctl"],
                            "action": "revert", "desc": "Back to baseline",
                            "eq_before": eq_after if tactical_w else eq_before,
                            "eq_after": eq_before,
                            "analog_quality": conf_result["analog_quality"],
                        })
                    current_tactical_action = "hold"
                    tactical_w = dict(BASELINE)

            except ValueError:
                tactical_w = dict(BASELINE)

    if tactical_w is None:
        tactical_w = current_w.get("T_tactical") or dict(BASELINE)

    w_base = dict(BASELINE)
    w_6040 = {a: 0.0 for a in avail}
    if "IVV" in avail: w_6040["IVV"] = 0.60
    if "VGLT" in avail: w_6040["VGLT"] = 0.40
    w_ivv = {a: (1.0 if a == "IVV" else 0.0) for a in avail}

    all_w = {"T_tactical": (tactical_w, is_ms), "B_baseline": (w_base, is_ms),
             "H_harvey": (harvey_w, is_ms), "bench_6040": (w_6040, is_ms),
             "bench_ivv": (w_ivv, False)}

    for cn, (new_w, trade) in all_w.items():
        if current_w[cn] is not None and not trade:
            w_used = current_w[cn]
        else:
            w_used = new_w
            to = sum(abs(new_w.get(a,0) - (current_w[cn] or {}).get(a,0)) for a in avail) / 2
            total_tc[cn] += to * TC
            if to > 0.01: n_trades[cn] += 1
            current_w[cn] = new_w
        strats[cn][day] = sum(w_used.get(a,0) * actual.get(a,0) for a in avail)

results = {c: pd.Series(d).sort_index() for c, d in strats.items() if d}
n_years = len(trading_days) / 252
ivv = results.get("bench_ivv")
labels = {"T_tactical": "Tactical", "B_baseline": "Baseline", "H_harvey": "Harvey",
          "bench_6040": "60/40", "bench_ivv": "IVV B&H"}

# ── Report ────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>12} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8} {'AnnTC':>6}")
print(f"  {'-'*56}")

for cn in ["T_tactical","B_baseline","H_harvey","bench_6040","bench_ivv"]:
    s = results.get(cn)
    if s is None or len(s)<252: continue
    ar=s.mean()*252; av=s.std()*np.sqrt(252); sh=ar/av if av>0 else 0
    cum=(1+s).cumprod(); dd=((cum-cum.expanding().max())/cum.expanding().max()).min()
    corr=s.corr(ivv) if ivv is not None else 0
    tc=total_tc.get(cn,0)/n_years
    print(f"  {labels[cn]:>12} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {corr:>7.2f} {tc:>5.2%}")

# Crisis
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2,cs,ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for cn in ["T_tactical","B_baseline","H_harvey","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        c=s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c)>0: print(f"    {labels[cn]:>12}: {((1+c).prod()-1):>+7.1%}")

# Signal history
print(f"\n{'='*80}")
print(f"  SIGNAL FIRING HISTORY")
print(f"{'='*80}")
sig_df = pd.DataFrame(signal_log)
if len(sig_df) > 0:
    bullish = sig_df[sig_df["action"]=="bullish"]
    bearish = sig_df[sig_df["action"]=="bearish"]
    reverts = sig_df[sig_df["action"]=="revert"]
    print(f"  Bullish fires:  {len(bullish)}")
    print(f"  Bearish fires:  {len(bearish)}")
    print(f"  Reverts:        {len(reverts)}")
    print(f"  Total signals:  {len(bullish)+len(bearish)} over {n_years:.0f} years = {(len(bullish)+len(bearish))/n_years:.1f}/year")
    print(f"\n  {'Date':>10} {'Action':>8} {'Pctl':>6} {'Eq%':>12} {'Desc'}")
    print(f"  {'-'*60}")
    for _,r in sig_df[sig_df["action"]!="revert"].iterrows():
        eq_s = f"{r['eq_before']:.0%}→{r['eq_after']:.0%}"
        print(f"  {r['date'].strftime('%Y-%m-%d'):>10} {r['action']:>8} {r['pctl']:>5.0%} {eq_s:>12} {r['desc']}")
else:
    print("  No signals fired")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'Tactic':>8} {'Base':>8} {'Harvey':>8} {'IVV':>8}")
for yr in range(2002,2025):
    row=f"  {yr:>6}"
    for cn in ["T_tactical","B_baseline","H_harvey","bench_ivv"]:
        s=results.get(cn)
        if s is None: row+=f" {'--':>8}"; continue
        y=s[s.index.year==yr]
        row+=f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in ["T_tactical","B_baseline","H_harvey","bench_6040","bench_ivv"]:
    s=results.get(cn)
    if s is None or len(s)<252: continue
    print(f"  {labels[cn]:>12}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

# Key years
print(f"\n{'='*80}")
print(f"  KEY YEAR ANALYSIS")
print(f"{'='*80}")
if len(sig_df) > 0:
    for yr, desc in [(2008,"GFC"),(2009,"Recovery"),(2013,"Bull"),(2022,"Bear")]:
        yr_sigs = sig_df[(sig_df["date"].dt.year==yr) & (sig_df["action"]!="revert")]
        print(f"\n  {yr} ({desc}): {len(yr_sigs)} signals fired")
        for _,r in yr_sigs.iterrows():
            print(f"    {r['date'].strftime('%Y-%m')}: {r['action']} (pctl={r['pctl']:.0%}, eq={r['eq_before']:.0%}→{r['eq_after']:.0%})")

# Excess return from overlay
s_t = results.get("T_tactical")
s_b = results.get("B_baseline")
if s_t is not None and s_b is not None:
    excess = (s_t.mean() - s_b.mean()) * 252
    print(f"\n  Annualized excess return from overlay: {excess:+.2%}")

print()
