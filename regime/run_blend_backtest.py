"""Daily backtest: Harvey + HMM blended allocation."""

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
from regime.hmm_trend import fetch_daily_data, compute_features, fit_and_predict_rolling, apply_persistence_filter
from regime.blend import blend_weights, EQUITY
from regime.run_daily_backtest import fetch_daily_etf_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
OUTPUT = Path("regime/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010
UNIVERSE = ["IVV", "QQQ", "VGLT", "IAU", "DBC", "VNQ", "cash"]

config = RegimeConfig()

print("=" * 80)
print("  HARVEY + HMM BLENDED ALLOCATION BACKTEST")
print("=" * 80)

# ── Data ──────────────────────────────────────────────────────────────────────

raw_macro = fetch_monthly_history(config)
transformed = transform_variables(raw_macro, config)
z_data = get_valid_zscored(transformed, config)
z_data_lagged = z_data.shift(1).dropna()

asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
if "VXUS" in asset_ret.columns: asset_ret = asset_ret.drop(columns=["VXUS"])
asset_ret_fwd = asset_ret.shift(-1)

daily_ret = fetch_daily_etf_returns()
spy_raw = fetch_daily_data()
features = compute_features(spy_raw)

print("\nFitting 3-feature HMM...")
hmm_pred = fit_and_predict_rolling(features)
print(f"  {len(hmm_pred)} daily predictions")
hmm_pred = hmm_pred.set_index("date")

# Persistence filter for zones
zone_raw = pd.Series("neutral", index=hmm_pred.index)
zone_raw[hmm_pred["bull_prob"] > 0.7] = "bull"
zone_raw[hmm_pred["bull_prob"] < 0.3] = "bear"
hmm_pred["zone"] = apply_persistence_filter(zone_raw)

# ── Backtest ──────────────────────────────────────────────────────────────────

common_start = max(daily_ret.dropna(how="all").index.min(),
                   hmm_pred.index.min(), pd.Timestamp("2002-01-01"))
trading_days = daily_ret.loc[common_start:].index
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

configs = {"H_harvey": {}, "HMM_only": {}, "BLEND": {},
           "bench_6040": {}, "bench_ivv": {}}
current_w = {c: None for c in configs}
total_tc = {c: 0.0 for c in configs}
n_trades = {c: 0 for c in configs}
harvey_w = {a: 0.0 for a in UNIVERSE}

# Override tracking
override_counts = {"bull_up": 0, "bear_down": 0, "neutral": 0}
weight_rows = []

spy_close = spy_raw["spy_close"]
spy_ret_daily = spy_close.pct_change()

for day in trading_days:
    if day not in daily_ret.index: continue
    dr = daily_ret.loc[day]
    avail = [a for a in UNIVERSE if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3: continue
    actual = {a: float(dr[a]) for a in avail}

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

    # Monthly Harvey signal
    if is_ms:
        z_cands = z_data_lagged.index[z_data_lagged.index < day]
        if len(z_cands) > 0:
            z_dt = z_cands[-1]
            try:
                sim = compute_distances(z_data_lagged, z_dt, config)
                sim_er = {}
                for a in avail:
                    if a not in asset_ret_fwd.columns: sim_er[a] = 0.0; continue
                    rets = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates
                            if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                    sim_er[a] = np.mean(rets) if rets else 0.0
                harvey_w = allocate_similarity_weighted(sim_er, max_single=0.50)
            except ValueError:
                pass

    # Daily HMM zone
    zone = hmm_pred.loc[day, "zone"] if day in hmm_pred.index else "neutral"
    bull_prob = hmm_pred.loc[day, "bull_prob"] if day in hmm_pred.index else 0.5

    # Config H: Harvey only
    w_h = dict(harvey_w)

    # Config HMM: standalone
    if zone == "bull":
        w_hmm = {"IVV": 1.0, **{a: 0.0 for a in UNIVERSE if a != "IVV"}}
    elif zone == "bear":
        w_hmm = {"cash": 1.0, **{a: 0.0 for a in UNIVERSE if a != "cash"}}
    else:
        w_hmm = {"IVV": 0.5, "cash": 0.5, **{a: 0.0 for a in UNIVERSE if a not in ("IVV", "cash")}}

    # Config BLEND
    w_blend = blend_weights(harvey_w, zone, bull_prob)

    # Track overrides
    if is_ms:
        h_eq = sum(harvey_w.get(a, 0) for a in EQUITY)
        if zone == "bull" and h_eq < 0.40:
            override_counts["bull_up"] += 1
        elif zone == "bear":
            override_counts["bear_down"] += 1
        else:
            override_counts["neutral"] += 1

        row = {"date": day, "zone": zone, "bull_prob": bull_prob}
        for a in UNIVERSE:
            row[f"h_{a}"] = harvey_w.get(a, 0)
            row[f"b_{a}"] = w_blend.get(a, 0)
        weight_rows.append(row)

    # Benchmarks
    w_6040 = {a: 0.0 for a in avail}
    if "IVV" in avail: w_6040["IVV"] = 0.60
    if "VGLT" in avail: w_6040["VGLT"] = 0.40
    w_ivv = {a: (1.0 if a == "IVV" else 0.0) for a in avail}

    all_configs = {
        "H_harvey": (w_h, is_ms), "HMM_only": (w_hmm, zone != (current_w.get("_prev_zone") or zone)),
        "BLEND": (w_blend, is_ms or zone != (current_w.get("_prev_zone") or zone)),
        "bench_6040": (w_6040, is_ms), "bench_ivv": (w_ivv, False),
    }
    current_w["_prev_zone"] = zone

    for cn, (new_w, trade) in all_configs.items():
        if cn.startswith("_"): continue
        if current_w[cn] is not None and not trade:
            w_used = current_w[cn]
        else:
            w_used = new_w
            to = sum(abs(new_w.get(a,0) - (current_w[cn] or {}).get(a,0)) for a in avail) / 2
            total_tc[cn] += to * TC
            if to > 0.01: n_trades[cn] += 1
            current_w[cn] = new_w

        configs[cn][day] = sum(w_used.get(a, 0) * actual.get(a, 0) for a in avail)

results = {c: pd.Series(d).sort_index() for c, d in configs.items() if d}
wdf = pd.DataFrame(weight_rows).set_index("date")

n_years = len(trading_days) / 252
ivv = results.get("bench_ivv")
labels = {"H_harvey": "Harvey Only", "HMM_only": "HMM Only", "BLEND": "BLEND",
          "bench_6040": "60/40", "bench_ivv": "IVV B&H"}

# ── Report ────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>14} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8} {'AnnTC':>6}")
print(f"  {'-'*58}")

for cn in ["H_harvey","HMM_only","BLEND","bench_6040","bench_ivv"]:
    s = results.get(cn)
    if s is None or len(s) < 252: continue
    ar = s.mean()*252; av = s.std()*np.sqrt(252)
    sh = ar/av if av>0 else 0
    cum = (1+s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    corr = s.corr(ivv) if ivv is not None else 0
    tc = total_tc.get(cn, 0) / n_years
    print(f"  {labels[cn]:>14} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {corr:>7.2f} {tc:>5.2%}")

# Crisis drawdowns
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2, cs, ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for cn in ["H_harvey","HMM_only","BLEND","bench_6040","bench_ivv"]:
        s = results.get(cn)
        if s is None: continue
        c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
        if len(c) > 0:
            print(f"    {labels[cn]:>14}: {((1+c).prod()-1):>+7.1%}")

# Bull market capture
print(f"\n{'='*80}")
print(f"  BULL MARKET CAPTURE")
print(f"{'='*80}")
for yr, desc in [(2013,"Harvey worst"),(2019,"Strong bull"),(2023,"Tech momentum")]:
    print(f"\n  {yr} ({desc}):")
    for cn in ["H_harvey","HMM_only","BLEND","bench_ivv"]:
        s = results.get(cn)
        if s is None: continue
        y = s[s.index.year == yr]
        if len(y) > 20:
            print(f"    {labels[cn]:>14}: {(1+y).prod()-1:>+7.1%}")

# 2009 recovery
print(f"\n{'='*80}")
print(f"  2009 RECOVERY — MONTHLY EQUITY & RETURN")
print(f"{'='*80}")
for cn in ["H_harvey","BLEND"]:
    col_prefix = "h_" if cn == "H_harvey" else "b_"
    sub = wdf[(wdf.index >= "2009-01") & (wdf.index <= "2009-06")]
    s = results.get(cn)
    if sub.empty or s is None: continue
    print(f"\n  {labels[cn]}:")
    for dt, row in sub.iterrows():
        eq = row.get(f"{col_prefix}IVV", 0) + row.get(f"{col_prefix}QQQ", 0)
        m_ret = s[(s.index >= dt) & (s.index < dt + pd.DateOffset(months=1))]
        mr = (1+m_ret).prod()-1 if len(m_ret) > 0 else 0
        print(f"    {dt.strftime('%Y-%m')}: eq={eq:.0%}, return={mr:>+5.1%}")

# 2022 test
print(f"\n{'='*80}")
print(f"  2022 SLOW BEAR — MONTHLY EQUITY")
print(f"{'='*80}")
sub22 = wdf[wdf.index.year == 2022]
if not sub22.empty:
    for cn, prefix in [("H_harvey","h_"),("BLEND","b_")]:
        print(f"\n  {labels[cn]}:")
        for dt, row in sub22.iterrows():
            eq = row.get(f"{prefix}IVV", 0) + row.get(f"{prefix}QQQ", 0)
            z = row.get("zone", "?")
            print(f"    {dt.strftime('%Y-%m')}: eq={eq:.0%} zone={z}")

# Override frequency
total_months = len(wdf)
print(f"\n{'='*80}")
print(f"  OVERRIDE FREQUENCY")
print(f"{'='*80}")
print(f"  Bull overrides (HMM pushed equity up):  {override_counts['bull_up']:>4} ({override_counts['bull_up']/total_months:.0%})")
print(f"  Bear reductions (HMM cut equity):       {override_counts['bear_down']:>4} ({override_counts['bear_down']/total_months:.0%})")
print(f"  Neutral (no modification):              {override_counts['neutral']:>4} ({override_counts['neutral']/total_months:.0%})")
print(f"  Total months: {total_months}")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'Harvey':>8} {'HMM':>8} {'BLEND':>8} {'60/40':>8} {'IVV':>8}")
for yr in range(2002, 2025):
    row = f"  {yr:>6}"
    for cn in ["H_harvey","HMM_only","BLEND","bench_6040","bench_ivv"]:
        s = results.get(cn)
        if s is None: row += f" {'--':>8}"; continue
        y = s[s.index.year == yr]
        row += f" {(1+y).prod()-1:>+7.1%}" if len(y) > 20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in ["H_harvey","HMM_only","BLEND","bench_6040","bench_ivv"]:
    s = results.get(cn)
    if s is None or len(s) < 252: continue
    print(f"  {labels[cn]:>14}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

print()
