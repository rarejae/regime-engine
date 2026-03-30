"""Head-to-head: Euclidean vs Mahalanobis distance in Harvey similarity engine."""

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
from regime.similarity import compute_distances, compute_distances_mahalanobis
from regime.multi_asset import allocate_similarity_weighted
from regime.run_daily_backtest import fetch_daily_etf_returns

logging.basicConfig(level=logging.WARNING)

config = RegimeConfig()
OUTPUT = Path("regime/output")
OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010
UNIVERSE = ["IVV", "QQQ", "VGLT", "IAU", "DBC", "VNQ", "cash"]

# Load data
raw_macro = fetch_monthly_history(config)
transformed = transform_variables(raw_macro, config)
z_data = get_valid_zscored(transformed, config)
z_data_lagged = z_data.shift(1).dropna()

asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
if "VXUS" in asset_ret.columns:
    asset_ret = asset_ret.drop(columns=["VXUS"])
asset_ret_fwd = asset_ret.shift(-1)

daily_ret = fetch_daily_etf_returns()

common_start = max(daily_ret.dropna(how="all").index.min(), pd.Timestamp("2002-01-01"))
trading_days = daily_ret.loc[common_start:].index

print("=" * 80)
print("  EUCLIDEAN vs MAHALANOBIS DISTANCE COMPARISON")
print("=" * 80)
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

# Run both configs
configs = {"E_euclid": {}, "M_mahal": {}, "bench_6040": {}, "bench_ivv": {}}
current_w = {c: None for c in configs}
harvey_w = {"E_euclid": {a: 0.0 for a in UNIVERSE}, "M_mahal": {a: 0.0 for a in UNIVERSE}}

# For analog comparison
analog_log = {"E_euclid": {}, "M_mahal": {}}

for day in trading_days:
    if day not in daily_ret.index:
        continue
    dr = daily_ret.loc[day]
    avail = [a for a in UNIVERSE if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3:
        continue
    actual = {a: float(dr[a]) for a in avail}

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

    if is_ms:
        z_cands = z_data_lagged.index[z_data_lagged.index < day]
        if len(z_cands) > 0:
            z_dt = z_cands[-1]

            for metric, dist_fn in [("E_euclid", compute_distances),
                                     ("M_mahal", compute_distances_mahalanobis)]:
                try:
                    sim = dist_fn(z_data_lagged, z_dt, config)
                    sim_er = {}
                    for a in avail:
                        if a not in asset_ret_fwd.columns:
                            sim_er[a] = 0.0; continue
                        rets = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates
                                if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                        sim_er[a] = np.mean(rets) if rets else 0.0
                    harvey_w[metric] = allocate_similarity_weighted(sim_er, max_single=0.50)

                    # Log analogs for key dates
                    key = z_dt.strftime("%Y-%m")
                    if key in ("2008-12", "2022-07", "2020-02"):
                        analog_log[metric][key] = {
                            "top5": sim.distances.nsmallest(5).to_dict(),
                            "n_similar": len(sim.similar_dates),
                            "min_dist": sim.min_distance,
                        }
                except ValueError:
                    pass

    w_6040 = {a: 0.0 for a in avail}
    if "IVV" in avail: w_6040["IVV"] = 0.60
    if "VGLT" in avail: w_6040["VGLT"] = 0.40
    w_ivv = {a: (1.0 if a == "IVV" else 0.0) for a in avail}

    all_w = {"E_euclid": (harvey_w["E_euclid"], is_ms),
             "M_mahal": (harvey_w["M_mahal"], is_ms),
             "bench_6040": (w_6040, is_ms),
             "bench_ivv": (w_ivv, False)}

    for cn, (new_w, trade) in all_w.items():
        if current_w[cn] is not None and not trade:
            w_used = current_w[cn]
        else:
            w_used = new_w
            current_w[cn] = new_w
        configs[cn][day] = sum(w_used.get(a, 0) * actual.get(a, 0) for a in avail)

results = {c: pd.Series(d).sort_index() for c, d in configs.items() if d}
ivv = results.get("bench_ivv")
labels = {"E_euclid": "Euclidean", "M_mahal": "Mahalanobis", "bench_6040": "60/40", "bench_ivv": "IVV B&H"}

# Report
print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>14} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8}")
print(f"  {'-'*52}")

for cn in ["E_euclid", "M_mahal", "bench_6040", "bench_ivv"]:
    s = results[cn]
    ar = s.mean()*252; av = s.std()*np.sqrt(252)
    sh = ar/av if av > 0 else 0
    cum = (1+s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    corr = s.corr(ivv)
    print(f"  {labels[cn]:>14} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {corr:>7.2f}")

# Crisis
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2, cs, ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for cn in ["E_euclid","M_mahal","bench_6040","bench_ivv"]:
        s = results[cn]
        c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
        if len(c) > 0:
            print(f"    {labels[cn]:>14}: {((1+c).prod()-1):>+7.1%}")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'Euclid':>8} {'Mahal':>8} {'60/40':>8} {'IVV':>8}")
for yr in range(2002, 2025):
    row = f"  {yr:>6}"
    for cn in ["E_euclid","M_mahal","bench_6040","bench_ivv"]:
        s = results[cn]; y = s[s.index.year == yr]
        row += f" {(1+y).prod()-1:>+7.1%}" if len(y) > 20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in ["E_euclid","M_mahal","bench_6040","bench_ivv"]:
    s = results[cn]
    print(f"  {labels[cn]:>14}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

# Analog comparison for key dates
print(f"\n{'='*80}")
print(f"  ANALOG COMPARISON FOR KEY DATES")
print(f"{'='*80}")
for key in ["2008-12", "2022-07", "2020-02"]:
    print(f"\n  Assessment: {key}")
    for metric in ["E_euclid", "M_mahal"]:
        data = analog_log[metric].get(key)
        if data is None:
            print(f"    {labels[metric]}: no data")
            continue
        print(f"    {labels[metric]} (n_similar={data['n_similar']}, min_d={data['min_dist']:.2f}):")
        for dt, d in sorted(data["top5"].items(), key=lambda x: x[1])[:5]:
            print(f"      {dt.date()} d={d:.2f}")

# Sharpe difference
se = results["E_euclid"]; sm = results["M_mahal"]
she = se.mean()*252 / (se.std()*np.sqrt(252))
shm = sm.mean()*252 / (sm.std()*np.sqrt(252))
diff = shm - she
print(f"\n{'='*80}")
print(f"  VERDICT")
print(f"{'='*80}")
print(f"  Euclidean Sharpe:   {she:.3f}")
print(f"  Mahalanobis Sharpe: {shm:.3f}")
print(f"  Difference:         {diff:+.3f}")
if diff > 0.05:
    print(f"  ADOPT MAHALANOBIS (improvement > 0.05)")
elif diff < -0.05:
    print(f"  KEEP EUCLIDEAN (Mahalanobis worse by > 0.05)")
else:
    print(f"  KEEP EUCLIDEAN (difference < 0.05, not worth the complexity)")
print()
