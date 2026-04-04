"""Roth IRA 7-ETF regime allocation backtest (IVV, QQQ, VGLT, IAU, DBC, VNQ, cash)."""

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

logging.basicConfig(level=logging.WARNING)

config = RegimeConfig()
OUTPUT = Path("experiments/signal_development/output")
OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010

UNIVERSE = ["IVV", "QQQ", "VGLT", "IAU", "DBC", "VNQ", "cash"]

# Load data
raw_macro = fetch_monthly_history(config)
transformed = transform_variables(raw_macro, config)
z_data = get_valid_zscored(transformed, config)

asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
# Drop VXUS
if "VXUS" in asset_ret.columns:
    asset_ret = asset_ret.drop(columns=["VXUS"])

# FIX: Lag z-scores by 1 month (use T-1 data for T allocation)
z_data_lagged = z_data.shift(1).dropna()
# FIX: Forward returns from similar months (return AFTER similar conditions)
asset_ret_fwd = asset_ret.shift(-1)

start = pd.Timestamp(config.backtest_start)
end = pd.Timestamp(config.backtest_end)
bt_dates = z_data_lagged.index[(z_data_lagged.index >= start) & (z_data_lagged.index <= end)]
bt_dates = bt_dates[bt_dates.isin(asset_ret.index)]

print("=" * 80)
print("  ROTH IRA 7-ETF REGIME ALLOCATION BACKTEST")
print("=" * 80)
print(f"\nUniverse: {UNIVERSE}")
print(f"Asset returns: {asset_ret.shape[0]} months × {asset_ret.shape[1]} assets")
for c in sorted(asset_ret.columns):
    n = asset_ret[c].notna().sum()
    fv = asset_ret[c].first_valid_index()
    print(f"  {c:>6}: {n:>4} months (from {fv.date() if fv else 'N/A'})")
print(f"Backtest: {len(bt_dates)} months ({bt_dates.min().date()} to {bt_dates.max().date()})")

# Run backtest
strats = {"regime": {}, "bench_6040": {}, "bench_ivv": {}}
weight_rows = []
prev_w = {s: None for s in strats}

for dt in bt_dates:
    avail = [a for a in asset_ret.columns if a in UNIVERSE and pd.notna(asset_ret.loc[dt, a])]
    avail_etfs = [a for a in avail if a != "cash"]
    if len(avail_etfs) < 2:
        continue

    actual = {a: asset_ret.loc[dt, a] for a in avail}

    try:
        sim = compute_distances(z_data_lagged, dt, config)
    except ValueError:
        continue

    sim_er = {}
    for a in avail:
        rets = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates
                if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
        sim_er[a] = np.mean(rets) if rets else 0.0

    w = allocate_similarity_weighted(sim_er, max_single=0.40)

    w_6040 = {a: 0.0 for a in avail}
    if "IVV" in avail: w_6040["IVV"] = 0.60
    if "VGLT" in avail: w_6040["VGLT"] = 0.40
    w_ivv = {a: (1.0 if a == "IVV" else 0.0) for a in avail}

    all_w = {"regime": w, "bench_6040": w_6040, "bench_ivv": w_ivv}

    row = {"date": dt}
    for a in avail:
        row[a] = w.get(a, 0.0)
    weight_rows.append(row)

    for sn, sw in all_w.items():
        ret = sum(sw.get(a, 0) * actual.get(a, 0) for a in avail)
        if prev_w[sn] is not None:
            to = sum(abs(sw.get(a, 0) - prev_w[sn].get(a, 0)) for a in avail) / 2
            ret -= to * TC
        strats[sn][dt] = ret
        prev_w[sn] = sw

results = {s: pd.Series(d).sort_index() for s, d in strats.items() if d}
weights_df = pd.DataFrame(weight_rows).set_index("date")
weights_df.to_parquet(OUTPUT / "roth7_weights.parquet")

ivv_rets = results.get("bench_ivv")

# ── Performance summary ──────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  PERFORMANCE SUMMARY (1985-2024)")
print(f"{'='*80}")
print(f"\n  {'Strategy':>14} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'HitR':>5} {'CorrIVV':>8}")
print(f"  {'-'*58}")

for sn, label in [("regime","Regime Alloc"),("bench_6040","60/40"),("bench_ivv","IVV B&H")]:
    s = results.get(sn)
    if s is None or len(s) < 12: continue
    ar = s.mean() * 12; av = s.std() * np.sqrt(12)
    sh = ar / av if av > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    hit = (s > 0).mean()
    corr = s.corr(ivv_rets) if ivv_rets is not None else 0
    print(f"  {label:>14} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {hit:>5.0%} {corr:>7.2f}")

# ── By decade ─────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  BY DECADE")
print(f"{'='*80}")
for dname, ds, de in [("1985-94","1985","1994"),("1995-04","1995","2004"),
                       ("2005-14","2005","2014"),("2015-24","2015","2024")]:
    print(f"\n  {dname}:")
    print(f"  {'':>14} {'AnnRet':>7} {'Sharpe':>7}")
    for sn, label in [("regime","Regime"),("bench_6040","60/40"),("bench_ivv","IVV")]:
        s = results.get(sn)
        if s is None: continue
        m = (s.index >= pd.Timestamp(ds)) & (s.index < pd.Timestamp(f"{int(de)+1}"))
        d = s[m]
        if len(d) < 12: continue
        print(f"  {label:>14} {d.mean()*12:>6.1%} {(d.mean()/d.std()*np.sqrt(12)) if d.std()>0 else 0:>7.2f}")

# ── Crisis drawdowns ──────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn, cs, ce in [("GFC","2008-09","2009-03"),("COVID","2020-02","2020-03"),("2022","2022-01","2022-10")]:
    print(f"\n  {cn} ({cs} to {ce}):")
    for sn, label in [("regime","Regime"),("bench_6040","60/40"),("bench_ivv","IVV")]:
        s = results.get(sn)
        if s is None: continue
        c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
        if len(c) == 0: continue
        print(f"    {label:>10}: {((1+c).prod()-1):>+7.1%}")

# ── Calendar years ────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'Regime':>8} {'60/40':>8} {'IVV':>8}")
print(f"  {'-'*34}")
for yr in range(1985, 2025):
    row = f"  {yr:>6}"
    for sn in ["regime","bench_6040","bench_ivv"]:
        s = results.get(sn)
        if s is None: row += f" {'--':>8}"; continue
        y = s[s.index.year == yr]
        if len(y)==0: row += f" {'--':>8}"
        else: row += f" {(1+y).prod()-1:>+7.1%}"
    print(row)

# ── Final values ──────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1 invested)")
print(f"{'='*80}")
for sn, label in [("regime","Regime"),("bench_6040","60/40"),("bench_ivv","IVV")]:
    s = results.get(sn)
    if s is None or len(s)<12: continue
    print(f"  {label:>10}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

# ── Allocation diagnostics ────────────────────────────────────────────────────

diag_lines = []
def p(s=""):
    diag_lines.append(s)
    print(s)

p(f"\n{'='*80}")
p(f"  ALLOCATION DIAGNOSTICS")
p(f"{'='*80}")

p(f"\n  Average weights:")
p(f"  {'ETF':>6} {'Mean':>7} {'Median':>7} {'%Zero':>7}")
p(f"  {'-'*30}")
for a in sorted(weights_df.columns):
    c = weights_df[a]
    p(f"  {a:>6} {c.mean():>6.1%} {c.median():>6.1%} {(c<0.005).mean():>6.0%}")

# IVV+QQQ combined
ivv_qqq = weights_df.get("IVV", 0) + weights_df.get("QQQ", 0)
if hasattr(ivv_qqq, 'mean'):
    p(f"\n  IVV+QQQ combined: avg={ivv_qqq.mean():.1%}, median={ivv_qqq.median():.1%}")

# Crisis weights
for cn, cs, ce in [("GFC","2007-07","2009-06"),("COVID","2020-01","2020-06"),("2022","2022-01","2022-12")]:
    m = (weights_df.index >= pd.Timestamp(cs)) & (weights_df.index <= pd.Timestamp(ce))
    period = weights_df[m]
    if len(period)==0: continue
    p(f"\n  {cn} weights ({cs} to {ce}):")
    hdr = f"    {'Date':>7}"
    for a in sorted(weights_df.columns): hdr += f" {a[:5]:>5}"
    p(hdr)
    for dt, row in period.iterrows():
        line = f"    {dt.strftime('%Y-%m'):>7}"
        for a in sorted(weights_df.columns):
            v = row.get(a, 0)
            line += f" {v:>4.0%}" if v > 0.005 else f" {'--':>4}"
        p(line)

# Concentration
n_nz = []; hhi = []
eq_largest = 0
for _, row in weights_df.iterrows():
    vals = row.dropna()
    n_nz.append((vals > 0.005).sum())
    hhi.append(float((vals ** 2).sum()))
    ivv_qqq_w = vals.get("IVV", 0) + vals.get("QQQ", 0)
    if ivv_qqq_w >= vals.max():
        eq_largest += 1

p(f"\n  Avg assets with non-zero weight: {np.mean(n_nz):.1f}")
p(f"  Avg HHI: {np.mean(hhi):.3f} (1/{len(weights_df.columns)}={1/len(weights_df.columns):.3f} = perfect)")
p(f"  Months equity (IVV+QQQ) is largest combined: {eq_largest}/{len(weights_df)} ({eq_largest/len(weights_df):.0%})")

# Turnover
to_list = []
prev = None
for _, row in weights_df.iterrows():
    if prev is not None:
        to_list.append(sum(abs(row.get(a,0) - prev.get(a,0)) for a in weights_df.columns) / 2)
    prev = row
if to_list:
    p(f"\n  Turnover: avg={np.mean(to_list):.1%}, median={np.median(to_list):.1%}, max={max(to_list):.1%}")

p()

# Save diagnostics
with open(OUTPUT / "roth7_diagnostics.txt", "w") as f:
    f.write("\n".join(diag_lines))
