"""Allocation diagnostics — trace what similarity_weighted actually does."""

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
from regime.multi_asset import (
    ASSETS, fetch_asset_returns, allocate_similarity_weighted,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

config = RegimeConfig()
config.output_dir.mkdir(parents=True, exist_ok=True)

raw_macro = fetch_monthly_history(config)
transformed = transform_variables(raw_macro, config)
z_data = get_valid_zscored(transformed, config)
asset_ret = fetch_asset_returns()

start = pd.Timestamp(config.backtest_start)
end = pd.Timestamp(config.backtest_end)
bt_dates = z_data.index[(z_data.index >= start) & (z_data.index <= end)]
bt_dates = bt_dates[bt_dates.isin(asset_ret.index)]

avail_assets = [a for a in asset_ret.columns if a in ASSETS or a in asset_ret.columns]

# ── Recompute all monthly weights ─────────────────────────────────────────────

weight_rows = []
er_rows = []  # expected returns
sim_detail_rows = {}  # for trace examples

for dt in bt_dates:
    avail = [a for a in asset_ret.columns if pd.notna(asset_ret.loc[dt, a])]
    if len(avail) < 3:
        continue

    try:
        sim_result = compute_distances(z_data, dt, config)
    except ValueError:
        continue

    sim_er = {}
    for a in avail:
        rets = [asset_ret.loc[d, a] for d in sim_result.similar_dates
                if d in asset_ret.index and pd.notna(asset_ret.loc[d, a])]
        sim_er[a] = np.mean(rets) if rets else 0.0

    w = allocate_similarity_weighted(sim_er)

    row = {"date": dt}
    er_row = {"date": dt}
    for a in avail:
        row[a] = w.get(a, 0.0)
        er_row[a] = sim_er.get(a, 0.0)
    weight_rows.append(row)
    er_rows.append(er_row)

    # Store detail for trace examples
    key = dt.strftime("%Y-%m")
    if key in ("2008-01", "2009-01"):
        top10 = sim_result.distances.nsmallest(10)
        sim_detail_rows[key] = {
            "date": dt,
            "top10_dates": top10.index.tolist(),
            "top10_dists": top10.values.tolist(),
            "expected_returns": dict(sim_er),
            "weights": dict(w),
            "n_similar": len(sim_result.similar_dates),
        }

weights_df = pd.DataFrame(weight_rows).set_index("date")
er_df = pd.DataFrame(er_rows).set_index("date")

# Save weights
weights_df.to_parquet(config.output_dir / "monthly_weights.parquet")

# ── Report ────────────────────────────────────────────────────────────────────

out = []

def p(s=""):
    out.append(s)
    print(s)

p("=" * 80)
p("  ALLOCATION DIAGNOSTICS — SIMILARITY-WEIGHTED STRATEGY")
p("=" * 80)
p(f"  Period: {weights_df.index.min().date()} to {weights_df.index.max().date()} ({len(weights_df)} months)")
p(f"  Assets: {list(weights_df.columns)}")

# ── Section 1: Full-sample average weights ────────────────────────────────────

p(f"\n{'─'*80}")
p("  1. FULL-SAMPLE AVERAGE ALLOCATION WEIGHTS")
p(f"{'─'*80}")
p(f"  {'Asset':>15} {'Mean':>7} {'Median':>7} {'Min':>7} {'Max':>7} {'%Zero':>7}")
p(f"  {'-'*55}")

for a in sorted(weights_df.columns):
    col = weights_df[a]
    mn = col.mean()
    md = col.median()
    mi = col.min()
    mx = col.max()
    pz = (col == 0).mean()
    p(f"  {a:>15} {mn:>6.1%} {md:>6.1%} {mi:>6.1%} {mx:>6.1%} {pz:>6.0%}")

# ── Section 2: Crisis period weights ─────────────────────────────────────────

p(f"\n{'─'*80}")
p("  2. ALLOCATION WEIGHTS DURING CRISIS PERIODS")
p(f"{'─'*80}")

crises = [
    ("GFC Window", "2007-07", "2009-06"),
    ("COVID", "2020-01", "2020-06"),
    ("2022 Rate Hikes", "2022-01", "2022-12"),
]

for cname, cs, ce in crises:
    mask = (weights_df.index >= pd.Timestamp(cs)) & (weights_df.index <= pd.Timestamp(ce))
    period = weights_df[mask]
    if len(period) == 0:
        p(f"\n  {cname}: no data")
        continue

    p(f"\n  {cname} ({cs} to {ce}):")
    header = f"    {'Date':>10}"
    for a in sorted(weights_df.columns):
        header += f" {a[:8]:>8}"
    p(header)
    p(f"    {'-' * (10 + 9 * len(weights_df.columns))}")

    for dt, row in period.iterrows():
        line = f"    {dt.strftime('%Y-%m'):>10}"
        for a in sorted(weights_df.columns):
            v = row.get(a, 0)
            line += f" {v:>7.0%}" if v > 0 else f" {'--':>7}"

        p(line)

# ── Section 3: Regime transition trace ────────────────────────────────────────

p(f"\n{'─'*80}")
p("  3. REGIME TRANSITION TRACE (Jan 2008 vs Jan 2009)")
p(f"{'─'*80}")

for key in ["2008-01", "2009-01"]:
    detail = sim_detail_rows.get(key)
    if detail is None:
        p(f"\n  {key}: not in backtest range")
        continue

    p(f"\n  {key} — {detail['n_similar']} similar months identified")
    p(f"  Top 10 similar months:")
    for d, dist in zip(detail["top10_dates"], detail["top10_dists"]):
        # Look up what happened after that month
        fwd = ""
        if d in asset_ret.index:
            spy_r = asset_ret.loc[d, "us_equity"] if "us_equity" in asset_ret.columns else None
            if spy_r is not None and pd.notna(spy_r):
                fwd = f"  (SPY fwd: {spy_r:+.1%})"
        p(f"    {d.date()} (d={dist:.2f}){fwd}")

    p(f"  Expected returns from similar months:")
    for a in sorted(detail["expected_returns"].keys()):
        er = detail["expected_returns"][a]
        p(f"    {a:>15}: {er:+.3%}")

    p(f"  Resulting weights:")
    for a in sorted(detail["weights"].keys()):
        w = detail["weights"][a]
        if w > 0:
            p(f"    {a:>15}: {w:.0%}")

# ── Section 4: Allocation concentration ───────────────────────────────────────

p(f"\n{'─'*80}")
p("  4. ALLOCATION CONCENTRATION")
p(f"{'─'*80}")

# Months where single asset > 50%
over_50 = 0
equity_largest = 0
n_nonzero = []
herfindahl = []

for _, row in weights_df.iterrows():
    vals = row.dropna()
    if vals.max() > 0.50:
        over_50 += 1
    if "us_equity" in vals.index and vals.get("us_equity", 0) == vals.max():
        equity_largest += 1
    nz = (vals > 0.005).sum()
    n_nonzero.append(nz)
    hhi = (vals ** 2).sum()
    herfindahl.append(hhi)

p(f"  Months with single asset >50%:     {over_50} ({over_50/len(weights_df):.0%})")
p(f"  Months equity is largest allocation:{equity_largest} ({equity_largest/len(weights_df):.0%})")
p(f"  Avg assets with non-zero weight:    {np.mean(n_nonzero):.1f}")
p(f"  Median assets with non-zero weight: {np.median(n_nonzero):.0f}")
p(f"  Avg Herfindahl (1/N={1/len(weights_df.columns):.2f} = perfect diversification):")
p(f"    Mean HHI:   {np.mean(herfindahl):.3f}")
p(f"    Median HHI: {np.median(herfindahl):.3f}")

# ── Section 5: Turnover ──────────────────────────────────────────────────────

p(f"\n{'─'*80}")
p("  5. TURNOVER ANALYSIS")
p(f"{'─'*80}")

turnover = []
prev_w = None
for dt, row in weights_df.iterrows():
    if prev_w is not None:
        to = sum(abs(row.get(a, 0) - prev_w.get(a, 0)) for a in weights_df.columns) / 2
        turnover.append((dt, to))
    prev_w = row

to_series = pd.Series(dict(turnover))

p(f"  Avg monthly turnover:  {to_series.mean():.1%}")
p(f"  Median turnover:       {to_series.median():.1%}")
p(f"  Max single-month:      {to_series.max():.1%} ({to_series.idxmax().date()})")
p(f"  Min turnover:          {to_series.min():.1%}")

# Turnover around crisis starts
p(f"\n  Turnover around crisis onsets (3 months before/after):")
crisis_starts = [
    ("GFC", "2008-09"),
    ("COVID", "2020-02"),
    ("2022 Bear", "2022-01"),
]
for cname, cs in crisis_starts:
    dt = pd.Timestamp(cs)
    window = to_series[(to_series.index >= dt - pd.DateOffset(months=3)) &
                       (to_series.index <= dt + pd.DateOffset(months=3))]
    if len(window) > 0:
        p(f"    {cname} ({cs}): avg={window.mean():.1%}, max={window.max():.1%}")
        for d, v in window.items():
            marker = " ← crisis start" if d.strftime("%Y-%m") == cs else ""
            p(f"      {d.strftime('%Y-%m')}: {v:.1%}{marker}")

p()

# Save report
with open(config.output_dir / "allocation_diagnostics.txt", "w") as f:
    f.write("\n".join(out))
