"""Faber trend filter backtest: baseline+Faber, Harvey, Harvey+Faber combined."""

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
from regime.faber_filter import compute_trend_signals, apply_faber_to_baseline, apply_faber_to_harvey
from regime.run_daily_backtest import fetch_daily_etf_returns

logging.basicConfig(level=logging.WARNING)
OUTPUT = Path("experiments/signal_development/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
UNIVERSE = list(BASELINE.keys())

config = RegimeConfig()
raw_macro = fetch_monthly_history(config)
transformed = transform_variables(raw_macro, config)
z_data = get_valid_zscored(transformed, config)
z_data_lagged = z_data.shift(1).dropna()

asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
for col in list(asset_ret.columns):
    if col not in UNIVERSE: asset_ret = asset_ret.drop(columns=[col])
asset_ret_fwd = asset_ret.shift(-1)

daily_ret = fetch_daily_etf_returns()
daily_cols = [c for c in daily_ret.columns if c in UNIVERSE]
daily_ret = daily_ret[daily_cols]

# Daily prices for SMA (reconstruct from returns)
# Better: use raw prices from yfinance
import yfinance as yf
etf_prices = {}
for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD"),("DBC","DBC")]:
    d = yf.download(ticker, start="1998-01-01", progress=False)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p, "columns"): p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        etf_prices[our] = p
prices_df = pd.DataFrame(etf_prices).sort_index()

# Trend signals (monthly) — SHIFTED by 1 month for PIT safety
# Signal at month T uses end-of-month T price, applied to month T+1 returns
trend_signals_df = compute_trend_signals(prices_df).shift(1)

common_start = max(daily_ret.dropna(how="all").index.min(), pd.Timestamp("2002-01-01"))
trading_days = daily_ret.loc[common_start:].index

print("=" * 80)
print("  FABER TREND FILTER BACKTEST")
print("=" * 80)
print(f"Universe: {UNIVERSE}")
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

# Run
strats = {"F_faber": {}, "H_harvey": {}, "HF_combo": {}, "B_baseline": {},
          "bench_6040": {}, "bench_ivv": {}}
current_w = {c: None for c in strats}
total_tc = {c: 0.0 for c in strats}
harvey_w = {a: 0.0 for a in UNIVERSE}
current_trends = {a: True for a in UNIVERSE if a != "cash"}

# Tracking
faber_cash_pcts = []
below_sma_counts = {a: 0 for a in UNIVERSE if a != "cash"}
total_months = 0
all_equity_off_months = []
sma_cross_log = []

for day in trading_days:
    if day not in daily_ret.index: continue
    dr = daily_ret.loc[day]
    avail = [a for a in UNIVERSE if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3: continue
    actual = {a: float(dr[a]) for a in avail}

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

    if is_ms:
        total_months += 1
        # Get trend signals for this month
        trend_month = trend_signals_df.index[trend_signals_df.index <= day]
        if len(trend_month) > 0:
            tm = trend_month[-1]
            for a in UNIVERSE:
                if a == "cash": continue
                if a in trend_signals_df.columns:
                    sig = trend_signals_df.loc[tm, a]
                    old = current_trends.get(a, True)
                    current_trends[a] = bool(sig) if pd.notna(sig) else True
                    if not current_trends[a]:
                        below_sma_counts[a] += 1
                    # Log SMA crossings for key assets
                    if a == "IVV" and old != current_trends[a]:
                        direction = "above" if current_trends[a] else "below"
                        sma_cross_log.append((day, direction))

        # Track cash allocation in Faber
        w_f = apply_faber_to_baseline(BASELINE, current_trends)
        faber_cash_pcts.append(w_f.get("cash", 0))

        # Check if all equity below SMA
        eq_trends = [current_trends.get(a, True) for a in ["IVV", "QQQ"]]
        if not any(eq_trends):
            all_equity_off_months.append(day)

        # Harvey signal
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

    # Compute weights for each config
    w_faber = apply_faber_to_baseline(BASELINE, current_trends)
    w_harvey = dict(harvey_w)
    w_hf = apply_faber_to_harvey(harvey_w, current_trends)
    w_base = dict(BASELINE)
    w_6040 = {a: 0.0 for a in avail}
    if "IVV" in avail: w_6040["IVV"] = 0.60
    if "VGLT" in avail: w_6040["VGLT"] = 0.40
    w_ivv = {a: (1.0 if a == "IVV" else 0.0) for a in avail}

    all_w = {"F_faber": (w_faber, is_ms), "H_harvey": (w_harvey, is_ms),
             "HF_combo": (w_hf, is_ms), "B_baseline": (w_base, is_ms),
             "bench_6040": (w_6040, is_ms), "bench_ivv": (w_ivv, False)}

    for cn, (new_w, trade) in all_w.items():
        if current_w[cn] is not None and not trade:
            w_used = current_w[cn]
        else:
            w_used = new_w
            to = sum(abs(new_w.get(a,0) - (current_w[cn] or {}).get(a,0)) for a in avail) / 2
            total_tc[cn] += to * TC
            current_w[cn] = new_w
        strats[cn][day] = sum(w_used.get(a,0) * actual.get(a,0) for a in avail)

results = {c: pd.Series(d).sort_index() for c, d in strats.items() if d}
n_years = len(trading_days) / 252
ivv = results.get("bench_ivv")
labels = {"F_faber": "Faber", "H_harvey": "Harvey", "HF_combo": "Harvey+Faber",
          "B_baseline": "Baseline", "bench_6040": "60/40", "bench_ivv": "IVV B&H"}

# ── Report ────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>14} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8} {'AnnTC':>6}")
print(f"  {'-'*58}")

for cn in ["F_faber","H_harvey","HF_combo","B_baseline","bench_6040","bench_ivv"]:
    s = results.get(cn)
    if s is None or len(s)<252: continue
    ar=s.mean()*252; av=s.std()*np.sqrt(252); sh=ar/av if av>0 else 0
    cum=(1+s).cumprod(); dd=((cum-cum.expanding().max())/cum.expanding().max()).min()
    corr=s.corr(ivv) if ivv is not None else 0
    tc=total_tc.get(cn,0)/n_years
    print(f"  {labels[cn]:>14} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {corr:>7.2f} {tc:>5.2%}")

# Crisis
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2,cs,ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for cn in ["F_faber","H_harvey","HF_combo","B_baseline","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        c=s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c)>0: print(f"    {labels[cn]:>14}: {((1+c).prod()-1):>+7.1%}")

# Bull market capture
print(f"\n{'='*80}")
print(f"  BULL MARKET CAPTURE")
print(f"{'='*80}")
for yr, desc in [(2013,"Harvey worst"),(2019,"Strong bull"),(2023,"Tech momentum")]:
    print(f"\n  {yr} ({desc}):")
    for cn in ["F_faber","H_harvey","HF_combo","B_baseline","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        y=s[s.index.year==yr]
        if len(y)>20: print(f"    {labels[cn]:>14}: {(1+y).prod()-1:>+7.1%}")

# Trend filter activity
print(f"\n{'='*80}")
print(f"  TREND FILTER ACTIVITY")
print(f"{'='*80}")
print(f"\n  Months below SMA (out of {total_months} total):")
for a in sorted(below_sma_counts.keys()):
    pct = below_sma_counts[a] / total_months if total_months > 0 else 0
    print(f"    {a:>6}: {below_sma_counts[a]:>4} months ({pct:.0%})")
print(f"\n  Average cash allocation (Faber config): {np.mean(faber_cash_pcts):.0%}")
print(f"  Months ALL equity below SMA: {len(all_equity_off_months)}")
for d in all_equity_off_months[:10]:
    print(f"    {d.date()}")

# IVV SMA crossings
print(f"\n{'='*80}")
print(f"  IVV 10-MONTH SMA CROSSINGS")
print(f"{'='*80}")
for d, direction in sma_cross_log:
    print(f"  {d.date()}: crossed {direction}")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'Faber':>8} {'Harvey':>8} {'H+F':>8} {'Base':>8} {'IVV':>8}")
for yr in range(2002,2025):
    row=f"  {yr:>6}"
    for cn in ["F_faber","H_harvey","HF_combo","B_baseline","bench_ivv"]:
        s=results.get(cn)
        if s is None: row+=f" {'--':>8}"; continue
        y=s[s.index.year==yr]
        row+=f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in ["F_faber","H_harvey","HF_combo","B_baseline","bench_6040","bench_ivv"]:
    s=results.get(cn)
    if s is None or len(s)<252: continue
    print(f"  {labels[cn]:>14}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

print()
