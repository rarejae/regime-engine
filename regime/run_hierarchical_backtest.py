"""Daily backtest: hierarchical signal system."""

import logging, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances
from regime.faber_filter import compute_trend_signals, apply_faber_to_baseline
from regime.hmm_trend import fetch_daily_data, compute_features, fit_and_predict_rolling, apply_persistence_filter
from regime.kritzman import (fetch_daily_basket, compute_turbulence, compute_turbulence_pctl,
                              compute_absorption_ratio, compute_ar_zscore)
from regime.hierarchical import BASELINE, ASSETS, EQUITY, run_hierarchy
from regime.run_daily_backtest import fetch_daily_etf_returns

logging.basicConfig(level=logging.WARNING)
OUTPUT = Path("regime/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010; UNIVERSE = list(BASELINE.keys())
config = RegimeConfig()

# Load data
raw_macro = fetch_monthly_history(config)
transformed = transform_variables(raw_macro, config)
z_data = get_valid_zscored(transformed, config)
z_data_lagged = z_data.shift(1).dropna()

asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
for c in list(asset_ret.columns):
    if c not in UNIVERSE: asset_ret = asset_ret.drop(columns=[c])
asset_ret_fwd = asset_ret.shift(-1)

daily_ret = fetch_daily_etf_returns()
daily_ret = daily_ret[[c for c in daily_ret.columns if c in UNIVERSE]]

# Faber SMA (shifted for PIT)
import yfinance as yf
etf_prices = {}
for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD"),("DBC","DBC")]:
    d = yf.download(ticker, start="1998-01-01", progress=False)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p,"columns"): p=p.iloc[:,0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        etf_prices[our] = p
prices_df = pd.DataFrame(etf_prices).sort_index()
trend_df = compute_trend_signals(prices_df).shift(1)

# HMM
spy_raw = fetch_daily_data()
hmm_feat = compute_features(spy_raw)
print("Fitting HMM...")
hmm_pred = fit_and_predict_rolling(hmm_feat).set_index("date")
zone_raw = pd.Series("neutral", index=hmm_pred.index)
zone_raw[hmm_pred["bull_prob"]>0.7]="bull"
zone_raw[hmm_pred["bull_prob"]<0.3]="bear"
hmm_pred["zone"] = apply_persistence_filter(zone_raw)

# Kritzman (shifted for PIT)
basket = fetch_daily_basket()
turb_smooth = compute_turbulence(basket)
turb_pctl = compute_turbulence_pctl(turb_smooth, window=252)
ar = compute_absorption_ratio(basket, n_components=2)
ar_z = compute_ar_zscore(ar)
turb_m = turb_pctl.resample("MS").last().shift(1)
ar_z_m = ar_z.resample("MS").last().shift(1)

# Backtest
common_start = max(daily_ret.dropna(how="all").index.min(), hmm_pred.index.min(), pd.Timestamp("2002-01-01"))
trading_days = daily_ret.loc[common_start:].index

print("=" * 80)
print("  HIERARCHICAL SIGNAL SYSTEM BACKTEST")
print("=" * 80)
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

strats = {"S_seq": {}, "F_faber": {}, "B_baseline": {}, "bench_6040": {}, "bench_ivv": {}}
current_w = {c: None for c in strats}
total_tc = {c: 0.0 for c in strats}
n_trades = {c: 0 for c in strats}

current_trends = {a: True for a in ASSETS if a != "cash"}
harvey_er = {a: 0.0 for a in ASSETS}
seq_w = dict(BASELINE)
trace_log = {}
pool_sizes = []
pool_destinations = []
kritz_fires = 0

for day in trading_days:
    if day not in daily_ret.index: continue
    dr = daily_ret.loc[day]
    avail = [a for a in UNIVERSE if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3: continue
    actual = {a: float(dr[a]) for a in avail}

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

    if is_ms:
        # Update Faber signals
        tm_cands = trend_df.index[trend_df.index <= day]
        if len(tm_cands) > 0:
            tm = tm_cands[-1]
            for a in ASSETS:
                if a == "cash": continue
                if a in trend_df.columns:
                    s = trend_df.loc[tm, a]
                    current_trends[a] = bool(s) if pd.notna(s) else True

        # Update Harvey
        z_cands = z_data_lagged.index[z_data_lagged.index < day]
        if len(z_cands) > 0:
            z_dt = z_cands[-1]
            try:
                sim = compute_distances(z_data_lagged, z_dt, config)
                for a in avail:
                    if a not in asset_ret_fwd.columns: harvey_er[a]=0; continue
                    rets = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates
                            if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                    harvey_er[a] = np.mean(rets) if rets else 0
            except ValueError:
                pass

        # HMM bull prob (monthly average from last month)
        prev_month_end = day - pd.DateOffset(months=1)
        hmm_month = hmm_pred[(hmm_pred.index >= prev_month_end) & (hmm_pred.index < day)]
        bp = hmm_month["bull_prob"].mean() if len(hmm_month) > 0 else 0.5
        if pd.isna(bp): bp = 0.5

        # Kritzman
        tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
        if pd.isna(tp): tp = 0.5
        arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
        if pd.isna(arz): arz = 0

        # Run hierarchy
        seq_w, trace = run_hierarchy(current_trends, harvey_er, bp, tp, arz)

        pool_sizes.append(trace["step1"]["pool"])
        if trace["step2"]["directed_to"]:
            pool_destinations.append(trace["step2"]["directed_to"])
        if trace["step4"]["fired"]:
            kritz_fires += 1

        # Log trace for key months
        key = day.strftime("%Y-%m")
        if key in ("2008-01","2009-01","2013-06","2020-03","2022-06"):
            trace_log[key] = trace

    # Faber alone
    w_fab = apply_faber_to_baseline(BASELINE, current_trends)
    w_base = dict(BASELINE)
    w_6040 = {a:0 for a in avail}
    if "IVV" in avail: w_6040["IVV"]=0.6
    if "VGLT" in avail: w_6040["VGLT"]=0.4
    w_ivv = {a:(1 if a=="IVV" else 0) for a in avail}

    all_w = {"S_seq":(seq_w,is_ms),"F_faber":(w_fab,is_ms),"B_baseline":(w_base,is_ms),
             "bench_6040":(w_6040,is_ms),"bench_ivv":(w_ivv,False)}

    for cn,(new_w,trade) in all_w.items():
        if current_w[cn] is not None and not trade: w_used=current_w[cn]
        else:
            w_used=new_w
            to=sum(abs(new_w.get(a,0)-(current_w[cn] or {}).get(a,0)) for a in avail)/2
            total_tc[cn]+=to*TC; n_trades[cn]+=1 if to>0.01 else 0
            current_w[cn]=new_w
        strats[cn][day]=sum(w_used.get(a,0)*actual.get(a,0) for a in avail)

results = {c:pd.Series(d).sort_index() for c,d in strats.items() if d}
n_years = len(trading_days)/252
ivv = results.get("bench_ivv")
labels = {"S_seq":"Sequential","F_faber":"Faber","B_baseline":"Baseline",
          "bench_6040":"60/40","bench_ivv":"IVV B&H"}

# Report
print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>12} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8} {'AnnTC':>6}")
print(f"  {'-'*56}")
for cn in ["S_seq","F_faber","B_baseline","bench_6040","bench_ivv"]:
    s=results.get(cn)
    if s is None or len(s)<252: continue
    ar=s.mean()*252;av=s.std()*np.sqrt(252);sh=ar/av if av>0 else 0
    cum=(1+s).cumprod();dd=((cum-cum.expanding().max())/cum.expanding().max()).min()
    corr=s.corr(ivv) if ivv is not None else 0
    tc=total_tc.get(cn,0)/n_years
    print(f"  {labels[cn]:>12} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {corr:>7.2f} {tc:>5.2%}")

# Crisis
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2,cs,ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for cn in ["S_seq","F_faber","B_baseline","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        c=s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c)>0: print(f"    {labels[cn]:>12}: {((1+c).prod()-1):>+7.1%}")

# Bull capture
print(f"\n{'='*80}")
print(f"  BULL MARKET CAPTURE")
print(f"{'='*80}")
for yr,desc in [(2013,"Harvey worst"),(2019,"Strong bull"),(2023,"Tech momentum")]:
    print(f"\n  {yr} ({desc}):")
    for cn in ["S_seq","F_faber","B_baseline","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        y=s[s.index.year==yr]
        if len(y)>20: print(f"    {labels[cn]:>12}: {(1+y).prod()-1:>+7.1%}")

# Decision traces
print(f"\n{'='*80}")
print(f"  DECISION TRACES")
print(f"{'='*80}")
for key in ["2008-01","2009-01","2013-06","2020-03","2022-06"]:
    t = trace_log.get(key)
    if t is None:
        print(f"\n  {key}: not in backtest range")
        continue
    print(f"\n  {key}:")
    s1 = t["step1"]
    print(f"    Step 1 (Faber): ineligible={s1['ineligible']}, pool={s1['pool']:.0%}")
    s2 = t["step2"]
    if s2["directed_to"]:
        dirs = ", ".join(f"{a}={v:.0%}" for a,v in s2["directed_to"].items())
        print(f"    Step 2 (Harvey): pool→ {dirs}")
    else:
        print(f"    Step 2 (Harvey): no pool to allocate")
    s3 = t["step3"]
    print(f"    Step 3 (HMM): {s3['action']} (bull_prob={s3['bull_prob']:.2f})")
    s4 = t["step4"]
    print(f"    Step 4 (Kritzman): {'FIRED' if s4['fired'] else 'no action'} (turb={s4['turb_pctl']:.2f}, ar_z={s4['ar_z']:+.1f})")
    final = t["final"]
    wstr = "  ".join(f"{a}={v:.0%}" for a,v in sorted(final.items()) if v > 0.005)
    print(f"    Final: {wstr}")

# Pool analysis
print(f"\n{'='*80}")
print(f"  REALLOCATION POOL ANALYSIS")
print(f"{'='*80}")
print(f"  Avg pool size: {np.mean(pool_sizes):.0%}")
print(f"  Pool > 0 months: {sum(1 for p in pool_sizes if p > 0.01)}/{len(pool_sizes)} ({sum(1 for p in pool_sizes if p>0.01)/len(pool_sizes):.0%})")
if pool_destinations:
    dest_counts = {}
    dest_amounts = {}
    for pd_dict in pool_destinations:
        for a, v in pd_dict.items():
            dest_counts[a] = dest_counts.get(a, 0) + 1
            dest_amounts[a] = dest_amounts.get(a, 0) + v
    print(f"  Pool directed to (frequency / total amount):")
    for a in sorted(dest_counts, key=dest_counts.get, reverse=True):
        print(f"    {a:>6}: {dest_counts[a]:>3} months, total {dest_amounts[a]:.1%}")
print(f"  Kritzman emergency fires: {kritz_fires}")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'Seq':>8} {'Faber':>8} {'Base':>8} {'IVV':>8}")
for yr in range(2002,2025):
    row=f"  {yr:>6}"
    for cn in ["S_seq","F_faber","B_baseline","bench_ivv"]:
        s=results.get(cn)
        if s is None: row+=f" {'--':>8}"; continue
        y=s[s.index.year==yr]
        row+=f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in ["S_seq","F_faber","B_baseline","bench_6040","bench_ivv"]:
    s=results.get(cn)
    if s is None or len(s)<252: continue
    print(f"  {labels[cn]:>12}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

print()
