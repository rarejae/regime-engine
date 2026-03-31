"""Hierarchical v3-fix with conditional ceiling removal during max conviction."""

import logging, sys
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
from regime.hierarchical_v3 import (BASELINE, ASSETS, EQUITY, compute_multi_sma_signals,
                                     step1_multi_faber, step2_harvey_invvol, step3_hmm,
                                     step4_kritzman, step5_normalize)
from regime.run_daily_backtest import fetch_daily_etf_returns

logging.basicConfig(level=logging.WARNING)
OUTPUT = Path("regime/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010; UNIVERSE = list(BASELINE.keys())
config = RegimeConfig()

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

import yfinance as yf
ep = {}
for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD"),("DBC","DBC")]:
    d = yf.download(ticker, start="1998-01-01", progress=False)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p,"columns"): p=p.iloc[:,0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        ep[our] = p
prices_df = pd.DataFrame(ep).sort_index()
multi_sma_df = compute_multi_sma_signals(prices_df).shift(1)
single_trend_df = compute_trend_signals(prices_df).shift(1)
daily_rets_raw = prices_df.pct_change()
rvol_63 = daily_rets_raw.rolling(63, min_periods=30).std() * np.sqrt(252)
rvol_monthly = rvol_63.resample("MS").last().shift(1)

print("Fitting HMM...")
spy_raw = fetch_daily_data()
hmm_feat = compute_features(spy_raw)
hmm_pred = fit_and_predict_rolling(hmm_feat).set_index("date")
zone_raw = pd.Series("neutral", index=hmm_pred.index)
zone_raw[hmm_pred["bull_prob"]>0.7]="bull"
zone_raw[hmm_pred["bull_prob"]<0.3]="bear"
hmm_pred["zone"] = apply_persistence_filter(zone_raw)

basket = fetch_daily_basket()
turb_smooth = compute_turbulence(basket)
turb_pctl = compute_turbulence_pctl(turb_smooth, window=252)
ar = compute_absorption_ratio(basket, n_components=2)
ar_z_s = compute_ar_zscore(ar)
turb_m = turb_pctl.resample("MS").last().shift(1)
ar_z_m = ar_z_s.resample("MS").last().shift(1)

common_start = max(daily_ret.dropna(how="all").index.min(), hmm_pred.index.min(), pd.Timestamp("2002-01-01"))
trading_days = daily_ret.loc[common_start:].index

print(f"\n{'='*80}")
print(f"  HIERARCHICAL v3 — CONVICTION-BASED CEILING REMOVAL")
print(f"{'='*80}")
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")


def step5_uncapped(weights):
    """Normalize with only min cash 3% — no equity or single-asset caps."""
    w = dict(weights)
    # Only enforce min cash
    if w.get("cash", 0) < 0.03:
        deficit = 0.03 - w["cash"]
        others = [a for a in w if a != "cash" and w[a] > 0]
        tot = sum(w[a] for a in others)
        if tot > deficit:
            for a in others: w[a] -= deficit * (w[a] / tot)
        w["cash"] = 0.03
    total = sum(max(v, 0) for v in w.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        w = {a: max(v, 0) / total for a, v in w.items()}
    return w


strats = {"UC": {}, "V3": {}, "B_base": {}, "bench_6040": {}, "bench_ivv": {}}
current_w = {c: None for c in strats}
total_tc = {c: 0.0 for c in strats}

current_strengths = {a: 3 for a in ASSETS if a != "cash"}
current_trends = {a: True for a in ASSETS if a != "cash"}
harvey_er = {a: 0.0 for a in ASSETS}
w_uc = dict(BASELINE); w_v3 = dict(BASELINE)
max_conv_log = []
total_months = 0

for day in trading_days:
    if day not in daily_ret.index: continue
    dr = daily_ret.loc[day]
    avail = [a for a in UNIVERSE if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3: continue
    actual = {a: float(dr[a]) for a in avail}

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

    if is_ms:
        total_months += 1

        # Multi-SMA
        ms_cands = multi_sma_df.index[multi_sma_df.index <= day]
        if len(ms_cands) > 0:
            ms = ms_cands[-1]
            for a in ASSETS:
                if a == "cash": continue
                if a in multi_sma_df.columns:
                    v = multi_sma_df.loc[ms, a]
                    current_strengths[a] = int(v) if pd.notna(v) else 0

        # Single SMA for Faber-only
        st_cands = single_trend_df.index[single_trend_df.index <= day]
        if len(st_cands) > 0:
            st = st_cands[-1]
            for a in ASSETS:
                if a == "cash": continue
                if a in single_trend_df.columns:
                    s = single_trend_df.loc[st, a]
                    current_trends[a] = bool(s) if pd.notna(s) else True

        # Harvey
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
            except ValueError: pass

        # Realized vols
        rv_cands = rvol_monthly.index[rvol_monthly.index <= day]
        rvols = {}
        if len(rv_cands) > 0:
            rv = rv_cands[-1]
            for a in ASSETS:
                if a == "cash": continue
                if a in rvol_monthly.columns:
                    v = rvol_monthly.loc[rv, a]
                    rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

        # HMM
        prev_m = day - pd.DateOffset(months=1)
        hm = hmm_pred[(hmm_pred.index >= prev_m) & (hmm_pred.index < day)]
        bp = hm["bull_prob"].mean() if len(hm) > 0 else 0.5
        if pd.isna(bp): bp = 0.5

        # Kritzman
        tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
        if pd.isna(tp): tp = 0.5
        arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
        if pd.isna(arz): arz = 0

        # Check max conviction conditions
        cond_faber = current_strengths.get("IVV", 0) >= 3 and current_strengths.get("QQQ", 0) >= 3
        cond_harvey = harvey_er.get("IVV", 0) > 0 and harvey_er.get("QQQ", 0) > 0
        cond_hmm = bp > 0.7
        cond_kritzman = not (tp > 0.95 and arz > 2.0)
        max_conviction = cond_faber and cond_harvey and cond_hmm and cond_kritzman

        # Run steps 1-4 (same for both configs)
        w3, pool, _ = step1_multi_faber(current_strengths)
        w3, _ = step2_harvey_invvol(w3, pool, harvey_er, rvols)
        w3, _ = step3_hmm(w3, bp)
        w3, _ = step4_kritzman(w3, tp, arz)

        # V3-fix: normal caps
        w_v3 = step5_normalize(dict(w3))

        # UC: conditional caps
        if max_conviction:
            w_uc = step5_uncapped(dict(w3))
        else:
            w_uc = step5_normalize(dict(w3))

        # Log
        eq_capped = sum(w_v3.get(a, 0) for a in EQUITY)
        eq_uncapped = sum(w_uc.get(a, 0) for a in EQUITY)
        if max_conviction:
            max_conv_log.append({
                "date": day, "eq_capped": eq_capped, "eq_uncapped": eq_uncapped,
                "diff": eq_uncapped - eq_capped,
                "faber": cond_faber, "harvey": cond_harvey, "hmm": cond_hmm, "kritz": cond_kritzman,
                "bp": bp, "ivv_w_uc": w_uc.get("IVV", 0), "qqq_w_uc": w_uc.get("QQQ", 0),
            })

    w_base = dict(BASELINE)
    w_6040 = {a:0 for a in avail}
    if "IVV" in avail: w_6040["IVV"]=0.6
    if "VGLT" in avail: w_6040["VGLT"]=0.4
    w_ivv = {a:(1 if a=="IVV" else 0) for a in avail}

    all_w = {"UC":(w_uc,is_ms),"V3":(w_v3,is_ms),"B_base":(w_base,is_ms),
             "bench_6040":(w_6040,is_ms),"bench_ivv":(w_ivv,False)}

    for cn,(new_w,trade) in all_w.items():
        if current_w[cn] is not None and not trade: w_used=current_w[cn]
        else:
            w_used=new_w
            to=sum(abs(new_w.get(a,0)-(current_w[cn] or {}).get(a,0)) for a in avail)/2
            total_tc[cn]+=to*TC; current_w[cn]=new_w
        strats[cn][day]=sum(w_used.get(a,0)*actual.get(a,0) for a in avail)

results = {c:pd.Series(d).sort_index() for c,d in strats.items() if d}
n_years = len(trading_days)/252
ivv = results.get("bench_ivv")
labels = {"UC":"Uncapped","V3":"V3-fix","B_base":"Baseline","bench_6040":"60/40","bench_ivv":"IVV B&H"}

# Report
print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>12} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8} {'AnnTC':>6}")
print(f"  {'-'*56}")
for cn in ["UC","V3","B_base","bench_6040","bench_ivv"]:
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
    for cn in ["UC","V3","B_base","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        c=s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c)>0: print(f"    {labels[cn]:>12}: {((1+c).prod()-1):>+7.1%}")

# Bull capture
print(f"\n{'='*80}")
print(f"  BULL CAPTURE")
print(f"{'='*80}")
for yr,desc in [(2013,"Harvey worst"),(2019,"Strong bull"),(2023,"Tech momentum")]:
    print(f"\n  {yr} ({desc}):")
    for cn in ["UC","V3","B_base","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        y=s[s.index.year==yr]
        if len(y)>20: print(f"    {labels[cn]:>12}: {(1+y).prod()-1:>+7.1%}")

# Max conviction analysis
print(f"\n{'='*80}")
print(f"  MAX CONVICTION ANALYSIS")
print(f"{'='*80}")
mc_df = pd.DataFrame(max_conv_log)
n_mc = len(mc_df)
print(f"  Max conviction months: {n_mc}/{total_months} ({n_mc/total_months:.0%})")
if n_mc > 0:
    print(f"  Max equity (uncapped): {mc_df['eq_uncapped'].max():.0%}")
    print(f"  Avg equity uplift: {mc_df['diff'].mean():+.1%} ({mc_df['diff'].mean()*100:+.1f}pp)")
    print(f"\n  {'Date':>10} {'Eq(cap)':>8} {'Eq(unc)':>8} {'Diff':>6} {'IVV':>5} {'QQQ':>5} {'BP':>5}")
    for _,r in mc_df.iterrows():
        print(f"  {r['date'].strftime('%Y-%m'):>10} {r['eq_capped']:>7.0%} {r['eq_uncapped']:>7.0%} "
              f"{r['diff']:>+5.0%} {r['ivv_w_uc']:>4.0%} {r['qqq_w_uc']:>4.0%} {r['bp']:>4.1f}")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'Uncap':>8} {'V3-fix':>8} {'Base':>8} {'IVV':>8}")
for yr in range(2002,2025):
    row=f"  {yr:>6}"
    for cn in ["UC","V3","B_base","bench_ivv"]:
        s=results.get(cn)
        if s is None: row+=f" {'--':>8}"; continue
        y=s[s.index.year==yr]
        row+=f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in ["UC","V3","B_base","bench_6040","bench_ivv"]:
    s=results.get(cn)
    if s is None or len(s)<252: continue
    print(f"  {labels[cn]:>12}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

print()
